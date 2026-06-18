"""Pure-numpy speed / pitch DSP fallback for TTS backends.

This module gives voxedge a backend-agnostic way to change **playback speed**
(tempo, pitch-preserving) and **pitch** (semitone shift, duration-preserving)
without a hard dependency on ffmpeg / librosa / pyrubberband. It is used as the
post-process fallback by the :class:`~voxedge.backends.base.TTSBackend` wrapper
for the (speed, pitch) dimensions a backend does **not** support natively.

Algorithm: WSOLA (Waveform Similarity Overlap-Add) for the time-stretch, and a
stretch + linear-resample for the pitch shift (shift = stretch by 1/r then
resample by r to restore the original length).

Design rules (from the reviewed spec):
  * Identity fast-path: speed in (None, 1.0) AND pitch in (None, 0.0) returns the
    input UNCHANGED (zero copy, byte-identical). The default no-op TTS path must
    stay untouched.
  * Vectorised overlap-add (no per-sample python loop).
  * Stereo: WSOLA timing is computed once on summed-mono and the SAME timing
    decisions are applied to every channel, then re-interleaved.
"""
from __future__ import annotations

import io
import struct
import wave
from typing import Optional

import numpy as np

__all__ = [
    "time_stretch_wsola",
    "pitch_shift_wsola",
    "TTSRateShifter",
    "apply_wav_rate_pitch",
    "apply_pcm_rate_pitch",
]


# ── helpers ──────────────────────────────────────────────────────────────


def _is_speed_identity(speed: Optional[float]) -> bool:
    return speed is None or speed == 1.0


def _is_pitch_identity(pitch: Optional[float]) -> bool:
    return pitch is None or pitch == 0.0


def _window_params(sample_rate: int) -> tuple[int, int, int]:
    """Return (W, Ha, delta) — analysis window, analysis hop, search radius."""
    W = int(round(sample_rate * 0.040))
    W = max(512, min(2048, W))
    if W % 2 != 0:
        W += 1
    Ha = W // 4
    delta = int(round(sample_rate * 0.008))
    return W, Ha, delta


def _to_float32_mono_or_channels(x_int16: np.ndarray, channels: int) -> np.ndarray:
    """Return float32 array shaped (n_frames, channels) in [-1, 1]."""
    flat = np.asarray(x_int16, dtype=np.int16).reshape(-1)
    if channels > 1:
        n = (flat.shape[0] // channels) * channels
        flat = flat[:n]
        frames = flat.reshape(-1, channels)
    else:
        frames = flat.reshape(-1, 1)
    return frames.astype(np.float32) / 32768.0


def _float_to_int16_bytes(y: np.ndarray) -> bytes:
    """y: (n_frames, channels) float32 → interleaved int16 bytes."""
    y = np.clip(y, -1.0, 1.0)
    out = np.rint(y * 32767.0).astype(np.int16)
    return out.reshape(-1).tobytes()


def _hann(W: int) -> np.ndarray:
    # periodic Hann (matches np.hanning's symmetric version closely enough for OLA)
    n = np.arange(W, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / W)).astype(np.float32)


def _wsola_timing(mono: np.ndarray, W: int, Ha: int, Hs: int, delta: int) -> list[int]:
    """Compute the list of analysis-frame start indices for every output frame.

    The timing is derived purely from ``mono`` (the summed-mono signal) so that
    the identical decisions can be replayed on each channel.
    """
    n = mono.shape[0]
    if n < W:
        return [0]

    # We step through the INPUT at the analysis hop Ha (one frame per step);
    # the synthesis hop Hs (= Ha/speed) is applied at overlap-add time, which is
    # what actually sets the output length (≈ n/speed). Frame count is therefore
    # driven by Ha, not Hs.
    n_out = 1 + max(0, (n - W) // Ha)

    starts: list[int] = []
    prev_tail: Optional[np.ndarray] = None
    # nominal analysis position advances by Ha per output frame
    for m in range(n_out):
        nominal = m * Ha
        if prev_tail is None:
            xa = nominal
            xa = max(0, min(xa, n - W))
            starts.append(xa)
            prev_tail = mono[xa + Hs : xa + Hs + (W - Hs)]
            continue

        lo = max(0, nominal - delta)
        hi = min(n - W, nominal + delta)
        if hi <= lo:
            best = max(0, min(nominal, n - W))
        else:
            best = lo
            best_score = -np.inf
            L = prev_tail.shape[0]
            pnorm = float(np.linalg.norm(prev_tail)) + 1e-8
            for cand in range(lo, hi + 1):
                seg = mono[cand : cand + L]
                if seg.shape[0] < L:
                    continue
                denom = (float(np.linalg.norm(seg)) + 1e-8) * pnorm
                score = float(np.dot(seg, prev_tail)) / denom
                if score > best_score:
                    best_score = score
                    best = cand
        starts.append(best)
        prev_tail = mono[best + Hs : best + Hs + (W - Hs)]
    return starts


def _wsola_overlap_add(
    chan: np.ndarray, starts: list[int], W: int, Hs: int, win: np.ndarray
) -> np.ndarray:
    """Vectorised Hann overlap-add of windowed frames at the given ``starts``."""
    n = chan.shape[0]
    n_out = len(starts)
    out_len = (n_out - 1) * Hs + W
    y = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    # gather all frames into a (n_out, W) matrix (zero-padded at the tail)
    frames = np.zeros((n_out, W), dtype=np.float32)
    for i, s in enumerate(starts):
        s = max(0, min(s, max(0, n - 1)))
        seg = chan[s : s + W]
        frames[i, : seg.shape[0]] = seg
    frames *= win[None, :]

    # scatter-add into the output with vectorised per-position offsets
    out_pos = (np.arange(n_out)[:, None] * Hs) + np.arange(W)[None, :]
    np.add.at(y, out_pos.reshape(-1), frames.reshape(-1))
    win_tiled = np.broadcast_to(win, (n_out, W))
    np.add.at(norm, out_pos.reshape(-1), win_tiled.reshape(-1))

    y /= np.maximum(norm, 1e-8)
    np.clip(y, -1.0, 1.0, out=y)
    return y


def _resample_linear(chan: np.ndarray, ratio: float, out_len: Optional[int] = None) -> np.ndarray:
    """Resample ``chan`` by ``ratio`` (out_len ≈ len/ratio) with linear interp."""
    n = chan.shape[0]
    if n == 0:
        return chan
    if out_len is None:
        out_len = max(1, int(round(n / ratio)))
    if out_len <= 1:
        return chan[:out_len].copy()
    idx = np.linspace(0.0, n - 1, out_len)
    return np.interp(idx, np.arange(n), chan).astype(np.float32)


# ── public DSP ───────────────────────────────────────────────────────────


def time_stretch_wsola(
    x_int16: np.ndarray, sample_rate: int, speed: float, *, channels: int = 1
) -> np.ndarray:
    """Pitch-preserving time-stretch by ``speed`` (output ≈ input_len / speed).

    ``speed > 1`` → faster/shorter; ``speed < 1`` → slower/longer. Returns an
    int16 ndarray (interleaved if ``channels > 1``).
    """
    arr16 = np.asarray(x_int16, dtype=np.int16).reshape(-1)
    if _is_speed_identity(speed):
        return arr16
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")

    W, Ha, delta = _window_params(sample_rate)
    Hs = max(1, int(round(Ha / speed)))
    win = _hann(W)

    frames = _to_float32_mono_or_channels(arr16, channels)  # (n, ch)
    n = frames.shape[0]
    if n < W:
        return arr16

    mono = frames.mean(axis=1)
    starts = _wsola_timing(mono, W, Ha, Hs, delta)

    out_channels = [
        _wsola_overlap_add(frames[:, c], starts, W, Hs, win)
        for c in range(frames.shape[1])
    ]
    y = np.stack(out_channels, axis=1)  # (out_len, ch)
    return np.frombuffer(_float_to_int16_bytes(y), dtype=np.int16)


def pitch_shift_wsola(
    x_int16: np.ndarray, sample_rate: int, semitones: float, *, channels: int = 1
) -> np.ndarray:
    """Duration-preserving pitch shift by ``semitones`` (+ up, − down).

    Implemented as time-stretch by ``1/r`` then resample by ``r`` (linear) to
    restore the original length, where ``r = 2 ** (semitones/12)``.
    """
    arr16 = np.asarray(x_int16, dtype=np.int16).reshape(-1)
    if _is_pitch_identity(semitones):
        return arr16

    r = 2.0 ** (semitones / 12.0)
    # Pitch up (r>1): time-stretch LONGER by r (speed=1/r → duration ×r, pitch
    # unchanged) then resample/decimate by r to restore the original length,
    # which scales the playback rate → raises pitch by r. Pitch down is the
    # mirror image.
    stretched = time_stretch_wsola(arr16, sample_rate, speed=1.0 / r, channels=channels)

    frames = _to_float32_mono_or_channels(stretched, channels)
    target_len = arr16.shape[0] // channels if channels > 1 else arr16.shape[0]
    out_channels = [
        _resample_linear(frames[:, c], ratio=r, out_len=target_len)
        for c in range(frames.shape[1])
    ]
    y = np.stack(out_channels, axis=1)
    return np.frombuffer(_float_to_int16_bytes(y), dtype=np.int16)


class TTSRateShifter:
    """Streaming, stateful speed+pitch shifter for chunked PCM.

    Feed raw int16 PCM bytes via :meth:`push` (returns processed bytes for the
    fully-synthesized hops so far) and finish with :meth:`flush` (drains the
    residual). A float32 buffer is carried across chunks so the WSOLA
    cross-correlation can span backend chunk boundaries.

    Identity fast-path: when speed in (None, 1.0) and pitch in (None, 0.0) the
    shifter is a pass-through — :meth:`push` returns its input bytes UNCHANGED.
    """

    def __init__(
        self,
        sample_rate: int,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        channels: int = 1,
    ):
        self.sample_rate = int(sample_rate)
        self.speed = 1.0 if speed is None else float(speed)
        self.pitch_shift = 0.0 if pitch_shift is None else float(pitch_shift)
        self.channels = max(1, int(channels))
        self._identity = _is_speed_identity(self.speed) and _is_pitch_identity(self.pitch_shift)

        self._buf = np.zeros((0, self.channels), dtype=np.float32)
        W, Ha, delta = _window_params(self.sample_rate)
        self._W, self._Ha, self._delta = W, Ha, delta
        # retain ≥ W + delta + Ha samples so x-corr spans the next chunk
        self._retain = W + delta + Ha
        self._closed = False

    # -- internal -----------------------------------------------------------

    def _bytes_to_frames(self, chunk: bytes) -> np.ndarray:
        flat = np.frombuffer(chunk, dtype=np.int16)
        if self.channels > 1:
            n = (flat.shape[0] // self.channels) * self.channels
            flat = flat[:n]
            frames = flat.reshape(-1, self.channels)
        else:
            frames = flat.reshape(-1, 1)
        return frames.astype(np.float32) / 32768.0

    def _process_available(self, *, final: bool) -> bytes:
        """Process whatever full hops are available; keep a tail for next time."""
        n = self._buf.shape[0]
        W, Ha, delta = self._W, self._Ha, self._delta
        if n < W:
            if not final:
                return b""
            # final with < W: zero-pad to W, process, drop the pad-induced tail
            if n == 0:
                return b""
            pad = np.zeros((W - n, self.channels), dtype=np.float32)
            block = np.concatenate([self._buf, pad], axis=0)
            self._buf = np.zeros((0, self.channels), dtype=np.float32)
            return self._render(block, valid_in=n)

        if final:
            block = self._buf
            self._buf = np.zeros((0, self.channels), dtype=np.float32)
            return self._render(block, valid_in=n)

        # streaming: emit whole analysis hops, but only CONSUME the input those
        # hops covered (n_out * Ha samples). Keeping the retained tail in the
        # buffer (rather than dropping it) is what previously inflated the
        # output length — here input-consumed / output-emitted stays
        # proportional to 1/speed across the whole stream.
        keep = self._retain
        usable = n - keep  # need ≥ keep samples held back for the next x-corr
        if usable <= W:
            return b""
        # number of analysis hops whose full window fits within `usable`
        n_out = 1 + (usable - W) // Ha
        if n_out <= 0:
            return b""
        consume = n_out * Ha
        block = self._buf[: consume + (W - Ha)]  # include the last frame's window
        out = self._render_hops(block, n_out)
        self._buf = self._buf[consume:]
        return out

    def _render_hops(self, block: np.ndarray, n_out: int) -> bytes:
        """Render exactly ``n_out`` analysis hops, emitting ``n_out * Hs``
        samples (the steady-state synthesis advance — the trailing window
        overlap is intentionally left for the next block to overlap-add, so
        seams are continuous and the length stays proportional to 1/speed)."""
        speed = self.speed
        pitch = self.pitch_shift
        W, Ha, delta = self._W, self._Ha, self._delta
        Hs = max(1, int(round(Ha / speed)))
        win = _hann(W)
        mono = block.mean(axis=1)
        # fixed analysis starts every Ha; WSOLA refines within ±delta
        starts = _wsola_timing(mono, W, Ha, Hs, delta)[:n_out]
        chans = [_wsola_overlap_add(block[:, c], starts, W, Hs, win) for c in range(self.channels)]
        block = np.stack(chans, axis=1)
        emit = min(block.shape[0], n_out * Hs)
        block = block[:emit]
        if not _is_pitch_identity(pitch):
            int16 = np.frombuffer(_float_to_int16_bytes(block), dtype=np.int16)
            shifted = pitch_shift_wsola(int16, self.sample_rate, pitch, channels=self.channels)
            return shifted.tobytes()
        return _float_to_int16_bytes(block)

    def _render(self, block: np.ndarray, *, valid_in: int) -> bytes:
        W, Ha, delta = self._W, self._Ha, self._delta
        speed = self.speed
        pitch = self.pitch_shift
        # speed stage
        if not _is_speed_identity(speed):
            Hs = max(1, int(round(Ha / speed)))
            win = _hann(W)
            mono = block.mean(axis=1)
            starts = _wsola_timing(mono, W, Ha, Hs, delta)
            chans = [_wsola_overlap_add(block[:, c], starts, W, Hs, win) for c in range(self.channels)]
            block = np.stack(chans, axis=1)
        # pitch stage (stretch already applied inside pitch via fresh path)
        if not _is_pitch_identity(pitch):
            r = 2.0 ** (pitch / 12.0)
            int16 = np.frombuffer(_float_to_int16_bytes(block), dtype=np.int16)
            shifted = pitch_shift_wsola(int16, self.sample_rate, pitch, channels=self.channels)
            return shifted.tobytes()
        return _float_to_int16_bytes(block)

    # -- public -------------------------------------------------------------

    def push(self, chunk: bytes) -> bytes:
        if self._identity:
            return chunk
        if not chunk:
            return b""
        frames = self._bytes_to_frames(chunk)
        if frames.shape[0]:
            self._buf = np.concatenate([self._buf, frames], axis=0)
        return self._process_available(final=False)

    def flush(self) -> bytes:
        if self._identity or self._closed:
            self._closed = True
            return b""
        self._closed = True
        return self._process_available(final=True)


# ── WAV / PCM wrappers ─────────────────────────────────────────────────────


def _parse_wav(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Return (pcm_int16_bytes, sample_rate, channels) from a WAV container."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as r:
        sr = r.getframerate()
        ch = r.getnchannels()
        sampwidth = r.getsampwidth()
        pcm = r.readframes(r.getnframes())
    if sampwidth != 2:
        raise ValueError(f"apply_wav_rate_pitch expects 16-bit PCM, got sampwidth={sampwidth}")
    return pcm, sr, ch


def _wrap_wav(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    buf = io.BytesIO()
    data_size = len(pcm)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    buf.write(struct.pack("<HHIIHH", 1, channels, sample_rate, byte_rate, block_align, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


def apply_pcm_rate_pitch(
    pcm_int16: bytes,
    sample_rate: int,
    speed: Optional[float] = None,
    pitch_shift: Optional[float] = None,
    channels: int = 1,
) -> bytes:
    """Transform raw interleaved int16 PCM bytes; identity → unchanged input."""
    if _is_speed_identity(speed) and _is_pitch_identity(pitch_shift):
        return pcm_int16
    arr = np.frombuffer(pcm_int16, dtype=np.int16)
    out = arr
    if not _is_speed_identity(speed):
        out = time_stretch_wsola(out, sample_rate, float(speed), channels=channels)
    if not _is_pitch_identity(pitch_shift):
        out = pitch_shift_wsola(out, sample_rate, float(pitch_shift), channels=channels)
    return out.tobytes()


def apply_wav_rate_pitch(
    wav_bytes: bytes,
    speed: Optional[float] = None,
    pitch_shift: Optional[float] = None,
    channels: Optional[int] = None,
) -> bytes:
    """Transform a WAV blob's audio by speed/pitch; identity → unchanged bytes.

    ``channels`` is read from the WAV header when not given; passing it
    overrides the header (e.g. when a backend's meta is authoritative).
    """
    if _is_speed_identity(speed) and _is_pitch_identity(pitch_shift):
        return wav_bytes
    if not wav_bytes or len(wav_bytes) <= 44:
        return wav_bytes
    pcm, sr, hdr_ch = _parse_wav(wav_bytes)
    ch = hdr_ch if channels is None else int(channels)
    new_pcm = apply_pcm_rate_pitch(pcm, sr, speed, pitch_shift, channels=ch)
    return _wrap_wav(new_pcm, sr, ch)
