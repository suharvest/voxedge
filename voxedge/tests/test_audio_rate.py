"""Pure-numpy speed/pitch DSP fallback tests (voxedge.audio.rate).

Covers: identity byte-equality, speed duration ratios, pitch direction +
duration preservation, mono + 48k stereo round-trips, int16 no-overflow, and
streaming push/flush parity + seam continuity vs the offline transform.
NumPy only, no CUDA / device deps.
"""
from __future__ import annotations

import io
import struct
import wave

import numpy as np

from voxedge.audio.rate import (
    TTSRateShifter,
    apply_pcm_rate_pitch,
    apply_wav_rate_pitch,
    pitch_shift_wsola,
    time_stretch_wsola,
)

SR = 24000


def _tone(seconds=1.0, freqs=(220.0, 440.0, 660.0), amps=(0.3, 0.15, 0.08), sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    x = np.zeros_like(t)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2 * np.pi * f * t)
    return np.rint(x * 32767).astype(np.int16)


def _fundamental(int16, sr=SR):
    s = int16.astype(np.float32) / 32768.0
    win = np.hanning(len(s))
    S = np.abs(np.fft.rfft(s * win))
    f = np.fft.rfftfreq(len(s), 1.0 / sr)
    return f[np.argmax(S)]


# ── identity ───────────────────────────────────────────────────────────────


def test_identity_speed_bytes_identical():
    x = _tone(0.5)
    out = time_stretch_wsola(x, SR, 1.0)
    assert out.tobytes() == x.tobytes()
    out_none = time_stretch_wsola(x, SR, None)  # type: ignore[arg-type]
    assert out_none.tobytes() == x.tobytes()


def test_identity_pitch_bytes_identical():
    x = _tone(0.5)
    assert pitch_shift_wsola(x, SR, 0.0).tobytes() == x.tobytes()
    assert pitch_shift_wsola(x, SR, None).tobytes() == x.tobytes()  # type: ignore[arg-type]


def test_apply_pcm_identity_returns_same_object():
    x = _tone(0.3)
    pcm = x.tobytes()
    assert apply_pcm_rate_pitch(pcm, SR, speed=1.0, pitch_shift=0.0) is pcm
    assert apply_pcm_rate_pitch(pcm, SR, speed=None, pitch_shift=None) is pcm


def test_apply_wav_identity_returns_same_object():
    wav = _wrap(_tone(0.3))
    assert apply_wav_rate_pitch(wav, speed=1.0, pitch_shift=0.0) is wav
    assert apply_wav_rate_pitch(wav, speed=None, pitch_shift=None) is wav


# ── speed ────────────────────────────────────────────────────────────────


def test_speed_duration_ratio_within_3pct():
    x = _tone(2.0)
    for speed in (0.8, 1.2, 1.5):
        y = time_stretch_wsola(x, SR, speed)
        ratio = len(y) / len(x)
        target = 1.0 / speed
        assert abs(ratio - target) / target < 0.03, (speed, ratio, target)


# ── pitch ──────────────────────────────────────────────────────────────────


def test_pitch_changes_fundamental_and_preserves_duration():
    x = _tone(1.5)
    base = _fundamental(x)
    for st in (-3, 3):
        y = pitch_shift_wsola(x, SR, st)
        # duration preserved within ~3%
        assert abs(len(y) / len(x) - 1.0) < 0.03
        expected = base * 2 ** (st / 12.0)
        got = _fundamental(y)
        # within ~6% of the expected semitone shift, correct direction
        assert abs(got - expected) / expected < 0.06, (st, got, expected)
        if st > 0:
            assert got > base
        else:
            assert got < base


# ── stereo / int16 ──────────────────────────────────────────────────────────


def test_stereo_48k_roundtrip_speed_and_pitch():
    sr = 48000
    t = np.arange(int(sr * 1.0)) / sr
    left = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    right = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    inter = np.empty(left.size * 2, dtype=np.int16)
    inter[0::2] = np.rint(left * 32767).astype(np.int16)
    inter[1::2] = np.rint(right * 32767).astype(np.int16)

    y = time_stretch_wsola(inter, sr, 1.25, channels=2)
    assert y.size % 2 == 0
    assert abs((y.size / 2) / (inter.size / 2) - 1 / 1.25) < 0.03

    p = pitch_shift_wsola(inter, sr, 2.0, channels=2)
    assert p.size % 2 == 0
    assert abs((p.size / 2) / (inter.size / 2) - 1.0) < 0.03


def test_no_int16_overflow_on_loud_input():
    # full-scale signal — the clip-before-cast must keep it in range
    t = np.arange(int(SR * 0.5)) / SR
    x = np.rint(0.99 * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)
    y = time_stretch_wsola(x, SR, 1.3)
    assert y.dtype == np.int16
    assert y.min() >= -32768 and y.max() <= 32767
    p = pitch_shift_wsola(x, SR, 4.0)
    assert p.min() >= -32768 and p.max() <= 32767


# ── streaming parity + continuity ───────────────────────────────────────────


def test_streaming_speed_matches_offline_length():
    x = _tone(2.0)
    offline = time_stretch_wsola(x, SR, 1.3).tobytes()
    sh = TTSRateShifter(SR, speed=1.3)
    out = b""
    b = x.tobytes()
    for i in range(0, len(b), 4000):
        out += sh.push(b[i : i + 4000])
    out += sh.flush()
    # hop-aligned streaming tracks the offline transform closely
    assert abs(len(out) - len(offline)) / len(offline) < 0.05


def test_streaming_identity_passthrough():
    x = _tone(0.5)
    sh = TTSRateShifter(SR, speed=1.0, pitch_shift=0.0)
    b = x.tobytes()
    out = b""
    for i in range(0, len(b), 3000):
        out += sh.push(b[i : i + 3000])
    out += sh.flush()
    assert out == b  # byte-identical pass-through


def test_streaming_no_large_seam_discontinuity():
    x = _tone(2.0)
    sh = TTSRateShifter(SR, speed=1.2)
    out = b""
    b = x.tobytes()
    for i in range(0, len(b), 2048):
        out += sh.push(b[i : i + 2048])
    out += sh.flush()
    y = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    # no abrupt sample-to-sample jump bigger than full scale (seam glitch guard)
    diffs = np.abs(np.diff(y))
    assert diffs.max() < 0.5, diffs.max()


# ── WAV wrapper ──────────────────────────────────────────────────────────────


def _wrap(int16, sr=SR, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(int16.tobytes())
    return buf.getvalue()


def test_apply_wav_speed_changes_duration():
    x = _tone(2.0)
    wav = _wrap(x)
    out = apply_wav_rate_pitch(wav, speed=1.5)
    with wave.open(io.BytesIO(out), "rb") as r:
        n = r.getnframes()
        assert r.getframerate() == SR
        assert r.getnchannels() == 1
    assert abs(n / len(x) - 1 / 1.5) < 0.03
