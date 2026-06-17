"""TTSBackend speed/pitch wrapper tests (voxedge.backends.base).

Verifies the base wrapper:
  * caps (False,False)        → both speed & pitch DSP'd, impl sees neither.
  * caps (True,False)         → speed passed to impl natively, only pitch DSP'd.
  * identity request          → impl output byte-identical (no DSP).
  * no double-apply           → impl records exactly what speed/pitch it got.

A FakeBackend emits a fixed-length tone so the wrapper's DSP changes the WAV
duration observably, while recording every (speed, pitch) it received.
"""
from __future__ import annotations

import io
import wave
from typing import Iterator, Optional

import numpy as np

from voxedge.backends.base import TTSBackend, TTSCapability

SR = 24000


def _tone_wav(seconds=1.0, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    x = np.rint(0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x.tobytes())
    return buf.getvalue()


def _wav_nframes(wav):
    with wave.open(io.BytesIO(wav), "rb") as r:
        return r.getnframes()


class FakeBackend(TTSBackend):
    def __init__(self, caps: tuple[bool, bool], sr: int = SR):
        self._caps = caps
        self._sr = sr
        self.synth_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> set:
        return {TTSCapability.BASIC_TTS, TTSCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return self._sr

    def is_ready(self) -> bool:
        return True

    def preload(self) -> None:
        pass

    def rate_pitch_caps(self) -> tuple[bool, bool]:
        return self._caps

    def _synthesize_impl(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        self.synth_calls.append({"speed": speed, "pitch_shift": pitch_shift})
        return _tone_wav(1.0, self._sr), {"sample_rate": self._sr}

    def _generate_streaming_impl(self, text: str, **kwargs) -> Iterator[bytes]:
        self.stream_calls.append(
            {"speed": kwargs.get("speed"), "pitch_shift": kwargs.get("pitch_shift")}
        )
        # one full-second tone, chunked
        wav = _tone_wav(1.0, self._sr)
        pcm = wav[44:]
        step = 4000
        for i in range(0, len(pcm), step):
            yield pcm[i : i + step]


# ── identity (no DSP, byte-identical) ───────────────────────────────────────


def test_identity_synthesize_byte_identical():
    be = FakeBackend((False, False))
    raw, _ = be._synthesize_impl("hi")
    wav, _ = be.synthesize("hi")  # no speed/pitch
    assert wav == raw
    assert be.synth_calls[-1] == {"speed": None, "pitch_shift": None}


def test_identity_streaming_passthrough():
    be = FakeBackend((False, False))
    impl = b"".join(be._generate_streaming_impl("hi"))
    out = b"".join(be.generate_streaming("hi"))
    assert out == impl


# ── caps (False, False): both DSP'd, impl sees neither ──────────────────────


def test_caps_false_false_speed_and_pitch_dsped():
    be = FakeBackend((False, False))
    base_n = _wav_nframes(_tone_wav(1.0))
    wav, _ = be.synthesize("hi", speed=1.5)
    # impl never received speed (popped → DSP)
    assert be.synth_calls[-1]["speed"] is None
    assert be.synth_calls[-1]["pitch_shift"] is None
    # duration shortened by ~1/1.5
    assert abs(_wav_nframes(wav) / base_n - 1 / 1.5) < 0.04


def test_caps_false_false_pitch_preserves_duration():
    be = FakeBackend((False, False))
    base_n = _wav_nframes(_tone_wav(1.0))
    wav, _ = be.synthesize("hi", pitch_shift=3.0)
    assert be.synth_calls[-1]["pitch_shift"] is None  # DSP'd, not passed
    assert abs(_wav_nframes(wav) / base_n - 1.0) < 0.04  # duration preserved


# ── caps (True, False): speed native, pitch DSP'd (no double-apply) ─────────


def test_caps_true_false_speed_passed_pitch_dsped():
    be = FakeBackend((True, False))
    base_n = _wav_nframes(_tone_wav(1.0))
    wav, _ = be.synthesize("hi", speed=1.5, pitch_shift=3.0)
    # speed handed to impl natively; pitch popped for DSP
    assert be.synth_calls[-1]["speed"] == 1.5
    assert be.synth_calls[-1]["pitch_shift"] is None
    # FakeBackend ignores native speed (emits 1s tone), so DSP only did pitch →
    # duration stays ~1s (no speed DSP double-apply).
    assert abs(_wav_nframes(wav) / base_n - 1.0) < 0.04


def test_caps_true_false_streaming_speed_passed_only():
    be = FakeBackend((True, False))
    list(be.generate_streaming("hi", speed=1.3))
    # native speed reaches the streaming impl; pitch absent
    assert be.stream_calls[-1]["speed"] == 1.3
    assert be.stream_calls[-1]["pitch_shift"] is None


def test_caps_true_true_no_dsp_passes_both():
    be = FakeBackend((True, True))
    base_n = _wav_nframes(_tone_wav(1.0))
    wav, _ = be.synthesize("hi", speed=1.5, pitch_shift=3.0)
    assert be.synth_calls[-1]["speed"] == 1.5
    assert be.synth_calls[-1]["pitch_shift"] == 3.0
    # both native → no DSP at all → duration unchanged (FakeBackend ignores them)
    assert _wav_nframes(wav) == base_n


# ── streaming DSP path ───────────────────────────────────────────────────────


def test_caps_false_false_streaming_speed_dsped():
    be = FakeBackend((False, False))
    base_pcm = b"".join(be._generate_streaming_impl("hi"))
    out = b"".join(be.generate_streaming("hi", speed=1.5))
    # impl saw no speed (popped); output PCM shortened by DSP
    assert be.stream_calls[-1]["speed"] is None
    assert abs(len(out) / len(base_pcm) - 1 / 1.5) < 0.05
