"""Kokoro TRT segmented synthesis instead of truncating (migration gap).

The hybrid / split_generator runtime has a fixed max sequence length; an
utterance longer than ``_hybrid_max_seq_len - 2`` tokens used to be silently
truncated. ``synthesize`` now splits over-long text into segments, synthesizes
each via ``_synthesize_one``, concatenates the PCM, and reports
``meta["segments"]`` / ``meta["truncated"] is False`` / summed ``num_tokens``.

The env gate (``KOKORO_SYNTH_SEGMENT_TEXT`` / ``...MAX_SEGMENT_TOKENS``) → config
is covered product-side; this locks the voxedge-side splitting/concatenation
algorithm, which had no voxedge coverage after the env-free rewrite. NumPy-only,
no CUDA: builds the backend via ``__new__`` and monkeypatches the token + per-
segment synth helpers.
"""

from __future__ import annotations

import numpy as np

from voxedge.backends.jetson import kokoro_trt
from voxedge.backends.jetson.kokoro_trt import KokoroTRTBackend, KokoroTRTConfig


def test_kokoro_synthesize_segments_instead_of_truncating(monkeypatch):
    backend = KokoroTRTBackend.__new__(KokoroTRTBackend)
    backend._config = KokoroTRTConfig()  # synth_segment_text=True by default
    backend._runtime_mode = "split_generator"
    backend._hybrid_max_seq_len = 10  # → max_tokens = 8, text below has 20 tokens

    monkeypatch.setattr(backend, "_text_to_token_ids", lambda text: list(range(20)))
    monkeypatch.setattr(
        backend, "_split_stream_text", lambda text, max_tokens: ["first", "second"]
    )

    def fake_one(text, speaker_id=None, speed=None):
        samples = np.ones(240, dtype=np.float32) * (0.1 if text == "first" else 0.2)
        return kokoro_trt._samples_to_wav(samples, kokoro_trt.SAMPLE_RATE), {
            "num_tokens": 6,
            "infer_ms": 1.5,
        }

    monkeypatch.setattr(backend, "_synthesize_one", fake_one)

    wav, meta = backend.synthesize("too long")

    assert len(wav) > 44  # real WAV payload, not just a header
    assert meta["segments"] == 2
    assert meta["truncated"] is False
    assert meta["num_tokens"] == 12  # 6 + 6 summed across segments


def test_kokoro_short_text_uses_single_shot(monkeypatch):
    """Below the cap → no segmentation, single _synthesize_one call."""
    backend = KokoroTRTBackend.__new__(KokoroTRTBackend)
    backend._config = KokoroTRTConfig()
    backend._runtime_mode = "split_generator"
    backend._hybrid_max_seq_len = 128

    monkeypatch.setattr(backend, "_text_to_token_ids", lambda text: list(range(5)))
    calls = []

    def fake_one(text, speaker_id=None, speed=None):
        calls.append(text)
        samples = np.ones(240, dtype=np.float32) * 0.1
        return kokoro_trt._samples_to_wav(samples, kokoro_trt.SAMPLE_RATE), {
            "num_tokens": 5,
            "segments": 1,
            "truncated": False,
        }

    monkeypatch.setattr(backend, "_synthesize_one", fake_one)

    _, meta = backend.synthesize("short")
    assert calls == ["short"]
    assert meta.get("segments", 1) == 1
