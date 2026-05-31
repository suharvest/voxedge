"""TRT-Edge-LLM ASR offline long-audio segmentation + concatenation (gap).

Long offline WAV uploads (> ``offline_segment_threshold_s``) overflow the engine
KV cache if sent whole. ``transcribe`` splits them at silence, transcribes each
bounded segment, and joins the per-segment texts (CJK-aware: drop trailing
punctuation between segments, no inter-word space for zh/ja/ko). ``meta`` reports
``segmented`` / ``segment_count``.

The env gate (``EDGE_LLM_ASR_OFFLINE_SEGMENT*``) → config is covered product-side;
this locks the voxedge-side split/transcribe/join orchestration, which had no
voxedge coverage after the env-free rewrite (and after the configured-VAD path
was intentionally dropped — voxedge ships no VAD backend, so the old
``_split_offline_audio uses create_vad`` test is obsolete, not ported).

NumPy-only, no CUDA / no worker: builds the backend via ``__new__`` and
monkeypatches the splitter + the inner per-segment ``transcribe``.
"""

from __future__ import annotations

import numpy as np

import voxedge.backends.jetson.trt_edge_llm_asr as asr_mod
from voxedge.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    TRTEdgeLLMASRConfig,
    TranscriptionResult,
    _float_audio_to_wav_bytes,
)


def _make_backend(threshold_s=2.0):
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._config = TRTEdgeLLMASRConfig(
        offline_segment_enabled=True,
        offline_segment_threshold_s=threshold_s,
        offline_segment_min_s=0.4,
    )
    backend._ready = True
    backend._worker_ready_meta = {}
    return backend


def test_offline_transcribe_segments_long_audio(monkeypatch):
    backend = _make_backend(threshold_s=2.0)

    audio = np.ones(16000 * 5, dtype=np.float32) * 0.02  # 5s > 2s threshold
    wav_bytes = _float_audio_to_wav_bytes(audio, 16000)

    def fake_split(samples, sample_rate, *, max_segment_s):
        assert sample_rate == 16000
        return [
            samples[: sample_rate * 2],
            samples[sample_rate * 2 : sample_rate * 4],
            samples[sample_rate * 4 :],
        ]

    monkeypatch.setattr(asr_mod, "_split_offline_audio", fake_split)

    calls = {"n": 0}

    def fake_transcribe(seg_wav, language="auto"):
        calls["n"] += 1
        return TranscriptionResult(
            text=f"第{calls['n']}段。",
            language="Chinese",
            meta={"inference_time_s": 0.1},
        )

    # Patch the *inner* per-segment transcribe; the segmented path calls
    # ``self.transcribe`` recursively for each bounded segment.
    monkeypatch.setattr(backend, "transcribe", fake_transcribe)

    result = backend._transcribe_segmented_offline(audio, 16000, "Chinese")

    # CJK join: trailing 。 stripped between segments, no inter-word space.
    assert result.text == "第1段第2段第3段。"
    assert result.language == "Chinese"
    assert result.meta["segmented"] is True
    assert result.meta["segment_count"] == 3
    assert calls["n"] == 3


def test_offline_segment_skips_sub_min_segments(monkeypatch):
    backend = _make_backend(threshold_s=2.0)
    audio = np.ones(16000 * 5, dtype=np.float32) * 0.02

    def fake_split(samples, sample_rate, *, max_segment_s):
        # One real segment + one too-short (< 0.4s) segment that must be skipped.
        return [samples[: sample_rate * 2], samples[: int(sample_rate * 0.2)]]

    monkeypatch.setattr(asr_mod, "_split_offline_audio", fake_split)

    seen = []

    def fake_transcribe(seg_wav, language="auto"):
        seen.append(len(seg_wav))
        return TranscriptionResult(text="ok", language="en", meta={})

    monkeypatch.setattr(backend, "transcribe", fake_transcribe)

    result = backend._transcribe_segmented_offline(audio, 16000, "en")
    # segment_count counts all returned segments, but only the >= min one is sent.
    assert result.meta["segment_count"] == 2
    assert len(seen) == 1
