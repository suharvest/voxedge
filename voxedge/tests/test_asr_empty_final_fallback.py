"""Streaming ASR empty-final → offline-transcribe rescue.

The worker withholds up to ``unfixed_token_num`` trailing tokens; a SHORT
utterance whose entire output fits inside that hold emits NO final text
(observed real-machine 2026-06-14: short English commands → empty asr_final,
while offline ``transcribe()`` — the POST /asr path — transcribes the same audio
cleanly). ``_TRTEdgeLLMStreamingASRStream.finalize()`` now falls back to the
offline path when the worker returned empty but enough audio was buffered.
"""
from __future__ import annotations

import numpy as np
import pytest

from voxedge.backends.jetson.trt_edge_llm_asr import _TRTEdgeLLMStreamingASRStream


class _FakeConfig:
    offline_segment_min_s = 0.4


class _R:
    def __init__(self, text, language="English"):
        self.text = text
        self.language = language


class _FakeBackend:
    def __init__(self, fallback_text):
        self._config = _FakeConfig()
        self._fallback_text = fallback_text
        self.transcribe_calls: list = []

    def transcribe(self, wav_bytes, language="auto"):
        self.transcribe_calls.append((len(wav_bytes), language))
        return _R(self._fallback_text)

    def _worker_request(self, ev):
        return {}


def _make_stream(backend, *, audio_s, final_text):
    s = _TRTEdgeLLMStreamingASRStream.__new__(_TRTEdgeLLMStreamingASRStream)
    s._backend = backend
    s._language = "auto"
    s._sample_rate = 16000
    s._audio_accum = np.zeros(int(16000 * audio_s), dtype=np.float32)
    s._final_text = final_text
    s._committed_text = ""
    s._partial_text = ""
    s._detected_language = None
    s._cancelled = False
    s._closed = False
    s._session_id = "test"
    return s


def test_empty_final_short_english_falls_back_to_offline(monkeypatch):
    be = _FakeBackend("Go home.")
    s = _make_stream(be, audio_s=0.667, final_text="")  # worker withheld everything
    monkeypatch.setattr(s, "_send_chunk", lambda **k: {})  # no real worker
    text, lang = s.finalize()
    assert text == "Go home."          # rescued via offline transcribe()
    assert lang == "English"
    assert len(be.transcribe_calls) == 1


def test_nonempty_final_takes_zero_new_code(monkeypatch):
    be = _FakeBackend("SHOULD NOT BE USED")
    s = _make_stream(be, audio_s=0.667, final_text="回家")  # worker emitted text
    monkeypatch.setattr(s, "_send_chunk", lambda **k: {})
    text, _ = s.finalize()
    assert text == "回家"
    assert be.transcribe_calls == []   # no fallback for non-empty finals


def test_empty_final_too_short_no_fallback(monkeypatch):
    be = _FakeBackend("x")
    s = _make_stream(be, audio_s=0.2, final_text="")  # < offline_segment_min_s
    monkeypatch.setattr(s, "_send_chunk", lambda **k: {})
    text, _ = s.finalize()
    assert text == ""
    assert be.transcribe_calls == []   # too little audio to rescue
