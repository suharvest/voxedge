"""Proactive long-audio segment cap for the TRT-Edge-LLM streaming ASR stream.

The qwen3_asr_worker prefills the cumulative audio every chunk; the engine KV
cache overflows at ~6.2s (prefill_failed). ``segment_cap_sec`` makes the Python
side rotate to a fresh worker segment at 5.5s — clean cut, no audio carryover,
so no boundary re-transcription/duplication. These tests use a mock worker (no
CUDA / no real worker) to verify:

  * short audio (< cap) never rotates -> single-segment behaviour unchanged;
  * long audio (> cap) rotates and concatenates committed segment text;
  * cap disabled (0) never rotates.

NOTE: this is correctness-of-wiring coverage only. On-device verification
(7.5 / 12.9 / 20s + short-audio latency unchanged) is still required before the
cap is relied on in production.
"""
from __future__ import annotations

import numpy as np

from voxedge.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRConfig,
    _TRTEdgeLLMStreamingASRStream,
)


class _MockBackend:
    """Stands in for TRTEdgeLLMASRBackend's worker IPC surface."""

    def __init__(self, config: TRTEdgeLLMASRConfig, finals):
        self._config = config
        self._finals = list(finals)
        self._final_idx = 0
        self.begin_count = 0
        self.last_true_count = 0

    def _worker_request(self, ev):
        e = ev.get("event")
        if e == "begin":
            self.begin_count += 1
            return {"event": "begin_ack"}
        if e == "end":
            return {"event": "final", "text": ""}
        if e == "chunk":
            if ev.get("last"):
                self.last_true_count += 1
                txt = (
                    self._finals[self._final_idx]
                    if self._final_idx < len(self._finals)
                    else "tail"
                )
                self._final_idx += 1
                return {"event": "final", "text": txt}
            return {"event": "partial", "text": "partial"}
        return {}

    def _strip_language_prefix(self, text):
        return text, None


def _feed(stream, seconds, sr=16000, chunk_s=0.25):
    samp = np.zeros(int(chunk_s * sr), dtype=np.float32)
    for _ in range(int(round(seconds / chunk_s))):
        stream.accept_waveform(sr, samp)


def test_short_audio_no_rotation():
    cfg = TRTEdgeLLMASRConfig(segment_cap_sec=5.5)
    be = _MockBackend(cfg, finals=["hello world"])
    s = _TRTEdgeLLMStreamingASRStream(be)
    assert be.begin_count == 1
    _feed(s, 3.0)  # < 5.5s cap -> never rotates
    assert be.begin_count == 1
    text, _ = s.finalize()
    assert text == "hello world"


def test_long_audio_rotates_and_concatenates():
    cfg = TRTEdgeLLMASRConfig(segment_cap_sec=5.5)
    be = _MockBackend(cfg, finals=["seg one", "seg two", "seg three"])
    s = _TRTEdgeLLMStreamingASRStream(be)
    _feed(s, 12.0)  # rotates at ~5.5s and ~11s -> 2 rotations
    # 1 initial begin + 2 rotation begins
    assert be.begin_count == 3
    text, _ = s.finalize()
    assert text == "seg one seg two seg three"


def test_cap_disabled_no_rotation():
    cfg = TRTEdgeLLMASRConfig(segment_cap_sec=0)
    be = _MockBackend(cfg, finals=["whole thing"])
    s = _TRTEdgeLLMStreamingASRStream(be)
    _feed(s, 12.0)  # would overflow on a real engine, but cap disabled -> no rotate
    assert be.begin_count == 1
    text, _ = s.finalize()
    assert text == "whole thing"


def test_partial_includes_committed_segments():
    cfg = TRTEdgeLLMASRConfig(segment_cap_sec=5.5)
    be = _MockBackend(cfg, finals=["alpha", "beta"])
    s = _TRTEdgeLLMStreamingASRStream(be)
    _feed(s, 6.0)  # one rotation at 5.5s -> "alpha" committed
    partial, is_final = s.get_partial()
    assert not is_final
    assert partial.startswith("alpha")  # committed prefix carried into partials


def test_rotate_uses_partial_when_worker_rotates_on_finalize():
    """B5 (codex): if the worker returns 'segment_rotation' instead of 'final'
    on the forced last=True chunk, _rotate_segment commits the latest partial
    rather than silently dropping the segment."""
    cfg = TRTEdgeLLMASRConfig(segment_cap_sec=5.5)
    be = _MockBackend(cfg, finals=[])
    s = _TRTEdgeLLMStreamingASRStream(be)
    s._partial_text = "partial words"
    # Worker rotates (no 'final') on the forced finalize; _final_text stays "".
    s._send_chunk = lambda *, last: {"event": "segment_rotation", "carryover_sec": 1.0}
    s._begin = lambda: None
    s._rotate_segment()
    assert s._committed_text == "partial words"
