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
    TRTEdgeLLMASRBackend,
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

    # 流式路径统一走 _postprocess_text（剥语言前缀 + 退化塌缩）。这里委托到真实
    # 实现而不是返回原文，否则 mock 会悄悄绕过塌缩守卫，测试就盖不住它。
    def _postprocess_text(self, text):
        return TRTEdgeLLMASRBackend._postprocess_text(self, text)


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


# ── 退化塌缩：partial 被晋升为定稿的两条路径 ────────────────────────────
# codex review 2026-08-08 指出原测试只覆盖轮转拼接，没验证新加的塌缩。

class _RotationBackend(_MockBackend):
    """强制 finalize 返回 segment_rotation 而非 final，逼走 partial 兜底路径。"""

    def __init__(self, config, partial_text):
        super().__init__(config, finals=[])
        self._partial_text = partial_text

    def _worker_request(self, ev):
        e = ev.get("event")
        if e == "begin":
            self.begin_count += 1
            return {"event": "begin_ack"}
        if e == "chunk":
            if ev.get("last"):
                self.last_true_count += 1
                return {"event": "segment_rotation", "carryover_sec": 0.0}
            return {"event": "partial", "text": self._partial_text}
        return {}


def _rotating_stream(partial_text, *, collapse=True):
    cfg = TRTEdgeLLMASRConfig(collapse_repetition=collapse)
    be = _RotationBackend(cfg, partial_text)
    return _TRTEdgeLLMStreamingASRStream(be), be


def test_rotation_fallback_collapses_degenerate_partial():
    stream, _ = _rotating_stream("帮我，" * 20)
    _feed(stream, seconds=1.0)
    stream._rotate_segment()
    assert stream._committed_text == "帮我", (
        f"轮转兜底把未塌缩的退化 partial 提交了: {stream._committed_text!r}"
    )


def test_rotation_fallback_respects_disable_switch():
    raw = "帮我，" * 20
    stream, _ = _rotating_stream(raw, collapse=False)
    _feed(stream, seconds=1.0)
    stream._rotate_segment()
    assert stream._committed_text == raw.strip(), "关掉开关后不应塌缩"


def test_cancel_and_finalize_collapses_degenerate_partial():
    stream, _ = _rotating_stream("帮我，" * 20)
    _feed(stream, seconds=1.0)
    stream._partial_text = "帮我，" * 20
    try:
        stream.cancel_and_finalize()
    except Exception:
        pass  # end 事件在 mock 上可能抛错，这里只关心 _final_text
    assert stream._final_text == "帮我", (
        f"cancel_and_finalize 晋升了未塌缩的 partial: {stream._final_text!r}"
    )
