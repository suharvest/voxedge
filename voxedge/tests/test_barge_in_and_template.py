"""Engine-level regression tests for the 3 barge-in / template fixes
(conversation.py).

Covers, with NO CUDA (mock backends + InProcessTransport):

  #7  template fast-path — after tool dispatch, if EVERY tool this round is
      response_mode=="template" with non-empty completion_text and a non-
      ``success:False`` result, the engine speaks completion_text and skips
      LLM round 2; otherwise round 2 runs. (T1-T5, end-to-end via engine.run)

  #5/#2  cooperative barge-in cancel + _bargein_tts cleanup (T6-T8, white-box
      against a directly-constructed Session).

These tests do NOT touch production code. The fake tool-aware LLM below uses a
``self.calls`` round counter so a test can assert whether round 2 ran:
``calls == 1`` → round 2 skipped, ``calls == 2`` → round 2 ran.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

import numpy as np

from voxedge.backends.base import LLMBackend, LLMEvent
from voxedge.backends.mock import MockASR, MockTTS, MockVAD
from voxedge.engine import ConversationEngine
from voxedge.engine.conversation import Session
from voxedge.engine.tool_registry import ToolRegistry
from voxedge.transport import InProcessTransport


# ── harness (copied from test_engine_inprocess.py; no pytest-asyncio) ──────


def run_async(coro_fn):
    """Decorate a zero-arg async test fn so pytest runs it via asyncio.run."""

    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


def _pcm(loud: bool, n: int = 512) -> bytes:
    """int16 PCM chunk: loud = speech, silent = below VAD threshold."""
    if loud:
        arr = np.ones(n, dtype=np.int16) * 8000
    else:
        arr = np.zeros(n, dtype=np.int16)
    return arr.tobytes()


async def _collect_events(transport: InProcessTransport, run_coro):
    await asyncio.wait_for(run_coro, timeout=10.0)
    events = transport.drain_events_nowait()
    audio = transport.drain_audio_nowait()
    return events, audio


# ── fake tool-aware LLM ────────────────────────────────────────────────────


class FakeToolLLM(LLMBackend):
    """Round 1 emits a tool_call (finish_reason="tool_calls"); round 2+ emits
    a plain text reply (finish_reason="stop").

    ``self.calls`` counts how many times ``stream_events`` was entered, so a
    test can assert round 2 ran (``calls == 2``) or was skipped (``calls == 1``).
    ``tool_names`` is the list of tool names to fire on round 1 — one
    tool_call_delta per name (distinct ``tool_call_index``).
    """

    def __init__(self, tool_names: list[str], round2_text: str = "二轮回复"):
        self.tool_names = tool_names
        self.round2_text = round2_text
        self.calls = 0

    @property
    def name(self) -> str:
        return "fake_tool_llm"

    async def stream(  # pragma: no cover - tool path uses stream_events
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        if False:
            yield ""

    async def stream_events(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        if self.calls == 1:
            for i, tname in enumerate(self.tool_names):
                yield LLMEvent(
                    kind="tool_call_delta",
                    tool_call_index=i,
                    tool_call_id=f"call_{i}",
                    name=tname,
                    arguments="{}",
                )
            yield LLMEvent(kind="finish", finish_reason="tool_calls")
        else:
            yield LLMEvent(kind="text", text=self.round2_text)
            yield LLMEvent(kind="finish", finish_reason="stop")


def _make_registry(tools: list[dict[str, Any]]) -> ToolRegistry:
    """Register local tools. Each spec dict: name, response_mode,
    completion_text, success (handler returns {"success": success})."""
    reg = ToolRegistry()
    for spec in tools:
        success = spec.get("success", True)

        def _handler(success=success) -> dict:
            return {"success": success}

        reg.register(
            spec["name"],
            {"type": "object", "properties": {}},
            _handler,
            response_mode=spec.get("response_mode", "await"),
            completion_text=spec.get("completion_text", ""),
            dispatch_mode="local",
        )
    return reg


async def _drive_tool_turn(engine: ConversationEngine):
    """Run one utterance (loud→silent → VAD endpoint → asr_final → tool loop)
    end-to-end through engine.run; return (events, audio)."""
    transport = InProcessTransport()
    for _ in range(2):
        await transport.feed_audio(_pcm(loud=True))
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=False))
    transport.end_input()
    return await _collect_events(transport, engine.run(transport))


def _engine_with(fake_llm: FakeToolLLM, registry: ToolRegistry) -> ConversationEngine:
    return ConversationEngine(
        backends={
            "asr": MockASR(transcript="请帮我开灯"),
            "vad": MockVAD(silence_chunks=2),
            "llm": fake_llm,
            "tts": MockTTS(),
        },
        tool_registry=registry,
        system_prompt="SYS",
        multi_utterance=False,
    )


def _spoken_sentences(events: list[dict]) -> str:
    """Concatenate the text the engine pushed to TTS (tts_sentence_done)."""
    return "".join(
        e.get("sentence", "") for e in events if e["type"] == "tts_sentence_done"
    )


# ═══════════════════ #7 template fast-path (end-to-end) ═══════════════════


@run_async
async def test_T1_template_success_skips_round2():
    """Single template tool, success → round2 skipped, completion_text spoken."""
    fake = FakeToolLLM(["set_light"])
    reg = _make_registry([
        {"name": "set_light", "response_mode": "template",
         "completion_text": "好的，完成了。", "success": True},
    ])
    engine = _engine_with(fake, reg)
    events, audio = await _drive_tool_turn(engine)

    assert fake.calls == 1, f"round2 should be skipped, calls={fake.calls}"
    spoken = _spoken_sentences(events)
    assert "好的，完成了。" in spoken, f"completion_text not in TTS: {spoken!r}"
    assert "二轮回复" not in spoken
    assert any(e["type"] == "tts_started" for e in events)
    assert sum(len(a) for a in audio) > 0


@run_async
async def test_T2_await_tool_runs_round2():
    """Single await tool → round2 runs, LLM round2 text spoken."""
    fake = FakeToolLLM(["lookup"])
    reg = _make_registry([
        {"name": "lookup", "response_mode": "await",
         "completion_text": "ignored", "success": True},
    ])
    engine = _engine_with(fake, reg)
    events, _ = await _drive_tool_turn(engine)

    assert fake.calls == 2, f"round2 should run, calls={fake.calls}"
    assert "二轮回复" in _spoken_sentences(events)


@run_async
async def test_T3_mixed_template_and_await_runs_round2():
    """Two tools in one round: one template + one await → round2 runs (the
    template fast-path requires ALL tools be template)."""
    fake = FakeToolLLM(["set_light", "lookup"])
    reg = _make_registry([
        {"name": "set_light", "response_mode": "template",
         "completion_text": "好的，完成了。", "success": True},
        {"name": "lookup", "response_mode": "await",
         "completion_text": "", "success": True},
    ])
    engine = _engine_with(fake, reg)
    events, _ = await _drive_tool_turn(engine)

    assert fake.calls == 2, f"mixed round should run round2, calls={fake.calls}"
    assert "二轮回复" in _spoken_sentences(events)


@run_async
async def test_T4_template_failure_falls_back_to_round2():
    """Template tool but handler returns {"success": False} → round2 runs."""
    fake = FakeToolLLM(["set_light"])
    reg = _make_registry([
        {"name": "set_light", "response_mode": "template",
         "completion_text": "好的，完成了。", "success": False},
    ])
    engine = _engine_with(fake, reg)
    events, _ = await _drive_tool_turn(engine)

    assert fake.calls == 2, f"failed template should fall back, calls={fake.calls}"
    assert "二轮回复" in _spoken_sentences(events)


@run_async
async def test_T5_template_empty_completion_runs_round2():
    """Template tool but empty completion_text → round2 runs (empty text
    cannot be spoken, so the fast-path is not taken)."""
    fake = FakeToolLLM(["set_light"])
    reg = _make_registry([
        {"name": "set_light", "response_mode": "template",
         "completion_text": "", "success": True},
    ])
    engine = _engine_with(fake, reg)
    events, _ = await _drive_tool_turn(engine)

    assert fake.calls == 2, f"empty completion should fall back, calls={fake.calls}"
    assert "二轮回复" in _spoken_sentences(events)


# ═══════════════════ #5 / #2 barge-in (white-box Session) ═════════════════


def _session(extra_backends: Optional[dict] = None) -> Session:
    """Construct a Session inside the running loop (the InProcess queues +
    asyncio.get_event_loop() must be created in the loop)."""
    backends: dict[str, Any] = {"tts": MockTTS(), "vad": MockVAD()}
    if extra_backends:
        backends.update(extra_backends)
    engine = ConversationEngine(backends=backends, multi_utterance=True)
    return Session(engine, InProcessTransport())


@run_async
async def test_T6_bargein_drains_queue_and_rebuilds_buffer():
    """#5: _bargein_tts drains _tts_q, rebuilds the TTS buffer, clears
    tts_flush, and sets llm_barged."""
    s = _session()
    # Stale queued sentences + a half-accumulated buffer fragment.
    await s._tts.q.put("残句一")
    await s._tts.q.put("残句二")
    assert s._tts.buffer is not None
    s._tts.buffer.add("半句")  # no terminator → stays buffered
    s.state.tts_flush = True
    old_buffer = s._tts.buffer

    await s._bargein_tts()

    assert s._tts.q.empty(), "queue should be drained"
    assert s.state.llm_barged is True
    assert s.state.tts_flush is False
    assert s._tts.buffer is not old_buffer, "buffer should be rebuilt (new object)"
    assert list(s._tts.buffer.flush()) == [], "rebuilt buffer should be empty"


@run_async
async def test_T7_bargein_awaits_cooperative_llm_task():
    """#2: a cooperative current_llm_task (loops until llm_barged) is stopped
    by the flag and bounded-awaited to completion by _bargein_tts."""
    s = _session()

    async def _coop_turn():
        # Self-stops once the barge-in flag flips.
        while not s.state.llm_barged:
            await asyncio.sleep(0.01)

    task = asyncio.create_task(_coop_turn())
    s.state.current_llm_task = task
    # Let it spin at least once before barging in.
    await asyncio.sleep(0.02)
    assert not task.done()

    await s._bargein_tts()

    assert task.done(), "cooperative task should have stopped and been awaited"
    assert s.state.llm_barged is True


@run_async
async def test_T8_barged_turn_returns_without_flushing():
    """#2: if llm_barged is already set when _llm_turn_with_tools is entered,
    it returns at the round-top checkpoint WITHOUT flushing the buffer / queue
    (no stale text reaches TTS)."""
    fake = FakeToolLLM(["set_light"])
    reg = _make_registry([
        {"name": "set_light", "response_mode": "template",
         "completion_text": "好的，完成了。", "success": True},
    ])
    s = _session(extra_backends={"llm": fake})
    s.engine.tool_registry = reg

    s.state.llm_barged = True  # barge-in landed before the turn started
    await s._llm_turn_with_tools([{"role": "user", "content": "请帮我开灯"}])

    assert fake.calls == 0, "barged turn must not call the LLM"
    assert s._tts.q.empty(), "barged turn must not flush any text to TTS"
    assert s.state.tts_flush is False
