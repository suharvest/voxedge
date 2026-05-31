"""Phase 1 tool-calling tests: registry dispatch + schema + engine pump.

Covers (spec §7 Phase 1 verification):
  * ToolRegistry.dispatch — sync / async / timeout / exception / unknown.
  * list_openai_tools schema shape + ctx exclusion from properties.
  * register_tool (programmatic) path.
  * Session._llm_turn_with_tools — one tool round then text, dispatch fired,
    tool result injected, second round gets text, iteration cap, no-tool
    short-circuit.
  * Regression: tool_registry=None leaves _on_asr_final on the plain path.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from voxedge.backends.base import LLMBackend, LLMEvent
from voxedge.engine import (
    ConversationEngine,
    ToolContext,
    ToolRegistry,
    register_tool,
)
from voxedge.engine.builtin_tools import register_builtins
from voxedge.backends.mock import MockTTS


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


# ─────────────────────────── registry.dispatch ──────────────────────────


@run_async
async def test_dispatch_sync_dict():
    r = ToolRegistry()

    @r.tool(description="add")
    def add(a: int, b: int) -> dict:
        return {"sum": a + b}

    out = await r.dispatch("add", {"a": 2, "b": 3}, None)
    assert out == {"sum": 5}


@run_async
async def test_dispatch_async_handler():
    r = ToolRegistry()

    @r.tool()
    async def slow_echo(msg: str) -> dict:
        await asyncio.sleep(0)
        return {"echo": msg}

    out = await r.dispatch("slow_echo", {"msg": "hi"}, None)
    assert out == {"echo": "hi"}


@run_async
async def test_dispatch_scalar_wrapped():
    r = ToolRegistry()

    @r.tool()
    def plain() -> str:
        return "hello"

    out = await r.dispatch("plain", {}, None)
    assert out == {"value": "hello"}


@run_async
async def test_dispatch_timeout():
    r = ToolRegistry()

    @r.tool(timeout_s=0.01)
    async def hang() -> dict:
        await asyncio.sleep(5)
        return {"ok": True}

    out = await r.dispatch("hang", {}, None)
    assert out["success"] is False
    assert "timed out" in out["error"]


@run_async
async def test_dispatch_exception():
    r = ToolRegistry()

    @r.tool()
    def boom() -> dict:
        raise ValueError("kaboom")

    out = await r.dispatch("boom", {}, None)
    assert out["success"] is False
    assert "kaboom" in out["error"]


@run_async
async def test_dispatch_unknown():
    r = ToolRegistry()
    out = await r.dispatch("nope", {}, None)
    assert out["success"] is False
    assert "unknown tool" in out["error"]


@run_async
async def test_dispatch_filters_unknown_args_and_injects_ctx():
    r = ToolRegistry()
    seen = {}

    @r.tool()
    def needs_ctx(x: int, ctx: Any) -> dict:
        seen["ctx"] = ctx
        return {"x": x}

    ctx = ToolContext(session_id="s1")
    # 'junk' is not in the schema → filtered; ctx never in schema → injected.
    out = await r.dispatch("needs_ctx", {"x": 7, "junk": 99}, ctx)
    assert out == {"x": 7}
    assert seen["ctx"] is ctx


# ─────────────────────────── list_openai_tools ──────────────────────────


def test_list_openai_tools_schema():
    r = ToolRegistry()

    @r.tool(description="weather lookup")
    def get_weather(city: str, ctx: Any) -> dict:
        return {}

    tools = r.list_openai_tools()
    assert len(tools) == 1
    fn = tools[0]
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "get_weather"
    assert fn["function"]["description"] == "weather lookup"
    props = fn["function"]["parameters"]["properties"]
    # ctx excluded from LLM-visible schema; city present + required.
    assert "ctx" not in props
    assert props["city"] == {"type": "string"}
    assert fn["function"]["parameters"]["required"] == ["city"]


def test_list_openai_tools_allow_filter():
    r = ToolRegistry()
    r.tool()(lambda: {})  # anonymous -> name '<lambda>'
    register_tool(r, "a", {"type": "object", "properties": {}}, lambda: {})
    register_tool(r, "b", {"type": "object", "properties": {}}, lambda: {})
    names = {t["function"]["name"] for t in r.list_openai_tools({"a"})}
    assert names == {"a"}


def test_register_tool_programmatic():
    r = ToolRegistry()

    def handler(n: int) -> dict:
        return {"n": n}

    register_tool(
        r,
        "double",
        {"type": "object", "properties": {"n": {"type": "integer"}}},
        handler,
        preamble_text="好的。",
        dispatch_mode="local",
    )
    t = r.get("double")
    assert t is not None
    assert t.preamble_text == "好的。"
    assert t.dispatch_mode == "local"
    tools = r.list_openai_tools()
    assert tools[0]["function"]["parameters"]["properties"]["n"]["type"] == "integer"


def test_builtins_register():
    r = ToolRegistry()
    register_builtins(r)
    assert r.has("time_now")
    assert not r.has("set_mode")  # agent-coupled, intentionally not ported


# ─────────────────────── _llm_turn_with_tools pump ──────────────────────


class _ScriptedLLM(LLMBackend):
    """Emits a pre-scripted list of LLMEvent *rounds*. Each call to
    stream_events pops the next round's events. Records calls for assertions."""

    def __init__(self, rounds: list[list[LLMEvent]]):
        self._rounds = rounds
        self.calls: list[dict] = []

    async def stream(self, messages, **kw) -> AsyncIterator[str]:  # pragma: no cover
        if False:
            yield ""

    async def stream_events(self, messages, **kw) -> AsyncIterator[LLMEvent]:
        self.calls.append({"messages": [dict(m) for m in messages], "kw": kw})
        round_events = self._rounds.pop(0) if self._rounds else [
            LLMEvent(kind="finish", finish_reason="stop")
        ]
        for ev in round_events:
            yield ev


def _make_session(llm, registry, max_tool_rounds=5):
    engine = ConversationEngine(
        backends={"tts": MockTTS(), "llm": llm},
        tool_registry=registry,
        max_tool_rounds=max_tool_rounds,
    )
    from voxedge.engine.conversation import Session

    class _DummyTransport:
        session_id = "sid-test"

        async def send_event(self, p):
            pass

        async def send_audio(self, b):
            pass

    return Session(engine, _DummyTransport())


@run_async
async def test_pump_one_tool_round_then_text():
    registry = ToolRegistry()
    dispatched = {}

    @registry.tool()
    def lookup(city: str) -> dict:
        dispatched["city"] = city
        return {"temp": 21}

    llm = _ScriptedLLM([
        # Round 1: a tool call.
        [
            LLMEvent(kind="tool_call_delta", tool_call_index=0,
                     tool_call_id="call_1", name="lookup",
                     arguments='{"city": "Paris"}'),
            LLMEvent(kind="finish", finish_reason="tool_calls"),
        ],
        # Round 2: final text.
        [
            LLMEvent(kind="text", text="It is 21 degrees."),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    sess = _make_session(llm, registry)
    await sess._llm_turn_with_tools([{"role": "user", "content": "weather in Paris?"}])

    # Tool dispatched with the right args.
    assert dispatched["city"] == "Paris"
    # Two LLM calls (round 1 + continuation).
    assert len(llm.calls) == 2
    # Tools schema was sent (non-None) on each call.
    assert llm.calls[0]["kw"]["tools"] is not None
    # Round-2 messages include the assistant(tool_calls) + role:tool result.
    r2_msgs = llm.calls[1]["messages"]
    roles = [m["role"] for m in r2_msgs]
    assert "assistant" in roles and "tool" in roles
    tool_msg = next(m for m in r2_msgs if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"
    assert "21" in tool_msg["content"]
    # Final text spoken → tts_flush set.
    assert sess.state["tts_flush"] is True


@run_async
async def test_pump_no_tool_short_circuits():
    registry = ToolRegistry()
    llm = _ScriptedLLM([
        [
            LLMEvent(kind="text", text="Hello there."),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    sess = _make_session(llm, registry)
    await sess._llm_turn_with_tools([{"role": "user", "content": "hi"}])
    assert len(llm.calls) == 1  # no continuation
    assert sess.state["tts_flush"] is True


@run_async
async def test_pump_iteration_cap():
    registry = ToolRegistry()

    @registry.tool()
    def loop_tool() -> dict:
        return {"again": True}

    # Every round emits a tool call → never terminates by content.
    forever = [[
        LLMEvent(kind="tool_call_delta", tool_call_index=0,
                 tool_call_id="c", name="loop_tool", arguments="{}"),
        LLMEvent(kind="finish", finish_reason="tool_calls"),
    ] for _ in range(20)]
    llm = _ScriptedLLM(forever)
    sess = _make_session(llm, registry, max_tool_rounds=3)
    await sess._llm_turn_with_tools([{"role": "user", "content": "go"}])
    # Capped at max_tool_rounds LLM calls.
    assert len(llm.calls) == 3
    assert sess.state["tts_flush"] is True


@run_async
async def test_pump_invalid_arguments_json_recovers():
    registry = ToolRegistry()
    fired = {"n": 0}

    @registry.tool()
    def t() -> dict:
        fired["n"] += 1
        return {"ok": True}

    llm = _ScriptedLLM([
        [  # round 1: malformed args
            LLMEvent(kind="tool_call_delta", tool_call_index=0,
                     tool_call_id="c", name="t", arguments="{not json"),
            LLMEvent(kind="finish", finish_reason="tool_calls"),
        ],
        [  # round 2: text
            LLMEvent(kind="text", text="recovered"),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    sess = _make_session(llm, registry)
    await sess._llm_turn_with_tools([{"role": "user", "content": "x"}])
    # Handler never ran (bad JSON), but loop continued and got the error dict
    # injected as the tool result, then terminated on text.
    assert fired["n"] == 0
    r2 = llm.calls[1]["messages"]
    tool_msg = next(m for m in r2 if m["role"] == "tool")
    assert "invalid arguments JSON" in tool_msg["content"]


@run_async
async def test_preamble_spoken_via_tts():
    registry = ToolRegistry()

    @registry.tool(preamble_text="好的。")
    def wave() -> dict:
        return {"started": True}

    llm = _ScriptedLLM([
        [
            LLMEvent(kind="tool_call_delta", tool_call_index=0,
                     tool_call_id="c", name="wave", arguments="{}"),
            LLMEvent(kind="finish", finish_reason="tool_calls"),
        ],
        [
            LLMEvent(kind="text", text="done"),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    sess = _make_session(llm, registry)
    # Drain the TTS queue to confirm the preamble was enqueued.
    await sess._llm_turn_with_tools([{"role": "user", "content": "wave"}])
    queued = []
    while not sess._tts_q.empty():
        queued.append(sess._tts_q.get_nowait())
    assert any("好的" in s for s in queued)


# ───────────────────── regression: None = no tool path ──────────────────


@run_async
async def test_tool_registry_none_uses_plain_path():
    """tool_registry=None → _on_asr_final never enters the tool pump; the LLM
    is called WITHOUT a tools schema (Phase 1 hard contract)."""
    llm = _ScriptedLLM([
        [
            LLMEvent(kind="text", text="plain reply"),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    engine = ConversationEngine(
        backends={"tts": MockTTS(), "llm": llm},
        tool_registry=None,
    )
    from voxedge.engine.conversation import Session

    class _DummyTransport:
        async def send_event(self, p):
            pass

        async def send_audio(self, b):
            pass

    sess = Session(engine, _DummyTransport())
    await sess._on_asr_final("hello")
    # Exactly one call, and NO tools kwarg passed (plain stream_events path).
    assert len(llm.calls) == 1
    assert "tools" not in llm.calls[0]["kw"]
    assert sess.state["tts_flush"] is True
