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


def test_description_lives_only_on_function_not_parameters():
    """Regression: the tool-level description must be emitted ONCE at
    function.description and NEVER duplicated inside parameters. The advertise
    path used to merge description into the schema, so list_openai_tools
    emitted it twice — a 12-tool prompt then overflowed the edge-llm 3000-token
    cap (input_too_long, every command 400). See cutover 3b-i."""
    r = ToolRegistry()

    # Explicit description arg (server-loop advertise path).
    register_tool(
        r,
        "dance",
        {"type": "object", "properties": {}},
        lambda ctx=None: {},
        dispatch_mode="remote",
        description="Make the arm perform a dance routine.",
    )
    # Legacy/foreign schema that leaked a root description into parameters.
    register_tool(
        r,
        "wave",
        {
            "type": "object",
            "description": "leaked tool description",
            "properties": {"hand": {"type": "string", "description": "which hand"}},
        },
        lambda hand="left", ctx=None: {},
        dispatch_mode="remote",
    )

    tools = {t["function"]["name"]: t["function"] for t in r.list_openai_tools()}

    # Explicit description surfaces at function-level, absent from parameters.
    assert tools["dance"]["description"] == "Make the arm perform a dance routine."
    assert "description" not in tools["dance"]["parameters"]

    # Leaked root description is stripped from parameters but preserved as the
    # tool description; nested property descriptions are left intact.
    assert tools["wave"]["description"] == "leaked tool description"
    assert "description" not in tools["wave"]["parameters"]
    assert tools["wave"]["parameters"]["properties"]["hand"]["description"] == "which hand"


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
    assert sess.state.tts_flush is True


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
    assert sess.state.tts_flush is True


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
    assert sess.state.tts_flush is True


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
    assert sess.state.tts_flush is True


# ─────────────────── Phase 2a: remote dispatch ──────────────────────


def _remote_registry(timeout_s: float = 5.0) -> ToolRegistry:
    r = ToolRegistry()

    @r.tool(description="set lamp", dispatch_mode="remote", timeout_s=timeout_s)
    def set_lamp(on: bool) -> dict:  # body never runs (remote)
        return {"unreachable": True}

    return r


@run_async
async def test_remote_dispatch_payload_and_result():
    r = _remote_registry()
    sent: list[dict] = []

    async def remote_send(frame: dict) -> None:
        sent.append(frame)
        # Simulate the client replying out-of-band.
        r.resolve_remote(frame["call_id"], result={"ok": True, "lamp": "on"})

    ctx = ToolContext(remote_send=remote_send)
    out = await r.dispatch("set_lamp", {"on": True}, ctx)

    assert out == {"ok": True, "lamp": "on"}
    assert len(sent) == 1
    frame = sent[0]
    assert frame["type"] == "tool_call"
    assert frame["name"] == "set_lamp"
    assert frame["arguments"] == {"on": True}
    assert isinstance(frame["call_id"], str) and frame["call_id"]
    # Future cleaned up after dispatch.
    assert r._pending_remote == {}


@run_async
async def test_remote_dispatch_deferred_resolve():
    r = _remote_registry()
    captured: dict = {}

    async def remote_send(frame: dict) -> None:
        captured["call_id"] = frame["call_id"]

    ctx = ToolContext(remote_send=remote_send)
    task = asyncio.ensure_future(r.dispatch("set_lamp", {"on": False}, ctx))
    # Let dispatch send the frame and register the pending future.
    await asyncio.sleep(0)
    assert captured["call_id"] in r._pending_remote
    r.resolve_remote(captured["call_id"], result={"value": 42})
    out = await task
    assert out == {"value": 42}
    assert r._pending_remote == {}


@run_async
async def test_remote_dispatch_error_becomes_error_dict():
    r = _remote_registry()

    async def remote_send(frame: dict) -> None:
        r.resolve_remote(frame["call_id"], error="device offline")

    ctx = ToolContext(remote_send=remote_send)
    out = await r.dispatch("set_lamp", {"on": True}, ctx)
    assert out["success"] is False
    assert "device offline" in out["error"]
    assert r._pending_remote == {}


@run_async
async def test_remote_dispatch_non_dict_result_wrapped():
    r = _remote_registry()

    async def remote_send(frame: dict) -> None:
        r.resolve_remote(frame["call_id"], result="plain-string")

    ctx = ToolContext(remote_send=remote_send)
    out = await r.dispatch("set_lamp", {"on": True}, ctx)
    assert out == {"value": "plain-string"}


@run_async
async def test_remote_dispatch_timeout():
    r = _remote_registry(timeout_s=0.05)

    async def remote_send(frame: dict) -> None:
        pass  # never resolves

    ctx = ToolContext(remote_send=remote_send)
    out = await r.dispatch("set_lamp", {"on": True}, ctx)
    assert out["success"] is False
    assert out["error"] == "timeout"
    assert out["timeout_s"] == 0.05
    # Future must be cleared even on timeout.
    assert r._pending_remote == {}


@run_async
async def test_remote_dispatch_no_transport():
    r = _remote_registry()
    out = await r.dispatch("set_lamp", {"on": True}, ToolContext(remote_send=None))
    assert out["success"] is False
    assert out["error"] == "no remote transport"
    assert r._pending_remote == {}


@run_async
async def test_remote_dispatch_no_transport_ctx_none():
    r = _remote_registry()
    out = await r.dispatch("set_lamp", {"on": True}, None)
    assert out["success"] is False
    assert out["error"] == "no remote transport"


@run_async
async def test_resolve_unknown_call_id_ignored():
    r = _remote_registry()
    # No exception; safe no-op.
    r.resolve_remote("does-not-exist", result={"x": 1})
    r.resolve_remote("does-not-exist", error="boom")
    assert r._pending_remote == {}


@run_async
async def test_resolve_already_done_ignored():
    r = _remote_registry()
    captured: dict = {}

    async def remote_send(frame: dict) -> None:
        captured["call_id"] = frame["call_id"]

    ctx = ToolContext(remote_send=remote_send)
    task = asyncio.ensure_future(r.dispatch("set_lamp", {"on": True}, ctx))
    await asyncio.sleep(0)
    cid = captured["call_id"]
    r.resolve_remote(cid, result={"first": True})
    out = await task
    # Second resolve after completion is ignored (no crash).
    r.resolve_remote(cid, result={"second": True})
    assert out == {"first": True}


@run_async
async def test_cancel_pending_remote_clears_futures():
    r = _remote_registry(timeout_s=10.0)
    captured: list[str] = []

    async def remote_send(frame: dict) -> None:
        captured.append(frame["call_id"])

    ctx = ToolContext(remote_send=remote_send)
    task = asyncio.ensure_future(r.dispatch("set_lamp", {"on": True}, ctx))
    await asyncio.sleep(0)
    assert len(r._pending_remote) == 1
    cleared = r.cancel_pending_remote()
    assert cleared == 1
    assert r._pending_remote == {}
    out = await task
    assert out["success"] is False
    assert out["error"] == "cancelled"


@run_async
async def test_local_dispatch_unchanged_by_remote_path():
    # Regression: a local tool in the same registry still runs in-process.
    r = ToolRegistry()

    @r.tool(description="add")
    def add(a: int, b: int) -> dict:
        return {"sum": a + b}

    @r.tool(description="remote", dispatch_mode="remote")
    def rem(x: int) -> dict:
        return {"unreachable": True}

    out = await r.dispatch("add", {"a": 4, "b": 5}, ToolContext())
    assert out == {"sum": 9}
    assert r._pending_remote == {}


# ──────────── system prompt + llm_params injection (#37 step) ────────────


def _make_session_cfg(llm, registry, *, system_prompt=None, llm_params=None,
                      max_tool_rounds=5):
    engine = ConversationEngine(
        backends={"tts": MockTTS(), "llm": llm},
        tool_registry=registry,
        system_prompt=system_prompt,
        llm_params=llm_params,
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
async def test_system_prompt_prepended_to_tool_pump():
    """A configured system_prompt is prepended once as the first message on
    every LLM round (stable prefix for the edge-LLM cache, spec §8)."""
    registry = ToolRegistry()
    llm = _ScriptedLLM([
        [
            LLMEvent(kind="text", text="Hi."),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    sess = _make_session_cfg(llm, registry, system_prompt="You are a robot.")
    await sess._llm_turn_with_tools([{"role": "user", "content": "hello"}])
    msgs = llm.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "You are a robot."}
    assert msgs[1]["role"] == "user"


@run_async
async def test_system_prompt_prepended_once_across_rounds():
    """Across a multi-round tool pump the system prompt appears exactly once
    (round-2 re-send must not duplicate it)."""
    registry = ToolRegistry()

    @registry.tool()
    def ping() -> dict:
        return {"pong": True}

    llm = _ScriptedLLM([
        [
            LLMEvent(kind="tool_call_delta", tool_call_index=0,
                     tool_call_id="c1", name="ping", arguments="{}"),
            LLMEvent(kind="finish", finish_reason="tool_calls"),
        ],
        [
            LLMEvent(kind="text", text="done"),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    sess = _make_session_cfg(llm, registry, system_prompt="SYS")
    await sess._llm_turn_with_tools([{"role": "user", "content": "go"}])
    assert len(llm.calls) == 2
    r2 = llm.calls[1]["messages"]
    assert [m for m in r2 if m["role"] == "system"] == [
        {"role": "system", "content": "SYS"}
    ]


@run_async
async def test_no_system_prompt_leaves_messages_unchanged():
    registry = ToolRegistry()
    llm = _ScriptedLLM([
        [LLMEvent(kind="text", text="ok"),
         LLMEvent(kind="finish", finish_reason="stop")],
    ])
    sess = _make_session_cfg(llm, registry, system_prompt=None)
    await sess._llm_turn_with_tools([{"role": "user", "content": "hi"}])
    assert llm.calls[0]["messages"][0]["role"] == "user"


@run_async
async def test_llm_params_forwarded_to_stream_events():
    """Configured llm_params (temperature / max_tokens) reach the backend
    stream_events kwargs on every round."""
    registry = ToolRegistry()
    llm = _ScriptedLLM([
        [LLMEvent(kind="text", text="ok"),
         LLMEvent(kind="finish", finish_reason="stop")],
    ])
    sess = _make_session_cfg(
        llm, registry, llm_params={"temperature": 0.3, "max_tokens": 64}
    )
    await sess._llm_turn_with_tools([{"role": "user", "content": "hi"}])
    kw = llm.calls[0]["kw"]
    assert kw["temperature"] == 0.3
    assert kw["max_tokens"] == 64


# ───────── tool_result wire frame → resolve_remote (receive side) ─────────


@run_async
async def test_client_tool_result_frame_resolves_remote():
    """End-to-end: a remote tool sends a SERVER_TOOL_CALL via the transport;
    the client's CLIENT_TOOL_RESULT frame is routed by the engine event loop
    to registry.resolve_remote, unblocking the dispatch."""
    from voxedge.engine.conversation import Session
    from voxedge.backends.base import LLMBackend

    registry = ToolRegistry()

    @registry.tool(description="wave", dispatch_mode="remote", timeout_s=5.0)
    def wave() -> dict:  # body never runs (remote)
        return {}

    # Scripted LLM: round 1 calls the remote tool, round 2 speaks.
    llm = _ScriptedLLM([
        [
            LLMEvent(kind="tool_call_delta", tool_call_index=0,
                     tool_call_id="c1", name="wave", arguments="{}"),
            LLMEvent(kind="finish", finish_reason="tool_calls"),
        ],
        [
            LLMEvent(kind="text", text="waved"),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    engine = ConversationEngine(
        backends={"tts": MockTTS(), "llm": llm},
        tool_registry=registry,
    )

    sent: list[dict] = []

    class _CaptureTransport:
        session_id = "sid"

        async def send_event(self, p):
            sent.append(p)

        async def send_audio(self, b):
            pass

    sess = Session(engine, _CaptureTransport())

    # Run the pump; concurrently feed the tool_result frame through the same
    # receive handler the real /v2v event loop uses.
    pump = asyncio.create_task(
        sess._llm_turn_with_tools([{"role": "user", "content": "wave"}])
    )

    # Wait until the remote dispatch frame has been sent.
    for _ in range(200):
        tool_calls = [s for s in sent if s.get("type") == "tool_call"]
        if tool_calls:
            break
        await asyncio.sleep(0.005)
    tool_calls = [s for s in sent if s.get("type") == "tool_call"]
    assert tool_calls, "remote tool_call frame never sent"
    call_id = tool_calls[0]["call_id"]

    # Simulate the device client's reply via the engine receive routing.
    # (Directly exercise the same call the _event_loop CLIENT_TOOL_RESULT
    # branch makes.)
    registry.resolve_remote(call_id, result={"started": True, "action": "wave"})

    await asyncio.wait_for(pump, timeout=5.0)
    # Round 2 ran → tool result injected as role:tool, final text spoken.
    assert len(llm.calls) == 2
    r2 = llm.calls[1]["messages"]
    tool_msg = next(m for m in r2 if m["role"] == "tool")
    assert "wave" in tool_msg["content"]
    assert registry._pending_remote == {}


@run_async
async def test_event_loop_routes_tool_result_to_resolve_remote():
    """The engine ``_event_loop`` routes an inbound CLIENT_TOOL_RESULT frame
    (ok / error variants) to ``registry.resolve_remote`` (the receive-side
    wire hook added for spec §4 Mode B)."""
    from voxedge.engine.conversation import Session, CLIENT_TOOL_RESULT

    registry = ToolRegistry()
    engine = ConversationEngine(
        backends={"tts": MockTTS()},
        tool_registry=registry,
    )

    frames = [
        {"type": CLIENT_TOOL_RESULT, "call_id": "ok1", "result": {"v": 1}},
        {"type": CLIENT_TOOL_RESULT, "call_id": "err1", "ok": False,
         "error": "boom"},
    ]

    class _FeedTransport:
        session_id = "sid"

        async def recv_event(self):
            for f in frames:
                yield f

        async def send_event(self, p):
            pass

        async def send_audio(self, b):
            pass

    sess = Session(engine, _FeedTransport())

    # Pre-register two pending remote futures so the event loop can resolve
    # them, mirroring what _dispatch_remote would have created.
    loop = asyncio.get_event_loop()
    ok_fut = loop.create_future()
    err_fut = loop.create_future()
    registry._pending_remote["ok1"] = ok_fut
    registry._pending_remote["err1"] = err_fut

    await sess._event_loop()

    assert ok_fut.result() == {"v": 1}
    assert isinstance(err_fut.exception(), RuntimeError)
    assert "boom" in str(err_fut.exception())


# ─────────────────────── tool advertise handshake ───────────────────────


@run_async
async def test_event_loop_tool_advertise_registers_remote_tools():
    """CLIENT_TOOL_ADVERTISE registers the advertised OpenAI schemas as
    dispatch_mode="remote" tools, exposes them via list_openai_tools, and
    overrides the engine system_prompt / merges llm_params (spec §4/§6)."""
    from voxedge.engine.conversation import Session, CLIENT_TOOL_ADVERTISE

    registry = ToolRegistry()
    engine = ConversationEngine(
        backends={"tts": MockTTS()},
        tool_registry=registry,
        system_prompt="old prompt",
        llm_params={"temperature": 0.1},
    )

    advertise = {
        "type": CLIENT_TOOL_ADVERTISE,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "look_up",
                    "description": "Tilt the arm up to look.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            # Bare (un-wrapped) form is also accepted.
            {
                "name": "wave",
                "description": "Wave the arm.",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
        "system_prompt": "You are a robot arm.",
        "llm_params": {"max_tokens": 64},
    }

    class _FeedTransport:
        session_id = "sid"

        async def recv_event(self):
            yield advertise

        async def send_event(self, p):
            pass

        async def send_audio(self, b):
            pass

    sess = Session(engine, _FeedTransport())
    await sess._event_loop()

    # Both tools registered as remote-dispatch.
    assert registry.has("look_up") and registry.has("wave")
    assert registry.get("look_up").dispatch_mode == "remote"
    assert registry.get("wave").dispatch_mode == "remote"

    # list_openai_tools surfaces them in OpenAI chat-completions shape.
    names = {t["function"]["name"] for t in registry.list_openai_tools()}
    assert names == {"look_up", "wave"}
    look = next(t for t in registry.list_openai_tools()
                if t["function"]["name"] == "look_up")
    assert look["type"] == "function"
    assert look["function"]["description"] == "Tilt the arm up to look."

    # System prompt overridden; llm_params merged (not replaced).
    assert engine.system_prompt == "You are a robot arm."
    assert engine.llm_params == {"temperature": 0.1, "max_tokens": 64}


@run_async
async def test_advertise_then_server_loop_picks_remote_tool_e2e():
    """Full server-loop handshake: client advertises a remote arm tool →
    scripted LLM selects it → engine proxies SERVER_TOOL_CALL to the client →
    client replies tool_result → LLM continuation speaks. Proves the advertised
    tool is visible to the LLM request AND flows through remote dispatch."""
    from voxedge.engine.conversation import (
        Session,
        CLIENT_TOOL_ADVERTISE,
        SERVER_TOOL_CALL,
    )

    registry = ToolRegistry()

    # LLM round 1 picks look_up; round 2 (after tool_result) speaks.
    llm = _ScriptedLLM([
        [
            LLMEvent(kind="tool_call_delta", tool_call_index=0,
                     tool_call_id="call_arm", name="look_up", arguments="{}"),
            LLMEvent(kind="finish", finish_reason="tool_calls"),
        ],
        [
            LLMEvent(kind="text", text="好的，我抬头看看。"),
            LLMEvent(kind="finish", finish_reason="stop"),
        ],
    ])
    engine = ConversationEngine(
        backends={"tts": MockTTS(), "llm": llm},
        tool_registry=registry,
    )

    # Advertise the arm tool first (handshake).
    sess0 = Session(engine, _CaptureTransport([]))
    sess0._handle_tool_advertise({
        "type": CLIENT_TOOL_ADVERTISE,
        "tools": [{
            "type": "function",
            "function": {
                "name": "look_up",
                "description": "Tilt the arm up to look.",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
        "system_prompt": "You are a robot arm assistant.",
    })
    # The advertise handshake fires a best-effort LLM prefix warm-up as an
    # un-awaited background task (Session._warm_llm_prefix → one extra
    # stream_events call). It shares this test's finite _ScriptedLLM, so if the
    # event loop schedules it mid-pump (more likely on slower hardware) it
    # consumes a scripted round and inflates llm.calls non-deterministically.
    # Cancel it now — before any await, so it never runs — so this test
    # measures only the tool pump. Warm-up has its own coverage in
    # test_llm_prefix_warmup.py.
    if sess0._warm_task is not None:
        sess0._warm_task.cancel()
    assert registry.get("look_up").dispatch_mode == "remote"

    # Drive the server-side pump on the SAME engine/registry. The capture
    # transport records the SERVER_TOOL_CALL frame and auto-replies with a
    # success tool_result so the remote dispatch future resolves.
    transport = _CaptureTransport([], registry=registry)
    sess = Session(engine, transport)
    await sess._llm_turn_with_tools(
        [{"role": "user", "content": "抬头看看"}]
    )

    # Server emitted a SERVER_TOOL_CALL for the advertised remote tool.
    tool_calls = [e for e in transport.sent if e.get("type") == SERVER_TOOL_CALL]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "look_up"

    # The tools schema sent to the LLM included the advertised tool.
    assert any(
        t["function"]["name"] == "look_up"
        for t in (llm.calls[0]["kw"].get("tools") or [])
    )

    # Continuation ran (round 2) after the tool_result, with the role:tool
    # result injected — the full closed loop completed.
    assert len(llm.calls) == 2
    r2_roles = [m["role"] for m in llm.calls[1]["messages"]]
    assert "tool" in r2_roles
    assert sess.state.tts_flush is True


class _CaptureTransport:
    """Records sent events. When ``registry`` is given, auto-resolves a
    SERVER_TOOL_CALL by feeding back a success tool_result (simulating a
    device client that ran the arm action), so remote dispatch completes
    without a live WS peer."""

    session_id = "sid"

    def __init__(self, events, registry=None):
        self._events = events
        self.sent: list[dict] = []
        self._registry = registry

    async def recv_event(self):
        for e in self._events:
            yield e

    async def send_event(self, p):
        self.sent.append(p)
        if (
            self._registry is not None
            and isinstance(p, dict)
            and p.get("type") == "tool_call"
        ):
            # Mimic the device client returning a successful arm result.
            self._registry.resolve_remote(
                p["call_id"], result={"started": True, "action": p["name"]}
            )

    async def send_audio(self, b):
        pass
