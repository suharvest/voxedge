"""Regression test for the server-loop LLM prefix warm-up (conversation.py).

In server-loop mode the agent skips its local LLM warmup (the LLM runs on the
server), so voxedge must prime edge-llm's prefix cache + CUDA graph once the
advertised system_prompt + tools are known — otherwise the FIRST user turn pays
a cold prefill. This regressed when voice-arm moved to server-loop (agent
skipped, voxedge didn't replace it). _handle_tool_advertise now fires a
fire-and-forget warm-up (gated per prefix signature so reconnect re-advertises
don't re-warm).

These tests do NOT touch production code.
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

from voxedge.backends.base import LLMBackend, LLMEvent
from voxedge.backends.mock import MockTTS, MockVAD
from voxedge.engine import ConversationEngine
from voxedge.engine.conversation import Session
from voxedge.engine.tool_registry import ToolRegistry
from voxedge.transport import InProcessTransport


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())
    wrapper.__name__ = coro_fn.__name__
    return wrapper


class RecordingLLM(LLMBackend):
    """Records every stream_events invocation so a test can assert the warm-up
    fired with the expected prefix + params."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "recording_llm"

    async def stream(self, messages, **kw) -> AsyncIterator[str]:  # pragma: no cover
        if False:
            yield ""

    async def stream_events(self, messages, *, tools=None, **kw) -> AsyncIterator[LLMEvent]:
        self.calls.append({"messages": messages, "tools": tools, "kw": kw})
        yield LLMEvent(kind="text", text="ok")
        yield LLMEvent(kind="finish", finish_reason="stop")


def _session_with_llm(llm: Optional[LLMBackend], with_registry: bool = True) -> Session:
    backends: dict[str, Any] = {"tts": MockTTS(), "vad": MockVAD()}
    if llm is not None:
        backends["llm"] = llm
    engine = ConversationEngine(
        backends=backends,
        tool_registry=ToolRegistry() if with_registry else None,
        system_prompt="BASE",
        multi_utterance=True,
    )
    return Session(engine, InProcessTransport())


def _advertise_payload(tool_names, system_prompt="SYS PROMPT /no_think"):
    return {
        "type": "tool_advertise",
        "system_prompt": system_prompt,
        "tools": [
            {"type": "function", "function": {
                "name": n, "description": f"do {n}",
                "parameters": {"type": "object", "properties": {}},
            }}
            for n in tool_names
        ],
    }


@run_async
async def test_warmup_fires_once_with_advertised_prefix():
    llm = RecordingLLM()
    s = _session_with_llm(llm)
    s._handle_tool_advertise(_advertise_payload(["wave", "home"]))
    assert s._warm_task is not None, "advertise should schedule a warm-up"
    await s._warm_task
    assert len(llm.calls) == 1, f"warm-up should fire exactly once, got {len(llm.calls)}"
    call = llm.calls[0]
    # prefix = advertised system_prompt + a trivial user message
    assert call["messages"][0] == {"role": "system", "content": "SYS PROMPT /no_think"}
    assert call["messages"][-1]["role"] == "user"
    # tools schema carried; generation capped
    assert call["tools"] is not None and len(call["tools"]) == 2
    assert call["kw"].get("max_tokens") == 1


@run_async
async def test_warmup_not_repeated_for_same_prefix():
    llm = RecordingLLM()
    s = _session_with_llm(llm)
    s._handle_tool_advertise(_advertise_payload(["wave", "home"]))
    await s._warm_task
    # Re-advertise the IDENTICAL prefix (e.g. reconnect) → no new warm-up.
    s._handle_tool_advertise(_advertise_payload(["wave", "home"]))
    if s._warm_task is not None and not s._warm_task.done():
        await s._warm_task
    assert len(llm.calls) == 1, "same prefix signature must not re-warm"


@run_async
async def test_warmup_refires_when_prefix_changes():
    llm = RecordingLLM()
    s = _session_with_llm(llm)
    s._handle_tool_advertise(_advertise_payload(["wave"]))
    await s._warm_task
    # Different tool set → prefix changed → warm again.
    s._handle_tool_advertise(_advertise_payload(["wave", "home"]))
    await s._warm_task
    assert len(llm.calls) == 2, "changed prefix signature must re-warm"


@run_async
async def test_no_warmup_without_llm_backend():
    s = _session_with_llm(None)  # no llm backend
    s._handle_tool_advertise(_advertise_payload(["wave"]))
    assert s._warm_task is None, "no LLM backend → no warm-up scheduled"


class FlakyLLM(RecordingLLM):
    """Fails the first warm-up, succeeds after — to prove a failed warm-up
    doesn't permanently suppress retries (signature is set on success only)."""

    def __init__(self, fail_first: int = 1):
        super().__init__()
        self.fail_first = fail_first

    async def stream_events(self, messages, *, tools=None, **kw) -> AsyncIterator[LLMEvent]:
        self.calls.append({"messages": messages, "tools": tools, "kw": kw})
        if len(self.calls) <= self.fail_first:
            raise RuntimeError("simulated edge-llm warm-up failure")
        yield LLMEvent(kind="finish", finish_reason="stop")


@run_async
async def test_failed_warmup_retries_on_next_advertise():
    llm = FlakyLLM(fail_first=1)
    s = _session_with_llm(llm)
    # First advertise: warm-up fails (swallowed) → signature must stay UNSET.
    s._handle_tool_advertise(_advertise_payload(["wave"]))
    await s._warm_task
    assert s._warmed_prefix_sig is None, "failed warm-up must not mark the prefix warmed"
    # Same prefix re-advertised (e.g. reconnect) → must RETRY, not be suppressed.
    s._handle_tool_advertise(_advertise_payload(["wave"]))
    assert s._warm_task is not None
    await s._warm_task
    assert len(llm.calls) == 2, "failed warm-up should retry on the next advertise"
    assert s._warmed_prefix_sig is not None, "successful retry must mark warmed"


@run_async
async def test_no_warmup_without_registry():
    llm = RecordingLLM()
    s = _session_with_llm(llm, with_registry=False)
    s._handle_tool_advertise(_advertise_payload(["wave"]))
    assert s._warm_task is None, "server-loop off (no registry) → no warm-up"
    assert len(llm.calls) == 0
