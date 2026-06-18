"""P2b: server-loop LLM-stream semantic timeout (turn_driver watchdog).

Directly drives ``run_turn`` with a hanging LLM to prove the first-token
watchdog fires and that the two error policies behave as the P2b design
requires:

  * server adapter (``reraise_errors=False``) → the timeout is swallowed into
    a graceful flush + ``None`` return (the turn ends instead of hanging to the
    backend's coarse httpx read timeout);
  * client shim (``reraise_errors=True``) → the timeout propagates so the
    agent's LLMTimeoutError / availability machinery can handle it.

See docs/plans/turn-driver-unification.md §6d (P2b).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

import pytest

from voxedge.backends.base import LLMEvent
from voxedge.engine.tool_registry import ToolContext, ToolRegistry
from voxedge.engine.turn_driver import run_turn


class _HangingLLM:
    """``stream_events`` that never yields its first event → first-token
    timeout territory."""

    def name(self) -> str:  # pragma: no cover - trivial
        return "hang"

    async def stream_events(
        self, messages, *, tools=None, **kw
    ) -> AsyncIterator[LLMEvent]:
        await asyncio.Event().wait()  # hangs until cancelled / aclosed
        yield LLMEvent(kind="finish", finish_reason="stop")  # unreachable


class _RecordingTextSink:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.flushed = 0

    async def text(self, s: str) -> None:  # pragma: no cover - unused here
        self.texts.append(s)

    async def preamble(self, s: str) -> None:  # pragma: no cover - unused
        self.texts.append(s)

    async def flush(self) -> None:
        self.flushed += 1


class _ListMessageSink:
    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []

    def working_messages(self) -> list[dict[str, Any]]:
        return self._messages

    def add_assistant_tool_calls(self, content, tool_calls) -> None:  # pragma: no cover
        self._messages.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls}
        )

    def add_assistant_text(self, content) -> None:  # pragma: no cover
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id, content) -> None:  # pragma: no cover
        self._messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )


def _ctx() -> ToolContext:
    return ToolContext(session_id=None, conversation=None, remote_send=None)


def _kwargs(text_sink, **over):
    base: dict[str, Any] = dict(
        llm=_HangingLLM(),
        registry=ToolRegistry(),
        msg_sink=_ListMessageSink(),
        text_sink=text_sink,
        should_abort=lambda: False,
        ctx=_ctx(),
        llm_params={},
        max_rounds=3,
        first_token_timeout_s=0.05,
        idle_timeout_s=0.05,
    )
    base.update(over)
    return base


def test_server_policy_swallows_timeout_into_graceful_end():
    """reraise_errors=False (server adapter): the first-token timeout ends the
    turn gracefully — run_turn returns None, the TTS sink is flushed, and no
    exception escapes."""
    sink = _RecordingTextSink()

    async def go():
        return await asyncio.wait_for(
            run_turn(**_kwargs(sink, reraise_errors=False)), timeout=5.0
        )

    result = asyncio.run(go())
    assert result is None
    assert sink.flushed >= 1  # graceful flush ran (turn ended, did not hang)


def test_client_policy_propagates_timeout():
    """reraise_errors=True (client shim): the timeout propagates so the agent
    layer can map it to LLMTimeoutError / recovery."""
    sink = _RecordingTextSink()

    async def go():
        return await asyncio.wait_for(
            run_turn(**_kwargs(sink, reraise_errors=True)), timeout=5.0
        )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
