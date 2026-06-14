"""Provider-agnostic, no-I/O multi-turn LLM ↔ tool pump.

This module hosts the *pure algorithm* of the server-side multi-round
LLM↔tool loop, extracted verbatim (byte-equivalent behaviour) from
``voxedge.engine.llm_turn._LLMTurn.run`` (turn-driver-unification spec P0).

The driver knows nothing about ``Session`` / ``ConversationEngine`` /
transport. It talks to the host only through three seams:

* ``TextSink`` — assistant text + completion_text + per-tool preamble, and an
  end-of-turn ``flush``/signal.
* ``MessageSink`` — owns the working message list (append-only here); the
  caller is responsible for the system-prompt prefix BEFORE calling
  ``run_turn`` (the sink protocol has no ``add_system``).
* ``should_abort: Callable[[], bool]`` — cooperative barge/cancel poll,
  checked at every barge checkpoint exactly as the original.

Strategy is explicit (never hard-coded):

* ``preamble_dedup``: ``"name"`` (server semantics) | ``"index"``.
* ``template_fastpath``: ``"all_join"`` (server semantics) | ``"any_first"``.

P0 callers (server) pass ``preamble_dedup="name"`` and
``template_fastpath="all_join"`` to preserve ``llm_turn.py`` semantics
byte-for-byte. Only the ``"name"`` / ``"all_join"`` branches are exercised by
P0; the ``"index"`` / ``"any_first"`` branches exist for the P1 client wiring
and are written to mirror the agent runner, but P0 must not depend on them.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol


logger = logging.getLogger(__name__)


@dataclass
class _ToolCallAcc:
    """Accumulator for one tool_call's streamed deltas (per OpenAI index slot).

    Mirrors agent/openvoicestream_agent/tools/runner.py:41-49."""

    id: str = ""
    name: str = ""
    arguments: str = ""


class TextSink(Protocol):
    """Where assistant text / completion_text / preamble go, plus end flush."""

    async def text(self, s: str) -> None: ...        # assistant text + completion_text
    async def preamble(self, s: str) -> None: ...    # tool preamble
    async def flush(self) -> None: ...               # end-of-turn flush/signal


class MessageSink(Protocol):
    """Owns the working message list. Append-only (server). No add_system —
    the system prefix is the caller's responsibility (spec §4 / D3)."""

    def add_assistant_tool_calls(
        self, content: Optional[str], tool_calls: list[dict[str, Any]]
    ) -> None: ...
    def add_assistant_text(self, content: str) -> None: ...
    def add_tool_result(self, tool_call_id: str, content: str) -> None: ...
    def working_messages(self) -> list[dict[str, Any]]: ...


async def run_turn(
    *,
    llm: Any,
    registry: Any,
    msg_sink: MessageSink,
    text_sink: TextSink,
    should_abort: Callable[[], bool],
    ctx: Any,
    llm_params: dict[str, Any],
    max_rounds: int,
    preamble_dedup: str = "name",
    template_fastpath: str = "all_join",
) -> None:
    """Pure multi-turn LLM ↔ tool pump (spec §2/§4).

    Byte-equivalent port of ``_LLMTurn.run`` for the server semantics
    (``preamble_dedup="name"``, ``template_fastpath="all_join"``).

    The caller MUST have already placed the system prompt at the head of
    ``msg_sink.working_messages()`` if one is desired — this function never
    inserts a system message.

    Loop (≤ ``max_rounds`` iterations):
      1. Stream ``llm.stream_events(messages, tools=schema)``.
      2. Text events → ``text_sink.text``.
      3. Accumulate ``tool_call_delta`` events per OpenAI tool-call index.
      4. finish_reason != "tool_calls" (or no tool calls) → done.
      5. Else: append assistant(tool_calls) via ``msg_sink``, dispatch each
         handler, append role:"tool" results, re-request the LLM.
    """
    tools_schema = registry.list_openai_tools() or None
    try:
        for _round in range(max_rounds):
            text_chunks: list[str] = []
            tool_accs: dict[int, _ToolCallAcc] = {}
            finish_reason: Optional[str] = None
            # preamble dedup key set: tool names ("name") or call indices ("index")
            preamble_fired: set = set()

            messages = msg_sink.working_messages()

            # ── latency instrumentation ───────────────────────────────
            _t_round_start = time.perf_counter()
            _ttft: Optional[float] = None
            _ctx_msgs = len(messages)

            # #2: barged in between rounds → drop this turn (discard any
            # partial text). Returning abandons the stream generator so the
            # LLM connection closes.
            if should_abort():
                return

            async for ev in llm.stream_events(
                messages, tools=tools_schema, **llm_params
            ):
                # #2: barge-in mid-stream — stop consuming deltas and drop
                # the turn WITHOUT flushing the partial sentence buffer.
                if should_abort():
                    return
                if _ttft is None:
                    _ttft = time.perf_counter() - _t_round_start
                if ev.kind == "text" and ev.text:
                    text_chunks.append(ev.text)
                    await text_sink.text(ev.text)
                elif ev.kind == "tool_call_delta":
                    idx = ev.tool_call_index if ev.tool_call_index is not None else 0
                    slot = tool_accs.setdefault(idx, _ToolCallAcc())
                    if ev.tool_call_id:
                        slot.id = ev.tool_call_id
                    if ev.name:
                        slot.name = ev.name
                        # Early-fire the per-tool preamble as soon as the
                        # tool name is known (lowest voice latency).
                        key = idx if preamble_dedup == "index" else ev.name
                        if key not in preamble_fired:
                            tool = registry.get(ev.name)
                            pre = (getattr(tool, "preamble_text", "") or "") if tool else ""
                            if pre:
                                preamble_fired.add(key)
                                await text_sink.preamble(pre)
                    if ev.arguments:
                        slot.arguments += ev.arguments
                elif ev.kind == "finish":
                    finish_reason = ev.finish_reason

            logger.info(
                "voxedge tool loop: round=%d ctx_msgs=%d ttft=%.3fs "
                "stream=%.3fs n_text=%d finish=%s n_tools=%d",
                _round, _ctx_msgs,
                (_ttft if _ttft is not None else -1.0),
                time.perf_counter() - _t_round_start,
                len(text_chunks), finish_reason, len(tool_accs),
            )

            # No tool call → terminal text answer. Flush + done.
            if not tool_accs or finish_reason != "tool_calls":
                # A barge-in can land in the await between the last stream
                # event and here (the per-event guard above won't catch it).
                if should_abort():
                    return
                final_text = "".join(text_chunks)
                if final_text:
                    msg_sink.add_assistant_text(final_text)
                await text_sink.flush()
                return

            # Commit assistant(tool_calls) to the message list.
            preamble_content = "".join(text_chunks) or None
            tc_payload: list[dict[str, Any]] = [
                {
                    "id": acc.id or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": acc.name,
                        "arguments": acc.arguments or "{}",
                    },
                }
                for idx, acc in sorted(tool_accs.items())
            ]
            msg_sink.add_assistant_tool_calls(preamble_content, tc_payload)

            # Dispatch each tool sequentially. registry.dispatch branches on
            # dispatch_mode internally: local tools run in-process, remote
            # tools proxy over ctx.remote_send and await a correlated
            # tool_result — both return a JSON-serialisable dict.
            dispatched: list[tuple[str, str, str, Any]] = []
            for idx, tc in enumerate(tc_payload):
                tname = tc["function"]["name"]
                # Fallback preamble: fire here if the streamed name delta never
                # triggered the early path above (e.g. backend sent the whole
                # tool_call in one finish chunk).
                key = idx if preamble_dedup == "index" else tname
                if tname and key not in preamble_fired:
                    tool = registry.get(tname)
                    pre = (getattr(tool, "preamble_text", "") or "") if tool else ""
                    if pre:
                        preamble_fired.add(key)
                        await text_sink.preamble(pre)
                args_raw = tc["function"]["arguments"]
                try:
                    args = json.loads(args_raw or "{}")
                except json.JSONDecodeError:
                    result: dict[str, Any] = {
                        "success": False,
                        "error": f"invalid arguments JSON: {args_raw!r}",
                    }
                else:
                    _t_disp = time.perf_counter()
                    result = await registry.dispatch(tname, args, ctx)
                    logger.info(
                        "voxedge tool loop: tool=%s dispatch=%.3fs",
                        tname, time.perf_counter() - _t_disp,
                    )
                content = json.dumps(result, ensure_ascii=False)
                msg_sink.add_tool_result(tc["id"], content)
                tool = registry.get(tname)
                dispatched.append((
                    tname,
                    (getattr(tool, "response_mode", "await") or "await") if tool else "await",
                    (getattr(tool, "completion_text", "") or "") if tool else "",
                    result,
                ))

            # #2: a barge-in landing during tool dispatch → drop the turn
            # before re-requesting the LLM.
            if should_abort():
                return

            # #7: template fast-path.
            if dispatched and _template_fires(dispatched, template_fastpath):
                spoken = _template_completion(dispatched, template_fastpath)
                logger.info(
                    "voxedge tool loop: template fast-path — skipping round2, "
                    "speaking completion_text (%r)", spoken,
                )
                await text_sink.text(spoken)
                await text_sink.flush()
                return
            # loop: re-request the LLM with the tool results appended.

        # Iteration cap hit — flush whatever was spoken and finish.
        logger.warning("voxedge tool loop hit max_tool_rounds=%d", max_rounds)
        if should_abort():
            return
        await text_sink.flush()
    except Exception:
        logger.exception("voxedge LLM tool loop failed")
        try:
            await text_sink.flush()
        except Exception:
            logger.exception("voxedge tool-loop flush after error failed")


def _template_fires(
    dispatched: list[tuple[str, str, str, Any]], mode_policy: str
) -> bool:
    """Whether the template fast-path triggers for this round.

    ``all_join`` (server): EVERY tool opted into ``response_mode="template"``
    with a non-empty completion_text AND succeeded.
    ``any_first`` (client): ANY such tool.
    """
    def ok(entry: tuple[str, str, str, Any]) -> bool:
        _, mode, comp, res = entry
        return (
            mode == "template"
            and bool(comp)
            and not (isinstance(res, dict) and res.get("success") is False)
        )

    if mode_policy == "any_first":
        return any(ok(e) for e in dispatched)
    return all(ok(e) for e in dispatched)


def _template_completion(
    dispatched: list[tuple[str, str, str, Any]], mode_policy: str
) -> str:
    """Spoken completion text for the template fast-path.

    ``all_join`` (server): join all completion_texts with a space.
    ``any_first`` (client): the first eligible tool's completion_text.
    """
    if mode_policy == "any_first":
        for _, mode, comp, res in dispatched:
            if (
                mode == "template"
                and comp
                and not (isinstance(res, dict) and res.get("success") is False)
            ):
                return comp
        return ""
    return " ".join(comp for _, _, comp, _ in dispatched)
