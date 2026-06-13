"""Server-side multi-turn LLM ↔ tool pump.

Conversation split step 7 (see seeed-local-voice docs/plans/conversation-split.md):
``Session._llm_turn_with_tools`` moves here as ``_LLMTurn.run()`` (plus its
``_emit_preamble`` helper and the ``_ToolCallAcc`` accumulator). ``_on_asr_final``
stays on ``Session`` (the ASR→LLM/TTS bridge) and drives this via the back-ref.

Holds a back-ref to the owning ``Session`` (``self._sess``) for backend / engine
/ transport / TTS access. Every method is a 1:1 behaviour port — no behaviour
change vs step 6.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from voxedge.engine.tool_registry import ToolContext

if TYPE_CHECKING:  # pragma: no cover
    from voxedge.engine.conversation import Session

logger = logging.getLogger(__name__)


@dataclass
class _ToolCallAcc:
    """Accumulator for one tool_call's streamed deltas (per OpenAI index slot).

    Mirrors agent/openvoicestream_agent/tools/runner.py:41-49."""

    id: str = ""
    name: str = ""
    arguments: str = ""


class _LLMTurn:
    """The server-side multi-round LLM↔tool pump."""

    def __init__(self, sess: "Session"):
        self._sess = sess

    async def _emit_preamble(self, preamble_text: str) -> None:
        """Speak a tool preamble (e.g. "好的。") via the TTS buffer.

        TODO(phase-2): if the TTS buffer / app grows a dedicated preamble
        interface (immediate flush, bypass sentence buffering for lowest
        latency), route through it. For now the preamble goes through the same
        sentence buffer as assistant text — correct, just not latency-tuned.
        """
        await self._sess._tts.enqueue_text(preamble_text)

    async def run(self, messages: list[dict[str, Any]]) -> None:
        """Server-side multi-turn LLM ↔ tool pump (spec §2/§4).

        Ported in shape from the agent runner
        (agent/openvoicestream_agent/tools/runner.py:116-443) but self-
        contained: it owns the local ``messages`` list (no agent ``Session``
        dependency) and streams assistant text into the engine TTS buffer
        instead of an SLV client.

        Loop (≤ ``engine.max_tool_rounds`` iterations):
          1. Stream ``llm.stream_events(messages, tools=schema)``.
          2. Text events → TTS sentence buffer.
          3. Accumulate ``tool_call_delta`` events per OpenAI tool-call index.
          4. finish_reason != "tool_calls" (or no tool calls) → done.
          5. Else: append assistant(tool_calls), dispatch each handler
             (local path), append role:"tool" results, re-request the LLM.

        Only invoked when ``tool_registry`` is non-None, so the no-tool path
        in :meth:`Session._on_asr_final` is unaffected (Phase 1 contract).
        """
        sess = self._sess
        registry = sess.engine.tool_registry
        tools_schema = registry.list_openai_tools() or None
        # Prepend the server-loop system prompt (spec §5). Done once on the
        # working message list so every round re-sends the same prefix (keeps
        # the edge-LLM prefix cache stable — append-only history per §8).
        sys_prompt = sess.engine.system_prompt
        if sys_prompt and not (messages and messages[0].get("role") == "system"):
            messages = [{"role": "system", "content": sys_prompt}, *messages]
        llm_params = sess.engine.llm_params
        ctx = ToolContext(
            session_id=getattr(sess.transport, "session_id", None),
            conversation=sess,
            # Phase 2a: remote-dispatch tools push their tool_call frame over
            # the same event channel as other server→client events; the
            # correlated tool_result is routed back via
            # ``registry.resolve_remote`` from the transport receive side.
            remote_send=sess.transport.send_event,
        )
        max_rounds = sess.engine.max_tool_rounds
        try:
            for _round in range(max_rounds):
                text_chunks: list[str] = []
                tool_accs: dict[int, _ToolCallAcc] = {}
                finish_reason: Optional[str] = None
                preamble_fired: set[str] = set()  # tool names already spoken

                # ── latency instrumentation ───────────────────────────────
                # Localise the round2 spike: a "round" here is one LLM request
                # in the tool loop (round 0 = initial command → tool_call,
                # round 1 = post-tool-result reply). _ttft is prefill→first
                # event (the true TTFT for THIS round's context); ctx_msgs is
                # the message-list length sent (proxy for context size — watch
                # it vs the engine's maxSupportedInputLength). This decisively
                # separates "round2 LLM is slow" from "user took N s to speak"
                # — the latter is OUTSIDE this span, in the next /asr turn.
                _t_round_start = time.perf_counter()
                _ttft: Optional[float] = None
                _ctx_msgs = len(messages)

                # #2: barged in between rounds → drop this turn (discard any
                # partial text; _bargein_tts owns the TTS cleanup). Returning
                # abandons the stream generator so the LLM connection closes.
                if sess.state.llm_barged:
                    return

                async for ev in sess._llm_be.stream_events(
                    messages, tools=tools_schema, **llm_params
                ):
                    # #2: barge-in mid-stream — stop consuming deltas and drop
                    # the turn WITHOUT flushing the partial sentence buffer.
                    if sess.state.llm_barged:
                        return
                    if _ttft is None:
                        _ttft = time.perf_counter() - _t_round_start
                    if ev.kind == "text" and ev.text:
                        text_chunks.append(ev.text)
                        await sess._tts.enqueue_text(ev.text)
                    elif ev.kind == "tool_call_delta":
                        idx = ev.tool_call_index if ev.tool_call_index is not None else 0
                        slot = tool_accs.setdefault(idx, _ToolCallAcc())
                        if ev.tool_call_id:
                            slot.id = ev.tool_call_id
                        if ev.name:
                            slot.name = ev.name
                            # Early-fire the per-tool preamble as soon as the
                            # tool name is known (lowest voice latency).
                            if ev.name not in preamble_fired:
                                tool = registry.get(ev.name)
                                pre = (getattr(tool, "preamble_text", "") or "") if tool else ""
                                if pre:
                                    preamble_fired.add(ev.name)
                                    await self._emit_preamble(pre)
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
                    # Don't flush stale text onto the freshly-drained queue —
                    # _bargein_tts owns cleanup for the interrupted turn.
                    if sess.state.llm_barged:
                        return
                    final_text = "".join(text_chunks)
                    if final_text:
                        messages.append({"role": "assistant", "content": final_text})
                    await sess._tts.flush_and_signal()
                    return

                # Commit assistant(tool_calls) to the local message list.
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
                messages.append({
                    "role": "assistant",
                    "content": preamble_content,
                    "tool_calls": tc_payload,
                })

                # Dispatch each tool sequentially. registry.dispatch branches
                # on dispatch_mode internally: local tools run in-process,
                # remote tools (dispatch_mode="remote", Phase 2a) proxy over
                # ctx.remote_send and await a correlated tool_result — both
                # return a JSON-serialisable dict, transparent to this loop.
                dispatched: list[tuple[str, str, str, Any]] = []
                for tc in tc_payload:
                    tname = tc["function"]["name"]
                    # Fallback preamble: fire here if the streamed name delta
                    # never triggered the early path above (e.g. backend sent
                    # the whole tool_call in one finish chunk).
                    if tname and tname not in preamble_fired:
                        tool = registry.get(tname)
                        pre = (getattr(tool, "preamble_text", "") or "") if tool else ""
                        if pre:
                            preamble_fired.add(tname)
                            await self._emit_preamble(pre)
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
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": content,
                    })
                    tool = registry.get(tname)
                    dispatched.append((
                        tname,
                        (getattr(tool, "response_mode", "await") or "await") if tool else "await",
                        (getattr(tool, "completion_text", "") or "") if tool else "",
                        result,
                    ))

                # #2: a barge-in landing during tool dispatch (cancel_pending_
                # remote unblocked the remote await with a "cancelled" result)
                # → drop the turn before re-requesting the LLM.
                if sess.state.llm_barged:
                    return

                # #7: template fast-path. If EVERY tool dispatched this round
                # opted into response_mode="template" with a non-empty
                # completion_text AND succeeded, skip the (slow) LLM round 2
                # and speak the fixed completion_text instead. A single
                # await/parallel tool, an empty completion_text, or a failed
                # result keeps round 2 so the LLM still synthesises a reply —
                # template is a per-tool default, NOT a global round-2 kill.
                if dispatched and all(
                    mode == "template"
                    and comp
                    and not (isinstance(res, dict) and res.get("success") is False)
                    for _, mode, comp, res in dispatched
                ):
                    spoken = " ".join(comp for _, _, comp, _ in dispatched)
                    logger.info(
                        "voxedge tool loop: template fast-path — skipping round2, "
                        "speaking completion_text (%r)", spoken,
                    )
                    await sess._tts.enqueue_text(spoken)
                    await sess._tts.flush_and_signal()
                    return
                # loop: re-request the LLM with the tool results appended.

            # Iteration cap hit — flush whatever was spoken and finish.
            logger.warning("voxedge tool loop hit max_tool_rounds=%d", max_rounds)
            if sess.state.llm_barged:
                return
            await sess._tts.flush_and_signal()
        except Exception:
            logger.exception("voxedge LLM tool loop failed")
            try:
                await sess._tts.flush_and_signal()
            except Exception:
                logger.exception("voxedge tool-loop flush after error failed")
