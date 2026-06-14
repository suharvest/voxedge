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

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Protocol


logger = logging.getLogger(__name__)

# Sentinel: distinguishes "caller did not supply tools_schema" (→ fall back
# to the registry) from "caller explicitly supplied None / [] = no tools".
_UNSET = object()


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
    tools_schema: Any = _UNSET,
    llm_params_for_round: Optional[Callable[[int], dict[str, Any]]] = None,
    on_iteration_limit: Optional[Callable[[], None]] = None,
    on_tool_started: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
    on_tool_completed: Optional[
        Callable[[dict[str, Any], Any, float], Awaitable[None]]
    ] = None,
    first_token_timeout_s: Optional[float] = None,
    idle_timeout_s: Optional[float] = None,
    on_timeout: Optional[Callable[[str, float, str], BaseException]] = None,
    reraise_errors: bool = False,
    record_template_text: bool = False,
    completion_text_cb: Optional[Callable[[str], Awaitable[None]]] = None,
    on_template_misconfig: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
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

    Optional seams (all default to server-equivalent no-ops so the server
    adapter stays byte-equivalent — see turn-driver-unification spec §6b/§6c):

    * ``tools_schema`` — pre-resolved OpenAI tool schema. When OMITTED
      (sentinel ``_UNSET``) → fall back to ``registry.list_openai_tools()``
      (server P0 behaviour). An EXPLICIT value is honoured verbatim, including
      ``None`` / ``[]`` which mean "no tools" — the client passes its
      allowlist-filtered schema (or ``None`` when tools are disabled).
    * ``llm_params_for_round(iter_idx) -> dict`` — per-round extra LLM
      params merged into ``llm_params`` (driver is agnostic to their
      meaning, e.g. prefix-cache injection). ``None`` → nothing merged.
    * ``on_iteration_limit()`` — called (after ``should_abort`` guard) when
      the round cap is hit, in lieu of (server) merely flushing. Caller owns
      any history rollback.
    * ``on_tool_started(tc)`` — fired before args parse / dispatch.
    * ``on_tool_completed(tc, result, dt_ms)`` — fired after the tool result
      is appended.
    * ``first_token_timeout_s`` / ``idle_timeout_s`` / ``on_timeout`` —
      stream watchdog (mirrors the agent runner). ``None`` → no watchdog
      (plain ``async for`` semantics).
    * ``reraise_errors`` — when ``True`` a non-cancel exception is re-raised
      to the caller (client path) instead of being swallowed+flushed
      (server P0 path).
    * ``record_template_text`` — when ``True`` the template fast-path also
      appends the spoken completion_text as an assistant message via
      ``msg_sink`` (client need for history coherence). Server P0 passes
      ``False`` → message list unchanged (byte-equivalent).
    * ``completion_text_cb`` — where the template fast-path's spoken
      completion_text goes. ``None`` (server) → spoken via ``text_sink.text``
      (P0 behaviour: same TTS channel as assistant text). Client passes a
      dedicated callback (its ``on_tool_completion_text``) so the emission is
      distinguishable from streamed assistant tokens.
    * ``on_template_misconfig(tool_name)`` — fired (``any_first`` only) when a
      template tool SUCCEEDED but declared an empty completion_text, so the
      fast-path is abandoned and a normal LLM round runs. Lets the client log
      the operator warning from its own logger. Server (``all_join``) unused.

    Returns the final assistant text (client path uses it; the server
    adapter ignores it). May be ``None`` when nothing terminal was spoken
    (iteration cap / error swallow path).
    """
    # ``_UNSET`` → caller (server) left it to the registry (P0 behaviour:
    # ``list_openai_tools() or None``). An explicit value (incl. ``None`` or
    # ``[]``) is honoured verbatim — the client passes its allowlist-filtered
    # schema, where ``None`` means "no tools" (NOT "all tools").
    if tools_schema is _UNSET:
        tools_schema = registry.list_openai_tools() or None
    try:
        for _round in range(max_rounds):
            text_chunks: list[str] = []
            tool_accs: dict[int, _ToolCallAcc] = {}
            finish_reason: Optional[str] = None
            # preamble dedup key set: tool names ("name") or call indices ("index")
            preamble_fired: set = set()

            messages = msg_sink.working_messages()

            # Per-round LLM params: base params + any caller-injected extras
            # (e.g. client prefix-cache on iter >0). Server passes no
            # injector → ``round_params`` is just ``llm_params`` (P0).
            round_params = dict(llm_params)
            if llm_params_for_round is not None:
                extra = llm_params_for_round(_round)
                if extra:
                    round_params.update(extra)

            # ── latency instrumentation ───────────────────────────────
            _t_round_start = time.perf_counter()
            _ttft: Optional[float] = None
            _ctx_msgs = len(messages)

            # #2: barged in between rounds → drop this turn (discard any
            # partial text). Returning abandons the stream generator so the
            # LLM connection closes.
            if should_abort():
                return None

            # Stream events. With a watchdog configured we drive the
            # iterator explicitly so we can apply first-token / idle
            # timeouts (mirrors agent runner.py:154-206); otherwise plain
            # ``async for`` (byte-equivalent to the server P0 path).
            stream = llm.stream_events(
                messages, tools=tools_schema, **round_params
            )
            _watchdog = (
                first_token_timeout_s is not None or idle_timeout_s is not None
            )
            received_payload = False
            it = stream.__aiter__()
            while True:
                if _watchdog:
                    use_first = (
                        first_token_timeout_s is not None
                        and not received_payload
                    )
                    use_idle = (
                        idle_timeout_s is not None and received_payload
                    )
                    try:
                        if use_first:
                            ev = await asyncio.wait_for(
                                it.__anext__(), timeout=first_token_timeout_s
                            )
                        elif use_idle:
                            ev = await asyncio.wait_for(
                                it.__anext__(), timeout=idle_timeout_s
                            )
                        else:
                            ev = await it.__anext__()
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        kind = (
                            "first_token" if not received_payload
                            else "stream_idle"
                        )
                        t_used = (
                            float(first_token_timeout_s)
                            if not received_payload
                            else float(idle_timeout_s)
                        )
                        aclose = getattr(stream, "aclose", None)
                        if callable(aclose):
                            try:
                                await aclose()
                            except Exception:  # pragma: no cover
                                logger.debug(
                                    "stream aclose during timeout failed",
                                    exc_info=True,
                                )
                        if on_timeout is not None:
                            raise on_timeout(
                                kind, t_used, "".join(text_chunks)
                            )
                        raise asyncio.TimeoutError(
                            f"LLM {kind} timeout after {t_used:.1f}s"
                        )
                else:
                    try:
                        ev = await it.__anext__()
                    except StopAsyncIteration:
                        break

                # #2: barge-in mid-stream — stop consuming deltas and drop
                # the turn WITHOUT flushing the partial sentence buffer.
                if should_abort():
                    return None
                if _ttft is None:
                    _ttft = time.perf_counter() - _t_round_start
                if ev.kind == "text" and ev.text:
                    received_payload = True
                    text_chunks.append(ev.text)
                    await text_sink.text(ev.text)
                elif ev.kind == "tool_call_delta":
                    received_payload = True
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
                    return None
                final_text = "".join(text_chunks)
                if final_text:
                    msg_sink.add_assistant_text(final_text)
                await text_sink.flush()
                return final_text

            # Commit assistant(tool_calls) to the message list.
            # Preserve the ORIGINAL tool_call indices alongside the payload
            # so the "index" dedup fallback uses the real OpenAI index, not
            # an enumerate() position (sparse indices are not equivalent —
            # spec §6c, mirrors agent runner.py:287-293).
            preamble_content = "".join(text_chunks) or None
            sorted_idxs = sorted(tool_accs.keys())
            tc_payload: list[dict[str, Any]] = [
                {
                    "id": tool_accs[idx].id or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_accs[idx].name,
                        "arguments": tool_accs[idx].arguments or "{}",
                    },
                }
                for idx in sorted_idxs
            ]
            msg_sink.add_assistant_tool_calls(preamble_content, tc_payload)

            # Dispatch each tool sequentially. registry.dispatch branches on
            # dispatch_mode internally: local tools run in-process, remote
            # tools proxy over ctx.remote_send and await a correlated
            # tool_result — both return a JSON-serialisable dict.
            dispatched: list[tuple[str, str, str, Any]] = []
            for tc_pos, tc in enumerate(tc_payload):
                tname = tc["function"]["name"]
                # Original tool_call index for "index"-mode dedup (sparse-
                # safe); position aligns with sorted_idxs.
                tc_idx = (
                    sorted_idxs[tc_pos] if tc_pos < len(sorted_idxs) else tc_pos
                )
                if on_tool_started is not None:
                    await on_tool_started(tc)
                # Fallback preamble: fire here if the streamed name delta never
                # triggered the early path above (e.g. backend sent the whole
                # tool_call in one finish chunk).
                key = tc_idx if preamble_dedup == "index" else tname
                if tname and key not in preamble_fired:
                    tool = registry.get(tname)
                    pre = (getattr(tool, "preamble_text", "") or "") if tool else ""
                    if pre:
                        preamble_fired.add(key)
                        await text_sink.preamble(pre)
                _t_disp = time.perf_counter()
                args_raw = tc["function"]["arguments"]
                try:
                    args = json.loads(args_raw or "{}")
                except json.JSONDecodeError:
                    result: dict[str, Any] = {
                        "success": False,
                        "error": f"invalid arguments JSON: {args_raw!r}",
                    }
                else:
                    result = await registry.dispatch(tname, args, ctx)
                    logger.info(
                        "voxedge tool loop: tool=%s dispatch=%.3fs",
                        tname, time.perf_counter() - _t_disp,
                    )
                content = json.dumps(result, ensure_ascii=False)
                msg_sink.add_tool_result(tc["id"], content)
                if on_tool_completed is not None:
                    dt_ms = (time.perf_counter() - _t_disp) * 1000.0
                    await on_tool_completed(tc, result, dt_ms)
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
                return None

            # #7: template fast-path.
            if template_fastpath == "any_first":
                # Client semantics (byte-equivalent to agent
                # runner.py:382-442): walk template tools in dispatch order.
                # First failed template tool → abandon fast-path. First
                # template tool with empty completion_text → warn (via
                # ``on_template_misconfig``) + abandon. Otherwise emit each
                # eligible tool's completion_text and skip round 2; the synth
                # assistant text is the FIRST eligible completion.
                template_handled = False
                for _tname, rmode, ctext, result in dispatched:
                    if rmode != "template":
                        continue
                    ok = not (
                        isinstance(result, dict)
                        and result.get("success") is False
                    )
                    if not ok:
                        template_handled = False
                        break
                    if not ctext:
                        if on_template_misconfig is not None:
                            on_template_misconfig(_tname)
                        template_handled = False
                        break
                    template_handled = True
                    if completion_text_cb is not None:
                        await completion_text_cb(ctext)
                    else:
                        await text_sink.text(ctext)
                if template_handled:
                    synth = next(
                        (
                            ct for _n, rm, ct, _r in dispatched
                            if rm == "template" and ct
                        ),
                        "",
                    )
                    if record_template_text and synth:
                        msg_sink.add_assistant_text(synth)
                    await text_sink.flush()
                    return synth
            elif dispatched and _template_fires(dispatched, template_fastpath):
                spoken = _template_completion(dispatched, template_fastpath)
                logger.info(
                    "voxedge tool loop: template fast-path — skipping round2, "
                    "speaking completion_text (%r)", spoken,
                )
                if completion_text_cb is not None:
                    if spoken:
                        await completion_text_cb(spoken)
                else:
                    await text_sink.text(spoken)
                # Record the synthesised assistant text so history stays
                # coherent for the next turn (client need — mirrors agent
                # runner.py:425-442). Server P0 path passed
                # ``record_template_text=False`` → no message-list change →
                # byte-equivalent.
                if record_template_text and spoken:
                    msg_sink.add_assistant_text(spoken)
                await text_sink.flush()
                return spoken
            # loop: re-request the LLM with the tool results appended.

        # Iteration cap hit — flush whatever was spoken and finish.
        logger.warning("voxedge tool loop hit max_tool_rounds=%d", max_rounds)
        if should_abort():
            return None
        if on_iteration_limit is not None:
            on_iteration_limit()
        await text_sink.flush()
        return None
    except Exception:
        if reraise_errors:
            raise
        logger.exception("voxedge LLM tool loop failed")
        try:
            await text_sink.flush()
        except Exception:
            logger.exception("voxedge tool-loop flush after error failed")
        return None


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
