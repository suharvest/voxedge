"""Server-side multi-turn LLM ↔ tool pump (thin adapter over ``turn_driver``).

Conversation split step 7 (see seeed-local-voice docs/plans/conversation-split.md):
``Session._llm_turn_with_tools`` moved here as ``_LLMTurn.run()``. The pump
*algorithm* now lives in ``voxedge.engine.turn_driver.run_turn`` (provider-
agnostic, no I/O — see docs/plans/turn-driver-unification.md, P0). ``_LLMTurn``
is the server adapter: it wires the driver's seams to the owning ``Session``
(TTS buffer, cooperative ``llm_barged`` flag, local message list). The driver
has a single behaviour (name dedup + all_join template fast-path); P2a removed
the strategy params.

Behaviour is byte-equivalent to the pre-extraction in-line pump. ``_on_asr_final``
stays on ``Session`` (the ASR→LLM/TTS bridge) and drives this via the back-ref.
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from voxedge.engine.tool_registry import ToolContext
from voxedge.engine.turn_driver import _ToolCallAcc, run_turn  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover
    from voxedge.engine.conversation import Session


class _TTSAdapter:
    """``TextSink`` over the engine TTS buffer (``Session._tts``).

    ``text`` and ``preamble`` both enqueue into the same sentence buffer
    (preamble has no dedicated low-latency path yet — phase-2 TODO), exactly
    as the pre-extraction pump did. ``flush`` calls ``flush_and_signal``.
    """

    def __init__(self, sess: "Session"):
        self._sess = sess

    async def text(self, s: str) -> None:
        await self._sess._tts.enqueue_text(s)

    async def preamble(self, s: str) -> None:
        await self._sess._tts.enqueue_text(s)

    async def flush(self) -> None:
        await self._sess._tts.flush_and_signal()


class _LocalMessageSink:
    """``MessageSink`` over a local append-only ``messages`` list.

    The server owns conversation state for the turn in this list; the system
    prompt is prepended by the adapter (caller responsibility) BEFORE the
    driver runs, so this sink never adds a system message.
    """

    def __init__(self, messages: list[dict[str, Any]]):
        self._messages = messages

    def working_messages(self) -> list[dict[str, Any]]:
        return self._messages

    def add_assistant_tool_calls(
        self, content: Optional[str], tool_calls: list[dict[str, Any]]
    ) -> None:
        self._messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

    def add_assistant_text(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })


class _LLMTurn:
    """The server-side multi-round LLM↔tool pump (adapter over ``run_turn``)."""

    def __init__(self, sess: "Session"):
        self._sess = sess

    async def _emit_preamble(self, preamble_text: str) -> None:
        """Speak a tool preamble (e.g. "好的。") via the TTS buffer.

        Retained for backwards compatibility / tests. The driver speaks
        preambles through the ``_TTSAdapter`` seam, which routes here-equivalent
        ``enqueue_text``.
        """
        await self._sess._tts.enqueue_text(preamble_text)

    async def run(self, messages: list[dict[str, Any]]) -> None:
        """Server-side multi-turn LLM ↔ tool pump (spec §2/§4).

        Wires the ``run_turn`` driver to this ``Session``:

          * system-prompt prefix is prepended HERE (caller responsibility) so
            the driver only appends assistant/tool messages;
          * ``TextSink`` → ``Session._tts`` (engine TTS buffer);
          * ``MessageSink`` → local append-only ``messages`` list;
          * ``should_abort`` → ``sess.state.llm_barged`` (cooperative poll).

        The driver's single behaviour is name-keyed preamble dedup + the
        all_join template fast-path (P2a).

        Only invoked when ``tool_registry`` is non-None, so the no-tool path in
        :meth:`Session._on_asr_final` is unaffected (Phase 1 contract).
        """
        sess = self._sess
        registry = sess.engine.tool_registry
        # Prepend the server-loop system prompt (spec §5). Done once on the
        # working message list so every round re-sends the same prefix (keeps
        # the edge-LLM prefix cache stable — append-only history per §8).
        # This is the adapter / caller's responsibility (spec §4 / D3): the
        # driver never inserts a system message.
        sys_prompt = sess.engine.system_prompt
        if sys_prompt and not (messages and messages[0].get("role") == "system"):
            messages = [{"role": "system", "content": sys_prompt}, *messages]
        ctx = ToolContext(
            session_id=getattr(sess.transport, "session_id", None),
            conversation=sess,
            # Phase 2a: remote-dispatch tools push their tool_call frame over
            # the same event channel as other server→client events; the
            # correlated tool_result is routed back via
            # ``registry.resolve_remote`` from the transport receive side.
            remote_send=sess.transport.send_event,
        )
        await run_turn(
            llm=sess._llm_be,
            registry=registry,
            msg_sink=_LocalMessageSink(messages),
            text_sink=_TTSAdapter(sess),
            should_abort=lambda: sess.state.llm_barged,
            ctx=ctx,
            llm_params=sess.engine.llm_params,
            max_rounds=sess.engine.max_tool_rounds,
            # Server P0: all client seams stay at their server-equivalent
            # defaults (None / False) so this adapter is byte-equivalent.
            # The driver's return value (final text) is intentionally
            # ignored here — the server speaks via the TTS sink.
            reraise_errors=False,
        )
