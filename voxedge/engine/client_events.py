"""Inbound control-event demux (CLIENT_* frames).

Conversation split step 6 (see seeed-local-voice docs/plans/conversation-split.md):
``Session._event_loop`` moves here as ``_ClientEvents.run()``. This is just the
demux of the inbound control-event stream; the handlers it calls stay where they
belong — TTS writes go through ``Session._tts``, barge-in through
``Session._bargein_tts``, the tool-advertise handshake through
``Session._handle_tool_advertise`` (registration that mutates engine
system_prompt / llm_params + warms the prefix), and remote tool-results resolve
on the registry. 1:1 behaviour port — no behaviour change vs step 5.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from voxedge.engine.protocol import (
    CLIENT_ABORT,
    CLIENT_ASR_EOS,
    CLIENT_TEXT,
    CLIENT_TOOL_ADVERTISE,
    CLIENT_TOOL_RESULT,
    CLIENT_TTS_FLUSH,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxedge.engine.conversation import Session

logger = logging.getLogger(__name__)


class _ClientEvents:
    """Demuxes the inbound control-event stream to the right handler."""

    def __init__(self, sess: "Session"):
        self._sess = sess

    async def run(self) -> None:
        """Inbound control events. Port of the text-frame branch of
        dispatcher() (app/main.py:2944-2983)."""
        sess = self._sess
        state = sess.state
        multi = sess.engine.multi_utterance
        try:
            async for payload in sess.transport.recv_event():
                if state.client_closed:
                    break
                typ = payload.get("type")
                if typ == CLIENT_TEXT:
                    await sess._tts.enqueue_text(payload.get("text", ""))
                elif typ == CLIENT_TTS_FLUSH:
                    await sess._tts.flush_and_signal()
                elif typ == CLIENT_ASR_EOS:
                    state.stamp_endpoint("client_eos")
                    if not multi:
                        state.asr_session_closed = True
                elif typ == CLIENT_ABORT:
                    # _bargein_tts now owns the full TTS cleanup (cancel synth +
                    # LLM turn, drain queue, reset buffer) for both barge-in
                    # paths. Here we additionally cancel the ASR turn — the user
                    # explicitly aborted, so (unlike VAD SPEECH_START) no new
                    # utterance is being opened.
                    await sess._bargein_tts()
                    if sess._asr_mgr is not None and state.asr_active:
                        await sess._asr_mgr.cancel("abort")
                        state.deactivate_asr()
                elif typ == CLIENT_TOOL_RESULT:
                    # Remote-tool wire (spec §4 Mode B): route a device client's
                    # tool result back to the awaiting _dispatch_remote future.
                    # ``ok=false`` carries an error string; unknown/late call_id
                    # is safely ignored by resolve_remote.
                    registry = sess.engine.tool_registry
                    if registry is not None:
                        call_id = payload.get("call_id") or payload.get("id")
                        if call_id:
                            ok = payload.get("ok", True)
                            if ok:
                                registry.resolve_remote(
                                    call_id, result=payload.get("result")
                                )
                            else:
                                registry.resolve_remote(
                                    call_id,
                                    error=str(payload.get("error") or "remote tool failed"),
                                )
                elif typ == CLIENT_TOOL_ADVERTISE:
                    sess._handle_tool_advertise(payload)
        except Exception:
            logger.exception("voxedge event_loop error")
            state.client_closed = True
