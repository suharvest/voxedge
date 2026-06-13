"""Inbound audio → VAD segmentation → ASR feed.

Conversation split step 5 (see seeed-local-voice docs/plans/conversation-split.md):
``Session._audio_loop`` moves here as ``_AudioDispatcher.run()``. It only
dispatches — opening/closing ASR turns is delegated to ``_ASRLoop`` (via the
Session back-ref) and barge-in to ``Session._bargein_tts``; this keeps the
generation-ID guards that span the audio/ASR boundary owned by the ASR side.

1:1 behaviour port (``conversation.py`` line refs in the comments) — no
behaviour change vs step 4.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from voxedge.engine.protocol import (
    SERVER_VAD_EVENT,
    VAD_EVENT_SPEECH_END,
    VAD_EVENT_SPEECH_START,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxedge.engine.conversation import Session

logger = logging.getLogger(__name__)


class _AudioDispatcher:
    """Feeds inbound audio frames through VAD and into the active ASR turn."""

    def __init__(self, sess: "Session"):
        self._sess = sess

    async def run(self) -> None:
        """Inbound audio frames → VAD segmentation + ASR feed.

        Port of the binary-frame branch of dispatcher() (app/main.py:2831-2943).
        """
        sess = self._sess
        state = sess.state
        multi = sess.engine.multi_utterance
        try:
            async for data in sess.transport.recv_audio():
                if state.client_closed:
                    break
                if not sess.asr_enabled or state.asr_session_closed:
                    continue
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                speech_ended_now = False

                if sess._vad is not None:
                    event = sess._vad.process(samples)
                    if event == sess._vad.SPEECH_START:
                        # Notify client first, then barge-in (app/main.py:2845-2893).
                        await sess._send_event({
                            "type": SERVER_VAD_EVENT,
                            "event": VAD_EVENT_SPEECH_START,
                        })
                        await sess._bargein_tts()
                        if not await sess._asr.open_turn():
                            continue
                    elif event == sess._vad.SPEECH_END:
                        # Defer endpoint flag until AFTER accepting this chunk
                        # (BUG 3, app/main.py:2894-2900).
                        speech_ended_now = True

                # No-VAD: open lazily on first audio (app/main.py:2901-2921).
                if sess._vad is None and not state.asr_active:
                    if not await sess._asr.open_turn():
                        continue

                if state.asr_active:
                    await sess._asr_mgr.accept_audio(samples)

                # Now safe to latch endpoint (app/main.py:2929-2942).
                if speech_ended_now:
                    state.stamp_endpoint("vad")
                    if not multi:
                        state.asr_session_closed = True
                    await sess._send_event({
                        "type": SERVER_VAD_EVENT,
                        "event": VAD_EVENT_SPEECH_END,
                    })
        except Exception:
            logger.exception("voxedge audio_loop error")
            state.client_closed = True
