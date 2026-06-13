"""TTS sentence channel + synth consumer loop.

Conversation split step 3 (see seeed-local-voice docs/plans/conversation-split.md):
this is the step-2 ``_TTSChannel`` facade moved out of ``conversation.py``,
now also owning the synth consumer loop (was ``Session._tts_out_task``) as
``run()``. It owns the TTS sentence queue + chunk buffer and is the single place
every TTS write happens — ``_event_loop``, ``_on_asr_final``,
``_llm_turn_with_tools`` and ``_bargein_tts`` go through its methods.

It still holds a back-ref to the owning ``Session`` (``self._sess``) for backend
/ transport / state / config access. Severing that into fully explicit deps is a
later cleanup; for now every method is a 1:1 behaviour port of the code it
replaced (``conversation.py`` line refs in the docstrings / comments are the
source of truth — there is no behaviour change vs step 2).
"""
from __future__ import annotations

import asyncio
import logging
import struct
import threading
from typing import TYPE_CHECKING

from voxedge.engine.protocol import (
    SERVER_TTS_DONE,
    SERVER_TTS_SENTENCE_DONE,
    SERVER_TTS_STARTED,
    _is_pool_saturated,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxedge.engine.conversation import Session

logger = logging.getLogger(__name__)


class _TTSChannel:
    """Owns the TTS sentence queue + chunk buffer and the synth consumer loop.

    ``_event_loop``, ``_on_asr_final``, ``_llm_turn_with_tools`` and
    ``_bargein_tts`` previously poked ``self._tts_q`` / ``self._tts_buffer`` /
    ``state.tts_flush`` directly from four places; they now go through these
    methods, so the TTS write surface is one object. Holds a back-ref to the
    owning Session for backend/state access; every method is a 1:1 behaviour
    port of the code it replaced."""

    def __init__(self, sess: "Session"):
        self._sess = sess
        self.q: asyncio.Queue = asyncio.Queue()
        self.buffer = sess._make_tts_buffer() if sess._tts_be else None

    async def enqueue_text(self, chunk: str) -> None:
        """Add assistant / preamble / client text to the sentence buffer and
        push any completed sentences to the synth queue (was
        ``_enqueue_tts_text`` and the inline ``buffer.add`` loops)."""
        if not chunk or self.buffer is None:
            return
        for sentence in self.buffer.add(chunk):
            await self.q.put(sentence)

    async def flush_and_signal(self) -> None:
        """Flush the buffer remainder to the queue and signal end-of-input so
        the consumer drains then emits its terminal ``tts_done`` — the
        ``for s in buffer.flush(): q.put(s)`` + ``state.tts_flush = True`` pair
        that was repeated at six sites."""
        if self.buffer is not None:
            for sentence in self.buffer.flush():
                await self.q.put(sentence)
        self._sess.state.tts_flush = True

    def interrupt_synth(self) -> None:
        """Barge-in step 1: stop the in-flight synth task + signal its stop
        event. Ordered BEFORE the LLM wind-down."""
        state = self._sess.state
        t = state.current_tts_task
        if t is not None and not t.done():
            t.cancel()
        stop = state.current_tts_stop
        if stop is not None:
            stop.set()

    def drain_and_reset(self) -> None:
        """Barge-in step 3: discard queued sentences + rebuild the buffer so
        nothing stale plays, and clear the flush signal. Ordered AFTER the LLM
        wind-down so the turn can't re-enqueue a sentence between drain
        iterations (#5)."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._sess._tts_be is not None:
            self.buffer = self._sess._make_tts_buffer()
        self._sess.state.tts_flush = False

    async def run(self) -> None:
        """Synth consumer loop (was ``Session._tts_out_task``, app/main.py:
        3207-3432). Dequeues sentences, streams each through the TTS backend
        with per-chunk + per-sentence watchdogs, and emits the close-out
        ``tts_done``."""
        sess = self._sess
        state = sess.state
        multi = sess.engine.multi_utterance
        sr_header_sent = False
        while not state.client_closed:
            # Exit / per-turn done when flush + drained (app/main.py:3217-3233).
            if state.tts_flush and self.q.empty():
                if multi and not state.asr_session_closed:
                    state.tts_flush = False
                    if not state.client_closed:
                        await sess._send_event({
                            "type": SERVER_TTS_DONE,
                            "session_complete": False,
                        })
                    continue
                break
            try:
                sentence = await asyncio.wait_for(self.q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            audio_queue: asyncio.Queue = asyncio.Queue()
            stop_event = threading.Event()
            state.current_tts_stop = stop_event

            # Engine-parity #15: resolve speaker / voice / speed kwargs once
            # per sentence (same dict the legacy _run_synth builds,
            # app/main.py:3473-3478). Empty when nothing injected → default
            # speaker, unchanged behavior.
            stream_kwargs = sess._tts_stream_kwargs()

            def _run_synth(s: str, ev: threading.Event, aq: asyncio.Queue):
                # Mirrors app/main.py:3246-3281 _run_synth (thread body).
                try:
                    for chunk in sess._tts_be.generate_streaming(
                        s,
                        language=sess.engine.tts_language,
                        cancel_token=ev,
                        **stream_kwargs,
                    ):
                        if ev.is_set():
                            break
                        sess._loop.call_soon_threadsafe(aq.put_nowait, chunk)
                except Exception as e:  # noqa: BLE001
                    # M4: a slot-pool saturation is "backend busy", NOT a
                    # synth fault — surface a typed reject marker, not a
                    # generic tts error (app/main.py:3342-3356).
                    sat, max_slots = _is_pool_saturated(e)
                    if sat:
                        sess._loop.call_soon_threadsafe(
                            aq.put_nowait, ("__saturated__", max_slots)
                        )
                    else:
                        logger.exception("voxedge tts synth failed for %r", s[:80])
                        sess._loop.call_soon_threadsafe(
                            aq.put_nowait, ("__error__", str(e))
                        )
                finally:
                    sess._loop.call_soon_threadsafe(aq.put_nowait, None)

            chunk_timeout_s = sess._tts_chunk_timeout_s

            async def drain(s: str, ev: threading.Event, aq: asyncio.Queue):
                nonlocal sr_header_sent
                # Mirrors app/main.py:3283-3361 drain().
                if not sr_header_sent:
                    sr = sess._tts_be.sample_rate
                    await sess._send_audio(struct.pack("<I", sr))
                    sr_header_sent = True
                await sess._send_event({"type": SERVER_TTS_STARTED, "sentence": s})
                # Slot acquire: a TTS synth is the GPU-heavy op; in
                # serialized/exclusive mode it must not overlap an ASR finalize.
                # Held across the whole synth so the synth thread runs alone.
                async with sess._acquire("tts"):
                    sess._loop.run_in_executor(None, _run_synth, s, ev, aq)
                    state.tts_started = True
                    # M3: per-chunk watchdog — a wedged backend that produces no
                    # chunk within the budget aborts the sentence + emits an
                    # error so the client never waits forever (app/main.py:3322-3339).
                    while True:
                        try:
                            item = await asyncio.wait_for(aq.get(), timeout=chunk_timeout_s)
                        except asyncio.TimeoutError:
                            logger.warning(
                                "voxedge tts watchdog: no chunk within %.1fs for "
                                "sentence=%r — aborting synth", chunk_timeout_s, s[:80],
                            )
                            ev.set()
                            await sess._send_error(
                                f"tts: synth produced no chunks within "
                                f"{chunk_timeout_s:.0f}s"
                            )
                            break
                        if item is None:
                            break
                        if isinstance(item, tuple) and item[0] == "__saturated__":
                            # M4: typed pool_saturated; keep the session alive.
                            await sess._emit_pool_saturated(item[1])
                            break
                        if isinstance(item, tuple) and item[0] == "__error__":
                            await sess._send_error(f"tts: {item[1]}")
                            break
                        await sess._send_audio(item)
                await sess._send_event({"type": SERVER_TTS_SENTENCE_DONE, "sentence": s})

            task = asyncio.create_task(drain(sentence, stop_event, audio_queue))
            state.current_tts_task = task
            # M3: outer per-sentence wall-clock deadline. Covers wedges BEFORE
            # the first chunk watchdog can fire (e.g. a backend that hangs in
            # generate_streaming setup before yielding) (app/main.py:3382-3409).
            sentence_timeout_s = sess._tts_sentence_timeout_s
            try:
                await asyncio.wait_for(task, timeout=sentence_timeout_s)
            except asyncio.TimeoutError:
                logger.warning(
                    "voxedge tts: per-sentence deadline %.1fs exceeded for "
                    "sentence=%r — cancelling drain", sentence_timeout_s, sentence[:80],
                )
                stop_event.set()
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                if not state.client_closed:
                    try:
                        await sess._send_error(
                            f"tts: per-sentence deadline "
                            f"{sentence_timeout_s:.0f}s exceeded"
                        )
                    except Exception:
                        logger.exception("voxedge send_error after tts deadline failed")
            except asyncio.CancelledError:
                # Barge-in: stop synth + drain residual chunks (app/main.py:3410-3420).
                stop_event.set()
                try:
                    while True:
                        item = audio_queue.get_nowait()
                        if item is None:
                            break
                except asyncio.QueueEmpty:
                    pass
            finally:
                state.current_tts_task = None
                state.current_tts_stop = None

        # Session-final tts_done (app/main.py:3424-3432).
        if not state.client_closed:
            payload = {"type": SERVER_TTS_DONE}
            if multi:
                payload["session_complete"] = True
            await sess._send_event(payload)
