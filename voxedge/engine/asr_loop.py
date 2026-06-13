"""ASR turn lifecycle + the streaming finalize/endpoint loop.

Conversation split step 4 (see seeed-local-voice docs/plans/conversation-split.md):
``Session._open_asr_turn`` and ``Session._asr_out_task`` move here as
``_ASRLoop.open_turn()`` / ``_ASRLoop.run()``. Per the Codex review,
``_on_asr_final`` stays on ``Session`` (it is the ASR→LLM/TTS bridge, not ASR
logic) and is invoked through the back-ref; ``_emit_pool_saturated`` and the
``ASRSessionManager`` also stay on ``Session`` (shared with the TTS path).

Holds a back-ref to the owning ``Session`` (``self._sess``) for manager / state /
transport access. Every method is a 1:1 behaviour port (``conversation.py`` line
refs in the docstrings/comments) — no behaviour change vs step 3.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from voxedge.engine.asr_session_manager import ASRSessionUnavailable
from voxedge.engine.protocol import (
    SERVER_ASR_ENDPOINT,
    SERVER_ASR_FINAL,
    SERVER_ASR_PARTIAL,
    SERVER_ERROR,
    _is_pool_saturated,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxedge.engine.conversation import Session

logger = logging.getLogger(__name__)


class _ASRLoop:
    """ASR turn open + the partial/endpoint/finalize streaming loop."""

    def __init__(self, sess: "Session"):
        self._sess = sess

    async def open_turn(self) -> bool:
        """Start a fresh ASR utterance via the session manager.

        Port of app/main.py:2862-2893 (VAD speech_start) / 2901-2921
        (no-VAD lazy open). On ``ASRSessionUnavailable`` (rebuild ladder
        exhausted — race #1) surface ``asr_unavailable`` and DON'T flag the
        session active, so audio isn't silently dropped with the client
        stuck. Returns True iff a turn was opened.
        """
        sess = self._sess
        state = sess.state
        try:
            new_gen = await sess._asr_mgr.on_speech_start()
        except ASRSessionUnavailable as e:
            state.asr_active = False
            state.clear_endpoint()
            # M4: the manager wraps create_stream failures in
            # ASRSessionUnavailable (raised ``from`` the original). If the
            # root cause was a slot-pool saturation, surface the typed
            # pool_saturated reject instead of a generic asr_unavailable so
            # the client knows to retry (saturation is "busy", not a fault).
            sat, max_slots = _is_pool_saturated(e.__cause__ or e)
            if sat:
                await sess._emit_pool_saturated(max_slots)
                return False
            logger.warning("voxedge: on_speech_start failed: %s", e)
            await sess._send_event({"type": SERVER_ERROR, "error": "asr_unavailable"})
            return False
        except Exception as e:  # noqa: BLE001
            sat, max_slots = _is_pool_saturated(e)
            if sat:
                # M4: backend slot-pool saturated → typed 4429, not a fault.
                await sess._emit_pool_saturated(max_slots)
                state.asr_active = False
                return False
            raise
        # M1: anchor the per-turn wall-clock deadline (app/main.py:2892).
        state.open_asr_generation(new_gen, sess._loop.time())
        return True

    async def run(self) -> None:
        """Partial-poll / endpoint-resolution / finalize loop (was
        ``Session._asr_out_task``, app/main.py:2992-3205)."""
        sess = self._sess
        state = sess.state
        multi = sess.engine.multi_utterance
        last_streamed_final = None
        last_partial: tuple[int, str] = (-1, "")
        asr_turn_timeout_s = sess._asr_turn_timeout_s
        while not state.client_closed:
            # ── M1: wall-clock per-turn deadline (app/main.py:3012-3077) ─
            # Active turn that hasn't finalized within the deadline → force
            # cancel + worker restart (via the manager's bounded cancel
            # ladder) so a wedged backend can't pin the session forever.
            turn_started = state.asr_turn_started_at
            if (
                state.asr_active
                and turn_started is not None
                and (sess._loop.time() - turn_started) > asr_turn_timeout_s
            ):
                elapsed = sess._loop.time() - turn_started
                logger.warning(
                    "voxedge ASR turn exceeded %.1fs wall-clock (elapsed=%.1fs); "
                    "aborting turn + force-cancel ASR session",
                    asr_turn_timeout_s, elapsed,
                )
                # Step 1: cooperative cancel with a tight budget; the manager
                # escalates to restart_worker internally on its own timeout
                # (app/main.py:3030-3054).
                if sess._asr_mgr is not None:
                    try:
                        await asyncio.wait_for(
                            sess._asr_mgr.cancel("turn_timeout"), timeout=2.0
                        )
                    except Exception as _exc:  # noqa: BLE001
                        logger.error(
                            "voxedge ASR cancel timed out / failed (%s)", _exc
                        )
                # Step 2: clear state + emit error so the client unwinds
                # (app/main.py:3055-3068).
                state.deactivate_asr()
                state.clear_endpoint()
                try:
                    await sess._send_error(
                        f"asr: per-turn deadline {asr_turn_timeout_s:.0f}s exceeded"
                    )
                except Exception:
                    logger.exception("voxedge send_error after asr turn timeout failed")
                if multi and not state.asr_session_closed:
                    await asyncio.sleep(0.05)
                    continue
                return

            # ── partial poll (app/main.py:3084-3096) ──────────────────
            if state.asr_active:
                try:
                    # M2: atomic (gen, partial, is_endpoint) snapshot under
                    # the manager's lock — no torn read against a stream that
                    # a barge-in is replacing.
                    partial_gen, partial, is_endpoint = (
                        await sess._asr_mgr.get_partial_for_generation()
                    )
                except Exception:
                    partial, is_endpoint, partial_gen = "", False, 0
                # Gen gate (BUG 4): drop partials from a replaced utterance.
                # Also dedupe identical consecutive partials so the fast poll
                # loop doesn't flood the client (prod tracks last_streamed_final
                # similarly, app/main.py:3001/3187).
                if (
                    partial
                    and partial_gen == state.asr_active_gen
                    and (partial_gen, partial) != last_partial
                ):
                    last_partial = (partial_gen, partial)
                    await sess._send_event({
                        "type": SERVER_ASR_PARTIAL,
                        "text": partial,
                        "is_stable": bool(is_endpoint),
                    })
            else:
                is_endpoint = False

            # ── endpoint resolution (app/main.py:3098-3119) ───────────
            endpoint_reason = state.endpoint_pending
            # Gen-race gate (app/main.py:3107-3114): endpoint stamped against a
            # generation that has since been preempted → drop on the floor.
            if (
                endpoint_reason
                and state.endpoint_pending_gen is not None
                and state.endpoint_pending_gen != state.asr_active_gen
            ):
                state.clear_endpoint()
                endpoint_reason = None

            endpoint_fired = bool(endpoint_reason) or (is_endpoint and state.asr_active)

            if endpoint_fired:
                state.clear_endpoint()
                if endpoint_reason != "client_eos":
                    await sess._send_event({"type": SERVER_ASR_ENDPOINT})

                if state.asr_active:
                    finalize_gen = state.asr_active_gen
                    # M2: finalize_with_status suppresses stale/cancelled
                    # results (accepted=False) so a finalize that raced a
                    # barge-in doesn't emit a spurious final
                    # (app/core/asr_session_manager.py:265-320).
                    # Slot acquire: ASR finalize is the GPU-heavy decode; in
                    # serialized/exclusive mode it must not overlap a TTS synth.
                    async with sess._acquire("asr"):
                        fin_gen, fin_text, accepted, detected_language = (
                            await sess._asr_mgr.finalize_with_status(
                                endpoint_reason or "vad_end"
                            )
                        )
                    if accepted:
                        final_text = fin_text
                    else:
                        final_text, detected_language = "", None
                    # Only clear active if generation still current (BUG 2).
                    if state.asr_active_gen == finalize_gen:
                        state.deactivate_asr()
                else:
                    final_text, detected_language = "", None

                # ── emit asr_final (app/main.py:3164-3197) ────────────
                if multi:
                    is_closing = state.asr_session_closed
                    # Close-out endpoint with no fresh utterance (e.g. a
                    # client_eos / input-end arriving after the turn already
                    # finalized via VAD): there is nothing new to transcribe,
                    # so the session-complete final reaffirms the last streamed
                    # final rather than emitting a spurious empty text. Mirrors
                    # prod's duplicate_of_streamed close-out (app/main.py:3166-
                    # 3176) — the client gets a coherent terminal result.
                    if is_closing and not (final_text or "") and last_streamed_final:
                        emit_text = last_streamed_final
                    else:
                        emit_text = final_text or ""
                    payload = {
                        "type": SERVER_ASR_FINAL,
                        "text": emit_text,
                        "session_complete": is_closing,
                    }
                    if is_closing:
                        payload["duplicate_of_streamed"] = (
                            emit_text == (last_streamed_final or "")
                        )
                    if detected_language:
                        payload["language"] = detected_language
                    await sess._send_event(payload)
                    # Closed-loop hook: feed final text to LLM→TTS (spec §4).
                    # Only drive the LLM on genuinely new ASR text — a close-out
                    # duplicate must not re-trigger another LLM→TTS turn.
                    if final_text and final_text.strip():
                        await sess._on_asr_final(final_text)
                    if is_closing:
                        return
                    last_streamed_final = final_text or ""
                else:
                    payload = {"type": SERVER_ASR_FINAL, "text": final_text or ""}
                    if detected_language:
                        payload["language"] = detected_language
                    await sess._send_event(payload)
                    await sess._on_asr_final(final_text or "")
                    return

            # Exit when closed + nothing left (app/main.py:3202-3203).
            if state.asr_session_closed and not state.asr_active:
                return
            await asyncio.sleep(0.02)
