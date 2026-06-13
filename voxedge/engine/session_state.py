"""Per-connection conversation state.

Step 1 of the conversation.py split (see seeed-local-voice
docs/plans/conversation-split.md). This replaces the ad-hoc ``state`` dict that
``Session`` carried (mirroring app/main.py:2751-2791) with a typed dataclass
PLUS grouped transition methods for the atomic, multi-field protocol operations.

Why methods and not just typed fields: several updates are *atomic* protocol
transitions spanning more than one field (opening an ASR generation, stamping /
clearing an endpoint against the current generation, deactivating the active
turn, closing the input stream). The Codex review of the split plan flagged that
leaving these as scattered field writes turns the refactor into "typed shared
globals" and lets later extractions re-implement them inconsistently. Each
method below is a faithful 1:1 port of the inline updates it replaces — the
``conversation.py`` line references are the source of truth, and there is **no
behaviour change** vs the dict (same defaults, same field writes, same order).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SessionState:
    """State for one conversation (one Transport).

    Field defaults mirror the original ``Session.state`` dict exactly
    (conversation.py:268-294).
    """

    client_closed: bool = False
    asr_active: bool = False
    asr_active_gen: int = 0
    asr_session_closed: bool = False
    endpoint_pending: Optional[str] = None
    endpoint_pending_gen: Optional[int] = None
    # Handle to the in-flight TTS synth task + its cooperative stop event.
    current_tts_task: Any = None
    current_tts_stop: Any = None
    # Handle to the in-flight server-loop LLM/tool turn so a barge-in can stop
    # it cooperatively (see ``llm_barged``).
    current_llm_task: Any = None
    # Cooperative barge-in flag — task.cancel() alone is unsafe because
    # _dispatch_remote swallows CancelledError; the turn polls this at each
    # checkpoint and self-terminates.
    llm_barged: bool = False
    tts_flush: bool = False
    tts_started: bool = False
    # M1: per-ASR-turn wall-clock deadline anchor (loop time, app/main.py:2790).
    asr_turn_started_at: Optional[float] = None

    # ── grouped transitions (atomic multi-field protocol ops) ─────────────

    def open_asr_generation(self, new_gen: int, now: float) -> None:
        """A fresh ASR utterance was opened (conversation.py:691-696).

        Clears any pending endpoint, marks the turn active under ``new_gen``,
        and anchors the per-turn wall-clock deadline at ``now`` (caller passes
        ``self._loop.time()`` so this stays loop-agnostic).
        """
        self.endpoint_pending = None
        self.endpoint_pending_gen = None
        self.asr_active = True
        self.asr_active_gen = new_gen
        self.asr_turn_started_at = now

    def stamp_endpoint(self, reason: str) -> None:
        """Stamp a pending endpoint against the current generation.

        Ports conversation.py:410-411 (VAD speech_end) and 441-442
        (client_eos). The generation tag lets ``_asr_out_task`` drop an
        endpoint that a later utterance has since preempted.
        """
        self.endpoint_pending = reason
        self.endpoint_pending_gen = self.asr_active_gen

    def clear_endpoint(self) -> None:
        """Drop any pending endpoint marker (conversation.py:669-670/749-750/
        800-801/807-808)."""
        self.endpoint_pending = None
        self.endpoint_pending_gen = None

    def deactivate_asr(self) -> None:
        """End the active ASR turn (conversation.py:747-748/832-833/454-455).

        Clears the active flag and the deadline anchor; does NOT touch the
        endpoint markers (callers that also need that call ``clear_endpoint``).
        """
        self.asr_active = False
        self.asr_turn_started_at = None

    def close_input(self) -> None:
        """Input stream closed (conversation.py:1414-1415).

        Sets the terminal-final flag and flushes TTS together — this pair is a
        single transition that drives both ASR terminal-final behaviour
        (conversation.py:839) and the final ``tts_done`` emission
        (conversation.py:1206); keep them coupled.
        """
        self.asr_session_closed = True
        self.tts_flush = True
