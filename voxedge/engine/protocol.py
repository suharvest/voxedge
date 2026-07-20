"""Wire-protocol constants + shared helpers for the conversation engine.

Extracted (conversation split step 3 prerequisite) so the engine submodules —
``conversation``, ``tts_sequencer``, ``asr_loop``, ``audio_dispatcher``,
``client_events``, ``llm_turn`` — share ONE definition of the V2V frame types
and the pool-saturation duck-type without importing each other (which would
create import cycles). Mirrors app/core/v2v.py:33-53. ``conversation`` re-exports
these names for backward compatibility (tests import them from there).
"""
from __future__ import annotations

from typing import Optional

# ── client → server frame types ────────────────────────────────────────
CLIENT_TEXT = "text"
CLIENT_ASR_EOS = "asr_eos"
CLIENT_TTS_FLUSH = "tts_flush"
CLIENT_ABORT = "abort"
# Remote-tool wire (spec §4 Mode B): the client returns a tool result that the
# engine routes back to the awaiting remote-dispatch future via resolve_remote.
CLIENT_TOOL_RESULT = "tool_result"
# Tool advertise handshake (spec §4/§6): right after opening the session the
# device client uploads the OpenAI-style tool schemas it can execute locally
# (plus an optional system_prompt / llm_params override). The engine registers
# them as dispatch_mode="remote" tools so the server-side LLM loop can pick one
# and proxy execution back to the client via SERVER_TOOL_CALL. Additive — a
# legacy client that never enables the server loop never sends this.
CLIENT_TOOL_ADVERTISE = "tool_advertise"
CLIENT_RESPONSE_CREATE = "response.create"

# ── server → client frame types ────────────────────────────────────────
SERVER_ASR_PARTIAL = "asr_partial"
SERVER_ASR_ENDPOINT = "asr_endpoint"
SERVER_ASR_FINAL = "asr_final"
SERVER_TTS_STARTED = "tts_started"
SERVER_TTS_SENTENCE_DONE = "tts_sentence_done"
SERVER_TTS_DONE = "tts_done"
SERVER_VAD_EVENT = "vad_event"
SERVER_ERROR = "error"
# Remote-tool wire (spec §4 Mode B): server asks a remote device client to run
# a tool and report back via CLIENT_TOOL_RESULT.
SERVER_TOOL_CALL = "tool_call"

VAD_EVENT_SPEECH_START = "speech_start"
VAD_EVENT_SPEECH_END = "speech_end"


def _is_pool_saturated(exc: BaseException) -> tuple[bool, Optional[int]]:
    """Duck-type a backend ``PoolSaturatedError`` (M4).

    Mirrors app/main.py:592-610: recognize by ``status == 4429`` (+ class
    name as belt-and-braces) and surface a clean 4429 reject. These are
    deliberately NOT worker-protocol errors — a saturation is "backend
    busy", never a worker fault, so it must NOT trigger a worker restart.

    Returns ``(is_saturated, max_slots_or_None)``.
    """
    if getattr(exc, "status", None) == 4429 or type(exc).__name__ == "PoolSaturatedError":
        ms = getattr(exc, "max_slots", None)
        return True, ms if isinstance(ms, int) else None
    return False, None
