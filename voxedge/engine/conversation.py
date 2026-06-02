"""ConversationEngine + Session — the V2V orchestration core.

Faithful port of the orchestration buried in app/main.py's /v2v/stream
handler (dispatcher / asr_out_task / tts_out_task, app/main.py:2814-3432),
lifted out of FastAPI into an importable engine that drives any
:class:`~voxedge.transport.base.Transport`.

PORTING MAP (source ranges are app/main.py snapshots, 2026-05-29):
  * dispatcher()    ← app/main.py:2814-2990   (audio/control frame state machine)
  * asr_out_task()  ← app/main.py:2992-3205   (partial poll, gen gate, finalize)
  * tts_out_task()  ← app/main.py:3207-3432   (sentence queue, barge-in, done)
  * orchestrate     ← app/main.py:3434-3457   (spawn work tasks, cancel dispatcher)

DELIBERATE SIMPLIFICATIONS (kept faithful in shape, infra dropped):
  * No BackendCoordinator ``coord.acquire(...)`` / SessionLimiter / slot-pool
    (app/core, out of Phase-1a scope). The lock-around-backend calls are
    replaced by direct calls; concurrency ceilings are a later phase.
    WS-level admission (SessionLimiter) stays in the transport/app layer.
  * ASRSessionManager IS used now (M2, 2026-05-30): copied verbatim into
    ``voxedge/engine/asr_session_manager.py`` (dedup after Phase 1b). This
    gives the engine atomic per-generation partial snapshots
    (``get_partial_for_generation``), bounded cancel + worker-restart ladder
    (``cancel`` / ERROR_REBUILD), and ``finalize_with_status`` accepted/stale
    suppression — replacing the earlier hand-rolled ``_AsrTurn`` helper.
  * Wall-clock turn (M1) + TTS chunk/sentence (M3) watchdogs are now wired,
    but their thresholds are CONSTRUCTOR-INJECTED (``timeouts`` dict), never
    read from env (spec §2). Defaults mirror prod's env defaults.
  * Blocking backend calls run through the ASRSessionManager's executor hop
    (ASR) or ``asyncio.to_thread`` / ``run_in_executor`` (TTS) so the event
    loop is never blocked — same intent as prod (app/main.py:3050, 3310).

LLM closed loop (spec §4): when an ``llm`` backend is provided the asr_final
text is fed to the LLM and its text deltas drive the TTS sentence buffer
(equivalent to prod's CLIENT_TEXT path, app/main.py:2953-2960). With no LLM
the engine is a pure ASR↔TTS pass-through plus a direct CLIENT_TEXT→TTS path.
When a ``tool_registry`` is supplied the asr_final text instead drives the
server-side multi-turn LLM↔tool pump (``Session._llm_turn_with_tools``, spec
§2/§4); with ``tool_registry=None`` (the default) that path is never taken and
the LLM call is byte-identical to before the tool migration (Phase 1 contract).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from voxedge.backends.base import ASRBackend, LLMBackend, TTSBackend, VADBackend
from voxedge.engine.tool_registry import ToolContext
from voxedge.engine.asr_session_manager import (
    ASRSessionManager,
    ASRSessionUnavailable,
)
from voxedge.engine.coordinator import BackendCoordinator
from voxedge.engine.tts_buffer import LowLatencyTTSBuffer
from voxedge.transport.base import Transport


@contextlib.asynccontextmanager
async def _passthrough():
    """No-op async context manager — used when no coordinator is wired so the
    engine keeps its current direct-call behavior (backward compatible)."""
    yield

logger = logging.getLogger(__name__)


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


# ── protocol constants (mirror app/core/v2v.py:33-53) ──────────────────
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


def _advertised_remote_noop(*_args, **_kwargs):  # pragma: no cover - never called
    """Placeholder local handler for a client-advertised remote tool.

    The registry's ``dispatch_mode="remote"`` path proxies execution to the
    device client over the wire and never invokes ``fn``; this exists only so
    ``ToolRegistry.register`` (which requires a callable) has something to hold.
    """
    raise RuntimeError("advertised remote tool must dispatch over the wire, not locally")


@dataclass
class _ToolCallAcc:
    """Accumulator for one tool_call's streamed deltas (per OpenAI index slot).

    Mirrors agent/openvoicestream_agent/tools/runner.py:41-49."""

    id: str = ""
    name: str = ""
    arguments: str = ""


# ── lightweight sentence buffer (regex fallback, mirrors v2v.py path) ──
import re as _re

_SENTENCE_END_RE = _re.compile(r"[。！？；\n]+|[!?.](?=\s|$)")


class _SentenceBuffer:
    """Minimal pure-Python sentence buffer.

    Equivalent in role to app/core/v2v.py SentenceBuffer's regex-fallback
    path (v2v.py:202-224). voxedge core stays dep-free; the pysbd upgrade is
    the optional ``voxedge[text]`` extra.
    """

    def __init__(self, min_chars: int = 2, max_buffer: int = 200):
        self._buf = ""
        self._min = min_chars
        self._max = max_buffer

    def add(self, chunk: str):
        if not chunk:
            return
        self._buf += chunk
        while True:
            s = self._extract()
            if s is None:
                return
            yield s

    def flush(self):
        leftover = self._buf.strip()
        self._buf = ""
        if leftover:
            yield leftover

    def _extract(self) -> Optional[str]:
        pos = 0
        while True:
            m = _SENTENCE_END_RE.search(self._buf, pos)
            if m is None:
                if len(self._buf) >= self._max:
                    out = self._buf.strip()
                    self._buf = ""
                    return out or None
                return None
            end = m.end()
            prefix = self._buf[:end]
            if len(prefix.strip()) >= self._min:
                self._buf = self._buf[end:]
                return prefix.strip()
            pos = end


class Session:
    """One conversation over one Transport. Owns the per-conn state machine."""

    def __init__(self, engine: "ConversationEngine", transport: Transport):
        self.engine = engine
        self.transport = transport

        # Optional slot coordinator (concurrency abstraction migrated from
        # app/core, spec §3.1). When None the engine runs direct passthrough
        # (current behavior, backward compatible). When present, ASR/TTS
        # backend calls are wrapped in ``coord.acquire(...)`` so serialized /
        # exclusive modes truly mutually-exclude and concurrent mode overlaps.
        self._coord: Optional[BackendCoordinator] = engine.coordinator

        be = engine.backends
        self._asr_be: Optional[ASRBackend] = be.get("asr")
        self._tts_be: Optional[TTSBackend] = be.get("tts")
        self._vad_be: Optional[VADBackend] = be.get("vad")
        self._llm_be: Optional[LLMBackend] = be.get("llm")

        self.asr_enabled = self._asr_be is not None
        self._vad = (
            self._vad_be.create_session(silence_ms=engine.silence_ms)
            if self._vad_be is not None
            else None
        )
        # M2: ASRSessionManager replaces the hand-rolled _AsrTurn — gives
        # atomic per-generation partial snapshots + bounded cancel/restart
        # ladder + finalize accepted/stale suppression (mirrors prod
        # app/main.py asr_manager).
        self._asr_mgr: Optional[ASRSessionManager] = (
            ASRSessionManager(
                self._asr_be,
                language=engine.asr_language,
                sample_rate=self._asr_be.sample_rate,
            )
            if self._asr_be
            else None
        )
        # Engine-parity #15: select the TTS chunk buffer. Default keeps the
        # back-compat ``_SentenceBuffer`` (existing tests unchanged); when the
        # engine is constructed with ``low_latency_tts=True`` use the ported
        # ``LowLatencyTTSBuffer`` (CJK clause / bounded-span early emit) — the
        # same buffer the legacy /v2v handler selects (app/main.py:2961-2970).
        self._tts_buffer = (
            self._make_tts_buffer() if self._tts_be else None
        )

        # Engine-parity #15: speaker / voice / speed kwargs forwarded to the
        # TTS backend's ``generate_streaming``. Constructor-injected (no env
        # / no app.core.tts_speakers import in voxedge) so the resolved
        # speaker dict is passed in by the caller — mirrors the legacy
        # ``tts_speaker_kwargs`` / ``tts_voice`` / ``tts_speed`` path
        # (app/main.py:3473-3478). Empty / None → backend uses its default
        # speaker (behavior unchanged from before this change).
        self._tts_speaker_kwargs: dict = dict(engine.tts_speaker_kwargs or {})
        self._tts_voice = engine.tts_voice
        self._tts_speed = engine.tts_speed

        self._tts_q: asyncio.Queue = asyncio.Queue()
        self._loop = asyncio.get_event_loop()

        # M1/M3: wall-clock watchdog thresholds — constructor-injected (spec
        # §2: no env reads in the engine). Defaults mirror prod env defaults
        # (OVS_ASR_TURN_TIMEOUT_S=45, OVS_TTS_CHUNK_TIMEOUT_S=10,
        # OVS_TTS_SENTENCE_TIMEOUT_S=15).
        self._asr_turn_timeout_s = engine.asr_turn_timeout_s
        self._tts_chunk_timeout_s = engine.tts_chunk_timeout_s
        self._tts_sentence_timeout_s = engine.tts_sentence_timeout_s

        # per-conn state (mirrors app/main.py:2751-2791 state dict)
        self.state = {
            "client_closed": False,
            "asr_active": False,
            "asr_active_gen": 0,
            "asr_session_closed": False,
            "endpoint_pending": None,
            "endpoint_pending_gen": None,
            "current_tts_task": None,
            "current_tts_stop": None,
            # #2: handle to the in-flight server-loop LLM/tool turn so a
            # barge-in can stop it (cooperatively — see ``llm_barged``) and
            # _asr_out_task can move on to the next utterance instead of
            # blocking on a turn whose (slow) round2 the user just interrupted.
            "current_llm_task": None,
            # #2: cooperative barge-in flag. task.cancel() alone is unsafe
            # here — _dispatch_remote swallows CancelledError (returns a
            # "cancelled" result) so a cancel landing mid remote-dispatch would
            # NOT stop the turn. The turn polls this flag at each checkpoint
            # (per stream event / per round / after dispatch) and self-
            # terminates; _bargein_tts also cancel_pending_remote() to unblock
            # an in-flight remote await fast.
            "llm_barged": False,
            "tts_flush": False,
            "tts_started": False,
            # M1: per-ASR-turn wall-clock deadline anchor (app/main.py:2790).
            "asr_turn_started_at": None,
        }
        self._work_tasks: list[asyncio.Task] = []
        # Server-loop LLM prefix warm-up state. The agent skips its local LLM
        # warmup in server-loop mode (the LLM runs here), so voxedge primes
        # edge-llm's prefix cache + CUDA graph once the advertised system_prompt
        # + tools are known (see _maybe_warm_llm_prefix). Track the last warmed
        # prefix signature so reconnect re-advertises don't re-warm needlessly.
        self._warmed_prefix_sig: "tuple | None" = None
        self._warm_task: "asyncio.Task | None" = None

    def _make_tts_buffer(self):
        """Build the TTS chunk buffer for this session (engine-parity #15).

        Mirrors the legacy buffer selection (app/main.py:2961-2970): when
        ``low_latency_tts`` is enabled use the clause-level
        ``LowLatencyTTSBuffer`` (lower TTFA), otherwise the conservative
        ``_SentenceBuffer``. Defaults to ``_SentenceBuffer`` so existing
        behavior / tests are unchanged.
        """
        if self.engine.low_latency_tts:
            return LowLatencyTTSBuffer(language=self.engine.tts_language)
        return _SentenceBuffer()

    def _tts_stream_kwargs(self) -> dict:
        """Build the kwargs forwarded to ``generate_streaming`` for speaker /
        voice / speed (engine-parity #15).

        Faithful port of the legacy ``_run_synth`` kwarg assembly
        (app/main.py:3473-3478): a resolved ``tts_speaker_kwargs`` dict wins;
        else a deprecated ``voice`` fallback; ``speed`` is always added when
        set. With nothing injected this returns ``{}`` → the backend uses its
        default speaker (behavior unchanged from before this change).
        """
        kwargs: dict = {}
        if self._tts_speaker_kwargs:
            kwargs.update(self._tts_speaker_kwargs)
        elif self._tts_voice is not None:
            kwargs["voice"] = self._tts_voice  # deprecated
        if self._tts_speed is not None:
            kwargs["speed"] = self._tts_speed
        return kwargs

    def _acquire(self, slot: str):
        """Acquire the slot via the coordinator, or a no-op when none wired.

        In ``serialized`` / ``exclusive`` mode the coordinator's shared lock
        makes ASR and TTS backend calls mutually exclusive; in ``concurrent``
        mode (or with no coordinator) this is a passthrough and they overlap.
        """
        if self._coord is None:
            return _passthrough()
        return self._coord.acquire(slot)  # type: ignore[arg-type]

    # ── transport send helpers (app/main.py:2795-2810) ─────────────────

    async def _send_event(self, payload: dict) -> None:
        try:
            await self.transport.send_event(payload)
        except Exception:
            self.state["client_closed"] = True

    async def _send_audio(self, data: bytes) -> None:
        try:
            await self.transport.send_audio(data)
        except Exception:
            self.state["client_closed"] = True

    async def _send_error(self, msg: str) -> None:
        await self._send_event({"type": SERVER_ERROR, "error": msg})

    # ══════════════════════════════════════════════════════════════════
    # dispatcher  ← app/main.py:2814-2990
    # ══════════════════════════════════════════════════════════════════

    async def _audio_loop(self) -> None:
        """Inbound audio frames → VAD segmentation + ASR feed.

        Port of the binary-frame branch of dispatcher() (app/main.py:2831-2943).
        """
        state = self.state
        multi = self.engine.multi_utterance
        try:
            async for data in self.transport.recv_audio():
                if state["client_closed"]:
                    break
                if not self.asr_enabled or state["asr_session_closed"]:
                    continue
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                speech_ended_now = False

                if self._vad is not None:
                    event = self._vad.process(samples)
                    if event == self._vad.SPEECH_START:
                        # Notify client first, then barge-in (app/main.py:2845-2893).
                        await self._send_event({
                            "type": SERVER_VAD_EVENT,
                            "event": VAD_EVENT_SPEECH_START,
                        })
                        await self._bargein_tts()
                        if not await self._open_asr_turn():
                            continue
                    elif event == self._vad.SPEECH_END:
                        # Defer endpoint flag until AFTER accepting this chunk
                        # (BUG 3, app/main.py:2894-2900).
                        speech_ended_now = True

                # No-VAD: open lazily on first audio (app/main.py:2901-2921).
                if self._vad is None and not state["asr_active"]:
                    if not await self._open_asr_turn():
                        continue

                if state["asr_active"]:
                    await self._asr_mgr.accept_audio(samples)

                # Now safe to latch endpoint (app/main.py:2929-2942).
                if speech_ended_now:
                    state["endpoint_pending"] = "vad"
                    state["endpoint_pending_gen"] = state["asr_active_gen"]
                    if not multi:
                        state["asr_session_closed"] = True
                    await self._send_event({
                        "type": SERVER_VAD_EVENT,
                        "event": VAD_EVENT_SPEECH_END,
                    })
        except Exception:
            logger.exception("voxedge audio_loop error")
            state["client_closed"] = True

    async def _event_loop(self) -> None:
        """Inbound control events. Port of the text-frame branch of
        dispatcher() (app/main.py:2944-2983)."""
        state = self.state
        multi = self.engine.multi_utterance
        try:
            async for payload in self.transport.recv_event():
                if state["client_closed"]:
                    break
                typ = payload.get("type")
                if typ == CLIENT_TEXT and self._tts_buffer is not None:
                    for sentence in self._tts_buffer.add(payload.get("text", "")):
                        await self._tts_q.put(sentence)
                elif typ == CLIENT_TTS_FLUSH:
                    if self._tts_buffer is not None:
                        for sentence in self._tts_buffer.flush():
                            await self._tts_q.put(sentence)
                    state["tts_flush"] = True
                elif typ == CLIENT_ASR_EOS:
                    state["endpoint_pending"] = "client_eos"
                    state["endpoint_pending_gen"] = state["asr_active_gen"]
                    if not multi:
                        state["asr_session_closed"] = True
                elif typ == CLIENT_ABORT:
                    # _bargein_tts now owns the full TTS cleanup (cancel synth +
                    # LLM turn, drain queue, reset buffer) for both barge-in
                    # paths. Here we additionally cancel the ASR turn — the user
                    # explicitly aborted, so (unlike VAD SPEECH_START) no new
                    # utterance is being opened.
                    await self._bargein_tts()
                    if self._asr_mgr is not None and state["asr_active"]:
                        await self._asr_mgr.cancel("abort")
                        state["asr_active"] = False
                        state["asr_turn_started_at"] = None
                elif typ == CLIENT_TOOL_RESULT:
                    # Remote-tool wire (spec §4 Mode B): route a device client's
                    # tool result back to the awaiting _dispatch_remote future.
                    # ``ok=false`` carries an error string; unknown/late call_id
                    # is safely ignored by resolve_remote.
                    registry = self.engine.tool_registry
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
                    self._handle_tool_advertise(payload)
        except Exception:
            logger.exception("voxedge event_loop error")
            state["client_closed"] = True

    def _handle_tool_advertise(self, payload: dict) -> None:
        """Register client-advertised tool schemas into the engine registry
        as remote-dispatch tools (spec §4 Mode B / §6 handshake).

        ``payload`` carries:
          * ``tools``  — a list of OpenAI-style ``{"type":"function",
            "function":{"name","description","parameters"}}`` dicts (the exact
            shape ``ToolRegistry.list_openai_tools`` emits), OR bare
            ``{"name","description","parameters"}`` dicts. Each becomes a
            ``dispatch_mode="remote"`` tool with a no-op local handler (the
            registry's remote path never calls ``fn``; execution is proxied to
            the client over SERVER_TOOL_CALL).
          * ``system_prompt`` — optional; overrides the engine system prompt for
            this session (read per-turn by the tool pump).
          * ``llm_params`` — optional dict merged into the engine LLM params.

        No-op (logged) when no registry is wired (server loop off). Re-advertise
        is allowed: a tool name already present is overwritten (re-registration),
        and the schema set defines what the LLM sees from the next turn on."""
        registry = self.engine.tool_registry
        if registry is None:
            logger.warning("tool_advertise ignored: no tool_registry (server loop off)")
            return
        tools = payload.get("tools") or []
        registered: list[str] = []
        for entry in tools:
            if not isinstance(entry, dict):
                continue
            # Accept both the full {"type":"function","function":{...}} wrapper
            # and a bare {"name","description","parameters"} dict.
            fn_block = entry.get("function") if isinstance(entry.get("function"), dict) else entry
            name = fn_block.get("name")
            if not name:
                continue
            schema = fn_block.get("parameters") or {"type": "object", "properties": {}}
            description = fn_block.get("description", "")
            registry.register(
                name,
                schema,
                _advertised_remote_noop,
                timeout_s=float(entry.get("timeout_s", 15.0)),
                preamble_text=str(entry.get("preamble_text", "")),
                completion_text=str(entry.get("completion_text", "")),
                response_mode=str(entry.get("response_mode", "await")),
                dispatch_mode="remote",
                description=description,
            )
            registered.append(name)
        sys_prompt = payload.get("system_prompt")
        if isinstance(sys_prompt, str) and sys_prompt:
            self.engine.system_prompt = sys_prompt
        llm_params = payload.get("llm_params")
        if isinstance(llm_params, dict) and llm_params:
            self.engine.llm_params = {**(self.engine.llm_params or {}), **llm_params}
        logger.info(
            "tool_advertise: registered %d remote tool(s) %s (system_prompt=%s)",
            len(registered), registered, "set" if sys_prompt else "unchanged",
        )
        # Warm-up regression fix: in server-loop the agent skips its local LLM
        # warmup (LLM runs here), so prime edge-llm now that the real prefix
        # (system_prompt + tools) is known — otherwise the FIRST user turn pays
        # a cold prefill + CUDA-graph capture. Fire-and-forget, gated per prefix
        # signature so reconnect re-advertises don't re-warm.
        self._maybe_warm_llm_prefix(registered)

    def _maybe_warm_llm_prefix(self, tool_names: list[str]) -> None:
        """Schedule a one-shot edge-llm prefix warm-up if the advertised prefix
        (system_prompt + tool set) hasn't been warmed yet. Fire-and-forget."""
        if self._llm_be is None or self.engine.tool_registry is None:
            return
        sig = (self.engine.system_prompt or "", tuple(sorted(tool_names)))
        if sig == self._warmed_prefix_sig:
            return  # already warmed (success)
        # A warm-up is already in flight → let it finish rather than cancel +
        # relaunch (a reconnect storm would otherwise thrash it). The next
        # advertise re-checks the signature once it completes.
        if self._warm_task is not None and not self._warm_task.done():
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no loop (shouldn't happen here)
            return
        # NOTE: _warmed_prefix_sig is set inside _warm_llm_prefix ON SUCCESS
        # only — if the warm-up fails/cancels, the signature stays unset so a
        # later advertise retries instead of being permanently suppressed.
        self._warm_task = asyncio.create_task(self._warm_llm_prefix(sig))

    async def _warm_llm_prefix(self, sig) -> None:
        """Drive one throwaway LLM request with the real system_prompt + tools
        so edge-llm caches the KV prefix and captures its CUDA graph here, not
        on the user's first command. Marks ``sig`` warmed ONLY on success, so a
        failed/cancelled warm-up doesn't permanently suppress future retries.
        Best-effort: any failure is logged and swallowed (the first real turn
        just pays cold-start)."""
        try:
            registry = self.engine.tool_registry
            if registry is None or self._llm_be is None:
                return
            tools_schema = registry.list_openai_tools() or None
            messages: list[dict[str, Any]] = []
            if self.engine.system_prompt:
                messages.append({"role": "system", "content": self.engine.system_prompt})
            messages.append({"role": "user", "content": "hi"})
            # Same sampling params the real turn uses, but cap generation — we
            # only need the prefill + graph capture, not a full reply.
            params = dict(self.engine.llm_params or {})
            params["max_tokens"] = 1
            t0 = self._loop.time()
            n = 0
            async for _ev in self._llm_be.stream_events(messages, tools=tools_schema, **params):
                n += 1
            self._warmed_prefix_sig = sig  # mark warmed only after success
            logger.info(
                "server-loop: edge-llm prefix warm-up done in %.2fs (%d events, %d tool(s))",
                self._loop.time() - t0, n, len(tools_schema or []),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.info(
                "server-loop: edge-llm prefix warm-up failed (non-fatal — first "
                "turn pays cold-start; will retry on next advertise)", exc_info=True,
            )

    async def _bargein_tts(self) -> None:
        """Stop everything the current turn is producing and discard any
        queued/buffered audio, so a barge-in leaves nothing stale to play
        (app/main.py:2856-2861, 2966-2972).

        Order matters: stop the producers (TTS synth + LLM turn) and wait for
        the LLM turn to actually wind down BEFORE draining the queue, else it
        could re-enqueue a sentence between drain iterations (#5). Used by both
        barge-in paths — VAD SPEECH_START and CLIENT_ABORT — so cleanup is
        identical.
        """
        # 1. Stop the in-flight TTS synth.
        t = self.state["current_tts_task"]
        if t is not None and not t.done():
            t.cancel()
        stop = self.state["current_tts_stop"]
        if stop is not None:
            stop.set()
        # 2. #2: stop the in-flight LLM/tool turn. Set the cooperative flag
        # (the turn polls it at each checkpoint), then unblock any in-flight
        # remote tool dispatch so the turn reaches its next checkpoint fast.
        self.state["llm_barged"] = True
        # Phase 2a: barge-in must not orphan in-flight remote tool calls
        # (spec §7). Clearing pending remote-dispatch futures makes their
        # awaiting _dispatch_remote return a recoverable "cancelled" dict.
        registry = self.engine.tool_registry
        if registry is not None:
            cleared = registry.cancel_pending_remote()
            if cleared:
                logger.debug("barge-in cleared %d pending remote tool call(s)", cleared)
        # Wait (bounded) for the turn to wind down. shield() so a timeout
        # never cancels the task — a cancel landing mid remote-dispatch would
        # be swallowed by _dispatch_remote and NOT stop the turn anyway; the
        # cooperative flag is the reliable stop. The bound keeps _audio_loop
        # responsive if the turn is wedged in a sync tool / silent backend.
        llm = self.state.get("current_llm_task")
        if llm is not None and not llm.done():
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(llm), timeout=2.0)
        # 3. #5: discard queued sentences + the half-accumulated buffer. The
        # producers are stopped (steps 1-2), so this can't race an enqueue.
        while not self._tts_q.empty():
            try:
                self._tts_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._tts_be is not None:
            self._tts_buffer = self._make_tts_buffer()
        self.state["tts_flush"] = False

    async def _open_asr_turn(self) -> bool:
        """Start a fresh ASR utterance via the session manager.

        Port of app/main.py:2862-2893 (VAD speech_start) / 2901-2921
        (no-VAD lazy open). On ``ASRSessionUnavailable`` (rebuild ladder
        exhausted — race #1) surface ``asr_unavailable`` and DON'T flag the
        session active, so audio isn't silently dropped with the client
        stuck. Returns True iff a turn was opened.
        """
        state = self.state
        try:
            new_gen = await self._asr_mgr.on_speech_start()
        except ASRSessionUnavailable as e:
            state["asr_active"] = False
            state["endpoint_pending"] = None
            state["endpoint_pending_gen"] = None
            # M4: the manager wraps create_stream failures in
            # ASRSessionUnavailable (raised ``from`` the original). If the
            # root cause was a slot-pool saturation, surface the typed
            # pool_saturated reject instead of a generic asr_unavailable so
            # the client knows to retry (saturation is "busy", not a fault).
            sat, max_slots = _is_pool_saturated(e.__cause__ or e)
            if sat:
                await self._emit_pool_saturated(max_slots)
                return False
            logger.warning("voxedge: on_speech_start failed: %s", e)
            await self._send_event({"type": SERVER_ERROR, "error": "asr_unavailable"})
            return False
        except Exception as e:  # noqa: BLE001
            sat, max_slots = _is_pool_saturated(e)
            if sat:
                # M4: backend slot-pool saturated → typed 4429, not a fault.
                await self._emit_pool_saturated(max_slots)
                state["asr_active"] = False
                return False
            raise
        state["endpoint_pending"] = None
        state["endpoint_pending_gen"] = None
        state["asr_active"] = True
        state["asr_active_gen"] = new_gen
        # M1: anchor the per-turn wall-clock deadline (app/main.py:2892).
        state["asr_turn_started_at"] = self._loop.time()
        return True

    async def _emit_pool_saturated(self, max_slots: Optional[int]) -> None:
        """M4: typed pool_saturated event (app/main.py:3348-3353)."""
        payload = {"type": SERVER_ERROR, "error": "pool_saturated", "status": 4429}
        if max_slots is not None:
            payload["max_slots"] = max_slots
        await self._send_event(payload)

    # ══════════════════════════════════════════════════════════════════
    # asr_out_task  ← app/main.py:2992-3205
    # ══════════════════════════════════════════════════════════════════

    async def _asr_out_task(self) -> None:
        state = self.state
        multi = self.engine.multi_utterance
        last_streamed_final = None
        last_partial: tuple[int, str] = (-1, "")
        asr_turn_timeout_s = self._asr_turn_timeout_s
        while not state["client_closed"]:
            # ── M1: wall-clock per-turn deadline (app/main.py:3012-3077) ─
            # Active turn that hasn't finalized within the deadline → force
            # cancel + worker restart (via the manager's bounded cancel
            # ladder) so a wedged backend can't pin the session forever.
            turn_started = state.get("asr_turn_started_at")
            if (
                state.get("asr_active")
                and turn_started is not None
                and (self._loop.time() - turn_started) > asr_turn_timeout_s
            ):
                elapsed = self._loop.time() - turn_started
                logger.warning(
                    "voxedge ASR turn exceeded %.1fs wall-clock (elapsed=%.1fs); "
                    "aborting turn + force-cancel ASR session",
                    asr_turn_timeout_s, elapsed,
                )
                # Step 1: cooperative cancel with a tight budget; the manager
                # escalates to restart_worker internally on its own timeout
                # (app/main.py:3030-3054).
                if self._asr_mgr is not None:
                    try:
                        await asyncio.wait_for(
                            self._asr_mgr.cancel("turn_timeout"), timeout=2.0
                        )
                    except Exception as _exc:  # noqa: BLE001
                        logger.error(
                            "voxedge ASR cancel timed out / failed (%s)", _exc
                        )
                # Step 2: clear state + emit error so the client unwinds
                # (app/main.py:3055-3068).
                state["asr_active"] = False
                state["asr_turn_started_at"] = None
                state["endpoint_pending"] = None
                state["endpoint_pending_gen"] = None
                try:
                    await self._send_error(
                        f"asr: per-turn deadline {asr_turn_timeout_s:.0f}s exceeded"
                    )
                except Exception:
                    logger.exception("voxedge send_error after asr turn timeout failed")
                if multi and not state["asr_session_closed"]:
                    await asyncio.sleep(0.05)
                    continue
                return

            # ── partial poll (app/main.py:3084-3096) ──────────────────
            if state["asr_active"]:
                try:
                    # M2: atomic (gen, partial, is_endpoint) snapshot under
                    # the manager's lock — no torn read against a stream that
                    # a barge-in is replacing.
                    partial_gen, partial, is_endpoint = (
                        await self._asr_mgr.get_partial_for_generation()
                    )
                except Exception:
                    partial, is_endpoint, partial_gen = "", False, 0
                # Gen gate (BUG 4): drop partials from a replaced utterance.
                # Also dedupe identical consecutive partials so the fast poll
                # loop doesn't flood the client (prod tracks last_streamed_final
                # similarly, app/main.py:3001/3187).
                if (
                    partial
                    and partial_gen == state["asr_active_gen"]
                    and (partial_gen, partial) != last_partial
                ):
                    last_partial = (partial_gen, partial)
                    await self._send_event({
                        "type": SERVER_ASR_PARTIAL,
                        "text": partial,
                        "is_stable": bool(is_endpoint),
                    })
            else:
                is_endpoint = False

            # ── endpoint resolution (app/main.py:3098-3119) ───────────
            endpoint_reason = state["endpoint_pending"]
            # Gen-race gate (app/main.py:3107-3114): endpoint stamped against a
            # generation that has since been preempted → drop on the floor.
            if (
                endpoint_reason
                and state["endpoint_pending_gen"] is not None
                and state["endpoint_pending_gen"] != state["asr_active_gen"]
            ):
                state["endpoint_pending"] = None
                state["endpoint_pending_gen"] = None
                endpoint_reason = None

            endpoint_fired = bool(endpoint_reason) or (is_endpoint and state["asr_active"])

            if endpoint_fired:
                state["endpoint_pending"] = None
                state["endpoint_pending_gen"] = None
                if endpoint_reason != "client_eos":
                    await self._send_event({"type": SERVER_ASR_ENDPOINT})

                if state["asr_active"]:
                    finalize_gen = state["asr_active_gen"]
                    # M2: finalize_with_status suppresses stale/cancelled
                    # results (accepted=False) so a finalize that raced a
                    # barge-in doesn't emit a spurious final
                    # (app/core/asr_session_manager.py:265-320).
                    # Slot acquire: ASR finalize is the GPU-heavy decode; in
                    # serialized/exclusive mode it must not overlap a TTS synth.
                    async with self._acquire("asr"):
                        fin_gen, fin_text, accepted, detected_language = (
                            await self._asr_mgr.finalize_with_status(
                                endpoint_reason or "vad_end"
                            )
                        )
                    if accepted:
                        final_text = fin_text
                    else:
                        final_text, detected_language = "", None
                    # Only clear active if generation still current (BUG 2).
                    if state["asr_active_gen"] == finalize_gen:
                        state["asr_active"] = False
                        state["asr_turn_started_at"] = None
                else:
                    final_text, detected_language = "", None

                # ── emit asr_final (app/main.py:3164-3197) ────────────
                if multi:
                    is_closing = state["asr_session_closed"]
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
                    await self._send_event(payload)
                    # Closed-loop hook: feed final text to LLM→TTS (spec §4).
                    # Only drive the LLM on genuinely new ASR text — a close-out
                    # duplicate must not re-trigger another LLM→TTS turn.
                    if final_text and final_text.strip():
                        await self._on_asr_final(final_text)
                    if is_closing:
                        return
                    last_streamed_final = final_text or ""
                else:
                    payload = {"type": SERVER_ASR_FINAL, "text": final_text or ""}
                    if detected_language:
                        payload["language"] = detected_language
                    await self._send_event(payload)
                    await self._on_asr_final(final_text or "")
                    return

            # Exit when closed + nothing left (app/main.py:3202-3203).
            if state["asr_session_closed"] and not state["asr_active"]:
                return
            await asyncio.sleep(0.02)

    async def _on_asr_final(self, text: str) -> None:
        """asr_final → LLM (optional) → TTS sentence buffer (spec §4).

        With no LLM this is a no-op (pure ASR/TTS pass-through; the client
        drives TTS directly via CLIENT_TEXT). With an LLM backend, stream its
        text deltas into the TTS sentence buffer — equivalent to prod's
        CLIENT_TEXT path (app/main.py:2953-2960).

        When the engine was constructed with a ``tool_registry`` (spec §4),
        the LLM call runs the server-side multi-turn tool pump
        (:meth:`_llm_turn_with_tools`) instead of the plain text stream. With
        ``tool_registry=None`` (the default) this method's behavior is
        byte-identical to before the tool migration (Phase 1 hard contract).
        """
        if self._llm_be is None or not text.strip():
            return
        if self._tts_buffer is None:
            return
        if self.engine.tool_registry is not None:
            # #2: run the tool loop as a cancellable task so a concurrent
            # barge-in (_audio_loop's VAD / CLIENT_ABORT → _bargein_tts) can
            # stop an in-flight turn. Reset the cooperative flag and create the
            # task in one synchronous step (no await between) so a barge-in
            # arriving right after create_task isn't lost to the reset.
            self.state["llm_barged"] = False
            task = asyncio.create_task(
                self._llm_turn_with_tools([{"role": "user", "content": text}])
            )
            self.state["current_llm_task"] = task
            try:
                await task
            except asyncio.CancelledError:
                if task.cancelled():
                    # Barge-in (or a defensive cancel) stopped this turn —
                    # swallow so _asr_out_task continues to the next utterance.
                    logger.info("voxedge LLM turn cancelled (barge-in)")
                else:
                    # _asr_out_task itself is being torn down: make sure the
                    # child task dies too, then propagate the teardown.
                    task.cancel()
                    raise
            finally:
                if self.state.get("current_llm_task") is task:
                    self.state["current_llm_task"] = None
            return
        messages = [{"role": "user", "content": text}]
        try:
            async for ev in self._llm_be.stream_events(messages):
                if ev.kind == "text" and ev.text:
                    for sentence in self._tts_buffer.add(ev.text):
                        await self._tts_q.put(sentence)
                # tool_call_delta is only reachable via the tool_registry path
                # above; with no registry the engine never sends a tools schema
                # so the backend never emits tool calls.
            for sentence in self._tts_buffer.flush():
                await self._tts_q.put(sentence)
            self.state["tts_flush"] = True
        except Exception:
            logger.exception("voxedge LLM stream failed")

    async def _enqueue_tts_text(self, chunk: str) -> None:
        """Push assistant text into the TTS sentence buffer (same path LLM
        text deltas take in :meth:`_on_asr_final`)."""
        if not chunk or self._tts_buffer is None:
            return
        for sentence in self._tts_buffer.add(chunk):
            await self._tts_q.put(sentence)

    async def _llm_turn_with_tools(self, messages: list[dict[str, Any]]) -> None:
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
        in :meth:`_on_asr_final` is unaffected (Phase 1 contract).
        """
        registry = self.engine.tool_registry
        tools_schema = registry.list_openai_tools() or None
        # Prepend the server-loop system prompt (spec §5). Done once on the
        # working message list so every round re-sends the same prefix (keeps
        # the edge-LLM prefix cache stable — append-only history per §8).
        sys_prompt = self.engine.system_prompt
        if sys_prompt and not (messages and messages[0].get("role") == "system"):
            messages = [{"role": "system", "content": sys_prompt}, *messages]
        llm_params = self.engine.llm_params
        ctx = ToolContext(
            session_id=getattr(self.transport, "session_id", None),
            conversation=self,
            # Phase 2a: remote-dispatch tools push their tool_call frame over
            # the same event channel as other server→client events; the
            # correlated tool_result is routed back via
            # ``registry.resolve_remote`` from the transport receive side.
            remote_send=self.transport.send_event,
        )
        max_rounds = self.engine.max_tool_rounds
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
                if self.state.get("llm_barged"):
                    return

                async for ev in self._llm_be.stream_events(
                    messages, tools=tools_schema, **llm_params
                ):
                    # #2: barge-in mid-stream — stop consuming deltas and drop
                    # the turn WITHOUT flushing the partial sentence buffer.
                    if self.state.get("llm_barged"):
                        return
                    if _ttft is None:
                        _ttft = time.perf_counter() - _t_round_start
                    if ev.kind == "text" and ev.text:
                        text_chunks.append(ev.text)
                        await self._enqueue_tts_text(ev.text)
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
                    if self.state.get("llm_barged"):
                        return
                    final_text = "".join(text_chunks)
                    if final_text:
                        messages.append({"role": "assistant", "content": final_text})
                    for sentence in self._tts_buffer.flush():
                        await self._tts_q.put(sentence)
                    self.state["tts_flush"] = True
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
                if self.state.get("llm_barged"):
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
                    await self._enqueue_tts_text(spoken)
                    for sentence in self._tts_buffer.flush():
                        await self._tts_q.put(sentence)
                    self.state["tts_flush"] = True
                    return
                # loop: re-request the LLM with the tool results appended.

            # Iteration cap hit — flush whatever was spoken and finish.
            logger.warning("voxedge tool loop hit max_tool_rounds=%d", max_rounds)
            if self.state.get("llm_barged"):
                return
            for sentence in self._tts_buffer.flush():
                await self._tts_q.put(sentence)
            self.state["tts_flush"] = True
        except Exception:
            logger.exception("voxedge LLM tool loop failed")
            try:
                for sentence in self._tts_buffer.flush():
                    await self._tts_q.put(sentence)
                self.state["tts_flush"] = True
            except Exception:
                logger.exception("voxedge tool-loop flush after error failed")

    async def _emit_preamble(self, preamble_text: str) -> None:
        """Speak a tool preamble (e.g. "好的。") via the TTS buffer.

        TODO(phase-2): if the TTS buffer / app grows a dedicated preamble
        interface (immediate flush, bypass sentence buffering for lowest
        latency), route through it. For now the preamble goes through the same
        sentence buffer as assistant text — correct, just not latency-tuned.
        """
        await self._enqueue_tts_text(preamble_text)

    # ══════════════════════════════════════════════════════════════════
    # tts_out_task  ← app/main.py:3207-3432
    # ══════════════════════════════════════════════════════════════════

    async def _tts_out_task(self) -> None:
        state = self.state
        multi = self.engine.multi_utterance
        sr_header_sent = False
        while not state["client_closed"]:
            # Exit / per-turn done when flush + drained (app/main.py:3217-3233).
            if state["tts_flush"] and self._tts_q.empty():
                if multi and not state["asr_session_closed"]:
                    state["tts_flush"] = False
                    if not state["client_closed"]:
                        await self._send_event({
                            "type": SERVER_TTS_DONE,
                            "session_complete": False,
                        })
                    continue
                break
            try:
                sentence = await asyncio.wait_for(self._tts_q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            audio_queue: asyncio.Queue = asyncio.Queue()
            stop_event = threading.Event()
            state["current_tts_stop"] = stop_event

            # Engine-parity #15: resolve speaker / voice / speed kwargs once
            # per sentence (same dict the legacy _run_synth builds,
            # app/main.py:3473-3478). Empty when nothing injected → default
            # speaker, unchanged behavior.
            stream_kwargs = self._tts_stream_kwargs()

            def _run_synth(s: str, ev: threading.Event, aq: asyncio.Queue):
                # Mirrors app/main.py:3246-3281 _run_synth (thread body).
                try:
                    for chunk in self._tts_be.generate_streaming(
                        s,
                        language=self.engine.tts_language,
                        cancel_token=ev,
                        **stream_kwargs,
                    ):
                        if ev.is_set():
                            break
                        self._loop.call_soon_threadsafe(aq.put_nowait, chunk)
                except Exception as e:  # noqa: BLE001
                    # M4: a slot-pool saturation is "backend busy", NOT a
                    # synth fault — surface a typed reject marker, not a
                    # generic tts error (app/main.py:3342-3356).
                    sat, max_slots = _is_pool_saturated(e)
                    if sat:
                        self._loop.call_soon_threadsafe(
                            aq.put_nowait, ("__saturated__", max_slots)
                        )
                    else:
                        logger.exception("voxedge tts synth failed for %r", s[:80])
                        self._loop.call_soon_threadsafe(
                            aq.put_nowait, ("__error__", str(e))
                        )
                finally:
                    self._loop.call_soon_threadsafe(aq.put_nowait, None)

            chunk_timeout_s = self._tts_chunk_timeout_s

            async def drain(s: str, ev: threading.Event, aq: asyncio.Queue):
                nonlocal sr_header_sent
                # Mirrors app/main.py:3283-3361 drain().
                if not sr_header_sent:
                    sr = self._tts_be.sample_rate
                    await self._send_audio(struct.pack("<I", sr))
                    sr_header_sent = True
                await self._send_event({"type": SERVER_TTS_STARTED, "sentence": s})
                # Slot acquire: a TTS synth is the GPU-heavy op; in
                # serialized/exclusive mode it must not overlap an ASR finalize.
                # Held across the whole synth so the synth thread runs alone.
                async with self._acquire("tts"):
                    self._loop.run_in_executor(None, _run_synth, s, ev, aq)
                    state["tts_started"] = True
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
                            await self._send_error(
                                f"tts: synth produced no chunks within "
                                f"{chunk_timeout_s:.0f}s"
                            )
                            break
                        if item is None:
                            break
                        if isinstance(item, tuple) and item[0] == "__saturated__":
                            # M4: typed pool_saturated; keep the session alive.
                            await self._emit_pool_saturated(item[1])
                            break
                        if isinstance(item, tuple) and item[0] == "__error__":
                            await self._send_error(f"tts: {item[1]}")
                            break
                        await self._send_audio(item)
                await self._send_event({"type": SERVER_TTS_SENTENCE_DONE, "sentence": s})

            task = asyncio.create_task(drain(sentence, stop_event, audio_queue))
            state["current_tts_task"] = task
            # M3: outer per-sentence wall-clock deadline. Covers wedges BEFORE
            # the first chunk watchdog can fire (e.g. a backend that hangs in
            # generate_streaming setup before yielding) (app/main.py:3382-3409).
            sentence_timeout_s = self._tts_sentence_timeout_s
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
                if not state["client_closed"]:
                    try:
                        await self._send_error(
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
                state["current_tts_task"] = None
                state["current_tts_stop"] = None

        # Session-final tts_done (app/main.py:3424-3432).
        if not state["client_closed"]:
            payload = {"type": SERVER_TTS_DONE}
            if multi:
                payload["session_complete"] = True
            await self._send_event(payload)

    # ══════════════════════════════════════════════════════════════════
    # orchestrate  ← app/main.py:3434-3457
    # ══════════════════════════════════════════════════════════════════

    async def run(self) -> None:
        """Drive one full conversation, then close the transport.

        Like app/main.py:3439-3457: spawn the receive loops + work tasks,
        wait for the work tasks (which terminate on session close), then
        cancel the still-looping receive tasks.

        Graceful end-of-input: the production dispatcher treats a
        ``websocket.disconnect`` (no more client frames) as the session-end
        signal (app/main.py:2819-2830). The in-process equivalent is the
        inbound audio + event streams both being exhausted (the client called
        ``end_input()``). When that happens we mark the ASR session closed so
        the work tasks emit their close-out finals — asr_final /
        tts_done with ``session_complete=True`` in multi-utterance — and
        terminate cleanly, instead of polling forever until force-cancelled.
        """
        recv_tasks = [
            asyncio.create_task(self._audio_loop()),
            asyncio.create_task(self._event_loop()),
        ]
        work_tasks: list[asyncio.Task] = []
        if self.asr_enabled:
            work_tasks.append(asyncio.create_task(self._asr_out_task()))
        if self._tts_be is not None:
            work_tasks.append(asyncio.create_task(self._tts_out_task()))
        self._work_tasks = work_tasks

        async def _watch_input_end() -> None:
            # Once the client stops feeding (both recv loops drained), the
            # session is over. This drain only happens when the transport
            # enqueues _CLOSE — i.e. a real WS close / disconnect. A quiet but
            # still-connected client keeps the pump blocked on ws.receive(),
            # so the recv loops never drain and this never fires (the critical
            # "don't kill a silent client" invariant lives in the transport).
            #
            # Slot-leak fix (3b-ii): on the ASR+TTS V2V path the close-out
            # MUST also terminate _tts_out_task, otherwise it spins forever in
            # `while not client_closed` and the Session.run() gather never
            # resolves → product finally never runs → limiter slot leaks. Set:
            #   * asr_session_closed → in-flight ASR turn finalizes (close-out)
            #   * tts_flush          → _tts_out_task drains its queue, emits its
            #                          final tts_done, then BREAKS — because with
            #                          asr_session_closed=True the `multi and not
            #                          asr_session_closed` re-arm guard at the top
            #                          of _tts_out_task is False, so the flush
            #                          branch falls through to `break`.
            # Previously the flush was gated behind `if not self.asr_enabled`,
            # so the full ASR+TTS loop never flushed and _tts_out_task hung
            # forever — that is the leak this removes.
            #
            # We deliberately do NOT set client_closed here: that would
            # short-circuit the work-task loops BEFORE they emit their terminal
            # close-out events (asr_final{session_complete=True} / final
            # tts_done) and abort an in-flight close-out synth. client_closed is
            # set by the run() finally, AFTER the gather has resolved (i.e. the
            # work tasks have drained and emitted their finals).
            await asyncio.gather(*recv_tasks, return_exceptions=True)
            self.state["asr_session_closed"] = True
            self.state["tts_flush"] = True
            # Don't orphan in-flight remote tool futures on disconnect: clear
            # them (same barge-in cancel mechanism, conversation.py:528-535 /
            # tool_registry.py:404-416) so any awaiting _dispatch_remote
            # returns a recoverable abort instead of hanging a multi-turn pump.
            registry = self.engine.tool_registry
            if registry is not None:
                try:
                    cleared = registry.cancel_pending_remote()
                    if cleared:
                        logger.debug(
                            "voxedge input-end cleared %d pending remote tool "
                            "call(s)", cleared,
                        )
                except Exception:
                    logger.exception("voxedge cancel_pending_remote on close failed")

        end_watcher = asyncio.create_task(_watch_input_end())

        try:
            if work_tasks:
                # Race the work tasks against input-end: a client WS close
                # makes _watch_input_end set client_closed=True, which breaks
                # both work-task loops so this gather resolves. We also include
                # end_watcher so a degenerate config (no work tasks) — and any
                # path where the work tasks somehow exit before the watcher —
                # still completes deterministically.
                await asyncio.gather(*work_tasks, return_exceptions=False)
            else:
                # No work tasks (degenerate config) — still wait for input end.
                await end_watcher
        except asyncio.CancelledError:
            pass
        finally:
            # M5: ordered teardown (app/main.py:3503-3537). Tear down the
            # backends BEFORE closing the transport so the worker doesn't
            # leak the session and the synth thread is released for the next
            # connection.
            self.state["client_closed"] = True
            if not end_watcher.done():
                end_watcher.cancel()
            # (a) cancel the recv loops + work tasks.
            for t in recv_tasks:
                if not t.done():
                    t.cancel()
            for t in work_tasks:
                if not t.done():
                    t.cancel()
            # Cancel a still-running prefix warm-up so it doesn't outlive the
            # session (fire-and-forget; harmless if already done).
            if self._warm_task is not None and not self._warm_task.done():
                self._warm_task.cancel()
            # (b) signal the synth thread to bail so the TTS executor frees up
            # (app/main.py:3513-3517).
            stop = self.state.get("current_tts_stop")
            if stop is not None:
                try:
                    stop.set()
                except Exception:
                    pass
            # (c) await the cancelled work tasks before releasing the backend
            # slot (app/main.py:3518-3528, race #6).
            _extra = [self._warm_task] if self._warm_task is not None else []
            await asyncio.gather(
                end_watcher, *recv_tasks, *work_tasks, *_extra, return_exceptions=True
            )
            # (d) cancel any in-flight ASR utterance so the worker doesn't
            # leak the session (app/main.py:3529-3535).
            if self._asr_mgr is not None:
                try:
                    await self._asr_mgr.cancel("ws_close")
                except Exception:
                    logger.exception("voxedge: asr cancel on close failed")
            # (e) finally close the transport (close-code 1003/1011 stays in
            # the transport/app layer — engine doesn't decide it).
            try:
                await self.transport.close()
            except Exception:
                pass


class ConversationEngine:
    """Importable V2V engine. Construct with resolved backends + config; no
    env reads (spec §2: ``__init__`` takes parsed config, not env).

    Args:
        backends: dict with keys ``asr`` / ``tts`` / ``vad`` / ``llm``; any
            subset. ASR-only, TTS-only, or full loop all valid.
        tool_registry: optional :class:`~voxedge.engine.tool_registry.ToolRegistry`.
            When provided, ``_on_asr_final`` runs the server-side multi-turn
            LLM↔tool pump (``_llm_turn_with_tools``, spec §2/§4). When ``None``
            (the default) the engine's LLM path is byte-identical to before the
            tool migration (Phase 1 hard contract: tool_registry=None = no-op).
        max_tool_rounds: iteration cap for the tool pump (spec §2, default 5).
        multi_utterance: keep the session alive across turns (True) vs
            single-shot (False).
        timeouts: optional dict of wall-clock watchdog knobs (seconds),
            constructor-injected (spec §2: no env reads). Recognized keys:
            ``asr_turn`` (M1, default 45), ``tts_chunk`` (M3, default 10),
            ``tts_sentence`` (M3, default 15). Defaults mirror prod's env
            defaults (OVS_ASR_TURN_TIMEOUT_S / OVS_TTS_CHUNK_TIMEOUT_S /
            OVS_TTS_SENTENCE_TIMEOUT_S).
        silence_ms: VAD silence threshold passed to ``create_session``.
        asr_language: language hint forwarded to ASR stream creation.
        tts_language: default language hint forwarded to TTS streaming.
        tts_speaker_kwargs: optional pre-resolved speaker dict forwarded to
            the TTS backend's ``generate_streaming`` (engine-parity #15).
            voxedge does NOT import ``app.core.tts_speakers`` — the caller
            resolves the speaker_id → kwargs (e.g. ``{"speaker_id": int,
            "speaker": str}`` or ``{"speaker_embedding": bytes}``) and injects
            the result here, mirroring the legacy ``tts_speaker_kwargs``
            (app/main.py:2823-2829, 3474-3475). Default ``None`` / empty →
            backend uses its default speaker (behavior unchanged).
        tts_voice: optional deprecated voice string fallback, only used when
            ``tts_speaker_kwargs`` is empty (legacy app/main.py:3476-3477).
        tts_speed: optional speech-rate multiplier forwarded to
            ``generate_streaming`` when set (legacy app/main.py:3478).
        low_latency_tts: when True select the ported ``LowLatencyTTSBuffer``
            (CJK clause / bounded-span early emit, lower TTFA) instead of the
            default ``_SentenceBuffer`` — mirrors the legacy buffer choice
            (app/main.py:2961-2970). Default ``False`` keeps the existing
            ``_SentenceBuffer`` behavior (back-compat; existing tests
            unchanged).
        coordinator: optional :class:`BackendCoordinator` (concurrency
            abstraction, spec §3.1). When provided, ASR/TTS backend calls are
            wrapped in ``coord.acquire(...)`` so serialized/exclusive modes
            truly mutually-exclude. When None (default) the engine runs direct
            passthrough — current behavior, backward compatible. Pass an
            instance built via ``BackendCoordinator.from_backends(...)`` to
            resolve the mode from backend capability.
    """

    def __init__(
        self,
        backends: dict[str, Any],
        *,
        tool_registry: Optional[Any] = None,
        max_tool_rounds: int = 5,
        system_prompt: Optional[str] = None,
        llm_params: Optional[dict] = None,
        multi_utterance: bool = False,
        timeouts: Optional[dict] = None,
        silence_ms: int = 400,
        asr_language: str = "auto",
        tts_language: Optional[str] = None,
        tts_speaker_kwargs: Optional[dict] = None,
        tts_voice: Optional[str] = None,
        tts_speed: Optional[float] = None,
        low_latency_tts: bool = False,
        coordinator: Optional[BackendCoordinator] = None,
    ):
        self.backends = backends
        self.tool_registry = tool_registry
        self.max_tool_rounds = max_tool_rounds
        # Server-loop system prompt + LLM params (spec §2/§5). Injected by the
        # product (env/config/profile) — the engine never reads env itself.
        # Used only on the tool-pump path; the no-tool plain-text path is
        # unaffected (Phase 1 contract).
        self.system_prompt = system_prompt
        self.llm_params = dict(llm_params or {})
        self.multi_utterance = multi_utterance
        self.timeouts = timeouts or {}
        self.silence_ms = silence_ms
        self.asr_language = asr_language
        self.tts_language = tts_language
        # Engine-parity #15: TTS speaker / voice / speed + buffer selection.
        self.tts_speaker_kwargs = tts_speaker_kwargs or {}
        self.tts_voice = tts_voice
        self.tts_speed = tts_speed
        self.low_latency_tts = low_latency_tts
        self.coordinator = coordinator

        # M1/M3: resolve watchdog thresholds from the injected dict (env-free).
        self.asr_turn_timeout_s = float(self.timeouts.get("asr_turn", 45.0))
        self.tts_chunk_timeout_s = float(self.timeouts.get("tts_chunk", 10.0))
        self.tts_sentence_timeout_s = float(self.timeouts.get("tts_sentence", 15.0))

        # Preload ready backends once (app/main.py does this at startup).
        for be in backends.values():
            preload = getattr(be, "preload", None)
            if callable(preload):
                try:
                    if not (getattr(be, "is_ready", lambda: False)()):
                        preload()
                except Exception:
                    logger.exception("voxedge backend preload failed: %r", be)

    async def run(self, transport: Transport) -> None:
        """Drive one conversation over ``transport`` to completion."""
        await Session(self, transport).run()


__all__ = ["ConversationEngine", "Session"]
