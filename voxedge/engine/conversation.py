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
from typing import Any, Optional

import numpy as np

from voxedge.backends.base import ASRBackend, LLMBackend, TTSBackend, VADBackend
from voxedge.engine.tool_registry import ToolContext
from voxedge.engine.asr_session_manager import (
    ASRSessionManager,
    ASRSessionUnavailable,
)
from voxedge.engine.asr_loop import _ASRLoop
from voxedge.engine.llm_turn import _LLMTurn
from voxedge.engine.audio_dispatcher import _AudioDispatcher
from voxedge.engine.client_events import _ClientEvents
from voxedge.engine.coordinator import BackendCoordinator
from voxedge.engine.session_state import SessionState
from voxedge.engine.tts_buffer import LowLatencyTTSBuffer
from voxedge.engine.tts_sequencer import _TTSChannel
from voxedge.transport.base import Transport


@contextlib.asynccontextmanager
async def _passthrough():
    """No-op async context manager — used when no coordinator is wired so the
    engine keeps its current direct-call behavior (backward compatible)."""
    yield

logger = logging.getLogger(__name__)

# Protocol constants + the pool-saturation duck-type now live in
# voxedge/engine/protocol.py so the split-out submodules can share them without
# importing conversation (cycle-free). Re-exported here for back-compat — tests
# and callers still do ``from voxedge.engine.conversation import SERVER_TOOL_CALL``.
from voxedge.engine.protocol import (  # noqa: E402
    CLIENT_TEXT,
    CLIENT_ASR_EOS,
    CLIENT_TTS_FLUSH,
    CLIENT_ABORT,
    CLIENT_TOOL_RESULT,
    CLIENT_TOOL_ADVERTISE,
    SERVER_ASR_PARTIAL,
    SERVER_ASR_ENDPOINT,
    SERVER_ASR_FINAL,
    SERVER_TTS_STARTED,
    SERVER_TTS_SENTENCE_DONE,
    SERVER_TTS_DONE,
    SERVER_VAD_EVENT,
    SERVER_ERROR,
    SERVER_TOOL_CALL,
    VAD_EVENT_SPEECH_START,
    VAD_EVENT_SPEECH_END,
    _is_pool_saturated,
)


def _advertised_remote_noop(*_args, **_kwargs):  # pragma: no cover - never called
    """Placeholder local handler for a client-advertised remote tool.

    The registry's ``dispatch_mode="remote"`` path proxies execution to the
    device client over the wire and never invokes ``fn``; this exists only so
    ``ToolRegistry.register`` (which requires a callable) has something to hold.
    """
    raise RuntimeError("advertised remote tool must dispatch over the wire, not locally")


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

        # Step 2 of the split: the TTS sentence queue + chunk buffer and every
        # write to them live in one facade (selects _SentenceBuffer vs
        # LowLatencyTTSBuffer via _make_tts_buffer). The consumer loop is still
        # Session._tts_out_task (reads self._tts.q); it relocates in step 3.
        self._tts = _TTSChannel(self)
        # Step 4: ASR turn open + the partial/endpoint/finalize loop live in
        # _ASRLoop; _on_asr_final / _emit_pool_saturated / the ASRSessionManager
        # stay on Session (bridge + shared with TTS).
        self._asr = _ASRLoop(self)
        # Step 5: inbound audio → VAD → ASR feed lives in _AudioDispatcher;
        # barge-in (_bargein_tts) + ASR turn open/close stay on Session/_ASRLoop.
        self._audio = _AudioDispatcher(self)
        # Step 6: inbound control-event demux lives in _ClientEvents; its
        # handlers stay on Session (_tts / _bargein_tts / _handle_tool_advertise).
        self._events = _ClientEvents(self)
        # Step 7: the server-side multi-round LLM↔tool pump lives in _LLMTurn;
        # _on_asr_final (the ASR→LLM/TTS bridge) stays on Session and drives it.
        self._llm = _LLMTurn(self)
        self._loop = asyncio.get_event_loop()

        # M1/M3: wall-clock watchdog thresholds — constructor-injected (spec
        # §2: no env reads in the engine). Defaults mirror prod env defaults
        # (OVS_ASR_TURN_TIMEOUT_S=45, OVS_TTS_CHUNK_TIMEOUT_S=10,
        # OVS_TTS_SENTENCE_TIMEOUT_S=15).
        self._asr_turn_timeout_s = engine.asr_turn_timeout_s
        self._tts_chunk_timeout_s = engine.tts_chunk_timeout_s
        self._tts_sentence_timeout_s = engine.tts_sentence_timeout_s

        # per-conn state (mirrors app/main.py:2751-2791 state dict). Step 1 of
        # the conversation.py split: a typed SessionState with grouped
        # transition methods, replacing the ad-hoc dict. Field defaults +
        # transitions are a 1:1 port — no behaviour change. See
        # voxedge/engine/session_state.py.
        self.state = SessionState()
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
            self.state.client_closed = True

    async def _send_audio(self, data: bytes) -> None:
        try:
            await self.transport.send_audio(data)
        except Exception:
            self.state.client_closed = True

    async def _send_error(self, msg: str) -> None:
        await self._send_event({"type": SERVER_ERROR, "error": msg})

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
        self._tts.interrupt_synth()
        # 2. #2: stop the in-flight LLM/tool turn. Set the cooperative flag
        # (the turn polls it at each checkpoint), then unblock any in-flight
        # remote tool dispatch so the turn reaches its next checkpoint fast.
        self.state.llm_barged = True
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
        llm = self.state.current_llm_task
        if llm is not None and not llm.done():
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(llm), timeout=2.0)
        # 3. #5: discard queued sentences + the half-accumulated buffer. The
        # producers are stopped (steps 1-2), so this can't race an enqueue.
        self._tts.drain_and_reset()

    async def _emit_pool_saturated(self, max_slots: Optional[int]) -> None:
        """M4: typed pool_saturated event (app/main.py:3348-3353)."""
        payload = {"type": SERVER_ERROR, "error": "pool_saturated", "status": 4429}
        if max_slots is not None:
            payload["max_slots"] = max_slots
        await self._send_event(payload)

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
        if self._tts.buffer is None:
            return
        if self.engine.tool_registry is not None:
            # #2: run the tool loop as a cancellable task so a concurrent
            # barge-in (_audio_loop's VAD / CLIENT_ABORT → _bargein_tts) can
            # stop an in-flight turn. Reset the cooperative flag and create the
            # task in one synchronous step (no await between) so a barge-in
            # arriving right after create_task isn't lost to the reset.
            self.state.llm_barged = False
            task = asyncio.create_task(
                self._llm.run([{"role": "user", "content": text}])
            )
            self.state.current_llm_task = task
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
                if self.state.current_llm_task is task:
                    self.state.current_llm_task = None
            return
        messages = [{"role": "user", "content": text}]
        try:
            async for ev in self._llm_be.stream_events(messages):
                if ev.kind == "text" and ev.text:
                    await self._tts.enqueue_text(ev.text)
                # tool_call_delta is only reachable via the tool_registry path
                # above; with no registry the engine never sends a tools schema
                # so the backend never emits tool calls.
            await self._tts.flush_and_signal()
        except Exception:
            logger.exception("voxedge LLM stream failed")


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
            asyncio.create_task(self._audio.run()),
            asyncio.create_task(self._events.run()),
        ]
        work_tasks: list[asyncio.Task] = []
        if self.asr_enabled:
            work_tasks.append(asyncio.create_task(self._asr.run()))
        if self._tts_be is not None:
            work_tasks.append(asyncio.create_task(self._tts.run()))
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
            self.state.close_input()
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
            self.state.client_closed = True
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
            stop = self.state.current_tts_stop
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
