"""Per-utterance ASR session manager.

COPIED FROM app/core/asr_session_manager.py (2026-05-30). Dedup after Phase 1b
(app/main.py still imports the original; once the v2v handler is migrated onto
voxedge this copy becomes the single source of truth and the app/core module
can re-export it). The original is stdlib-only with no env/profile reads, so
this is a verbatim port — the only intentional difference is the constructor
``sample_rate`` injection (M2: the production code hardcoded 16000 in
``accept_audio``; here it is passed in so voxedge stays env-free and works with
any backend sample rate).

Owns the lifecycle of streaming ASR sessions for a single connection:
fresh ``ASRStream`` per utterance, generation tokens guarding against stale
finals, bounded cancellation with worker-restart fallback, and ERROR_REBUILD
recovery on worker protocol errors.

State machine
-------------

    IDLE ──speech_start──► ACTIVE ──speech_end / asr_eos──► FINALIZING ──ack──► IDLE
                              │                                  │
                              └────────── cancel ────────────────┴─► CANCELLING ──► IDLE
                                                                          │
                                                                          ▼
                                                              (waits ≤500ms for end-ack;
                                                               on timeout calls restart_worker())

    Any ──worker error──► ERROR_REBUILD ──(retry ≤3 / backoff 50,150,400ms)──► IDLE
                                       └─ exhausted ──► restart_worker() ──► IDLE

Each transition into ``ACTIVE`` issues a fresh ``generation_id``; finals tagged
with a stale generation are silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ASRSessionUnavailable(RuntimeError):
    """Raised when on_speech_start cannot produce a working ASR stream.

    Signals to the caller that the ASR worker is unrecoverable for this
    turn — caller MUST NOT flag the session as active (race #1: silent
    no-op accept_audio loop with client stuck THINKING).
    """


class SessionState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    FINALIZING = "finalizing"
    CANCELLING = "cancelling"
    ERROR_REBUILD = "error_rebuild"


# Worker-protocol error types (mirrored on trt_edge_llm_asr backend);
# duck-typed via class name so tests / non-jetson backends don't need to
# import the jetson module.
_WORKER_ERROR_NAMES = {
    "NoActiveSessionError",
    "SessionAlreadyActiveError",
    "WorkerExitError",
    "WorkerProtocolError",
}


def _is_worker_protocol_error(exc: BaseException) -> bool:
    if exc is None:
        return False
    for cls in type(exc).__mro__:
        if cls.__name__ in _WORKER_ERROR_NAMES:
            return True
    return False


def _safe_close_stream(stream: Any) -> None:
    """Release per-stream backend resources (TRT contexts, device buffers).

    Default ASRStream.close() is a no-op; backends like paraformer_trt
    override it to drop per-stream TRT IExecutionContext + cudaMalloc'd
    buffers. We swallow any exception so close() never breaks lifecycle
    teardown.
    """
    if stream is None:
        return
    close = getattr(stream, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        logger.exception("ASRSessionManager: stream.close raised; ignoring")


class ASRSessionManager:
    """Async-safe per-utterance ASR session orchestrator.

    Backends are synchronous; all calls into them are hopped through
    ``loop.run_in_executor`` to avoid blocking the event loop. A single
    instance-level ``asyncio.Lock`` serializes state transitions.

    Worker-op serialization (F1): the underlying ASR worker is single-
    concurrency (one C++ IPC at a time). Every worker operation — create_stream,
    accept_waveform, finalize, get_partial, cancel — therefore runs while
    holding ``self._lock``, so no two can hit the worker concurrently from the
    three driver tasks (audio loop / asr-out / event loop). The introspection
    properties (``state`` / ``current_generation`` / ``stream``) read their
    fields WITHOUT the lock, so callers polling state stay responsive while a
    worker op holds it. INVARIANT: any ``except``/error path inside a held-lock
    worker op MUST call ``_handle_error_locked`` DIRECTLY (never ``async with
    self._lock`` again — asyncio.Lock is not re-entrant → instant deadlock).
    """

    # Retry/backoff schedule for ERROR_REBUILD (≤3 attempts before
    # falling back to a full worker restart).
    _REBUILD_BACKOFF_S = (0.05, 0.15, 0.40)
    _CANCEL_ACK_TIMEOUT_S = 0.5

    def __init__(
        self,
        backend: Any,
        language: str = "auto",
        coord: Any = None,
        *,
        sample_rate: int = 16000,
        executor: Any = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._backend = backend
        self._language = language
        self._coord = coord  # BackendCoordinator (optional)
        # M2: sample_rate injected (prod hardcoded 16000 in accept_audio,
        # app/core/asr_session_manager.py:235). Falls back to the backend's
        # own sample_rate if it exposes one.
        self._sample_rate = int(getattr(backend, "sample_rate", sample_rate) or sample_rate)
        self._executor = executor  # asr executor (optional)
        self._loop = loop  # late-bound if None
        self._lock = asyncio.Lock()
        self._state: SessionState = SessionState.IDLE
        self._stream: Any = None
        self._generation: int = 0
        # Lock-free preemption signal. ``cancel``/``on_speech_start`` bump this
        # BEFORE contending for ``_lock`` so an in-flight ``finalize`` (which,
        # per F1, holds ``_lock`` across its whole worker op) can detect that it
        # was preempted and discard its now-stale result instead of committing a
        # barge-in/old-utterance final. Mutated only from the event loop thread.
        self._abort_epoch: int = 0
        self._last_error: Optional[BaseException] = None
        self._recovery_in_progress: bool = False
        self._recovery_future: Optional[asyncio.Future] = None

    # ── public introspection ───────────────────────────────────────────
    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def current_generation(self) -> int:
        return self._generation

    @property
    def stream(self) -> Any:
        return self._stream

    # ── helpers ────────────────────────────────────────────────────────
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        return asyncio.get_event_loop()

    async def _run_sync(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        loop = self._get_loop()
        if kwargs:
            def _bound():
                return fn(*args, **kwargs)
            return await loop.run_in_executor(self._executor, _bound)
        return await loop.run_in_executor(self._executor, fn, *args)

    def _new_stream_sync(self) -> Any:
        return self._backend.create_stream(language=self._language)

    async def _create_stream(self) -> Any:
        return await self._run_sync(self._new_stream_sync)

    # ── public API ─────────────────────────────────────────────────────
    async def on_speech_start(self) -> int:
        """Transition IDLE→ACTIVE (cancelling any prior session first).

        Returns the new generation id. Stale finals from a previous
        generation must be ignored by the caller.
        """
        # Preempt any in-flight finalize before blocking on the lock (see
        # ``_abort_epoch``): U2 starting while U1 is finalizing must discard U1.
        self._abort_epoch += 1
        async with self._lock:
            if self._state in (SessionState.ACTIVE, SessionState.FINALIZING):
                await self._inner_cancel(reason="speech_start_preempt")
            elif self._state == SessionState.CANCELLING:
                pass

            self._generation += 1
            try:
                self._stream = await self._create_stream()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ASRSessionManager: create_stream failed: %s", exc)
                await self._handle_error_locked(exc)
                if self._stream is None:
                    self._state = SessionState.IDLE
                    raise ASRSessionUnavailable(
                        "ASR worker unavailable after rebuild"
                    ) from exc
            self._state = SessionState.ACTIVE
            return self._generation

    async def accept_audio(self, samples) -> None:
        """Push a chunk of audio at the current stream.

        No-op outside ACTIVE. Failures route to ERROR_REBUILD.
        """
        # F1: hold _lock across accept_waveform so it can't run concurrently
        # with finalize/get_partial/cancel/create on the single ASR worker.
        async with self._lock:
            if self._state != SessionState.ACTIVE:
                return
            stream = self._stream
            if stream is None:
                try:
                    stream = await self._create_stream()
                    self._stream = stream
                except Exception as exc:  # noqa: BLE001
                    await self._handle_error_locked(exc)
                    return
            try:
                await self._run_sync(stream.accept_waveform, self._sample_rate, samples)
            except Exception as exc:  # noqa: BLE001
                # Lock already held — call directly (re-acquiring would deadlock).
                await self._handle_error_locked(exc)

    async def finalize(self, reason: str = "vad_end") -> str:
        """Transition ACTIVE→FINALIZING→IDLE; return final text."""
        _gen, text = await self.finalize_with_generation(reason)
        return text

    async def finalize_with_generation(self, reason: str = "vad_end") -> tuple[int, str]:
        """Like :meth:`finalize` but returns ``(generation_id, text)``."""
        gen, text, _accepted, _lang = await self.finalize_with_status(reason)
        return gen, text

    async def finalize_with_status(
        self, reason: str = "vad_end"
    ) -> tuple[int, str, bool, Optional[str]]:
        """Returns ``(generation, text, accepted, detected_language)``.

        ``accepted`` is False when the manager discarded the finalize
        result because the stream was cancelled, no longer finalizable, or
        superseded by another generation.
        """
        # F1: hold _lock across the whole finalize (transition → worker op →
        # commit) so it can't overlap accept/get_partial/cancel on the worker.
        async with self._lock:
            if self._state not in (SessionState.ACTIVE,):
                return self._generation, "", False, None
            gen = self._generation
            self._state = SessionState.FINALIZING
            # Snapshot the preemption signal before the (lock-held) worker op.
            abort_epoch0 = self._abort_epoch
            stream = self._stream
            if stream is None:
                self._state = SessionState.IDLE
                return gen, "", False, None
            try:
                raw = await self._run_sync(stream.finalize)
            except Exception as exc:  # noqa: BLE001
                await self._handle_error_locked(exc)  # lock held — call directly
                return gen, "", False, None
            # Backends MUST return ``(text, language)`` per the ASRStream ABC.
            final_text, detected_language = raw
            # A cancel()/on_speech_start() that arrived while we held the lock
            # bumped _abort_epoch (lock-free, before it blocked on the lock).
            # Discard the now-stale result so a barge-in / preempting utterance
            # does not leak the old final downstream — the caller suppresses on
            # accepted=False. (F1 holds the lock across the whole finalize, so
            # this lock-free epoch is the only signal that can reach us here.)
            if self._abort_epoch != abort_epoch0:
                logger.info(
                    "ASRSessionManager: finalize result discarded (preempted mid-flight)"
                )
                return gen, "", False, None
            # Defensive invariants (kept; normally unreachable under the lock).
            if self._state != SessionState.FINALIZING:
                logger.info("ASRSessionManager: finalize result discarded (state=%s)", self._state)
                return gen, "", False, None
            if self._generation != gen:
                logger.info(
                    "ASRSessionManager: finalize result discarded (stale gen %d != current %d)",
                    gen, self._generation,
                )
                return gen, "", False, None
            _safe_close_stream(self._stream)
            self._stream = None
            self._state = SessionState.IDLE
            return gen, final_text or "", True, detected_language

    async def prepare_finalize_for_generation(
        self, generation: int | None = None
    ) -> tuple[int, bool]:
        """Precompute final ASR work for the active generation when supported.

        Dialogue clients often know an utterance is about to end before they
        commit EOS. This method lets callers hide backend ``prepare_finalize``
        work under that EOU lead time without changing the authoritative
        ``finalize_with_status`` lifecycle. It returns ``(generation,
        prepared)``; ``prepared`` is false when the requested generation is no
        longer active or the stream has no prepare hook.
        """
        async with self._lock:
            gen = self._generation
            if generation is not None and generation != gen:
                return gen, False
            if self._state != SessionState.ACTIVE or self._stream is None:
                return gen, False
            stream = self._stream
            prepare = getattr(stream, "prepare_finalize", None)
            if prepare is None:
                return gen, False
            try:
                await self._run_sync(prepare)
            except Exception as exc:  # noqa: BLE001
                await self._handle_error_locked(exc)
                return gen, False
            return gen, bool(
                self._generation == gen
                and self._state == SessionState.ACTIVE
                and self._stream is stream
            )

    async def get_partial_for_generation(self) -> tuple[int, str, bool]:
        """Snapshot ``(generation, partial_text, is_endpoint)`` atomically.

        Returns ``(generation, "", False)`` if there's no active stream.
        """
        # F1: hold _lock across get_partial so it can't overlap accept/finalize/
        # cancel on the single worker.
        async with self._lock:
            gen = self._generation
            stream = self._stream
            if stream is None or self._state != SessionState.ACTIVE:
                return gen, "", False
            try:
                partial, is_endpoint = await self._run_sync(stream.get_partial)
            except Exception:  # noqa: BLE001
                return gen, "", False
            return gen, partial or "", bool(is_endpoint)

    async def cancel(self, reason: str = "bargein") -> None:
        # Flag the in-flight finalize (if any) before contending for the lock so
        # a barge-in discards rather than commits its result (see _abort_epoch).
        self._abort_epoch += 1
        async with self._lock:
            await self._inner_cancel(reason=reason)

    async def _inner_cancel(self, *, reason: str) -> None:
        """Lock must be held by caller."""
        if self._state in (SessionState.IDLE,):
            return
        prev_state = self._state
        self._state = SessionState.CANCELLING
        stream = self._stream
        self._stream = None
        if stream is None:
            self._state = SessionState.IDLE
            return

        def _cancel_call():
            if hasattr(stream, "cancel"):
                stream.cancel()
            else:
                stream.cancel_and_finalize()

        loop = self._get_loop()
        fut = loop.run_in_executor(self._executor, _cancel_call)
        # P3: only close the stream if the cancel executor call actually
        # FINISHED (returned or raised). On timeout the worker thread may still
        # be inside _cancel_call on this C++ stream object; a synchronous
        # stream.close() from the coroutine side would then race that thread
        # (the asyncio lock doesn't extend to executor threads). Leave it to
        # restart_worker to reclaim the worker-side resources instead.
        thread_done = False
        try:
            await asyncio.wait_for(fut, timeout=self._CANCEL_ACK_TIMEOUT_S)
            thread_done = True
        except asyncio.TimeoutError:
            logger.warning(
                "ASRSessionManager: cancel(%s) timed out from state=%s; restarting "
                "worker (NOT closing stream — _cancel_call thread may still hold it)",
                reason, prev_state,
            )
            await self._maybe_restart_worker()
        except Exception as exc:  # noqa: BLE001
            thread_done = True  # future raised → the executor call has returned
            if _is_worker_protocol_error(exc):
                logger.warning(
                    "ASRSessionManager: cancel(%s) raised worker error %s; restarting",
                    reason, type(exc).__name__,
                )
                await self._maybe_restart_worker()
            else:
                logger.info("ASRSessionManager: cancel(%s) swallowed exc=%s", reason, exc)
        if thread_done:
            _safe_close_stream(stream)
        self._state = SessionState.IDLE

    def mark_error(self, exc: BaseException) -> None:
        """Synchronous shim so accept_waveform threads / partial pollers
        can flag the manager. Defers to the next async tick."""
        self._last_error = exc
        try:
            loop = self._get_loop()
        except Exception:
            return
        if loop.is_running():
            asyncio.ensure_future(self._async_mark_error(exc), loop=loop)

    async def _async_mark_error(self, exc: BaseException) -> None:
        fut: Optional[asyncio.Future] = None
        own_recovery = False
        if self._recovery_future is not None and not self._recovery_future.done():
            fut = self._recovery_future
        else:
            loop = self._get_loop()
            self._recovery_future = loop.create_future()
            own_recovery = True
            fut = self._recovery_future

        if not own_recovery:
            try:
                await fut
            except Exception:
                pass
            return

        try:
            async with self._lock:
                await self._handle_error_locked(exc)
        finally:
            if not fut.done():
                fut.set_result(None)
            self._recovery_future = None

    async def _handle_error_locked(self, exc: BaseException) -> None:
        self._last_error = exc
        if self._recovery_in_progress:
            return
        self._recovery_in_progress = True
        _safe_close_stream(self._stream)
        self._stream = None
        self._state = SessionState.ERROR_REBUILD
        if not _is_worker_protocol_error(exc):
            logger.info("ASRSessionManager: non-protocol error during ASR: %s", exc)
        try:
            await self._do_rebuild_locked()
        finally:
            self._recovery_in_progress = False

    async def _do_rebuild_locked(self) -> None:
        for attempt, delay in enumerate(self._REBUILD_BACKOFF_S):
            await asyncio.sleep(delay)
            try:
                self._stream = await self._create_stream()
                self._state = SessionState.ACTIVE
                logger.info(
                    "ASRSessionManager: ERROR_REBUILD recovered on attempt %d",
                    attempt + 1,
                )
                return
            except Exception as inner:
                logger.warning(
                    "ASRSessionManager: ERROR_REBUILD attempt %d failed: %s",
                    attempt + 1, inner,
                )
                self._last_error = inner
        await self._maybe_restart_worker()
        try:
            self._stream = await self._create_stream()
            self._state = SessionState.ACTIVE
        except Exception as inner:
            logger.warning("ASRSessionManager: post-restart create_stream failed: %s", inner)
            _safe_close_stream(self._stream)
            self._stream = None
            self._state = SessionState.IDLE

    async def _maybe_restart_worker(self) -> None:
        backend = self._backend
        fn = getattr(backend, "restart_worker", None)
        if fn is None:
            return
        # IMPORTANT: do NOT submit to ``self._executor`` (the single-thread
        # ASR slot that may be wedged). The default executor (None) is a
        # multi-thread pool and is always free.
        loop = self._get_loop()
        try:
            await loop.run_in_executor(None, fn)
            logger.info("ASRSessionManager: backend.restart_worker() completed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ASRSessionManager: restart_worker failed: %s", exc)


__all__ = ["ASRSessionManager", "ASRSessionUnavailable", "SessionState"]
