"""Generic subprocess-worker IO multiplexer — voxedge adapter.

adapted from app/core/worker_io.py (2026-05-30), dedup after registry switch.

This is the framework-layer abstraction that demuxes a single JSON-line
subprocess (one stdin / one stdout) into N in-flight per-request streams,
keyed by ``request_id`` / ``id``. Used by the TRT-Edge-LLM ASR/TTS adapters.

The body is byte-equivalent to the production copy: it has ZERO env reads,
ZERO ``app.*`` imports, and only depends on the stdlib. It is reproduced here
(not imported) so the voxedge trt package stays free of any ``app`` import.

Public API:

    wio = WorkerIO(proc, concurrency)

    # Async (preferred for new code)
    async for event in wio.send_request(rid, payload):
        ...
    wio.cancel(rid)
    wio.close()

    # Sync (legacy shim retained for the TTS path that runs inside a
    # ThreadPoolExecutor / generator-of-PCM-chunks pipeline).
    for event in wio.request(payload):
        ...

Both APIs share the same underlying ``_inflight`` map, ``_stdin_lock``,
reader thread, and semaphore, so they coexist safely on the same instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import subprocess
import threading
import time
from typing import AsyncIterator, Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class WorkerExitError(RuntimeError):
    """Raised when the worker subprocess dies while a request is in flight."""


class WorkerIO:
    """Per-worker stdin writer + stdout reader thread, multiplexing N in-flight requests.

    Replaces a coarse per-request lock (which would serialize full
    request→response cycles end-to-end) with:

      * a single ``_stdin_lock`` that only protects the single-line JSON write,
      * a daemon reader thread that demuxes stdout events to per-request
        ``queue.Queue`` instances keyed by ``request_id``/``id``,
      * a ``threading.Semaphore`` bounding in-flight requests to ``concurrency``.

    When the worker subprocess EOFs (crash / restart), the reader thread wakes
    every in-flight caller with a sentinel ``{"event": "_worker_exit"}`` so
    they raise ``WorkerExitError`` instead of hanging on ``q.get(timeout=...)``.

    A given ``WorkerIO`` instance is bound to ONE subprocess. To restart the
    worker, discard the old instance and create a new one (handled by the
    owning backend, e.g. ``_ensure_worker`` / ``_restart_worker``).
    """

    # Class-level temporary instrumentation: counts every cancel() invocation
    # across all WorkerIO instances since process start.
    _cancel_count: int = 0
    _cancel_count_lock = threading.Lock()

    def __init__(
        self,
        proc: subprocess.Popen,
        concurrency: int,
        *,
        telemetry_callback: Optional[Callable[[dict], None]] = None,
    ):
        self._proc = proc
        # Optional sink for id-less worker telemetry (``asr_ifb_health`` /
        # ``asr_ifb_trace``). Default None keeps the TTS/control API unchanged.
        # Telemetry is NEVER routed into request queues or semaphore accounting,
        # even when a telemetry line carries an external or internal id.
        self._telemetry_callback = telemetry_callback
        self._stdin_lock = threading.Lock()
        self._inflight: dict[str, "queue.Queue"] = {}
        self._inflight_lock = threading.Lock()
        # Keep the configured concurrency bound for the opt-in cancel-receipt
        # admission cap (see ``cancel_and_wait``). Never used for request
        # semaphore accounting.
        self._concurrency = max(1, int(concurrency))
        # Opt-in per-id cancellation receipt queues (native ASR only). A
        # cancelled terminal for a known id is emitted immediately by the
        # native worker, but an IDLE session has no inflight queue, so the
        # reader would drop it as stale. A registered receipt queue gives the
        # reader a second fan-out destination WITHOUT replacing the inflight
        # queue, the request semaphore, or synthesizing any event.
        self._cancel_receipts: dict[str, "queue.Queue"] = {}
        # Number of currently active cancel_and_wait waiters; bounded by
        # ``_concurrency`` before any stdin write happens.
        self._cancel_waiter_count = 0
        self._sem = threading.Semaphore(max(1, int(concurrency)))
        # Set by close(); requests that acquire the semaphore after this
        # is True must abort instead of writing to a dead worker stdin.
        self._closed = False
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="worker-io-stdout",
            daemon=True,
        )
        self._reader_thread.start()

    async def send_request(
        self, request_id: str, payload: dict
    ) -> AsyncIterator[dict]:
        """Async-iterate worker events for one request.

        The semaphore + per-request queue + sentinel-on-worker-exit semantics
        match ``request()`` exactly; the only difference is awaitable
        ``q.get`` (offloaded to the default thread executor) so callers
        integrate into an asyncio event loop without blocking it.

        If the consumer breaks out of the ``async for`` (or ``aclose()`` is
        invoked), the ``finally`` arm fires ``cancel(request_id)`` so the
        worker emits a terminal ``cancelled`` event.
        """
        # Ensure payload carries the request_id the caller passed in.
        payload = dict(payload)
        payload.setdefault("id", request_id)
        if payload.get("id") != request_id:
            raise ValueError(
                f"payload['id']={payload.get('id')!r} != request_id={request_id!r}"
            )

        loop = asyncio.get_running_loop()
        # Run the blocking semaphore acquire off the loop. The semaphore is
        # a back-pressure gate; it can block when concurrency is saturated.
        # If the awaiting task is cancelled while the executor thread is
        # still blocked in self._sem.acquire(), the thread cannot be
        # interrupted — it may eventually grab the token after we're gone.
        # Attach a done-callback so the late-acquired token is released
        # back to the pool instead of leaking the slot.
        _fut = loop.run_in_executor(None, self._sem.acquire)
        try:
            await _fut
        except BaseException:
            def _release_if_late_acquire(f: "asyncio.Future") -> None:
                try:
                    if f.result() is True:
                        self._sem.release()
                except BaseException:
                    pass
            _fut.add_done_callback(_release_if_late_acquire)
            raise

        # Hold semaphore from here. If close() ran while we were waiting,
        # abort immediately and release the slot back.
        if self._closed:
            self._sem.release()
            raise WorkerExitError("WorkerIO closed before request could start")

        q: "queue.Queue" = queue.Queue()
        with self._inflight_lock:
            self._inflight[request_id] = q

        finished_naturally = False
        try:
            assert self._proc.stdin is not None
            try:
                with self._stdin_lock:
                    # Re-check under the same lock that close() uses to set
                    # the flag; this closes the TOCTOU window.
                    if self._closed:
                        raise WorkerExitError(
                            "WorkerIO closed between acquire and stdin write"
                        )
                    self._proc.stdin.write(
                        json.dumps(payload, ensure_ascii=False) + "\n"
                    )
                    self._proc.stdin.flush()
            except Exception:
                with self._inflight_lock:
                    self._inflight.pop(request_id, None)
                raise

            while True:
                try:
                    event = await loop.run_in_executor(None, q.get, True, 60.0)
                except queue.Empty:
                    raise TimeoutError(
                        f"WorkerIO.send_request: no event for {request_id} in 60s"
                    )
                if event.get("event") == "_worker_exit":
                    raise WorkerExitError("worker subprocess died mid-request")
                yield event
                if event.get("event") in ("done", "cancelled"):
                    finished_naturally = True
                    return
        finally:
            with self._inflight_lock:
                self._inflight.pop(request_id, None)
            self._sem.release()
            if not finished_naturally:
                try:
                    self.cancel(request_id)
                except Exception:
                    logger.debug(
                        "cancel() during async cleanup failed", exc_info=True
                    )

    def close(self) -> None:
        """Tear down: wake every in-flight caller and stop the reader.

        After ``close()``, in-flight ``send_request``/``request`` generators
        observe a ``_worker_exit`` sentinel and surface ``WorkerExitError``.
        Sets a closed flag so that any request currently blocked in
        ``self._sem.acquire()`` will abort on wake-up instead of writing to
        a dead worker's stdin. ``close()`` does not kill the subprocess —
        that is the owning backend's responsibility.
        """
        with self._stdin_lock:
            self._closed = True
        with self._inflight_lock:
            queues = list(self._inflight.values())
            self._inflight.clear()
            receipt_queues = list(self._cancel_receipts.values())
            self._cancel_receipts.clear()
        for q in queues:
            q.put({"event": "_worker_exit"})
        for rq in receipt_queues:
            # Non-blocking: a receipt queue is bounded (maxsize=1) and may
            # already hold a real terminal. Waking it must NEVER block
            # close(); dropping the wake on a full queue is safe because the
            # retained terminal is the one the waiter must observe first.
            self._offer_receipt(rq, {"event": "_worker_exit"})

    def request(
        self,
        payload: dict,
        *,
        cancel_event: threading.Event | None = None,
        on_written: "Callable[[], None] | None" = None,
    ) -> Iterator[dict]:
        """Send ``payload`` to the worker and yield response events until ``done``.

        Caller must include a unique ``id`` field in ``payload``. The generator
        terminates when an ``event=="done"`` or ``event=="cancelled"`` is
        received, or raises ``WorkerExitError`` if the worker dies mid-request.

        ``on_written`` is an OPTIONAL, default-off callback invoked exactly once
        AFTER the payload line has been written AND flushed to the worker stdin,
        and (preferably) after the stdin lock has been released. It is the ONLY
        sound "begin is now on the wire" barrier: receivers that must not write
        a dependent line for an unknown native id (e.g. a native cancel that a
        worker silently drops for an unknown SID) wait on it. On any failure
        that prevents an actual write+flush (closed worker, stdin-lock timeout,
        broken pipe) the callback is NEVER invoked, so the barrier stays
        unset and the dependent operation fails closed. A callback exception is
        NOT swallowed (the write already succeeded, so the caller must know);
        keep the callback non-blocking and non-raising.

        ``cancel_event`` provides out-of-band cancellation for a generator
        currently blocked inside ``q.get``. This is required by synchronous
        TTS generators running in an executor: calling ``generator.close()``
        from the ASGI event-loop thread cannot interrupt a generator that is
        already executing in another thread.
        """
        if cancel_event is None:
            self._sem.acquire()
        else:
            # A prefetched sentence can already be waiting for a Python
            # WorkerIO slot when its HTTP client disconnects. Do not let that
            # stale task acquire a later slot and write a brand-new worker
            # request after cancellation.
            while not self._sem.acquire(timeout=0.1):
                if cancel_event.is_set():
                    return
        req_id: str | None = None
        cancel_sent = False
        last_event_at = time.monotonic()
        try:
            if self._closed:
                raise WorkerExitError("WorkerIO closed before request could start")
            # Close the race between the successful semaphore acquire and the
            # first stdin write.
            if cancel_event is not None and cancel_event.is_set():
                return
            req_id = payload["id"]
            q: "queue.Queue" = queue.Queue()
            # CRITICAL ordering: insert the queue BEFORE writing stdin so
            # the reader thread can never observe an event for ``req_id``
            # before the queue exists (would otherwise be dropped as "stale").
            with self._inflight_lock:
                self._inflight[req_id] = q
            assert self._proc.stdin is not None
            with self._stdin_lock:
                if self._closed:
                    raise WorkerExitError(
                        "WorkerIO closed between acquire and stdin write"
                    )
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            if on_written is not None:
                # Fired AFTER the actual write+flush and after the stdin lock is
                # released, so a waiter woken by this barrier may itself write
                # to stdin without deadlocking on the lock we just dropped.
                on_written()
            while True:
                if (
                    cancel_event is not None
                    and cancel_event.is_set()
                    and not cancel_sent
                ):
                    self.cancel(req_id)
                    cancel_sent = True
                try:
                    event = q.get(timeout=0.1 if cancel_event is not None else 60.0)
                except queue.Empty:
                    if time.monotonic() - last_event_at >= 60.0:
                        raise
                    continue
                last_event_at = time.monotonic()
                if event.get("event") == "_worker_exit":
                    raise WorkerExitError("worker subprocess died mid-request")
                # A cancel can race the worker thread's per-request
                # registration. Retry until it reports tripped=true or a
                # terminal event arrives.
                if (
                    event.get("event") == "cancel_ack"
                    and event.get("tripped") is False
                    and cancel_event is not None
                    and cancel_event.is_set()
                ):
                    cancel_sent = False
                yield event
                if event.get("event") in ("done", "cancelled"):
                    return
        finally:
            if req_id is not None:
                with self._inflight_lock:
                    self._inflight.pop(req_id, None)
            self._sem.release()

    def cancel_and_wait(self, req_id: str, timeout_s: float) -> dict:
        """Send a cancel and WAIT for the native ``cancelled`` terminal event.

        Intended ONLY for the native ASR worker, whose cancelled terminal for
        a known id is
        ``{"event": "cancelled", "id": <sid>, "ok": false, "epoch": <n>}``
        emitted immediately (unknown ids are silently dropped by the worker).
        Returns the ACTUAL matching cancelled event dict as received — ``ok``
        being false is the EXPECTED native shape, not an error. No
        ``cancel_ack`` / epoch assumptions are made (legacy protocols do not
        emit them here).

        Admission (before ANY stdin write):
          * ``req_id`` must be a non-empty str, ``timeout_s`` a positive,
            finite, non-bool number — otherwise ValueError.
          * at most ONE active waiter per id and at most ``concurrency``
            waiters total — otherwise RuntimeError.

        Semantics:
          * A bounded per-id receipt queue is registered BEFORE the cancel
            line is written; the reader thread fans a matching cancelled
            terminal to BOTH the existing inflight queue (if any) AND this
            receipt queue. A concurrent ``request()``/``send_request()`
            consumer keeps its own queue and semaphore — nothing is replaced
            and no synthetic terminal is generated.
          * Unsolicited or mismatched terminals never reach the receipt queue
            and cannot satisfy the wait.
          * One absolute ``time.monotonic()`` deadline: validated before
            registration/write, before the stdin write, after the write, and
            before returning the receipt. The stdin lock is acquired with the
            actual remaining budget; the queue wait uses the same absolute
            deadline with NO renewal and NO floor. Late arrival fails.
          * Worker exit / close() wakes pending receipts with
            ``WorkerExitError`` — a cancelled receipt is never fabricated.
          * The waiter is always unregistered in ``finally``.

        KNOWN LIMITATION (deliberate, do not claim more): the whole-line
        ``TextIO.write`` + ``flush`` under ``_stdin_lock`` may block for an
        opaque duration inside the OS/pipe layer. This method therefore does
        NOT guarantee hard termination of the entire call within
        ``timeout_s``; callers that need loop liveness must run it off the
        event loop with a supervisor and retain live writers on timeout.
        """
        if not isinstance(req_id, str) or not req_id:
            raise ValueError(
                f"cancel_and_wait: invalid req_id={req_id!r}"
            )
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError(
                f"cancel_and_wait: invalid timeout_s={timeout_s!r}"
            )

        deadline = time.monotonic() + float(timeout_s)

        # --- Admission: register receipt queue before writing anything. ---
        with self._inflight_lock:
            if self._closed:
                raise WorkerExitError("WorkerIO closed before cancel_and_wait")
            if req_id in self._cancel_receipts:
                raise RuntimeError(
                    f"cancel_and_wait: duplicate active waiter for id={req_id!r}"
                )
            if self._cancel_waiter_count >= self._concurrency:
                raise RuntimeError(
                    "cancel_and_wait: waiter capacity exceeded "
                    f"({self._cancel_waiter_count} >= {self._concurrency})"
                )
            receipt_q: "queue.Queue" = queue.Queue(maxsize=1)
            self._cancel_receipts[req_id] = receipt_q
            self._cancel_waiter_count += 1

        try:
            # --- Write the cancel line under the absolute deadline. ---
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"cancel_and_wait: deadline expired before write for {req_id!r}"
                )
            try:
                assert self._proc.stdin is not None
                if not self._stdin_lock.acquire(timeout=remaining):
                    raise TimeoutError(
                        f"cancel_and_wait: stdin lock not acquired in budget for {req_id!r}"
                    )
                try:
                    if self._closed:
                        raise WorkerExitError(
                            "WorkerIO closed before cancel could be written"
                        )
                    # The lock acquire may itself have consumed the entire
                    # remaining budget (it returns as soon as the lock is
                    # free, but the wall clock is not ours to control).
                    # Re-check the SAME absolute deadline here, inside the
                    # lock and immediately BEFORE the write, so a late
                    # acquisition fails closed with ZERO bytes written.
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "cancel_and_wait: deadline expired after acquiring "
                            f"stdin lock for {req_id!r}"
                        )
                    # NOTE: this write+flush may block for an opaque duration
                    # (see docstring); the deadline is re-checked AFTER it.
                    self._proc.stdin.write(
                        json.dumps({"type": "cancel", "id": req_id}) + "\n"
                    )
                    self._proc.stdin.flush()
                finally:
                    self._stdin_lock.release()
            except WorkerExitError:
                raise
            except TimeoutError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"cancel_and_wait: cancel write failed for {req_id!r}"
                ) from exc

            # Deadline is fresh after the write; no renewal, no floors.
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"cancel_and_wait: deadline expired after write for {req_id!r}"
                )

            # --- Wait for the receipt on the same absolute deadline. ---
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"cancel_and_wait: no cancelled receipt for {req_id!r} "
                        "within budget"
                    )
                try:
                    event = receipt_q.get(timeout=remaining)
                except queue.Empty:
                    raise TimeoutError(
                        f"cancel_and_wait: no cancelled receipt for {req_id!r} "
                        "within budget"
                    )
                if not isinstance(event, dict):
                    continue
                if event.get("event") == "_worker_exit":
                    raise WorkerExitError(
                        "worker subprocess died while awaiting cancel receipt"
                    )
                # Defensive re-validation: only an exact matching cancelled
                # terminal satisfies the receipt (ok:false is the EXPECTED
                # native shape, not an error).
                if (
                    event.get("event") == "cancelled"
                    and event.get("id") == req_id
                ):
                    # Fresh check immediately before returning the receipt.
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"cancel_and_wait: receipt for {req_id!r} arrived late"
                        )
                    return event
                # Anything else (defensively enqueued) is ignored.
        finally:
            with self._inflight_lock:
                self._cancel_receipts.pop(req_id, None)
                self._cancel_waiter_count -= 1

    def cancel(self, req_id: str) -> None:
        """Best-effort cancel for an in-flight request.

        Writes a cancel JSON to the worker's stdin. The worker will check
        its per-request atomic flag at the next chunk boundary and emit
        a ``{"event":"cancelled", ...}`` terminal event in lieu of ``done``.

        Safe to call from any thread. Safe to call after the request has
        naturally completed (worker silently drops unknown cancels).
        """
        with WorkerIO._cancel_count_lock:
            WorkerIO._cancel_count += 1
            count_snapshot = WorkerIO._cancel_count
        logger.info(
            "WorkerIO.cancel: req_id=%s total_cancel_count=%d",
            req_id,
            count_snapshot,
        )
        try:
            assert self._proc.stdin is not None
            with self._stdin_lock:
                if self._closed:
                    return
                self._proc.stdin.write(
                    json.dumps({"type": "cancel", "id": req_id}) + "\n"
                )
                self._proc.stdin.flush()
        except Exception:
            logger.debug(
                "cancel() write failed; worker may be exiting",
                exc_info=True,
            )

    @staticmethod
    def _offer_receipt(rq: "queue.Queue", event: dict) -> None:
        """Non-blocking put onto a bounded cancel-receipt queue.

        Receipt queues are ``queue.Queue(maxsize=1)``: they exist to retain
        the FIRST actual matching cancelled terminal for a waiting
        ``cancel_and_wait`` caller. The reader thread, ``close()`` and the
        EOF finalizer must NEVER block on them, so every receipt fan-out goes
        through this helper. When the queue is already full the offer is
        dropped (the retained first terminal is the one the waiter must
        observe).
        """
        try:
            rq.put_nowait(event)
        except queue.Full:
            pass

    def _reader_loop(self) -> None:
        """Drain worker stdout, dispatching events to per-request queues."""
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    logger.debug("worker emitted non-JSON line: %r", line[:200])
                    continue
                if not isinstance(event, dict):
                    # Malformed JSON primitives (ints, strings, lists, null) are
                    # logged and skipped — the reader must NOT crash on them.
                    logger.debug(
                        "worker emitted non-object JSON line: %r", line[:200]
                    )
                    continue
                # Health/trace lines are telemetry even when they carry an
                # external_id / internal id: they must never land in a request
                # inflight queue or touch semaphore accounting. The NATIVE
                # worker emits these with the canonical ``type`` key (e.g.
                # qwen3_asr_worker lifecycle health_final:
                # {"type":"asr_ifb_health",...}); ``event`` is accepted only
                # as a source-compat alias.
                kind = event.get("type") or event.get("event")
                if kind in ("asr_ifb_health", "asr_ifb_trace"):
                    callback = self._telemetry_callback
                    if callback is not None:
                        try:
                            callback(dict(event))
                        except Exception:
                            # A broken consumer must never kill the reader (and
                            # with it every in-flight request).
                            logger.exception(
                                "worker telemetry callback raised; reader continues"
                            )
                    continue
                rid = event.get("request_id") or event.get("id")
                with self._inflight_lock:
                    q = self._inflight.get(rid) if rid else None
                    # Opt-in fan-out: a matching cancelled terminal also goes
                    # to the per-id cancel-receipt queue, WITHOUT replacing
                    # the inflight queue. Mismatched/unsolicited ids have no
                    # receipt queue and are dropped as before.
                    rq = None
                    if (
                        rid
                        and event.get("event") == "cancelled"
                        and event.get("id") == rid
                    ):
                        rq = self._cancel_receipts.get(rid)
                if q is not None:
                    q.put(event)
                if rq is not None:
                    # Bounded receipt queue (maxsize=1): never block the
                    # reader. Retain the FIRST actual matching terminal; a
                    # later duplicate is dropped, not queued.
                    self._offer_receipt(rq, dict(event))
                # else: stale / unsolicited. Drop silently.
        except Exception:
            logger.exception("worker stdout reader crashed")
        finally:
            with self._inflight_lock:
                for q in self._inflight.values():
                    q.put({"event": "_worker_exit"})
                self._inflight.clear()
                for rq in self._cancel_receipts.values():
                    self._offer_receipt(rq, {"event": "_worker_exit"})
                self._cancel_receipts.clear()


# Backwards-compat alias.
_WorkerIO = WorkerIO
