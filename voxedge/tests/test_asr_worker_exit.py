"""TRT-Edge-LLM ASR worker-death + cancel-timeout error contracts (migration gap).

The session manager relies on the streaming ASR backend surfacing a typed
``WorkerExitError`` (not a raw IOError / silent hang) so it can route to
ERROR_REBUILD and respawn the worker. Two paths must honor that contract:

  * ``_worker_request``: when the worker subprocess dies mid-request, the voxedge
    ``WorkerIO`` raises its internal exit sentinel, which the backend re-raises as
    the backend-level ``WorkerExitError`` and clears ``_worker`` so the next call
    rebuilds;
  * ``cancel_and_finalize``: bounded 500ms wait for the ``end`` ack — an
    unresponsive worker must raise ``WorkerExitError`` (and mark the stream
    closed), not block the barge-in path forever.

The old product copy of these tests wired ``app.core.worker_io.WorkerIO`` onto
the backend, which the env-free voxedge backend no longer translates (it only
catches its *own* ``WorkerIO`` exit type). These rewrite them against voxedge's
own ``WorkerIO`` + a fake subprocess (no CUDA, no real worker).
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time

import pytest

from voxedge.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    TRTEdgeLLMASRConfig,
    WorkerExitError,
    WorkerProtocolError,
    _TRTEdgeLLMStreamingASRStream,
)
from voxedge.engine.asr_session_manager import ASRSessionManager, SessionState
from voxedge.backends.jetson.worker_io import WorkerIO


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self.writes.append(s)
        return len(s)

    def flush(self) -> None:
        pass


class _FakeStdoutQueue:
    def __init__(self) -> None:
        self._q: "queue.Queue[str | None]" = queue.Queue()

    def feed(self, line: str) -> None:
        self._q.put(line if line.endswith("\n") else line + "\n")

    def eof(self) -> None:
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdoutQueue()


def _make_backend_with_wio():
    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._config = TRTEdgeLLMASRConfig()
    backend._worker = proc
    backend._wio = wio
    backend._worker_lock = threading.Lock()
    backend._restart_lock = threading.Lock()
    backend._worker_stderr_tail = []  # consumed by _stderr_tail_text()
    backend._ensure_worker = lambda: None  # already wired
    return backend, proc, wio


def _feed_responses(proc, builders):
    """Feed one fake WorkerIO response for each request written by the backend."""
    seen = 0
    while seen < len(builders):
        if len(proc.stdin.writes) > seen:
            req = json.loads(proc.stdin.writes[seen])
            response = builders[seen](req) if callable(builders[seen]) else builders[seen]
            proc.stdout.feed(json.dumps(response))
            seen += 1
        else:
            time.sleep(0.005)


def _begin_ack(req):
    return {"id": req["id"], "event": "begin_ack"}


def _empty_end(req):
    return {
        "id": req["id"],
        "event": "error",
        "ok": False,
        "error": "no_audio_accumulated",
    }


def test_empty_end_error_is_normal_only_for_matching_end_request():
    backend, proc, _wio = _make_backend_with_wio()
    feeder = threading.Thread(
        target=_feed_responses,
        args=(proc, [lambda req: _empty_end(req)]),
        daemon=True,
    )
    feeder.start()
    response = backend._worker_request({"event": "end", "id": "sess-empty"})
    assert response["error"] == "no_audio_accumulated"

    backend, proc, _wio = _make_backend_with_wio()
    feeder = threading.Thread(
        target=_feed_responses,
        args=(proc, [{"id": "sess-begin", "event": "error", "ok": False,
                      "error": "no_audio_accumulated"}]),
        daemon=True,
    )
    feeder.start()
    with pytest.raises(WorkerProtocolError):
        backend._worker_request({"event": "begin", "id": "sess-begin"})


def test_end_other_error_still_raises():
    backend, proc, _wio = _make_backend_with_wio()
    feeder = threading.Thread(
        target=_feed_responses,
        args=(proc, [lambda req: {
            "id": req["id"], "event": "error", "ok": False, "error": "other_error"
        }]),
        daemon=True,
    )
    feeder.start()
    with pytest.raises(WorkerProtocolError):
        backend._worker_request({"event": "end", "id": "sess-end"})


def test_end_mismatched_error_id_still_raises():
    backend, _proc, _wio = _make_backend_with_wio()

    class _OneResponseWIO:
        def request(self, _request):
            yield {
                "id": "wrong",
                "event": "error",
                "ok": False,
                "error": "no_audio_accumulated",
            }

    backend._wio = _OneResponseWIO()
    with pytest.raises(WorkerProtocolError):
        backend._worker_request({"event": "end", "id": "sess-end"})


def test_close_empty_end_is_idempotent_and_manager_does_not_restart():
    backend, proc, _wio = _make_backend_with_wio()
    feeder = threading.Thread(
        target=_feed_responses,
        args=(proc, [_begin_ack, _empty_end]),
        daemon=True,
    )
    feeder.start()
    stream = _TRTEdgeLLMStreamingASRStream(backend)
    stream.close()
    stream.close()
    assert stream._closed is True
    assert len(proc.stdin.writes) == 2

    backend2, proc2, _wio2 = _make_backend_with_wio()
    feeder2 = threading.Thread(
        target=_feed_responses,
        args=(proc2, [_begin_ack, _empty_end]),
        daemon=True,
    )
    feeder2.start()
    stream2 = _TRTEdgeLLMStreamingASRStream(backend2)
    restart_calls = []
    backend2.restart_worker = lambda: restart_calls.append(True)
    manager = ASRSessionManager(backend2)
    manager._state = SessionState.ACTIVE
    manager._stream = stream2
    asyncio.run(manager._inner_cancel(reason="empty-audio"))
    assert restart_calls == []
    assert manager.state is SessionState.IDLE


def test_worker_request_worker_exit_raises_worker_exit_error():
    backend, proc, _wio = _make_backend_with_wio()

    def _kill():
        time.sleep(0.02)
        proc.stdout.eof()  # reader thread observes EOF → exit sentinel

    threading.Thread(target=_kill, daemon=True).start()
    with pytest.raises(WorkerExitError):
        backend._worker_request({"event": "begin", "id": "sess-1"})
    # _worker cleared so the next call rebuilds via _ensure_worker.
    assert backend._worker is None


def test_cancel_and_finalize_timeout_raises_worker_exit_error():
    backend, proc, _wio = _make_backend_with_wio()

    # Feed begin_ack so the stream constructor's _begin() returns, then leave the
    # queue idle so the subsequent 'end' never gets acked → 500ms timeout trips.
    def _feeder_begin_only():
        for _ in range(50):
            if proc.stdin.writes:
                break
            time.sleep(0.01)
        first = json.loads(proc.stdin.writes[0])
        proc.stdout.feed(json.dumps({"id": first["id"], "event": "begin_ack"}))

    threading.Thread(target=_feeder_begin_only, daemon=True).start()
    stream = _TRTEdgeLLMStreamingASRStream(backend)

    start = time.time()
    with pytest.raises(WorkerExitError):
        stream.cancel_and_finalize()
    elapsed = time.time() - start
    assert 0.4 < elapsed < 2.0, f"cancel timeout took {elapsed:.3f}s, expected ~0.5s"
    assert stream._closed is True


# ── supports_hot_reload tracks worker vs in-process mode (config-driven) ──────


def test_supports_hot_reload_true_when_worker_mode():
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._config = TRTEdgeLLMASRConfig(use_worker=True)
    assert backend.supports_hot_reload is True


def test_supports_hot_reload_false_when_inprocess():
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._config = TRTEdgeLLMASRConfig(use_worker=False)
    assert backend.supports_hot_reload is False
