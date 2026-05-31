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

import json
import queue
import threading
import time

import pytest

from voxedge.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    TRTEdgeLLMASRConfig,
    WorkerExitError,
    _TRTEdgeLLMStreamingASRStream,
)
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
