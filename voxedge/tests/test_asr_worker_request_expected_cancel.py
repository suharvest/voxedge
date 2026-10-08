"""Focused tests for the opt-in ``_worker_request(expected_cancel=...)`` arm.

Status: AUTHORED, UNRUN (static/compile author pass only).

What this file pins (expected-cancel carve-out ONLY):

  1. GENUINE OVERLAP, same SID: a real-stdio scripted child holds the ordinary
     request ``sess-1``; an actual ``_worker_request`` thread runs with an
     ARMED ``expected_cancel`` event; the canonical
     ``_worker_cancel_and_wait`` (same SID) triggers the native ``cancelled``
     terminal which WorkerIO fans out to BOTH the receipt queue and the
     ordinary inflight queue. The ordinary consumer must receive the EXACT
     SAME actual payload (ok=false, epoch=7) as the helper, its generator is
     closed, the inflight entry is released, and the marker event REMAINS SET
     (no clear anywhere).
  2. NO opt-in (``expected_cancel=None``): the same cancelled terminal keeps
     the existing generic behavior — ``WorkerProtocolError``.
  3. UNARMED event (never set): same — ``WorkerProtocolError``, no widening.
  4. WRONG ID: a matching-looking ``cancelled`` terminal whose ``id`` differs
     from the request id must NOT route to success even with an armed event.

No GPU import, no server/core involvement, no mocked WIO queues/reader/
semaphore: real ``WorkerIO`` over real OS pipes with stdlib Python children.
Every child teardown is graceful and finite (stdin EOF → bounded wait →
polite TERM), never SIGKILL; the reader thread is joined with a finite budget.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

_wio_mod = importlib.import_module("voxedge.backends.jetson.worker_io")
WorkerIO = _wio_mod.WorkerIO

_asr_mod = importlib.import_module("voxedge.backends.jetson.trt_edge_llm_asr")
TRTEdgeLLMASRBackend = _asr_mod.TRTEdgeLLMASRBackend
WorkerProtocolError = _asr_mod.WorkerProtocolError

EPOCH = 7

# Holds ordinary request ``sess-1`` until a cancel input arrives, then emits
# the native cancelled terminal (epoch 7) — for the cancel's own id, which in
# the overlap test IS the same SID as the held ordinary request. Any other
# id request gets an immediate echo_ack.
_HOLD_CHILD_SOURCE = r"""
import json
import sys

EPOCH = 7
pending = None
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        obj = json.loads(raw)
    except Exception:
        continue
    if not isinstance(obj, dict):
        continue
    if obj.get("type") == "cancel":
        sid = obj.get("id")
        sys.stdout.write(json.dumps(
            {"event": "cancelled", "id": sid, "ok": False, "epoch": EPOCH}
        ) + "\n")
        sys.stdout.flush()
        if pending is not None:
            pending = None
        continue
    sid = obj.get("id")
    if sid == "sess-1":
        pending = sid
        sys.stderr.write("HELD\n")
        sys.stderr.flush()
        continue
    if sid is not None:
        sys.stdout.write(json.dumps(
            {"event": "echo_ack", "id": sid, "ok": True}
        ) + "\n")
        sys.stdout.flush()
"""

# Replies to the ordinary request ``sess-1`` with a cancelled terminal whose
# id does NOT match (wrong id): the narrow arm must stay closed.
_WRONG_ID_CHILD_SOURCE = r"""
import json
import sys

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        obj = json.loads(raw)
    except Exception:
        continue
    if not isinstance(obj, dict):
        continue
    sid = obj.get("id")
    if sid == "sess-1":
        sys.stdout.write(json.dumps(
            {"event": "cancelled", "id": "sess-OTHER", "ok": False,
             "epoch": 7}
        ) + "\n")
        sys.stdout.flush()
    elif sid is not None:
        sys.stdout.write(json.dumps(
            {"event": "echo_ack", "id": sid, "ok": True}
        ) + "\n")
        sys.stdout.flush()
"""

_ERROR_CHILD_SOURCE = r"""
import json
import sys

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    obj = json.loads(raw)
    sid = obj.get("id", "")
    print(json.dumps({
        "id": sid,
        "event": "error",
        "ok": False,
        "error": "pcm rejected",
    }), flush=True)
    # Keep the child alive: the consumer must close its generator immediately.
"""

_DONE_CHILD_SOURCE = r"""
import json
import sys

for raw in sys.stdin:
    raw = raw.strip()
    if raw:
        obj = json.loads(raw)
        print(json.dumps({
            "id": obj.get("id", ""),
            "event": "done",
            "ok": True,
            "responses": [{"output_text": "hello"}],
        }), flush=True)
"""

_CANCELLED_CHILD_SOURCE = r"""
import json
import sys

for raw in sys.stdin:
    raw = raw.strip()
    if raw:
        obj = json.loads(raw)
        print(json.dumps({
            "id": obj.get("id", ""),
            "event": "cancelled",
            "ok": False,
        }), flush=True)
"""


class _Child:
    """One scripted real-stdio child + its canonical WorkerIO.

    Finite graceful teardown ONLY: stdin EOF, bounded wait, polite TERM as
    bounded escalation — never SIGKILL; reader joined with finite budget.
    """

    def __init__(self, source: str, concurrency: int = 4) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.wio = WorkerIO(self.proc, concurrency)

    def close(self) -> list[str]:
        problems: list[str] = []
        try:
            if self.proc.stdin is not None and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"stdin.close() raised {exc!r}")
        try:
            self.wio.close()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"wio.close() raised {exc!r}")
        try:
            self.proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                problems.append("child did not exit after terminate()")
        for stream in (self.proc.stdout, self.proc.stderr):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"stream.close() raised {exc!r}")
        thread = getattr(self.wio, "_reader_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
            if thread.is_alive():
                problems.append("WorkerIO _reader_thread still alive")
        return problems


@pytest.fixture
def make_child():
    children: list[_Child] = []

    def _factory(source: str = _HOLD_CHILD_SOURCE, concurrency: int = 4) -> _Child:
        c = _Child(source=source, concurrency=concurrency)
        children.append(c)
        return c

    yield _factory

    problems: list[str] = []
    for idx, c in enumerate(children):
        for p in c.close():
            problems.append(f"child[{idx}]: {p}")
    if problems:
        raise AssertionError("child teardown failures: " + "; ".join(problems))


def _make_backend(child: _Child):
    """``__new__`` backend: only the fields ``_worker_request`` touches.

    ``_ensure_worker`` is a booby trap — the narrow arm must never spawn.
    """
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._worker_lock = threading.Lock()
    backend._worker = child.proc
    backend._wio = child.wio
    backend._worker_failed = False
    backend._worker_failed_reason = None
    backend._worker_ready_meta = {}
    backend._worker_stderr_tail = deque(maxlen=80)

    def _forbidden(*_a, **_k):
        raise AssertionError("_ensure_worker must NOT be called")

    backend._ensure_worker = _forbidden  # type: ignore[method-assign]
    return backend


def _wait_inflight(wio, req_id: str) -> None:
    """Bounded MONOTONIC-deadline barrier: the ordinary request is registered
    in WorkerIO's real ``_inflight`` map (stdin write landed, child holding).
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with wio._inflight_lock:
            if req_id in wio._inflight:
                return
        time.sleep(0.01)
    raise AssertionError(f"{req_id!r} never registered in WorkerIO _inflight")


# ---------------------------------------------------------------------------
# 1. Genuine overlap: same-SID cancel, armed event, actual dict to consumer.
# ---------------------------------------------------------------------------

def test_armed_expected_cancel_returns_actual_dict_to_ordinary_consumer(make_child):
    child = make_child()
    backend = _make_backend(child)
    marker = threading.Event()
    marker.set()  # armed BEFORE the request is sent
    box: dict = {}
    started = threading.Event()

    def _ordinary():
        started.set()
        box["result"] = backend._worker_request(
            {"event": "begin", "id": "sess-1"}, expected_cancel=marker
        )

    t = threading.Thread(target=_ordinary, daemon=True)
    t.start()
    assert started.wait(timeout=5.0)
    _wait_inflight(child.wio, "sess-1")
    assert t.is_alive(), "ordinary request finished before cancel (hold broken)"

    # Canonical helper, SAME SID: native cancelled ok=false epoch=7 receipt.
    receipt = backend._worker_cancel_and_wait("sess-1", 5.0)
    assert receipt == {
        "event": "cancelled",
        "id": "sess-1",
        "ok": False,
        "epoch": 7,
    }

    t.join(timeout=5.0)
    assert not t.is_alive(), "ordinary _worker_request thread did not finish"
    # EXACT SAME actual payload — fanned out by WorkerIO, returned verbatim.
    assert box["result"] == receipt
    # The marker is NEVER cleared by the backend.
    assert marker.is_set(), "expected_cancel marker was cleared"
    # The ordinary generator closed and the inflight slot is released.
    with child.wio._inflight_lock:
        assert "sess-1" not in child.wio._inflight
    # The worker stays healthy: no failure marker, no respawn.
    assert backend._worker_failed is False
    assert backend._worker is child.proc


# ---------------------------------------------------------------------------
# 2. No opt-in: existing generic ok=false behavior preserved.
# ---------------------------------------------------------------------------

def test_no_optin_cancelled_terminal_raises_worker_protocol_error(make_child):
    child = make_child()
    backend = _make_backend(child)
    box: dict = {}
    started = threading.Event()

    def _ordinary():
        started.set()
        try:
            backend._worker_request({"event": "begin", "id": "sess-1"})
        except BaseException as exc:  # noqa: BLE001 - captured, not suppressed
            box["error"] = exc

    t = threading.Thread(target=_ordinary, daemon=True)
    t.start()
    assert started.wait(timeout=5.0)
    _wait_inflight(child.wio, "sess-1")

    receipt = backend._worker_cancel_and_wait("sess-1", 5.0)
    assert receipt["event"] == "cancelled"

    t.join(timeout=5.0)
    assert not t.is_alive()
    assert isinstance(box.get("error"), WorkerProtocolError), (
        f"expected generic WorkerProtocolError, got {box.get('error')!r}"
    )


# ---------------------------------------------------------------------------
# 3. Unarmed event: no widening to success.
# ---------------------------------------------------------------------------

def test_unarmed_expected_cancel_raises_worker_protocol_error(make_child):
    child = make_child()
    backend = _make_backend(child)
    marker = threading.Event()  # NOT set
    box: dict = {}
    started = threading.Event()

    def _ordinary():
        started.set()
        try:
            backend._worker_request(
                {"event": "begin", "id": "sess-1"}, expected_cancel=marker
            )
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=_ordinary, daemon=True)
    t.start()
    assert started.wait(timeout=5.0)
    _wait_inflight(child.wio, "sess-1")

    receipt = backend._worker_cancel_and_wait("sess-1", 5.0)
    assert receipt["event"] == "cancelled"

    t.join(timeout=5.0)
    assert not t.is_alive()
    assert isinstance(box.get("error"), WorkerProtocolError), (
        f"unarmed event must NOT route to success, got {box.get('error')!r}"
    )
    assert not marker.is_set()


# ---------------------------------------------------------------------------
# 4. Wrong id: armed event still refuses a mismatched cancelled terminal.
# ---------------------------------------------------------------------------

def test_wrong_id_cancelled_terminal_raises_even_when_armed(make_child):
    child = make_child(source=_WRONG_ID_CHILD_SOURCE)
    backend = _make_backend(child)
    marker = threading.Event()
    marker.set()
    with pytest.raises(WorkerProtocolError):
        backend._worker_request(
            {"event": "begin", "id": "sess-1"}, expected_cancel=marker
        )
    # Gate failure never clears the marker.
    assert marker.is_set()


def test_transcribe_worker_error_terminates_and_releases_workerio(make_child):
    """An immediate native error must not leave _transcribe_worker in a 60s wait."""
    child = make_child(source=_ERROR_CHILD_SOURCE, concurrency=1)
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._worker_lock = threading.Lock()
    backend._worker = child.proc
    backend._wio = child.wio
    backend._worker_failed = False
    backend._worker_failed_reason = None
    backend._worker_stderr_tail = deque(maxlen=80)
    backend._config = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        max_generate_length=200,
    )
    backend._ensure_worker = lambda: None  # type: ignore[method-assign]

    started = time.monotonic()
    with pytest.raises(WorkerProtocolError, match="pcm rejected"):
        backend._transcribe_worker("/tmp/mel.safetensors", 0.0)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"error response was delayed by WorkerIO timeout: {elapsed:.3f}s"
    with child.wio._inflight_lock:
        assert not child.wio._inflight
    assert child.wio._sem.acquire(timeout=0.2), "WorkerIO semaphore slot leaked"
    child.wio._sem.release()
    assert child.proc.poll() is None, "error response must not kill the live worker"


def _make_transcribe_backend(child: _Child):
    backend = TRTEdgeLLMASRBackend.__new__(TRTEdgeLLMASRBackend)
    backend._worker_lock = threading.Lock()
    backend._worker = child.proc
    backend._wio = child.wio
    backend._worker_failed = False
    backend._worker_failed_reason = None
    backend._worker_ready_meta = {}
    backend._worker_stderr_tail = deque(maxlen=80)
    backend._config = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        max_generate_length=200,
    )
    backend._ensure_worker = lambda: None  # type: ignore[method-assign]
    backend._postprocess_text = lambda text: (text, None)  # type: ignore[method-assign]
    return backend


def test_transcribe_worker_done_preserves_parsed_text(make_child):
    child = make_child(source=_DONE_CHILD_SOURCE, concurrency=1)
    backend = _make_transcribe_backend(child)
    result = backend._transcribe_worker("/tmp/mel.safetensors", 0.0)
    assert result.text == "hello"
    with child.wio._inflight_lock:
        assert not child.wio._inflight
    assert child.wio._sem.acquire(timeout=0.2)
    child.wio._sem.release()


def test_transcribe_worker_cancelled_preserves_original_exception(make_child):
    child = make_child(source=_CANCELLED_CHILD_SOURCE, concurrency=1)
    backend = _make_transcribe_backend(child)
    with pytest.raises(RuntimeError, match="ASR worker failed"):
        backend._transcribe_worker("/tmp/mel.safetensors", 0.0)
    with child.wio._inflight_lock:
        assert not child.wio._inflight
    assert child.wio._sem.acquire(timeout=0.2)
    child.wio._sem.release()
