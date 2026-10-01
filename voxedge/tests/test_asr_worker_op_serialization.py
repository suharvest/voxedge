"""Regression test for F1: ASR worker-op serialization.

The ASR worker is single-concurrency (one C++ IPC at a time). Before the fix,
accept_audio / finalize_with_status / get_partial_for_generation snapshotted the
stream under _lock then RELEASED it before the executor call, so those ops could
run concurrently with each other / with cancel / create on the same worker from
the three driver tasks — corrupting IPC ordering. The fix holds _lock across
every worker op so at most one runs at a time.

This test drives accept/get_partial/finalize/cancel concurrently against an
instrumented stream that records the MAX number of overlapping worker ops
(across real executor threads) and asserts it never exceeds 1.
"""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from voxedge.engine.asr_session_manager import ASRSessionManager


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())
    wrapper.__name__ = coro_fn.__name__
    return wrapper


class _ConcurrencyProbe:
    """Thread-safe live-op counter that records the max overlap seen."""

    def __init__(self):
        self._lock = threading.Lock()
        self.live = 0
        self.max_live = 0

    def enter(self):
        with self._lock:
            self.live += 1
            if self.live > self.max_live:
                self.max_live = self.live

    def exit(self):
        with self._lock:
            self.live -= 1


class _InstrumentedStream:
    """Every worker op bumps the probe, sleeps to widen the overlap window,
    then drops it. If two ops ever run at once, probe.max_live > 1."""

    def __init__(self, probe: _ConcurrencyProbe):
        self._probe = probe

    def _op(self, dur=0.02):
        self._probe.enter()
        try:
            time.sleep(dur)
        finally:
            self._probe.exit()

    def accept_waveform(self, sample_rate, samples):
        self._op()

    def get_partial(self):
        self._op()
        return ("partial", False)

    def finalize(self):
        self._op(dur=0.05)
        return ("final text", "zh")

    def prepare_finalize(self):
        self._op(dur=0.05)

    def cancel(self):
        self._op()

    def close(self):
        pass


class _InstrumentedBackend:
    sample_rate = 16000

    def __init__(self, probe: _ConcurrencyProbe):
        self._probe = probe

    def create_stream(self, language="auto"):
        # create_stream is itself a worker op — count it too.
        self._probe.enter()
        try:
            time.sleep(0.01)
        finally:
            self._probe.exit()
        return _InstrumentedStream(self._probe)


@run_async
async def test_worker_ops_never_overlap():
    probe = _ConcurrencyProbe()
    # executor=None → default multi-thread pool: ops WOULD run in parallel if
    # the manager didn't serialize them under _lock.
    mgr = ASRSessionManager(_InstrumentedBackend(probe), sample_rate=16000)
    await mgr.on_speech_start()

    samples = b"\x00\x00" * 256

    async def feed():
        for _ in range(8):
            await mgr.accept_audio(samples)

    async def poll():
        for _ in range(8):
            await mgr.get_partial_for_generation()

    # Fire accept + partial pollers concurrently, then a finalize, racing a
    # cancel — the exact cross-task contention the fix serializes.
    feeders = [asyncio.create_task(feed()) for _ in range(3)]
    pollers = [asyncio.create_task(poll()) for _ in range(2)]
    await asyncio.sleep(0.03)
    fin = asyncio.create_task(mgr.finalize_with_status("vad_end"))
    await asyncio.gather(*feeders, *pollers, fin, return_exceptions=True)

    assert probe.max_live == 1, (
        f"worker ops overlapped (max concurrent = {probe.max_live}); "
        "single-worker IPC ordering would be corrupted"
    )


@run_async
async def test_cancel_does_not_overlap_accept():
    probe = _ConcurrencyProbe()
    mgr = ASRSessionManager(_InstrumentedBackend(probe), sample_rate=16000)
    await mgr.on_speech_start()
    samples = b"\x00\x00" * 256

    async def feed():
        for _ in range(10):
            await mgr.accept_audio(samples)

    feeders = [asyncio.create_task(feed()) for _ in range(3)]
    await asyncio.sleep(0.02)
    await mgr.cancel("bargein")  # races the in-flight accept feeders
    await asyncio.gather(*feeders, return_exceptions=True)
    assert probe.max_live == 1, f"cancel overlapped accept (max={probe.max_live})"


@run_async
async def test_prepare_finalize_does_not_overlap_worker_ops():
    probe = _ConcurrencyProbe()
    mgr = ASRSessionManager(_InstrumentedBackend(probe), sample_rate=16000)
    gen = await mgr.on_speech_start()
    samples = b"\x00\x00" * 256

    async def feed():
        for _ in range(6):
            await mgr.accept_audio(samples)

    feeders = [asyncio.create_task(feed()) for _ in range(2)]
    await asyncio.sleep(0.02)
    prep = asyncio.create_task(mgr.prepare_finalize_for_generation(gen))
    await asyncio.gather(*feeders, prep, return_exceptions=True)

    assert prep.result() == (gen, True)
    assert probe.max_live == 1, (
        f"prepare_finalize overlapped worker ops (max={probe.max_live})"
    )


class _BlockingFinalizeStream:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel_entered = threading.Event()
        self.overlap = False
        self._lock = threading.Lock()
        self._in_finalize = False

    def finalize(self):
        with self._lock:
            self._in_finalize = True
            self.started.set()
        self.release.wait(3.0)
        with self._lock:
            self._in_finalize = False
        return ("old", None)

    def cancel(self):
        with self._lock:
            self.overlap = self._in_finalize
            self.cancel_entered.set()

    def close(self):
        pass

    def accept_waveform(self, sample_rate, samples):
        pass

    def get_partial(self):
        return ("", False)


class _BlockingFinalizeBackend:
    sample_rate = 16000

    def __init__(self):
        self.stream = _BlockingFinalizeStream()
        self.restart_called = threading.Event()

    def create_stream(self, language="auto"):
        return self.stream

    def restart_worker(self):
        self.restart_called.set()


@run_async
async def test_cancel_waits_for_cancelled_finalize_worker_op():
    backend = _BlockingFinalizeBackend()
    executor = ThreadPoolExecutor(max_workers=2)
    mgr = ASRSessionManager(backend, executor=executor, sample_rate=16000)
    await mgr.on_speech_start()
    finalize = asyncio.create_task(mgr.finalize_with_status("vad_end"))
    while not backend.stream.started.is_set():
        await asyncio.sleep(0.005)
    finalize.cancel()
    try:
        await finalize
    except asyncio.CancelledError:
        pass
    cancel = asyncio.create_task(mgr.cancel("bargein"))
    await asyncio.sleep(0.1)
    assert not backend.stream.cancel_entered.is_set()
    assert not backend.stream.overlap
    backend.stream.release.set()
    await cancel
    assert backend.stream.cancel_entered.is_set()
    assert not backend.stream.overlap
    executor.shutdown(wait=True)


@run_async
async def test_cancel_timeout_restarts_wedged_worker_op():
    backend = _BlockingFinalizeBackend()
    executor = ThreadPoolExecutor(max_workers=2)
    mgr = ASRSessionManager(backend, executor=executor, sample_rate=16000)
    await mgr.on_speech_start()
    finalize = asyncio.create_task(mgr.finalize_with_status("vad_end"))
    while not backend.stream.started.is_set():
        await asyncio.sleep(0.005)
    finalize.cancel()
    try:
        await finalize
    except asyncio.CancelledError:
        pass
    await mgr.cancel("bargein")
    assert backend.restart_called.is_set()
    backend.stream.release.set()
    executor.shutdown(wait=True)


class _RestartingBackend:
    sample_rate = 16000

    def __init__(self):
        self.streams = []
        self.restarted = threading.Event()

    def create_stream(self, language="auto"):
        stream = _BlockingFinalizeStream()
        self.streams.append(stream)
        return stream

    def restart_worker(self):
        self.restarted.set()


class _FailedRestartBackend(_RestartingBackend):
    def restart_worker(self):
        self.restarted.set()
        raise RuntimeError("restart unavailable")


@run_async
async def test_restart_retires_queued_old_cancel_before_new_stream():
    backend = _RestartingBackend()
    executor = ThreadPoolExecutor(max_workers=2)
    mgr = ASRSessionManager(backend, executor=executor, sample_rate=16000)
    await mgr.on_speech_start()
    old = backend.streams[0]
    finalize = asyncio.create_task(mgr.finalize_with_status("vad_end"))
    while not old.started.is_set():
        await asyncio.sleep(0.005)
    finalize.cancel()
    try:
        await finalize
    except asyncio.CancelledError:
        pass
    await mgr.cancel("bargein")
    assert backend.restarted.is_set()
    new_gen = await mgr.on_speech_start()
    assert new_gen == 2
    new = backend.streams[1]
    await mgr.accept_audio(b"\0\0" * 64)
    old.release.set()
    await asyncio.sleep(0.1)
    assert not old.cancel_entered.is_set()
    assert not old.overlap
    assert new is not old
    executor.shutdown(wait=True)


@run_async
async def test_failed_restart_keeps_old_mutex_until_native_op_returns():
    """Restart failure must not publish a lock that bypasses old native work."""
    backend = _FailedRestartBackend()
    executor = ThreadPoolExecutor(max_workers=2)
    mgr = ASRSessionManager(backend, executor=executor, sample_rate=16000)
    await mgr.on_speech_start()
    old = backend.streams[0]
    finalize = asyncio.create_task(mgr.finalize_with_status("vad_end"))
    while not old.started.is_set():
        await asyncio.sleep(0.005)
    finalize.cancel()
    try:
        await finalize
    except asyncio.CancelledError:
        pass
    await mgr.cancel("bargein")
    new_start = asyncio.create_task(mgr.on_speech_start())
    await asyncio.sleep(0.1)
    assert not new_start.done()
    assert len(backend.streams) == 1
    assert not old.overlap
    old.release.set()
    await new_start
    assert len(backend.streams) == 2
    executor.shutdown(wait=True)
