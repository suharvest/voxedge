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
