import threading

import numpy as np
import pytest

from voxedge.backends.rk.tts import RKTTSBackend
from voxedge.backends.base import TTSCapability


class FakeStream:
    def __init__(self, items):
        self.items = items
        self.closed = 0

    def __iter__(self):
        return iter(self.items)

    def close(self):
        self.closed += 1


class FakeInner:
    name = "fake"

    def __init__(self):
        self.closed = 0
        self.calls = []
        self.fail_close = False
        self.fail_preload = False
        self.stream = None

    def get_sample_rate(self):
        return 24000

    def is_ready(self):
        return True

    def preload(self):
        self.calls.append(("preload",))
        if self.fail_preload:
            raise RuntimeError("preload failed")

    def synthesize(self, **kwargs):
        self.calls.append(("synthesize", kwargs))
        return b"wav", {"ok": True}

    def synthesize_stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        self.stream = FakeStream([(np.array([0.5], np.float32), {"x": 1})])
        return self.stream

    def close(self):
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("close failed")


def backend_with_inner():
    backend = RKTTSBackend()
    backend._inner = FakeInner()
    return backend


def test_unload_calls_close_once_and_duplicate_is_harmless():
    backend = backend_with_inner()
    inner = backend._inner
    backend.unload()
    backend.unload()
    assert inner.closed == 1
    assert backend._inner is None


@pytest.mark.parametrize("supports, expect_streaming", [(False, False), (True, True), (None, True)])
def test_capabilities_follow_explicit_inner_streaming_support(supports, expect_streaming):
    backend = backend_with_inner()
    backend._inner.name = "kokoro_convonly"
    if supports is not None:
        backend._inner.supports_streaming = supports
    assert (TTSCapability.STREAMING in backend.capabilities) is expect_streaming


def test_legacy_inner_with_abc_false_and_stream_override_keeps_streaming():
    backend = backend_with_inner()
    backend._inner.name = "qwen3_rknn"
    backend._inner.supports_streaming = False  # inherited-ABC-shaped legacy value
    backend._inner.synthesize_stream = lambda **_: FakeStream([])  # actual override
    assert TTSCapability.STREAMING in backend.capabilities


def test_unload_failure_retains_owner_and_propagates():
    backend = backend_with_inner()
    inner = backend._inner
    inner.fail_close = True
    with pytest.raises(RuntimeError, match="close failed"):
        backend.unload()
    assert backend._inner is inner
    assert backend.is_ready() is False
    with pytest.raises(RuntimeError, match="not loaded"):
        backend._synthesize_impl("blocked")


def test_stream_close_runs_when_consumer_stops_and_cancel_forwards():
    backend = backend_with_inner()
    cancel = threading.Event()
    chunks = list(backend._generate_streaming_impl("hi", cancel_event=cancel))
    assert len(chunks) == 1  # one float32 sample becomes one int16 PCM sample
    inner = backend._inner
    assert inner.stream.closed == 1
    assert inner.calls[0][0] == "stream"
    assert inner.calls[0][1]["cancel_event"] is cancel


def test_synchronous_call_forwards_cancel_event():
    backend = backend_with_inner()
    out = backend._synthesize_impl("hi", cancel_event=threading.Event())
    assert out[0] == b"wav"
    assert backend._inner.calls[0][1]["cancel_event"] is not None


def test_synchronous_call_holds_lifecycle_lock_against_concurrent_unload():
    backend = backend_with_inner()
    entered = threading.Event()
    release = threading.Event()
    original = backend._inner.synthesize

    def blocked_synthesize(**kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original(**kwargs)

    backend._inner.synthesize = blocked_synthesize
    call = threading.Thread(target=lambda: backend._synthesize_impl("hi"))
    call.start()
    assert entered.wait(timeout=1)

    unloaded = threading.Event()
    unload = threading.Thread(target=lambda: (backend.unload(), unloaded.set()))
    unload.start()
    assert not unloaded.wait(timeout=0.1)

    release.set()
    call.join(timeout=2)
    unload.join(timeout=2)
    assert not call.is_alive()
    assert not unload.is_alive()
    assert unloaded.is_set()
    assert backend._inner is None


def test_runtime_info_forwards_inner_diagnostics():
    backend = backend_with_inner()
    backend._inner.runtime_info = {"route": "rk3576"}
    assert backend.runtime_info() == {
        "route": "rk3576", "lifecycle_busy": False,
        "lifecycle_ready": True, "ready": True,
    }


def test_preload_double_failure_retains_owner_and_stays_unready():
    backend = backend_with_inner()
    inner = backend._inner
    inner.fail_preload = True
    inner.fail_close = True
    with pytest.raises(RuntimeError, match="close failed"):
        backend.preload()
    assert backend._inner is inner
    assert backend.is_ready() is False
    with pytest.raises(RuntimeError, match="retained an inner owner"):
        backend.preload()
    with pytest.raises(RuntimeError, match="not loaded"):
        backend._synthesize_impl("blocked")
    with pytest.raises(RuntimeError, match="not loaded"):
        list(backend._generate_streaming_impl("blocked"))
    inner.fail_close = False
    backend.unload()
    assert backend._inner is None


def test_stream_can_close_from_a_different_thread():
    backend = backend_with_inner()
    stream = backend._generate_streaming_impl("hi")
    got = []
    t1 = threading.Thread(target=lambda: got.append(next(stream)))
    t1.start()
    t1.join()
    t2 = threading.Thread(target=stream.close)
    t2.start()
    t2.join()
    assert got
    assert backend._inner.stream.closed == 1
    backend.unload()


def test_runtime_info_does_not_deadlock_while_stream_is_paused():
    backend = backend_with_inner()
    stream = backend._generate_streaming_impl("hi")
    next(stream)
    info = backend.runtime_info()
    assert info["lifecycle_busy"] is True
    stream.close()
    backend.unload()


def test_public_dsp_stream_close_releases_inner_before_unload():
    backend = backend_with_inner()
    backend._cached_sample_rate = 24000
    stream = backend.generate_streaming("hi", speed=1.2)
    next(stream)
    stream.close()
    assert backend._inner.stream.closed == 1
    backend.unload()


def test_public_identity_stream_closes_delegate_exactly_once():
    backend = backend_with_inner()
    stream = backend.generate_streaming("hi")
    next(stream)
    stream.close()
    assert backend._inner.stream.closed == 1
    backend.unload()


def test_runtime_info_after_unload_cannot_report_stale_ready():
    backend = backend_with_inner()
    backend._inner.runtime_info = {"ready": True, "route": "rk3576"}
    backend._runtime_info_cache = dict(backend._inner.runtime_info)
    backend.unload()
    info = backend.runtime_info()
    assert info["ready"] is False
    assert info["lifecycle_ready"] is False


def test_runtime_info_preserves_inner_not_ready_state():
    backend = backend_with_inner()
    backend._inner.runtime_info = {"ready": False, "route": "rk3576"}
    info = backend.runtime_info()
    assert info["ready"] is False
    assert info["lifecycle_ready"] is True


def test_runtime_info_busy_without_cache_is_not_ready():
    backend = backend_with_inner()
    acquired = backend._lifecycle_lock.acquire(blocking=False)
    assert acquired
    try:
        info = backend.runtime_info()
    finally:
        backend._lifecycle_lock.release()
    assert info["ready"] is False
    assert info["lifecycle_busy"] is True


def test_repreload_busy_does_not_reuse_unload_ready_cache():
    backend = backend_with_inner()
    backend._inner.runtime_info = {"ready": True, "route": "rk3576"}
    assert backend.runtime_info()["ready"] is True
    backend.unload()
    inner = FakeInner()
    entered = threading.Event()
    release = threading.Event()

    def blocked_preload():
        entered.set()
        release.wait(timeout=2)

    inner.preload = blocked_preload
    backend._ensure_inner = lambda: setattr(backend, "_inner", inner)
    worker = threading.Thread(target=backend.preload)
    worker.start()
    assert entered.wait(timeout=1)
    info = backend.runtime_info()
    assert info["ready"] is False
    assert info["lifecycle_busy"] is True
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    backend.unload()
