"""Admission ceiling for the Whisper backend (encoder + CPU KV decoder).

The encoder configures its bindings and output buffer once and the decoder
carries a KV cache, so two threads inside ``transcribe_array`` would write the
same buffers. Execution therefore stays serialized on ``WhisperASR._lock``;
``WhisperASRConfig.max_concurrent`` only decides how many callers may be
ADMITTED and queue behind that lock instead of taking a 429.

These tests build the config plus a config-bearing stub (no encoder, no
onnxruntime) and read ``concurrency_capability()`` — Mac-safe.
"""

from __future__ import annotations

import threading

import pytest

from voxedge.backends.whisper import WhisperASRConfig
from voxedge.backends.whisper.asr import WhisperASR


def _cfg(**kw) -> WhisperASRConfig:
    base = dict(
        encoder_kind="tensorrt",
        encoder_path="/nonexistent/encoder.plan",
        decoder_dir="/nonexistent/decoder",
        vocab_dir="/nonexistent/vocab",
        window_s=30.0,
        language="en",
    )
    base.update(kw)
    return WhisperASRConfig(**base)


def _cap(config: WhisperASRConfig):
    # __new__ + inject config: skip __init__, exactly how the product
    # capability probe (concurrency_capability_for_spec) does it.
    stub = WhisperASR.__new__(WhisperASR)
    stub._cfg = config
    return stub.concurrency_capability()


def test_default_cap_is_one():
    cfg = _cfg()
    assert cfg.max_concurrent == 1
    cap = _cap(cfg)
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False


def test_admission_ceiling_raised_stays_serialized():
    # The point of the change: ceiling > 1 WITHOUT claiming parallelism, so the
    # coordinator queues the extra callers rather than rejecting them.
    cap = _cap(_cfg(max_concurrent=8))
    assert cap.max_concurrent == 8
    assert cap.supports_parallel is False
    assert cap.scaling_mode == "single_runtime_multiplex"


def test_exclusive_device_still_tracks_encoder_kind():
    # HailoRT hands /dev/hailo0 to one process; TRT shares the Jetson GPU.
    assert _cap(_cfg(encoder_kind="tensorrt")).requires_exclusive_device is False
    assert (
        _cap(
            _cfg(encoder_kind="hailo", encoder_path="/nonexistent/e.hef", window_s=10.0)
        ).requires_exclusive_device
        is True
    )


@pytest.mark.parametrize("bad", [0, -3])
def test_non_positive_ceiling_rejected(bad):
    with pytest.raises(ValueError, match="max_concurrent"):
        _cfg(max_concurrent=bad)


@pytest.mark.parametrize("bad", [1.5, True, "4"])
def test_non_int_ceiling_rejected(bad):
    with pytest.raises(ValueError, match="max_concurrent"):
        _cfg(max_concurrent=bad)


def test_backend_holds_a_reentrant_execution_lock():
    # transcribe_array takes the lock and calls into helpers that may take it
    # again (unload during hot reload), so a plain Lock would self-deadlock.
    backend = WhisperASR(_cfg(max_concurrent=4))
    assert backend._lock.acquire(blocking=False)
    try:
        assert backend._lock.acquire(blocking=False), "lock must be re-entrant"
        backend._lock.release()
    finally:
        backend._lock.release()


def test_execution_is_serialized_across_threads():
    """Two threads must not be inside the transcribe body at once."""
    backend = WhisperASR(_cfg(max_concurrent=4))
    inside = 0
    overlap = False
    gate = threading.Barrier(2, timeout=5)

    def fake_body(samples, language="auto"):
        nonlocal inside, overlap
        inside += 1
        if inside > 1:
            overlap = True
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            # Expected: the second thread cannot reach the barrier while the
            # first holds the lock, so it times out. That IS the serialization.
            pass
        inside -= 1
        return "done"

    backend._transcribe_array_locked = fake_body
    threads = [
        threading.Thread(target=backend.transcribe_array, args=(None,))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in threads)
    assert overlap is False
