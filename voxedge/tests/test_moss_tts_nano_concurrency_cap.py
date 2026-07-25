"""MOSS-TTS-Nano concurrency capability follows its native worker slots."""

from voxedge.backends.jetson.moss_tts_nano import (
    MossTtsNanoBackend,
    MossTtsNanoConfig,
)


def _cap(max_slots: int):
    backend = MossTtsNanoBackend(MossTtsNanoConfig(max_slots=max_slots))
    return backend.concurrency_capability()


def test_moss_single_slot_is_serial():
    cap = _cap(1)
    assert cap.supports_parallel is False
    assert cap.max_concurrent == 1
    assert cap.is_stateful is True
    assert cap.requires_exclusive_device is True
    assert cap.scaling_mode == "single_runtime_multiplex"


def test_moss_two_slots_advertises_parallel():
    cap = _cap(2)
    assert cap.supports_parallel is True
    assert cap.max_concurrent == 2


def test_moss_nonpositive_slots_fail_safe_to_one():
    cap = _cap(0)
    assert cap.supports_parallel is False
    assert cap.max_concurrent == 1
