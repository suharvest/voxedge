"""G2: bounded concurrency cap for the Paraformer TRT ASR backend.

Each ``ParaformerTRTStream`` builds its own per-stream TRT execution contexts +
device buffers (``_ParaformerCtxBundle``). The pre-G2 capability returned
``max_concurrent=None`` (unbounded), so a client burst could open arbitrarily
many streams and OOM the device. ``ParaformerTRTConfig.max_concurrent`` (default
2, conservative) now bounds it.

These tests build the config + a config-bearing backend stub (no model load, no
CUDA) and read ``concurrency_capability()`` — Mac-safe.
"""

from __future__ import annotations

from voxedge.backends.jetson.paraformer_trt import (
    ParaformerTRTConfig,
    ParaformerTRTBackend,
)


def _cap(config: ParaformerTRTConfig):
    # __new__ + inject config: skip __init__ (no engines, no CUDA), exactly how
    # the product capability probe (concurrency_capability_for_spec) does it.
    stub = ParaformerTRTBackend.__new__(ParaformerTRTBackend)
    stub._config = config
    return stub.concurrency_capability()


def test_default_cap_is_bounded_two():
    cfg = ParaformerTRTConfig()
    assert cfg.max_concurrent == 2
    cap = _cap(cfg)
    assert cap.max_concurrent == 2
    assert cap.supports_parallel is True
    assert cap.scaling_mode == "multi_runtime_per_slot"


def test_cap_one_is_serial():
    cap = _cap(ParaformerTRTConfig(max_concurrent=1))
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False


def test_cap_override_higher():
    cap = _cap(ParaformerTRTConfig(max_concurrent=4))
    assert cap.max_concurrent == 4
    assert cap.supports_parallel is True


def test_cap_clamped_to_at_least_one():
    # __post_init__ clamps to >= 1 (no unbounded / zero / negative ceilings).
    cfg = ParaformerTRTConfig(max_concurrent=0)
    assert cfg.max_concurrent == 1
    cfg_neg = ParaformerTRTConfig(max_concurrent=-5)
    assert cfg_neg.max_concurrent == 1
