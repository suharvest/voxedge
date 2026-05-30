"""G3: Kokoro TRT TTS concurrency capability reads config (config → capability).

The product-side env/profile → config precedence is covered in the product
test (app/tests/test_voxedge_backend_config.py + its G3 section) since voxedge
must not import ``app.*``. Here we lock the voxedge half: a KokoroTRTConfig with
a given ``stream_max_workers`` produces the matching ConcurrencyCapability.

Mac-safe: builds a config-bearing backend stub via ``__new__`` (no model load,
no CUDA).

Hardware K=2 smoke (dual-client burst, byte-identical audio, 0 CUDA errors) is
a deploy-time TODO — see memory ``kokoro_trt_hot_reload_verified`` for the
existing single-runtime-multiplex N=2 evidence pattern; re-run on the target
Jetson after this cap wiring lands.
"""

from __future__ import annotations

from voxedge.backends.jetson.kokoro_trt import (
    KokoroTRTConfig,
    KokoroTRTBackend,
)


def _cap(config: KokoroTRTConfig):
    stub = KokoroTRTBackend.__new__(KokoroTRTBackend)
    stub._config = config
    return stub.concurrency_capability()


def test_default_k_is_two():
    cfg = KokoroTRTConfig()
    assert cfg.stream_max_workers == 2
    cap = _cap(cfg)
    assert cap.max_concurrent == 2
    assert cap.supports_parallel is True
    assert cap.scaling_mode == "single_runtime_multiplex"


def test_k_one_is_serial():
    cap = _cap(KokoroTRTConfig(stream_max_workers=1))
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False


def test_k_override_higher():
    cap = _cap(KokoroTRTConfig(stream_max_workers=3))
    assert cap.max_concurrent == 3
    assert cap.supports_parallel is True


def test_k_clamped_at_least_one():
    cfg = KokoroTRTConfig(stream_max_workers=0)
    assert cfg.stream_max_workers == 1
    assert _cap(cfg).max_concurrent == 1
