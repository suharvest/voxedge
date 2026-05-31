"""Matcha TRT TTS concurrency capability reads config (config → capability).

Migration gap: the env/profile → config precedence
(``OVS_TTS_STREAM_MAX_WORKERS`` / profile ``tts_stream_max_workers``) is covered
product-side in ``app/tests/test_voxedge_backend_config.py``. voxedge must not
import ``app.*``, so this locks the voxedge half: a ``MatchaTRTConfig`` with a
given ``stream_max_workers`` (K) produces the matching ``ConcurrencyCapability``.
Engines (weights) are shared; each slot holds its own per-stream context →
``single_runtime_multiplex``. K>1 flips ``supports_parallel``.

Matcha is an N=2 Jetson workhorse (see memory ``matcha_trt_unload_vram_release``
for the leak-free hot-reload evidence). Mac-safe: builds a config-bearing backend
stub via ``__new__`` (no model load, no CUDA).
"""

from __future__ import annotations

from voxedge.backends.jetson.matcha_trt import (
    MatchaTRTConfig,
    MatchaTRTBackend,
)


def _cap(config: MatchaTRTConfig):
    stub = MatchaTRTBackend.__new__(MatchaTRTBackend)
    stub._config = config
    return stub.concurrency_capability()


def test_default_k_is_two():
    cfg = MatchaTRTConfig()
    assert cfg.stream_max_workers == 2
    cap = _cap(cfg)
    assert cap.max_concurrent == 2
    assert cap.supports_parallel is True
    assert cap.scaling_mode == "single_runtime_multiplex"


def test_k_one_is_serial():
    cap = _cap(MatchaTRTConfig(stream_max_workers=1))
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False


def test_k_override_higher():
    cap = _cap(MatchaTRTConfig(stream_max_workers=3))
    assert cap.max_concurrent == 3
    assert cap.supports_parallel is True


def test_k_clamped_at_least_one():
    cfg = MatchaTRTConfig(stream_max_workers=0)
    assert cfg.stream_max_workers == 1
    assert _cap(cfg).max_concurrent == 1
    cfg_neg = MatchaTRTConfig(stream_max_workers=-2)
    assert cfg_neg.stream_max_workers == 1
    assert _cap(cfg_neg).max_concurrent == 1
