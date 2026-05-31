"""TRT-Edge-LLM TTS worker concurrency capability reads config (config → cap).

Migration gap: the env/profile → config precedence (``OVS_TTS_WORKER_CONCURRENCY``
/ profile ``tts_worker_concurrency``) is covered product-side in
``app/tests/test_voxedge_backend_config.py`` (G3). voxedge must not import
``app.*``, so this locks the voxedge half: a ``TRTEdgeLLMTTSConfig`` with a given
``worker_concurrency`` produces the matching ``ConcurrencyCapability``. N>1 (the
WorkerIO slot-pool) flips ``supports_parallel``.

This is the N=2 Jetson workhorse path (see memory
``tts_n2_phase_b_stability_landed``) — the cap declaration is what the session
limiter reads, so it must stay pinned. Mac-safe: builds a config-bearing backend
stub via ``__new__`` (no model load, no CUDA).
"""

from __future__ import annotations

from voxedge.backends.jetson.trt_edge_llm_tts import (
    TRTEdgeLLMTTSConfig,
    TRTEdgeLLMTTSBackend,
)


def _cap(config: TRTEdgeLLMTTSConfig):
    stub = TRTEdgeLLMTTSBackend.__new__(TRTEdgeLLMTTSBackend)
    stub._config = config
    return stub.concurrency_capability()


def test_default_n_is_one_serial():
    cfg = TRTEdgeLLMTTSConfig()
    assert cfg.worker_concurrency == 1
    cap = _cap(cfg)
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False
    assert cap.scaling_mode == "single_runtime_multiplex"
    assert cap.requires_exclusive_device is True
    assert cap.is_stateful is True


def test_n_two_enables_parallel():
    cap = _cap(TRTEdgeLLMTTSConfig(worker_concurrency=2))
    assert cap.max_concurrent == 2
    assert cap.supports_parallel is True
    assert cap.scaling_mode == "single_runtime_multiplex"


def test_n_override_higher():
    cap = _cap(TRTEdgeLLMTTSConfig(worker_concurrency=4))
    assert cap.max_concurrent == 4
    assert cap.supports_parallel is True


def test_n_clamped_at_least_one():
    # __post_init__ clamps to >= 1 (no zero / negative ceilings).
    cfg = TRTEdgeLLMTTSConfig(worker_concurrency=0)
    assert cfg.worker_concurrency == 1
    assert _cap(cfg).max_concurrent == 1
    cfg_neg = TRTEdgeLLMTTSConfig(worker_concurrency=-3)
    assert cfg_neg.worker_concurrency == 1
    assert _cap(cfg_neg).max_concurrent == 1
