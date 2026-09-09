"""Configurable concurrency cap for the sherpa-onnx CPU ASR backend.

sherpa-onnx recognizers are re-entrant across streams, so this is a real
parallelism declaration, not an admission-only ceiling. The cap used to be the
hardcoded class default 4; it now comes from ``SherpaASRConfig.max_concurrent``
so a deployment can size it against its own core count.

The method moved from a classmethod to an instance method for that reason. The
product probe (``concurrency_capability_for_spec``) builds a config-bearing
stub via ``__new__``, so no model is loaded to answer the question — these
tests do the same and stay Mac-safe.
"""

from __future__ import annotations

import pytest

from voxedge.backends.sherpa.asr import SherpaASRBackend, SherpaASRConfig


def _cap(config: SherpaASRConfig):
    stub = SherpaASRBackend.__new__(SherpaASRBackend)
    stub._config = config
    return stub.concurrency_capability()


def test_default_cap_matches_historical_desktop_value():
    cfg = SherpaASRConfig()
    assert cfg.max_concurrent == 4
    cap = _cap(cfg)
    assert cap.max_concurrent == 4
    assert cap.supports_parallel is True
    assert cap.scaling_mode == "external_managed"
    assert cap.requires_exclusive_device is False


def test_cap_can_be_raised_from_config():
    cap = _cap(SherpaASRConfig(max_concurrent=8))
    assert cap.max_concurrent == 8
    assert cap.supports_parallel is True


def test_cap_of_one_is_still_declared_parallel_capable():
    # supports_parallel describes the recognizer, not the ceiling: sherpa
    # streams stay independent even when only one slot is handed out.
    cap = _cap(SherpaASRConfig(max_concurrent=1))
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is True


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_is_clamped_to_one(bad):
    cfg = SherpaASRConfig(max_concurrent=bad)
    assert cfg.max_concurrent == 1
    assert _cap(cfg).max_concurrent == 1


def test_capability_is_reachable_on_a_constructed_backend():
    # __init__ does not load models, so the real object answers too — this is
    # the path server/main.py takes on an already-loaded backend.
    backend = SherpaASRBackend(SherpaASRConfig(max_concurrent=6))
    assert backend.concurrency_capability().max_concurrent == 6
    # ``profile`` stays accepted for callers that still pass it positionally.
    assert backend.concurrency_capability(None).max_concurrent == 6
