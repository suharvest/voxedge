"""Regression: MossTtsNanoBackend must expose ``model_id``.

The product's ``_request_voice_kwargs`` (app/main.py) reads ``backend.model_id``
on every /tts request. MOSS shipped without it, so after the voxedge migration
every /tts/stream returned 500
(``'MossTtsNanoBackend' object has no attribute 'model_id'``) even though the
backend loaded fine — surfaced on real hardware during the profiling sweep
(#39, orin-nano). All TTS backends carry a ``model_id`` (config field + property,
same as matcha_trt / qwen3_trt).
"""

from voxedge.backends.jetson.moss_tts_nano import (
    MossTtsNanoBackend,
    MossTtsNanoConfig,
)


def _backend(config: MossTtsNanoConfig) -> MossTtsNanoBackend:
    # __init__ wires worker state but spawns nothing; the model_id property
    # only reads self._config. Bypass it for a pure, hardware-free probe.
    b = MossTtsNanoBackend.__new__(MossTtsNanoBackend)
    b._config = config
    return b


def test_moss_model_id_default():
    assert _backend(MossTtsNanoConfig()).model_id == "moss-tts-nano"


def test_moss_model_id_from_config():
    assert _backend(MossTtsNanoConfig(model_id="moss-custom")).model_id == "moss-custom"


def test_moss_model_id_is_a_property_not_missing():
    # The exact failure mode: attribute access must not raise.
    assert hasattr(_backend(MossTtsNanoConfig()), "model_id")
