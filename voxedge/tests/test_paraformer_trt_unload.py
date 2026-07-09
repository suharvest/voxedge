"""ParaformerTRTBackend.unload() — hot-swap resource release.

paraformer_trt was the only Jetson backend that declared no hot-reload support,
so the manager returned ``hot_reload_not_supported`` when switching *away* from
it. These tests lock in that it now (a) advertises hot-reload and (b) releases
its shared engines idempotently. No CUDA/TensorRT needed — unload's ``from cuda
import cudart`` fails gracefully off-device and the rest is pure ref-clearing.
"""

from voxedge.backends.jetson.paraformer_trt import ParaformerTRTBackend


def test_supports_hot_reload_flag():
    assert ParaformerTRTBackend.supports_hot_reload is True


def test_unload_releases_engines_and_marks_not_ready():
    b = ParaformerTRTBackend()
    # Simulate a preloaded backend without touching CUDA.
    b._engines = {"enc": object(), "dec": object()}
    b._enc_ort_session = object()
    b._enc_profile_ranges = [(1, 2, 3)]
    b._ready = True

    b.unload()

    assert b._engines == {}
    assert b._enc_ort_session is None
    assert b._enc_profile_ranges == []
    assert b.is_ready() is False


def test_unload_is_idempotent_and_safe_when_never_preloaded():
    b = ParaformerTRTBackend()
    # Fresh backend: no engines, not ready — must no-op, not raise.
    b.unload()
    assert b.is_ready() is False
    # Second call after a real unload must also be a no-op.
    b._engines = {"enc": object(), "dec": object()}
    b._ready = True
    b.unload()
    b.unload()
    assert b._engines == {}
    assert b.is_ready() is False
