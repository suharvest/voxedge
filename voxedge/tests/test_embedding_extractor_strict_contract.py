import sys
import threading
import types

import numpy as np
import pytest

from voxedge.capabilities import embedding_extractor as mod


class _Cuda:
    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    def __init__(self, *, execute=True):
        self.execute = execute
        self.next_handle = 10

    def cudaMalloc(self, _size):
        self.next_handle += 1
        return 0, self.next_handle

    def cudaStreamCreate(self):
        return 0, 77

    def cudaMemcpyAsync(self, *_args):
        return 0

    def cudaStreamSynchronize(self, _stream):
        return 0

    def cudaFree(self, _ptr):
        return 0

    def cudaStreamDestroy(self, _stream):
        return 0


class _Context:
    def __init__(self, *, execute=True, entered=None, release=None):
        self.execute = execute
        self.entered = entered
        self.release = release
        self.calls = []

    def set_input_shape(self, _name, shape):
        self.calls.append(("shape", shape))
        return True

    def get_tensor_shape(self, _name):
        return (1, 192)

    def set_tensor_address(self, name, address):
        self.calls.append(("address", name, address))
        return True

    def execute_async_v3(self, _stream):
        self.calls.append("execute")
        if self.entered is not None:
            self.entered.set()
            self.release.wait(timeout=2)
        return self.execute


class _Engine:
    num_io_tensors = 2

    def __init__(self, ctx=None, profile=((1, 40, 80), (1, 60, 80), (1, 80, 80))):
        self.ctx = ctx or _Context()
        self.profile = profile

    def get_tensor_name(self, i):
        return ("features", "embedding")[i]

    def get_tensor_mode(self, name):
        return _TRT.TensorIOMode.INPUT if name == "features" else _TRT.TensorIOMode.OUTPUT

    def get_tensor_shape(self, name):
        return (1, -1, 80) if name == "features" else (1, 192)

    def get_tensor_dtype(self, _name):
        return "float32"

    def get_tensor_profile_shape(self, _name, _profile):
        return self.profile

    def create_execution_context(self):
        return self.ctx


class _Runtime:
    last = None

    def __init__(self, logger):
        self.logger = logger
        self.engine = None
        _Runtime.last = self

    def deserialize_cuda_engine(self, _blob):
        return self.engine or type(self).engine


class _TRT:
    class TensorIOMode:
        INPUT = object()
        OUTPUT = object()

    class Logger:
        WARNING = 1

        def __init__(self, level):
            self.level = level

    Runtime = _Runtime

    @staticmethod
    def nptype(dtype):
        return np.dtype(dtype)


def _install_runtime(monkeypatch, engine, *, fbank=True):
    _Runtime.last = None
    _Runtime.engine = engine
    monkeypatch.setitem(sys.modules, "tensorrt", _TRT)
    monkeypatch.setitem(sys.modules, "cuda", types.SimpleNamespace(cudart=_Cuda()))
    if fbank:
        monkeypatch.setitem(sys.modules, "kaldi_native_fbank", types.ModuleType("kaldi_native_fbank"))


def _make_plan(tmp_path):
    plan = tmp_path / "campplus.plan"
    plan.write_bytes(b"fake-plan")
    return str(plan)


def test_strict_retains_runtime_logger_and_reports_actual_profile(monkeypatch, tmp_path):
    _install_runtime(monkeypatch, _Engine())
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    assert backend.is_ready is True
    assert backend.frame_bounds == (40, 80)
    assert backend._runtime is _Runtime.last
    assert backend._logger.level == _TRT.Logger.WARNING


def test_strict_rejects_bad_profile_and_default_keeps_none(monkeypatch, tmp_path):
    engine = _Engine(profile=((1, -1, 80), (1, 1, 80), (1, -1, 80)))
    _install_runtime(monkeypatch, engine)
    with pytest.raises(mod.EmbeddingBackendError, match="profile"):
        mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)

    legacy = mod.JetsonCampplusTRT(_make_plan(tmp_path))
    assert legacy.extract(np.ones(1600, dtype=np.float32), 16000) is None


@pytest.mark.parametrize("frames, error", [(39, mod.EmbeddingInputError), (81, mod.EmbeddingInputError)])
def test_strict_checks_min_and_max_frame_boundaries(monkeypatch, tmp_path, frames, error):
    _install_runtime(monkeypatch, _Engine())
    monkeypatch.setattr(mod, "compute_fbank", lambda _audio, _sr: np.zeros((frames, 80), np.float32))
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    with pytest.raises(error, match="frameT"):
        backend.extract(np.ones(4, dtype=np.float32), 16000)


def test_strict_shape_false_raises_and_legacy_returns_none(monkeypatch, tmp_path):
    class FalseContext(_Context):
        def set_input_shape(self, _name, _shape):
            return False

    _install_runtime(monkeypatch, _Engine(ctx=FalseContext()))
    monkeypatch.setattr(mod, "compute_fbank", lambda _audio, _sr: np.zeros((40, 80), np.float32))
    strict = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    with pytest.raises(mod.EmbeddingBackendError, match="shape"):
        strict.extract(np.ones(4, dtype=np.float32), 16000)

    legacy = mod.JetsonCampplusTRT(_make_plan(tmp_path))
    assert legacy.extract(np.ones(4, dtype=np.float32), 16000) is None


@pytest.mark.parametrize("frames", [40, 80])
def test_strict_accepts_profile_boundaries(monkeypatch, tmp_path, frames):
    _install_runtime(monkeypatch, _Engine())
    monkeypatch.setattr(mod, "compute_fbank", lambda _audio, _sr: np.zeros((frames, 80), np.float32))
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    backend._infer = lambda _feat: np.ones(192, dtype=np.float32)
    result = backend.extract(np.ones(4, dtype=np.float32), 16000)
    assert result.shape == (192,)
    assert np.isclose(np.linalg.norm(result), 1.0)


@pytest.mark.parametrize("bad", [np.zeros(191, dtype=np.float32), np.full(192, np.nan, dtype=np.float32)])
def test_strict_rejects_wrong_or_nonfinite_embedding(monkeypatch, tmp_path, bad):
    _install_runtime(monkeypatch, _Engine())
    monkeypatch.setattr(mod, "compute_fbank", lambda _audio, _sr: np.zeros((40, 80), np.float32))
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    backend._infer = lambda _feat: bad
    with pytest.raises(mod.EmbeddingBackendError, match="finite|shape"):
        backend.extract(np.ones(4, dtype=np.float32), 16000)


def test_strict_stable_norm_handles_large_finite_embedding(monkeypatch, tmp_path):
    _install_runtime(monkeypatch, _Engine())
    monkeypatch.setattr(mod, "compute_fbank", lambda _audio, _sr: np.zeros((40, 80), np.float32))
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    backend._infer = lambda _feat: np.full(192, 1e30, dtype=np.float32)
    result = backend.extract(np.ones(4, dtype=np.float32), 16000)
    assert np.isfinite(result).all()
    assert np.isclose(np.linalg.norm(result.astype(np.float64)), 1.0)


class _CleanupCuda(_Cuda):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.fail_free = True
        self.fail_destroy = True

    def cudaFree(self, ptr):
        self.calls.append(("free", ptr))
        return 1 if self.fail_free else 0

    def cudaStreamDestroy(self, stream):
        self.calls.append(("destroy", stream))
        return 1 if self.fail_destroy else 0


def test_strict_cleanup_attempts_all_resources_and_preserves_execute_error(monkeypatch, tmp_path):
    ctx = _Context(execute=False)
    _install_runtime(monkeypatch, _Engine(ctx=ctx))
    monkeypatch.setattr(mod, "compute_fbank", lambda _audio, _sr: np.zeros((40, 80), np.float32))
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    cleanup = _CleanupCuda()
    backend._cudart = cleanup
    with pytest.raises(mod.EmbeddingBackendError, match="execute_async_v3"):
        backend.extract(np.ones(4, dtype=np.float32), 16000)
    assert cleanup.calls == [("free", 11), ("free", 12), ("destroy", 77)]


def test_strict_cleanup_failure_is_typed_when_execution_succeeds(monkeypatch, tmp_path):
    _install_runtime(monkeypatch, _Engine())
    monkeypatch.setattr(mod, "compute_fbank", lambda _audio, _sr: np.zeros((40, 80), np.float32))
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    cleanup = _CleanupCuda()
    backend._cudart = cleanup
    with pytest.raises(mod.EmbeddingBackendError, match="cleanup"):
        backend._infer(np.zeros((1, 40, 80), np.float32))
    assert cleanup.calls == [("free", 11), ("free", 12), ("destroy", 77)]


def test_strict_scalar_audio_is_input_error(monkeypatch, tmp_path):
    _install_runtime(monkeypatch, _Engine())
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    with pytest.raises(mod.EmbeddingInputError, match="1-D"):
        backend.extract(np.array(1, dtype=np.float32), 16000)


def test_default_nonpositive_min_frames_remains_legacy(monkeypatch, tmp_path):
    _install_runtime(monkeypatch, _Engine())
    assert mod.JetsonCampplusTRT(_make_plan(tmp_path), min_frames=0).is_ready is True
    with pytest.raises(ValueError, match="min_frames"):
        mod.JetsonCampplusTRT(_make_plan(tmp_path), min_frames=0, strict=True)


def test_shared_context_is_serialized(monkeypatch, tmp_path):
    entered = threading.Event()
    release = threading.Event()
    ctx = _Context(entered=entered, release=release)
    _install_runtime(monkeypatch, _Engine(ctx=ctx))
    monkeypatch.setattr(mod, "compute_fbank", lambda _audio, _sr: np.zeros((40, 80), np.float32))
    backend = mod.JetsonCampplusTRT(_make_plan(tmp_path), strict=True)
    # The fake CUDA transport does not copy a synthetic embedding back; this
    # test only asserts shared-context exclusion, so retain legacy output rules.
    backend._strict = False
    results = []
    first = threading.Thread(target=lambda: results.append(backend.extract(np.ones(4), 16000)))
    second = threading.Thread(target=lambda: results.append(backend.extract(np.ones(4), 16000)))
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    # The second call cannot touch the context while the first execute holds the lock.
    assert len(ctx.calls) == 4
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert len(results) == 2
    assert ctx.calls.count("execute") == 2
