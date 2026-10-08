"""Behavioral contract tests for the TensorRT Whisper encoder boundary."""

import gc
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from voxedge.backends.whisper.encoders import TensorRTEncoder


class FakePool:
    def __init__(self, events, *, fail=False):
        self.events = events
        self.fail = fail
        self.next_ptr = 100
        self.destroyed = 0
        if fail:
            raise RuntimeError("pool construction failed")

    def allocate(self, _size):
        ptr = self.next_ptr
        self.next_ptr += 1
        return ptr

    def copy_htod(self, *_args):
        self.events.append("copy_htod")

    def copy_dtoh(self, _ptr, output):
        self.events.append("copy_dtoh")
        output.fill(7)

    def stream_handle(self):
        return 9

    def synchronize(self):
        self.events.append("sync")

    def destroy(self):
        self.destroyed += 1
        self.events.append("pool.destroy")


class FakeContext:
    def __init__(self, events, *, shape_ok=True, address_ok=True, output_shape=(1, 1500, 512)):
        self.events = events
        self.shape_ok = shape_ok
        self.address_ok = address_ok
        self.output_shape = output_shape
        self.execute_calls = 0

    def set_input_shape(self, *_args):
        self.events.append("set_input_shape")
        return self.shape_ok

    def get_tensor_shape(self, _name):
        return self.output_shape

    def set_tensor_address(self, name, _ptr):
        self.events.append(f"set_tensor_address:{name}")
        return self.address_ok

    def execute_async_v3(self, _stream):
        self.execute_calls += 1
        self.events.append("execute")
        return True


class FakeEngine:
    def __init__(self, events, *, context=True, io=None, input_rank=3, output_shape=(1, 1500, 512)):
        self.events = events
        self.context = context
        self.io = io or [("mel", "INPUT"), ("hidden", "OUTPUT")]
        self.input_rank = input_rank
        self.output_shape = output_shape
        self.num_io_tensors = len(self.io)
        self.ctx = FakeContext(events, output_shape=output_shape)

    def __del__(self):
        self.events.append("engine.del")

    def get_tensor_name(self, index):
        return self.io[index][0]

    def get_tensor_mode(self, name):
        return next(mode for tensor, mode in self.io if tensor == name)

    def get_tensor_shape(self, name):
        if name == "mel" and self.input_rank == 2:
            return (1, 80)
        return (1, 80, 3000) if name == "mel" and self.input_rank == 3 else (1, 80, 1, 3000)

    def get_tensor_dtype(self, _name):
        return "float32"

    def create_execution_context(self):
        return self.ctx if self.context else None


class FakeRuntime:
    def __init__(self, events, engine):
        self.events = events
        self.engine = engine

    def deserialize_cuda_engine(self, _blob):
        self.events.append("deserialize")
        return self.engine

    def __del__(self):
        self.events.append("runtime.del")


class FakeLogger:
    WARNING = 1

    def __init__(self, _level):
        self.events = []

    def __del__(self):
        self.events.append("logger.del")


def install_fakes(monkeypatch, events, *, engine=None, runtime_error=False, pool_error=False):
    engine = engine or FakeEngine(events)

    class Logger(FakeLogger):
        def __init__(self, level):
            super().__init__(level)
            self.events = events

    class Runtime:
        def __init__(self, _logger):
            events.append("runtime.new")

        def deserialize_cuda_engine(self, blob):
            if runtime_error:
                events.append("deserialize")
                return None
            return FakeRuntime(events, engine).deserialize_cuda_engine(blob)

    trt = types.ModuleType("tensorrt")
    trt.Logger = Logger
    trt.Runtime = Runtime
    trt.TensorIOMode = types.SimpleNamespace(INPUT="INPUT", OUTPUT="OUTPUT")
    trt.nptype = lambda dtype: np.float32
    monkeypatch.setitem(sys.modules, "tensorrt", trt)

    import voxedge.backends.jetson._util as util

    pools = []

    def pool_factory(_size):
        pool = FakePool(events, fail=pool_error)
        pools.append(pool)
        return pool

    monkeypatch.setattr(util, "CudaMemoryPool", pool_factory)
    monkeypatch.setattr(util, "arena_size_bytes", lambda _mb: 123)
    return engine, pools


def plan_file(tmp_path: Path):
    plan = tmp_path / "encoder.plan"
    plan.write_bytes(b"fake-plan")
    return plan


def test_construct_run_device_and_legacy_run_preserve_dtype_and_owner(monkeypatch, tmp_path):
    events = []
    engine, pools = install_fakes(monkeypatch, events)
    enc = TensorRTEncoder(plan_file(tmp_path), 10.0)
    ptr, shape, dtype, owner = enc.run_device(np.zeros((80, 3000), dtype=np.float32))
    assert (ptr, shape, dtype, owner) == (101, (1, 1500, 512), np.dtype("float32"), enc)
    assert "copy_dtoh" not in events
    output = enc.run(np.zeros((80, 3000), dtype=np.float32))
    assert output.shape == (1, 1500, 512)
    assert output.dtype == np.dtype("float32")
    assert np.all(output == 7)
    assert events.count("copy_dtoh") == 1
    assert engine.ctx.execute_calls == 2
    enc.close()
    enc.close()
    assert pools[0].destroyed == 1


@pytest.mark.parametrize("failure", ["deserialize", "context", "io", "rank", "nptype", "pool"])
def test_constructor_failure_closes_partial_resources(monkeypatch, tmp_path, failure):
    events = []
    engine = FakeEngine(events, context=failure != "context")
    if failure == "io":
        engine.io = [("mel", "INPUT"), ("hidden", "OUTPUT"), ("extra", "OUTPUT")]
        engine.num_io_tensors = 3
    if failure == "rank":
        engine.input_rank = 2
    install_fakes(monkeypatch, events, engine=engine, runtime_error=failure == "deserialize", pool_error=failure == "pool")
    if failure == "nptype":
        import tensorrt

        tensorrt.nptype = lambda _dtype: (_ for _ in ()).throw(TypeError("bad dtype"))
    with pytest.raises((RuntimeError, TypeError)):
        TensorRTEncoder(plan_file(tmp_path), 10.0)
    # Constructor failure must release pool first when one exists, and must not
    # leave an owned runtime alive ahead of an engine/context.
    assert not any(event == "execute" for event in events)
    if failure == "pool":
        assert "pool.destroy" not in events
    gc.collect()


@pytest.mark.parametrize("which", ["shape", "input_address", "output_address"])
def test_false_trt_contract_result_fails_before_execute(monkeypatch, tmp_path, which):
    events = []
    engine, _pools = install_fakes(monkeypatch, events)
    enc = TensorRTEncoder(plan_file(tmp_path), 10.0)
    if which == "shape":
        engine.ctx.shape_ok = False
    else:
        original = engine.ctx.set_tensor_address

        def address(name, ptr):
            if (which == "input_address" and name == "mel") or (which == "output_address" and name == "hidden"):
                return False
            return original(name, ptr)

        engine.ctx.set_tensor_address = address
    with pytest.raises(RuntimeError):
        enc.run_device(np.zeros((80, 3000), dtype=np.float32))
    assert "execute" not in events
    if which == "shape":
        assert enc._bufs == {}
        assert enc._sizes == (0, 0)
    enc.close()


@pytest.mark.parametrize("output_shape", [(1, -1, 512), (1, 0, 512), (1, True, 512), ()])
def test_non_concrete_output_shape_fails_closed(monkeypatch, tmp_path, output_shape):
    events = []
    engine = FakeEngine(events, output_shape=output_shape)
    install_fakes(monkeypatch, events, engine=engine)
    enc = TensorRTEncoder(plan_file(tmp_path), 10.0)
    with pytest.raises(RuntimeError):
        enc.run_device(np.zeros((80, 3000), dtype=np.float32))
    assert "execute" not in events
    assert enc._bufs == {}
    enc.close()
