"""Encoder execution paths: Hailo HEF, Rockchip RKNN, TensorRT.

Each one takes a mel of shape ``[n_mels, frames]`` and returns encoder hidden
states. They differ only in how the graph is executed and in the input layout
the compiled graph expects.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Encoder(ABC):
    window_s: float
    padding_cutoff_s: float = 0.0

    @abstractmethod
    def run(self, mel: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class HailoEncoder(Encoder):
    """Hailo-8 HEF.

    Hailo grants /dev/hailo0 to a single process, and two VDevices inside one
    process collide the same way, so one VDevice is created per backend
    instance. Anything else holding the device surfaces as
    HAILO_OUT_OF_PHYSICAL_DEVICES (74).

    The shipped HEFs are tiny at a 10 s window and base at 5 s. The mel front
    end crops one second (``padding_cutoff_s``) before padding back out, which
    is Hailo's boundary-hallucination guard, so the usable window is one second
    shorter than the compiled one.
    """

    def __init__(self, hef_path: str | Path, window_s: float, padding_cutoff_s: float = 1.0,
                 timeout_ms: int = 10_000):
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice

        self.window_s = window_s
        self.padding_cutoff_s = padding_cutoff_s
        self._timeout_ms = timeout_ms

        params = VDevice.create_params()
        # Without a scheduling algorithm the network group is never activated and
        # every inference returns HAILO_STREAM_NOT_ACTIVATED (72) — while still
        # filling the output buffer, so the decoder reads zeros and emits a
        # single token per chunk rather than raising.
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdev = VDevice(params)

        self._model = self._vdev.create_infer_model(str(hef_path))
        # BOTH sides, not just the input: the output otherwise stays the HEF's
        # native UINT8 and the float32 buffer handed to it is 4x the expected
        # size (HAILO_INVALID_OPERATION, "768000 is different than expected
        # 192000").
        self._model.input().set_format_type(FormatType.FLOAT32)
        self._model.output().set_format_type(FormatType.FLOAT32)

        # configure() and create_bindings() once, then reuse. Rebuilding the
        # bindings per call is pure overhead on a path whose whole point is a
        # 24 ms encoder. Safe because this backend declares max_concurrent=1.
        self._configured = self._model.configure()
        self._bindings = self._configured.create_bindings()
        self._out = np.zeros(self._model.output().shape, dtype=np.float32)

    def run(self, mel: np.ndarray) -> np.ndarray:
        # HEF encoders take NHWC: [1, 1, frames, n_mels]
        inp = np.ascontiguousarray(mel.T[None, None, :, :], dtype=np.float32)
        self._bindings.input().set_buffer(inp)
        self._bindings.output().set_buffer(self._out)
        # `timeout` is positional in hailo_platform 4.21; there is no
        # `timeout_ms` keyword.
        self._configured.run([self._bindings], self._timeout_ms)
        return self._out.copy()

    def close(self) -> None:
        self._bindings = None
        self._configured = None
        self._model = None
        try:
            self._vdev.release()
        except Exception:
            # release() is known to segfault on some HailoRT builds after a
            # clean run; the data is already out by then.
            pass


class RknnEncoder(Encoder):
    """Rockchip RKNN.

    The exported ONNX rank must match what is fed here. Rockchip's official
    encoder is 3D ``[1, 80, frames]``; an encoder exported for Hailo is 4D NCHW.
    When the element counts match, rknn-lite raises nothing and silently
    reinterprets the buffer — the only symptom is the decoder emitting
    ``(chiming)``-style non-speech annotations. The rank is therefore read off
    the model rather than assumed.

    Binding all three NPU cores on RK3588 measured no faster than two cores on
    RK3576, so ``core_mask`` defaults to automatic.
    """

    def __init__(self, rknn_path: str | Path, window_s: float, all_cores: bool = False):
        from rknnlite.api import RKNNLite

        self.window_s = window_s
        self._rt = RKNNLite()
        if self._rt.load_rknn(str(rknn_path)) != 0:
            raise RuntimeError(f"load_rknn failed: {rknn_path}")
        kwargs = {}
        if all_cores:
            kwargs["core_mask"] = RKNNLite.NPU_CORE_0_1_2
        if self._rt.init_runtime(**kwargs) != 0:
            raise RuntimeError(f"init_runtime failed: {rknn_path}")

    def run(self, mel: np.ndarray) -> np.ndarray:
        return self._rt.inference(inputs=[mel[None, :, :]])[0]

    def close(self) -> None:
        try:
            self._rt.release()
        except Exception:
            pass


class TensorRTEncoder(Encoder):
    """Bare TensorRT — no onnxruntime TRT EP, no torch2trt.

    Build the whisper-base encoder with ``--bf16``, NOT ``--fp16``: the fp16
    build of that graph scores cosine 0.826 against onnxruntime (bf16: 0.9996,
    fp32: 1.0), deterministic run-to-run, so it is kernel selection rather than
    a race. The engine raises no error — the decoder just produces fluent text
    that drifts off-topic. tiny's fp16 encoders are fine, so this is
    model-specific. Passing ``--fp16 --bf16`` together yields an engine
    bit-identical to the pure fp16 one; TRT does not pick bf16 per layer.

    Verify with ``cmp_engine_precision.py`` in the seeed-local-voice repo before
    trusting any engine.
    """

    def __init__(self, plan_path: str | Path, window_s: float, arena_size_mb: int = 16):
        # Set before anything that can fail, so close() has a consistent object
        # to work with no matter how far construction got.
        self._pool = None
        self._runtime = self._logger = self._engine = self._ctx = None
        self._dev: dict[str, int] = {}
        self._bufs: dict[str, int] = {}
        self._sizes: tuple[int, int] = (0, 0)

        import tensorrt as trt

        from voxedge.backends.jetson._util import CudaMemoryPool, arena_size_bytes

        self.window_s = window_s
        # The Logger and Runtime must OUTLIVE the engine and its execution
        # context — NVIDIA's lifetime contract. Building the engine from a
        # temporary `trt.Runtime(...)` leaves both destroyed by the end of the
        # statement, and everything after that is undefined behaviour that
        # happens to work until it does not.
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(plan_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize {plan_path}")
        self._ctx = self._engine.create_execution_context()
        self._in_name = self._engine.get_tensor_name(0)
        self._out_name = self._engine.get_tensor_name(1)
        self._in_rank = len(self._engine.get_tensor_shape(self._in_name))
        # Shared with the other Jetson backends: owns the stream, checks every
        # cudaError_t, and bump-allocates from one arena instead of a cudaMalloc
        # per call. The encoder's two buffers are fixed-size once the window is
        # fixed, so the arena is reset (not freed) between calls.
        self._pool = CudaMemoryPool(arena_size_bytes(arena_size_mb))

    def run(self, mel: np.ndarray) -> np.ndarray:
        inp = mel[None, :, :]
        if self._in_rank == 4:
            inp = inp[:, :, None, :]
        inp = np.ascontiguousarray(inp, dtype=np.float32)
        self._ctx.set_input_shape(self._in_name, inp.shape)
        out = np.empty(tuple(self._ctx.get_tensor_shape(self._out_name)), dtype=np.float32)

        if not self._bufs:
            self._bufs[self._in_name] = self._pool.allocate(inp.nbytes)
            self._bufs[self._out_name] = self._pool.allocate(out.nbytes)
            self._sizes = (inp.nbytes, out.nbytes)
        elif (inp.nbytes, out.nbytes) != self._sizes:
            # The mel is always padded to the full window, so this is constant
            # in practice. Raising beats reusing an undersized device buffer,
            # which would corrupt silently.
            raise RuntimeError(
                f"whisper encoder: buffer size changed {self._sizes} -> "
                f"{(inp.nbytes, out.nbytes)}; window_s must match the engine"
            )
        self._pool.copy_htod(inp, self._bufs[self._in_name])
        self._ctx.set_tensor_address(self._in_name, self._bufs[self._in_name])
        self._ctx.set_tensor_address(self._out_name, self._bufs[self._out_name])
        if not self._ctx.execute_async_v3(self._pool.stream_handle()):
            raise RuntimeError("whisper encoder: TRT execute_async_v3 returned False")
        self._pool.synchronize()
        self._pool.copy_dtoh(self._bufs[self._out_name], out)
        return out

    def close(self) -> None:
        """Release everything, in reverse construction order, exactly once.

        Tolerates a half-constructed object throughout: ``preload`` calls this
        from its failure path, so an AttributeError here would replace the real
        cause — a bad plan, a missing file — with a complaint about a buffer
        dict. Every attribute is read defensively rather than assuming
        ``__init__`` ran to completion.
        """
        cudart = getattr(self, "_cudart", None)
        if cudart is not None:
            for ptr in getattr(self, "_dev", {}).values():
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass
        getattr(self, "_dev", {}).clear()
        getattr(self, "_bufs", {}).clear()

        pool = getattr(self, "_pool", None)
        if pool is not None:
            try:
                pool.destroy()
            except Exception:
                logger.exception("whisper TRT encoder: pool destroy raised; continuing")
        self._pool = None

        # The Runtime must outlive the engine and the context, so it goes last.
        self._ctx = None
        self._engine = None
        self._runtime = None
        self._logger = None


def build_encoder(kind: str, path: str | Path, window_s: float, **kw) -> Encoder:
    if kind == "hailo":
        return HailoEncoder(path, window_s, kw.get("padding_cutoff_s", 1.0),
                            kw.get("timeout_ms", 10_000))
    if kind == "rknn":
        return RknnEncoder(path, window_s, kw.get("all_cores", False))
    if kind == "tensorrt":
        return TensorRTEncoder(path, window_s, kw.get("arena_size_mb", 16))
    raise ValueError(f"unknown encoder kind: {kind!r}")
