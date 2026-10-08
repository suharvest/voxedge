"""Speaker-embedding extraction backends — `EmbeddingExtractor` abstraction.

Per spec `diarization-capability.md` §7. The clustering layer is device-agnostic
(numpy); only the "emit a vector" step is device-specific. This module provides
the abstraction plus the **Jetson TRT engine** backend, which runs the CAM++
(3D-Speaker campplus) model as a resident TensorRT engine instead of
onnxruntime.

  EmbeddingExtractor.extract(audio, sr) -> np.ndarray   # 192-d, L2-normalized

Backends:
  * ``JetsonCampplusTRT`` — Python fbank front-end (kaldi-native-fbank) →
    TRT engine ``[1,T,80] -> [1,192]`` → L2-norm.  CPU sherpa fallback lives in
    ``speaker_embedding.SpeakerEmbedder`` and is wrapped separately.

Env-free per voxedge convention: the engine file path is injected at
construction; flag gating / path resolution stay in the product layer
(env ``DIAR_CAMPPLUS_ENGINE_FILE`` points at the ``.plan`` file, not a dir).

Heavy deps (``tensorrt``, ``cuda-python``, ``kaldi_native_fbank``) are imported
lazily so this module loads on any image.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "campplus_sv_zh_en_3dspeaker"
EMBEDDING_DIM = 192
_TARGET_SR = 16000


class EmbeddingInputError(ValueError):
    """The requested audio cannot be represented by the CAM++ profile."""


class EmbeddingBackendError(RuntimeError):
    """The CAM++ TensorRT backend is unavailable or failed during inference."""


class EmbeddingExtractor(ABC):
    """Produce one L2-normalized speaker embedding per utterance."""

    @abstractmethod
    def extract(self, audio: np.ndarray, sr: int) -> "np.ndarray | None":
        """``audio``: mono float32 in [-1, 1]. Returns 192-d L2-norm or None."""
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM


# ── fbank front-end (kaldi-native-fbank, mirrors 3D-Speaker / sherpa-onnx) ────

def compute_fbank(samples: np.ndarray, sr: int = _TARGET_SR) -> np.ndarray:
    """80-dim kaldi fbank ``[T, 80]`` with global-mean CMN.

    Mirrors sherpa-onnx's speaker front-end *exactly* (the TRT engine was
    exported from the sherpa/3D-Speaker ONNX, so the query front-end must match
    sherpa's config, not the raw training config). Verified to parity cosine
    against ``sherpa_onnx.SpeakerEmbeddingExtractor`` on-device.

    Critical knobs (do NOT change without re-running the parity gate):
      * waveform fed in [-1, 1] — **no x32768 scaling** (normalize_samples=1)
      * ``snip_edges = False``
      * ``high_freq = -400.0`` (sherpa override; NOT 0/Nyquist)
      * ``preemph_coeff = 0.97``, povey window, power fbank, log
      * CMN = global per-bin mean subtraction, no variance division
    """
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = float(sr)
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.frame_opts.dither = 0.0
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0
    opts.mel_opts.is_librosa = False
    opts.use_energy = False
    opts.use_log_fbank = True
    opts.use_power = True

    fb = knf.OnlineFbank(opts)
    x = np.ascontiguousarray(samples, dtype=np.float32)   # [-1, 1], no scaling
    fb.accept_waveform(sr, x.tolist())
    fb.input_finished()
    frames = [fb.get_frame(i) for i in range(fb.num_frames_ready)]
    if not frames:
        return np.zeros((0, 80), dtype=np.float32)
    feat = np.array(frames, dtype=np.float32)              # [T, 80]
    feat = feat - feat.mean(axis=0, keepdims=True)          # global-mean CMN
    return feat


class JetsonCampplusTRT(EmbeddingExtractor):
    """CAM++ via a resident TensorRT engine (dynamic time profile).

    Engine I/O: input ``[1, T, 80]`` fbank → output ``[1, 192]`` embedding.
    The default mode is lazy and preserves the historical ``None`` on failure.
    ``strict=True`` makes initialization and inference failures explicit.
    """

    def __init__(self, engine_path: str, min_frames: int = 40, *, strict: bool = False):
        self._engine_path = engine_path
        self._min_frames = min_frames
        self._strict = strict
        self._lock = threading.Lock()
        self._engine = None
        self._ctx = None
        self._logger = None
        self._runtime = None
        self._in_name = None
        self._out_name = None
        self._trt = None
        self._cudart = None
        self._frame_bounds = None
        self._failed = False

        if strict and min_frames < 1:
            raise ValueError("min_frames must be positive")
        if strict:
            try:
                import kaldi_native_fbank  # noqa: F401
            except Exception as exc:
                raise EmbeddingBackendError(
                    "strict CAM++ requires kaldi_native_fbank"
                ) from exc
            self._ensure()

    def _ck(self, ret):
        err = ret[0] if isinstance(ret, tuple) else ret
        if int(err) != 0:
            raise RuntimeError(f"CUDA error {err}")
        return ret[1] if isinstance(ret, tuple) and len(ret) > 1 else None

    def _ensure(self) -> bool:
        if self._ctx is not None:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._ctx is not None:
                return True
            if self._failed:
                return False
            try:
                import tensorrt as trt
                from cuda import cudart

                # Runtime and Logger must outlive the engine and execution
                # context (see the Whisper TRT encoder lifecycle).
                logger_trt = trt.Logger(trt.Logger.WARNING)
                runtime = trt.Runtime(logger_trt)
                with open(self._engine_path, "rb") as f:
                    engine = runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    raise EmbeddingBackendError("deserialize_cuda_engine returned None")
                ctx = engine.create_execution_context()
                if ctx is None:
                    raise EmbeddingBackendError("failed to create CAM++ execution context")
                inputs, outputs = [], []
                for i in range(engine.num_io_tensors):
                    nm = engine.get_tensor_name(i)
                    mode = engine.get_tensor_mode(nm)
                    if mode == trt.TensorIOMode.INPUT:
                        inputs.append(nm)
                    elif mode == trt.TensorIOMode.OUTPUT:
                        outputs.append(nm)
                if self._strict and (len(inputs) != 1 or len(outputs) != 1):
                    raise EmbeddingBackendError(
                        f"CAM++ expects one input/output, got {inputs}/{outputs}"
                    )
                if not inputs or not outputs:
                    raise EmbeddingBackendError("CAM++ engine has no input/output tensors")
                in_name, out_name = inputs[0], outputs[0]
                if self._strict:
                    in_shape = tuple(engine.get_tensor_shape(in_name))
                    out_shape = tuple(engine.get_tensor_shape(out_name))
                    if len(in_shape) != 3 or in_shape[0] != 1 or in_shape[2] != 80:
                        raise EmbeddingBackendError(
                            f"CAM++ input must be [1,T,80], got {in_shape}"
                        )
                    if len(out_shape) != 2 or out_shape[0] != 1 or out_shape[1] != EMBEDDING_DIM:
                        raise EmbeddingBackendError(
                            f"CAM++ output must be [1,192], got {out_shape}"
                        )
                    in_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(in_name)))
                    out_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(out_name)))
                    if in_dtype != np.dtype(np.float32) or out_dtype != np.dtype(np.float32):
                        raise EmbeddingBackendError(
                            f"CAM++ tensors must be float32, got {in_dtype}/{out_dtype}"
                        )
                    min_t, max_t = self._profile_bounds(engine, in_name, in_shape)
                    if min_t > max_t:
                        raise EmbeddingBackendError(f"invalid CAM++ frame profile {min_t}..{max_t}")
                    if self._min_frames > max_t:
                        raise EmbeddingBackendError(
                            f"min_frames={self._min_frames} exceeds CAM++ profile max T={max_t}"
                        )
                else:
                    # Metadata is best-effort in legacy mode: an old engine
                    # lacking profile introspection must keep its None-on-
                    # failure behavior.
                    try:
                        min_t, max_t = self._profile_bounds(
                            engine, in_name, tuple(engine.get_tensor_shape(in_name))
                        )
                    except Exception:
                        min_t, max_t = self._min_frames, None
                self._trt, self._cudart = trt, cudart
                self._logger, self._runtime = logger_trt, runtime
                self._engine, self._ctx = engine, ctx
                self._in_name, self._out_name = in_name, out_name
                self._frame_bounds = (max(min_t, self._min_frames), max_t)
                logger.info("JetsonCampplusTRT loaded (%s, in=%s out=%s).",
                            self._engine_path, in_name, out_name)
            except Exception as exc:
                self._failed = True
                logger.exception("Failed to load CAM++ TRT engine; disabled.")
                if self._strict:
                    if isinstance(exc, EmbeddingBackendError):
                        raise
                    raise EmbeddingBackendError("failed to initialize CAM++ TRT backend") from exc
                return False
        return True

    @staticmethod
    def _profile_bounds(engine, in_name, in_shape):
        time_dim = in_shape[1]
        if time_dim > 0:
            return int(time_dim), int(time_dim)
        try:
            profile = engine.get_tensor_profile_shape(in_name, 0)
        except Exception as exc:
            raise EmbeddingBackendError("CAM++ dynamic input has no readable profile") from exc
        if not isinstance(profile, (tuple, list)) or len(profile) != 3:
            raise EmbeddingBackendError(f"invalid CAM++ profile for {in_name}: {profile!r}")
        mins, _, maxs = (tuple(x) for x in profile)
        if (len(mins) != 3 or len(maxs) != 3 or
                mins[0] != 1 or maxs[0] != 1 or mins[2] != 80 or maxs[2] != 80):
            raise EmbeddingBackendError(f"invalid CAM++ profile shapes: {profile!r}")
        min_t, max_t = int(mins[1]), int(maxs[1])
        if min_t < 1 or max_t < min_t:
            raise EmbeddingBackendError(f"invalid CAM++ profile T bounds: {min_t}..{max_t}")
        return min_t, max_t

    @property
    def is_ready(self) -> bool:
        return self._ensure()

    def ready(self) -> bool:
        """Compatibility alias retained for existing callers."""
        return self.is_ready

    @property
    def frame_bounds(self):
        self._ensure()
        return self._frame_bounds

    def _infer(self, feat: np.ndarray) -> np.ndarray:
        with self._lock:
            cudart = self._cudart
            ctx = self._ctx
            shape_result = ctx.set_input_shape(self._in_name, tuple(feat.shape))
            if shape_result is False:
                raise EmbeddingBackendError(
                    f"CAM++ rejected input shape {tuple(feat.shape)}"
                )
            out_shape = tuple(ctx.get_tensor_shape(self._out_name))
            if self._strict and out_shape != (1, EMBEDDING_DIM):
                raise EmbeddingBackendError(f"CAM++ resolved output shape is {out_shape}")
            inp = np.ascontiguousarray(feat, dtype=np.float32)
            out = np.empty(out_shape, dtype=np.float32)
            d_in = d_out = stream = None
            primary_error = None
            try:
                d_in = self._ck(cudart.cudaMalloc(inp.nbytes))
                d_out = self._ck(cudart.cudaMalloc(out.nbytes))
                stream = self._ck(cudart.cudaStreamCreate())
                if d_in is None or d_out is None or stream is None:
                    raise EmbeddingBackendError("CUDA allocation returned no handle")
                self._ck(cudart.cudaMemcpyAsync(
                    d_in, inp.ctypes.data, inp.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
                if ctx.set_tensor_address(self._in_name, int(d_in)) is False:
                    raise EmbeddingBackendError("CAM++ rejected input tensor address")
                if ctx.set_tensor_address(self._out_name, int(d_out)) is False:
                    raise EmbeddingBackendError("CAM++ rejected output tensor address")
                if not ctx.execute_async_v3(stream):
                    raise EmbeddingBackendError("execute_async_v3 failed")
                self._ck(cudart.cudaMemcpyAsync(
                    out.ctypes.data, d_out, out.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
                self._ck(cudart.cudaStreamSynchronize(stream))
            except BaseException as exc:
                primary_error = exc
            finally:
                cleanup_errors = []
                for resource, release in (
                    (d_in, cudart.cudaFree),
                    (d_out, cudart.cudaFree),
                    (stream, cudart.cudaStreamDestroy),
                ):
                    if resource is not None:
                        try:
                            self._ck(release(resource))
                        except BaseException as exc:
                            cleanup_errors.append(exc)
                if primary_error is None and cleanup_errors:
                    raise EmbeddingBackendError(
                        "CAM++ CUDA resource cleanup failed"
                    ) from cleanup_errors[0]
            if primary_error is not None:
                raise primary_error
            return out.reshape(-1).astype(np.float32)

    def extract(self, audio: np.ndarray, sr: int) -> "np.ndarray | None":
        if not self._ensure():
            return None
        try:
            if audio is None:
                raise EmbeddingInputError("CAM++ received empty audio")
            samples = np.asarray(audio, dtype=np.float32)
            if samples.ndim != 1:
                raise EmbeddingInputError(
                    f"CAM++ audio must be a mono 1-D array, got shape {samples.shape}"
                )
            if samples.size == 0:
                raise EmbeddingInputError("CAM++ received empty audio")
            feat = compute_fbank(samples, sr)
            if feat.ndim != 2 or feat.shape[1] != 80:
                raise EmbeddingBackendError(f"CAM++ fbank must be [T,80], got {feat.shape}")
            min_t, max_t = self._frame_bounds or (self._min_frames, None)
            if feat.shape[0] < min_t:
                raise EmbeddingInputError(
                    f"CAM++ input has frameT={feat.shape[0]}, below minimum frameT={min_t}"
                )
            if max_t is not None and feat.shape[0] > max_t:
                raise EmbeddingInputError(
                    f"CAM++ input has frameT={feat.shape[0]}, above maximum frameT={max_t}"
                )
            emb = self._infer(feat[None])          # [1,T,80] -> [192]
            if self._strict:
                if emb.shape != (EMBEDDING_DIM,) or not np.all(np.isfinite(emb)):
                    raise EmbeddingBackendError("CAM++ output must be finite with shape [192]")
                norm = float(np.linalg.norm(emb.astype(np.float64)))
                if not np.isfinite(norm) or norm <= 0:
                    raise EmbeddingBackendError("CAM++ output has invalid norm")
                emb = (emb.astype(np.float64) / norm).astype(np.float32)
                post_norm = float(np.linalg.norm(emb.astype(np.float64)))
                if not np.isfinite(post_norm) or not np.isclose(post_norm, 1.0, rtol=1e-5, atol=1e-5):
                    raise EmbeddingBackendError("CAM++ output is not unit-normalized")
            else:
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
            return emb
        except Exception as exc:
            if self._strict:
                if isinstance(exc, (EmbeddingInputError, EmbeddingBackendError)):
                    raise
                raise EmbeddingBackendError("CAM++ inference failed") from exc
            logger.exception("JetsonCampplusTRT.extract failed.")
            return None
