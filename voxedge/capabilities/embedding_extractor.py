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
    Lazy, thread-safe, sticky on hard failure; ``extract`` never raises.
    """

    def __init__(self, engine_path: str, min_frames: int = 40):
        self._engine_path = engine_path
        self._min_frames = min_frames
        self._lock = threading.Lock()
        self._engine = None
        self._ctx = None
        self._in_name = None
        self._out_name = None
        self._trt = None
        self._cudart = None
        self._failed = False

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

                logger_trt = trt.Logger(trt.Logger.WARNING)
                with open(self._engine_path, "rb") as f, trt.Runtime(logger_trt) as rt:
                    engine = rt.deserialize_cuda_engine(f.read())
                if engine is None:
                    raise RuntimeError("deserialize_cuda_engine returned None")
                ctx = engine.create_execution_context()
                in_name = out_name = None
                for i in range(engine.num_io_tensors):
                    nm = engine.get_tensor_name(i)
                    if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                        in_name = nm
                    else:
                        out_name = nm
                self._trt, self._cudart = trt, cudart
                self._engine, self._ctx = engine, ctx
                self._in_name, self._out_name = in_name, out_name
                logger.info("JetsonCampplusTRT loaded (%s, in=%s out=%s).",
                            self._engine_path, in_name, out_name)
            except Exception:
                self._failed = True
                logger.exception("Failed to load CAM++ TRT engine; disabled.")
                return False
        return True

    def ready(self) -> bool:
        return self._ensure()

    def _infer(self, feat: np.ndarray) -> np.ndarray:
        cudart = self._cudart
        ctx = self._ctx
        ctx.set_input_shape(self._in_name, tuple(feat.shape))
        out_shape = tuple(ctx.get_tensor_shape(self._out_name))
        inp = np.ascontiguousarray(feat, dtype=np.float32)
        out = np.empty(out_shape, dtype=np.float32)
        d_in = self._ck(cudart.cudaMalloc(inp.nbytes))
        d_out = self._ck(cudart.cudaMalloc(out.nbytes))
        stream = self._ck(cudart.cudaStreamCreate())
        try:
            self._ck(cudart.cudaMemcpyAsync(
                d_in, inp.ctypes.data, inp.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
            ctx.set_tensor_address(self._in_name, int(d_in))
            ctx.set_tensor_address(self._out_name, int(d_out))
            if not ctx.execute_async_v3(stream):
                raise RuntimeError("execute_async_v3 failed")
            self._ck(cudart.cudaMemcpyAsync(
                out.ctypes.data, d_out, out.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
            self._ck(cudart.cudaStreamSynchronize(stream))
        finally:
            cudart.cudaFree(d_in)
            cudart.cudaFree(d_out)
            cudart.cudaStreamDestroy(stream)
        return out.reshape(-1).astype(np.float32)

    def extract(self, audio: np.ndarray, sr: int) -> "np.ndarray | None":
        if not self._ensure():
            return None
        if audio is None or len(audio) == 0:
            return None
        try:
            feat = compute_fbank(np.asarray(audio, dtype=np.float32), sr)
            if feat.shape[0] < self._min_frames:
                return None
            emb = self._infer(feat[None])          # [1,T,80] -> [192]
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception:
            logger.exception("JetsonCampplusTRT.extract failed.")
            return None
