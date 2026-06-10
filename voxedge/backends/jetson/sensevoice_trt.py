"""SenseVoice offline ASR — encoder + CTC via a standalone TensorRT engine.

SenseVoice-small is an encoder+CTC model: a single forward over LFR features
yields ``[1, T, 25055]`` CTC logits. The 4 SenseVoice prompt embeddings
(language/event/speech/textnorm) are prepended as the first 4 frames; the engine
is built from a fixed-shape ONNX (``T_FIXED=344``). On Jetson the engine is a
pure TensorRT ``.plan`` driven by the tensorrt + cuda-python runtime — the slim
Jetson image's onnxruntime is CPU-only, so we do NOT use ORT here.

RK3588 / Jetson fp16 NOTE: the block-48 FFN overflows fp16 on Chinese
activations. The engine MUST be built from the **activation-rescaled** ONNX
(``...scaled.fixed.onnx``, K=8, math-exact); plain fp16 yields all-NaN on zh.
Verified on real Jetson (orin-nano, TRT 10.4): zh + en both decode correctly.

Front end matches the lovemefan/sherpa export (identical to the RK backend):
80-dim kaldi fbank (dither=0, hamming, snip_edges) -> LFR(m=7,n=6)=560 -> CMVN
(am.mvn) -> prepend 4 prompt frames. CTC greedy + sentencepiece, strip <|...|>.

env-free per voxedge convention: paths injected via SenseVoiceTRTConfig.
``tensorrt`` / ``cuda`` / ``kaldi_native_fbank`` / ``sentencepiece`` imports stay
method-local so this module imports without the optional jetson extra.
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from voxedge.backends.base import ASRBackend, ASRCapability, ASRStream, TranscriptionResult
from voxedge.engine.concurrency_capability import ConcurrencyCapability

logger = logging.getLogger(__name__)

T_FIXED = 344
LFR_DIM = 560
BLANK_ID = 0
VOCAB = 25055
_LANG_IDS = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12}
_TEXTNORM_IDS = {"withitn": 14, "woitn": 15}
_LANGUAGE_MAP = {
    "auto": "auto", "chinese": "zh", "mandarin": "zh", "english": "en",
    "japanese": "ja", "korean": "ko", "cantonese": "yue", "yue": "yue",
    "zh": "zh", "zh-cn": "zh", "zh-tw": "zh", "en": "en", "en-us": "en",
    "en-gb": "en", "ja": "ja", "ko": "ko",
}


def _map_language(language: str) -> str:
    return _LANGUAGE_MAP.get((language or "auto").lower(), "auto")


@dataclass
class SenseVoiceTRTConfig:
    """Construction-time config (no os.environ reads inside the backend).

    ``engine`` is the prebuilt TensorRT ``.plan`` (built per device/TRT version
    from the rescaled fixed ONNX). ``model_dir`` holds the decode assets
    (am.mvn, embedding.npy, the sentencepiece model).
    """

    engine: str = "/opt/models/sensevoice-trt/sensevoice.plan"
    model_dir: str = "/opt/models/sensevoice-trt"
    bpe_model: Optional[str] = None  # default: <model_dir>/chn_jpn_yue_eng_ko_spectok.bpe.model

    def __post_init__(self) -> None:
        if self.bpe_model is None:
            self.bpe_model = os.path.join(self.model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")


class SenseVoiceTRTBackend(ASRBackend):
    """SenseVoice offline ASR on the Jetson GPU via a standalone TensorRT engine."""

    # Opt into the generic offline→streaming adapter (OfflineAccumulateStream):
    # accumulate audio, transcribe the whole utterance on finalize, endpointing
    # via the OVS server-side VAD. Unlocks /asr/stream + /v2v/stream.
    supports_offline_streaming = True

    def __init__(self, config: Optional[SenseVoiceTRTConfig] = None):
        self._cfg = config or SenseVoiceTRTConfig()
        self._engine = None
        self._ctx = None
        self._in_name = None
        self._out_name = None
        self._out_shape = (1, T_FIXED, VOCAB)
        self._cmvn_add = None
        self._cmvn_scale = None
        self._emb = None
        self._sp = None
        self._lock = threading.Lock()  # single shared context; offline is serialized
        self._ready = False

    @property
    def name(self) -> str:
        return "sensevoice_trt"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.MULTI_LANGUAGE}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready and self._ctx is not None

    @classmethod
    def concurrency_capability(cls, profile=None):
        # Single shared execution context, serialized by _lock. Offline /asr is
        # request-at-a-time; keep it conservative.
        return ConcurrencyCapability.default()

    # ------------------------------------------------------------------
    # Preload
    # ------------------------------------------------------------------

    def preload(self) -> None:
        import tensorrt as trt
        import sentencepiece as spm

        cfg = self._cfg
        if not os.path.isfile(cfg.engine):
            raise FileNotFoundError(f"SenseVoice TRT engine not found: {cfg.engine!r}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(cfg.engine, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"deserialize_cuda_engine failed: {cfg.engine!r}")
        self._ctx = self._engine.create_execution_context()

        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        self._in_name = next(n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self._out_name = next(n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self._ctx.set_input_shape(self._in_name, (1, T_FIXED, LFR_DIM))
        self._out_shape = tuple(self._ctx.get_tensor_shape(self._out_name))

        self._cmvn_add, self._cmvn_scale = self._load_cmvn(os.path.join(cfg.model_dir, "am.mvn"))
        self._emb = np.load(os.path.join(cfg.model_dir, "embedding.npy"))
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(cfg.bpe_model)

        self._ready = True
        logger.info("SenseVoice TRT backend ready (engine=%s, out=%s).", cfg.engine, self._out_shape)

    def unload(self) -> None:
        self._ctx = None
        self._engine = None
        self._ready = False

    # ------------------------------------------------------------------
    # Transcribe (offline)
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("SenseVoice TRT backend not ready — call preload() first")
        return self.transcribe_array(self._decode_audio(audio_bytes), language)

    def transcribe_array(self, samples: np.ndarray, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("SenseVoice TRT backend not ready — call preload() first")
        tag = _map_language(language)
        speech, valid = self._build_speech(samples, lang=tag)
        logits = self._infer(speech)
        if logits is None:
            return TranscriptionResult(text="", language=None, meta={})
        return TranscriptionResult(
            text=self._ctc_decode(logits, valid), language=None, meta={}
        )

    def _infer(self, speech: np.ndarray):
        from cuda import cudart

        speech = np.ascontiguousarray(speech, dtype=np.float32)
        out = np.empty(self._out_shape, dtype=np.float32)
        with self._lock:
            err, d_in = cudart.cudaMalloc(speech.nbytes)
            err, d_out = cudart.cudaMalloc(out.nbytes)
            err, stream = cudart.cudaStreamCreate()
            try:
                self._ctx.set_tensor_address(self._in_name, int(d_in))
                self._ctx.set_tensor_address(self._out_name, int(d_out))
                cudart.cudaMemcpy(d_in, speech.ctypes.data, speech.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
                ok = self._ctx.execute_async_v3(stream)
                cudart.cudaStreamSynchronize(stream)
                if not ok:
                    logger.error("SenseVoice TRT execute_async_v3 failed")
                    return None
                cudart.cudaMemcpy(out.ctypes.data, d_out, out.nbytes,
                                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            finally:
                cudart.cudaStreamDestroy(stream)
                cudart.cudaFree(d_in)
                cudart.cudaFree(d_out)
        return out.reshape(self._out_shape)[0]

    # ------------------------------------------------------------------
    # Front end + decode (identical contract to the RK backend)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_cmvn(path: str):
        txt = open(path).read()
        vals = [np.array(b.split(), dtype=np.float32) for b in re.findall(r"\[([^\]]*)\]", txt)]
        big = [v for v in vals if v.size == LFR_DIM]
        return big[0], big[1]

    @staticmethod
    def _compute_feats(audio: np.ndarray) -> np.ndarray:
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.dither = 0.0
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        fb = knf.OnlineFbank(opts)
        fb.accept_waveform(16000, (audio * 32768).tolist())
        fb.input_finished()
        return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])

    @staticmethod
    def _apply_lfr(feats: np.ndarray, m: int = 7, n: int = 6) -> np.ndarray:
        T = feats.shape[0]
        pad = (m - 1) // 2
        feats = np.vstack([np.tile(feats[0], (pad, 1)), feats])
        T2 = feats.shape[0]
        out = []
        i = 0
        while i * n < T:
            idx0 = i * n
            if idx0 + m <= T2:
                out.append(feats[idx0:idx0 + m].reshape(-1))
            else:
                chunk = feats[idx0:T2]
                need = m - chunk.shape[0]
                chunk = np.vstack([chunk, np.tile(feats[-1], (need, 1))])
                out.append(chunk.reshape(-1))
            i += 1
        return np.stack(out).astype(np.float32)

    def _build_speech(self, audio: np.ndarray, lang: str = "auto", textnorm: str = "withitn"):
        lfr = self._apply_lfr(self._compute_feats(audio))
        # NOTE: do NOT apply external CMVN. The lovemefan SenseVoice encoder ONNX
        # normalizes internally (first LayerNorm); applying am.mvn on top
        # double-normalizes and degrades accuracy (mean CER 0.048→0.032 across 5
        # zh samples when removed). am.mvn kept in bundle as reference only.
        prefix = np.stack([
            self._emb[_LANG_IDS.get(lang, 0)],
            self._emb[1],
            self._emb[2],
            self._emb[_TEXTNORM_IDS[textnorm]],
        ]).astype(np.float32)
        sp_in = np.concatenate([prefix, lfr], axis=0).astype(np.float32)
        valid = sp_in.shape[0]
        if valid > T_FIXED:
            sp_in = sp_in[:T_FIXED]
            valid = T_FIXED
        else:
            sp_in = np.vstack([sp_in, np.zeros((T_FIXED - valid, LFR_DIM), dtype=np.float32)])
        return sp_in[None], valid

    def _ctc_decode(self, logits: np.ndarray, valid: int) -> str:
        ids = logits.argmax(-1).tolist()[:valid]
        collapsed = []
        prev = -1
        for x in ids:
            if x != prev and x != BLANK_ID:
                collapsed.append(x)
            prev = x
        pieces = [self._sp.id_to_piece(i) for i in collapsed if 0 <= i < self._sp.get_piece_size()]
        text = "".join(pieces).replace("▁", " ")
        text = re.sub(r"<\|[^|]*\|>", "", text)
        return text.strip()

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> np.ndarray:
        import soundfile as sf

        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            raise ValueError(f"Cannot decode audio: {exc}") from exc
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            n_out = int(round(len(audio) * 16000 / sr))
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        return audio.astype(np.float32)
