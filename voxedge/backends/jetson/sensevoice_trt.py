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

Concurrency / buffers: ONE execution context, serialized by ``_lock``. Its device
buffers (d_in/d_out) and CUDA stream are allocated once in ``preload()`` and
reused by every ``_infer()`` call. ``config.max_concurrent`` does NOT add
contexts — it only raises the server-side admission ceiling so extra callers
queue instead of getting a 429; see ``concurrency_capability()``.

Measured on orin-nano (trtexec 10.3 + in-process timing), two things this file
deliberately does NOT do:

* **No execution-context pool.** GR3D is already 98% at 1 stream and throughput
  tops out at ~30 qps; ``--streams=2/4`` bought only 1.11x/1.13x (throughput is
  enqueue-bound: CPU enqueue 36.98 ms vs GPU 37.02 ms) while costing +216/+302
  MB and degrading latency to 135 ms at N=4. A single context with CUDA graphs
  reaches the same 29.45 qps at a quarter of the memory.
* **No pinned host buffer.** Pinned D2H really is faster (1.24 vs 4.77 ms), but
  the shared block has to be copied out before the next request overwrites it,
  and that copy costs 7.71 ms — host bandwidth (~4.5 GB/s), not a pinned
  artefact, since a pageable numpy->numpy copy of the same size measured 7.70
  ms. Net it was 0.7 ms *slower* than copying straight into a fresh array, plus
  32.9 MB of page-locked memory.

What does pay, and is kept: reusing the allocations (per-call ``cudaMalloc x2 +
cudaFree x2 + StreamCreate/Destroy`` measured 5.36 ms) and transferring only the
``valid`` frames instead of all ``T_FIXED`` (on a 3 s clip that is 54 rows of
344 — 84% of a 34.5 MB copy that the decoder then discarded).

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

from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    TranscriptionResult,
    resolve_reported_language,
)
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
    # ADMISSION ceiling, not a parallelism knob. The backend keeps exactly one
    # execution context and serializes on _lock; this value only tells the
    # coordinator how many requests may be admitted and QUEUED before it starts
    # rejecting with 429. Execution stays serialized either way — see
    # concurrency_capability(). Raising it costs no VRAM (no extra contexts);
    # it trades 429s for queueing latency, so size it against the per-request
    # ~68 ms and the client's timeout.
    max_concurrent: int = 1

    def __post_init__(self) -> None:
        if self.bpe_model is None:
            self.bpe_model = os.path.join(self.model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        self.max_concurrent = max(1, int(self.max_concurrent))


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
        self._warned_languages: set[str] = set()
        # Resident device buffers + stream, created once in preload(), reused by
        # every _infer(), released in unload(). This is the part that pays: the
        # old per-call cudaMalloc/cudaFree/cudaStreamCreate churn measured
        # 5.36 ms per request on orin-nano.
        #
        # There is deliberately NO pinned host buffer. Pinned D2H is genuinely
        # faster (1.24 ms vs 4.77 ms for the full tensor), but it lands in a
        # shared block the next request overwrites, so the result has to be
        # copied out again — and that host copy measured 7.71 ms. That is the
        # board's memory bandwidth (34.5 MB / 7.7 ms ~ 4.5 GB/s), not a pinned
        # artefact: a plain pageable numpy->numpy copy of the same size measured
        # 7.70 ms. Full pinned path 1.24 + 7.71 = 8.95 ms vs 8.26 ms for a D2H
        # straight into a fresh array — pinned was net NEGATIVE and cost 32.9 MB
        # of page-locked host memory. Measured on device; do not reintroduce it
        # without re-measuring.
        self._d_in = 0
        self._d_out = 0
        self._stream = None
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

    def concurrency_capability(self, profile=None) -> ConcurrencyCapability:
        # supports_parallel=False WITH max_concurrent=N>1 is DELIBERATE, not a
        # typo. The two fields answer different questions:
        #   supports_parallel -> may the coordinator run 2 requests at once?
        #                        No: one execution context, serialized by _lock.
        #   max_concurrent    -> how many requests may be ADMITTED?
        #                        N: extra callers queue behind the lock instead
        #                        of being rejected outright with 429.
        # Verified against the OVS resolve(): this pair yields ceiling=N with
        # mode=serialized. Do NOT "fix" it to supports_parallel=cap>1 — the
        # on-device measurement (GR3D 98% at 1 stream, --streams=2 = 1.11x for
        # +216 MB, enqueue-bound) is why there is no context pool here.
        return ConcurrencyCapability(
            supports_parallel=False,
            max_concurrent=max(1, int(self._cfg.max_concurrent)),
        )

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
        self._alloc_buffers()

        self._cmvn_add, self._cmvn_scale = self._load_cmvn(os.path.join(cfg.model_dir, "am.mvn"))
        self._emb = np.load(os.path.join(cfg.model_dir, "embedding.npy"))
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(cfg.bpe_model)

        self._ready = True
        logger.info("SenseVoice TRT backend ready (engine=%s, out=%s).", cfg.engine, self._out_shape)

    # ------------------------------------------------------------------
    # Resident buffers (allocated once — NOT per request)
    # ------------------------------------------------------------------

    def _alloc_buffers(self) -> None:
        """Allocate d_in / d_out / stream once.

        Shapes are static (fixed-shape engine), so nothing here has to be redone
        per request. Measured saving vs the old per-call path: 5.36 ms of
        cudaMalloc x2 + cudaFree x2 + StreamCreate/Destroy.

        VRAM is unchanged versus the per-call version — the same two device
        buffers, just held for the process lifetime instead of churned:
        d_out (1, 344, 25055) fp32 = 34.5 MB, d_in (1, 344, 560) fp32 = 0.77 MB.
        No host buffer is allocated here; see the module docstring for why the
        pinned one was removed.
        """
        from cuda import cudart

        in_nbytes = int(np.prod((1, T_FIXED, LFR_DIM))) * 4
        out_nbytes = int(np.prod(self._out_shape)) * 4
        try:
            err, d_in = cudart.cudaMalloc(in_nbytes)
            if int(err) != 0:
                raise RuntimeError(f"cudaMalloc(in={in_nbytes}) failed: {err}")
            self._d_in = int(d_in)
            err, d_out = cudart.cudaMalloc(out_nbytes)
            if int(err) != 0:
                raise RuntimeError(f"cudaMalloc(out={out_nbytes}) failed: {err}")
            self._d_out = int(d_out)
            err, stream = cudart.cudaStreamCreate()
            if int(err) != 0:
                raise RuntimeError(f"cudaStreamCreate failed: {err}")
            self._stream = stream
            # The context and the buffers are both resident, so bind once here
            # rather than on every _infer().
            self._ctx.set_tensor_address(self._in_name, self._d_in)
            self._ctx.set_tensor_address(self._out_name, self._d_out)
        except Exception:
            self._free_buffers()
            raise

    def _free_buffers(self) -> None:
        """Release everything _alloc_buffers took. Idempotent."""
        try:
            from cuda import cudart
        except Exception:  # pragma: no cover - no CUDA on this host
            self._stream = None
            self._d_in = self._d_out = 0
            return
        if self._stream is not None:
            stream, self._stream = self._stream, None
            try:
                cudart.cudaStreamSynchronize(stream)
            except Exception:
                pass
            try:
                cudart.cudaStreamDestroy(stream)
            except Exception:
                pass
        for attr in ("_d_in", "_d_out"):
            ptr = getattr(self, attr)
            if ptr:
                setattr(self, attr, 0)
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass

    def unload(self) -> None:
        self._ready = False
        with self._lock:
            self._free_buffers()
            self._ctx = None
            self._engine = None

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
        # The tag really is applied (it selects the language token prepended to
        # the input), so report it rather than None. _map_language falls back to
        # "auto" for anything unsupported, which resolve_reported_language then
        # surfaces as an ignored request instead of silently pretending.
        reported = resolve_reported_language(
            language, honoured=tag, backend=self.name, warned=self._warned_languages,
        )
        speech, valid = self._build_speech(samples, lang=tag)
        logits = self._infer(speech, valid)
        if logits is None:
            return TranscriptionResult(text="", language=reported, meta={})
        return TranscriptionResult(
            text=self._ctc_decode(logits, valid), language=reported, meta={}
        )

    def _infer(self, speech: np.ndarray, valid: Optional[int] = None):
        from cuda import cudart

        speech = np.ascontiguousarray(speech, dtype=np.float32)
        # Fixed-shape engine: _build_speech always pads/truncates to T_FIXED, so
        # the resident d_in fits exactly. Guard anyway — with a resident buffer a
        # shape bug becomes an out-of-bounds H2D write instead of a fresh malloc.
        if speech.shape != (1, T_FIXED, LFR_DIM):
            raise ValueError(
                f"SenseVoice TRT expects speech shape (1, {T_FIXED}, {LFR_DIM}), "
                f"got {speech.shape}"
            )
        # Only the first `valid` frames carry audio; the rest is the zero pad
        # _build_speech added to reach T_FIXED, and _ctc_decode throws it away.
        # The engine output is row-major (1, T, V), so those frames are a
        # contiguous prefix and the D2H can simply stop early. On a 3 s clip
        # valid is 54 of 344, i.e. 84% of the 34.5 MB transfer was pure waste —
        # and at 4.5 GB/s of host bandwidth that transfer is the single most
        # expensive non-GPU item in the request.
        rows = int(self._out_shape[1]) if valid is None else max(
            1, min(int(valid), int(self._out_shape[1]))
        )
        out = np.empty((rows, int(self._out_shape[2])), dtype=np.float32)
        out_nbytes = out.nbytes
        with self._lock:
            # Buffers/stream are resident: no cudaMalloc / cudaStreamCreate /
            # cudaFree per request (measured 5.00 ms of the old path). Tensor
            # addresses were bound in _alloc_buffers(); re-bind is cheap and keeps
            # the pairing explicit.
            self._ctx.set_tensor_address(self._in_name, self._d_in)
            self._ctx.set_tensor_address(self._out_name, self._d_out)
            cudart.cudaMemcpy(self._d_in, speech.ctypes.data, speech.nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            ok = self._ctx.execute_async_v3(self._stream)
            cudart.cudaStreamSynchronize(self._stream)
            if not ok:
                logger.error("SenseVoice TRT execute_async_v3 failed")
                return None
            # D2H straight into the caller's array. It is freshly allocated per
            # request, so nothing is shared and no second copy is needed — the
            # pinned variant needed one and lost 0.7 ms net doing it.
            cudart.cudaMemcpy(out.ctypes.data, self._d_out, out_nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return out

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
        # External am.mvn CMVN — device-validated default. Removing it improves
        # FP32 onnxruntime CER but is NOT a clean win on the fp16 device (per-
        # sample instability); see sensevoice_rknn for the on-device finding.
        lfr = (lfr + self._cmvn_add) * self._cmvn_scale
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
