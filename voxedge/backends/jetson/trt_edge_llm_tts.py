"""TTS backend via TRT-Edge-LLM C++ worker (qwen3_tts_worker / qwen3_tts_inference).

adapted from app/backends/jetson/trt_edge_llm_tts.py + app/core/worker_io.py
(2026-05-30), dedup after registry switch.

The Python side spawns a C++ worker subprocess and talks JSON-line IPC, so it
imports cleanly on a machine with no CUDA / tensorrt. Audio output is WAV via
the Code2Wav (vocoder) engine.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (TTSBackend / TTSCapability)
    and ``ConcurrencyCapability`` from ``voxedge.engine.concurrency_capability``.
  * ALL module-scope ``os.environ.get(...)`` reads (sampling defaults, artifact
    dirs, worker/segmentation flags, streaming-profile chunk frames, model_id
    /OVS_TTS_MODEL_ID) replaced by an explicit ``TRTEdgeLLMTTSConfig`` dataclass
    injected at construction time. voxedge has ZERO module-scope or hardcoded
    env reads.
  * ``WorkerIO`` imported from the sibling ``voxedge.backends.jetson.worker_io``;
    ``resolve_speaker_kwargs`` from ``._util`` (env-free, registry-free).
  * The speaker-encoder ONNX path uses lazy ``import onnxruntime`` / ``soundfile``.
  * ``concurrency_capability`` is an instance method (voxedge base contract)
    reading ``config.worker_concurrency`` instead of env/profile; the N>1
    ``--max_slots`` conditional (main fix b1cb1a5) is preserved.

Supports: BASIC_TTS, MULTI_LANGUAGE, STREAMING (+ VOICE_CLONE in worker mode).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Iterator, Optional

from voxedge.backends.base import TTSBackend, TTSCapability
from voxedge.engine.concurrency_capability import ConcurrencyCapability

from ._trt_edge_llm_util import resolve_speaker_kwargs
from .trt_edge_llm_ipc import run_binary
from .worker_io import WorkerExitError, WorkerIO

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var (first non-empty wins for alias groups) → field
#   EDGE_LLM_TTS_BIN                                  → tts_binary
#   EDGE_LLM_TTS_WORKER_BIN                           → worker_binary
#   EDGELLM_PLUGIN_PATH                              → plugin_path
#   EDGE_LLM_TTS_TALKER_DIR                           → talker_dir
#   EDGE_LLM_TTS_TALKER_BACKEND                       → talker_backend ("")         ← explicit-KV worker flag
#   EDGE_LLM_TTS_TALKER_ENGINE                        → talker_engine ("")          ← explicit-KV worker flag
#   EDGE_LLM_TTS_CODE_PREDICTOR_BACKEND               → code_predictor_backend ("") ← explicit-KV worker flag
#   EDGE_LLM_TTS_TEXT_PROJECTION                      → text_projection ("")        ← explicit-KV worker flag
#   EDGE_LLM_TTS_PROMPT_KV_CACHE                      → prompt_kv_cache ("")         ← explicit-KV worker flag
#   EDGE_LLM_TTS_CP_DIR                              → code_predictor_dir
#   EDGE_LLM_TTS_TOKENIZER_DIR                       → tokenizer_dir
#   EDGE_LLM_TTS_CODE2WAV_DIR                        → code2wav_dir
#   QWEN3_SPEAKER_ENCODER                            → speaker_encoder
#   OVS_TTS_MODEL_ID                                → model_id ("trt_edgellm")
#   OVS_TTS_BACKEND/EDGE_LLM_TTS_BACKEND            → backend_mode ("edgellm_worker")
#   EDGE_LLM_TTS_WORKER                              → use_worker (True)
#   OVS_TTS_WORKER_CONCURRENCY                       → worker_concurrency (1) ← N>1 gates --max_slots
#   EDGE_LLM_QWEN3_PROFILE/OVS_QWEN3_PROFILE        → qwen3_runtime_profile ("highperf")
#   EDGE_LLM_TTS_PERF_PROFILE                        → perf_profile ("quality")
#   EDGE_LLM_TTS_STATEFUL_CODE2WAV                   → stateful_code2wav (None→profile-derived)
#   OVS_TTS_SEED                                     → seed (42)
#   OVS_TTS_TALKER_TEMPERATURE/TTS_TALKER_TEMPERATURE → talker_temperature (0.9)
#   OVS_TTS_TALKER_TOP_K/TTS_TALKER_TOP_K            → talker_top_k (50)
#   OVS_TTS_TOP_P/TTS_TOP_P                          → talker_top_p (1.0)
#   OVS_TTS_PREDICTOR_TEMPERATURE/...                → predictor_temperature (0.9)
#   OVS_TTS_PREDICTOR_TOP_K/...                      → predictor_top_k (50)
#   OVS_TTS_PREDICTOR_TOP_P/...                      → predictor_top_p (1.0)
#   TTS_MAX_AUDIO_LENGTH                              → max_audio_length (1024)
#   TTS_MIN_AUDIO_LENGTH                              → min_audio_length (30)
#   TTS_REPETITION_PENALTY                            → repetition_penalty (1.05)
#   TTS_CODEC_EOS_LOGIT_OFFSET                        → codec_eos_logit_offset (0.0)
#   EDGE_LLM_TTS_SEGMENT_TEXT                         → segment_text (True)
#   EDGE_LLM_TTS_SEGMENT_MAX_CHARS/..._CJK_...       → segment_max_chars_latin/cjk (120/48)
#   EDGE_LLM_TTS_SEGMENT_PAUSE_MS/HARD_...           → segment_pause_ms/hard (80/120)
#   EDGE_LLM_TTS_STREAMING_PROFILE                   → streaming_profile ("continuous_playback")
#   EDGE_LLM_TTS_FIRST_CHUNK_FRAMES etc.             → chunk-frame overrides (None→profile-derived)
#   (product_model_base / product_overlay_dir removed — product_explicit_kv mode deleted)


_HIGHPERF_PROFILES = ("highperf", "perf", "performance", "v2v")
_FAST_PERF_PROFILES = ("fast", "v2v", "low_latency")


@dataclass
class TRTEdgeLLMTTSConfig:
    """Explicit construction-time config for :class:`TRTEdgeLLMTTSBackend`.

    Every field default is identical to the production env default. Artifact
    path fields default to empty strings (production resolved them from
    ``~/...`` artifact trees via env at import); a working backend MUST supply
    them. ``stateful_code2wav`` defaults to ``None`` → derived from
    ``qwen3_runtime_profile`` (mirrors the production env-default expression).
    """

    # Binaries / engines / plugin / artifact dirs.
    tts_binary: str = ""
    worker_binary: str = ""
    plugin_path: str = ""
    talker_dir: str = ""
    # Explicit-KV (highperf) worker flags. When set, ``_ensure_worker`` passes the
    # corresponding ``--qwen3Tts*`` / ``--codePredictor*`` flags to the C++
    # ``qwen3_tts_worker`` so a single-optimization-profile w8a16 talker engine is
    # loaded by the explicit-KV runner instead of the generic 2-profile
    # ``LLMEngineRunner`` (which rejects the single-profile engine). Each is
    # emitted independently; empty → flag omitted (legacy generic-runner path,
    # byte-equivalent at N=1). Mirrors profiling patch a9995c6c.
    #   talker_backend         → --qwen3TtsTalkerBackend
    #   talker_engine          → --qwen3TtsTalkerEngine
    #   code_predictor_backend → --codePredictorBackend
    #   text_projection        → --qwen3TtsTextProjection
    #   prompt_kv_cache        → --qwen3TtsPromptKvCache
    talker_backend: str = ""
    talker_engine: str = ""
    code_predictor_backend: str = ""
    text_projection: str = ""
    prompt_kv_cache: str = ""
    code_predictor_dir: str = ""
    tokenizer_dir: str = ""
    code2wav_dir: str = ""
    speaker_encoder: str = ""
    # Fixed BASE-model speaker conditioning. The Qwen3-TTS *base* checkpoint
    # conditions on an external 1024-d speaker embedding (vs CustomVoice's named
    # speaker ids). When set, this precomputed base64 embedding is injected as
    # ``speaker_embedding_b64`` on every request that does not carry its own
    # speaker (speaker_id / speaker / per-request speaker_embedding still win).
    # Empty → unchanged CustomVoice/named-speaker behavior (backward compatible).
    base_speaker_embedding_b64: str = ""

    model_id: str = "trt_edgellm"
    backend_mode: str = "edgellm_worker"
    use_worker: bool = True
    worker_concurrency: int = 1  # N>1 gates --max_slots (main fix b1cb1a5)
    qwen3_runtime_profile: str = "highperf"
    perf_profile: str = "quality"
    stateful_code2wav: Optional[bool] = None
    # Streaming-native worker (TensorRT-Edge-LLM v0.9.0 lean code2wav): the
    # worker only emits audio via streamingChunkFrames/onChunkReady and has no
    # output_file (write-whole-WAV) mode. When True, the non-streaming synth
    # path aggregates streaming chunks into a WAV instead of requesting an
    # output_file (which the v090 worker rejects with KeyError('output_file')).
    # Defaults False to preserve the v0.8.0 non-stateful output_file path.
    streaming_only_worker: bool = False
    seed: int = 42

    # Sampling.
    talker_temperature: float = 0.9
    talker_top_k: int = 50
    talker_top_p: float = 1.0
    predictor_temperature: float = 0.9
    predictor_top_k: int = 50
    predictor_top_p: float = 1.0
    max_audio_length: int = 1024
    min_audio_length: int = 30
    repetition_penalty: float = 1.05
    codec_eos_logit_offset: float = 0.0

    # Text segmentation.
    segment_text: bool = True
    segment_max_chars_latin: int = 120
    segment_max_chars_cjk: int = 48
    segment_pause_ms: int = 80
    segment_hard_pause_ms: int = 120

    # Streaming.
    streaming_profile: str = "continuous_playback"
    # Per-stream chunk-frame overrides (None -> streaming_profile-derived
    # default). These were the EDGE_LLM_TTS_FIRST_CHUNK_FRAMES / _CHUNK_FRAMES /
    # _ADAPTIVE_CHUNKS / _MAX_CHUNK_FRAMES / _CHUNK_GROWTH_FRAMES env reads in
    # the pre-migration backend; the product config-builder wires them.
    first_chunk_frames: Optional[int] = None
    chunk_frames: Optional[int] = None
    adaptive_chunks: Optional[bool] = None
    max_chunk_frames: Optional[int] = None
    chunk_growth_frames: Optional[int] = None

    # Extra env passed through to the worker subprocess.
    extra_worker_env: dict = field(default_factory=dict)

    # Optional stable artifact name for the runtime-artifact manifest
    # (voxedge.artifacts). None preserves the existing host-mounted behaviour.
    artifact_ref: Optional[str] = None

    def __post_init__(self) -> None:
        self.worker_concurrency = max(1, int(self.worker_concurrency))
        self.qwen3_runtime_profile = (
            (self.qwen3_runtime_profile or "highperf").strip().lower().replace("-", "_")
        )
        self.backend_mode = (
            (self.backend_mode or "edgellm_worker").strip().lower().replace("-", "_")
        )
        self.perf_profile = (self.perf_profile or "quality").strip().lower()

    # -- derived helpers (no env reads) --

    def highperf_enabled(self) -> bool:
        return self.qwen3_runtime_profile in _HIGHPERF_PROFILES

    def stateful_code2wav_enabled(self) -> bool:
        if self.stateful_code2wav is not None:
            return bool(self.stateful_code2wav)
        return self.highperf_enabled()

    def fast_perf_profile(self) -> bool:
        return self.perf_profile in _FAST_PERF_PROFILES


class PoolSaturatedError(RuntimeError):
    """TTS worker rejected a request because every decoder slot is busy.

    The C++ ``qwen3_tts_worker`` launched with ``--max_slots N`` returns a
    ``status:4429`` payload when an N+1st concurrent request arrives. A plain
    RuntimeError (NOT routed into the destructive worker-restart path).
    """

    status: int = 4429

    def __init__(self, message: str, max_slots: Optional[int] = None) -> None:
        super().__init__(message)
        self.max_slots = max_slots


def _tts_pool_saturated_error(event: dict) -> Optional[PoolSaturatedError]:
    """Return a PoolSaturatedError if ``event`` is a worker saturation reject."""
    if not isinstance(event, dict):
        return None
    if event.get("event") != "error" and event.get("ok") is not False:
        return None
    msg = ""
    for key in ("error", "message", "reason", "detail"):
        v = event.get(key)
        if isinstance(v, str) and v:
            msg = v
            break
    low = msg.lower()
    if (
        event.get("status") == 4429
        or "pool_saturated" in low
        or "too_many_tts" in low
        or "too many tts" in low
    ):
        return PoolSaturatedError(msg or str(event), max_slots=event.get("max_slots"))
    return None


_SPEAKER_ENC_SESSION = None  # cached ort session — onnx load is non-trivial


def _qwen3_speaker_embed_inproc(audio_wav_bytes: bytes, encoder_path: str) -> bytes:
    """In-process 1024-d speaker embedding from a reference WAV.

    Pure numpy mel + ONNX Runtime. ``onnxruntime`` / ``soundfile`` are imported
    lazily so this module imports without those optional deps.
    """
    global _SPEAKER_ENC_SESSION
    import numpy as np
    import onnxruntime as ort
    import soundfile as sf

    data, sr_in = sf.read(io.BytesIO(audio_wav_bytes), always_2d=False, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)
    if sr_in != 24000:
        n_in = len(data)
        n_out = int(round(n_in * 24000 / sr_in))
        spec = np.fft.rfft(data)
        n_spec_out = n_out // 2 + 1
        if n_spec_out < len(spec):
            spec = spec[:n_spec_out]
        else:
            spec = np.concatenate(
                [spec, np.zeros(n_spec_out - len(spec), dtype=spec.dtype)]
            )
        data = (np.fft.irfft(spec, n=n_out) * (n_out / n_in)).astype(np.float32)
    sr = 24000

    N_FFT, HOP, WIN, N_MEL = 1024, 256, 1024, 128
    FMIN, FMAX = 0.0, 12000.0

    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _slaney_mel_filterbank() -> "np.ndarray":
        mel_pts = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MEL + 2)
        hz_pts = _mel_to_hz(mel_pts)
        bin_freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
        fb = np.zeros((N_MEL, N_FFT // 2 + 1), dtype=np.float32)
        for i in range(N_MEL):
            lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
            lt = (bin_freqs >= lo) & (bin_freqs <= mid)
            rt = (bin_freqs >= mid) & (bin_freqs <= hi)
            fb[i, lt] = (bin_freqs[lt] - lo) / (mid - lo + 1e-12)
            fb[i, rt] = (hi - bin_freqs[rt]) / (hi - mid + 1e-12)
        enorm = 2.0 / (hz_pts[2:] - hz_pts[:-2])
        fb *= enorm[:, None]
        return fb

    mel_basis = _slaney_mel_filterbank()
    hann = np.hanning(WIN).astype(np.float32)
    pad = (N_FFT - HOP) // 2
    y = np.pad(data, pad, mode="reflect")
    num_frames = 1 + (len(y) - WIN) // HOP
    if num_frames < 1:
        raise ValueError(f"reference audio too short: {len(data)/sr:.2f}s (need >0.5s)")
    frames = np.lib.stride_tricks.sliding_window_view(y, WIN)[::HOP][:num_frames] * hann
    spec = np.fft.rfft(frames, n=N_FFT, axis=-1)
    mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9).astype(np.float32)
    mel_spec = mag @ mel_basis.T
    mel_spec = np.log(np.clip(mel_spec, 1e-5, None)).astype(np.float32)

    if _SPEAKER_ENC_SESSION is None or _SPEAKER_ENC_SESSION[0] != encoder_path:
        sess = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
        _SPEAKER_ENC_SESSION = (encoder_path, sess)
    sess = _SPEAKER_ENC_SESSION[1]
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: mel_spec[None, ...]})
    emb = out[0].squeeze().astype(np.float32)
    if emb.shape != (1024,):
        raise RuntimeError(f"unexpected speaker embedding shape: {emb.shape}")
    return emb.tobytes()


def _code2wav_engine_path(code2wav_dir: str) -> str:
    """Return the Code2Wav engine path used by current Qwen3 artifact sets."""
    candidates = [
        os.path.join(code2wav_dir, "code2wav.engine"),
        os.path.join(code2wav_dir, "code2wav_stateful.engine"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _detect_language(text: str) -> str:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return "chinese"
        if 0x3040 <= cp <= 0x30FF:
            return "japanese"
        if 0xAC00 <= cp <= 0xD7AF:
            return "korean"
    return "english"


def _contains_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3040 <= cp <= 0x30FF or 0xAC00 <= cp <= 0xD7AF:
            return True
    return False


def _split_tts_text(
    text: str,
    max_chars: Optional[int] = None,
    *,
    max_chars_latin: int = 120,
    max_chars_cjk: int = 48,
) -> list[str]:
    """Split long TTS text into independently stable synthesis requests."""
    normalized = " ".join(text.split()) if not _contains_cjk(text) else text.strip()
    if not normalized:
        return []

    if max_chars is None:
        max_chars = max_chars_cjk if _contains_cjk(normalized) else max_chars_latin
    max_chars = max(8, int(max_chars))

    hard_breaks = set("。！？!?；;\n")
    soft_breaks = set("，,、：:")
    segments: list[str] = []
    current: list[str] = []
    is_cjk = _contains_cjk(normalized)
    max_overrun = 0 if is_cjk else max(2, min(8, max_chars // 2))
    abbreviations = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
        "etc.", "e.g.", "i.e.",
    }

    def is_nonterminal_period(buffer: str, next_ch: str) -> bool:
        stripped = buffer.strip().lower()
        if next_ch.isdigit() and len(stripped) >= 2 and stripped[-2].isdigit():
            return True
        return any(stripped.endswith(abbrev) for abbrev in abbreviations)

    def flush() -> None:
        part = "".join(current).strip()
        current.clear()
        if part:
            segments.append(part)

    for idx, ch in enumerate(normalized):
        next_ch = normalized[idx + 1] if idx + 1 < len(normalized) else ""
        current.append(ch)
        if ch in hard_breaks:
            if not is_cjk and ch == "." and is_nonterminal_period("".join(current), next_ch):
                continue
            flush()
            continue
        current_text = "".join(current).strip()
        if len(current_text) >= max_chars:
            text_so_far = "".join(current)
            cut = max(text_so_far.rfind(p) for p in soft_breaks)
            if cut >= max_chars // 3:
                head = text_so_far[: cut + 1].strip()
                tail = text_so_far[cut + 1:].lstrip()
                current.clear()
                if head:
                    segments.append(head)
                if tail:
                    current.extend(tail)
            else:
                if is_cjk and len(current_text) < max_chars + max_overrun:
                    continue
                flush()
    flush()

    if not is_cjk:
        packed: list[str] = []
        for part in segments:
            if len(part) <= max_chars:
                packed.append(part)
                continue
            words = part.split(" ")
            buf: list[str] = []
            for word in words:
                candidate = " ".join(buf + [word]).strip()
                if buf and len(candidate) > max_chars:
                    packed.append(" ".join(buf))
                    buf = [word]
                else:
                    buf.append(word)
            if buf:
                packed.append(" ".join(buf))
        segments = packed

    merged: list[str] = []
    min_chars = max(4, min(12, max_chars // 3))
    for part in segments:
        if merged and all(ch in hard_breaks or ch in soft_breaks for ch in part):
            merged[-1] = f"{merged[-1]}{part}"
            continue
        sep = "" if merged and _contains_cjk(part + merged[-1]) else " "
        if merged and len(part) < min_chars and len(merged[-1]) + len(sep) + len(part) <= max_chars:
            merged[-1] = f"{merged[-1]}{sep}{part}"
        else:
            merged.append(part)
    return merged


def _segment_pause_ms(segment: str, pause_ms: int = 80, hard_pause_ms: int = 120) -> int:
    """Silence to insert *after* a synthesized segment when concatenating."""
    if not segment:
        return 0
    stripped = segment.rstrip()
    if stripped.endswith(("。", "！", "？", "!", "?", ";", "；")):
        return max(0, hard_pause_ms)
    if stripped.endswith(("，", ",", "、", "：", ":")):
        return max(0, pause_ms)
    return 0


def _concat_wav_bytes(parts: list[bytes], pauses_ms: Optional[list[int]] = None) -> bytes:
    non_empty = [part for part in parts if part]
    if not non_empty:
        return b""
    if len(non_empty) == 1:
        return non_empty[0]

    params = None
    frames: list[bytes] = []
    for idx, part in enumerate(non_empty):
        with wave.open(io.BytesIO(part), "rb") as reader:
            current = reader.getparams()
            comparable = (current.nchannels, current.sampwidth, current.framerate, current.comptype, current.compname)
            if params is None:
                params = comparable
            elif comparable != params:
                raise RuntimeError(f"Cannot concatenate WAV segments with different formats: {comparable} != {params}")
            frames.append(reader.readframes(reader.getnframes()))
            if pauses_ms and idx < len(non_empty) - 1:
                pause_samples = int(current.framerate * max(0, pauses_ms[idx]) / 1000)
                if pause_samples > 0:
                    frames.append(b"\x00" * pause_samples * current.nchannels * current.sampwidth)

    nchannels, sampwidth, framerate, comptype, compname = params
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(nchannels)
        writer.setsampwidth(sampwidth)
        writer.setframerate(framerate)
        writer.setcomptype(comptype, compname)
        for frame_bytes in frames:
            writer.writeframes(frame_bytes)
    return out.getvalue()


def _wav_duration_and_samples(wav_bytes: bytes) -> tuple[float, int]:
    if not wav_bytes:
        return 0.0, 0
    with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
        samples = reader.getnframes()
        rate = reader.getframerate()
    return (samples / rate if rate > 0 else 0.0), samples


def _event_request_id(event: dict) -> str | None:
    rid = event.get("request_id")
    if rid:
        return rid
    rid = event.get("id")
    return rid if rid else None


def _pcm16_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class TRTEdgeLLMTTSBackend(TTSBackend):
    """TTS via TRT-Edge-LLM qwen3_tts_worker subprocess."""

    def concurrency_capability(self) -> ConcurrencyCapability:
        """Declare the TTS slot-pool ceiling.

        WorkerIO multiplexes N in-flight requests with a single subprocess.
        Reads ``config.worker_concurrency`` (was env
        ``OVS_TTS_WORKER_CONCURRENCY`` / profile ``tts_worker_concurrency``).
        N>1 enables ``supports_parallel``.
        """
        n = max(1, int(self._config.worker_concurrency))
        return ConcurrencyCapability(
            supports_parallel=n > 1,
            max_concurrent=n,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    @property
    def supports_hot_reload(self) -> bool:  # type: ignore[override]
        return True

    def __init__(self, config: Optional[TRTEdgeLLMTTSConfig] = None):
        self._config = config or TRTEdgeLLMTTSConfig()
        self._ready = False
        self._worker: Optional[subprocess.Popen] = None
        self._worker_lock = threading.Lock()
        self._worker_io: Optional[WorkerIO] = None
        self._worker_concurrency: int = max(1, int(self._config.worker_concurrency))
        self._worker_ready_meta: dict = {}
        self._worker_stderr_tail = deque(maxlen=80)
        # Artifact paths captured from the injected config (was resolved from
        # the current env at __init__ in the production copy).
        self._talker_dir = self._config.talker_dir
        self._talker_backend = (self._config.talker_backend or "").strip()
        self._talker_engine = (self._config.talker_engine or "").strip()
        self._code_predictor_backend = (self._config.code_predictor_backend or "").strip()
        self._text_projection = (self._config.text_projection or "").strip()
        self._prompt_kv_cache = (self._config.prompt_kv_cache or "").strip()
        self._code_predictor_dir = self._config.code_predictor_dir
        self._tokenizer_dir = self._config.tokenizer_dir
        self._speaker_encoder = self._config.speaker_encoder
        # Decode the fixed base-model speaker embedding once (None if unset).
        self._base_speaker_embedding: Optional[bytes] = None
        _b64 = (self._config.base_speaker_embedding_b64 or "").strip()
        if _b64:
            try:
                self._base_speaker_embedding = base64.b64decode(_b64)
            except Exception:
                logger.warning("invalid base_speaker_embedding_b64; ignoring")
                self._base_speaker_embedding = None
        self._code2wav_dir = self._config.code2wav_dir
        self._worker_binary = self._config.worker_binary
        self._qwen3_runtime_profile = self._config.qwen3_runtime_profile

    # -- TTSBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def model_id(self) -> str:
        """Model-scope key — injected via config (was ``OVS_TTS_MODEL_ID``)."""
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {TTSCapability.BASIC_TTS, TTSCapability.MULTI_LANGUAGE, TTSCapability.STREAMING,
                TTSCapability.VOICE_CLONE}
        return caps

    @property
    def sample_rate(self) -> int:
        return 24000

    def is_ready(self) -> bool:
        return self._ready

    def _backend_mode(self) -> str:
        return self._config.backend_mode

    def preload(self) -> None:
        """Verify all required files exist."""
        mode = self._backend_mode()
        if mode not in ("edgellm", "edgellm_worker", "official"):
            raise ValueError(
                f"Unsupported backend_mode {mode!r}; expected edgellm_worker"
            )

        required = [
            (self._worker_binary if self._use_worker() else self._config.tts_binary, "TTS binary"),
            (self._config.plugin_path, "TRT-Edge-LLM plugin"),
            (os.path.join(self._talker_dir, "config.json"), "talker config"),
            (os.path.join(self._tokenizer_dir, "tokenizer.json"), "tokenizer"),
            (os.path.join(self._talker_dir, "llm.engine"), "talker engine"),
        ]
        missing = []
        for path, label in required:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")
        if missing:
            raise FileNotFoundError(
                "TTS preload failed — missing:\n  " + "\n  ".join(missing)
            )

        c2w_path = _code2wav_engine_path(self._code2wav_dir)
        if os.path.exists(c2w_path):
            logger.info("Code2Wav engine found at %s", c2w_path)
        else:
            logger.warning(
                "Code2Wav not found at %s — will output RVQ codes only", c2w_path
            )

        logger.info(
            "TTS backend preload OK (profile=%s binary=%s talker=%s)",
            self._qwen3_runtime_profile,
            self._worker_binary if self._use_worker() else self._config.tts_binary,
            self._talker_dir,
        )
        if self._use_worker():
            self._ensure_worker()
        self._ready = True

    def unload(self) -> None:
        """Kill the resident worker subprocess so GPU memory is fully released."""
        if not self._ready and self._worker is None:
            return
        try:
            with self._worker_lock:
                old = self._worker
                self._worker = None
                self._worker_io = None
                if old is not None:
                    try:
                        old.terminate()
                        old.wait(timeout=5)
                    except Exception:
                        try:
                            old.kill()
                        except Exception:
                            pass
                self._worker_stderr_tail.clear()
                self._worker_ready_meta = {}
        except Exception:
            logger.exception("TRTEdgeLLMTTSBackend.unload failed; continuing")
        finally:
            self._ready = False

    def _use_worker(self) -> bool:
        return bool(self._config.use_worker)

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        env.update(self._config.extra_worker_env)
        env["EDGELLM_PLUGIN_PATH"] = self._config.plugin_path
        env.setdefault("EDGE_LLM_TTS_CUDA_GRAPH", "0")
        env.setdefault("EDGE_LLM_TTS_LAZY_CODE2WAV", "0")
        env.setdefault(
            "EDGE_LLM_TTS_STATEFUL_CODE2WAV",
            "1" if self._config.highperf_enabled() else "0",
        )
        if self._config.stateful_code2wav_enabled():
            env.setdefault("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", "0")
            env.setdefault("QWEN3_TTS_CP_DECODE_CUDA_GRAPH", "1")
            env.setdefault("QWEN3_TTS_ACTIVE_CP_GROUPS", "13")
        else:
            env.setdefault("EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES", "3")
        return env

    def _worker_stderr_snip(self) -> str:
        return "".join(self._worker_stderr_tail)[-2000:] or "(empty)"

    def _drain_worker_stderr(self) -> None:
        worker = self._worker
        if worker is None or worker.stderr is None:
            return
        for line in worker.stderr:
            self._worker_stderr_tail.append(line)
            if "[JV_MEM]" in line:
                logger.info("TTS worker: %s", line.rstrip())
            else:
                logger.debug("TTS worker stderr: %s", line.rstrip())

    def _explicit_kv_flags(self) -> list:
        """Explicit-KV (highperf) worker CLI flags for the C++ ``qwen3_tts_worker``.

        Each flag is emitted independently when its config field is set, so a
        single-optimization-profile w8a16 talker engine is loaded by the
        explicit-KV runner instead of the generic 2-profile ``LLMEngineRunner``.
        All-empty (default) → no flags (legacy generic-runner path, byte-equivalent
        at N=1). Mirrors profiling patch a9995c6c.
        """
        flags: list = []
        if self._talker_backend:
            flags += ["--qwen3TtsTalkerBackend", self._talker_backend]
        if self._talker_engine:
            flags += ["--qwen3TtsTalkerEngine", self._talker_engine]
        if self._code_predictor_backend:
            flags += ["--codePredictorBackend", self._code_predictor_backend]
        if self._text_projection:
            flags += ["--qwen3TtsTextProjection", self._text_projection]
        if self._prompt_kv_cache:
            flags += ["--qwen3TtsPromptKvCache", self._prompt_kv_cache]
        return flags

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        self._worker_concurrency = max(1, int(self._config.worker_concurrency))
        cmd = [
            self._worker_binary,
            "--talkerEngineDir", self._talker_dir,
            "--codePredictorEngineDir", self._code_predictor_dir,
            "--tokenizerDir", self._tokenizer_dir,
            "--code2wavEngineDir", self._code2wav_dir,
        ]
        cmd += self._explicit_kv_flags()
        # Only emit --max_slots when N>1 (main fix b1cb1a5): at N=1 omit it for
        # legacy single-slot byte-equivalent behavior + back-compat with older
        # worker binaries that reject the unknown flag.
        if self._worker_concurrency and self._worker_concurrency > 1:
            cmd += ["--max_slots", str(self._worker_concurrency)]
        self._worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        threading.Thread(target=self._drain_worker_stderr, name="tts-worker-stderr", daemon=True).start()
        assert self._worker.stdout is not None
        ready_line = self._worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"TTS worker failed to start: {self._worker_stderr_snip()}")
        ready = json.loads(ready_line)
        if ready.get("event") != "ready":
            raise RuntimeError(f"TTS worker did not become ready: {ready}")
        self._worker_ready_meta = ready
        self._worker_concurrency = max(1, int(self._config.worker_concurrency))
        self._worker_io = WorkerIO(self._worker, self._worker_concurrency)

    def _restart_worker_locked(self, reason: str) -> None:
        """Restart the resident TTS worker. Called from inside ``_worker_lock``."""
        logger.warning("Restarting TTS worker: %s", reason)
        old = self._worker
        self._worker = None
        self._worker_io = None
        if old is not None:
            try:
                old.terminate()
                old.wait(timeout=5)
            except Exception:
                try:
                    old.kill()
                except Exception:
                    pass
        self._worker_stderr_tail.clear()
        self._ensure_worker()

    def _synthesize_worker(self, text: str, language: Optional[str], **kwargs) -> tuple[bytes, dict]:
        if self._config.stateful_code2wav_enabled() or self._config.streaming_only_worker:
            # Streaming worker (stateful v080, or streaming-native v090 lean):
            # no output_file mode → aggregate streaming chunks into a WAV.
            return self._synthesize_worker_via_stream(text, language=language, **kwargs)
        req_id = uuid.uuid4().hex
        with tempfile.NamedTemporaryFile(prefix="trt_edgellm_tts_", suffix=".wav", delete=False) as f:
            output_file = f.name
        request = {
            "id": req_id,
            "text": text,
            "output_file": output_file,
            "language": language or _detect_language(text),
            "talker_temperature": self._config.talker_temperature,
            "talker_top_k": self._config.talker_top_k,
            "talker_top_p": self._config.talker_top_p,
            "repetition_penalty": self._config.repetition_penalty,
            "codec_eos_logit_offset": self._config.codec_eos_logit_offset,
            "predictor_temperature": self._config.predictor_temperature,
            "predictor_top_k": self._config.predictor_top_k,
            "predictor_top_p": self._config.predictor_top_p,
            "max_audio_length": kwargs.get("max_audio_length", self._config.max_audio_length),
            "min_audio_length": kwargs.get("min_audio_length", self._config.min_audio_length),
            "seed": int(kwargs.get("seed", self._config.seed)),
        }
        voice_kwargs = self._resolve_voice_kwargs(kwargs)
        speaker_embedding = voice_kwargs.get("speaker_embedding")
        if speaker_embedding:
            request["speaker_embedding_b64"] = base64.b64encode(speaker_embedding).decode("ascii")
        else:
            self._add_speaker_request_fields(request, voice_kwargs)
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker_io is not None
        t0 = time.time()
        response = None
        try:
            for event in self._worker_io.request(request):
                response = event
        except WorkerExitError as exc:
            self._worker = None
            self._worker_io = None
            raise RuntimeError(f"TTS worker exited before response: {self._worker_stderr_snip()}") from exc
        elapsed = time.time() - t0
        if response is None:
            self._worker = None
            self._worker_io = None
            raise RuntimeError(f"TTS worker returned no events: {self._worker_stderr_snip()}")
        saturated = _tts_pool_saturated_error(response)
        if saturated is not None:
            raise saturated
        if not response.get("ok"):
            raise RuntimeError(f"TTS worker failed: {response}")
        with open(response["output_file"], "rb") as f:
            wav_bytes = f.read()
        try:
            os.unlink(response["output_file"])
        except OSError:
            pass
        audio_s = float(response.get("audio_s", 0.0))
        meta = {
            "inference_time_s": round(elapsed, 3),
            "sample_rate": int(response.get("sample_rate", 24000)),
            "duration_s": audio_s,
            "samples": int(response.get("samples", 0)),
            "rtf": round(float(response.get("rtf", 0.0)), 3),
            "generation_ms": round(float(response.get("generation_ms", 0.0)), 1),
            "code2wav_ms": round(float(response.get("code2wav_ms", 0.0)), 1),
            "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        }
        return wav_bytes, meta

    def _synthesize_worker_via_stream(
        self, text: str, language: Optional[str] = None, **kwargs
    ) -> tuple[bytes, dict]:
        """Aggregate streaming PCM chunks into a single WAV."""
        t0 = time.time()
        done_meta: dict = {}
        stream_kwargs = dict(kwargs)
        stream_kwargs["segment_text"] = False
        stream_kwargs["language"] = language
        pcm = bytearray()
        for chunk in self._generate_streaming_single(text, meta_out=done_meta, **stream_kwargs):
            pcm.extend(chunk)
        elapsed = time.time() - t0
        sample_rate = int(done_meta.get("sample_rate", 24000))
        wav_bytes = _pcm16_to_wav(bytes(pcm), sample_rate=sample_rate)
        meta = {
            "inference_time_s": round(elapsed, 3),
            "sample_rate": sample_rate,
            "duration_s": float(done_meta.get("audio_s", 0.0)),
            "samples": int(done_meta.get("samples", len(pcm) // 2)),
            "rtf": round(float(done_meta.get("rtf", 0.0)), 3),
            "generation_ms": round(float(done_meta.get("generation_ms", 0.0)), 1),
            "code2wav_ms": round(float(done_meta.get("code2wav_ms", 0.0)), 1),
            "first_chunk_ms": round(float(done_meta.get("first_chunk_ms", 0.0)), 1),
            "chunk_count": int(done_meta.get("chunk_count", 0)),
            "stateful_code2wav": bool(done_meta.get("stateful_code2wav", True)),
            "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
        }
        return wav_bytes, meta

    def rate_pitch_caps(self) -> tuple[bool, bool]:
        # No native speed/pitch → both via DSP fallback. (When a product
        # backend is delegated to, the base wrapper has already popped
        # speed/pitch, so the delegated public call is an identity pass-through
        # and there is no double-apply.)
        return (False, False)

    def _generate_streaming_impl(self, text: str, **kwargs):
        """Yield raw PCM int16 chunks from the resident EdgeLLM TTS worker."""
        if self._config.segment_text and kwargs.get("segment_text", True):
            segments = _split_tts_text(
                text,
                kwargs.get("segment_max_chars"),
                max_chars_latin=self._config.segment_max_chars_latin,
                max_chars_cjk=self._config.segment_max_chars_cjk,
            )
            if len(segments) > 1:
                segment_kwargs = dict(kwargs)
                segment_kwargs["segment_text"] = False
                segment_kwargs.setdefault("seed", self._config.seed)
                for segment in segments:
                    yield from self._generate_streaming_impl(segment, **segment_kwargs)
                return

        yield from self._generate_streaming_single(text, **kwargs)

    def _generate_streaming_single(self, text: str, meta_out: Optional[dict] = None, **kwargs):
        """Yield raw PCM int16 chunks for one already-bounded TTS request."""
        retry_empty = bool(kwargs.pop("_retry_empty", True))
        req_id = uuid.uuid4().hex
        streaming_profile = str(
            kwargs.get("streaming_profile", self._config.streaming_profile)
        ).lower()
        if streaming_profile in ("v2v", "voice_to_voice", "eos_to_first_audio"):
            default_first_chunk_frames = 1
            default_chunk_frames = 97
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 97
            default_adaptive_chunks = False
        elif streaming_profile in ("instant_feedback", "low_latency"):
            default_first_chunk_frames = 1
            default_chunk_frames = 25
            default_chunk_growth_frames = 50
            default_max_chunk_frames = 150
            default_adaptive_chunks = True
        elif streaming_profile in ("playback", "smooth", "buffered"):
            default_first_chunk_frames = 20
            default_chunk_frames = 20
            default_chunk_growth_frames = 30
            default_max_chunk_frames = 120
            default_adaptive_chunks = True
        else:
            default_first_chunk_frames = 50
            default_chunk_frames = 97
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 97
            default_adaptive_chunks = False
        if self._config.stateful_code2wav_enabled():
            if self._config.fast_perf_profile():
                default_first_chunk_frames = 4
            elif self._config.perf_profile == "balanced":
                default_first_chunk_frames = 6
            else:
                default_first_chunk_frames = 7
            default_chunk_frames = 10
            default_chunk_growth_frames = 0
            default_max_chunk_frames = 10
            default_adaptive_chunks = False
        request = {
            "id": req_id,
            "text": text,
            "output_file": f"/tmp/trt_edgellm_tts_stream_{req_id}.wav",
            "language": kwargs.get("language") or _detect_language(text),
            "talker_temperature": self._config.talker_temperature,
            "talker_top_k": self._config.talker_top_k,
            "talker_top_p": self._config.talker_top_p,
            "repetition_penalty": self._config.repetition_penalty,
            "codec_eos_logit_offset": self._config.codec_eos_logit_offset,
            "predictor_temperature": self._config.predictor_temperature,
            "predictor_top_k": self._config.predictor_top_k,
            "predictor_top_p": self._config.predictor_top_p,
            "max_audio_length": kwargs.get("max_audio_length", self._config.max_audio_length),
            "min_audio_length": kwargs.get("min_audio_length", self._config.min_audio_length),
            "seed": int(kwargs.get("seed", self._config.seed)),
            "stream": True,
            "stream_only": True,
            # Precedence: per-call kwargs > config override (deploy) > profile default.
            "first_chunk_frames": kwargs.get(
                "first_chunk_frames",
                default_first_chunk_frames if self._config.first_chunk_frames is None
                else self._config.first_chunk_frames,
            ),
            "chunk_frames": kwargs.get(
                "chunk_frames",
                default_chunk_frames if self._config.chunk_frames is None
                else self._config.chunk_frames,
            ),
            "adaptive_chunks": kwargs.get(
                "adaptive_chunks",
                default_adaptive_chunks if self._config.adaptive_chunks is None
                else self._config.adaptive_chunks,
            ),
            "max_chunk_frames": kwargs.get(
                "max_chunk_frames",
                default_max_chunk_frames if self._config.max_chunk_frames is None
                else self._config.max_chunk_frames,
            ),
            "chunk_growth_frames": kwargs.get(
                "chunk_growth_frames",
                default_chunk_growth_frames if self._config.chunk_growth_frames is None
                else self._config.chunk_growth_frames,
            ),
            "chunk_format": "pcm_s16le",
            "chunk_transport": "base64",
        }
        voice_kwargs = self._resolve_voice_kwargs(kwargs)
        speaker_embedding = voice_kwargs.get("speaker_embedding")
        if speaker_embedding:
            request["speaker_embedding_b64"] = base64.b64encode(speaker_embedding).decode("ascii")
        else:
            self._add_speaker_request_fields(request, voice_kwargs)

        retry_after_empty = False
        emitted_chunks = 0
        done_event: dict | None = None
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker_io is not None
            worker_io = self._worker_io
        try:
            for event in worker_io.request(request):
                event_rid = _event_request_id(event)
                if event_rid is not None and event_rid != req_id and event_rid != "__worker__":
                    logger.debug(
                        "TTS worker event id mismatch: expected=%s got=%s event=%s",
                        req_id, event_rid, event.get("event"),
                    )
                if event.get("event") == "cancelled":
                    logger.info(
                        "TTS worker acknowledged cancel for %s (reason=%s)",
                        req_id, event.get("reason"),
                    )
                    return
                saturated = _tts_pool_saturated_error(event)
                if saturated is not None:
                    raise saturated
                if not event.get("ok"):
                    raise RuntimeError(f"TTS streaming worker failed: {event}")
                if event.get("event") == "chunk":
                    if event.get("chunk_transport") == "base64":
                        emitted_chunks += 1
                        yield base64.b64decode(event.get("audio_b64", ""))
                    elif event.get("chunk_file"):
                        with open(event["chunk_file"], "rb") as f:
                            payload = f.read()
                        try:
                            os.unlink(event["chunk_file"])
                        except OSError:
                            pass
                        if event.get("chunk_format") == "wav" and len(payload) > 44:
                            payload = payload[44:]
                        emitted_chunks += 1
                        yield payload
                elif event.get("event") == "done":
                    done_event = event
                    if meta_out is not None and isinstance(meta_out, dict):
                        meta_out.update(event)
                    if (
                        retry_empty
                        and self._config.stateful_code2wav_enabled()
                        and emitted_chunks == 0
                    ):
                        retry_after_empty = True
                        with self._worker_lock:
                            self._restart_worker_locked(
                                f"stateful stream returned 0 chunks for request {req_id}"
                            )
                    break
        except GeneratorExit:
            logger.info(
                "generator exit during TTS stream; cancelling worker for %s", req_id
            )
            try:
                worker_io.cancel(req_id)
            except Exception:
                logger.debug("worker_io.cancel() failed during GeneratorExit", exc_info=True)
            raise
        except WorkerExitError as exc:
            self._worker = None
            self._worker_io = None
            raise RuntimeError(
                f"TTS worker exited during stream: {self._worker_stderr_snip()}"
            ) from exc
        if retry_after_empty:
            logger.warning(
                "Retrying TTS stream after empty stateful result (done=%s stderr_tail=%s)",
                done_event, self._worker_stderr_snip(),
            )
            yield from self._generate_streaming_single(
                text, meta_out=meta_out, _retry_empty=False, **kwargs
            )

    def _synthesize_impl(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Run TTS inference via subprocess. Returns (wav_bytes, meta_dict)."""
        if not self._ready:
            raise RuntimeError("TTS backend not preloaded")

        if self._config.segment_text and kwargs.get("segment_text", True):
            segments = _split_tts_text(
                text,
                kwargs.get("segment_max_chars"),
                max_chars_latin=self._config.segment_max_chars_latin,
                max_chars_cjk=self._config.segment_max_chars_cjk,
            )
            if len(segments) > 1:
                segment_kwargs = dict(kwargs)
                segment_kwargs["segment_text"] = False
                segment_kwargs.setdefault("seed", self._config.seed)
                wav_parts: list[bytes] = []
                segment_meta: list[dict] = []
                total_elapsed = 0.0
                total_duration = 0.0
                total_samples = 0
                for segment in segments:
                    wav, meta = self._synthesize_impl(
                        segment, speaker_id=speaker_id, speed=speed,
                        pitch_shift=pitch_shift, language=language, **segment_kwargs,
                    )
                    wav_parts.append(wav)
                    segment_meta.append({"text": segment, **meta})
                    total_elapsed += float(meta.get("inference_time_s", 0.0))
                    wav_duration, wav_samples = _wav_duration_and_samples(wav)
                    total_duration += wav_duration
                    total_samples += wav_samples

                pauses_ms = [
                    _segment_pause_ms(
                        segment,
                        pause_ms=self._config.segment_pause_ms,
                        hard_pause_ms=self._config.segment_hard_pause_ms,
                    )
                    for segment in segments[:-1]
                ]
                wav_bytes = _concat_wav_bytes(wav_parts, pauses_ms)
                meta = {
                    "inference_time_s": round(total_elapsed, 3),
                    "sample_rate": self.sample_rate,
                    "duration_s": round(total_duration + sum(pauses_ms) / 1000.0, 3),
                    "samples": total_samples + int(self.sample_rate * sum(pauses_ms) / 1000.0),
                    "rtf": round(total_elapsed / total_duration, 3) if total_duration > 0 else 0.0,
                    "segmented": True,
                    "segment_count": len(segments),
                    "segment_pauses_ms": pauses_ms,
                    "segments": segment_meta,
                }
                return wav_bytes, meta

        return self._synthesize_single(
            text, speaker_id=speaker_id, speed=speed,
            pitch_shift=pitch_shift, language=language, **kwargs,
        )

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        if len(speaker_embedding) % 4 != 0:
            raise ValueError("speaker_embedding must be a float32 byte vector")
        return self._synthesize_impl(
            text, speed=speed, language=language,
            speaker_embedding=speaker_embedding, **kwargs,
        )

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        if not self._speaker_encoder or not os.path.exists(self._speaker_encoder):
            raise NotImplementedError(f"speaker encoder not found: {self._speaker_encoder}")
        return _qwen3_speaker_embed_inproc(audio_wav_bytes, self._speaker_encoder)

    def _synthesize_single(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Run one already-bounded TTS request."""
        if self._use_worker():
            return self._synthesize_worker(text, language, speaker_id=speaker_id, **kwargs)

        input_data = {
            "requests": [
                {
                    "messages": [{"role": "user", "content": text}],
                    "speaker": "",
                }
            ],
            "batch_size": 1,
            "apply_chat_template": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "talker_temperature": self._config.talker_temperature,
            "talker_top_k": self._config.talker_top_k,
            "talker_top_p": self._config.talker_top_p,
            "repetition_penalty": self._config.repetition_penalty,
            "codec_eos_logit_offset": self._config.codec_eos_logit_offset,
            "predictor_temperature": self._config.predictor_temperature,
            "predictor_top_k": self._config.predictor_top_k,
            "predictor_top_p": self._config.predictor_top_p,
            "max_audio_length": kwargs.get("max_audio_length", self._config.max_audio_length),
            "min_audio_length": kwargs.get("min_audio_length", self._config.min_audio_length),
        }
        voice_kwargs = self._resolve_voice_kwargs({"speaker_id": speaker_id, **kwargs})
        self._add_speaker_request_fields(input_data["requests"][0], voice_kwargs)

        with tempfile.TemporaryDirectory(prefix="trt_edgellm_tts_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            output_path = os.path.join(tmpdir, "output.json")
            audio_dir = os.path.join(tmpdir, "audio_out")
            os.makedirs(audio_dir, exist_ok=True)

            with open(input_path, "w") as f:
                json.dump(input_data, f)

            cli_args = [
                "--inputFile", input_path,
                "--talkerEngineDir", self._talker_dir,
                "--codePredictorEngineDir", self._code_predictor_dir,
                "--tokenizerDir", self._tokenizer_dir,
                "--outputFile", output_path,
                "--outputAudioDir", audio_dir,
            ]
            # Explicit-KV (highperf) flags — see _explicit_kv_flags.
            cli_args += self._explicit_kv_flags()

            c2w_path = _code2wav_engine_path(self._code2wav_dir)
            if os.path.exists(c2w_path):
                cli_args += ["--code2wavEngineDir", self._code2wav_dir]

            t0 = time.time()
            result = run_binary(
                self._config.tts_binary, cli_args, timeout=120,
                plugin_path=self._config.plugin_path,
            )
            elapsed = time.time() - t0

            if result.returncode != 0 or not os.path.exists(output_path):
                raise RuntimeError(
                    f"TTS subprocess failed (exit={result.returncode}): "
                    f"stdout={result.stdout[-300:]}, stderr={result.stderr[-300:]}"
                )

            with open(output_path) as f:
                output_data = json.load(f)

            responses = output_data.get("responses", [])
            if not responses:
                raise RuntimeError(f"TTS produced no responses: {output_data}")

            r = responses[0]
            audio_file = r.get("audio_file")
            wav_bytes = b""
            meta = {"inference_time_s": round(elapsed, 3), "sample_rate": 24000}

            if audio_file and os.path.exists(audio_file):
                with open(audio_file, "rb") as f:
                    wav_bytes = f.read()
                meta["duration_s"] = r.get("audio_duration_ms", 0) / 1000.0
                meta["samples"] = r.get("audio_samples", 0)
            else:
                logger.warning("No audio WAV in output, returning RVQ codes only")
                meta["rvq_file"] = r.get("rvq_file")
                if not meta.get("rvq_file"):
                    raise RuntimeError(
                        f"TTS output has neither audio nor RVQ: {list(r.keys())}"
                    )

            return wav_bytes, meta

    def _resolve_voice_kwargs(self, kwargs: dict) -> dict:
        sid = kwargs.get("speaker_id", kwargs.get("sid"))
        forward = {k: v for k, v in kwargs.items() if k not in ("speaker_id", "sid")}
        resolved = resolve_speaker_kwargs(self.model_id, speaker_id=sid, **forward)
        # BASE model: default to the fixed precomputed speaker embedding when the
        # caller supplied no speaker (no per-request embedding / id / name). A
        # per-request speaker always wins. No-op for CustomVoice (base unset).
        if self._base_speaker_embedding and not (resolved or {}).get("speaker_embedding") \
                and "speaker_embedding" not in kwargs \
                and not (resolved or {}).get("speaker_id") and "speaker_id" not in kwargs \
                and not (resolved or {}).get("speaker"):
            resolved = dict(resolved or {})
            resolved["speaker_embedding"] = self._base_speaker_embedding
        return resolved

    @staticmethod
    def _add_speaker_request_fields(request: dict, voice_kwargs: dict) -> None:
        if not voice_kwargs:
            return
        if "speaker_id" in voice_kwargs:
            request["speaker_id"] = int(voice_kwargs["speaker_id"])
        if "speaker" in voice_kwargs:
            request["speaker"] = str(voice_kwargs["speaker"])


def build_config_from_env(env: "dict | None" = None) -> TRTEdgeLLMTTSConfig:
    """Build TRTEdgeLLMTTSConfig from environment variables.

    All path fields are resolved from the passed ``env`` dict (or
    ``os.environ`` when None). Resolution logic mirrors the ``_deploy_paths``
    resolver functions but uses the supplied dict so callers can pass an
    explicit env override without touching the real process environment.
    Non-path fields (sampling, concurrency, streaming) are read the same way.
    Mirrors ``server.core.voxedge_backend_config.build_trt_edge_llm_tts_config``
    field-for-field.
    """
    import os as _os
    from . import _deploy_paths as dp

    if env is None:
        env = _os.environ

    # ---------------------------------------------------------------------------
    # Inline env-dict-aware path resolvers (mirror _deploy_paths but use ``env``)
    # ---------------------------------------------------------------------------

    def _e(key: str, default: str = "") -> str:
        return env.get(key, default) or default

    def _prefer_existing_e(primary: str, fallback: str) -> str:
        if primary and _os.path.exists(primary):
            return primary
        return fallback

    def _first_existing_dir_e(*paths: str) -> str:
        for p in paths:
            if p and _os.path.exists(p):
                return p
        return paths[-1] if paths else ""

    # qwen3 runtime profile (from env dict)
    def _qwen3_profile_e() -> str:
        raw = env.get("EDGE_LLM_QWEN3_PROFILE", env.get("OVS_QWEN3_PROFILE", "highperf"))
        return (raw or "highperf").strip().lower().replace("-", "_")

    def _highperf_e() -> bool:
        return _qwen3_profile_e() in ("highperf", "perf", "performance", "v2v")

    # TTS default root (env-based, mirrors dp._TTS_DEFAULT_ROOT logic)
    _tts_fixed = _os.path.expanduser("~/qwen3-tts-edgellm-runtime")
    _tts_export = _os.path.expanduser("~/qwen3-tts-trt-edge-llm-export")
    _tts_root = (
        _tts_fixed
        if _os.path.exists(_os.path.join(_tts_fixed, "engines", "talker", "llm.engine"))
        else _tts_export
    )

    # TTS worker binary
    def _resolve_worker_binary_e() -> str:
        explicit = env.get("EDGE_LLM_TTS_WORKER_BIN")
        if explicit:
            return explicit
        edge_base = env.get("EDGE_LLM_BASE", _os.path.expanduser("~/project/tensorrt-edge-llm"))
        edge_build = _os.path.join(edge_base, env.get("EDGE_LLM_BUILD_DIR", "build_sm87"))
        ovs_base = env.get("OVS_BASE", "")
        ovs_build = env.get(
            "OVS_WORKER_BUILD",
            _os.path.join(ovs_base, "build", "edgellm_voice_worker", "workers") if ovs_base else "",
        )
        return _prefer_existing_e(
            _os.path.join(ovs_build, "qwen3_tts_worker") if ovs_build else "",
            _os.path.join(edge_build, "examples/omni/qwen3_tts_worker"),
        )

    # TTS talker dir
    def _resolve_talker_dir_e() -> str:
        explicit = env.get("EDGE_LLM_TTS_TALKER_DIR")
        if explicit:
            return explicit
        default_talker = _os.path.join(_tts_root, "engines", "talker")
        full_dir = env.get("EDGE_LLM_TTS_FULL_TALKER_DIR", default_talker)
        pruned_dir = env.get("EDGE_LLM_TTS_PRUNED_TALKER_DIR", default_talker)
        vocab = env.get("EDGE_LLM_TTS_VOCAB_PRUNED", env.get("QWEN3_TTS_VOCAB_PRUNED", "0")).lower()
        if vocab in ("1", "true", "yes"):
            return pruned_dir
        if vocab in ("0", "false", "no"):
            return full_dir
        return default_talker

    # TTS code-predictor dir
    def _resolve_cp_dir_e() -> str:
        explicit = env.get("EDGE_LLM_TTS_CP_DIR")
        if explicit:
            return explicit
        talker_dir = _resolve_talker_dir_e()
        default_cp = _os.path.join(_os.path.dirname(talker_dir), "code_predictor")
        bf16_io_cp = env.get(
            "EDGE_LLM_TTS_CP_BF16_IO_DIR",
            "/tmp/qwen3_tts_cp_lmhead_pretranspose_0510/cp_dir",
        )
        if _highperf_e():
            return _first_existing_dir_e(bf16_io_cp, default_cp)
        return default_cp

    # TTS tokenizer dir
    def _resolve_tokenizer_dir_e() -> str:
        explicit = env.get("EDGE_LLM_TTS_TOKENIZER_DIR")
        if explicit:
            return explicit
        if _os.path.exists(_os.path.join(_tts_root, "processed_chat_template.json")):
            return _tts_root
        return _tts_export

    # TTS code2wav dir
    def _resolve_code2wav_dir_e() -> str:
        explicit = env.get("EDGE_LLM_TTS_CODE2WAV_DIR")
        if explicit:
            return explicit
        return _first_existing_dir_e(
            _os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder100_compat/code2wav"),
            _os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder_vocoder50_compat/code2wav"),
            _os.path.join(_tts_root, "engines", "code2wav"),
            _os.path.expanduser("~/qwen3-tts-trt-edge-llm-export/engines/tokenizer_decoder/code2wav"),
        )

    # TTS binary / plugin (import-time constants from _deploy_paths, but env override wins)
    def _tts_binary_e() -> str:
        explicit = env.get("EDGE_LLM_TTS_BIN")
        if explicit:
            return explicit
        edge_base = env.get("EDGE_LLM_BASE", _os.path.expanduser("~/project/tensorrt-edge-llm"))
        edge_build = _os.path.join(edge_base, env.get("EDGE_LLM_BUILD_DIR", "build_sm87"))
        return _os.path.join(edge_build, "examples/omni/qwen3_tts_inference")

    def _plugin_path_e() -> str:
        explicit = env.get("EDGELLM_PLUGIN_PATH")
        if explicit:
            return explicit
        edge_base = env.get("EDGE_LLM_BASE", _os.path.expanduser("~/project/tensorrt-edge-llm"))
        edge_build = _os.path.join(edge_base, env.get("EDGE_LLM_BUILD_DIR", "build_sm87"))
        return _os.path.join(edge_build, "libNvInfer_edgellm_plugin.so")

    # ---------------------------------------------------------------------------
    # Non-path field helpers
    # ---------------------------------------------------------------------------

    def _first(*names: str, default: str = "") -> str:
        """First non-empty env value among ``names``."""
        for name in names:
            v = env.get(name)
            if v not in (None, ""):
                return v
        return default

    def _fl(default: float, *names: str) -> float:
        try:
            return float(_first(*names, default=str(default)))
        except (TypeError, ValueError):
            return default

    def _in(default: int, *names: str) -> int:
        try:
            return int(_first(*names, default=str(default)))
        except (TypeError, ValueError):
            return default

    def _flag(name: str, default: bool) -> bool:
        v = env.get(name)
        if v is None:
            return default
        return v.lower() not in ("0", "false", "no", "off")

    # -- speaker encoder: QWEN3_SPEAKER_ENCODER → QWEN3_ARTIFACT_ROOT probe →
    #    <model_base>/onnx/speaker_encoder.onnx  (legacy _resolve_speaker_encoder)
    qwen3_tts_model_base = _first(
        "OVS_TTS_MODEL_BASE", "QWEN3_MODEL_BASE", default="/opt/models/qwen3-tts"
    )
    speaker_encoder = env.get("QWEN3_SPEAKER_ENCODER", "") or ""
    if not speaker_encoder:
        qwen3_root = env.get("QWEN3_ARTIFACT_ROOT", "")
        candidate = ""
        if qwen3_root:
            candidate = _os.path.join(
                qwen3_root, "tts", "speaker_encoder", "speaker_encoder.onnx"
            )
            if not _os.path.exists(candidate):
                candidate = ""
        speaker_encoder = candidate or _os.path.join(
            qwen3_tts_model_base, "onnx", "speaker_encoder.onnx"
        )

    # -- worker_concurrency: env → 1.
    env_conc = env.get("OVS_TTS_WORKER_CONCURRENCY")
    if env_conc is not None:
        try:
            worker_concurrency = int(env_conc)
        except ValueError:
            worker_concurrency = 1
    else:
        worker_concurrency = 1
    worker_concurrency = max(1, worker_concurrency)

    # -- stateful_code2wav: leave None when unset so dataclass derives from
    #    runtime profile; pass explicit bool otherwise.
    stateful_raw = env.get("EDGE_LLM_TTS_STATEFUL_CODE2WAV")
    if stateful_raw is None:
        stateful_code2wav = None
    else:
        stateful_code2wav = stateful_raw.lower() not in ("0", "false", "no", "off")

    # Streaming-native worker (v0.9.0 lean code2wav has no output_file mode).
    streaming_only_worker = str(
        env.get("EDGE_LLM_TTS_STREAMING_ONLY", "0")
    ).lower() not in ("0", "false", "no", "off", "")

    model_id = env.get("OVS_TTS_MODEL_ID") or "trt_edgellm"

    # BASE-model fixed speaker embedding
    def _resolve_base_spk_embed_b64() -> str:
        direct = (env.get("EDGE_LLM_TTS_BASE_SPK_EMBED_B64") or "").strip()
        if direct:
            return direct
        path = (env.get("EDGE_LLM_TTS_BASE_SPK_EMBED_PATH") or "").strip()
        if path and _os.path.exists(path):
            try:
                return open(path).read().strip()
            except Exception:
                return ""
        return ""
    base_speaker_embedding_b64 = _resolve_base_spk_embed_b64()

    return TRTEdgeLLMTTSConfig(
        # --- paths (all resolved from passed env dict) ---
        tts_binary=_tts_binary_e(),
        worker_binary=_resolve_worker_binary_e(),
        plugin_path=_plugin_path_e(),
        talker_dir=_resolve_talker_dir_e(),
        talker_backend=_first("EDGE_LLM_TTS_TALKER_BACKEND"),
        talker_engine=_first("EDGE_LLM_TTS_TALKER_ENGINE"),
        code_predictor_backend=_first("EDGE_LLM_TTS_CODE_PREDICTOR_BACKEND"),
        text_projection=_first("EDGE_LLM_TTS_TEXT_PROJECTION"),
        prompt_kv_cache=_first("EDGE_LLM_TTS_PROMPT_KV_CACHE"),
        code_predictor_dir=_resolve_cp_dir_e(),
        tokenizer_dir=_resolve_tokenizer_dir_e(),
        code2wav_dir=_resolve_code2wav_dir_e(),
        speaker_encoder=speaker_encoder,
        base_speaker_embedding_b64=base_speaker_embedding_b64,
        # --- identity ---
        model_id=model_id,
        backend_mode=_first("OVS_TTS_BACKEND", "EDGE_LLM_TTS_BACKEND", default="edgellm_worker"),
        # --- concurrency ---
        use_worker=_flag("EDGE_LLM_TTS_WORKER", True),
        worker_concurrency=worker_concurrency,
        # --- runtime profile ---
        qwen3_runtime_profile=_qwen3_profile_e(),
        perf_profile=env.get("EDGE_LLM_TTS_PERF_PROFILE", "quality"),
        stateful_code2wav=stateful_code2wav,
        streaming_only_worker=streaming_only_worker,
        # --- sampling ---
        seed=_in(42, "OVS_TTS_SEED"),
        talker_temperature=_fl(0.9, "OVS_TTS_TALKER_TEMPERATURE", "TTS_TALKER_TEMPERATURE"),
        talker_top_k=_in(50, "OVS_TTS_TALKER_TOP_K", "TTS_TALKER_TOP_K"),
        talker_top_p=_fl(1.0, "OVS_TTS_TOP_P", "TTS_TOP_P"),
        predictor_temperature=_fl(0.9, "OVS_TTS_PREDICTOR_TEMPERATURE", "TTS_PREDICTOR_TEMPERATURE"),
        predictor_top_k=_in(50, "OVS_TTS_PREDICTOR_TOP_K", "TTS_PREDICTOR_TOP_K"),
        predictor_top_p=_fl(1.0, "OVS_TTS_PREDICTOR_TOP_P", "TTS_PREDICTOR_TOP_P"),
        max_audio_length=_in(1024, "TTS_MAX_AUDIO_LENGTH"),
        min_audio_length=_in(30, "TTS_MIN_AUDIO_LENGTH"),
        repetition_penalty=_fl(1.05, "TTS_REPETITION_PENALTY"),
        codec_eos_logit_offset=_fl(0.0, "TTS_CODEC_EOS_LOGIT_OFFSET"),
        # --- text segmentation ---
        segment_text=_flag("EDGE_LLM_TTS_SEGMENT_TEXT", True),
        segment_max_chars_latin=_in(120, "EDGE_LLM_TTS_SEGMENT_MAX_CHARS"),
        segment_max_chars_cjk=_in(48, "EDGE_LLM_TTS_CJK_SEGMENT_MAX_CHARS"),
        segment_pause_ms=_in(80, "EDGE_LLM_TTS_SEGMENT_PAUSE_MS"),
        segment_hard_pause_ms=_in(120, "EDGE_LLM_TTS_HARD_SEGMENT_PAUSE_MS"),
        # --- streaming ---
        streaming_profile=env.get("EDGE_LLM_TTS_STREAMING_PROFILE", "continuous_playback"),
        first_chunk_frames=(
            int(env["EDGE_LLM_TTS_FIRST_CHUNK_FRAMES"])
            if "EDGE_LLM_TTS_FIRST_CHUNK_FRAMES" in env else None
        ),
        chunk_frames=(
            int(env["EDGE_LLM_TTS_CHUNK_FRAMES"])
            if "EDGE_LLM_TTS_CHUNK_FRAMES" in env else None
        ),
        adaptive_chunks=(
            env["EDGE_LLM_TTS_ADAPTIVE_CHUNKS"].strip().lower() in ("1", "true", "yes", "on")
            if "EDGE_LLM_TTS_ADAPTIVE_CHUNKS" in env else None
        ),
        max_chunk_frames=(
            int(env["EDGE_LLM_TTS_MAX_CHUNK_FRAMES"])
            if "EDGE_LLM_TTS_MAX_CHUNK_FRAMES" in env else None
        ),
        chunk_growth_frames=(
            int(env["EDGE_LLM_TTS_CHUNK_GROWTH_FRAMES"])
            if "EDGE_LLM_TTS_CHUNK_GROWTH_FRAMES" in env else None
        ),
    )
