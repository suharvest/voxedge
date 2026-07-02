"""SparkTTS controllable-TTS backend (Jetson) via the ``spark_tts_worker`` subprocess.

The worker (jetson-voice-engine ``native/edgellm_voice_worker/spark_tts_worker``) speaks
the JSON-line protocol over stdio: a ``ready`` event on startup, then per-request
``token_progress`` / ``chunk`` (base64 PCM s16le) / ``done`` / ``cancelled`` / ``error``
events keyed by request ``id``. We reuse :class:`WorkerIO` for the same request-id
multiplexing + ``threading.Semaphore(worker_concurrency)`` back-pressure used by the
qwen3 TTS backend, and ``{"type":"cancel","id":...}`` for mid-stream cancel.

SparkTTS supports TWO voice-selection modes over the SAME worker / LLM / vocoder
(spec §7, they share the engine and differ only in the prompt the worker builds):

  * CONTROLLABLE — voice selected via ``gender`` / ``pitch`` / ``speed`` style labels.
    The controllable prompt is built INSIDE the worker; this backend forwards the three
    style fields (``mode:"controllable"``, the worker default).
  * CLONE (zero-shot) — voice selected via a ``voice``/``speaker`` value that names a
    registered VoiceProfile (host-enrolled, spec §10). This backend looks the id up in a
    :class:`VoiceRegistry` and forwards ``mode:"clone"`` + ``global_ids[32]`` (+ optional
    strategy-B ``ref_semantic_ids`` / ``ref_text``); the worker conditions timbre on the
    reference's global tokens. NO reference-audio analysis happens here or on-device — the
    analysis chain runs once at enrollment on a GPU host (spec §3.2).

Routing: a ``voice``/``speaker`` string that hits the registry → clone; otherwise it is
parsed as a controllable ``gender_pitch_speed`` spec (back-compat). Output is 16 kHz mono
PCM s16le for both modes.

Zero env reads at import / construction (the ``trt_edge_llm_tts_env_staleness`` pitfall):
every path/param is an explicit :class:`SparkTTSConfig` field, read once in ``preload`` /
``_ensure_worker``. The product layer builds the config from env+profile.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import threading
import uuid
import wave
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from voxedge.backends.base import TTSBackend, TTSCapability
from voxedge.backends.jetson.worker_io import WorkerIO, WorkerExitError
from voxedge.backends.jetson.voice_registry import VoiceRegistry, VoiceProfile
from voxedge.engine.concurrency_capability import ConcurrencyCapability

logger = logging.getLogger(__name__)


# ── env/profile → config map (defaults match the spike layout; product overrides) ──
#   SPARKTTS_WORKER_BINARY        → worker_binary
#   SPARKTTS_PLUGIN_PATH          → plugin_path (LD_PRELOAD'd edge-llm plugin .so)
#   SPARKTTS_LLM_ENGINE_DIR       → llm_engine_dir (mixed-precision LLM engine dir)
#   SPARKTTS_TOKENIZER_DIR        → tokenizer_dir (defaults to llm_engine_dir)
#   SPARKTTS_BICODEC_ENGINE       → bicodec_engine (.engine file)
#   SPARKTTS_SPEAKER_DECODER_ENGINE → speaker_decoder_engine (.engine file)
#   SPARKTTS_LD_LIBRARY_PATH      → ld_library_path (prepended; edge-llm build dir)
#   profile sparktts_worker_concurrency / tts_worker_concurrency → worker_concurrency (1)


@dataclass
class SparkTTSConfig:
    """Explicit construction-time config for :class:`SparkTTSBackend` (no env reads)."""

    worker_binary: str = "/opt/jv-workers/spark_tts_worker"
    plugin_path: str = "/opt/edgellm/libNvInfer_edgellm_plugin.so"
    llm_engine_dir: str = "/opt/models/sparktts-0p5b/llm_engine"
    tokenizer_dir: Optional[str] = None  # defaults to llm_engine_dir
    bicodec_engine: str = "/opt/models/sparktts-0p5b/bicodec_decoder_dynT.fp16.engine"
    speaker_decoder_engine: str = "/opt/models/sparktts-0p5b/sparktts_speaker_decoder.fp32.engine"
    ld_library_path: Optional[str] = None  # prepended to LD_LIBRARY_PATH (edge-llm build dir)
    sample_rate: int = 16000

    # streaming / generation
    first_chunk_tokens: int = 6   # S3 sweet spot (~0.92s TTFA on Orin NX)
    chunk_tokens: int = 16
    left_overlap_tokens: int = 12
    max_tokens: int = 800         # runaway cap (mixed LLM may not emit EOS on some ZH)
    max_semantic: int = 600       # BiCodec dynamic-T profile ceiling
    temperature: float = 1.0
    top_k: int = 1                # greedy (matches validated e2e)
    top_p: float = 1.0

    # default style (controllable) — overridable per request
    default_gender: str = "female"
    default_pitch: str = "moderate"
    default_speed: str = "moderate"

    # voice clone (registry of host-enrolled VoiceProfiles; spec §4.3)
    #   voices_dir: directory of <voice_id>.json + .npz pairs (None → clone disabled).
    #   clone_use_ref_semantic: strategy B (ref-semantic in-context prefix) when a profile
    #     carries it; False forces strategy A (global-only, shorter prompt / faster TTFA).
    voices_dir: Optional[str] = None
    clone_use_ref_semantic: bool = False  # spike P0: strategy A slightly better + faster

    # concurrency: gates worker --max_slots (N>1 enables supports_parallel)
    worker_concurrency: int = 1

    model_id: str = "sparktts-0p5b"
    extra_env: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.tokenizer_dir is None:
            self.tokenizer_dir = self.llm_engine_dir
        self.worker_concurrency = max(1, int(self.worker_concurrency))


class SparkTTSBackend(TTSBackend):
    """SparkTTS controllable TTS via the spark_tts_worker subprocess."""

    supports_hot_reload = False
    _REQUEST_TIMEOUT_S = 120.0

    def __init__(self, config: Optional[SparkTTSConfig] = None):
        self._config = config or SparkTTSConfig()
        self._ready = False
        self._worker: Optional[subprocess.Popen] = None
        self._worker_lock = threading.Lock()
        self._worker_io: Optional[WorkerIO] = None
        self._worker_concurrency = max(1, int(self._config.worker_concurrency))
        self._worker_stderr_tail: deque[str] = deque(maxlen=80)
        self._sample_rate = int(self._config.sample_rate)
        # Clone voice registry (spec §4.3); None voices_dir → empty registry (clone off).
        self._voices = VoiceRegistry(self._config.voices_dir)

    @property
    def voices(self) -> VoiceRegistry:
        """Clone voice registry (``voice_id`` → :class:`VoiceProfile`). The product/OVS
        layer uses ``.reload()`` after registering/deleting a profile."""
        return self._voices

    # -- TTSBackend interface ------------------------------------------------
    @property
    def name(self) -> str:
        return "jetson.sparktts"

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = {
            TTSCapability.BASIC_TTS,
            TTSCapability.MULTI_LANGUAGE,
            TTSCapability.STREAMING,
        }
        # Advertise voice clone only when a registry directory is configured. The
        # selection key is a registered ``voice_id`` (NOT a raw speaker embedding):
        # clone here is reference-token based (host-enrolled VoiceProfile), so this
        # backend does not consume ``speaker_embedding`` bytes the way CustomVoice does.
        if self._config.voices_dir:
            caps.add(TTSCapability.VOICE_CLONE)
            caps.add(TTSCapability.VOICE_CLONE_ICL)  # strategy B (ref-semantic in-context)
        return caps

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def rate_pitch_caps(self) -> tuple[bool, bool]:
        # SparkTTS exposes pitch/speed as discrete STYLE LABELS (very_low..very_high),
        # not continuous factors, so we do NOT advertise native continuous speed/pitch;
        # continuous requests go through the DSP fallback. Discrete style control is via
        # the gender/pitch/speed request fields handled in _build_request.
        return (False, False)

    def concurrency_capability(self) -> ConcurrencyCapability:
        """Slot-pool ceiling. WorkerIO multiplexes N in-flight requests over one
        subprocess whose --max_slots == worker_concurrency (each slot = own LLM
        execution context + KV + vocoder contexts, per-slot isolated)."""
        n = max(1, int(self._config.worker_concurrency))
        return ConcurrencyCapability(
            supports_parallel=n > 1,
            max_concurrent=n,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    def is_ready(self) -> bool:
        return self._ready

    # -- lifecycle -----------------------------------------------------------
    def preload(self) -> None:
        """Verify required artifacts exist, then start + warm the worker."""
        cfg = self._config
        required = [
            (cfg.worker_binary, "spark_tts_worker binary"),
            (cfg.plugin_path, "edge-llm plugin .so"),
            (os.path.join(cfg.llm_engine_dir, "llm.engine"), "LLM engine"),
            (os.path.join(cfg.tokenizer_dir, "tokenizer.json"), "tokenizer"),
            (cfg.bicodec_engine, "BiCodec vocoder engine"),
            (cfg.speaker_decoder_engine, "speaker_decoder engine"),
        ]
        missing = [f"{label}: {path}" for path, label in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError("SparkTTS preload failed — missing:\n  " + "\n  ".join(missing))
        with self._worker_lock:
            self._ensure_worker()
        self._ready = True
        logger.info("SparkTTS backend preload OK (binary=%s concurrency=%d)",
                    cfg.worker_binary, self._worker_concurrency)

    def _worker_env(self) -> dict:
        env = dict(os.environ)
        # LD_PRELOAD the edge-llm plugin (the worker loads it via dlopen too, but the
        # relative-path fallback warns; preloading is the proven invocation).
        env["LD_PRELOAD"] = self._config.plugin_path
        if self._config.ld_library_path:
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = self._config.ld_library_path + (":" + prev if prev else "")
        if self._config.extra_env:
            env.update({str(k): str(v) for k, v in self._config.extra_env.items()})
        return env

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        cfg = self._config
        self._worker_concurrency = max(1, int(cfg.worker_concurrency))
        cmd = [
            cfg.worker_binary,
            "--llmEngineDir", cfg.llm_engine_dir,
            "--speakerDecoderEngine", cfg.speaker_decoder_engine,
            "--bicodecEngine", cfg.bicodec_engine,
            "--sampleRate", str(int(cfg.sample_rate)),
        ]
        # spark_tts_worker uses camelCase --maxSlots (accepts it at N=1 too).
        cmd += ["--maxSlots", str(self._worker_concurrency)]
        logger.info("Starting spark_tts_worker: %s", " ".join(cmd))
        self._worker = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=self._worker_env(),
        )
        threading.Thread(target=self._drain_stderr, name="sparktts-worker-stderr", daemon=True).start()
        assert self._worker.stdout is not None
        ready_line = self._worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"spark_tts_worker failed to start: {self._stderr_snip()}")
        ready = json.loads(ready_line)
        if ready.get("event") != "ready":
            raise RuntimeError(f"spark_tts_worker did not become ready: {ready}")
        logger.info("spark_tts_worker ready: %s", ready)
        self._worker_io = WorkerIO(self._worker, self._worker_concurrency)

    def _drain_stderr(self) -> None:
        proc = self._worker
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._worker_stderr_tail.append(line.rstrip())
        except Exception:
            pass

    def _stderr_snip(self) -> str:
        return " | ".join(list(self._worker_stderr_tail)[-8:])

    def shutdown(self) -> None:
        with self._worker_lock:
            wio, self._worker_io = self._worker_io, None
            proc, self._worker = self._worker, None
            self._ready = False
        if wio is not None:
            try:
                wio.close()
            except Exception:
                pass
        if proc is not None:
            try:
                proc.terminate(); proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def unload(self) -> None:
        self.shutdown()

    # -- request construction (controllable style mapping) -------------------
    @staticmethod
    def _norm_level(v: Any, default: str) -> str:
        """Map a request value to a SparkTTS style label. Accepts the label
        strings directly (very_low/low/moderate/high/very_high) or 0-4."""
        if v is None:
            return default
        s = str(v).strip().lower()
        return s if s else default

    def _parse_style(self, kwargs: dict) -> tuple[str, str, str]:
        """Resolve SparkTTS controllable style (gender, pitch, speed) for a request.

        IMPORTANT: the public ``generate_streaming``/``synthesize`` wrapper reserves
        ``speed``/``pitch``/``pitch_shift`` for CONTINUOUS DSP rate/pitch shifting, so
        SparkTTS's DISCRETE style labels must NOT be read from those keys (a string like
        "moderate" would crash the DSP shifter). Style is resolved, in priority order:
          1. explicit ``style_gender`` / ``style_pitch`` / ``style_speed`` kwargs
          2. a ``speaker``/``voice`` spec string ``"<gender>_<pitch>_<speed>"``
             (e.g. ``"female_moderate_high"``; missing parts fall through to defaults)
          3. config defaults
        """
        cfg = self._config
        gender = pitch = speed = None
        spec = kwargs.get("speaker") or kwargs.get("voice")
        if isinstance(spec, str) and spec.strip():
            parts = spec.strip().lower().split("_")
            # gender token may itself be "very_low"-free; gender ∈ {female,male}
            if parts and parts[0] in ("female", "male"):
                gender = parts[0]
                rest = parts[1:]
            else:
                rest = parts
            # remaining tokens map to pitch then speed; allow "very_low"/"very_high" (2 tokens)
            def _take_level(toks: list[str]) -> tuple[Optional[str], list[str]]:
                if not toks:
                    return None, toks
                if toks[0] == "very" and len(toks) >= 2 and toks[1] in ("low", "high"):
                    return f"very_{toks[1]}", toks[2:]
                return toks[0], toks[1:]
            pitch, rest = _take_level(rest)
            speed, rest = _take_level(rest)
        gender = self._norm_level(kwargs.get("style_gender", gender), cfg.default_gender)
        pitch = self._norm_level(kwargs.get("style_pitch", pitch), cfg.default_pitch)
        speed = self._norm_level(kwargs.get("style_speed", speed), cfg.default_speed)
        return gender, pitch, speed

    def _resolve_voice_profile(self, kwargs: dict) -> Optional[VoiceProfile]:
        """Return a clone VoiceProfile if the request selects a registered voice.

        Selection keys (first hit wins):
          1. ``voice_profile`` — an already-loaded :class:`VoiceProfile` (callers may
             inject one directly, bypassing the registry).
          2. ``voice_id`` — explicit registry key.
          3. ``voice`` / ``speaker`` — the generic selection string; a hit in the
             registry means clone, a miss falls through to controllable style parsing.
        Returns ``None`` → controllable mode.
        """
        vp = kwargs.get("voice_profile")
        if isinstance(vp, VoiceProfile):
            return vp
        for key in ("voice_id", "voice", "speaker"):
            val = kwargs.get(key)
            if isinstance(val, str) and val.strip():
                prof = self._voices.get(val.strip())
                if prof is not None:
                    return prof
        return None

    def _build_request(self, req_id: str, text: str, *, stream: bool, kwargs: dict) -> dict:
        cfg = self._config
        req = {
            "id": req_id,
            "text": text,
            "top_k": int(kwargs.get("top_k", cfg.top_k)),
            "temperature": float(kwargs.get("temperature", cfg.temperature)),
            "top_p": float(kwargs.get("top_p", cfg.top_p)),
            "max_tokens": int(kwargs.get("max_tokens", cfg.max_tokens)),
            "max_semantic": int(kwargs.get("max_semantic", cfg.max_semantic)),
            "stream_audio": bool(stream),
            "first_chunk_tokens": int(kwargs.get("first_chunk_tokens", cfg.first_chunk_tokens)),
            "chunk_tokens": int(kwargs.get("chunk_tokens", cfg.chunk_tokens)),
            "left_overlap_tokens": int(kwargs.get("left_overlap_tokens", cfg.left_overlap_tokens)),
            "chunk_transport": "base64",
        }
        profile = self._resolve_voice_profile(kwargs)
        if profile is not None:
            # CLONE: forward mode + global_ids[32] (+ strategy-B ref-semantic/ref-text).
            # Per-request override of strategy B via use_ref_semantic kwarg.
            use_ref = bool(kwargs.get("use_ref_semantic", cfg.clone_use_ref_semantic))
            req.update(profile.worker_request_fields(use_ref_semantic=use_ref))
        else:
            # CONTROLLABLE: gender/pitch/speed style labels (worker default mode).
            gender, pitch, speed = self._parse_style(kwargs)
            req["gender"] = gender
            req["pitch"] = pitch
            req["speed"] = speed
        return req

    def _worker_io_locked(self) -> WorkerIO:
        with self._worker_lock:
            self._ensure_worker()
            assert self._worker_io is not None
            return self._worker_io

    # -- streaming -----------------------------------------------------------
    def _generate_streaming_impl(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        cancel_token: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[bytes]:
        """Yield raw PCM s16le @16k chunks from the resident SparkTTS worker."""
        if not text or not text.strip():
            return
        # ``speaker`` is captured as a named kwarg by the public wrapper; fold it back
        # so voice/clone routing in _build_request sees it (registry hit → clone).
        if speaker is not None:
            kwargs.setdefault("speaker", speaker)
        req_id = uuid.uuid4().hex
        request = self._build_request(req_id, text, stream=True, kwargs=kwargs)
        worker_io = self._worker_io_locked()
        try:
            for event in worker_io.request(request):
                if cancel_token is not None and getattr(cancel_token, "cancelled", False):
                    worker_io.cancel(req_id)
                ev = event.get("event")
                if ev == "cancelled":
                    logger.info("SparkTTS cancelled %s", req_id)
                    return
                if not event.get("ok", True) and ev not in ("token_progress",):
                    raise RuntimeError(f"SparkTTS worker failed: {event}")
                if ev == "chunk":
                    b64 = event.get("audio_b64")
                    if b64:
                        yield base64.b64decode(b64)
                # token_progress / done: nothing to yield (done terminates the iterator)
        except WorkerExitError as e:
            raise RuntimeError(f"SparkTTS worker died mid-request: {self._stderr_snip()}") from e

    # -- non-streaming -------------------------------------------------------
    def _synthesize_impl(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[bytes, dict]:
        """Synthesize to WAV bytes (mono s16le @16k). Drives the streaming path and
        assembles the PCM, so the worker stays in one code path."""
        pcm = bytearray()
        for chunk in self._generate_streaming_impl(text, language=language, **kwargs):
            pcm += chunk
        wav_bytes = self._pcm_to_wav(bytes(pcm), self._sample_rate, channels=1)
        meta = {"sample_rate": self._sample_rate, "channels": 1, "format": "wav",
                "samples": len(pcm) // 2, "backend": self.name}
        return wav_bytes, meta

    @staticmethod
    def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()
