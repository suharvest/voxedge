"""voxedge backend abstract base classes.

These mirror the **public** API of the production backend ABCs in
``app/core/`` and ``agent/openvoicestream_agent/llm/`` but are deliberately
DECOUPLED from env / profile loading: every constructor takes explicit
parameters only and nothing here reads ``os.environ`` or calls
``current_profile()``.

Source-of-truth mapping (read-only reference, NOT imported):
  * ASRBackend / ASRStream  ← app/core/asr_backend.py:36-147
  * TTSBackend              ← app/core/tts_backend.py:31-142
  * VADBackend / VADSession ← app/core/vad.py:90-109, 244-257
  * LLMBackend / LLMEvent   ← agent/openvoicestream_agent/llm/base.py

The LLM contract is reproduced here (not imported) on purpose: importing the
``agent`` package would drag in openai / httpx and break the "pure Python, no
heavy deps" guarantee. When the agent layer is itself packaged into voxedge
(spec §0.1 layer ②) these can be unified.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, ClassVar, Iterator, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ASR
# ═══════════════════════════════════════════════════════════════════════


class ASRCapability(str, Enum):
    """Mirrors app/core/asr_backend.py:21-26."""

    OFFLINE = "offline"
    STREAMING = "streaming"
    TIMESTAMPS = "timestamps"
    MULTI_LANGUAGE = "multi_language"
    LANGUAGE_ID = "language_id"


@dataclass
class TranscriptionResult:
    """Mirrors app/core/asr_backend.py:29-33.

    ``language`` is what the backend ACTUALLY decoded with — never an echo of
    an unhonoured request. ``None`` means the backend performs no language
    selection at all (a fixed single/dual-language model). Use
    :func:`resolve_reported_language` to derive it.
    """

    text: str
    language: Optional[str] = None
    meta: Optional[dict] = None


def resolve_reported_language(
    requested: Optional[str],
    *,
    honoured: Optional[str],
    backend: str,
    warned: set,
) -> Optional[str]:
    """Reconcile the language a caller asked for with the one a backend can deliver.

    Backends differ in how much of a per-request language they can honour:
    SenseVoice binds it when the recognizer is constructed, SenseVoice-TRT maps
    it per call, Paraformer has no language switch whatsoever. What they must
    NOT do is accept the parameter, ignore it, and hand it back in
    ``TranscriptionResult.language`` — a caller asking for ``zh`` then gets a
    reply claiming ``zh`` from a decoder that never saw the request.

    ``honoured`` is what the backend will really use (``None`` when it selects
    no language). The return value is what belongs in the result. A request
    that cannot be satisfied is reported once per distinct value via ``warned``,
    so a per-utterance caller does not flood the log.
    """
    # Both sides are normalised: ``honoured`` often comes from a config/env
    # value that was only stripped, so comparing it raw made a pin of "ZH" and
    # a request of "zh" look like a conflict — a spurious warning plus the odd
    # casing leaking into the reported language.
    requested = (requested or "auto").strip().lower() or "auto"
    honoured = honoured.strip().lower() or None if honoured is not None else None
    if requested == "auto":
        return honoured
    if requested != (honoured or ""):
        if requested not in warned:
            warned.add(requested)
            if honoured is None:
                logger.warning(
                    "%s: per-request language %r ignored — this backend selects no "
                    "language; its model is fixed.", backend, requested,
                )
            else:
                logger.warning(
                    "%s: per-request language %r ignored — decoding as %r. Pin it in "
                    "the backend config instead.", backend, requested, honoured,
                )
    return honoured


class ASRStream(ABC):
    """A streaming ASR session that accumulates audio and produces text.

    Public API mirrors app/core/asr_backend.py:36-91.
    """

    prefer_backend_endpoint_vad: bool = False
    """Whether this stream owns endpoint VAD finalization."""

    immediate_client_eos_cancel_safe: bool = False
    """Whether partial abort can run outside normal ASR serialization."""

    #: 本类的 ``close()`` 是否需要释放东西。**每个子类必须显式声明。**
    #:
    #: 这不是样板：``close()`` 在基类里曾是空实现，于是
    #: ``close = getattr(stream, "close", None); if close is not None: close()``
    #: 这种鸭子类型兜底永远拿得到一个什么都不做的方法，判断从来拦不住任何东西。
    #: 子类忘了实现 close 会静默继承空壳 —— 调用"成功"、资源不释放、零报错。
    #: 2026-08-06 的 ASR worker 槽位泄漏正是这么活下来的：
    #: ``_TRTEdgeLLMStreamingASRStream`` 从没实现 close，服务端每次都"成功"调用了
    #: 继承来的空函数，而 max_slots=1 的槽位一直漏，第二轮起 ASR 直接死掉。
    #:
    #: 声明为 True 却没实现 ``close()`` 会在**类定义时**（即 import 时）抛
    #: TypeError —— 漏写立刻炸，而不是几个月后有人去查延迟才发现。
    OWNS_RESOURCES: ClassVar[Optional[bool]] = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        owns = getattr(cls, "OWNS_RESOURCES", None)
        if owns is None:
            raise TypeError(
                f"{cls.__name__} 必须显式声明 OWNS_RESOURCES："
                "True 表示 close() 要释放资源（并且必须实现 close），"
                "False 表示确认没有需要释放的每流资源。"
                "不允许留空 —— 那正是让忘记实现 close 变得不可察觉的原因。"
            )
        if owns and cls.close is ASRStream.close:
            raise TypeError(
                f"{cls.__name__} 声明了 OWNS_RESOURCES=True 却没有实现 close()，"
                "继承来的基类实现是空操作，资源不会被释放。"
            )

    @abstractmethod
    def accept_waveform(self, sample_rate: int, samples: "np.ndarray") -> None:
        """Feed audio samples (float32, [-1,1]) into the stream."""
        ...

    @abstractmethod
    def finalize(self) -> tuple[str, Optional[str]]:
        """Signal end-of-audio. Returns ``(final_text, detected_language)``.

        ``detected_language`` is the human-readable language name if the
        backend supports language ID, otherwise ``None``.
        """
        ...

    def get_partial(self) -> tuple[str, bool]:
        """Return ``(partial_text, is_endpoint)``. Default: no partials."""
        return "", False

    def prepare_finalize(self) -> None:
        """Optional: pre-encode remaining audio so finalize() runs decoder only."""

    def cancel_and_finalize(self) -> None:
        """Hard-cancel in-flight decode + skip residual tail encode (barge-in)."""

    def cancel(self) -> None:
        """Symmetric alias for :meth:`cancel_and_finalize`."""
        self.cancel_and_finalize()

    def close(self) -> None:
        """Release per-stream resources. Safe to call twice.

        Subclasses declaring ``OWNS_RESOURCES = True`` must override this;
        ``__init_subclass__`` refuses to define a class that declares ownership
        without implementing it. For ``OWNS_RESOURCES = False`` this no-op is
        the correct behaviour and callers can invoke it unconditionally.
        """


def _resample_linear(samples, src_sr: int, target_sr: int):
    """Dependency-free resample: box-filter decimation, then linear interpolation.

    Interpolation alone is fine going UP, and wrong going down: content above
    the new Nyquist folds back into the band instead of disappearing. A 12 kHz
    tone in 48 kHz input became a full-amplitude 4 kHz component after
    resampling to 16 kHz — squarely inside the speech band, and indistinguishable
    from real audio to everything downstream.

    A moving average over the decimation ratio is the cheapest honest answer:
    it is a crude low-pass, but it attenuates rather than folds, and it keeps
    this helper numpy-only. It is still not a resampler; feeding the backend's
    own rate remains the right thing to do.
    """
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32)
    if src_sr == target_sr or samples.size == 0:
        return samples
    if target_sr < src_sr:
        width = int(src_sr // target_sr)
        if width > 1 and samples.size >= width:
            kernel = np.ones(width, dtype=np.float32) / width
            samples = np.convolve(samples, kernel, mode="same").astype(np.float32)
    n_out = int(round(samples.size * target_sr / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


class OfflineAccumulateStream(ASRStream):
    """Generic offline→streaming adapter.

    Wraps any offline backend (one that implements ``transcribe_array``) into a
    streaming session by accumulating audio and transcribing the whole utterance
    on ``finalize()``. Endpointing is delegated to the OVS server-side VAD (which
    finalizes + recreates the stream per utterance), so there is NO internal VAD
    and NO incremental partial — first visible text lands at finalize.

    Any backend that sets ``supports_offline_streaming = True`` gets a streaming
    session for free via ``ASRBackend.create_stream`` — no per-backend stream code.
    """

    #: 每流累积缓冲 + 自实现 close 清理它。
    OWNS_RESOURCES = True

    def __init__(self, backend: "ASRBackend", language: str = "auto") -> None:
        self._backend = backend
        self._language = language
        self._buf: list = []
        self._warned_rates: set = set()
        self._rate: Optional[int] = None

    def accept_waveform(self, sample_rate: int, samples: "np.ndarray") -> None:
        import numpy as np

        # 采样率不匹配要处理掉，不能忽略：这个 stream 只累积，喂进来的样本会
        # 被当成后端采样率下的音频 —— 8k 的 8000 个样本被读成 16k 的 0.5 秒，
        # 音高和语速全错且不报任何错。（SenseVoice-TRT 把 16000 硬编码进了自己
        # 的 fbank，所以这条路一直是静默错的。）
        #
        # 但重采样要留到 finalize 一次做完，**不能逐块做**：每块独立插值会在
        # 每个块边界留下不连续点，实测 8k→16k 的 440Hz 正弦上是 0.17 幅度的
        # 周期性毛刺（信号峰值 1.0），而且非整数倍率下长度会累积漂移
        # （22050→16000、200 块累计 +4.6ms）。这里只记采样率。
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            # Not audio. Appending it reaches np.stack([]) in some backends,
            # and letting it pin the rate would reject the first real chunk.
            return

        expected = self._backend.sample_rate
        if sample_rate != expected and sample_rate not in self._warned_rates:
            self._warned_rates.add(sample_rate)
            logger.warning(
                "%s: will resample %d Hz -> %d Hz at finalize. Feeding the "
                "backend's own rate avoids this; linear interpolation is "
                "adequate for speech but is not a resampler.",
                self._backend.name, sample_rate, expected,
            )
        if self._rate is None:
            self._rate = sample_rate
        elif self._rate != sample_rate:
            raise ValueError(
                f"{self._backend.name}: sample rate changed mid-utterance "
                f"({self._rate} -> {sample_rate}); one utterance is one rate"
            )
        # copy(): np.asarray 对已经是 float32 的输入返回**同一个对象**，于是
        # 缓冲区里存的是调用方的数组。调用方复用或清零它之后，finalize 转写的
        # 就是被改过的内容。
        self._buf.append(samples.copy())

    def get_partial(self) -> tuple[str, bool]:
        # Offline models produce no incremental partials; the server-side VAD
        # (or explicit EOS) drives finalize().
        return "", False

    def finalize(self) -> tuple[str, Optional[str]]:
        import numpy as np

        if not self._buf:
            return "", None
        audio = np.concatenate(self._buf) if len(self._buf) > 1 else self._buf[0]
        # ONE resample over the whole utterance — see accept_waveform.
        if self._rate is not None and self._rate != self._backend.sample_rate:
            audio = _resample_linear(audio, self._rate, self._backend.sample_rate)
        self._buf = []
        self._rate = None
        result = self._backend.transcribe_array(audio, self._language)
        return result.text, result.language

    def close(self) -> None:
        self._buf = []
        self._rate = None


class ASRBackend(ABC):
    """Streaming/offline ASR backend.

    Public API mirrors app/core/asr_backend.py:94-147. The production
    factory ``create_asr_backend()`` reads ``current_profile()`` — voxedge
    drops that: callers construct the concrete backend with explicit params.
    """

    # Backends whose unload() truly releases GPU/NPU resources set True.
    supports_hot_reload: bool = False
    prefer_backend_endpoint_vad: bool = False
    """Whether streams from this backend should receive audio before frontend
    VAD speech_start and rely on backend endpointing for finalization."""

    # Offline backends set this True + implement transcribe_array() to get a
    # streaming session (via OfflineAccumulateStream) + the STREAMING capability
    # for free — no per-backend stream code, no internal VAD.
    supports_offline_streaming: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> set[ASRCapability]:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def preload(self) -> None:
        """Load models and warm up. Called once before use."""
        ...

    @abstractmethod
    def transcribe(
        self, audio_bytes: bytes, language: str = "auto"
    ) -> TranscriptionResult:
        """One-shot offline transcription."""
        ...

    def transcribe_array(
        self, samples: "np.ndarray", language: str = "auto"
    ) -> TranscriptionResult:
        """One-shot offline transcription on a float32 mono 16k sample array.

        Implemented by offline backends that opt into ``supports_offline_streaming``;
        the generic OfflineAccumulateStream calls this on finalize().
        """
        raise NotImplementedError(f"{self.name} does not implement transcribe_array")

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Create a streaming ASR session. Requires STREAMING capability.

        Offline backends with ``supports_offline_streaming`` get a generic
        accumulate-then-transcribe stream for free.
        """
        if self.supports_offline_streaming:
            return OfflineAccumulateStream(self, language)
        raise NotImplementedError(f"{self.name} does not support streaming")

    def has_capability(self, cap: ASRCapability) -> bool:
        if cap in self.capabilities:
            return True
        # Offline backends that opt into pseudo-streaming advertise STREAMING.
        if cap == ASRCapability.STREAMING and self.supports_offline_streaming:
            return True
        return False

    def unload(self) -> None:
        """Release GPU/NPU resources. Default no-op."""

    def concurrency_capability(self) -> "ConcurrencyCapability":
        """Describe runtime concurrency properties (N, mode).

        Instance method (no profile/env coupling). Returns a typed
        ``ConcurrencyCapability`` — the same type concrete backends override
        with and the capability resolver aggregates via ``.max_concurrent``.
        Default is conservative (serialized, single-slot) so backends that do
        not override stay N=1-safe and still resolve cleanly.
        """
        # Local import: a top-level one would trigger voxedge.engine.__init__
        # (which imports conversation -> backends.base) and deadlock the import.
        from voxedge.engine.concurrency_capability import ConcurrencyCapability
        return ConcurrencyCapability.default()


# ═══════════════════════════════════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════════════════════════════════


class TTSCapability(str, Enum):
    """Mirrors app/core/tts_backend.py:21-28."""

    BASIC_TTS = "basic_tts"
    VOICE_CLONE = "voice_clone"
    VOICE_CLONE_ICL = "voice_clone_icl"
    STREAMING = "streaming"
    MULTI_SPEAKER = "multi_speaker"
    MULTI_LANGUAGE = "multi_language"


class TTSBackend(ABC):
    """Text-to-speech backend.

    Public API mirrors app/core/tts_backend.py:31-142. The production
    ``model_id`` property reads ``OVS_TTS_MODEL_ID`` from env — voxedge takes
    ``model_id`` as an explicit constructor param on concrete backends
    instead (no env read here).
    """

    supports_hot_reload: bool = False

    # Honest device-side enrollment signal. True only when the backend can turn
    # a reference WAV into a speaker embedding *on this host* (e.g. a CPU-ONNX
    # speaker encoder is present). Distinct from ``VOICE_CLONE`` capability,
    # which only means the backend can *consume* a raw embedding. Backends
    # override this (usually as a property gated on the encoder file existing).
    supports_voice_enrollment: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> set[TTSCapability]:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def preload(self) -> None:
        """Load models and warm up. Called once before use."""
        ...

    # ── Unified speed / pitch interface ──────────────────────────────────
    #
    # Each backend declares which of (speed, pitch) it supports NATIVELY via
    # :meth:`rate_pitch_caps`. The public ``synthesize`` / ``generate_streaming``
    # below are WRAPPERS: they hand the natively-supported dimension to the
    # backend impl and apply a pure-numpy DSP fallback (see
    # ``voxedge.audio.rate``) for the rest. Concrete backends implement
    # ``_synthesize_impl`` / ``_generate_streaming_impl`` instead of overriding
    # the public methods.
    #
    # Backward compatibility: a backend (or test mock) that overrides
    # ``synthesize`` / ``generate_streaming`` directly bypasses the wrapper
    # entirely — its old behaviour is preserved (no DSP, no double-apply).

    def rate_pitch_caps(self) -> tuple[bool, bool]:
        """Return ``(native_speed, native_pitch)`` for this backend.

        Default ``(False, False)`` → both dimensions handled by the DSP
        fallback. Backends with a native ``speed`` (e.g. Matcha ``length_scale``)
        or native pitch override this.
        """
        return (False, False)

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize text to WAV bytes. Returns ``(wav_bytes, metadata)``.

        Wrapper: passes the natively-supported (speed, pitch) dims to
        ``_synthesize_impl`` and DSP-post-processes the non-native dims on the
        returned WAV. The identity request (no speed/pitch) is byte-identical
        to the impl's output (DSP module's identity fast-path).
        """
        native_speed, native_pitch = self.rate_pitch_caps()
        need_speed = speed not in (None, 1.0) and not native_speed
        need_pitch = pitch_shift not in (None, 0.0) and not native_pitch

        impl_speed = speed if native_speed else None
        impl_pitch = pitch_shift if native_pitch else None

        wav, meta = self._synthesize_impl(
            text,
            speaker_id=speaker_id,
            speed=impl_speed,
            pitch_shift=impl_pitch,
            language=language,
            **kwargs,
        )
        if need_speed or need_pitch:
            from voxedge.audio.rate import apply_wav_rate_pitch

            channels = None
            if isinstance(meta, dict) and meta.get("channels"):
                channels = int(meta["channels"])
            wav = apply_wav_rate_pitch(
                wav,
                speed=speed if need_speed else None,
                pitch_shift=pitch_shift if need_pitch else None,
                channels=channels,
            )
        return wav, meta

    def _synthesize_impl(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Backend synthesis. Receives speed/pitch ONLY for natively-supported
        dims (the wrapper pops the rest). Returns ``(wav_bytes, metadata)``.

        NOT abstract: a backend (or mock) may instead override the public
        ``synthesize`` directly, which bypasses the wrapper (and this method).
        """
        raise NotImplementedError(
            f"Backend '{self.name}' must implement _synthesize_impl or override synthesize"
        )

    def generate_streaming(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        cancel_token: Optional[Any] = None,
        **kwargs,
    ) -> Iterator[bytes]:
        """Generator yielding raw PCM chunks. Requires STREAMING capability.

        Wrapper: routes the non-native (speed, pitch) dims through a streaming
        :class:`~voxedge.audio.rate.TTSRateShifter` initialised from
        ``self.sample_rate``. The identity request is a pass-through (chunks
        yielded unchanged).
        """
        speed = kwargs.get("speed")
        pitch_shift = kwargs.get("pitch_shift", kwargs.get("pitch"))
        native_speed, native_pitch = self.rate_pitch_caps()
        need_speed = speed not in (None, 1.0) and not native_speed
        need_pitch = pitch_shift not in (None, 0.0) and not native_pitch

        impl_kwargs = dict(kwargs)
        # Pop the dims we will DSP so the impl does not also apply them.
        if need_speed:
            impl_kwargs.pop("speed", None)
        if need_pitch:
            impl_kwargs.pop("pitch_shift", None)
            impl_kwargs.pop("pitch", None)

        impl_iter = self._generate_streaming_impl(
            text,
            language=language,
            speaker=speaker,
            cancel_token=cancel_token,
            **impl_kwargs,
        )

        if not (need_speed or need_pitch):
            yield from impl_iter
            return

        from voxedge.audio.rate import TTSRateShifter

        shifter = TTSRateShifter(
            sample_rate=self.sample_rate,
            speed=speed if need_speed else 1.0,
            pitch_shift=pitch_shift if need_pitch else 0.0,
            channels=1,
        )
        for chunk in impl_iter:
            out = shifter.push(chunk)
            if out:
                yield out
        tail = shifter.flush()
        if tail:
            yield tail

    def _generate_streaming_impl(
        self,
        text: str,
        *,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        cancel_token: Optional[Any] = None,
        **kwargs,
    ) -> Iterator[bytes]:
        """Backend streaming impl. Receives speed/pitch (in kwargs) ONLY for
        natively-supported dims. Default: not supported."""
        raise NotImplementedError(f"Backend '{self.name}' does not support streaming")

    def clone_voice(
        self,
        text: str,
        speaker_embedding: bytes,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        """Synthesize with voice cloning. Requires VOICE_CLONE capability."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support voice cloning"
        )

    def extract_speaker_embedding(self, audio_wav_bytes: bytes) -> bytes:
        raise NotImplementedError(
            f"Backend '{self.name}' does not support speaker embedding extraction"
        )

    def has_capability(self, cap: TTSCapability) -> bool:
        return cap in self.capabilities

    def unload(self) -> None:
        """Release GPU/NPU resources. Default no-op."""

    def concurrency_capability(self) -> "ConcurrencyCapability":
        """See :meth:`ASRBackend.concurrency_capability`."""
        from voxedge.engine.concurrency_capability import ConcurrencyCapability
        return ConcurrencyCapability.default()


# ═══════════════════════════════════════════════════════════════════════
# VAD
# ═══════════════════════════════════════════════════════════════════════


class VADSession(ABC):
    """Per-connection VAD state. Mirrors app/core/vad.py:90-109.

    Feed PCM chunks via :meth:`process`; get speech-start / speech-end
    transition events back.
    """

    SPEECH_START: Literal["speech_start"] = "speech_start"
    SPEECH_END: Literal["speech_end"] = "speech_end"
    NONE = None

    @abstractmethod
    def process(self, samples: "np.ndarray") -> Optional[str]:
        """Feed one chunk of int16/float32 PCM (16 kHz mono assumed).

        Returns ``"speech_start"`` (onset), ``"speech_end"`` (endpoint), or
        ``None`` (no transition this chunk).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset state (e.g. after a forced finalize)."""
        ...


class VADBackend(ABC):
    """Factory for per-connection VAD sessions.

    voxedge folds the production module-level ``create_vad(...)`` factory
    (app/core/vad.py:244-257) into a backend object whose ``create_session``
    takes explicit ``sample_rate`` / ``silence_ms`` — no ``SILERO_VAD_ONNX_PATH``
    env read (the model path is a constructor arg on concrete backends).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def create_session(
        self, sample_rate: int = 16000, silence_ms: int = 400, **kwargs
    ) -> VADSession:
        ...


# ═══════════════════════════════════════════════════════════════════════
# Translator
# ═══════════════════════════════════════════════════════════════════════


class TranslatorCapability(str, Enum):
    """Capabilities a translation backend may advertise.

    Mirrors the ASR/TTS capability-enum style — a small, env-free vocabulary
    the capability resolver / callers can query via :meth:`has_capability`.
    """

    TEXT = "text"
    MULTI_LANGUAGE = "multi_language"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class TranslatorConfig:
    """Explicit, env-free construction parameters for a translator backend.

    Mirrors the production ``services/translator/server.py`` env reads
    (``TRANSLATOR_MODEL_PATH`` / ``TRANSLATOR_DEVICE`` / ``TRANSLATOR_DEVICE_INDEX``
    …) but as plain fields — voxedge never reads ``os.environ``; callers pass
    a fully-resolved config.
    """

    model_path: str
    src_lang: str = "zho_Hans"
    tgt_lang: str = "eng_Latn"
    device: str = "cuda"
    device_index: int = 0
    compute_type: str = "default"
    beam_size: int = 1
    max_batch_size: int = 1


@dataclass
class TranslationResult:
    """One translation output. Mirrors :class:`TranscriptionResult` style."""

    text: str
    src_lang: str
    tgt_lang: str
    meta: Optional[dict] = None


class TranslatorBackend(ABC):
    """Text-to-text translation backend.

    Env-free, pure-config-driven — mirrors :class:`ASRBackend` /
    :class:`TTSBackend`. Concrete backends are constructed with an explicit
    :class:`TranslatorConfig`; nothing here reads ``os.environ`` or a profile.
    """

    # Backends whose unload() truly releases GPU/NPU resources set True.
    supports_hot_reload: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> set[TranslatorCapability]:
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def preload(self) -> None:
        """Load models and warm up. Called once before use."""
        ...

    @abstractmethod
    def translate(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> TranslationResult:
        """Translate ``text``. ``src_lang`` / ``tgt_lang`` default to config."""
        ...

    def translate_batch(
        self,
        texts: list[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> list[TranslationResult]:
        """Translate several texts. Requires BATCH capability."""
        raise NotImplementedError(
            f"Backend '{self.name}' does not support batch translation"
        )

    def has_capability(self, cap: TranslatorCapability) -> bool:
        return cap in self.capabilities

    def unload(self) -> None:
        """Release GPU/NPU resources. Default no-op."""

    def concurrency_capability(self) -> "ConcurrencyCapability":
        """See :meth:`ASRBackend.concurrency_capability`."""
        from voxedge.engine.concurrency_capability import ConcurrencyCapability
        return ConcurrencyCapability.default()


# ═══════════════════════════════════════════════════════════════════════
# LLM  (contract reproduced from agent/openvoicestream_agent/llm/base.py)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LLMEvent:
    """One unit of streaming output from an LLM backend.

    Equivalent to agent/openvoicestream_agent/llm/base.py:9-30. Reproduced
    here (not imported) so voxedge core stays free of the agent package's
    openai/httpx deps. When layer ② is packaged this can re-export the
    canonical definition.
    """

    kind: Literal["text", "tool_call_delta", "finish"]
    text: Optional[str] = None
    tool_call_index: Optional[int] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    finish_reason: Optional[str] = None


class LLMBackend(ABC):
    """Streaming LLM backend.

    Equivalent ABC to agent/openvoicestream_agent/llm/base.py:33-118:
      * ``stream_events`` (preferred) — yields :class:`LLMEvent`.
      * ``stream`` (back-compat) — yields plain text deltas.
      * ``warmup`` / ``aclose`` lifecycle hooks (default no-op, idempotent).
    """

    async def stream_events(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[LLMEvent]:
        """Yield :class:`LLMEvent`s. Default delegates to :meth:`stream`."""
        async for tok in self.stream(messages, **kw):
            if tok:
                yield LLMEvent(kind="text", text=tok)
        yield LLMEvent(kind="finish", finish_reason="stop")

    @abstractmethod
    async def stream(
        self, messages: list[dict[str, Any]], **kw: Any
    ) -> AsyncIterator[str]:
        """Yield already-decoded text deltas."""
        if False:  # pragma: no cover
            yield ""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release held network/transport resources. Default no-op."""
        return None

    async def warmup(
        self,
        *,
        system_prompt: str = "",
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: bool = False,
        timeout_s: Optional[float] = 60.0,
    ) -> dict[str, Any]:
        """Optional fire-and-forget pre-flight warmup. Default returns {}."""
        return {}


__all__ = [
    "ASRCapability",
    "TranscriptionResult",
    "ASRStream",
    "ASRBackend",
    "TTSCapability",
    "TTSBackend",
    "VADSession",
    "VADBackend",
    "TranslatorCapability",
    "TranslatorConfig",
    "TranslationResult",
    "TranslatorBackend",
    "LLMEvent",
    "LLMBackend",
]
