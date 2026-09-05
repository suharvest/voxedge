"""RK TTS adapter — voxedge adapter.

adapted from app/backends/rk/tts.py + app/core/rk_*.py (2026-05-30), dedup
after registry switch.

Wraps ``rkvoice_stream.create_tts()`` output. rkvoice-stream's TTSBackend ABC
is smaller than ours (no ``capabilities``, no ``language`` arg, ``speaker_id``
is int with default 0); the adapter forwards everything the voxedge contract
requires and exposes a conservative default capability set.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (not ``app.core.*``);
    ``ConcurrencyCapability`` from ``voxedge.engine.concurrency_capability``.
  * The ``model_id`` (was ``OVS_TTS_MODEL_ID`` env, read by the production
    base ``TTSBackend.model_id`` property) is injected via ``RKTTSConfig``.
    voxedge has no module-scope or hardcoded env reads.
  * ``detect_zh_en`` / ``resolve_speaker_kwargs`` reproduced in ``._util``
    (no ``app.*`` import).
  * ``import rkvoice_stream`` stays inside ``__init__`` (lazy) so this module
    imports cleanly without the optional ``voxedge[rk]`` extra / rknn runtime.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from voxedge.backends.base import TTSBackend, TTSCapability
from voxedge.engine.concurrency_capability import ConcurrencyCapability

from ._util import detect_zh_en, normalize_auto_language, resolve_speaker_kwargs

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                  → RKTTSConfig field
#   OVS_TTS_MODEL_ID                → model_id   (default "rk"; production base
#                                                 fell back to backend name)

# rkvoice-stream's TTSBackend doesn't expose a capability set. The shipped
# backends (matcha_rknn, piper_rknn, qwen3_rknn) all do basic + streaming TTS,
# so declare that as the floor. The wire layer feature-detects optional things
# (voice clone, etc.) via has_capability().
_DEFAULT_RK_TTS_CAPS = {
    TTSCapability.BASIC_TTS,
    TTSCapability.STREAMING,
    TTSCapability.MULTI_LANGUAGE,
}


@dataclass
class RKTTSConfig:
    """Explicit construction-time config for :class:`RKTTSBackend`.

    ``model_id`` was the production ``OVS_TTS_MODEL_ID`` env (used to key the
    speaker registry). The RK adapter is single-speaker so the value is only
    passed to ``resolve_speaker_kwargs`` and otherwise unused; default ``"rk"``
    matches the production fallback to the backend name.
    """

    model_id: str = "rk"
    # Optional stable artifact name for the runtime-artifact manifest
    # (voxedge.artifacts). None preserves the existing host-mounted behaviour.
    artifact_ref: Optional[str] = None


class RKTTSBackend(TTSBackend):
    """Adapter around rkvoice_stream.create_tts(). Backend selection is
    delegated to rkvoice-stream via the ``TTS_BACKEND`` env var (set in the
    rk3576/rk3588 profile / process env); that env is read by rkvoice-stream,
    not by this adapter."""

    @classmethod
    def concurrency_capability(cls, profile=None):
        """Declare concurrency for RK NPU TTS.

        rkvoice-stream owns the NPU lifecycle, serializes through one NPU
        device, and cannot be safely multiplexed across slots. Single-session
        only.
        """
        return ConcurrencyCapability(
            supports_parallel=False,
            max_concurrent=1,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="external_managed",
        )

    def __init__(self, config: Optional[RKTTSConfig] = None):
        # Lazy init (matches the Jetson backends' lifecycle): __init__ only
        # stores config — the heavy ``rkvoice_stream.create_tts()`` NPU init is
        # deferred to ``preload()``. This keeps construction cheap so the
        # capability resolver / health wiring can build the object without
        # triggering NPU init or requiring the aarch64-only ``voxedge[rk]``
        # extra (a Mac / x86_64 dev box can construct but not preload). The
        # BackendManager always calls ``preload()`` after the factory, so the
        # runtime methods below still see a live ``_inner``.
        self._config = config or RKTTSConfig()
        # The lock is held for the complete synchronous call or streaming
        # iterator lifetime.  unload therefore cannot drop a native owner
        # while a generator still has borrowed state.
        # Generators may be advanced/closed by different executor threads;
        # Lock is not thread-bound like RLock and can be released by close().
        self._lifecycle_lock = threading.Lock()
        self._lifecycle_failed = False
        self._runtime_info_cache: dict = {}
        self._inner = None
        # Sensible cached defaults until ``preload()`` populates the real
        # values from the inner backend (also the post-unload fallback so
        # status queries don't crash on ``self._inner is None``).
        self._cached_name = "rk:unknown"
        self._cached_sample_rate = 0

    def _ensure_inner(self) -> None:
        """Create the rkvoice-stream inner backend (NPU init) on first use.

        Deferred out of ``__init__`` so construction stays cheap. Idempotent:
        a second call is a no-op once ``_inner`` exists. The friendly
        dependency check (naming the ``rk`` extra) runs here — the aarch64-only
        wheel is never present on a Mac / x86_64 dev box, so we only require it
        at the moment NPU init is actually needed.
        """
        if self._inner is not None:
            return
        from voxedge.backends._deps import check_rk_deps

        check_rk_deps()
        import rkvoice_stream

        create_tts = getattr(rkvoice_stream, "create_tts", None)
        if not callable(create_tts):
            raise ImportError(
                "rkvoice-stream>=0.2.0 is required: missing create_tts factory; "
                "install 'voxedge[rk]'"
            )

        self._inner = create_tts()
        try:
            self._cached_name = f"rk:{self._inner.name}"
        except Exception:
            self._cached_name = "rk:unknown"
        try:
            self._cached_sample_rate = int(self._inner.get_sample_rate())
        except Exception:
            self._cached_sample_rate = 0

    @property
    def name(self) -> str:
        inner = self._inner
        if inner is None:
            return self._cached_name
        return f"rk:{inner.name}"

    @property
    def model_id(self) -> str:
        """Model-scope key for speaker tables — injected via config (was
        ``OVS_TTS_MODEL_ID``)."""
        return self._config.model_id

    @property
    def capabilities(self) -> set[TTSCapability]:
        caps = set(_DEFAULT_RK_TTS_CAPS)
        # The default is intentionally conservative until the lazy inner is
        # loaded.  Once loaded, an explicit ``False`` is authoritative: the
        # product layer must not route this backend through streaming APIs.
        if (self._inner is not None and getattr(self._inner, "name", None) == "kokoro_convonly"
                and getattr(self._inner, "supports_streaming", True) is False):
            caps.discard(TTSCapability.STREAMING)
        return caps

    @property
    def sample_rate(self) -> int:
        inner = self._inner
        if inner is None:
            return self._cached_sample_rate
        return inner.get_sample_rate()

    def is_ready(self) -> bool:
        inner = self._inner
        if inner is None or self._lifecycle_failed:
            return False
        return inner.is_ready()

    def preload(self) -> None:
        # Lazy first-load: build the inner backend (NPU init) here rather than
        # in __init__. After ``unload()`` this re-creates it, which matches the
        # BackendManager reload contract (factory → preloader).
        with self._lifecycle_lock:
            if self._lifecycle_failed:
                raise RuntimeError(
                    "RKTTSBackend retained an inner owner after cleanup failure; "
                    "unload it successfully before retrying preload"
                )
            self._ensure_inner()
            try:
                self._inner.preload()
                self._refresh_runtime_info_locked()
            except Exception as original:
                cleanup_error = None
                try:
                    self._cleanup_inner(self._inner)
                except Exception as exc:
                    cleanup_error = exc
                if cleanup_error is not None:
                    self._lifecycle_failed = True
                    raise RuntimeError(
                        "RKTTSBackend preload failed and cleanup failed: "
                        f"{cleanup_error}"
                    ) from original
                self._inner = None
                raise

    @staticmethod
    def _cleanup_inner(inner) -> None:
        """Invoke exactly one advertised teardown hook, if present."""
        for name in ("close", "unload", "cleanup"):
            hook = getattr(inner, name, None)
            if callable(hook):
                hook()
                return

    def _refresh_runtime_info_locked(self) -> None:
        inner = self._inner
        info = getattr(inner, "runtime_info", None) if inner is not None else None
        if callable(info):
            info = info()
        if isinstance(info, dict):
            self._runtime_info_cache = dict(info)

    def unload(self) -> None:
        """Drop the rkvoice-stream inner backend handle. Idempotent.

        ``supports_hot_reload`` stays False — the NPU is held by the
        rkvoice-stream backend and a deeper teardown contract belongs to that
        repo. Provide a best-effort release here so future support can plug in
        without touching the manager.
        """
        with self._lifecycle_lock:
            inner = self._inner
            if inner is None:
                return
            # Keep ownership until teardown has succeeded.  On failure the
            # caller can retry and the manager can report the live resource.
            try:
                self._cleanup_inner(inner)
            except Exception:
                self._lifecycle_failed = True
                raise
            self._inner = None
            self._lifecycle_failed = False
            self._runtime_info_cache = {}

    def rate_pitch_caps(self) -> tuple[bool, bool]:
        # rkvoice's native speed/pitch is bypassed (wrapper pops them) → DSP
        # fallback handles both, so there is no double-apply with rkvoice.
        return (False, False)

    def _resolve_language(self, text: str, language: Optional[str]) -> str:
        """Resolve auto language only for the loaded Kokoro ConvOnly backend."""
        auto = normalize_auto_language(language) is None
        name = getattr(self._inner, "name", None)
        if name is None and self._cached_name == "rk:kokoro_convonly":
            name = "kokoro_convonly"
        if auto and name == "kokoro_convonly" and re.search(r"[ぁ-ゖァ-ヺㇰ-ㇿｦ-ﾟ]", text):
            return "ja"
        return detect_zh_en(text, language)

    def _synthesize_impl(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        with self._lifecycle_lock:
            if self._inner is None or self._lifecycle_failed:
                raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
            voice = resolve_speaker_kwargs(
                self.model_id, allow_embedding=False, speaker_id=speaker_id, **kwargs
            )
            sid = voice.get("speaker_id", 0)
            language = self._resolve_language(text, language)
            kwargs.setdefault("language", language)
            return self._inner.synthesize(
                text=text, speaker_id=sid, speed=speed, pitch_shift=pitch_shift,
                **kwargs,
            )

    def _generate_streaming_impl(self, text: str, **kwargs):
        """Bridge our base-class generate_streaming() to rkvoice-stream's
        synthesize_stream().

        rkvoice-stream yields ``(audio, metadata)`` tuples where ``audio`` is
        either float32 [-1,1], int16 PCM, or raw bytes. The wire layer
        (`/tts/stream`) expects int16 PCM bytes per chunk, so coerce here —
        starlette's StreamingResponse calls ``.encode()`` on non-bytes items
        and explodes on tuples (`'tuple' object has no attribute 'encode'`).
        """
        with self._lifecycle_lock:
            if self._inner is None or self._lifecycle_failed:
                raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
            voice = resolve_speaker_kwargs(self.model_id, allow_embedding=False, **kwargs)
            speaker_id = voice.get("speaker_id", 0)
            kwargs.pop("speaker_id", None)
            speed = kwargs.pop("speed", None)
            pitch_shift = kwargs.pop("pitch_shift", None)
            language = self._resolve_language(text, kwargs.pop("language", None))
            kwargs.setdefault("language", language)
            source = self._inner.synthesize_stream(
                text=text, speaker_id=speaker_id, speed=speed,
                pitch_shift=pitch_shift, **kwargs,
            )
            try:
                for item in source:
                    audio = item[0] if isinstance(item, tuple) else item
                    if audio is None:
                        continue
                    if isinstance(audio, (bytes, bytearray)):
                        if audio:
                            yield bytes(audio)
                    elif isinstance(audio, np.ndarray) and audio.size:
                        if audio.dtype == np.int16:
                            yield audio.tobytes()
                        else:
                            a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
                            yield (a * 32767.0).astype(np.int16).tobytes()
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    close()

    def synthesize_stream(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Iterator[tuple[np.ndarray, dict]]:
        with self._lifecycle_lock:
            if self._inner is None or self._lifecycle_failed:
                raise RuntimeError("RKTTSBackend not loaded (was unloaded)")
            language = self._resolve_language(text, language)
            kwargs.setdefault("language", language)
            source = self._inner.synthesize_stream(
                text=text, speaker_id=speaker_id if speaker_id is not None else 0,
                speed=speed, pitch_shift=pitch_shift, **kwargs,
            )
            try:
                for item in source:
                    yield item
            finally:
                close = getattr(source, "close", None)
                if callable(close):
                    close()

    def runtime_info(self):
        if not self._lifecycle_lock.acquire(blocking=False):
            info = dict(self._runtime_info_cache)
            reported_ready = bool(info.get("ready", False))
            info["lifecycle_busy"] = True
            info["lifecycle_ready"] = bool(self._inner is not None and not self._lifecycle_failed)
            info["ready"] = reported_ready and info["lifecycle_ready"]
            return info
        try:
            self._refresh_runtime_info_locked()
            info = dict(self._runtime_info_cache)
            if "ready" in info:
                reported_ready = bool(info["ready"])
            else:
                inner = self._inner
                probe = getattr(inner, "is_ready", None) if inner is not None else None
                reported_ready = bool(probe()) if callable(probe) else False
            info["lifecycle_busy"] = False
            info["lifecycle_ready"] = bool(self._inner is not None and not self._lifecycle_failed)
            info["ready"] = reported_ready and info["lifecycle_ready"]
            return info
        finally:
            self._lifecycle_lock.release()
