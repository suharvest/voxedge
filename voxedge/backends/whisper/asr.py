"""Whisper ASR backend — NPU/GPU encoder + CPU KV-cache decoder.

See the package docstring for why the decoder is not on the accelerator and why
this backend is offline-only.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from voxedge.audio.segment import split_at_silence_energy, split_at_silence_vad
from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    TranscriptionResult,
    resolve_reported_language,
)
# Lives under capabilities/ because speaker enrollment needed it first, but it
# is plain stdlib+numpy audio decoding: PCM16 WAV or raw PCM in, mono float32 @
# 16 kHz out, stereo downmixed and resampled. Reused rather than re-parsed here.
from voxedge.capabilities.speaker_embedding import decode_audio_to_16k_mono
from voxedge.text.degenerate import collapse_repetition, collapse_segment_repeats
from voxedge.text.join import join_segments

from .decoder import OnnxKVDecoder, detokenize, read_vocab
from .encoders import build_encoder
from .frontend import SAMPLE_RATE, load_mel_filters, log_mel

logger = logging.getLogger(__name__)

# Whisper's shipped encoders are English-or-Chinese here; the language token is
# forced rather than detected, because the multilingual detection pass costs a
# whole extra decoder prefill for a choice the deployment already knows.
_SUPPORTED = ("en", "zh")


@dataclass
class WhisperASRConfig:
    """Explicit construction-time config. Nothing here reads ``os.environ``.

    ``window_s`` is not a free knob: it must equal the window the encoder graph
    was compiled at (the .hef, the .rknn, or the shape the .plan was built
    with). Feeding a different length silently reinterprets the buffer on
    rknn-lite and raises on the others.
    """

    encoder_kind: str                    # "hailo" | "rknn" | "tensorrt"
    encoder_path: str
    decoder_dir: str                     # optimum ONNX export (2 graphs)
    vocab_dir: str                       # vocab_en.txt / vocab_zh.txt / mel filters
    window_s: float = 10.0
    language: str = "en"
    #: Hailo's boundary guard: crop this much off the window before padding
    #: back out. Non-zero only for HEFs built with it.
    padding_cutoff_s: float = 0.0
    #: Silence quieter than this counts as a cut point when audio is longer
    #: than one window; a run this long is needed before a cut is taken there.
    split_rms: float = 0.003
    split_min_silence_ms: int = 80
    #: Bind all three RK3588 NPU cores. Measured no faster than the default on
    #: this graph; left available for boards where it does help.
    all_cores: bool = False
    #: Hard cap on generated tokens per chunk. ``None`` uses the
    #: duration-proportional budget, which assumes the decoder emits EOS.
    #: The Hailo pairing does not: it transcribes correctly and then repeats
    #: the sentence until the budget runs out, so the vendor pipeline caps the
    #: sequence instead. Set it for that path; leave it None elsewhere.
    max_new_tokens: Optional[int] = None
    #: 0 lets onnxruntime pick. The decoder is the wall-clock bottleneck on
    #: every board here, so this is the knob that actually moves RTF.
    decoder_threads: int = 0
    warmup_runs: int = 1
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        lang = (self.language or "en").strip().lower()
        if lang not in _SUPPORTED:
            raise ValueError(
                f"whisper: language {self.language!r} not supported by this backend "
                f"(built for {'/'.join(_SUPPORTED)}); use Paraformer/SenseVoice/"
                f"Qwen3-ASR for other languages"
            )
        self.language = lang
        # nan and inf reach here from anything that is not the product env
        # parser — a caller constructing the dataclass directly, a YAML float,
        # a computed value. Every comparison against nan is False, so the range
        # checks below cannot catch it: the invariant has to be stated here.
        for name in ("window_s", "padding_cutoff_s", "overlap_check"):
            value = getattr(self, name, 0.0)
            if not math.isfinite(value):
                raise ValueError(f"whisper: {name} must be finite, got {value!r}")
        if self.window_s <= 0:
            raise ValueError(f"whisper: window_s must be > 0, got {self.window_s}")
        if self.padding_cutoff_s < 0:
            # A negative cutoff makes the usable window LONGER than the graph,
            # and the front end then truncates the excess silently — the one
            # failure mode this class exists to prevent.
            raise ValueError(
                f"whisper: padding_cutoff_s must be >= 0, got {self.padding_cutoff_s}"
            )
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError(
                f"whisper: max_new_tokens must be >= 1, got {self.max_new_tokens}"
            )
        # Compare in SAMPLES, not seconds: a cutoff of 4.99999 against a 5 s
        # window is "less than" by the float check and still leaves zero
        # samples, which divides by zero when segments are capped.
        if int((self.window_s - self.padding_cutoff_s) * SAMPLE_RATE) < SAMPLE_RATE // 10:
            raise ValueError(
                f"whisper: window_s={self.window_s} minus "
                f"padding_cutoff_s={self.padding_cutoff_s} leaves under 100 ms "
                f"of usable audio"
            )
        if self.padding_cutoff_s >= self.window_s:
            raise ValueError(
                f"whisper: padding_cutoff_s {self.padding_cutoff_s} leaves no audio "
                f"inside a {self.window_s}s window"
            )


def _enforce_window(chunks: list[np.ndarray], usable_s: float) -> list[np.ndarray]:
    """Hard-cap every segment at the encoder window.

    The silence splitter can hand back a segment LONGER than the max it was
    given: its final pass folds a short tail into its neighbour, which is the
    right trade for a decoder that merely degrades on over-long input. Whisper's
    window is a compiled-in shape, so an over-long segment is not degraded, it
    is truncated — the tail is dropped with no error anywhere. Splitting it
    evenly keeps the pieces balanced instead of leaving a runt at the end.
    """
    limit = int(usable_s * SAMPLE_RATE)
    out: list[np.ndarray] = []
    for chunk in chunks:
        if len(chunk) <= limit:
            out.append(chunk)
            continue
        n = -(-len(chunk) // limit)
        step = -(-len(chunk) // n)
        out.extend(chunk[i : i + step] for i in range(0, len(chunk), step))
    return out


class WhisperASR(ASRBackend):
    """Offline Whisper. Streaming comes from ``OfflineAccumulateStream``."""

    supports_hot_reload = True
    supports_offline_streaming = True

    def __init__(self, config: WhisperASRConfig) -> None:
        self._cfg = config
        self._encoder = None
        self._decoder: Optional[OnnxKVDecoder] = None
        self._filters: Optional[np.ndarray] = None
        self._vocab: Optional[dict] = None
        self._warned: set = set()

    # ── identity ────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return f"whisper-{self._cfg.encoder_kind}"

    @property
    def capabilities(self) -> set[ASRCapability]:
        # No LANGUAGE_ID: the language token is forced, not detected. Claiming
        # it would make the reported language an echo of the config.
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING}

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_ready(self) -> bool:
        return self._encoder is not None and self._decoder is not None

    # ── lifecycle ───────────────────────────────────────────────────────
    def preload(self) -> None:
        """Load and warm up. Either everything works or the backend stays unready.

        Nothing is published to ``self`` until the warmup inference has
        succeeded. Assigning as we go looked harmless and was not: a warmup
        failure left ``is_ready()`` True, the server logged the exception and
        carried on, and the first real utterance then reused a runtime that had
        already failed to run once.
        """
        if self.is_ready():
            return
        cfg = self._cfg
        vocab_dir = Path(cfg.vocab_dir)
        filters = load_mel_filters(vocab_dir / "mel_80_filters.txt")
        vocab = read_vocab(vocab_dir / f"vocab_{cfg.language}.txt")
        decoder = OnnxKVDecoder(cfg.decoder_dir, cfg.decoder_threads)
        encoder = build_encoder(
            cfg.encoder_kind,
            cfg.encoder_path,
            cfg.window_s,
            padding_cutoff_s=cfg.padding_cutoff_s,
            all_cores=cfg.all_cores,
        )
        try:
            for _ in range(max(0, cfg.warmup_runs)):
                # First inference on every one of these runtimes pays a one-off
                # setup cost (JIT, memory pool, engine context). Paying it here
                # keeps it out of the first user utterance's TTFT.
                encoder.run(
                    log_mel(
                        np.zeros(int(cfg.window_s * SAMPLE_RATE), dtype=np.float32),
                        filters,
                        cfg.window_s,
                        cfg.padding_cutoff_s,
                    )
                )
        except Exception:
            # Release the accelerator handle; on Hailo it is the whole device,
            # and holding it would block the next attempt as well.
            try:
                encoder.close()
            except Exception:
                logger.exception("whisper: encoder close after failed warmup raised")
            raise

        self._filters, self._vocab = filters, vocab
        self._decoder, self._encoder = decoder, encoder
        logger.info(
            "whisper: %s encoder @%.1fs window, CPU KV decoder, lang=%s",
            cfg.encoder_kind, cfg.window_s, cfg.language,
        )

    def unload(self) -> None:
        if self._encoder is not None:
            self._encoder.close()
        self._encoder = None
        self._decoder = None
        self._vocab = None
        self._filters = None

    def concurrency_capability(self):
        from voxedge.engine.concurrency_capability import ConcurrencyCapability

        # One runtime handle, one in-flight call. Nothing is carried between
        # calls, so this is a mutex rather than a session.
        #
        # ``requires_exclusive_device`` differs by path: HailoRT hands
        # /dev/hailo0 to a single process and a second VDevice anywhere raises
        # HAILO_OUT_OF_PHYSICAL_DEVICES, whereas a TRT engine happily shares the
        # Jetson GPU with the TTS stack.
        return ConcurrencyCapability(
            supports_parallel=False,
            max_concurrent=1,
            is_stateful=False,
            requires_exclusive_device=self._cfg.encoder_kind in ("hailo", "rknn"),
            scaling_mode="single_runtime_multiplex",
        )

    def _split(self, audio: np.ndarray, usable_s: float) -> list[np.ndarray]:
        """Cut long audio, preferring VAD over frame energy.

        Both matter here. Frame RMS finds a gap only where the waveform is
        quiet, and on the shortest windows there often is not one: Hailo's base
        HEF leaves 4 s of usable audio, and continuous speech rarely goes quiet
        inside 4 s, so the energy splitter falls back to a hard cut mid-phrase.
        VAD decides on speech rather than loudness and finds boundaries the
        energy pass cannot. Measured cost of not doing this: English long-form
        on Hailo tiny came in at 40.3% against the vendor harness's 21.6%,
        entirely on segmentation.

        webrtcvad is optional, hence the fallback — the same order the
        TRT-Edge-LLM backend uses.
        """
        cfg = self._cfg
        try:
            return split_at_silence_vad(audio, SAMPLE_RATE, max_seg_s=usable_s)
        except ImportError:
            logger.debug("whisper: webrtcvad absent, splitting on frame energy")
        except Exception as exc:
            logger.warning("whisper: VAD splitter failed (%s); using frame energy", exc)
        return split_at_silence_energy(
            audio,
            SAMPLE_RATE,
            split_rms=cfg.split_rms,
            min_silence_ms=cfg.split_min_silence_ms,
            max_seg_s=usable_s,
        )

    # ── transcription ───────────────────────────────────────────────────
    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        return self.transcribe_array(decode_audio_to_16k_mono(audio_bytes), language)

    def transcribe_array(
        self, samples: np.ndarray, language: str = "auto"
    ) -> TranscriptionResult:
        if not self.is_ready():
            self.preload()
        cfg = self._cfg
        lang = resolve_reported_language(
            language, honoured=cfg.language, backend=self.name, warned=self._warned
        )
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return TranscriptionResult(text="", language=lang, meta={"chunks": 0})

        # Whisper's window is fixed at compile time, so audio longer than one
        # window has to be cut. Cutting at silence rather than at a fixed hop is
        # what the RK and TRT-Edge-LLM backends already do for their own
        # fixed-context decoders, and it removes the need to stitch overlapping
        # transcripts: segments no longer share audio, so they no longer share
        # words.
        usable_s = cfg.window_s - cfg.padding_cutoff_s
        chunks = _enforce_window(self._split(audio, usable_s), usable_s)

        t0 = time.perf_counter()
        enc_ms = dec_ms = 0.0
        ttft_ms: Optional[float] = None
        parts: list[str] = []
        for chunk in chunks:
            mel = log_mel(chunk, self._filters, cfg.window_s, cfg.padding_cutoff_s)
            te = time.perf_counter()
            enc_out = self._encoder.run(mel)
            enc_ms += (time.perf_counter() - te) * 1000
            td = time.perf_counter()
            raw, token_times = self._decoder.decode(
                enc_out,
                self._vocab,
                cfg.language,
                audio_s=len(chunk) / SAMPLE_RATE,
                max_new=cfg.max_new_tokens,
            )
            dec_ms += (time.perf_counter() - td) * 1000
            if ttft_ms is None:
                # TTFT here is encoder + decoder prefill, i.e. how long until
                # the first token exists — not how long until text is emitted,
                # which for an offline backend is the whole utterance.
                ttft_ms = enc_ms + (token_times[0] if token_times else 0.0)
            # Per-chunk, before joining: a runaway loop inside one chunk is
            # invisible once the segments are concatenated.
            text, collapsed = collapse_repetition(detokenize(raw, cfg.language))
            if collapsed:
                logger.debug("whisper: collapsed a degenerate chunk transcript")
            parts.append(text)

        # Cross-chunk: a whole segment repeating the previous one is the other
        # face of the same degeneration.
        parts, dropped = collapse_segment_repeats(parts)
        merged = join_segments(parts, cfg.language)

        total_ms = (time.perf_counter() - t0) * 1000
        audio_s = audio.size / SAMPLE_RATE
        return TranscriptionResult(
            text=merged,
            language=lang,
            meta={
                "chunks": len(chunks),
                "encoder_ms": round(enc_ms, 2),
                "decoder_ms": round(dec_ms, 2),
                "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
                "rtf": round(total_ms / 1000.0 / audio_s, 4) if audio_s else None,
                "segments_dropped": dropped,
            },
        )
