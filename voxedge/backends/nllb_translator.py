"""NLLB-200 translation backend via CTranslate2 + SentencePiece.

Env-free port of ``services/translator/server.py`` onto the voxedge
:class:`TranslatorBackend` ABC. Heavy runtime imports (``ctranslate2`` /
``sentencepiece``) are deferred to :meth:`preload` so the pure-Python core stays
installable on a CUDA-less / x86_64 dev box without the ``translator`` extra.

The translate path bakes in three tokenizer/runtime quirks that only surfaced
on real-device e2e (all handled here):

  1. tokenize with ``EncodeAsPieces`` (piece STRINGS), NOT ``EncodeAsIds`` —
     CT2 ``translate_batch`` expects token pieces, not int IDs.
  2. the FLORES src-lang code + ``</s>`` go AFTER the source pieces
     (``pieces + ["</s>", src_lang]``), not prefixed.
  3. ``device_index`` is forced to ``int`` — passing a str crashes CT2.
"""
from __future__ import annotations

from typing import Optional

from voxedge.backends._deps import check_translator_deps
from voxedge.backends.base import (
    TranslationResult,
    TranslatorBackend,
    TranslatorCapability,
    TranslatorConfig,
)


class NLLBTranslatorBackend(TranslatorBackend):
    """CTranslate2-backed NLLB-200 translator.

    Construct with an explicit :class:`TranslatorConfig`; the model and
    tokenizer are loaded lazily in :meth:`preload`.
    """

    supports_hot_reload = True

    def __init__(self, config: Optional[TranslatorConfig] = None) -> None:
        # heavy load deferred to preload() — mirror the other lazy backends.
        self._config = config or TranslatorConfig(model_path="")
        self._translator = None
        self._tokenizer = None

    # ── identity / capabilities ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "nllb_translator"

    @property
    def capabilities(self) -> set[TranslatorCapability]:
        return {
            TranslatorCapability.TEXT,
            TranslatorCapability.MULTI_LANGUAGE,
            TranslatorCapability.BATCH,
        }

    def is_ready(self) -> bool:
        return self._translator is not None and self._tokenizer is not None

    # ── lifecycle ───────────────────────────────────────────────────────

    def preload(self) -> None:
        """Load CT2 model + SentencePiece tokenizer from ``config.model_path``."""
        # Fail fast with a friendly install hint if the extra is missing.
        check_translator_deps()
        import os

        import ctranslate2
        import sentencepiece

        cfg = self._config
        if not cfg.model_path:
            raise ValueError(
                "NLLBTranslatorBackend.preload(): config.model_path is empty"
            )

        # CUDA guard: CT2 wheels on some platforms (e.g. the pypi arm64 wheel)
        # are CPU-only. Requesting device="cuda" against such a build silently
        # falls over deep inside CT2 — surface a clear, actionable error here
        # (memory nllb_translator_slim_cuda_jetson: needs a bare-metal CUDA
        # build of CTranslate2 on Jetson).
        if cfg.device == "cuda":
            try:
                cuda_count = ctranslate2.get_cuda_device_count()
            except Exception:
                cuda_count = 0
            if not cuda_count:
                raise RuntimeError(
                    "NLLBTranslatorBackend: config.device='cuda' but this "
                    "CTranslate2 build reports no CUDA devices (CPU-only wheel?). "
                    "Install a bare-metal CUDA build of CTranslate2 (see memory "
                    "nllb_translator_slim_cuda_jetson) or set device='cpu'."
                )

        tokenizer = sentencepiece.SentencePieceProcessor()
        tokenizer.Load(os.path.join(cfg.model_path, "sentencepiece.bpe.model"))

        translator = ctranslate2.Translator(
            cfg.model_path,
            device=cfg.device,
            device_index=int(cfg.device_index),  # bug #3: force int
            compute_type=cfg.compute_type,
        )

        self._tokenizer = tokenizer
        self._translator = translator

    def unload(self) -> None:
        """Drop references so GPU/CPU resources are released."""
        self._translator = None
        self._tokenizer = None

    # ── translation ─────────────────────────────────────────────────────

    def _encode(self, text: str, src_lang: str) -> list[str]:
        # bug #1: EncodeAsPieces (piece strings), NOT EncodeAsIds.
        # bug #2: src_lang code goes AFTER the source pieces, not prefixed.
        return self._tokenizer.EncodeAsPieces(text) + ["</s>", src_lang]

    def translate(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> TranslationResult:
        if not self.is_ready():
            raise RuntimeError(
                "NLLBTranslatorBackend.translate() called before preload()"
            )
        src = src_lang or self._config.src_lang
        tgt = tgt_lang or self._config.tgt_lang

        tokens = self._encode(text, src)
        results = self._translator.translate_batch(
            [tokens],
            target_prefix=[[tgt]],
            beam_size=self._config.beam_size,
        )
        # Skip the tgt_lang token CT2 echoes back at position 0.
        translated_tokens = results[0].hypotheses[0][1:]
        out = self._tokenizer.DecodePieces(translated_tokens)
        return TranslationResult(text=out, src_lang=src, tgt_lang=tgt)

    def translate_batch(
        self,
        texts: list[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> list[TranslationResult]:
        if not self.is_ready():
            raise RuntimeError(
                "NLLBTranslatorBackend.translate_batch() called before preload()"
            )
        src = src_lang or self._config.src_lang
        tgt = tgt_lang or self._config.tgt_lang

        batch_tokens = [self._encode(t, src) for t in texts]
        results = self._translator.translate_batch(
            batch_tokens,
            target_prefix=[[tgt]] * len(texts),
            beam_size=self._config.beam_size,
            max_batch_size=self._config.max_batch_size,
        )
        out: list[TranslationResult] = []
        for res in results:
            translated_tokens = res.hypotheses[0][1:]
            out.append(
                TranslationResult(
                    text=self._tokenizer.DecodePieces(translated_tokens),
                    src_lang=src,
                    tgt_lang=tgt,
                )
            )
        return out


__all__ = ["NLLBTranslatorBackend"]
