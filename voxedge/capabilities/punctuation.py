"""Punctuation restoration (CT-Transformer via sherpa-onnx) — voxedge engine.

Env-free per voxedge convention: ``model_path`` + ``num_threads`` are injected
at construction. Flag gating (OVS_PUNCT), path resolution and model download
stay in the product layer (e.g. seeed-local-voice ``server/core/punctuation``).

The CT-Transformer tokenizer + 272727-token vocab are embedded in the ONNX
file; we drive it through sherpa-onnx ``OfflinePunctuation`` so tokenization
matches the upstream model exactly (reproducing it by hand is fragile). CPU
only — backend/device independent (Jetson / RK / RPi). ``import sherpa_onnx``
is lazy so this module imports without the optional ``voxedge[sherpa]`` extra.
"""

from __future__ import annotations

import logging
import re
import threading

logger = logging.getLogger(__name__)

# Stable identifier surfaced in payloads so consumers can detect a model swap.
PUNCT_MODEL_NAME = "ct_transformer_zh_en_vocab272727_2024-04-12"

# Sentence/clause punctuation the CT-Transformer re-inserts. We strip these
# from the input first so already-punctuated ASR output (e.g. Qwen3/SenseVoice,
# which punctuate inline) doesn't get DOUBLED (`。` → `。。`). CT-Transformer is
# designed for unpunctuated text, so stripping is the correct usage and makes
# the result idempotent + backend-agnostic. Word-internal marks (apostrophe,
# hyphen, decimal point) are intentionally NOT stripped.
_STRIP_PUNCT = re.compile(r"[。，、！？；：,!?;:]+")


def _strip_existing_punct(text: str) -> str:
    # Replace stripped punctuation with a space so token boundaries survive,
    # then collapse runs of whitespace.
    return re.sub(r"\s+", " ", _STRIP_PUNCT.sub(" ", text)).strip()


class Punctuator:
    """Lazy, thread-safe wrapper around sherpa-onnx ``OfflinePunctuation``.

    The model loads once on first use; a hard load failure is sticky (we don't
    retry every call). ``add_punctuation`` never raises to the caller.
    """

    def __init__(self, model_path: str, num_threads: int = 2):
        self._model_path = model_path
        self._num_threads = num_threads
        self._punct = None
        self._lock = threading.Lock()
        self._failed = False

    def _ensure(self):
        if self._punct is not None:
            return self._punct
        if self._failed:
            return None
        with self._lock:
            if self._punct is not None:
                return self._punct
            if self._failed:
                return None
            try:
                import sherpa_onnx

                config = sherpa_onnx.OfflinePunctuationConfig(
                    model=sherpa_onnx.OfflinePunctuationModelConfig(
                        ct_transformer=self._model_path,
                        num_threads=self._num_threads,
                        provider="cpu",
                        debug=False,
                    ),
                )
                self._punct = sherpa_onnx.OfflinePunctuation(config)
                logger.info(
                    "Punctuator loaded (%s, threads=%d).",
                    self._model_path, self._num_threads,
                )
            except Exception:
                self._failed = True
                logger.exception("Failed to load punctuation model; disabled.")
                return None
        return self._punct

    def ready(self) -> bool:
        return self._ensure() is not None

    def add_punctuation(self, text: str) -> str:
        """Return ``text`` with restored punctuation, unchanged if the model is
        unavailable or the text is empty. Never raises.
        """
        if not text or not text.strip():
            return text
        punct = self._ensure()
        if punct is None:
            return text
        try:
            # Strip any punctuation the ASR backend already produced so we don't
            # double it; CT-Transformer wants unpunctuated input.
            cleaned = _strip_existing_punct(text)
            if not cleaned:
                return text
            return punct.add_punctuation(cleaned)
        except Exception:
            logger.exception("add_punctuation failed; returning original text.")
            return text
