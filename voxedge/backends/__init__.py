"""voxedge backend ABCs + mock implementations."""
from __future__ import annotations

from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    LLMBackend,
    LLMEvent,
    TranscriptionResult,
    TranslationResult,
    TranslatorBackend,
    TranslatorCapability,
    TranslatorConfig,
    TTSBackend,
    TTSCapability,
    VADBackend,
    VADSession,
)

__all__ = [
    "ASRBackend",
    "ASRCapability",
    "ASRStream",
    "LLMBackend",
    "LLMEvent",
    "TranscriptionResult",
    "TranslationResult",
    "TranslatorBackend",
    "TranslatorCapability",
    "TranslatorConfig",
    "TTSBackend",
    "TTSCapability",
    "VADBackend",
    "VADSession",
]
