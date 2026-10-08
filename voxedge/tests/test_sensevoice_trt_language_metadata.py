"""SenseVoice TRT language metadata from the CTC token prefix.

These tests use synthetic logits and the real backend decode/transcribe methods;
they do not load a TensorRT engine or a downloadable model.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxedge.backends.jetson.sensevoice_trt import (
    BLANK_ID,
    LFR_DIM,
    SenseVoiceTRTBackend,
    SenseVoiceTRTConfig,
    T_FIXED,
)


PIECES = {
    1: "<|zh|>",
    2: "<|en|>",
    3: "<|ja|>",
    4: "<|ko|>",
    5: "<|yue|>",
    6: "▁hello",
    7: "▁世",
    8: "界",
    9: "<|unknown|>",
    10: "<|xx|>",
    11: "<|SAD|>",
}


class _SentencePiece:
    def get_piece_size(self):
        return 32

    def id_to_piece(self, idx):
        return PIECES.get(idx, "")


def _logits(ids: list[int]) -> np.ndarray:
    out = np.full((T_FIXED, 32), -1.0, dtype=np.float32)
    out[:, BLANK_ID] = 0.0
    for row, idx in enumerate(ids):
        out[row, idx] = 1.0
    return out


def _backend(ids: list[int] | None) -> SenseVoiceTRTBackend:
    backend = SenseVoiceTRTBackend(SenseVoiceTRTConfig())
    backend._ctx = object()
    backend._ready = True
    backend._sp = _SentencePiece()
    backend._build_speech = lambda audio, lang="auto", textnorm="withitn": (
        np.zeros((1, T_FIXED, LFR_DIM), dtype=np.float32),
        len(ids or []),
    )
    backend._infer = lambda speech, valid: None if ids is None else _logits(ids)
    return backend


@pytest.mark.parametrize(
    ("tag", "language"),
    [("<|en|>", "en"), ("<|zh|>", "zh"), ("<|ja|>", "ja"),
     ("<|ko|>", "ko"), ("<|yue|>", "yue")],
)
def test_auto_reports_unique_leading_language_tag(tag, language):
    tag_id = next(idx for idx, piece in PIECES.items() if piece == tag)
    result = _backend([tag_id, 6]).transcribe_array(np.zeros(1), language="auto")
    assert result.language == language
    assert result.text == "hello"


@pytest.mark.parametrize("ids", [[], [9, 6], [10, 6], [1, 2, 6], [1, 10, 6], [6, 1]])
def test_auto_keeps_auto_without_unambiguous_leading_language(ids):
    result = _backend(ids).transcribe_array(np.zeros(1), language="auto")
    assert result.language == "auto"


@pytest.mark.parametrize(
    ("language_id", "expected"),
    [(1, "zh"), (2, "en"), (3, "ja")],
)
def test_known_emotion_header_does_not_hide_unique_language(language_id, expected):
    result = _backend([language_id, 11, 6]).transcribe_array(
        np.zeros(1), language="auto"
    )
    assert result.language == expected


def test_unknown_language_shaped_header_still_rejects_detection():
    result = _backend([1, 10, 6]).transcribe_array(np.zeros(1), language="auto")
    assert result.language == "auto"


def test_conflicting_language_headers_still_reject_detection():
    result = _backend([1, 11, 2, 6]).transcribe_array(np.zeros(1), language="auto")
    assert result.language == "auto"


def test_language_tag_in_body_is_not_used_for_detection_and_text_cleanup_is_unchanged():
    result = _backend([6, 1, 7, 8]).transcribe_array(np.zeros(1), language="auto")
    assert result.language == "auto"
    assert result.text == "hello 世界"


def test_explicit_language_is_preserved_even_when_header_differs():
    result = _backend([1, 6]).transcribe_array(np.zeros(1), language="en")
    assert result.language == "en"
    assert result.text == "hello"


def test_unsupported_language_still_warns_and_reports_honoured_auto(caplog):
    result = _backend([1, 6]).transcribe_array(np.zeros(1), language="fr")
    assert result.language == "auto"
    assert "per-request language 'fr' ignored" in caplog.text


def test_empty_logits_preserve_original_empty_result_language():
    result = _backend(None).transcribe_array(np.zeros(1), language="auto")
    assert result.text == ""
    assert result.language == "auto"


def test_legacy_ctc_decode_returns_only_the_original_text_shape():
    backend = _backend([1, 1, 6, 6, BLANK_ID, 7, 8])
    assert backend._ctc_decode(_logits([1, 1, 6, 6, BLANK_ID, 7, 8]), 7) == "hello 世界"
