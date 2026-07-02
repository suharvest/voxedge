"""RK ASR finalize must not stringify tuple-returning inner backends.

qwen3_rk's ``finalize()`` returns ``(text, language)`` per the newer
ASRStream contract, while paraformer/sensevoice return plain strings.
Regression: the tuple reached ``_clean_segment_text``/``str()`` and
captions rendered as ``"('今天天气怎么样？', '')"`` (seen live on
radxa/RK3588, 2026-07-02).
"""

from voxedge.backends.rk.asr import _clean_segment_text, _unpack_inner_final


def test_unpack_tuple_inner():
    assert _unpack_inner_final(("你好", "zh")) == ("你好", "zh")


def test_unpack_tuple_empty_lang():
    assert _unpack_inner_final(("你好", "")) == ("你好", None)


def test_unpack_plain_string_inner():
    assert _unpack_inner_final("你好") == ("你好", None)


def test_unpack_none_and_empty():
    assert _unpack_inner_final(None) == ("", None)
    assert _unpack_inner_final("") == ("", None)
    assert _unpack_inner_final(()) == ("", None)


def test_clean_segment_text_never_sees_tuple_repr():
    text, lang = _unpack_inner_final(("今天天气怎么样？", ""))
    cleaned = _clean_segment_text(text)
    assert cleaned == "今天天气怎么样？"
    assert "('" not in cleaned
    assert lang is None
