import numpy as np

from voxedge.backends.rk.tts import RKTTSBackend


class Inner:
    name = "kokoro_convonly"

    def __init__(self):
        self.languages = []

    def synthesize(self, **kwargs):
        self.languages.append(kwargs["language"])
        return b"wav", {}

    def synthesize_stream(self, **kwargs):
        self.languages.append(kwargs["language"])
        yield np.zeros(1, dtype=np.int16), {}


def backend():
    b = RKTTSBackend(); b._inner = Inner(); return b


def test_kokoro_auto_kana_and_han_routes_ja_and_other_backends_keep_old():
    b = backend()
    assert b._resolve_language("今日はテストです", None) == "ja"
    assert b._resolve_language("テスト", "auto") == "ja"
    assert b._resolve_language("ﾃｽﾄ", "detect") == "ja"
    assert b._resolve_language("かな", "default") == "ja"
    assert b._resolve_language("ㇰ", None) == "ja"
    assert b._resolve_language("漢字ㇰ", None) == "ja"
    assert b._resolve_language("你好", None) == "zh"
    assert b._resolve_language("Hello", None) == "en"
    b._inner.name = "matcha_rknn"
    assert b._resolve_language("こんにちは", None) == "en"


def test_kokoro_explicit_languages_are_preserved_on_all_paths():
    b = backend()
    b._synthesize_impl("こんにちは", language="en-GB")
    list(b._generate_streaming_impl("こんにちは", language="zh"))
    list(b.synthesize_stream("こんにちは", language="ja"))
    assert b._inner.languages == ["en-GB", "zh", "ja"]


def test_kokoro_auto_language_is_forwarded_on_all_three_paths():
    b = backend()
    b._synthesize_impl("漢字かな", language="detect")
    list(b._generate_streaming_impl("カナ", language="default"))
    list(b.synthesize_stream("ﾃｽﾄ", language=" auto "))
    assert b._inner.languages == ["ja", "ja", "ja"]


def test_forwarding_preserves_han_latin_and_non_kokoro_auto_behavior():
    b = backend()
    b._synthesize_impl("漢字", language=None)
    list(b._generate_streaming_impl("Hello", language=None))
    list(b.synthesize_stream("漢字かな", language=None))
    assert b._inner.languages == ["zh", "en", "ja"]
    b = backend(); b._inner.name = "matcha_rknn"
    b._synthesize_impl("かな", language=None)
    list(b._generate_streaming_impl("かな", language=None))
    list(b.synthesize_stream("かな", language=None))
    assert b._inner.languages == ["en", "en", "en"]
