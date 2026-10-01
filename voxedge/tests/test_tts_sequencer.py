import re

from voxedge.engine.tts_sequencer import _to_speakable


def test_emphasis_removes_only_paired_markers():
    assert _to_speakable("这是 *重要* 和 **加粗**。") == "这是 重要 和 加粗。"
    assert _to_speakable("请**查看天气**再出门。") == "请查看天气再出门。"
    assert _to_speakable("这是**重点**。") == "这是重点。"
    assert _to_speakable("**a `code` b**") == "a code b"
    assert _to_speakable("**a `x` and `y` b**") == "a x and y b"


def test_literal_asterisk_and_underscore_are_preserved():
    for text in ("20*3=60", "x*y", "foo__bar", "未配对 * 标记"):
        assert _to_speakable(text) == text


def test_inline_code_and_markdown_link_behavior_remains():
    assert _to_speakable("`https://weather.com`") == "https://weather.com"
    assert _to_speakable("`__name__`") == "__name__"
    assert _to_speakable("`**literal**`") == "**literal**"
    assert _to_speakable("[Weather](https://weather.com)") == "Weather"


def test_cjk_bare_domains_are_spoken_without_character_loss():
    cases = {
        "请访问 example.org。": "请访问 example 点 o r g。",
        "网址是 www.nmc.cn。": "网址是 w w w 点 n m c 点 c n。",
        "Weather.com 很可靠。": "Weather 点 c o m 很可靠。",
        "网址是 www.example.com.": "网址是 w w w 点 example 点 c o m.",
        "有 example.org 和 example.com。": "有 example 点 o r g 和 example 点 c o m。",
    }
    for text, expected in cases.items():
        actual = _to_speakable(text, language="chinese")
        assert actual == expected
        reconstructed = actual.replace(" 点 ", ".")
        reconstructed = re.sub(r"(?<=[A-Za-z]) (?=[A-Za-z])", "", reconstructed)
        assert reconstructed == text


def test_cjk_domain_normalization_does_not_rewrite_other_literals():
    for text in (
        "foo@example.org 不改。",
        "版本 v1.2 不改。",
        "数学 3.14 不改。",
        "访问 https://example.org/path?q=1 不改。",
        "代码 `example.org` 不改。",
    ):
        expected = text.replace("`", "")
        assert _to_speakable(text, language="chinese") == expected


def test_pure_english_domain_is_unchanged():
    assert _to_speakable("Visit example.org.", language="english") == "Visit example.org."


def test_domain_normalization_requires_explicit_chinese_language():
    assert _to_speakable("请访问 example.org。") == "请访问 example.org。"
    assert _to_speakable("請訪問 example.org。", language="japanese") == "請訪問 example.org。"


def test_domain_candidates_with_url_or_email_context_are_untouched():
    for text in (
        "中文 https://example.org 不改。",
        "中文 https://example.org/path?q=1 不改。",
        "中文 a@sub.example.org 不改。",
        "中文 -foo.example.com 和 foo-.example.com 不改。",
        "版本 v1.2.3 不改。",
        "中文 www.example.com/path 不改。",
    ):
        assert _to_speakable(text, language="zh") == text


def test_domain_dns_length_boundaries_are_all_or_nothing():
    label62 = "a" * 62
    label63 = "b" * 63
    label64 = "c" * 64
    assert "点 c n" in _to_speakable(f"中文 {label62}.cn。", language="zh")
    assert "点 c n" in _to_speakable(f"中文 {label63}.cn。", language="zh")
    assert _to_speakable(f"中文 {label64}.cn。", language="zh") == f"中文 {label64}.cn。"

    total253 = ".".join(("d" * 63, "e" * 63, "f" * 63, "g" * 61))
    total254 = ".".join(("d" * 63, "e" * 63, "f" * 63, "g" * 62))
    assert _to_speakable(f"中文 {total253}。", language="zh") != f"中文 {total253}。"
    assert _to_speakable(f"中文 {total254}。", language="zh") == f"中文 {total254}。"


def test_markup_only_is_empty():
    assert _to_speakable("**__*") == ""
