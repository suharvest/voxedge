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
        "访问 https://example.org/path?q=1 不改。",
        "代码 `example.org` 不改。",
    ):
        expected = text.replace("`", "")
        assert _to_speakable(text, language="chinese") == expected

    # A decimal is a quantity, not one of the literals the domain pass protects,
    # so it is rewritten — by _normalize_spoken_numbers, not by this pass. The
    # pair below is here rather than in its own test so the boundary between the
    # two passes stays visible.
    assert _to_speakable("圆周率 3.14。", language="chinese") == "圆周率 三点一四。"


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


# ── Arabic numerals ───────────────────────────────────────────────────
# The matcha frontend classifies by script and hands every digit to espeak-ng,
# which speaks English: "28 度" comes out "twenty-eight 度". Chinese output
# gets the digits rewritten before the backend sees them. The expected strings
# below are what the reply should say, not a token sequence.


def test_chinese_quantities_are_spoken_as_numerals():
    assert _to_speakable("现在亮度是33。", language="zh") == "现在亮度是三十三。"
    assert _to_speakable("调到 28 度", language="zh") == "调到 二十八 度"
    assert _to_speakable("温度 3.5 度", language="zh") == "温度 三点五 度"
    assert _to_speakable("完成度 50%", language="zh") == "完成度 百分之五十"
    assert _to_speakable("客厅有 2 盏灯", language="zh") == "客厅有 二 盏灯"


def test_chinese_identifiers_are_spoken_digit_by_digit():
    # A cardinal reading would turn 01-00 into 一杠零, which is worse than the
    # English it replaces; the separator is dropped rather than spoken.
    assert (
        _to_speakable("中弘空调-01-00关好了。", language="zh")
        == "中弘空调零一零零关好了。"
    )
    assert (
        _to_speakable("其他几台（01-00、01-02）都不支持。", language="zh")
        == "其他几台（零一零零、零一零二）都不支持。"
    )


def test_chinese_dates_times_and_years():
    assert (
        _to_speakable("日期 2026-10-16", language="zh")
        == "日期 二零二六年十月十六日"
    )
    assert _to_speakable("现在是 14:05", language="zh") == "现在是 十四点零五"
    assert _to_speakable("现在是 8:30", language="zh") == "现在是 八点三十"
    assert _to_speakable("2026年见", language="zh") == "二零二六年见"


def test_number_normalization_requires_explicit_chinese_language():
    # An English deployment's TTS reads its own digits; rewriting them here
    # would be wrong, and an unset language must not opt in either.
    assert _to_speakable("brightness 33", language="en") == "brightness 33"
    assert _to_speakable("brightness 33", language="english") == "brightness 33"
    assert _to_speakable("现在亮度是33。") == "现在亮度是33。"


def test_digits_glued_to_letters_are_left_alone():
    # A model id or a version number is not a quantity.
    assert _to_speakable("版本 v0.0.15a3", language="zh") == "版本 v0.0.15a3"
    assert _to_speakable("RK3576 平台", language="zh") == "RK3576 平台"


def test_int_to_chinese_reads_the_way_it_is_said():
    from voxedge.engine.tts_sequencer import _int_to_chinese

    assert _int_to_chinese(0) == "零"
    assert _int_to_chinese(10) == "十"
    assert _int_to_chinese(18) == "十八"      # not 一十八
    assert _int_to_chinese(20) == "二十"
    assert _int_to_chinese(105) == "一百零五"
    assert _int_to_chinese(1005) == "一千零五"
    assert _int_to_chinese(1100) == "一千一百"
    assert _int_to_chinese(10001) == "一万零一"
    assert _int_to_chinese(100000005) == "一亿零五"
    assert _int_to_chinese(12345678) == "一千二百三十四万五千六百七十八"
