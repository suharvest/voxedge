"""跨段复读的塌缩。

``collapse_repetition`` 只看单段。退化也会横跨切段：每段各吐一次同一句幻觉，
段内看都只出现一次，拼接后才叠成一片。实测一段 35 分钟音频结尾处
「中国在国际上的话语权和影响力在不断增强，国际地位在不断提升」连续占了 11 段。
"""
from voxedge.text.degenerate import collapse_segment_repeats

HALLUCINATION = "一方面，中国在国际上的话语权和影响力在不断增强，国际地位在不断提升。"


def test_run_of_identical_segments_collapsed() -> None:
    texts = ["前面的真实内容。"] + [HALLUCINATION] * 11 + ["后面的真实内容。"]
    got, dropped = collapse_segment_repeats(texts)
    assert dropped == 10
    assert got == ["前面的真实内容。", HALLUCINATION, "后面的真实内容。"]


def test_trailing_punctuation_differences_still_match() -> None:
    """同一句在不同段的收尾标点常不一样，比较前要剥掉尾部标点。"""
    a = "这是一句足够长的话用来触发长段门槛"
    got, dropped = collapse_segment_repeats([a + "。", a + "，", a + "！", "B"])
    assert dropped == 2
    assert got == [a + "。", "B"]


def test_inner_punctuation_is_not_stripped() -> None:
    """只剥尾部标点：剥掉句中标点会让语义不同的段撞键。"""
    texts = ["不，行。", "不行。", "不，行！"]
    got, dropped = collapse_segment_repeats(texts)
    assert dropped == 0
    assert got == texts


def test_short_segments_need_more_repeats() -> None:
    """VAD 把连续的短语切成三段是正常语音，不是退化。"""
    for texts in (["啊"] * 3, ["对不起。"] * 3, ["嗯"] * 3):
        got, dropped = collapse_segment_repeats(texts)
        assert dropped == 0, texts
        assert got == texts


def test_short_segments_collapsed_once_run_is_long() -> None:
    """短段连着六份就不再像正常强调了。"""
    got, dropped = collapse_segment_repeats(["对不起。"] * 6)
    assert dropped == 5
    assert got == ["对不起。"]


def test_two_identical_segments_left_alone() -> None:
    """两段说同一句话可能是正常的重复强调，门槛与段内一致取 3 份。"""
    texts = ["同一句话在这里。", "同一句话在这里。", "别的内容"]
    got, dropped = collapse_segment_repeats(texts)
    assert dropped == 0
    assert got == texts


def test_non_adjacent_repeats_left_alone() -> None:
    """隔开的重复是正常复述，不收。"""
    texts = ["X句", "Y句", "X句", "Y句", "X句"]
    got, dropped = collapse_segment_repeats(texts)
    assert dropped == 0
    assert got == texts


def test_normal_transcript_untouched() -> None:
    texts = ["大家好。", "今天聊芝加哥。", "一九九八年。"]
    got, dropped = collapse_segment_repeats(texts)
    assert dropped == 0
    assert got == texts


def test_empty_segments_are_not_treated_as_repeats() -> None:
    """空串不该被当成互相重复而被丢掉。"""
    texts = ["", "", "", "实际内容"]
    got, dropped = collapse_segment_repeats(texts)
    assert dropped == 0
    assert got == texts
