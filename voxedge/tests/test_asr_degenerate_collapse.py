"""退化塌缩守卫：既要收掉真的复读，也不能误伤正常口语重复。

用例里的退化样本取自 2026-08-08 在 orin-nx 上的实测输出（离线与流式逐字相同）。
"""
from __future__ import annotations

import pytest

from voxedge.text.degenerate import collapse_repetition


@pytest.mark.parametrize(
    "text, want",
    [
        # 实测：300ms 片段
        ("帮我，" * 128, "帮我"),
        # 实测：600ms 片段（尾部是句号不是逗号）
        ("帮我查一下，帮我查一下，帮我查一下。", "帮我查一下"),
        # 实测：900ms 片段
        ("帮我查一下，帮我查一下。", "帮我查一下，帮我查一下。"),  # 只有 2 份，不塌缩
        # 被 max_generate_length 截断，尾部残缺一份
        ("好的好的好的好的好", "好的"),
        # 单字退化要到 8 份才收
        ("啊" * 12, "啊"),
    ],
)
def test_collapses_degenerate(text: str, want: str) -> None:
    got, _ = collapse_repetition(text)
    assert got == want


@pytest.mark.parametrize(
    "text",
    [
        "帮我查一下M6螺母的库存。",   # 正常整句
        "对对对",                      # 正常口语重复（单字 3 份）
        "好好好",
        "啊啊啊",                      # 单字 3 份，低于门槛
        "是的，我知道了。",
        "一二三四五六七八九十",
        "",
        "嗯",
        "行行行行",                    # 单字 4 份，仍低于门槛 8
        # 英文：以下都是合法说法，按字符切分会被黏成 thethethe / byebyebye
        # 而误砍，所以空格分隔文本按词判定且门槛 6 份。
        "the the the",
        "bye bye bye",
        "no no no no",
        "very very very good",
        "I need to check the M6 nut stock",
        "ha ha ha ha ha",
    ],
)
def test_leaves_normal_text_alone(text: str) -> None:
    got, collapsed = collapse_repetition(text)
    assert got == text
    assert collapsed is False


def test_reports_whether_it_collapsed() -> None:
    _, c1 = collapse_repetition("帮我，" * 10)
    assert c1 is True
    _, c2 = collapse_repetition("帮我查一下M6螺母的库存。")
    assert c2 is False


def test_picks_shortest_period() -> None:
    """「帮我，」×4 不能塌成「帮我，帮我」。"""
    got, _ = collapse_repetition("帮我，帮我，帮我，帮我，")
    assert got == "帮我"


def test_english_degeneration_still_caught() -> None:
    """英文门槛提高到 6 份，但真的退化（几十上百份）仍要收掉。"""
    got, collapsed = collapse_repetition("help me " * 30)
    assert collapsed is True
    assert got == "help me"


def test_english_punctuated_repeats() -> None:
    """逐词剥标点，"hello, hello, ..." 与无标点走同一判定。"""
    got, collapsed = collapse_repetition("hello, " * 10)
    assert collapsed is True
    assert got == "hello"


def test_mixed_content_not_collapsed() -> None:
    """局部重复但整段还有别的内容 —— 覆盖率不够，不动。"""
    text = "帮我，帮我，帮我，然后查一下M6螺母的库存和价格还有供应商信息"
    got, collapsed = collapse_repetition(text)
    assert got == text
    assert collapsed is False


# --- 尾部锚定：整段锚定要求周期从 index 0 起算，一个不重复的前缀就让守卫失效 ---


def test_degeneration_behind_hallucinated_prefix() -> None:
    """实测形态：模型先吐一个幻觉词再开始复读，前缀要留住、复读要收掉。

    spark 上一个 4 秒片段返回「一方面，」+「我们看到，」×66，共 334 字。
    只从 index 0 找周期时整段判不出周期，334 字原样返回。
    """
    got, collapsed = collapse_repetition("一方面，" + "我们看到，" * 66)
    assert collapsed is True
    assert got == "一方面，我们看到"


def test_long_prefix_survives_tail_collapse() -> None:
    """前缀是真实转写内容时，不能连它一起丢。"""
    prefix = "是芝加哥种族单一化程度最高的社区之一"
    got, collapsed = collapse_repetition(prefix + "我们看到，" * 40)
    assert collapsed is True
    assert got.startswith(prefix)
    assert got == prefix + "我们看到"


def test_english_tail_collapse_keeps_word_spacing() -> None:
    """英文走词切分，拼回去时前缀与单元之间要留空格。"""
    got, collapsed = collapse_repetition("so anyway " + "we see " * 20)
    assert collapsed is True
    assert got == "so anyway we see"


def test_short_tail_repeat_not_collapsed() -> None:
    """尾部只重复 2 份 —— 与正常强调无法区分，放过。"""
    text = "他说了这句话。他说了这句话。"
    got, collapsed = collapse_repetition(text)
    assert collapsed is False
    assert got == text


# --- 误伤边界：正常语音里的周期性内容不能被当成解码退化 ---


def test_periodic_phone_number_not_collapsed() -> None:
    """口述的号码天然是周期型的，份数少，与退化的几十上百份区分得开。"""
    for text in ("客服电话是123123123", "订单号 AB12AB12AB12", "他说了123123123这个号"):
        got, collapsed = collapse_repetition(text)
        assert collapsed is False, text
        assert got == text


def test_truncated_trailing_period_leaves_no_fragment() -> None:
    """结尾被 max_generate_length 切在半份上时，不能选中旋转过的周期。"""
    got, collapsed = collapse_repetition("前缀" + "我们看到" * 10 + "我们")
    assert collapsed is True
    assert got == "前缀我们看到"
    got, collapsed = collapse_repetition("prefix " + "we see " * 10 + "we")
    assert collapsed is True
    assert got == "prefix we see"


def test_oversized_input_skips_tail_scan() -> None:
    """尾部搜索最坏是 O(n^2)，超长输入宁可放过也不能成为性能雷区。"""
    from voxedge.text.degenerate import _MAX_TAIL_SCAN_UNITS

    text = "前缀" + "循环" * (_MAX_TAIL_SCAN_UNITS + 10)
    got, collapsed = collapse_repetition(text)
    # 整段锚定仍可能命中；这里只要求调用能在瞬间返回而不是卡死。
    assert isinstance(collapsed, bool)
    assert isinstance(got, str)


# ── 第三个锚点：段中间的复读 ────────────────────────────────────────────
#
# 前两个锚点一个从 index 0 起算、一个贴着结尾，两侧都有正常内容的复读它们都
# 判不出来。实测来源：Whisper-base 在 RK3588 上按 10s 窗口切分时，中文长句里
# 出现「…上下文语经中找到×18并能针对特定问题…」，前后都是正常转写。


def test_interior_run_is_collapsed_keeping_both_sides():
    text = "学生们可以在他文章的上下文语经中" + "找到" * 18 + "并能针对特定问题提出自己的观点"
    out, collapsed = collapse_repetition(text)
    assert collapsed
    assert out == "学生们可以在他文章的上下文语经中找到并能针对特定问题提出自己的观点"


def test_interior_run_in_spaced_language_keeps_the_word_break():
    # 塌缩点两侧必须还是两个词：拼成 "delayeduntil" 等于制造一个新的错词。
    out, collapsed = collapse_repetition("the meeting is " + "delayed " * 8 + "until friday")
    assert collapsed
    assert out == "the meeting is delayed until friday"


@pytest.mark.parametrize("text", [
    # 口语里合法的重复，份数都够不到门槛 —— 这一档没有覆盖率兜底，只靠份数，
    # 所以门槛比整段锚定更严正是为了这些。
    "对对对我明白了",
    "very very very good",
    "I said no no no no to that",
    "他说不行不行不行这样不行",
    "正常的一句中文转写没有任何复读",
])
def test_normal_speech_repetition_survives(text):
    out, collapsed = collapse_repetition(text)
    assert not collapsed
    assert out == text


def test_interior_guard_does_not_fire_on_a_long_clean_transcript():
    # 排比句式：结构重复但内容不同，不该被当成解码退化。
    text = "第一要看清楚，第二要想明白，第三要说得准，第四要做得实，第五要收得住"
    out, collapsed = collapse_repetition(text)
    assert not collapsed and out == text


# ── 门槛按单元长度分档 ──────────────────────────────────────────────────
#
# 空格分词语言原本统一要求 6 份，因为 "the the the" / "no no no no" 是合法英语。
# 但那是为**单个词**定的：一整句逐字重复 3 遍不是说话方式。实测来源：Hailo-8 上
# Whisper 不吐 EOS，把正确转写的句子原样重复到 token 预算耗尽，10 词单元 4 份卡
# 在 6 份门槛外，整段原样返回，WER 168%。

_SENTENCE = "Television reports show white smoke coming from the plant. "


def test_a_whole_sentence_repeated_four_times_is_collapsed():
    out, collapsed = collapse_repetition(_SENTENCE * 4 + "Television reports")
    assert collapsed
    assert out.count("Television") == 1


def test_a_long_repeated_prefix_is_collapsed_and_the_rest_kept():
    out, collapsed = collapse_repetition(
        "However, due to the slow communication channels, " * 3
        + "Styles in the West could lag behind by 25 30 years."
    )
    assert collapsed
    assert out.count("However") == 1
    assert out.endswith("25 30 years.")


@pytest.mark.parametrize("text", [
    # Short units keep the higher bar — these are all ordinary speech.
    "very very very good",
    "I said no no no no to that",
    "bye bye bye",
    # Two repeats of a long unit stay: two is indistinguishable from emphasis,
    # which is the module's standing rule.
    "I do not know what to do I do not know what to do",
    "the cat sat on the mat and then went away",
])
def test_short_units_and_double_repeats_survive(text):
    out, collapsed = collapse_repetition(text)
    assert not collapsed and out == text


def test_two_full_repeats_plus_a_started_third_is_collapsed():
    # 一个长单元重复两遍再起第三遍的头，是解码退化被 token 上限截断的样子。
    # 按完整份数只数到 2，正好落进"2 份一律放过"的规则里。实测来源：Hailo-8 上
    # cap=32 恰好放得下 2.5 遍短句。
    out, collapsed = collapse_repetition(
        "He referred to the rumors as political chatter and silliness. " * 2
        + "He referred to the rumors as political chatter"
    )
    assert collapsed
    assert out.count("He referred") == 1


def test_a_short_unit_twice_plus_a_started_third_survives():
    # 同样的形状，短单元：这是真实说法，长度门槛（>=6 词）就是为了保住它。
    text = "I love you. I love you. I love you so much"
    out, collapsed = collapse_repetition(text)
    assert not collapsed and out == text
