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
