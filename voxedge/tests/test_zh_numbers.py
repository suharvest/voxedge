"""阿拉伯数字读法的回归测试。

锁住的不是「读得好听」，而是「数字根本发不出声」这个失效模式。matcha-icefall-zh-en
的 tokens.txt / lexicon.txt 里没有阿拉伯数字（tokens.txt 中 0-9 只有 "1"），
后端查表未命中即静默跳过，于是「945」不是读错而是不发音。仓库场景里数量、批次、
库位全是数字，等于整条信息丢失，而日志、字幕一切正常 —— 没有任何东西会报错。
"""
import pytest

from voxedge.text.zh_numbers import cardinal, digit_by_digit, normalize


@pytest.mark.parametrize("n,want", [
    (0, "零"), (5, "五"), (10, "十"), (11, "十一"), (15, "十五"), (20, "二十"),
    (100, "一百"), (101, "一百零一"), (105, "一百零五"), (110, "一百一十"),
    (945, "九百四十五"), (1000, "一千"), (1024, "一千零二十四"), (1100, "一千一百"),
    (10000, "一万"), (100000, "十万"),
    # 整段缺位只补一个「零」，且后面还有非零位时才补
    (1000000, "一百万"), (1000005, "一百万零五"),
    (100000000, "一亿"),
    (20260806, "二千零二十六万零八百零六"),
    (-7, "负七"),
])
def test_cardinal(n, want):
    assert cardinal(n) == want


def test_ten_keeps_one_when_not_highest_place():
    """15 读「十五」，115 读「一百一十五」—— 省略只在十位是最高位时成立。"""
    assert cardinal(15) == "十五"
    assert cardinal(115) == "一百一十五"


@pytest.mark.parametrize("src,want", [
    ("库存945个", "库存九百四十五个"),
    ("共16个批次", "共十六个批次"),
    # 与字母/连字符相连 = 编号，逐位读
    ("位于A-02-01", "位于A 零二 零一"),
    ("M6 螺母", "M六 螺母"),
    ("SKU-0003", "SKU 零零零三"),
    # 年份按惯例逐位
    ("现在是 2026 年", "现在是 二零二六 年"),
    # 无数字不动
    ("没有数字的句子", "没有数字的句子"),
    ("", ""),
])
def test_normalize(src, want):
    assert normalize(src) == want


def test_year_rule_does_not_hit_plain_counts():
    """「2026 个」是计数不是年份，仍读整数。"""
    assert normalize("2026 个") == "二千零二十六 个"


def test_long_digit_run_reads_digit_by_digit():
    """超长数字串当编号读 —— 整数读法在语音里没人跟得上。"""
    assert normalize("1234567890") == digit_by_digit("1234567890")


def test_no_arabic_digit_survives():
    """兜底：规范化后不应再有阿拉伯数字，否则后端仍会静默丢弃。"""
    for src in ["库存945个，共16个批次，位于A-02-01",
                "轴承-NJ409MC3 在 C2-2-07，还有 5 个",
                "订单 SKU-0003 数量 1,200"]:
        assert not any(c.isdigit() for c in normalize(src)), src
