"""ASR 退化输出（自回归循环）的塌缩守卫。

Qwen3-ASR 在短音频上会退化成整段复读：300ms 片段实测输出「帮我，」×128，
600ms 输出「帮我查一下，」×3。这不是流式路径的问题 —— 同一段音频走离线
/asr 得到逐字相同的结果（2026-08-08 实测），根因是贪心解码（top_k=1）下的
自回归退化。worker 只接受 temperature/top_k/top_p/max_generate_length，
不支持 repetition_penalty 或 no_repeat_ngram，解码侧无法治，只能在文本侧兜。

守卫刻意保守：宁可漏掉一些退化，也不能把正常的口语重复误伤成单份。
"""
from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

# 中文按字符切分，「对对对」「好好好」是正常说法，所以单字门槛远高于多字。
_MIN_REPEATS_SINGLE_CHAR = 8
_MIN_REPEATS_MULTI_CHAR = 3
# 空格分隔的语言（英文等）按词切分，门槛必须更高：英文里 "the the the"、
# "bye bye bye"、"no no no no" 都是合法说法，按字符切分还会把它们黏成
# "thethethe" 从而误判。真正的解码退化通常复读几十上百份，6 份足以区分。
_MIN_REPEATS_SPACED = 6
# 重复部分必须占到整段的这个比例，才认为整段是退化产物而非局部口吃。
_MIN_COVERAGE = 0.8
_SEPARATORS = "，,。.、！!？?；;：: \t\n"


def _find_period(
    seq: Sequence, min_repeats: int, min_repeats_single: Optional[int] = None
) -> Optional[Sequence]:
    """找出能解释整段的最短周期单元；找不到返回 None。

    单元长度从短到长试，取最短周期 —— 否则「帮我，」×4 会得到「帮我，帮我」。
    ``min_repeats_single`` 只对单元长度为 1 的情况生效（中文单字重复门槛更高）；
    不传则与 ``min_repeats`` 相同。
    """
    n = len(seq)
    if n < 4:
        return None
    # 单元最长只需试到 n // 最小门槛：更长的单元不可能重复够份数。
    max_unit_len = n // min(min_repeats, min_repeats_single or min_repeats)
    for unit_len in range(1, max_unit_len + 1):
        unit = seq[:unit_len]
        repeats = 1
        while seq[repeats * unit_len:(repeats + 1) * unit_len] == unit:
            repeats += 1
        need = min_repeats_single if unit_len == 1 and min_repeats_single else min_repeats
        if repeats < need:
            continue
        covered = repeats * unit_len
        # 尾部残缺一份（被 max_generate_length 截断）时整段仍算被覆盖，
        # 否则截断反而让守卫失效。
        if unit[:n - covered] == seq[covered:]:
            covered = n
        if covered < _MIN_COVERAGE * n:
            continue
        return unit
    return None


def collapse_repetition(text: str) -> Tuple[str, bool]:
    """把整段复读塌缩成一份。返回 (结果, 是否发生塌缩)。

    分两种切分：
    - 含空白的文本（英文等）按**词**求周期，门槛 6 份。按字符会把
      "the the the" 黏成 "thethethe" 并砍成 "the"，那是合法英文。
    - 无空白的文本（中日韩）按**字符**求周期，且在剥离标点后进行 ——
      退化的最后一份常以句号收尾而前面都是逗号（实测
      「帮我查一下，帮我查一下，帮我查一下。」），逐字比对会数成 2 份而漏判。

    已知局限：只重复 2 份不会被收 —— 2 份与正常的强调式重复无法从文本上
    区分，宁可放过。
    """
    stripped = text.strip() if text else ""
    if not stripped:
        return text, False

    if any(ch.isspace() for ch in stripped):
        # 逐词剥掉首尾标点再比较，让 "hello, hello, hello" 与
        # "hello hello hello." 走同一条判定。
        units: Sequence = [
            w for w in (x.strip(_SEPARATORS) for x in re.split(r"\s+", stripped)) if w
        ]
        thresholds, joiner = (_MIN_REPEATS_SPACED,), " "
    else:
        units = [ch for ch in stripped if ch not in _SEPARATORS]
        thresholds = (_MIN_REPEATS_MULTI_CHAR, _MIN_REPEATS_SINGLE_CHAR)
        joiner = ""

    unit = _find_period(units, *thresholds)
    return (joiner.join(unit), True) if unit else (text, False)


__all__ = ["collapse_repetition"]
