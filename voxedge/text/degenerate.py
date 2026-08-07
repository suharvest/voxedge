"""ASR 退化输出（自回归循环）的塌缩守卫。

Qwen3-ASR 在短音频上会退化成整段复读：300ms 片段实测输出「帮我，」×128，
600ms 输出「帮我查一下，」×3。这不是流式路径的问题 —— 同一段音频走离线
/asr 得到逐字相同的结果（2026-08-08 实测），根因是贪心解码（top_k=1）下的
自回归退化。worker 只接受 temperature/top_k/top_p/max_generate_length，
不支持 repetition_penalty 或 no_repeat_ngram，解码侧无法治，只能在文本侧兜。

守卫刻意保守：宁可漏掉一些退化，也不能把正常的口语重复（「对对对」「好好好」）
误伤成单字。
"""
from __future__ import annotations

from typing import Tuple

# 单字重复的门槛远高于多字：中文口语里「对对对」「好好好」完全正常，
# 而「啊」连出八次以上基本只可能是解码退化。
_MIN_REPEATS_SINGLE_CHAR = 8
_MIN_REPEATS_MULTI_CHAR = 3
# 重复部分必须占到整段的这个比例，才认为整段是退化产物而非局部口吃。
_MIN_COVERAGE = 0.8
# 分隔符跟在重复单元尾部（「帮我，」），塌缩后要去掉，避免留下孤立标点。
_TRAILING_SEPARATORS = "，,。.、！!？?；; \t\n"


def collapse_repetition(text: str) -> Tuple[str, bool]:
    """把整段复读塌缩成一份。返回 (结果, 是否发生塌缩)。

    周期在**剥离标点后**的字符序列上求：退化输出的最后一份往往以句号收尾而
    前面都是逗号（实测「帮我查一下，帮我查一下，帮我查一下。」），逐字比对会
    把它数成 2 份而漏判。

    单元长度从短到长试，取最短周期，这样「帮我，帮我，帮我，」得到「帮我」
    而不是「帮我，帮我」。

    已知局限：只重复 2 份（如「帮我查一下，帮我查一下。」）不会被收 —— 2 份
    与正常的强调式重复无法从文本上区分，宁可放过。
    """
    if not text:
        return text, False

    core = "".join(ch for ch in text if ch not in _TRAILING_SEPARATORS)
    n = len(core)
    if n < 4:
        return text, False

    for unit_len in range(1, n // _MIN_REPEATS_MULTI_CHAR + 1):
        unit = core[:unit_len]
        repeats = 1
        while core[repeats * unit_len:(repeats + 1) * unit_len] == unit:
            repeats += 1
        if repeats < 2:
            continue

        need = _MIN_REPEATS_SINGLE_CHAR if unit_len == 1 else _MIN_REPEATS_MULTI_CHAR
        if repeats < need:
            continue

        covered = repeats * unit_len
        # 尾部允许残缺一份（被 max_generate_length 截断），同样算进覆盖率，
        # 否则截断反而让守卫失效。
        tail = core[covered:]
        if tail and unit.startswith(tail):
            covered += len(tail)
        if covered < _MIN_COVERAGE * n:
            continue

        return unit, True

    return text, False


__all__ = ["collapse_repetition"]
