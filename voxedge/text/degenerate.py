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
# 尾部锚定时前面还留着有效内容，覆盖率不可能像整段锚定那样高，所以改用**份数**
# 兜底：正常语音里周期性重复的尾巴（电话号码「123123123」、编号「AB12AB12」）份数
# 都很少，而解码退化动辄几十上百份。只放宽覆盖率不加份数门槛会误砍号码 ——
# 「客服电话是123123123」的尾部覆盖率有 0.64。
_MIN_TAIL_COVERAGE = 0.5
_MIN_TAIL_REPEATS = 6
_MIN_TAIL_REPEATS_SINGLE_CHAR = 12
# 超过这个单元数就不做尾部搜索：该搜索最坏是 O(n²)，而正常的 ASR 段只有百来个
# 单元。真有超长输入时宁可放过，也不能让守卫本身变成性能雷区 ——
# 实测 2048 单元约 0.06s，4096 就要 0.43s。
_MAX_TAIL_SCAN_UNITS = 2048
# 段**中间**的复读：两侧都有正常内容，既不从 index 0 起算也不贴着结尾，前两个
# 锚点都判不出来。实测 Whisper 在 RK3588 上把「…上下文语境中找到自己的立场，
# 并能够…」吐成「…语经中找到×18并能针对…」——前后都是正常转写。
# 这一档没有覆盖率兜底（复读只占整段一小部分），所以份数门槛取得比整段锚定更
# 严：宁可放过，也不能把「好好好好」这类正常说法从句子中间挖掉。
_MIN_INTERIOR_REPEATS = 6
_MIN_INTERIOR_REPEATS_SINGLE = 12
# 单元最长扫到这么多个单位：再长的"复读"更可能是正常的排比句式。
_MAX_INTERIOR_UNIT_LEN = 12
# 跨段：短段（少于这么多有效字符）要更多份数才当退化，否则 VAD 把「对不起。」
# 切成三段就会被删掉两段。
_MIN_SEGMENT_KEY_LEN = 8
_MIN_SEGMENT_RUN_SHORT = 6
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


def _find_tail_period(
    seq: Sequence, min_repeats: int, min_repeats_single: Optional[int] = None
) -> Tuple[Optional[Sequence], int]:
    """找出**结尾处**的复读周期；返回 (单元, 复读起始下标)，找不到返回 (None, len)。

    ``_find_period`` 要求周期从 index 0 起算，于是任何一个不重复的前缀都能让
    守卫整段失效。实测里这个前缀几乎总在：模型在音乐/低信噪比段先吐一个幻觉
    词再开始复读，例如「一方面，我们看到，我们看到，×66」——整段锚定判不出周期，
    334 字原样返回。改从尾部锚定后，前缀保留、复读收成一份。
    """
    n = len(seq)
    if n < 4 or n > _MAX_TAIL_SCAN_UNITS:
        return None, n
    max_unit_len = n // _MIN_TAIL_REPEATS
    for unit_len in range(1, max_unit_len + 1):
        need = _MIN_TAIL_REPEATS_SINGLE_CHAR if unit_len == 1 else _MIN_TAIL_REPEATS
        # `phase` 允许最后一份是残缺的：max_generate_length 截断常把结尾切在半份
        # 上（「…我们看到我们看到我们」）。同一个 unit_len 下多个相位都可能成立，
        # 但它们是同一个周期的旋转，取覆盖最大的那个 —— 否则「前缀+我们看到×10+
        # 我们」会选中旋转版「看到我们」，塌缩后留下「前缀我们看到我们」的残片。
        best = None
        for phase in range(unit_len):
            end = n - phase
            unit = seq[end - unit_len:end]
            if phase and seq[end:] != unit[:phase]:
                continue
            repeats = 1
            while True:
                stop = end - repeats * unit_len
                start = stop - unit_len
                if start < 0 or seq[start:stop] != unit:
                    break
                repeats += 1
            if repeats < need:
                continue
            covered = repeats * unit_len + phase
            if covered < _MIN_TAIL_COVERAGE * n:
                continue
            if best is None or covered > best[0]:
                best = (covered, unit)
        if best is not None:
            return best[1], n - best[0]
    return None, n


def _find_interior_run(
    seq: Sequence, min_repeats: int, min_repeats_single: Optional[int] = None
) -> Optional[Tuple[int, int, int]]:
    """找出段中间连续复读的一段；返回 (起始下标, 单元长度, 份数)。

    与另外两个锚点的区别是它不要求复读解释整段，也不要求贴着结尾 —— 代价是
    没有覆盖率可以兜底，所以只靠份数判定。有多处命中时取覆盖字数最多的那处。
    """
    n = len(seq)
    if n < 4 or n > _MAX_TAIL_SCAN_UNITS:
        return None
    best: Optional[Tuple[int, int, int, int]] = None   # (covered, start, unit_len, repeats)
    for unit_len in range(1, min(_MAX_INTERIOR_UNIT_LEN, n // min_repeats) + 1):
        need = min_repeats_single if (unit_len == 1 and min_repeats_single) else min_repeats
        i = 0
        while i + unit_len * need <= n:
            unit = seq[i:i + unit_len]
            repeats = 1
            while seq[i + repeats * unit_len:i + (repeats + 1) * unit_len] == unit:
                repeats += 1
            if repeats >= need:
                covered = repeats * unit_len
                if best is None or covered > best[0]:
                    best = (covered, i, unit_len, repeats)
                i += repeats * unit_len
            else:
                i += 1
    return (best[1], best[2], best[3]) if best else None


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

    # 同时记下每个单元在 stripped 里的起点：尾部锚定要把前缀按原样切回来，
    # 只有单元序列是不够的（标点在切分时已被剥掉）。
    units: list = []
    offsets: list = []
    if any(ch.isspace() for ch in stripped):
        # 逐词剥掉首尾标点再比较，让 "hello, hello, hello" 与
        # "hello hello hello." 走同一条判定。
        for match in re.finditer(r"\S+", stripped):
            word = match.group(0).strip(_SEPARATORS)
            if word:
                units.append(word)
                offsets.append(match.start())
        thresholds, joiner = (_MIN_REPEATS_SPACED,), " "
    else:
        for index, ch in enumerate(stripped):
            if ch not in _SEPARATORS:
                units.append(ch)
                offsets.append(index)
        thresholds = (_MIN_REPEATS_MULTI_CHAR, _MIN_REPEATS_SINGLE_CHAR)
        joiner = ""

    unit = _find_period(units, *thresholds)
    if unit:
        return joiner.join(unit), True

    tail_unit, tail_start = _find_tail_period(units, *thresholds)
    if tail_unit and tail_start > 0:
        prefix = stripped[:offsets[tail_start]].rstrip()
        return joiner.join([prefix, joiner.join(tail_unit)]), True

    interior_thresholds = (
        (_MIN_INTERIOR_REPEATS,) if len(thresholds) == 1
        else (_MIN_INTERIOR_REPEATS, _MIN_INTERIOR_REPEATS_SINGLE)
    )
    found = _find_interior_run(units, *interior_thresholds)
    if found:
        start, unit_len, repeats = found
        end = start + unit_len * repeats
        # 保留第一份，删掉其余。按 stripped 的原始下标切，标点和两侧文本原样保留。
        keep_to = offsets[start + unit_len]
        resume_from = offsets[end] if end < len(offsets) else len(stripped)
        head = stripped[:keep_to].rstrip(_SEPARATORS)
        return joiner.join([head, stripped[resume_from:]]) if joiner else head + stripped[resume_from:], True
    return text, False


def collapse_segment_repeats(texts: Sequence[str]) -> Tuple[list, int]:
    """把连续多段吐出同一句话的情况收成一份。返回 (结果, 被丢弃的段数)。

    ``collapse_repetition`` 只看单段，段内重复够份数才收。但退化也会横跨切段：
    每段各吐一次同一句幻觉，段内看都只出现一次，拼接后才叠成一片。实测一段
    35 分钟音频结尾处「中国在国际上的话语权和影响力在不断增强，国际地位在不断
    提升」连续占了 11 段，段内守卫一段都收不掉。

    只剥**尾部**标点后比较：同一句在不同段的收尾标点常不一样（「A句。」「A句，」），
    但剥掉句中标点会让语义不同的段撞键 —— 「不，行。」和「不行。」并不是同一句。

    门槛分两档，因为短段落太容易假阳性：VAD 把连续的「对不起。」「啊」切成三段是
    正常语音，不是退化。实测的退化形态是一整句幻觉（30+ 字）连占十几段，所以长段
    3 份即收，短段要 6 份。
    """
    kept: list = []
    dropped = 0
    run_start = 0  # index in `kept` where the current run of identical texts began

    def key(s: str) -> str:
        return s.strip().rstrip(_SEPARATORS)

    def enough(run_len: int, text: str) -> bool:
        if run_len < _MIN_REPEATS_MULTI_CHAR:
            return False
        return run_len >= _MIN_SEGMENT_RUN_SHORT or len(key(text)) >= _MIN_SEGMENT_KEY_LEN

    for text in texts:
        if kept and key(text) == key(kept[-1]) and key(text):
            kept.append(text)
        else:
            # A run ended — trim it back to one entry if it was long enough.
            run_len = len(kept) - run_start
            if kept and enough(run_len, kept[run_start]):
                del kept[run_start + 1:]
                dropped += run_len - 1
            kept.append(text)
            run_start = len(kept) - 1

    run_len = len(kept) - run_start
    if kept and enough(run_len, kept[run_start]):
        del kept[run_start + 1:]
        dropped += run_len - 1
    return kept, dropped


__all__ = ["collapse_repetition", "collapse_segment_repeats"]
