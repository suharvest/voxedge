"""Arabic digits -> spoken Chinese, for TTS front-ends that cannot pronounce them.

The lexicon-based Chinese front-ends in this package (matcha, kokoro, sherpa)
convert text by looking each token up in ``tokens.txt`` / ``lexicon.txt``. On
the shipped ``matcha-icefall-zh-en`` assets those tables contain every Chinese
numeral (零一二三四五六七八九十) and **no Arabic digit at all** — of "0123456789"
only "1" appears in tokens.txt, and lexicon.txt has none. A lookup miss is
silently skipped, so digits are not mispronounced: they produce no audio.

Measured on an Orin NX (16 kHz mono, seeed-voice v0.9.1):

    库存945个        1.47 s   |  库存九百四十五个     2.24 s
    共16个批次       1.60 s   |  共十六个批次         1.90 s
    位于A-02-01      1.34 s   |  位于A零二零一        2.10 s

Each digit-bearing utterance is about as long as the same sentence with the
number deleted. For a warehouse assistant this is not cosmetic — quantities,
batch counts and bin codes are the entire payload.

Two readings, chosen by context:

* counts -> cardinal              945 个   -> 九百四十五个
* identifiers -> digit by digit   A-02-01  -> A 零二 零一

A run of digits is treated as an identifier when it is glued to letters or
hyphens (``M6``, ``SKU-0003``, ``C2-2-07``). Four digits before 年 read
digit-by-digit, as Chinese convention wants 二零二六年 rather than
二千零二十六年.
"""
from __future__ import annotations

import re

_DIGITS = "零一二三四五六七八九"
_UNITS = ["", "十", "百", "千"]
_BIG = ["", "万", "亿"]


def cardinal(n: int) -> str:
    """Spoken form of an integer.

    Scans most-significant first, carrying a "zero pending" flag so any run of
    zeros contributes exactly one 零, and only when a non-zero digit follows
    (1000005 -> 一百万零五, but 1000000 -> 一百万). A leading 十 drops its 一
    only when it is the whole number's highest place (15 -> 十五, but
    115 -> 一百一十五).
    """
    if n == 0:
        return _DIGITS[0]
    sign = "负" if n < 0 else ""
    digits = str(abs(n))
    length = len(digits)
    out: list[str] = []
    zero_pending = False
    for i, ch in enumerate(digits):
        d = int(ch)
        pos = length - 1 - i          # distance from the ones place
        big = _BIG[pos // 4]          # 万 / 亿
        unit = _UNITS[pos % 4]        # 千 / 百 / 十
        if d == 0:
            # A group boundary still needs its 万/亿 even when this digit is 0
            # (100000000 -> 一亿).
            if pos % 4 == 0 and any(c != "0" for c in digits[max(0, i - 3):i]):
                out.append(big)
            zero_pending = bool(out)
            continue
        if zero_pending:
            out.append(_DIGITS[0])
            zero_pending = False
        if d == 1 and unit == "十" and i == 0:
            out.append(unit)
        else:
            out.append(_DIGITS[d] + unit)
        if pos % 4 == 0:
            out.append(big)
    return sign + "".join(out)


def digit_by_digit(s: str) -> str:
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for c in s)


# Digits glued to letters or hyphens: a part number or a bin code.
_CODE = re.compile(r"[A-Za-z]+[A-Za-z0-9\-]*\d[A-Za-z0-9\-]*|\d+(?:-\d+)+")
# A bare integer, thousands separators allowed.
_PLAIN = re.compile(r"\d[\d,]*")
_YEAR = re.compile(r"(?<!\d)(\d{4})(?=\s*年)")


def normalize(text: str) -> str:
    """Rewrite Arabic digits into their spoken Chinese form."""
    if not text:
        return text

    def _code(m: re.Match) -> str:
        # A hyphen inside a code reads as a short pause; saying 杠 is worse.
        return digit_by_digit(m.group(0)).replace("-", " ")

    text = _CODE.sub(_code, text)
    text = _YEAR.sub(lambda m: digit_by_digit(m.group(1)), text)

    def _plain(m: re.Match) -> str:
        raw = m.group(0).rstrip(",")
        digits = raw.replace(",", "")
        if not digits.isdigit():
            return m.group(0)
        # Very long runs are almost never a quantity; reading them as a cardinal
        # produces something nobody can follow.
        if len(digits) > 8:
            return digit_by_digit(digits)
        return cardinal(int(digits)) + m.group(0)[len(raw):]

    return _PLAIN.sub(_plain, text)
