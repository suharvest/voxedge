"""Join per-segment ASR transcripts back into one utterance.

Every backend that splits long audio (see ``voxedge.audio.segment``) needs this
on the way out, and two of them had grown identical copies. Two details carry
the weight:

* **Trailing punctuation is trimmed off all but the last segment.** A model
  handed a 4 s slice ends it with a full stop because that slice ended, not
  because the sentence did — keeping those turns one sentence into four.
* **CJK joins with no separator.** Inserting spaces between Chinese segments is
  visible in the transcript and, downstream, changes CER.
"""

from __future__ import annotations

from typing import Optional, Sequence

_TRAIL_PUNCT = "。，、！？；,.!?;"
_CJK_LANGS = {"Chinese", "Japanese", "Korean", "Cantonese", "zh", "ja", "ko"}
_CJK_PREFIXES = ("zh", "ja", "ko")


def join_segments(texts: Sequence[str], language: Optional[str]) -> str:
    parts = [t.strip() for t in texts if t and t.strip()]
    if not parts:
        return ""
    if len(parts) > 1:
        parts = [t.rstrip(_TRAIL_PUNCT).rstrip() for t in parts[:-1]] + [parts[-1]]
    lang = language or ""
    is_cjk = lang in _CJK_LANGS or any(lang.startswith(p) for p in _CJK_PREFIXES)
    return ("" if is_cjk else " ").join(parts).strip()
