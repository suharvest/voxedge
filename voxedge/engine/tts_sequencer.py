"""TTS sentence channel + synth consumer loop.

Conversation split step 3 (see seeed-local-voice docs/plans/conversation-split.md):
this is the step-2 ``_TTSChannel`` facade moved out of ``conversation.py``,
now also owning the synth consumer loop (was ``Session._tts_out_task``) as
``run()``. It owns the TTS sentence queue + chunk buffer and is the single place
every TTS write happens — ``_event_loop``, ``_on_asr_final``,
``_llm_turn_with_tools`` and ``_bargein_tts`` go through its methods.

It still holds a back-ref to the owning ``Session`` (``self._sess``) for backend
/ transport / state / config access. Severing that into fully explicit deps is a
later cleanup; for now every method is a 1:1 behaviour port of the code it
replaced (``conversation.py`` line refs in the docstrings / comments are the
source of truth — there is no behaviour change vs step 2).
"""
from __future__ import annotations

import asyncio
import logging
import re
import struct
import threading
from typing import TYPE_CHECKING

from voxedge.engine.protocol import (
    SERVER_TTS_DONE,
    SERVER_TTS_SENTENCE_DONE,
    SERVER_TTS_STARTED,
    _is_pool_saturated,
)

if TYPE_CHECKING:  # pragma: no cover
    from voxedge.engine.conversation import Session

logger = logging.getLogger(__name__)


# ── markdown → speakable text ─────────────────────────────────────────
# An LLM answering a spoken turn still tends to reply in Markdown (bold,
# ATX headings, bullets, code spans). The sentence buffer splits on
# punctuation, so a reply like "### 1. **如果想表达的是：**" yields fragments
# that are pure markup — '**', '：**', '”**'. A TTS backend has nothing to
# say for those: matcha logs "No audio produced for text" and returns 0.1 s
# of silence per fragment, which both wastes NPU time and injects audible
# gaps. Stripping the markup is not cosmetic — it is what makes the text
# speakable at all.
#
# Deliberately conservative: only structural markers are removed, prose is
# never rewritten. A system prompt should also ask for plain speech, but
# prompt compliance is not guaranteed, so this is the backstop.
_MD_FENCE_RE = re.compile(r"```[^`]*```|`([^`]*)`")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_MD_QUOTE_RE = re.compile(r"(?m)^\s{0,3}>\s*")
_MD_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d{1,3}[.)])\s+")
_MD_EMPHASIS_RE = re.compile(r"(?<![A-Za-z0-9_*])(\*{1,3}|__)(?=\S)(.*?\S)(?<!\*)\1(?![A-Za-z0-9_*])")
_BARE_DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9_.@:/?-])"
    r"((?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,})"
    r"(?![A-Za-z0-9_/?&=#@:-])(?!(?:\.[A-Za-z0-9]))"
)
# Speakable = at least one letter, digit or CJK codepoint. Anything else is
# punctuation/markup only and must not reach the backend.
_SPEAKABLE_RE = re.compile(r"[0-9A-Za-z぀-ヿ一-鿿가-힯]")


def _normalize_cjk_domains(text: str, *, language: str | None = None) -> str:
    """Make bare domains legible in CJK speech without changing characters."""
    if language not in {"zh", "chinese"} or not re.search(r"[一-鿿]", text):
        return text

    def expand(match: re.Match[str]) -> str:
        domain = match.group(1)
        labels = domain.split(".")
        if len(domain) > 253 or any(len(label) > 63 for label in labels):
            return domain
        spoken: list[str] = []
        for index, label in enumerate(labels):
            if index:
                spoken.append(" 点 ")
            if index == len(labels) - 1 or (label.isascii() and label.isalpha() and len(label) <= 3):
                spoken.append(" ".join(label))
            else:
                spoken.append(label)
        return "".join(spoken)

    return _BARE_DOMAIN_RE.sub(expand, text)


# ── Arabic numerals → what the frontend should say ────────────────────
# The RK matcha frontend classifies text by script: CJK runs go through the
# pinyin lexicon, Latin words through espeak-ng, and everything else — which
# includes every digit — through espeak-ng as well. espeak speaks English, so a
# Chinese reply carrying "28 度" is synthesized as "twenty-eight 度" and a
# device id "01-00" as "zero one dash zero zero". Measured with the shipped
# voice (`espeak-ng -q --ipa`): 28 -> "twˈɛnti ˈeɪt", 01-00 ->
# "zˈiəɹəʊ wˈɒn dˈaʃ zˈiəɹəʊ zˈiəɹəʊ".
#
# The frontend has no language to consult, so the decision belongs here, in the
# one place that already turns a reply into speakable text. Gated on the same
# condition as the domain pass above: Chinese output only.
#
# Two readings, because a single rule gets one of them wrong. Reading an id as
# a cardinal turns "01-00" into "一杠零", worse than the English; reading a
# quantity digit by digit turns "28" into "二八".
#   quantities / dates / times / money / units   "28 度" -> "二十八 度"
#   identifiers                                  "01-00" -> "零一零零"
_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ("", "十", "百", "千")
_CN_GROUPS = ("", "万", "亿", "兆")

# Claimed in this order. Each pass replaces digits with Chinese characters, so
# no later pattern can re-match what an earlier one produced.
_NUM_TIME_RE = re.compile(r"(?<![\dA-Za-z:.])(\d{1,2}):(\d{2})(?![\dA-Za-z:.])")
_NUM_DATE_RE = re.compile(
    r"(?<![\dA-Za-z\-/])(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?![\dA-Za-z\-/])"
)
_NUM_YEAR_RE = re.compile(r"(?<![\dA-Za-z])(\d{4})(?=\s*年)")
_NUM_PERCENT_RE = re.compile(r"(?<![\dA-Za-z.])(\d+(?:\.\d+)?)\s*%(?![\dA-Za-z.])")
_NUM_DECIMAL_RE = re.compile(r"(?<![\dA-Za-z.])(\d+\.\d+)(?![\dA-Za-z.])")
# An identifier is dash/underscore/slash-joined digit groups, or a run with a
# leading zero. Both are things a cardinal reading would mangle.
_NUM_ID_RE = re.compile(r"(?<![\dA-Za-z])(\d+(?:[-_/]\d+)+|0\d+)(?![\dA-Za-z])")
_NUM_CARDINAL_RE = re.compile(r"(?<![\dA-Za-z.])(\d+)(?![\dA-Za-z.])")

# Digits that belong to a URL, an email or a bare domain are part of that
# token and must survive verbatim: "…/path?q=1" is not "…/path?q=一". Split
# them out the same way the Markdown stripper protects code spans.
_NUM_PROTECT_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s，。；！？、）】》]+"
    r"|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"
    r"|(?<![A-Za-z0-9_.@:/?-])"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}"
    r"[A-Za-z0-9_/?&=#@:.\-]*"
)


def _under_ten_thousand(value: int) -> str:
    """1..9999 -> 一千二百三十四, 1005 -> 一千零五."""
    out: list[str] = []
    pending_zero = False
    for position in (3, 2, 1, 0):
        digit = (value // 10 ** position) % 10
        if digit == 0:
            if out:
                pending_zero = True
            continue
        if pending_zero:
            out.append(_CN_DIGITS[0])
            pending_zero = False
        out.append(_CN_DIGITS[digit])
        if position:
            out.append(_CN_UNITS[position])
    return "".join(out)


def _int_to_chinese(value: int) -> str:
    """0..99999999 spoken the way it is said, not the way it is written.

    18 -> 十八 rather than 一十八; 1005 -> 一千零五; 10001 -> 一万零一.
    Beyond 亿 the run falls back to digit by digit, which is how such numbers
    are read aloud anyway.
    """
    if value < 0:
        return "负" + _int_to_chinese(-value)
    if value == 0:
        return _CN_DIGITS[0]

    parts: list[str] = []
    for index in (3, 2, 1, 0):
        chunk = (value // 10000 ** index) % 10000
        if not chunk:
            continue
        spoken = _under_ten_thousand(chunk)
        # A group that is neither the leading one nor four digits wide needs a
        # 零 between it and what came before: 10001 -> 一万零一.
        if parts and chunk < 1000:
            spoken = _CN_DIGITS[0] + spoken
        parts.append(spoken + _CN_GROUPS[index])

    out = "".join(parts)
    if 10 <= value < 20:
        out = out[1:]          # 一十八 -> 十八
    return out


def _spoken_digit_run(digits: str) -> str:
    """Digit by digit, separators dropped: "01-00" -> 零一零零."""
    return "".join(_CN_DIGITS[int(c)] for c in digits if c.isdigit())


def _spoken_decimal(number: str) -> str:
    whole, _, fraction = number.partition(".")
    return _int_to_chinese(int(whole)) + "点" + _spoken_digit_run(fraction)


def _rewrite_numbers(text: str) -> str:
    """The numeral passes, in the order that keeps them from overlapping."""
    text = _NUM_TIME_RE.sub(
        lambda m: _int_to_chinese(int(m.group(1)))
        + "点"
        # 14:05 reads 十四点零五 — a leading zero minute is digit by digit.
        + (_spoken_digit_run(m.group(2)) if m.group(2).startswith("0")
           else _int_to_chinese(int(m.group(2)))),
        text,
    )
    text = _NUM_DATE_RE.sub(
        lambda m: _spoken_digit_run(m.group(1))
        + "年" + _int_to_chinese(int(m.group(2)))
        + "月" + _int_to_chinese(int(m.group(3))) + "日",
        text,
    )
    text = _NUM_YEAR_RE.sub(lambda m: _spoken_digit_run(m.group(1)), text)
    text = _NUM_PERCENT_RE.sub(
        lambda m: "百分之"
        + (_spoken_decimal(m.group(1)) if "." in m.group(1)
           else _int_to_chinese(int(m.group(1)))),
        text,
    )
    text = _NUM_DECIMAL_RE.sub(lambda m: _spoken_decimal(m.group(1)), text)
    text = _NUM_ID_RE.sub(lambda m: _spoken_digit_run(m.group(1)), text)
    text = _NUM_CARDINAL_RE.sub(lambda m: _int_to_chinese(int(m.group(1))), text)

    # The dash in 中弘空调-01-00 joins a name to its id, so it reads as one run
    # rather than being spoken. The matcha frontend drops a bare dash on its
    # own, but the text handed to a backend should already say what is meant
    # instead of depending on one frontend's punctuation handling.
    return re.sub(
        r"(?<=[一-鿿])[-‐-―](?=[零一二三四五六七八九十])", "", text
    )


def _normalize_spoken_numbers(text: str, *, language: str | None = None) -> str:
    """Rewrite Arabic numerals for speech. Chinese output only."""
    if language not in {"zh", "chinese"} or not re.search(r"\d", text):
        return text

    # Rewrite between the protected spans and never inside them, so a URL or an
    # address survives the pass unchanged.
    out: list[str] = []
    cursor = 0
    for match in _NUM_PROTECT_RE.finditer(text):
        out.append(_rewrite_numbers(text[cursor:match.start()]))
        out.append(match.group(0))
        cursor = match.end()
    out.append(_rewrite_numbers(text[cursor:]))
    return "".join(out)


def _to_speakable(text: str, *, language: str | None = None) -> str:
    """Strip Markdown markup; return "" if nothing speakable remains."""
    if not text:
        return ""

    # Locate code/fence spans in the original text before touching emphasis.
    # Delimiters outside those spans may wrap a body that contains inline code;
    # delimiters inside code remain literal.
    code_spans = [(m.start(), m.end()) for m in _MD_FENCE_RE.finditer(text)]

    def in_code(pos: int) -> bool:
        return any(start <= pos < end for start, end in code_spans)

    def emphasis_sub(match: re.Match[str]) -> str:
        opening = match.start()
        closing = match.end() - len(match.group(1))
        if in_code(opening) or in_code(closing):
            return match.group(0)
        return match.group(2)

    s = _MD_EMPHASIS_RE.sub(emphasis_sub, text)

    def clean_markup(chunk: str) -> str:
        chunk = _MD_LINK_RE.sub(r"\1", chunk)
        chunk = _MD_HEADING_RE.sub("", chunk)
        chunk = _MD_QUOTE_RE.sub("", chunk)
        chunk = _MD_BULLET_RE.sub("", chunk)
        return chunk

    # Remove other structural markup outside code; inline code contributes its
    # literal body and fenced code remains non-speech, as before.
    chunks: list[tuple[str, bool]] = []
    cursor = 0
    for match in _MD_FENCE_RE.finditer(s):
        chunks.append((clean_markup(s[cursor:match.start()]), False))
        chunks.append((match.group(1) if match.group(1) is not None else " ", True))
        cursor = match.end()
    chunks.append((clean_markup(s[cursor:]), False))

    def speakable_chunk(chunk: str) -> str:
        return _normalize_spoken_numbers(
            _normalize_cjk_domains(chunk, language=language), language=language
        )

    s = "".join(
        speakable_chunk(chunk) if not protected else chunk
        for chunk, protected in chunks
    ).strip()
    return s if _SPEAKABLE_RE.search(s) else ""


class _TTSChannel:
    """Owns the TTS sentence queue + chunk buffer and the synth consumer loop.

    ``_event_loop``, ``_on_asr_final``, ``_llm_turn_with_tools`` and
    ``_bargein_tts`` previously poked ``self._tts_q`` / ``self._tts_buffer`` /
    ``state.tts_flush`` directly from four places; they now go through these
    methods, so the TTS write surface is one object. Holds a back-ref to the
    owning Session for backend/state access; every method is a 1:1 behaviour
    port of the code it replaced."""

    def __init__(self, sess: "Session"):
        self._sess = sess
        self.q: asyncio.Queue = asyncio.Queue()
        self.buffer = sess._make_tts_buffer() if sess._tts_be else None

    async def enqueue_text(self, chunk: str) -> None:
        """Add assistant / preamble / client text to the sentence buffer and
        push any completed sentences to the synth queue (was
        ``_enqueue_tts_text`` and the inline ``buffer.add`` loops)."""
        if not chunk or self.buffer is None:
            return
        for sentence in self.buffer.add(chunk):
            await self._put_speakable(sentence)

    async def flush_and_signal(self) -> None:
        """Flush the buffer remainder to the queue and signal end-of-input so
        the consumer drains then emits its terminal ``tts_done`` — the
        ``for s in buffer.flush(): q.put(s)`` + ``state.tts_flush = True`` pair
        that was repeated at six sites."""
        if self.buffer is not None:
            for sentence in self.buffer.flush():
                await self._put_speakable(sentence)
        self._sess.state.tts_flush = True

    async def _put_speakable(self, sentence: str) -> None:
        """Queue a sentence for synthesis, dropping markup-only fragments.

        Single choke point for both enqueue paths: nothing reaches the synth
        queue that the backend can only answer with silence.
        """
        speakable = _to_speakable(sentence)
        if not speakable:
            logger.debug("voxedge tts: dropped non-speakable fragment %r", sentence[:40])
            return
        await self.q.put(speakable)

    def interrupt_synth(self) -> None:
        """Barge-in step 1: stop the in-flight synth task + signal its stop
        event. Ordered BEFORE the LLM wind-down."""
        state = self._sess.state
        t = state.current_tts_task
        if t is not None and not t.done():
            t.cancel()
        stop = state.current_tts_stop
        if stop is not None:
            stop.set()

    def drain_and_reset(self) -> None:
        """Barge-in step 3: discard queued sentences + rebuild the buffer so
        nothing stale plays, and clear the flush signal. Ordered AFTER the LLM
        wind-down so the turn can't re-enqueue a sentence between drain
        iterations (#5)."""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._sess._tts_be is not None:
            self.buffer = self._sess._make_tts_buffer()
        self._sess.state.tts_flush = False

    async def run(self) -> None:
        """Synth consumer loop (was ``Session._tts_out_task``, app/main.py:
        3207-3432). Dequeues sentences, streams each through the TTS backend
        with per-chunk + per-sentence watchdogs, and emits the close-out
        ``tts_done``."""
        sess = self._sess
        state = sess.state
        multi = sess.engine.multi_utterance
        sr_header_sent = False
        while not state.client_closed:
            # Exit / per-turn done when flush + drained (app/main.py:3217-3233).
            if state.tts_flush and self.q.empty():
                if multi and not state.asr_session_closed:
                    state.tts_flush = False
                    if not state.client_closed:
                        await sess._send_event({
                            "type": SERVER_TTS_DONE,
                            "session_complete": False,
                        })
                    continue
                break
            try:
                sentence = await asyncio.wait_for(self.q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue

            audio_queue: asyncio.Queue = asyncio.Queue()
            stop_event = threading.Event()
            state.current_tts_stop = stop_event

            # Engine-parity #15: resolve speaker / voice / speed kwargs once
            # per sentence (same dict the legacy _run_synth builds,
            # app/main.py:3473-3478). Empty when nothing injected → default
            # speaker, unchanged behavior.
            stream_kwargs = sess._tts_stream_kwargs()

            def _run_synth(s: str, ev: threading.Event, aq: asyncio.Queue):
                # Mirrors app/main.py:3246-3281 _run_synth (thread body).
                try:
                    for chunk in sess._tts_be.generate_streaming(
                        s,
                        language=sess.engine.tts_language,
                        cancel_token=ev,
                        **stream_kwargs,
                    ):
                        if ev.is_set():
                            break
                        sess._loop.call_soon_threadsafe(aq.put_nowait, chunk)
                except Exception as e:  # noqa: BLE001
                    # M4: a slot-pool saturation is "backend busy", NOT a
                    # synth fault — surface a typed reject marker, not a
                    # generic tts error (app/main.py:3342-3356).
                    sat, max_slots = _is_pool_saturated(e)
                    if sat:
                        sess._loop.call_soon_threadsafe(
                            aq.put_nowait, ("__saturated__", max_slots)
                        )
                    else:
                        logger.exception("voxedge tts synth failed for %r", s[:80])
                        sess._loop.call_soon_threadsafe(
                            aq.put_nowait, ("__error__", str(e))
                        )
                finally:
                    sess._loop.call_soon_threadsafe(aq.put_nowait, None)

            chunk_timeout_s = sess._tts_chunk_timeout_s

            async def drain(s: str, ev: threading.Event, aq: asyncio.Queue):
                nonlocal sr_header_sent
                # Mirrors app/main.py:3283-3361 drain().
                if not sr_header_sent:
                    sr = sess._tts_be.sample_rate
                    await sess._send_audio(struct.pack("<I", sr))
                    sr_header_sent = True
                await sess._send_event({"type": SERVER_TTS_STARTED, "sentence": s})
                # Slot acquire: a TTS synth is the GPU-heavy op; in
                # serialized/exclusive mode it must not overlap an ASR finalize.
                # Held across the whole synth so the synth thread runs alone.
                async with sess._acquire("tts"):
                    sess._loop.run_in_executor(None, _run_synth, s, ev, aq)
                    state.tts_started = True
                    # M3: per-chunk watchdog — a wedged backend that produces no
                    # chunk within the budget aborts the sentence + emits an
                    # error so the client never waits forever (app/main.py:3322-3339).
                    while True:
                        try:
                            item = await asyncio.wait_for(aq.get(), timeout=chunk_timeout_s)
                        except asyncio.TimeoutError:
                            logger.warning(
                                "voxedge tts watchdog: no chunk within %.1fs for "
                                "sentence=%r — aborting synth", chunk_timeout_s, s[:80],
                            )
                            ev.set()
                            await sess._send_error(
                                f"tts: synth produced no chunks within "
                                f"{chunk_timeout_s:.0f}s"
                            )
                            break
                        if item is None:
                            break
                        if isinstance(item, tuple) and item[0] == "__saturated__":
                            # M4: typed pool_saturated; keep the session alive.
                            await sess._emit_pool_saturated(item[1])
                            break
                        if isinstance(item, tuple) and item[0] == "__error__":
                            await sess._send_error(f"tts: {item[1]}")
                            break
                        await sess._send_audio(item)
                await sess._send_event({"type": SERVER_TTS_SENTENCE_DONE, "sentence": s})

            task = asyncio.create_task(drain(sentence, stop_event, audio_queue))
            state.current_tts_task = task
            # M3: outer per-sentence wall-clock deadline. Covers wedges BEFORE
            # the first chunk watchdog can fire (e.g. a backend that hangs in
            # generate_streaming setup before yielding) (app/main.py:3382-3409).
            sentence_timeout_s = sess._tts_sentence_timeout_s
            try:
                await asyncio.wait_for(task, timeout=sentence_timeout_s)
            except asyncio.TimeoutError:
                logger.warning(
                    "voxedge tts: per-sentence deadline %.1fs exceeded for "
                    "sentence=%r — cancelling drain", sentence_timeout_s, sentence[:80],
                )
                stop_event.set()
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                if not state.client_closed:
                    try:
                        await sess._send_error(
                            f"tts: per-sentence deadline "
                            f"{sentence_timeout_s:.0f}s exceeded"
                        )
                    except Exception:
                        logger.exception("voxedge send_error after tts deadline failed")
            except asyncio.CancelledError:
                # Barge-in: stop synth + drain residual chunks (app/main.py:3410-3420).
                stop_event.set()
                try:
                    while True:
                        item = audio_queue.get_nowait()
                        if item is None:
                            break
                except asyncio.QueueEmpty:
                    pass
            finally:
                state.current_tts_task = None
                state.current_tts_stop = None

        # Session-final tts_done (app/main.py:3424-3432).
        if not state.client_closed:
            payload = {"type": SERVER_TTS_DONE}
            if multi:
                payload["session_complete"] = True
            await sess._send_event(payload)
