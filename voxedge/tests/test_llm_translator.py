"""Unit tests for the LLM-backed translator backend.

The wrapped :class:`LLMBackend` is faked: its ``stream_events`` yields a fixed
sequence of text deltas (plus noise events we must ignore). We assert the
*contract*: prompt construction (system + user carrying src/tgt/text), text
concatenation + cleanup, src/tgt defaulting/override, async↔sync bridge (both
from outside a loop and from inside a running loop), and the delegated
``is_ready`` / ``preload`` / ``capabilities`` surface.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from voxedge.backends.base import (
    LLMBackend,
    LLMEvent,
    TranslationResult,
    TranslatorCapability,
    TranslatorConfig,
)
from voxedge.backends.llm_translator import LLMTranslatorBackend


# ── fake LLM ───────────────────────────────────────────────────────────────


class _FakeLLM(LLMBackend):
    """Records the messages it was given; replays scripted events."""

    def __init__(self, text_deltas, *, ready=True, has_ready=True):
        self._deltas = text_deltas
        self._ready = ready
        self._has_ready = has_ready
        self.calls: list[list[dict[str, Any]]] = []
        self.preloaded = False

    async def stream(self, messages, **kw):  # pragma: no cover — unused path
        for d in self._deltas:
            yield d

    async def stream_events(self, messages, **kw):
        self.calls.append(messages)
        # interleave noise events the translator must ignore
        yield LLMEvent(kind="tool_call_delta", name="noise", arguments="{}")
        for d in self._deltas:
            yield LLMEvent(kind="text", text=d)
        yield LLMEvent(kind="text", text=None)  # empty delta — skipped
        yield LLMEvent(kind="finish", finish_reason="stop")

    # optional surface the wrapper probes for via getattr
    def is_ready(self):
        if not self._has_ready:
            raise AttributeError  # pragma: no cover
        return self._ready

    def preload(self):
        self.preloaded = True


# ── identity / capabilities ─────────────────────────────────────────────────


def test_name_and_capabilities():
    b = LLMTranslatorBackend(_FakeLLM(["x"]))
    assert b.name == "llm_translator"
    caps = b.capabilities
    assert TranslatorCapability.TEXT in caps
    assert TranslatorCapability.MULTI_LANGUAGE in caps
    # deliberately NOT advertised
    assert TranslatorCapability.BATCH not in caps
    assert TranslatorCapability.STREAMING not in caps


def test_has_capability_delegates_to_set():
    b = LLMTranslatorBackend(_FakeLLM(["x"]))
    assert b.has_capability(TranslatorCapability.TEXT) is True
    assert b.has_capability(TranslatorCapability.BATCH) is False


def test_supports_hot_reload_false():
    assert LLMTranslatorBackend.supports_hot_reload is False


# ── is_ready / preload delegation ───────────────────────────────────────────


def test_is_ready_delegates_true():
    b = LLMTranslatorBackend(_FakeLLM(["x"], ready=True))
    assert b.is_ready() is True


def test_is_ready_delegates_false():
    b = LLMTranslatorBackend(_FakeLLM(["x"], ready=False))
    assert b.is_ready() is False


def test_is_ready_without_probe_method():
    class _NoReady(LLMBackend):
        async def stream(self, messages, **kw):  # pragma: no cover
            yield ""

    b = LLMTranslatorBackend(_NoReady())
    # no is_ready on the LLM → considered ready since llm is non-None
    assert b.is_ready() is True


def test_preload_delegates():
    llm = _FakeLLM(["x"])
    b = LLMTranslatorBackend(llm)
    b.preload()
    assert llm.preloaded is True


def test_preload_noop_when_llm_has_no_preload():
    class _NoPreload(LLMBackend):
        async def stream(self, messages, **kw):  # pragma: no cover
            yield ""

    # must not raise
    LLMTranslatorBackend(_NoPreload()).preload()


def test_unload_default_is_noop():
    LLMTranslatorBackend(_FakeLLM(["x"])).unload()


# ── prompt construction ─────────────────────────────────────────────────────


def test_prompt_construction_system_and_user():
    llm = _FakeLLM(["译文"])
    b = LLMTranslatorBackend(
        llm,
        TranslatorConfig(model_path="", src_lang="zho_Hans", tgt_lang="eng_Latn"),
    )
    b.translate("你好世界")
    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "你是翻译引擎，只输出译文，不解释不寒暄"
    assert messages[1]["role"] == "user"
    user = messages[1]["content"]
    # user message carries src, tgt and the source text
    assert "zho_Hans" in user
    assert "eng_Latn" in user
    assert "你好世界" in user


# ── translate: concatenation + cleanup ──────────────────────────────────────


def test_translate_concatenates_text_deltas():
    llm = _FakeLLM(["Hello", ", ", "world"])
    b = LLMTranslatorBackend(
        llm, TranslatorConfig(model_path="", src_lang="zho_Hans", tgt_lang="eng_Latn")
    )
    res = b.translate("你好，世界")
    assert isinstance(res, TranslationResult)
    assert res.text == "Hello, world"
    assert res.src_lang == "zho_Hans"
    assert res.tgt_lang == "eng_Latn"


def test_translate_strips_wrapping_quotes_and_whitespace():
    llm = _FakeLLM(["  “Hello world”\n"])
    b = LLMTranslatorBackend(llm)
    res = b.translate("x")
    assert res.text == "Hello world"


def test_translate_strips_ascii_quotes():
    llm = _FakeLLM(['"Bonjour"'])
    res = LLMTranslatorBackend(llm).translate("x")
    assert res.text == "Bonjour"


def test_translate_ignores_non_text_events():
    # only the text deltas (and not tool_call/finish/None) make it into output
    llm = _FakeLLM(["A", "B"])
    res = LLMTranslatorBackend(llm).translate("x")
    assert res.text == "AB"


# ── src/tgt defaulting + override ───────────────────────────────────────────


def test_translate_uses_config_defaults():
    llm = _FakeLLM(["out"])
    b = LLMTranslatorBackend(
        llm, TranslatorConfig(model_path="", src_lang="A", tgt_lang="B")
    )
    res = b.translate("x")
    assert (res.src_lang, res.tgt_lang) == ("A", "B")
    user = llm.calls[0][1]["content"]
    assert "A" in user and "B" in user


def test_translate_explicit_override_beats_config():
    llm = _FakeLLM(["out"])
    b = LLMTranslatorBackend(
        llm, TranslatorConfig(model_path="", src_lang="A", tgt_lang="B")
    )
    res = b.translate("x", src_lang="fra_Latn", tgt_lang="deu_Latn")
    assert (res.src_lang, res.tgt_lang) == ("fra_Latn", "deu_Latn")
    user = llm.calls[0][1]["content"]
    assert "fra_Latn" in user and "deu_Latn" in user


def test_translate_default_config_when_none_passed():
    # no config → TranslatorConfig defaults (zho_Hans → eng_Latn)
    llm = _FakeLLM(["out"])
    res = LLMTranslatorBackend(llm).translate("x")
    assert res.src_lang == "zho_Hans"
    assert res.tgt_lang == "eng_Latn"


# ── async variant ───────────────────────────────────────────────────────────


def test_atranslate_async_variant():
    llm = _FakeLLM(["Hola"])
    b = LLMTranslatorBackend(
        llm, TranslatorConfig(model_path="", src_lang="A", tgt_lang="C")
    )

    async def _run():
        return await b.atranslate("x")

    res = asyncio.run(_run())
    assert res.text == "Hola"
    assert (res.src_lang, res.tgt_lang) == ("A", "C")


# ── async↔sync bridge: calling sync translate from INSIDE a running loop ─────


def test_translate_sync_from_inside_running_loop():
    """The loop-aware bridge must not deadlock / raise when translate() is
    called from within a running event loop (offloads to a worker thread)."""
    llm = _FakeLLM(["bridged"])
    b = LLMTranslatorBackend(llm)

    async def _outer():
        # call the SYNC translate() while a loop is running on this thread
        return await asyncio.to_thread(b.translate, "x")

    res = asyncio.run(_outer())
    assert res.text == "bridged"


def test_translate_sync_directly_in_coroutine_thread():
    """Even calling translate() directly (not via to_thread) from a coroutine
    must work via the worker-thread fallback."""
    llm = _FakeLLM(["ok"])
    b = LLMTranslatorBackend(llm)

    async def _outer():
        # directly invoke sync translate on the loop thread
        return b.translate("x")

    res = asyncio.run(_outer())
    assert res.text == "ok"


# ── bridge error propagation ─────────────────────────────────────────────────


def test_translate_propagates_llm_error():
    class _BoomLLM(LLMBackend):
        async def stream(self, messages, **kw):  # pragma: no cover
            yield ""

        async def stream_events(self, messages, **kw):
            raise ValueError("llm exploded")
            yield  # pragma: no cover — make it a generator

    b = LLMTranslatorBackend(_BoomLLM())
    with pytest.raises(ValueError, match="llm exploded"):
        b.translate("x")


def test_translate_propagates_llm_error_from_running_loop():
    class _BoomLLM(LLMBackend):
        async def stream(self, messages, **kw):  # pragma: no cover
            yield ""

        async def stream_events(self, messages, **kw):
            raise ValueError("boom inside loop")
            yield  # pragma: no cover

    b = LLMTranslatorBackend(_BoomLLM())

    async def _outer():
        return b.translate("x")

    with pytest.raises(ValueError, match="boom inside loop"):
        asyncio.run(_outer())
