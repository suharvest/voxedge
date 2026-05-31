"""Unit tests for the NLLB translator ABC + CTranslate2 backend.

Heavy runtime (ctranslate2 / sentencepiece) is mocked — these tests assert the
*contract*: env-free config, the ABC default methods, and especially the three
real-device bugs (memory ``nllb_translator_service_bugs``):
  1. ``EncodeAsPieces`` (not ``EncodeAsIds``)
  2. ``</s>`` + src_lang appended AFTER source pieces (not prefixed)
  3. ``device_index`` forced to ``int``
plus the CUDA guard (device="cuda" on a CPU-only CT2 build → preload raises).
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from voxedge.backends.base import (
    TranslationResult,
    TranslatorBackend,
    TranslatorCapability,
    TranslatorConfig,
)
from voxedge.backends.nllb_translator import NLLBTranslatorBackend
from voxedge.engine.concurrency_capability import ConcurrencyCapability


# ── fakes ────────────────────────────────────────────────────────────────


class _FakeSP:
    """Records EncodeAsPieces / DecodePieces calls."""

    def __init__(self):
        self.load_path = None
        self.encode_calls = []
        self.decode_calls = []

    def Load(self, path):  # noqa: N802 (mirror sentencepiece API)
        self.load_path = path

    def EncodeAsPieces(self, text):  # noqa: N802
        self.encode_calls.append(text)
        # deterministic fake pieces
        return [f"▁{text}", "piece2"]

    def DecodePieces(self, pieces):  # noqa: N802
        self.decode_calls.append(list(pieces))
        return " ".join(pieces)


class _FakeHyp:
    def __init__(self, tokens):
        self.hypotheses = [tokens]


class _FakeTranslator:
    """Records translate_batch args and returns a fixed hypothesis."""

    last_ctor_kwargs = None

    def __init__(self, *args, **kwargs):
        type(self).last_ctor_kwargs = {"args": args, "kwargs": kwargs}
        self.batch_calls = []

    def translate_batch(self, batch, **kwargs):
        self.batch_calls.append({"batch": batch, "kwargs": kwargs})
        # echo a tgt_lang token at position 0 (CT2 behaviour) + real tokens
        prefix = kwargs.get("target_prefix", [["xxx"]])
        return [_FakeHyp([prefix[i][0], "out", "tok"]) for i in range(len(batch))]


def _install_fake_modules(monkeypatch, *, cuda_count=1):
    """Inject fake ctranslate2 + sentencepiece modules into sys.modules."""
    ct2 = types.ModuleType("ctranslate2")
    ct2.Translator = _FakeTranslator
    ct2.get_cuda_device_count = lambda: cuda_count

    sp = types.ModuleType("sentencepiece")
    sp.SentencePieceProcessor = _FakeSP

    monkeypatch.setitem(sys.modules, "ctranslate2", ct2)
    monkeypatch.setitem(sys.modules, "sentencepiece", sp)
    return ct2, sp


# ── ABC default methods ────────────────────────────────────────────────────


class _BareTranslator(TranslatorBackend):
    """Concrete backend that overrides only the abstract surface."""

    @property
    def name(self):
        return "bare"

    @property
    def capabilities(self):
        return {TranslatorCapability.TEXT}

    def is_ready(self):
        return True

    def preload(self):
        pass

    def translate(self, text, src_lang=None, tgt_lang=None):
        return TranslationResult(text=text, src_lang="a", tgt_lang="b")


def test_has_capability_default():
    b = _BareTranslator()
    assert b.has_capability(TranslatorCapability.TEXT) is True
    assert b.has_capability(TranslatorCapability.BATCH) is False


def test_unload_default_is_noop():
    # Should not raise; default no-op.
    _BareTranslator().unload()


def test_translate_batch_default_raises_without_capability():
    with pytest.raises(NotImplementedError):
        _BareTranslator().translate_batch(["x"])


def test_concurrency_capability_default():
    cap = _BareTranslator.__new__(_BareTranslator).concurrency_capability()
    assert isinstance(cap, ConcurrencyCapability)
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False


def test_capability_enum_values():
    assert TranslatorCapability.TEXT.value == "text"
    assert TranslatorCapability.MULTI_LANGUAGE.value == "multi_language"
    assert TranslatorCapability.BATCH.value == "batch"
    assert TranslatorCapability.STREAMING.value == "streaming"


# ── config dataclass ───────────────────────────────────────────────────────


def test_config_defaults():
    cfg = TranslatorConfig(model_path="/m")
    assert cfg.model_path == "/m"
    assert cfg.src_lang == "zho_Hans"
    assert cfg.tgt_lang == "eng_Latn"
    assert cfg.device == "cuda"
    assert cfg.device_index == 0
    assert cfg.compute_type == "default"
    assert cfg.beam_size == 1
    assert cfg.max_batch_size == 1


def test_config_overrides():
    cfg = TranslatorConfig(
        model_path="/m",
        src_lang="eng_Latn",
        tgt_lang="fra_Latn",
        device="cpu",
        device_index=2,
        compute_type="int8",
        beam_size=4,
        max_batch_size=8,
    )
    assert (cfg.src_lang, cfg.tgt_lang, cfg.device) == ("eng_Latn", "fra_Latn", "cpu")
    assert cfg.device_index == 2
    assert cfg.beam_size == 4


# ── NLLB backend: identity / lazy load ─────────────────────────────────────


def test_init_does_not_load(monkeypatch):
    _install_fake_modules(monkeypatch)
    b = NLLBTranslatorBackend(TranslatorConfig(model_path="/m", device="cpu"))
    assert b.is_ready() is False
    assert b.name == "nllb_translator"
    assert TranslatorCapability.BATCH in b.capabilities
    assert TranslatorCapability.MULTI_LANGUAGE in b.capabilities


def test_preload_loads_tokenizer_and_translator(monkeypatch):
    _install_fake_modules(monkeypatch, cuda_count=1)
    b = NLLBTranslatorBackend(
        TranslatorConfig(model_path="/models/nllb", device="cuda")
    )
    b.preload()
    assert b.is_ready() is True
    # tokenizer loaded the bpe model from model_path
    assert b._tokenizer.load_path == "/models/nllb/sentencepiece.bpe.model"


def test_preload_empty_model_path_raises(monkeypatch):
    _install_fake_modules(monkeypatch)
    b = NLLBTranslatorBackend()  # default config has empty model_path
    with pytest.raises(ValueError):
        b.preload()


# ── BUG 3: device_index forced to int ──────────────────────────────────────


def test_preload_forces_device_index_int(monkeypatch):
    _install_fake_modules(monkeypatch, cuda_count=1)
    # pass device_index as a str to prove it gets int()-coerced
    cfg = TranslatorConfig(model_path="/m", device="cuda")
    cfg.device_index = "3"  # type: ignore[assignment]
    b = NLLBTranslatorBackend(cfg)
    b.preload()
    kwargs = _FakeTranslator.last_ctor_kwargs["kwargs"]
    assert kwargs["device_index"] == 3
    assert isinstance(kwargs["device_index"], int)
    assert kwargs["device"] == "cuda"
    assert kwargs["compute_type"] == "default"


# ── CUDA guard ─────────────────────────────────────────────────────────────


def test_cuda_guard_raises_on_cpu_only_build(monkeypatch):
    _install_fake_modules(monkeypatch, cuda_count=0)
    b = NLLBTranslatorBackend(TranslatorConfig(model_path="/m", device="cuda"))
    with pytest.raises(RuntimeError, match="no CUDA devices"):
        b.preload()


def test_cuda_guard_not_triggered_for_cpu_device(monkeypatch):
    _install_fake_modules(monkeypatch, cuda_count=0)
    b = NLLBTranslatorBackend(TranslatorConfig(model_path="/m", device="cpu"))
    b.preload()  # must not raise — device is cpu
    assert b.is_ready()


# ── BUGS 1 & 2: tokenization order ─────────────────────────────────────────


def test_translate_token_handling(monkeypatch):
    _install_fake_modules(monkeypatch, cuda_count=1)
    b = NLLBTranslatorBackend(
        TranslatorConfig(
            model_path="/m", device="cuda", src_lang="zho_Hans", tgt_lang="eng_Latn"
        )
    )
    b.preload()
    res = b.translate("你好")

    # BUG 1: EncodeAsPieces was used (recorded on the fake)
    assert b._tokenizer.encode_calls == ["你好"]

    # BUG 2: </s> + src_lang APPENDED after source pieces, not prefixed
    call = b._translator.batch_calls[0]
    sent_tokens = call["batch"][0]
    assert sent_tokens == ["▁你好", "piece2", "</s>", "zho_Hans"]
    assert sent_tokens[-2:] == ["</s>", "zho_Hans"]
    assert sent_tokens[0] != "zho_Hans"  # NOT prefixed

    # target_prefix is the tgt_lang
    assert call["kwargs"]["target_prefix"] == [["eng_Latn"]]
    assert call["kwargs"]["beam_size"] == 1

    # tgt_lang token at position 0 of hypothesis is stripped before decode
    decoded = b._tokenizer.decode_calls[0]
    assert "eng_Latn" not in decoded
    assert decoded == ["out", "tok"]

    assert isinstance(res, TranslationResult)
    assert res.src_lang == "zho_Hans"
    assert res.tgt_lang == "eng_Latn"
    assert res.text == "out tok"


def test_translate_lang_override_falls_back_to_config(monkeypatch):
    _install_fake_modules(monkeypatch, cuda_count=1)
    b = NLLBTranslatorBackend(
        TranslatorConfig(model_path="/m", device="cpu", src_lang="A", tgt_lang="B")
    )
    b.preload()
    # explicit override beats config
    res = b.translate("x", src_lang="fra_Latn", tgt_lang="deu_Latn")
    sent_tokens = b._translator.batch_calls[0]["batch"][0]
    assert sent_tokens[-2:] == ["</s>", "fra_Latn"]
    assert b._translator.batch_calls[0]["kwargs"]["target_prefix"] == [["deu_Latn"]]
    assert (res.src_lang, res.tgt_lang) == ("fra_Latn", "deu_Latn")

    # no override → config defaults
    b.translate("y")
    sent_tokens2 = b._translator.batch_calls[1]["batch"][0]
    assert sent_tokens2[-2:] == ["</s>", "A"]
    assert b._translator.batch_calls[1]["kwargs"]["target_prefix"] == [["B"]]


def test_translate_before_preload_raises(monkeypatch):
    _install_fake_modules(monkeypatch)
    b = NLLBTranslatorBackend(TranslatorConfig(model_path="/m"))
    with pytest.raises(RuntimeError, match="before preload"):
        b.translate("x")


def test_translate_batch_token_handling(monkeypatch):
    _install_fake_modules(monkeypatch, cuda_count=1)
    b = NLLBTranslatorBackend(
        TranslatorConfig(
            model_path="/m", device="cpu", src_lang="zho_Hans", tgt_lang="eng_Latn"
        )
    )
    b.preload()
    out = b.translate_batch(["a", "b"])
    assert len(out) == 2
    call = b._translator.batch_calls[0]
    # both rows get </s>+src appended
    for row in call["batch"]:
        assert row[-2:] == ["</s>", "zho_Hans"]
    assert call["kwargs"]["target_prefix"] == [["eng_Latn"], ["eng_Latn"]]
    assert all(r.text == "out tok" for r in out)


def test_unload_clears_state(monkeypatch):
    _install_fake_modules(monkeypatch, cuda_count=1)
    b = NLLBTranslatorBackend(TranslatorConfig(model_path="/m", device="cpu"))
    b.preload()
    assert b.is_ready()
    b.unload()
    assert b.is_ready() is False


def test_preload_missing_deps_raises_friendly(monkeypatch):
    # Simulate the translator extra not being installed.
    monkeypatch.setitem(sys.modules, "ctranslate2", None)
    monkeypatch.setitem(sys.modules, "sentencepiece", None)
    with mock.patch(
        "voxedge.backends._deps.importlib.import_module",
        side_effect=ImportError("nope"),
    ):
        b = NLLBTranslatorBackend(TranslatorConfig(model_path="/m"))
        with pytest.raises(ImportError, match=r"voxedge\[translator\]"):
            b.preload()
