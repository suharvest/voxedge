"""TRT-Edge-LLM TTS ``supports_voice_enrollment`` signal (config → capability).

Honest device-side enrollment signal: the CPU-ONNX speaker encoder turns a
reference WAV into a float32[1024] embedding *without torch*, so the Qwen3 BASE
backend can self-enroll on a torch-less Jetson TRT deployment. The signal must
be True only when the encoder ONNX actually exists on disk (mirrors the guard in
``extract_speaker_embedding``), so the server /tts/capabilities view is honest.

Mac-safe: ``__init__`` only records config paths (no model load, no CUDA).
"""

from __future__ import annotations

from voxedge.backends.base import TTSBackend
from voxedge.backends.jetson.trt_edge_llm_tts import (
    TRTEdgeLLMTTSConfig,
    TRTEdgeLLMTTSBackend,
)


def test_base_default_is_false():
    # Base ABC declares the attribute so getattr on any backend is safe.
    assert TTSBackend.supports_voice_enrollment is False


def test_no_encoder_configured_is_false():
    be = TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(speaker_encoder=""))
    assert be.supports_voice_enrollment is False


def test_encoder_path_missing_is_false(tmp_path):
    missing = str(tmp_path / "does_not_exist.onnx")
    be = TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(speaker_encoder=missing))
    assert be.supports_voice_enrollment is False


def test_encoder_present_is_true(tmp_path):
    enc = tmp_path / "speaker_encoder.onnx"
    enc.write_bytes(b"\x00")  # existence is all the property checks
    be = TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(speaker_encoder=str(enc)))
    assert be.supports_voice_enrollment is True


def test_base_voice_clone_capability_unchanged():
    """The embedding-conditioned Base variant remains clone-capable."""
    from voxedge.backends.base import TTSCapability
    be = TRTEdgeLLMTTSBackend(
        TRTEdgeLLMTTSConfig(model_id="qwen3-tts", speaker_encoder="")
    )
    assert be.supports_voice_cloning is True
    assert TTSCapability.VOICE_CLONE in be.capabilities


def test_customvoice_does_not_advertise_clone_or_enrollment(tmp_path):
    """CustomVoice has preset speakers but cannot consume clone embeddings."""
    from voxedge.backends.base import TTSCapability

    enc = tmp_path / "speaker_encoder.onnx"
    enc.write_bytes(b"\x00")
    be = TRTEdgeLLMTTSBackend(
        TRTEdgeLLMTTSConfig(
            model_id="qwen3-tts-customvoice",
            speaker_encoder=str(enc),
        )
    )

    assert be.supports_voice_cloning is False
    assert be.supports_voice_enrollment is False
    assert TTSCapability.VOICE_CLONE not in be.capabilities


def test_customvoice_clone_fails_before_worker_dispatch(monkeypatch):
    be = TRTEdgeLLMTTSBackend(
        TRTEdgeLLMTTSConfig(model_id="qwen3-tts-customvoice")
    )
    dispatched = False

    def _unexpected_dispatch(*args, **kwargs):
        nonlocal dispatched
        dispatched = True
        raise AssertionError("worker dispatch must not run")

    monkeypatch.setattr(be, "_synthesize_impl", _unexpected_dispatch)

    import pytest

    with pytest.raises(NotImplementedError, match="built-in speakers"):
        be.clone_voice("hello", b"\x00\x00\x00\x00")
    assert dispatched is False


def test_base_embedding_default_allows_explicit_none_but_preserves_voice_id(monkeypatch):
    """An explicit None means no per-request speaker; id 0 remains an override."""
    import voxedge.backends.jetson.trt_edge_llm_tts as tts_mod

    embedding = b"\x00\x01" * 16
    be = TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(model_id="qwen3-tts"))
    be._base_speaker_embedding = embedding

    def fake_resolve(_model_id, *, speaker_id=None, **kwargs):
        return {"speaker_id": speaker_id} if speaker_id is not None else {}

    monkeypatch.setattr(tts_mod, "resolve_speaker_kwargs", fake_resolve)
    assert be._resolve_voice_kwargs({})["speaker_embedding"] == embedding
    assert be._resolve_voice_kwargs({"speaker_id": None})["speaker_embedding"] == embedding
    zero = be._resolve_voice_kwargs({"speaker_id": 0})
    assert zero.get("speaker_id") == 0
    assert "speaker_embedding" not in zero


def test_speaker_resolver_preserves_embedding_and_id_priority():
    from voxedge.backends.jetson._trt_edge_llm_util import resolve_speaker_kwargs

    emb = b"\x00\x01\x02\x03"
    assert resolve_speaker_kwargs("qwen3", speaker_embedding=emb, speaker="serena") == {
        "speaker_embedding": emb
    }
    assert resolve_speaker_kwargs("qwen3", speaker_id=0, speaker="serena") == {
        "speaker_id": 0
    }
    assert resolve_speaker_kwargs("qwen3", sid=7, speaker="serena") == {
        "speaker_id": 7
    }
    assert resolve_speaker_kwargs("qwen3", speaker="serena") == {}
    assert resolve_speaker_kwargs("qwen3", speaker=None) == {}


def test_customvoice_public_api_resolves_named_speaker(monkeypatch):
    """Both public APIs retain named-speaker kwargs through the real resolver."""
    from voxedge.backends.jetson import trt_edge_llm_tts as tts_mod

    be = TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(model_id="qwen3-tts-customvoice"))
    calls = []

    def fake_impl(text, **kwargs):
        calls.append(kwargs)
        return b"wav", {}

    monkeypatch.setattr(be, "_synthesize_impl", fake_impl)
    monkeypatch.setattr(be, "_generate_streaming_impl", fake_impl)
    be.synthesize("hello", language="english", speaker="serena", seed=42)
    list(be.generate_streaming("hello", language="english", speaker="serena", seed=42))
    assert len(calls) == 2
    assert all(x["speaker"] == "serena" and x["seed"] == 42 for x in calls)


def test_customvoice_rejects_embedding_before_worker_dispatch():
    be = TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(model_id="qwen3-tts-customvoice"))
    import pytest
    with pytest.raises(ValueError, match="does not support voice clone embeddings"):
        be._resolve_voice_kwargs({"speaker_embedding": b"\x00\x01\x02\x03"})
