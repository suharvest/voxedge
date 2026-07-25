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
