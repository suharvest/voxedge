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


def test_voice_clone_capability_unchanged(tmp_path):
    """Zero-regression: VOICE_CLONE is still advertised regardless of enroller."""
    from voxedge.backends.base import TTSCapability
    be = TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(speaker_encoder=""))
    assert TTSCapability.VOICE_CLONE in be.capabilities
