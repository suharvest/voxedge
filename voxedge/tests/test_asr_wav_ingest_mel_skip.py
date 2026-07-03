"""WAV-ingest mode (TensorRT-Edge-LLM v0.9.0+) skips the host-side mel-asset
requirement.

The v0.9.0 ASR worker writes the received PCM to a temp WAV and lets the
runtime's audio front-end extract mel internally, so the host-side
mel_settings/mel_filters assets are no longer needed. The Python preload guard
``_require_streaming_worker_assets`` must therefore skip the mel check when
``EDGELLM_REQUEST_AUDIO_WAV`` is truthy, while preserving the v0.8.0 contract
(mel required) when it is unset — the default.
"""
from __future__ import annotations

import os

import pytest

from voxedge.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    TRTEdgeLLMASRConfig,
    build_config_from_env,
)


def test_wav_mode_skips_mel_requirement():
    # stream_mode=worker + no mel assets, but WAV-ingest is on → no raise.
    cfg = TRTEdgeLLMASRConfig(
        use_worker=True,
        stream_mode="worker",
        mel_settings_path="",
        mel_filters_path="",
        request_audio_wav=True,
    )
    be = TRTEdgeLLMASRBackend(cfg)
    be._require_streaming_worker_assets()  # must not raise


def test_legacy_mode_still_requires_mel():
    # v0.8.0 contract preserved: worker mode without WAV-ingest and without mel
    # assets raises.
    cfg = TRTEdgeLLMASRConfig(
        use_worker=True,
        stream_mode="worker",
        mel_settings_path="",
        mel_filters_path="",
        request_audio_wav=False,
    )
    be = TRTEdgeLLMASRBackend(cfg)
    with pytest.raises(FileNotFoundError, match="mel"):
        be._require_streaming_worker_assets()


def test_build_config_reads_wav_flag():
    assert build_config_from_env({"EDGELLM_REQUEST_AUDIO_WAV": "1"}).request_audio_wav is True
    # Default (no key) preserves the legacy mel-required behavior.
    assert build_config_from_env({}).request_audio_wav is False
    assert build_config_from_env({"EDGELLM_REQUEST_AUDIO_WAV": "0"}).request_audio_wav is False


def test_worker_env_propagates_wav_flag():
    cfg = TRTEdgeLLMASRConfig(request_audio_wav=True, plugin_path="/tmp/plugin.so")
    be = TRTEdgeLLMASRBackend(cfg)
    env = be._worker_env()
    assert env["EDGELLM_REQUEST_AUDIO_WAV"] == "1"

    cfg2 = TRTEdgeLLMASRConfig(request_audio_wav=False, plugin_path="/tmp/plugin.so")
    env2 = TRTEdgeLLMASRBackend(cfg2)._worker_env()
    assert env2["EDGELLM_REQUEST_AUDIO_WAV"] == "0"


def test_worker_env_explicit_env_wins():
    # An explicit ambient value is not clobbered (setdefault semantics).
    cfg = TRTEdgeLLMASRConfig(request_audio_wav=False, plugin_path="/tmp/plugin.so")
    be = TRTEdgeLLMASRBackend(cfg)
    prev = os.environ.get("EDGELLM_REQUEST_AUDIO_WAV")
    os.environ["EDGELLM_REQUEST_AUDIO_WAV"] = "1"
    try:
        assert be._worker_env()["EDGELLM_REQUEST_AUDIO_WAV"] == "1"
    finally:
        if prev is None:
            os.environ.pop("EDGELLM_REQUEST_AUDIO_WAV", None)
        else:
            os.environ["EDGELLM_REQUEST_AUDIO_WAV"] = prev
