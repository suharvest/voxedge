"""Non-streaming synth on a streaming-native worker (v0.9.0 lean code2wav).

The v0.9.0 lean code2wav worker emits audio only via
streamingChunkFrames/onChunkReady and has no output_file (write-whole-WAV)
mode. The non-streaming ``_synthesize_worker`` path used to request an
output_file, which the v090 worker rejects with KeyError('output_file'). When
``streaming_only_worker`` is set, the non-streaming path must aggregate the
streaming chunks into a WAV instead (``_synthesize_worker_via_stream``).
"""
from __future__ import annotations

from voxedge.backends.jetson.trt_edge_llm_tts import (
    TRTEdgeLLMTTSBackend,
    TRTEdgeLLMTTSConfig,
    build_config_from_env,
)


def test_build_config_reads_streaming_only_flag():
    assert build_config_from_env(
        {"EDGE_LLM_TTS_STREAMING_ONLY": "1"}
    ).streaming_only_worker is True
    # Default preserves the legacy v0.8.0 output_file path.
    assert build_config_from_env({}).streaming_only_worker is False
    assert build_config_from_env(
        {"EDGE_LLM_TTS_STREAMING_ONLY": "0"}
    ).streaming_only_worker is False


def test_streaming_only_routes_nonstreaming_synth_via_stream():
    # stateful disabled but streaming_only set → must take the aggregate-stream
    # path, never the output_file path.
    cfg = TRTEdgeLLMTTSConfig(stateful_code2wav=False, streaming_only_worker=True)
    be = TRTEdgeLLMTTSBackend(cfg)
    calls = {"via_stream": 0}

    def _fake_via_stream(text, language=None, **kwargs):
        calls["via_stream"] += 1
        return b"RIFFfake", {"sample_rate": 24000, "duration_s": 0.1}

    be._synthesize_worker_via_stream = _fake_via_stream  # type: ignore[assignment]
    wav, meta = be._synthesize_worker("hello", language="english")
    assert calls["via_stream"] == 1
    assert wav == b"RIFFfake"


def test_non_streaming_worker_still_uses_output_file_path():
    # Legacy v0.8.0 non-stateful worker (streaming_only False) keeps the
    # output_file protocol — routing must NOT divert to via_stream.
    cfg = TRTEdgeLLMTTSConfig(stateful_code2wav=False, streaming_only_worker=False)
    be = TRTEdgeLLMTTSBackend(cfg)
    called = {"via_stream": 0}
    be._synthesize_worker_via_stream = lambda *a, **k: called.__setitem__(  # type: ignore[assignment]
        "via_stream", called["via_stream"] + 1
    )
    # Force an early, controlled failure right after the routing decision so we
    # don't spawn a real worker: a missing worker IO raises, proving we passed
    # the routing branch WITHOUT calling via_stream.
    try:
        be._synthesize_worker("hello", language="english")
    except Exception:
        pass
    assert called["via_stream"] == 0
