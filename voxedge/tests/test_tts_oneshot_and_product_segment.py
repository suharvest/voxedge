"""Non-worker TTS synth + Qwen3 product-segmentation orchestration (gap).

After the env-free migration these stayed real but lost coverage (the old product
tests patched module constants ``TTS_BINARY`` / ``PLUGIN_PATH`` + env, which no
longer exist — they are ``config.tts_binary`` / ``config.plugin_path`` now):

  * ``_synthesize_single`` one-shot binary path passes ``--codePredictorEngineDir``
    + the talker sampling params + ``min_audio_length`` to the TTS binary;
  * ``synthesize`` segments long CJK text and concatenates the per-segment WAVs
    with ``segment_pauses_ms`` inserted;
  * Qwen3 ``synthesize`` with ``product_segment_text`` splits CJK at clause
    boundaries, synthesizes each segment with the same fixed seed, and reports
    ``product_segmented`` / ``segment_pauses_ms``.

Mac-safe: backends via ``__new__`` + injected config; the binary is a fake
``run_binary`` that writes a WAV; the engine/tokenizer are fakes.
"""

from __future__ import annotations

import io
import json
import subprocess
import wave

import voxedge.backends.jetson.trt_edge_llm_tts as tts_mod
from voxedge.backends.jetson.trt_edge_llm_tts import (
    TRTEdgeLLMTTSBackend,
    TRTEdgeLLMTTSConfig,
)


def _make_wav_bytes(frame_count: int, sample_rate: int = 24000) -> bytes:
    payload = b"\x00\x00" * frame_count
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(payload)
    return out.getvalue()


def _make_oneshot_backend(**config_kwargs):
    """Non-worker backend wired for the one-shot binary synth path."""
    cfg = TRTEdgeLLMTTSConfig(
        use_worker=False,
        tts_binary="/tmp/qwen3_tts_inference",
        plugin_path="/tmp/plugin.so",
        segment_text=True,
        **config_kwargs,
    )
    backend = TRTEdgeLLMTTSBackend.__new__(TRTEdgeLLMTTSBackend)
    backend._config = cfg
    backend._product_backend = None
    backend._ready = True
    backend._talker_dir = "/models/talker"
    backend._code_predictor_dir = "/models/code_predictor"
    backend._tokenizer_dir = "/models/tokenizer"
    backend._code2wav_dir = "/models/code2wav"
    # Explicit-KV flag inputs (default empty → legacy generic-runner path).
    backend._talker_backend = ""
    backend._talker_engine = ""
    backend._code_predictor_backend = ""
    backend._text_projection = ""
    backend._prompt_kv_cache = ""
    return backend


def test_one_shot_passes_code_predictor_and_sampling(monkeypatch):
    backend = _make_oneshot_backend()
    captured = {}

    def fake_run_binary(binary, args, timeout, plugin_path=None):
        captured["binary"] = binary
        captured["args"] = args
        input_path = args[args.index("--inputFile") + 1]
        with open(input_path) as f:
            captured["input"] = json.load(f)
        output_path = args[args.index("--outputFile") + 1]
        audio_dir = args[args.index("--outputAudioDir") + 1]
        audio_path = f"{audio_dir}/audio_req0.wav"
        with open(audio_path, "wb") as f:
            f.write(b"RIFFtest")
        with open(output_path, "w") as f:
            json.dump(
                {"responses": [{"audio_file": audio_path, "audio_duration_ms": 10, "audio_samples": 240}]},
                f,
            )
        return subprocess.CompletedProcess([binary] + args, 0, "", "")

    monkeypatch.setattr(tts_mod, "run_binary", fake_run_binary)
    monkeypatch.setattr(tts_mod, "_code2wav_engine_path", lambda d: "/nonexistent")

    wav, _ = backend.synthesize("你好", max_audio_length=8, segment_text=False)

    assert wav == b"RIFFtest"
    assert captured["binary"] == "/tmp/qwen3_tts_inference"
    cp_idx = captured["args"].index("--codePredictorEngineDir") + 1
    assert captured["args"][cp_idx] == "/models/code_predictor"
    req = captured["input"]["requests"][0]
    inp = captured["input"]
    assert inp["talker_top_k"] == 50
    assert inp["talker_top_p"] == 1.0
    assert inp["predictor_top_k"] == 50
    assert inp["min_audio_length"] == 30
    assert req["messages"][0]["content"] == "你好"


def test_segmented_concatenates_one_shot_wavs(monkeypatch):
    backend = _make_oneshot_backend()
    calls = []

    def fake_run_binary(binary, args, timeout, plugin_path=None):
        input_path = args[args.index("--inputFile") + 1]
        with open(input_path) as f:
            input_data = json.load(f)
        calls.append(input_data)
        output_path = args[args.index("--outputFile") + 1]
        audio_dir = args[args.index("--outputAudioDir") + 1]
        audio_path = f"{audio_dir}/audio_req0.wav"
        with open(audio_path, "wb") as f:
            f.write(_make_wav_bytes(240))
        with open(output_path, "w") as f:
            json.dump(
                {"responses": [{"audio_file": audio_path, "audio_duration_ms": 10, "audio_samples": 240}]},
                f,
            )
        return subprocess.CompletedProcess([binary] + args, 0, "", "")

    monkeypatch.setattr(tts_mod, "run_binary", fake_run_binary)
    monkeypatch.setattr(tts_mod, "_code2wav_engine_path", lambda d: "/nonexistent")

    text = "你好，很高兴认识你。今天我们来测试一下语音合成的稳定性，看看这段稍微长一点的中文是不是能清楚自然地读出来。"
    wav, meta = backend.synthesize(text, max_audio_length=64, segment_max_chars=24)

    assert len(calls) > 1
    assert meta["segmented"] is True
    assert meta["segment_count"] == len(calls)
    assert meta["samples"] > 240 * len(calls)  # pauses inserted between segments
    assert len(meta["segment_pauses_ms"]) == len(calls) - 1
    with wave.open(io.BytesIO(wav), "rb") as reader:
        assert reader.getframerate() == 24000
        assert reader.getnframes() == meta["samples"]


