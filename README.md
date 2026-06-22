# voxedge

> **English** | [中文](README.zh-CN.md)

<p align="center">
  <img src="media/banner.png" alt="voxedge banner" width="100%">
</p>

[![PyPI](https://img.shields.io/pypi/v/voxedge)](https://pypi.org/project/voxedge)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/voxedge)](https://pypi.org/project/voxedge)

**Native TensorRT · RKNN · sherpa-onnx voice pipelines for Jetson, Rockchip, and Raspberry Pi — fully on-device, verified on real hardware, zero cloud.**

<!-- TODO: Add demo GIF — recommend a ~15s terminal recording showing ASR→TTS on Jetson Orin (place in media/demo.gif) -->

## What is voxedge?

voxedge is an embeddable Python library that drives real-time, on-device voice conversations by calling directly into each platform's native inference runtime — TensorRT on Jetson Orin, RKNN on RK3576/RK3588, sherpa-onnx on CPU. No cloud STT/TTS APIs, no internet at runtime, no intermediate abstraction overhead. The same `ConversationEngine` API works across all three backends; you swap only the backend constructor — N=2 concurrent sessions verified on Orin Nano 8 GB, byte-identical output, zero CUDA errors.

voxedge is the open-core engine behind **[OpenVoiceStream](https://github.com/suharvest/openvoicestream)** — the deployable FastAPI/WebSocket server, device profiles, and agent gallery. Want a container? Start there. Want to embed real-time edge voice in your own app? You're in the right place.

## Key Features

- **Native runtimes, full performance** — calls directly into TensorRT (Jetson), RKNN (Rockchip), and sherpa-onnx (CPU); no wrapper overhead, no cross-platform abstraction tax
- **Fully on-device** — no speech API key, no per-call bill, no internet dependency at runtime
- **Verified on real hardware** — N=2 concurrent sessions on Orin Nano 8 GB: byte-identical output vs. single-stream, zero CUDA errors
- **Streaming + barge-in** — partial + final ASR while the user speaks; sentence-level TTS streaming with first-audio latency low enough for live dialogue and cooperative barge-in
- **Swap hardware, not code** — same `ConversationEngine` API across Jetson, Rockchip, and sherpa-onnx CPU; only the backend constructor changes
- **Test on any machine** — mock backends require only numpy; the whole engine runs end-to-end on a Mac with no CUDA or GPU

## Quickstart

Runs on any machine — no GPU needed. Swap the backend constructors for a real device; the engine, transport, and event contract never change.

```bash
pip install voxedge
```

```python
import asyncio
from voxedge.engine import ConversationEngine
from voxedge.transport import InProcessTransport
from voxedge.backends.mock import MockASR, MockTTS, MockVAD

engine = ConversationEngine(
    backends={"asr": MockASR(transcript="hello world"), "tts": MockTTS(), "vad": MockVAD()},
    multi_utterance=True,
)

async def main():
    t = InProcessTransport()
    await t.feed_audio(b"\x01\x02" * 8000)   # speech frames (int16 PCM)
    await t.feed_audio(b"\x00\x00" * 8000)   # silence → VAD endpoints the utterance
    t.end_input()
    await engine.run(t)                       # drives ASR → (LLM) → TTS
    for ev in t.drain_events_nowait():        # asr_final / tts_* / ...
        print(ev["type"], ev.get("text", ""))

asyncio.run(main())
```

On a real device, swap **only the backend constructors** — everything else is identical:

```python
# Jetson Orin — pip install voxedge[jetson]
from voxedge.backends.jetson import (
    TRTEdgeLLMASRBackend, TRTEdgeLLMASRConfig,
    TRTEdgeLLMTTSBackend, TRTEdgeLLMTTSConfig,
)

engine = ConversationEngine(backends={
    "asr": TRTEdgeLLMASRBackend(TRTEdgeLLMASRConfig(...)),   # Qwen3-ASR, native TRT
    "tts": TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(...)),   # Qwen3-TTS, streaming
}, multi_utterance=True)
```

> `import voxedge` is **numpy-only** — TensorRT, RKNN, and sherpa-onnx are lazy-imported by their backend adapters and pulled in via extras. The example above imports cleanly on a Mac even though the TRT engine only runs on a Jetson.

## Install

```bash
pip install voxedge            # pure-Python core (numpy only)
pip install voxedge[sherpa]    # sherpa-onnx CPU ASR/TTS
pip install voxedge[jetson]    # Jetson TensorRT backends (aarch64)
pip install voxedge[rk]        # Rockchip RK3576/RK3588 NPU (aarch64)
pip install voxedge[llm]       # OpenAI-compatible LLM backend (httpx)
```

The `jetson` / `rk` extras declare only pure-Python deps; the CUDA/TensorRT and RKNN runtime wheels ship from the platform (JetPack L4T / Rockchip NPU userspace) or the engine repos — you bring the platform runtime.

## Architecture

Four layers, all importable without CUDA.

### Backends (`voxedge/backends/`)

Clean ABCs in `backends/base.py` — every constructor takes explicit params only, no env coupling:

- `ASRBackend` / `ASRStream` — streaming recognition
- `TTSBackend` — `synthesize()` (batch) + `generate_streaming()` (sentence-level chunks, cooperative cancel via `cancel_token` for barge-in)
- `VADBackend` / `VADSession` — voice-activity detection for speech / barge-in segmentation
- `LLMBackend` / `LLMEvent` — token-streaming LLM for the conversation loop

Concrete adapters live under `backends/{jetson,rk,sherpa}/` and import their heavy runtimes **lazily** (inside methods), so all modules import on any machine:

| Backend | Platform | Models | Extra |
|---------|----------|--------|-------|
| `backends/jetson/` | Jetson Orin (TensorRT) | Qwen3-ASR/TTS, Matcha, Kokoro, Paraformer, SenseVoice, MOSS-TTS-Nano | `voxedge[jetson]` aarch64 |
| `backends/rk/` | Rockchip RK3576/RK3588 (RKNN) | `rkvoice_stream` engine | `voxedge[rk]` aarch64 |
| `backends/sherpa/` | CPU (any arch) | Paraformer, Zipformer, SenseVoice, Matcha, Kokoro ONNX | `voxedge[sherpa]` |
| `backends/llm/` | Any | OpenAI-compatible LLM over httpx | `voxedge[llm]` |
| `backends/mock.py` | Dev / CI | MockASR, MockTTS, MockVAD, MockLLM | core |

### Transport (`voxedge/transport/`)

`Transport` ABC + two implementations:

- `InProcessTransport` — zero-IPC asyncio queues; default, used everywhere in tests
- `WebSocketTransport` — duck-typed ws adapter with no FastAPI dependency; idle-watchdog timeout injected by caller, reads no env

### Conversation Engine (`voxedge/engine/`)

`ConversationEngine` + per-connection `Session` coordinator, split into focused collaborators: `audio_dispatcher` (VAD → speech / barge-in), `asr_loop`, `client_events`, `tts_sequencer` / `tts_buffer`, `session_state`, and the LLM↔tool loop — `llm_turn` over the provider-agnostic `turn_driver.run_turn` pump, with `tool_registry` (`@tool` → JSON schema) and `coordinator` / `concurrency_capability` for multi-stream concurrency.

### Capabilities (`voxedge/capabilities/`)

Optional, default-off, stateless add-ons (punctuation, speaker embedding) via sherpa-onnx. Opt in explicitly; byte-level no-op when off.

## Design Constraints

- **Pure Python core** — `import voxedge` is numpy-only. Heavy adapters live under `backends/{jetson,rk,sherpa}/` with deferred runtime imports.
- **No env reads in the library** — all config injected as explicit params. Profiles and deployment knobs are the product's job ([OpenVoiceStream](https://github.com/suharvest/openvoicestream)).

## Status

In production — the open-core engine behind a shipped edge voice stack. ~270 mock-based tests; the whole engine runs end-to-end on a Mac with no CUDA.

## Contributing

Issues and PRs welcome. The mock backend suite runs on any machine with no hardware:

```bash
pip install voxedge
uv run pytest
```

## Acknowledgements

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — CPU ASR/TTS runtime
- [OpenVoiceStream](https://github.com/suharvest/openvoicestream) — the deployable server product built on this engine

## License

Apache-2.0. See [LICENSE](LICENSE).
