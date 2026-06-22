# voxedge

> **English** | [中文](README.zh-CN.md)

**Run real-time speech recognition and synthesis locally on edge devices — Jetson Orin, RK3576/RK3588, Raspberry Pi.** Low latency, multiple concurrent streams, and swap models without changing your code.

```bash
pip install voxedge
```

```python
import asyncio
from voxedge.engine import ConversationEngine
from voxedge.transport import InProcessTransport
from voxedge.backends.mock import MockASR, MockTTS, MockVAD

# Laptop / CI path — mock backends, no CUDA needed. The whole engine runs here.
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

On a real device you swap **only the backend constructors** — the engine, transport, and event contract are identical:

```python
# Device path — Jetson Orin, TensorRT engines.  pip install voxedge[jetson]
from voxedge.backends.jetson import (
    TRTEdgeLLMASRBackend, TRTEdgeLLMASRConfig,
    TRTEdgeLLMTTSBackend, TRTEdgeLLMTTSConfig,
)

engine = ConversationEngine(backends={
    "asr": TRTEdgeLLMASRBackend(TRTEdgeLLMASRConfig(...)),   # Qwen3-ASR, on-device
    "tts": TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(...)),   # Qwen3-TTS, streaming
}, multi_utterance=True)
```

> `import voxedge` is **numpy-only** — heavy runtimes (TensorRT, RKNN, sherpa-onnx) are lazy-imported by their backend adapters and pulled in via extras. The example above imports cleanly on a Mac even though the TRT engine only runs on the Jetson.

## Why voxedge

- **Runs on the edge, not the cloud.** Fully on-device ASR + TTS (and the conversation / LLM-tool loop). No speech API key, no per-call bill, no runtime internet dependency.
- **Real-time / streaming.** Partial + final ASR as the user speaks; sentence-level streaming TTS with first-audio latency low enough for live dialogue and cooperative barge-in.
- **Multi-stream concurrency.** Concurrent sessions on a single board — **N=2 verified on Orin Nano 8 GB** (byte-identical concurrent-vs-solo output, zero CUDA errors).
- **Swap models, not code.** The same `ConversationEngine` API runs across Jetson TensorRT, Rockchip NPU (RKNN), and sherpa-onnx CPU — pick a backend at construction time; your client and transport code never change.
- **Pure-Python core.** `import voxedge` needs only numpy; bring the platform runtime via extras. Install, test, and run end-to-end on a laptop with the mock backends.

voxedge is the open-core engine behind a shipped edge voice product, **[OpenVoiceStream](https://github.com/suharvest/openvoicestream)** (the FastAPI/WebSocket server, device profiles, deploy tooling, and agent gallery). Want a deployable container? Start there. Want to embed real-time edge voice in your own app? You're in the right place.

*(Think of it as "Pipecat for the edge" — but the lead is the concrete capability above, not the analogy.)*

## Install

```bash
pip install voxedge            # pure-Python core (numpy only)
pip install voxedge[sherpa]    # sherpa-onnx CPU ASR/TTS
pip install voxedge[jetson]    # Jetson TensorRT backends (aarch64)
pip install voxedge[rk]        # Rockchip RK3576/RK3588 NPU (aarch64)
pip install voxedge[llm]       # OpenAI-compatible LLM backend (httpx)
```

The `jetson` / `rk` extras declare only the pure-Python deps; the CUDA/TensorRT and RKNN runtime wheels ship from the platform (JetPack L4T / Rockchip NPU userspace) or the engine repos — you bring the platform runtime.

## Architecture

For when you've decided you're in. Four layers, all importable without CUDA.

### Backends (`voxedge/backends/`)

Clean ABCs in `backends/base.py` with **no env / profile coupling** — every constructor takes explicit params only:

- `ASRBackend` / `ASRStream` — streaming recognition.
- `TTSBackend` — `synthesize()` (batch) + `generate_streaming()` (sentence-level chunks, cooperative cancel via `cancel_token` for barge-in).
- `VADBackend` / `VADSession` — voice-activity detection that segments speech / barge-in.
- `LLMBackend` / `LLMEvent` — token-streaming LLM for the conversation loop.

Concrete adapters live under `backends/{jetson,rk,sherpa}/` and import their heavy runtimes **lazily** (inside methods), so the modules import on any machine:

- `backends/jetson/` — TensorRT path: `TRTEdgeLLMASRBackend` / `TRTEdgeLLMTTSBackend` (Qwen3 ASR/TTS), plus Matcha / Kokoro / MOSS-TTS-Nano / Paraformer / SenseVoice TRT backends. *(`voxedge[jetson]`, aarch64.)*
- `backends/rk/` — Rockchip RK3576/RK3588 RKNN/RKLLM glue over the `rkvoice_stream` engine. *(`voxedge[rk]`, aarch64.)*
- `backends/sherpa/` — sherpa-onnx CPU ASR/TTS (Paraformer/Zipformer/SenseVoice + Matcha/Kokoro ONNX). *(`voxedge[sherpa]`.)*
- `backends/llm/` — generic OpenAI-compatible `LLMBackend` over httpx; product layers subclass it for provider-specific flags. *(`voxedge[llm]`.)*
- `backends/mock.py` — `MockASR` / `MockTTS` / `MockVAD` / `MockLLM` for laptop, no-CUDA, end-to-end runs and CI.

### Transport (`voxedge/transport/`)

`Transport` ABC + two implementations:

- `InProcessTransport` — zero-IPC asyncio queues; the default, used everywhere in tests.
- `WebSocketTransport` — duck-typed ws adapter with **no FastAPI dependency** (the server product wires it to FastAPI; the library does not depend on it). Reads no env — the idle-watchdog timeout is injected by the caller.

### Conversation engine (`voxedge/engine/`)

`ConversationEngine` + a per-connection `Session` coordinator, split into focused collaborators: `audio_dispatcher` (VAD → speech / barge-in), `asr_loop`, `client_events`, `tts_sequencer` / `tts_buffer`, `session_state`, and the LLM↔tool loop — `llm_turn` over the provider-agnostic `turn_driver.run_turn` pump, with `tool_registry` (`@tool` → JSON schema) and `coordinator` / `concurrency_capability` for multi-stream concurrency.

### Capabilities (`voxedge/capabilities/`)

Optional, **default-off, stateless** add-ons (punctuation, speaker embedding) via sherpa-onnx. Opt in explicitly; byte-level no-op when off.

## Design constraints

- **Pure Python core.** No CUDA / torch / tensorrt in the core — `import voxedge` is numpy-only. Heavy adapters live under `backends/{jetson,rk,sherpa}/` and defer their runtime imports.
- **No env reads in the library.** All config is injected as explicit params. Profiles, env vars, and deployment knobs are the *product's* job ([OpenVoiceStream](https://github.com/suharvest/openvoicestream)), not the engine's.

## Status

In production — the open-core engine behind a shipped edge voice stack. ~270 mock-based tests; the whole engine runs end-to-end on a Mac with no CUDA.

License: Apache-2.0.
