# voxedge

> Edge-native, local-first real-time voice conversation library — "Pipecat for the edge".

**Status: in production.** The open-core engine behind a shipped edge voice
stack (Jetson Orin / RK3576 / Raspberry Pi). The core stays pure-Python and
numpy-only; heavy inference lives behind lazy-imported backend adapters.

voxedge is an edge-native library for low-latency, local-first real-time voice
conversation. It was originally extracted as the open-core foundation of a
production edge voice stack and now hosts the full conversation engine.

## What's here

- `voxedge/backends/base.py` — clean backend ABCs (`ASRBackend`/`ASRStream`,
  `TTSBackend`, `VADBackend`/`VADSession`, `LLMBackend`/`LLMEvent`) with **no
  env / profile coupling** — constructors take explicit params only. Concrete
  adapters live under `voxedge/backends/{jetson,rk,sherpa}/` (lazy heavy imports)
  plus `mock.py` for laptop, no-CUDA end-to-end runs.
- `voxedge/transport/base.py` — `Transport` ABC + `InProcessTransport`
  (zero-IPC asyncio queues, the default) + `WebSocketTransport` (duck-typed
  ws adapter, no FastAPI dependency).
- `voxedge/engine/` — the conversation engine, split into focused collaborators:
  `conversation.py` (`ConversationEngine` + the `Session` coordinator),
  `audio_dispatcher` (VAD → speech/barge-in), `asr_loop`, `client_events`,
  `tts_sequencer`/`tts_buffer`, `session_state`, and the LLM↔tool loop —
  `llm_turn` (server adapter) over the provider-agnostic `turn_driver.run_turn`
  pump, with `tool_registry` (`@tool` → JSON schema) and `coordinator` /
  `capability_resolver` for concurrency.
- `voxedge/capabilities/` — optional, default-off, stateless add-ons
  (punctuation, speaker embedding) via sherpa-onnx.
- `voxedge/tests/` — ~225 mock-based tests; the whole engine runs end-to-end on
  a Mac with no CUDA.

## Design constraints

- **Pure Python.** No CUDA / torch / tensorrt in the core — `import voxedge` is
  numpy-only. Heavy adapters live under `backends/{jetson,rk,sherpa}/` and import
  their runtimes lazily; the optional extras (`voxedge[trt]`, `voxedge[rknn]`)
  intentionally declare no deps — you bring the platform runtime.
- **No env reads in the library.** All config is injected as explicit params.

## Quickstart (mock backends)

```python
import asyncio
from voxedge.engine import ConversationEngine
from voxedge.transport import InProcessTransport
from voxedge.backends.mock import MockASR, MockTTS, MockVAD

engine = ConversationEngine(
    backends={"asr": MockASR(), "tts": MockTTS(), "vad": MockVAD()},
    multi_utterance=True,
)
transport = InProcessTransport()
asyncio.run(engine.run(transport))
```
