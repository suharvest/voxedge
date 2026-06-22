# voxedge

> [English](README.md) | **中文**

**在边缘设备上本地运行实时语音识别与合成 —— Jetson Orin、RK3576/RK3588、树莓派。** 低延迟、多路并发、换模型不改代码。

```bash
pip install voxedge
```

```python
import asyncio
from voxedge.engine import ConversationEngine
from voxedge.transport import InProcessTransport
from voxedge.backends.mock import MockASR, MockTTS, MockVAD

# 笔记本 / CI 路径 —— mock 后端，无需 CUDA，整个引擎都能跑。
engine = ConversationEngine(
    backends={"asr": MockASR(transcript="hello world"), "tts": MockTTS(), "vad": MockVAD()},
    multi_utterance=True,
)

async def main():
    t = InProcessTransport()
    await t.feed_audio(b"\x01\x02" * 8000)   # 语音帧（int16 PCM）
    await t.feed_audio(b"\x00\x00" * 8000)   # 静音 → VAD 切分出一句话
    t.end_input()
    await engine.run(t)                       # 驱动 ASR → (LLM) → TTS
    for ev in t.drain_events_nowait():        # asr_final / tts_* / ...
        print(ev["type"], ev.get("text", ""))

asyncio.run(main())
```

在真实设备上，你**只需替换后端构造器** —— 引擎、传输层、事件契约完全不变：

```python
# 设备路径 —— Jetson Orin，TensorRT 引擎。 pip install voxedge[jetson]
from voxedge.backends.jetson import (
    TRTEdgeLLMASRBackend, TRTEdgeLLMASRConfig,
    TRTEdgeLLMTTSBackend, TRTEdgeLLMTTSConfig,
)

engine = ConversationEngine(backends={
    "asr": TRTEdgeLLMASRBackend(TRTEdgeLLMASRConfig(...)),   # Qwen3-ASR，端侧
    "tts": TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(...)),   # Qwen3-TTS，流式
}, multi_utterance=True)
```

> `import voxedge` **只依赖 numpy** —— 重型运行时（TensorRT、RKNN、sherpa-onnx）由各自的后端适配器惰性导入，通过 extras 安装。所以上面的例子在 Mac 上也能干净导入，即便 TRT 引擎只在 Jetson 上真正运行。

## 为什么选 voxedge

- **跑在边缘，不依赖云。** 完全端侧的 ASR + TTS（以及对话 / LLM-工具循环）。无需语音 API key、无按次计费、运行时不依赖联网。
- **实时 / 流式。** 用户说话时即出 partial + final ASR；句级流式 TTS，首音延迟低到足以支撑实时对话与协作式打断（barge-in）。
- **多路并发。** 单板上多会话并发 —— **Orin Nano 8GB 实测 N=2**（并发与单路输出逐字节一致，零 CUDA 错误）。
- **换模型，不换代码。** 同一套 `ConversationEngine` API 横跨 Jetson TensorRT、瑞芯微 NPU（RKNN）、sherpa-onnx CPU —— 构造时选后端，客户端与传输代码永不改动。
- **纯 Python 核心。** `import voxedge` 只需 numpy；平台运行时由 extras 自带。用 mock 后端即可在笔记本上安装、测试、端到端运行。

voxedge 是一个已上线的边缘语音产品 **[OpenVoiceStream](https://github.com/suharvest/openvoicestream)** 背后的开源内核（产品侧含 FastAPI/WebSocket 服务、设备 profile、部署工具与 agent 库）。想要可部署的容器？从那里开始。想把实时边缘语音嵌进自己的应用？这里就对了。

*（可以理解为"边缘版的 Pipecat" —— 但主线是上面那条具体能力，不是这个类比。）*

## 安装

```bash
pip install voxedge            # 纯 Python 核心（仅 numpy）
pip install voxedge[sherpa]    # sherpa-onnx CPU ASR/TTS
pip install voxedge[jetson]    # Jetson TensorRT 后端（aarch64）
pip install voxedge[rk]        # 瑞芯微 RK3576/RK3588 NPU（aarch64）
pip install voxedge[llm]       # OpenAI 兼容 LLM 后端（httpx）
```

`jetson` / `rk` extras 只声明纯 Python 依赖；CUDA/TensorRT 与 RKNN 运行时 wheel 来自平台（JetPack L4T / 瑞芯微 NPU 用户态）或引擎仓库 —— 平台运行时由你自带。

## 架构

给已经决定上车的人看。四层，全部无需 CUDA 即可导入。

### 后端（`voxedge/backends/`）

`backends/base.py` 里是干净的 ABC，**不耦合 env / profile** —— 每个构造器只接受显式参数：

- `ASRBackend` / `ASRStream` —— 流式识别。
- `TTSBackend` —— `synthesize()`（整段）+ `generate_streaming()`（句级 chunk，通过 `cancel_token` 协作式取消以支持 barge-in）。
- `VADBackend` / `VADSession` —— 切分语音 / 打断的语音活动检测。
- `LLMBackend` / `LLMEvent` —— 对话循环用的 token 流式 LLM。

具体适配器位于 `backends/{jetson,rk,sherpa}/`，**惰性导入**各自的重型运行时（在方法内部），所以模块在任意机器上都能导入：

- `backends/jetson/` —— TensorRT 路径：`TRTEdgeLLMASRBackend` / `TRTEdgeLLMTTSBackend`（Qwen3 ASR/TTS），外加 Matcha / Kokoro / MOSS-TTS-Nano / Paraformer / SenseVoice 的 TRT 后端。*（`voxedge[jetson]`，aarch64。）*
- `backends/rk/` —— 瑞芯微 RK3576/RK3588 的 RKNN/RKLLM 胶水，封装 `rkvoice_stream` 引擎。*（`voxedge[rk]`，aarch64。）*
- `backends/sherpa/` —— sherpa-onnx CPU ASR/TTS（Paraformer/Zipformer/SenseVoice + Matcha/Kokoro ONNX）。*（`voxedge[sherpa]`。）*
- `backends/llm/` —— 通用 OpenAI 兼容 `LLMBackend`（基于 httpx）；产品层可子类化以注入特定厂商的请求标志。*（`voxedge[llm]`。）*
- `backends/mock.py` —— `MockASR` / `MockTTS` / `MockVAD` / `MockLLM`，用于笔记本、无 CUDA、端到端运行与 CI。

### 传输层（`voxedge/transport/`）

`Transport` ABC + 两个实现：

- `InProcessTransport` —— 零 IPC 的 asyncio 队列；默认实现，测试中处处使用。
- `WebSocketTransport` —— 鸭子类型的 ws 适配器，**不依赖 FastAPI**（服务端产品把它接到 FastAPI；库本身不依赖）。不读 env —— 空闲看门狗超时由调用方注入。

### 对话引擎（`voxedge/engine/`）

`ConversationEngine` + 每连接一个的 `Session` 协调器，拆成聚焦的协作体：`audio_dispatcher`（VAD → 语音 / 打断）、`asr_loop`、`client_events`、`tts_sequencer` / `tts_buffer`、`session_state`，以及 LLM↔工具循环 —— `llm_turn` 跑在与厂商无关的 `turn_driver.run_turn` pump 之上，配 `tool_registry`（`@tool` → JSON schema）与 `coordinator` / `concurrency_capability` 做多路并发。

### Capabilities（`voxedge/capabilities/`）

可选、**默认关闭、无状态**的附加能力（标点、声纹）走 sherpa-onnx。需显式开启；关闭时为字节级 no-op。

## 设计约束

- **纯 Python 核心。** 核心不含 CUDA / torch / tensorrt —— `import voxedge` 只依赖 numpy。重型适配器位于 `backends/{jetson,rk,sherpa}/`，运行时导入被推迟。
- **库内不读 env。** 所有配置以显式参数注入。profile、环境变量、部署开关是*产品*（[OpenVoiceStream](https://github.com/suharvest/openvoicestream)）的职责，不是引擎的。

## 状态

已上线 —— 一个已发货的边缘语音栈背后的开源内核。约 270 个基于 mock 的测试；整个引擎能在无 CUDA 的 Mac 上端到端运行。

许可证：Apache-2.0。
