# voxedge

> [English](README.md) | **中文**

<p align="center">
  <img src="media/banner.png" alt="voxedge banner" width="100%">
</p>

[![PyPI](https://img.shields.io/pypi/v/voxedge)](https://pypi.org/project/voxedge)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/voxedge)](https://pypi.org/project/voxedge)

**原生 TensorRT · RKNN · sherpa-onnx 语音流水线，适配 Jetson、瑞芯微与树莓派 —— 完全离线，在真实硬件上验证，零云依赖。**

<!-- TODO: 添加演示 GIF —— 建议录制约 15 秒的终端视频，展示 Jetson Orin 上的 ASR→TTS 流程（存放至 media/demo.gif）-->

## voxedge 是什么？

voxedge 是一个可嵌入的 Python 库，通过直接调用各平台原生推理运行时来驱动实时、端侧语音对话 —— Jetson Orin 上是 TensorRT，RK3576/RK3588 上是 RKNN，CPU 上是 sherpa-onnx。无需云端 STT/TTS API，运行时不依赖网络，没有中间抽象层的性能损耗。同一套 `ConversationEngine` 契约横跨三类后端，但具体功能与并发上限仍取决于模型、运行时和产物组合。

voxedge 是已上线产品 **[OpenVoiceStream](https://github.com/suharvest/openvoicestream)** 的开源内核（产品侧含 FastAPI/WebSocket 服务、设备 profile、部署工具与 agent 库）。想要可部署的容器？从那里开始。想把实时边缘语音嵌进自己的应用？这里就是对的地方。

## 核心特性

- **原生运行时，充分发挥性能** —— 直接调用 TensorRT（Jetson）、RKNN（瑞芯微）、sherpa-onnx（CPU），无封装开销，无跨平台抽象损耗
- **完全离线** —— 无需语音 API key，无按次计费，运行时不依赖网络
- **硬件证据有明确边界** —— 部分 Jetson slot-pool 组合已在 Orin Nano 8 GB 通过 N=2 冒烟测试；这不是对所有后端、模型、设备或产物构建的普遍 N=2 保证
- **流式 + 打断（barge-in）** —— 用户说话时即出 partial + final ASR；句级 TTS 流式输出，首音延迟低到足以支撑实时对话与协作式打断
- **换硬件，不换代码** —— 同一套 `ConversationEngine` API 横跨 Jetson、瑞芯微、sherpa-onnx CPU，只需替换后端构造器
- **任何机器均可测试** —— mock 后端只依赖 numpy；整个引擎在无 CUDA、无 GPU 的 Mac 上即可端到端运行

## 快速上手

在任意机器上即可运行，无需 GPU。换到真实设备时只替换后端构造器，引擎、传输层、事件契约完全不变。

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
    await t.feed_audio(b"\x01\x02" * 8000)   # 语音帧（int16 PCM）
    await t.feed_audio(b"\x00\x00" * 8000)   # 静音 → VAD 切分出一句话
    t.end_input()
    await engine.run(t)                       # 驱动 ASR → (LLM) → TTS
    for ev in t.drain_events_nowait():        # asr_final / tts_* / ...
        print(ev["type"], ev.get("text", ""))

asyncio.run(main())
```

在真实设备上，只替换**后端构造器**，其余完全不变：

```python
# Jetson Orin —— pip install voxedge[jetson]
from voxedge.backends.jetson import (
    TRTEdgeLLMASRBackend, TRTEdgeLLMASRConfig,
    TRTEdgeLLMTTSBackend, TRTEdgeLLMTTSConfig,
)

engine = ConversationEngine(backends={
    "asr": TRTEdgeLLMASRBackend(TRTEdgeLLMASRConfig(...)),   # Qwen3-ASR，原生 TRT
    "tts": TRTEdgeLLMTTSBackend(TRTEdgeLLMTTSConfig(...)),   # Qwen3-TTS，流式
}, multi_utterance=True)
```

> `import voxedge` **只依赖 numpy** —— TensorRT、RKNN、sherpa-onnx 由各自的后端适配器惰性导入，通过 extras 安装。因此上面的例子在 Mac 上也能干净导入，即便 TRT 引擎只在 Jetson 上真正运行。

## 安装

```bash
pip install voxedge            # 纯 Python 核心（仅 numpy）
pip install voxedge[sherpa]    # sherpa-onnx CPU ASR/TTS
pip install voxedge[jetson]    # Jetson TensorRT 后端（aarch64）
pip install "voxedge[rk]==0.0.13a0" # 瑞芯微 RK3576/RK3588 NPU（aarch64）
pip install voxedge[llm]       # OpenAI 兼容 LLM 后端（httpx）
```

`jetson` extra 只安装包索引中可获取的 Python 侧依赖；它**不会**安装 TensorRT/CUDA、C++/pybind worker、plugin、TensorRT engine 或模型产物。这些需来自 JetPack 与引擎/产物 release。因此，单独执行 `pip install voxedge[jetson]` 不会得到可运行的 Jetson 部署。RKNN 与 `voxedge[rk]` 同样存在平台运行时边界。

### Rockchip 上的 Kokoro ConvOnly

Kokoro ConvOnly 已经是 RKVoice Stream 的一级 TTS backend，由 VoxEdge
统一适配，并被 OpenVoiceStream（OVS）直接使用。在 aarch64 设备上安装：
`pip install "voxedge[rk]==0.0.13a0"`；`rk` extra 依赖
`rkvoice-stream>=0.2.0`。

RK3576 与 RK3588 使用同一套统一应用镜像。RKNN 模型和可选的日语词典
作为外置只读 platform bundle 挂载，切换模型或区域不需要重新制作应用
镜像。OVS 通过 backend profile 配置 bundle 路径。`language=auto` 会按
文本路由语言；平假名、片假名、半角片假名及相关日文假名文本路由到
`ja`。

Kokoro ConvOnly 支持通过 RKVoice Stream 的句级 streaming，以及 HTTP
有限请求取消。消费者关闭 stream 会传递到底层 native iterator，并释放
backend 生命周期所有权。对于底层一次性完成的 RKNN 调用，不强制中止
当前推理；这类 backend 会先完成当前调用再清理资源。

## Python 库与服务端的边界

voxedge 是 Python 库：提供后端接口、对话引擎以及进程内/WebSocket 传输适配器。它不会启动 FastAPI，也不直接暴露 HTTP route。**OpenVoiceStream（OVS）**才是可部署的服务/产品层，负责 profile 选择、产物、容器和网络 API。

OVS 提供的 OpenAI 兼容语音接口包括：

- `POST /v1/audio/speech` —— 通过 HTTP chunked transfer 流式返回音频，而不是缓冲完整音频后一次返回；
- `GET /v1/models` —— 列出该 OVS 部署可用的模型；
- `GET /v1/capabilities` —— 返回该部署检测到的运行时/后端能力。

上述 route 属于 OVS，不是本仓库实现的 endpoint。

## 架构

四层，全部无需 CUDA 即可导入。

### 后端（`voxedge/backends/`）

`backends/base.py` 中是干净的 ABC —— 每个构造器只接受显式参数，不耦合 env：

- `ASRBackend` / `ASRStream` —— 流式识别
- `TTSBackend` —— `synthesize()`（整段）+ `generate_streaming()`（句级 chunk，通过 `cancel_token` 协作式取消以支持 barge-in）
- `VADBackend` / `VADSession` —— 切分语音 / 打断的语音活动检测
- `LLMBackend` / `LLMEvent` —— 对话循环用的 token 流式 LLM

具体适配器位于 `backends/{jetson,rk,sherpa}/`，**惰性导入**各自的重型运行时（在方法内部），所以模块在任意机器上都能导入：

| 后端 | 平台 | 模型 | Extra | 底层引擎源码 |
|------|------|------|-------|------------|
| `backends/jetson/` | Jetson Orin（TensorRT） | Qwen3-ASR/TTS、Matcha、Kokoro、Paraformer、SenseVoice、MOSS-TTS-Nano | `voxedge[jetson]` aarch64 | [jetson-voice-engine](https://github.com/suharvest/qwen3-edgellm-jetson) |
| `backends/rk/` | 瑞芯微 RK3576/RK3588（RKNN） | Qwen3-ASR、Matcha、Piper、Kokoro、Paraformer、SenseVoice | `voxedge[rk]` aarch64 | [rkvoice-stream](https://github.com/suharvest/rkvoice-stream) |
| `backends/sherpa/` | CPU（任意架构） | Paraformer、Zipformer、SenseVoice、Matcha、Kokoro ONNX | `voxedge[sherpa]` | — |
| `backends/llm/` | 任意 | OpenAI 兼容 LLM（httpx） | `voxedge[llm]` | — |
| `backends/mock.py` | 开发 / CI | MockASR、MockTTS、MockVAD、MockLLM | 核心包 | — |

### Jetson TTS 音色、克隆与速度能力矩阵

| 模型/后端 | 音色选择 | 音色克隆 / 录入 | 速度行为 |
|---|---|---|---|
| Matcha TRT | 固定/单音色 | 不支持克隆 | 原生连续速度（`length_scale`） |
| Qwen3-TTS Base | speaker embedding；可选配置固定 base embedding 作为默认音色 | 可消费可复用 embedding；只有存在 speaker-encoder 产物时才能从参考 WAV 录入 | 连续速度走 numpy DSP fallback |
| Qwen3-TTS CustomVoice | 内置 named speakers | 不支持外部音色克隆或录入 | 连续速度走 numpy DSP fallback |
| MOSS-TTS-Nano | 每次请求用参考音频条件化 | 支持 reference-audio prompt-prefix 克隆；不提供可复用 speaker embedding API | 连续速度走 numpy DSP fallback |
| SparkTTS | 可控 gender/pitch/speed **离散 style label** | 仅配置 `voices_dir` 时启用 registry clone，通过已录入 `voice_id` 选择 | 离散 speed label 是原生控制；连续倍率走 DSP fallback |

“DSP fallback”表示 voxedge 对 PCM 做后处理，它不是模型原生控制，音质和时延特性可能不同。

### 传输层（`voxedge/transport/`）

`Transport` ABC + 两个实现：

- `InProcessTransport` —— 零 IPC 的 asyncio 队列；默认实现，测试中处处使用
- `WebSocketTransport` —— 鸭子类型的 ws 适配器，不依赖 FastAPI；空闲看门狗超时由调用方注入，不读 env

### 对话引擎（`voxedge/engine/`）

`ConversationEngine` + 每连接一个的 `Session` 协调器，拆成聚焦的协作体：`audio_dispatcher`（VAD → 语音 / 打断）、`asr_loop`、`client_events`、`tts_sequencer` / `tts_buffer`、`session_state`，以及 LLM↔工具循环 —— `llm_turn` 跑在与厂商无关的 `turn_driver.run_turn` pump 之上，配 `tool_registry`（`@tool` → JSON schema）与 `coordinator` / `concurrency_capability` 做多路并发。

### Capabilities（`voxedge/capabilities/`）

可选、默认关闭、无状态的附加能力（标点、声纹）走 sherpa-onnx。需显式开启；关闭时为字节级 no-op。

## 设计约束

- **纯 Python 核心** —— `import voxedge` 只依赖 numpy。重型适配器位于 `backends/{jetson,rk,sherpa}/`，运行时导入被推迟。
- **显式运行时配置** —— 后端配置以参数注入；profile 和部署开关属于 [OpenVoiceStream](https://github.com/suharvest/openvoicestream)。可选 artifact downloader 是唯一明确的例外：它在显式 endpoint 之后、manifest endpoint 之前遵循 `HF_ENDPOINT`。

## 状态

voxedge **0.0.9a0** 为当前版本；通过 TensorRT Edge-LLM v0.9.1 验收的 OVS 运行时
自 2026-08-09 起固定在该版本。

0.0.9a0 把 sherpa SenseVoice 后端里两个本该由部署方决定、却写死在库里的选择放了出来。
`use_itn` 原本恒为 `True`；ITN 不只是标点，它还决定数字形态（`9点`/`九点`）、英文大小写，
在 2024-07-17 导出上甚至会改词（`开放时间`/`开饭时间`）。而在 2025-09-09 导出上，
`use_itn=True` 会直接吞掉首字 —— 写死的这个值在那里是有害的，调用方却无从关闭。
现由 `SherpaASRConfig.offline_use_itn` 控制，默认仍为 `True`，现有部署行为逐字节不变。

`transcribe(audio, language=...)` 另有一处：它收下 per-request 的 language 后完全不用，
却把它原样填回 `TranscriptionResult.language` —— 调用方请求 `zh`，拿到一个自称 `zh` 的
结果，而 recognizer 实际跑在 `auto`。SenseVoice 在构造 recognizer 时绑定语言，要让
per-request 生效就得每种语言各驻留一份模型（各约 237 MB）。因此改为部署级 pin
（`offline_language`），并让结果诚实汇报 recognizer 实际构造时所用的语言，对冲突的请求
按取值各警告一次。需注意该 pin 只是弱提示：设定后，其他受支持语种的音频依然按其本身
语言输出，差异主要在标点位置。

0.0.9a0 同时统一了 `TranscriptionResult.language` 的语义。此前各后端错法不一且方向相反：
sherpa 与 Paraformer-TRT 把调用方的请求原样回显 —— 而 Paraformer 压根没有语言开关，
该参数从未传到解码器；SenseVoice-TRT 则相反，语言真的生效了却恒报 `None`。
现在该字段一律表示**后端实际解码时所用的语言**，后端不做语言选择时为 `None`。
三种形态（配置级 pin / 逐次生效 / 完全不选语言）由
`backends.base.resolve_reported_language()` 统一承载，避免同一套判断在各后端重复推导、
并各自推导歪掉。

0.0.8a0 新增短音频解码退化的塌缩守卫。Qwen3-ASR 在贪心解码（`top_k=1`）下会退化
成整段复读：300ms 片段实测输出「帮我，」×128。这**不是**流式路径的问题 —— 同一段
音频走离线 `/asr` 得到逐字相同的结果。worker 只接受
`temperature`/`top_k`/`top_p`/`max_generate_length`，不支持重复惩罚，所以守卫放在
文本侧、两条路径的公共出口上。取舍刻意保守：空格分隔语言按词、门槛 6 份（`the the
the` 不会被误伤），中日韩按字符（单字 8 份、多字 3 份），两者都要求重复部分覆盖全段
80% 以上；只重复 2 份不收。

0.0.7a0 含两处现场修复与一处**破坏性**变更：

* 阿拉伯数字在查词典前转成中文读法。matcha-icefall-zh-en 的 token 表里没有阿拉伯
  数字，查表未命中会被静默跳过 —— 于是数字不是读错，而是**完全不发音**。
* 流式 ASR 的 worker 槽位现在在所有 finalize 路径上都会归还。此前只有「零音频」
  分支会归还，于是在 `max_slots=1` 下每次成功识别都漏一个槽位，第二轮起必然
  `PoolSaturatedError`。
* **破坏性**：所有 `ASRStream` 子类必须显式声明 `OWNS_RESOURCES`。不声明、或声明
  `True` 却未实现 `close()`，会在**类定义时**抛 `TypeError`。此前 `close()` 是空
  实现，使得「忘了实现」与「无需实现」无法区分 —— 上面那个槽位泄漏正是因此长期
  未被发现。该版本已通过 mock 测试套件和 Orin NX
目标设备验证；平台运行时、worker 与模型 engine 仍作为独立部署产物提供。
特别是上述 OpenAI HTTP route 属于 OVS 行为，不是 voxedge 的兼容性承诺。

## 参与贡献

欢迎提 Issue 和 PR。mock 后端测试套件在任何机器上无需硬件即可运行：

```bash
pip install voxedge
uv run pytest
```

## 项目生态

voxedge 是一个系列仓库中的一层：

| 仓库 | 定位 | 适合去看的场景 |
|------|------|--------------|
| **voxedge**（本仓库） | 可嵌入的 Python 引擎 | 把实时语音嵌入自己的应用 |
| [openvoicestream](https://github.com/suharvest/openvoicestream) | 可部署的 FastAPI/WebSocket 服务、Docker profile、agent 库 | 端到端的部署案例和完整示例；开箱即用的容器 |
| [rkvoice-stream](https://github.com/suharvest/rkvoice-stream) | 瑞芯微 NPU 引擎（`backends/rk/` 包装此库） | RK3576/RK3588 模型格式、RKNN 性能数据、TTS/ASR 后端内部实现 |
| [jetson-voice-engine](https://github.com/suharvest/qwen3-edgellm-jetson) | Jetson TensorRT 构建脚本、模型导出、产物（`backends/jetson/` 包装此库） | Jetson 模型转换、TRT 引擎构建、Orin 专属优化 |

## 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) —— CPU ASR/TTS 运行时
- [OpenVoiceStream](https://github.com/suharvest/openvoicestream) —— 基于本引擎构建的可部署服务端产品

## 许可证

Apache-2.0，详见 [LICENSE](LICENSE)。
