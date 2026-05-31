# voxedge 使用指南：从纯 ASR 到 agent 闭环

voxedge 是边缘语音库。**同一套后端 + 引擎，按需叠加** —— 纯转写叠 0，同传叠翻译，agent 才叠 LLM+工具+TTS 的完整闭环。

## 分层

```
backends/                 后端适配器（纯能力，吃 config、无 env 耦合）
  ├─ ASRBackend           transcribe() 一句性 / create_stream() 流式
  ├─ TTSBackend           synthesize() 流式音频
  ├─ VADBackend           语音端点检测
  ├─ LLMBackend           stream_events() （agent 用，如 EdgeLLMBackend）
  └─ （Translator         ← 见文末「翻译器归属」，当前是外部 service）

engine/
  ├─ ASRSessionManager    纯 ASR 会话：partial/final/VAD 端点/取消/worker 重启
  ├─ ConversationEngine   对话闭环：ASR→(LLM+tools)→TTS（agent 才需要）
  ├─ tool_registry        工具注册 + 多轮 pump（远程/本地分发）
  └─ tts_buffer           低延迟句子缓冲
```

**核心判断：要做什么，就用到哪一层为止。** ConversationEngine 是给对话闭环的；纯 ASR / 同传不要碰它。

---

## 场景 A：只实时显示 ASR（无 LLM 无 TTS）

最轻路径 = ASR backend + ASRSessionManager。

```python
from app.core.asr_backend import create_asr_backend          # 产品工厂：按 profile 选 voxedge 后端
from voxedge.engine.asr_session_manager import ASRSessionManager

backend = create_asr_backend(); backend.preload()
mgr = ASRSessionManager(backend, language="zh")              # 或 "auto"

await mgr.on_speech_start()                                  # VAD 检测到说话起点
await mgr.accept_audio(samples)                              # 循环喂音频帧（16k PCM）
gen, partial, is_endpoint = await mgr.get_partial_for_generation()  # 轮询 partial → 显示
gen, final, accepted, lang = await mgr.finalize_with_status("vad_end")  # 端点 → final
```

一句性离线：`backend.transcribe(audio_bytes, language)` → `TranscriptionResult`。

**生产已封装成 `/asr/stream` WS**（推 `asr_partial`/`asr_final` 帧）。实时字幕/转写显示**直接连这个 WS 即可**，什么都不用自己写。配一个 VADBackend（silero）驱动 `on_speech_start`/`finalize`。

不需要：LLM / TTS / tool_registry / ConversationEngine。

---

## 场景 B：同声传译（ASR → 翻译 → 字幕）

```
voxedge ASR（同 A，每句 final）
        → 翻译器（NLLB / CTranslate2，见文末归属）
        → 字幕显示
```

- **不走 TTS**：InterpreterMode 的决策是字幕路线，不切 TTS backend（多语 TTS out-of-scope）。
- 即：voxedge ASR + 翻译器 + 字幕 UI。无 LLM、无 TTS。

---

## 场景 C：agent 对话闭环（ASR→LLM→工具→TTS）

```python
from voxedge.engine.conversation import ConversationEngine
from voxedge.engine.tool_registry import ToolRegistry
from voxedge.backends... import EdgeLLMBackend  # LLM hop（走 edge-llm /v1/chat/completions）

registry = ToolRegistry()                       # 注册本地/远程工具
engine = ConversationEngine(
    backends={"asr": asr_be, "llm": llm_be, "tts": tts_be},
    tool_registry=registry,                     # 非 None → 启用服务端多轮工具 pump
    system_prompt=..., llm_params=...,
)
# 引擎跑 ASR→LLM(+tools)→TTS；服务端 tool_call 经 wire 派发到客户端执行
```

- `backends` 只给 `asr` → 退化成 ASR 出 partial/final（但纯显示用场景 A 更轻）。
- 给 `llm` 无 `tool_registry` → ASR→LLM→TTS 普通对话。
- 给 `tool_registry` → 服务端工具闭环（机械臂 server-loop 就是这条）。

---

## 场景 D：只 TTS（文本→语音）

```python
tts_be = create_tts_backend(); tts_be.preload()
async for pcm in tts_be.synthesize(text="你好", speaker_id=..., language="zh"):
    play(pcm)
```

生产封装成 `/tts` + `/tts/stream`。

---

## 一句话对比

| 场景 | 要的层 | 不要的 |
|---|---|---|
| 实时 ASR 显示 | ASR backend + ASRSessionManager（或 `/asr/stream`）+ VAD | LLM / TTS / tools / ConversationEngine |
| 同声传译 | ASR backend + 翻译器 + 字幕 | LLM / TTS / tools |
| agent 对话 | ConversationEngine + {asr, llm, tts} + tool_registry | — |
| 纯 TTS | TTS backend | ASR / LLM / VAD |

---

## 翻译器（NLLB）归属

**现状**：NLLB 翻译器是独立微服务 `services/translator/server.py`（CTranslate2），既不在 voxedge 后端抽象里，也不在产品 `app/core` 里 —— 是个孤立的边车 service。

**建议归属**：翻译能力应抽成 voxedge 的第 5 个后端类型 **`TranslatorBackend`**，与 ASR/TTS/VAD/LLM 并列。理由：

1. **同传是一等的边缘语音用例**（ASR→翻译→字幕），翻译是语音栈原语。
2. **边缘性能工程（CT2 slim CUDA build on Jetson）是护城河，应在库里**，而非散在边车。
3. **对称性**：voxedge 已抽象 ASR/TTS/VAD/LLM，Translator 是自然的第 5 类。

**落点拆分**（镜像 ASR 的做法）：

- **voxedge**：`TranslatorBackend` ABC（base.py）+ CT2/NLLB 适配器（`backends/.../translator`）。纯翻译能力 + 边缘优化适配器。
- **产品**：翻译 service 的部署 / dashboard / 字幕 UI。`services/translator/server.py` 变成包一层 voxedge `TranslatorBackend` 的薄 server（如同 `/asr/stream` 包 ASR 后端）。InterpreterMode 编排（ASR→翻译→字幕）可做成 voxedge 组合或产品编排。

> 这是设计建议，尚未实施 —— 当前 translator 仍是 `services/translator/` 独立 service。要不要抽成 `TranslatorBackend` 是一个明确的架构决策点。
