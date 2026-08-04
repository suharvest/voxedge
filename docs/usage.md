# voxedge 使用指南：从纯 ASR 到 agent 闭环

voxedge 是边缘语音库。**同一套后端 + 引擎，按需叠加** —— 纯转写叠 0，同传叠翻译，agent 才叠 LLM+工具+TTS 的完整闭环。

## 分层

```
backends/                 后端适配器（纯能力，吃 config、无 env 耦合）
  ├─ ASRBackend           transcribe() 一句性 / create_stream() 流式
  ├─ TTSBackend           synthesize() 整段 WAV / generate_streaming() PCM chunk
  ├─ VADBackend           语音端点检测
  ├─ LLMBackend           stream_events() （agent 用）
  └─ TranslatorBackend    translate() / translate_batch()

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
from voxedge.backends.sherpa import SherpaASRBackend, SherpaASRConfig
from voxedge.engine.asr_session_manager import ASRSessionManager

backend = SherpaASRBackend(SherpaASRConfig(streaming_provider="cpu"))
backend.preload()
mgr = ASRSessionManager(backend, language="zh")              # 或 "auto"

await mgr.on_speech_start()                                  # VAD 检测到说话起点
await mgr.accept_audio(samples)                              # 循环喂音频帧（16k PCM）
gen, partial, is_endpoint = await mgr.get_partial_for_generation()  # 轮询 partial → 显示
gen, final, accepted, lang = await mgr.finalize_with_status("vad_end")  # 端点 → final
```

一句性离线：`backend.transcribe(audio_bytes, language)` → `TranscriptionResult`。

OpenVoiceStream（OVS）服务端另行封装了 `/asr/stream` WebSocket（推送
`asr_partial`/`asr_final` 帧）。这是 OVS 的 HTTP/WS route，不是 voxedge
库自带的服务器；直接嵌入库时，由调用方配置 VAD 并驱动
`on_speech_start()` / `finalize_with_status()`。

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
from voxedge.backends.llm import OpenAICompatBackend

llm_be = OpenAICompatBackend("http://edge-llm:8000", model="local-model")
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
from voxedge.backends.sherpa import SherpaTTSBackend, SherpaTTSConfig

tts_be = SherpaTTSBackend(SherpaTTSConfig(provider="cpu"))
tts_be.preload()
wav_bytes, metadata = tts_be.synthesize(text="你好", speaker_id=0, language="zh")
open("hello.wav", "wb").write(wav_bytes)

# 需要流式时改用同步迭代器；chunk 是 raw PCM，不是 WAV。
for pcm in tts_be.generate_streaming(text="你好", language="zh"):
    play(pcm)
```

OVS 产品层另行封装 `/tts` 与 `/tts/stream`；voxedge 本身只提供
Python backend / engine API。

---

## 一句话对比

| 场景 | 要的层 | 不要的 |
|---|---|---|
| 实时 ASR 显示 | ASR backend + ASRSessionManager + VAD（或选用 OVS `/asr/stream`） | LLM / TTS / tools / ConversationEngine |
| 同声传译 | ASR backend + 翻译器 + 字幕 | LLM / TTS / tools |
| agent 对话 | ConversationEngine + {asr, llm, tts} + tool_registry | — |
| 纯 TTS | TTS backend | ASR / LLM / VAD |

---

## 翻译器（NLLB）

`TranslatorBackend` 已是 voxedge 的一等后端接口，NLLB/CTranslate2 实现为
`NLLBTranslatorBackend`：

```python
from voxedge.backends.base import TranslatorConfig
from voxedge.backends.nllb_translator import NLLBTranslatorBackend

translator = NLLBTranslatorBackend(TranslatorConfig(
    model_path="/opt/models/nllb-200-distilled-600M-ct2",
    device="cpu",
))
translator.preload()
result = translator.translate("你好", src_lang="zho_Hans", tgt_lang="eng_Latn")
print(result.text)
```

模型路径、CPU/CUDA 选择以显式 `TranslatorConfig` 注入。OVS 如需提供翻译
HTTP 服务，应在产品层封装这个 backend；部署、dashboard 和字幕 UI
仍不属于 voxedge Python 库。
