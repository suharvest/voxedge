"""LLM-backed translation backend.

A second, *optional* implementation of the voxedge :class:`TranslatorBackend`
contract that wraps an already-constructed :class:`LLMBackend` instead of
loading a dedicated translation model. It is meant for the
"this device is already running an LLM agent and we don't want to pay for a
second model" scenario — the translation prompt is just driven through the
existing chat backend.

TRADE-OFFS (read before choosing this over NLLB):
  * **Latency / determinism are noticeably worse** than the NLLB CTranslate2
    path: autoregressive decode over a chat model is slower per token, the
    model may add preamble / chit-chat ("Sure, here's the translation:"), and
    output is non-deterministic across runs (sampling). We mitigate the
    chatter with a strict system prompt + an output-cleanup pass, but cannot
    guarantee a clean string the way a dedicated MT model does.
  * For **simultaneous interpretation** (the latency-sensitive subtitle path)
    prefer :class:`NLLBTranslatorBackend`. Reach for this wrapper only when the
    agent LLM is *already loaded* and adding a second model is not worth the
    memory.

async↔sync bridge: :meth:`LLMBackend.stream_events` is an async generator, but
:meth:`TranslatorBackend.translate` is a **synchronous** contract (mirrors
:class:`NLLBTranslatorBackend.translate`). We drive the async generator to
completion from sync code with :func:`_run_sync`, which is loop-aware: if no
event loop is running it uses :func:`asyncio.run`; if a loop is already running
on the calling thread it offloads the coroutine to a dedicated worker thread
with its own loop (so we never raise "asyncio.run() cannot be called from a
running event loop" and never deadlock by re-entering the live loop). An
``async`` convenience variant :meth:`atranslate` is also provided for callers
that are already inside a loop and want to await directly.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

from voxedge.backends.base import (
    LLMBackend,
    TranslationResult,
    TranslatorBackend,
    TranslatorCapability,
    TranslatorConfig,
)

# System prompt: force the model into a pure translation engine. Kept terse so
# small edge LLMs follow it reliably; the cleanup pass below mops up the rest.
_SYSTEM_PROMPT = "你是翻译引擎，只输出译文，不解释不寒暄"

# Characters we strip from the model output edges: assorted quote styles plus
# whitespace. A chat LLM frequently wraps the translation in quotes.
_STRIP_CHARS = " \t\r\n\"'“”‘’「」『』《》"


def _run_sync(coro: Any) -> Any:
    """Drive an awaitable to completion from synchronous code, loop-aware.

    * No running loop on this thread  → :func:`asyncio.run` (fast path).
    * A loop *is* already running here → run the coroutine on a separate
      thread with a fresh loop and block until it finishes. This avoids both
      the ``RuntimeError`` from re-entering :func:`asyncio.run` and the
      deadlock from blocking the very loop that must drive the coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running on this thread — safe to own one.
        return asyncio.run(coro)

    # A loop is already running on this thread: offload to a worker thread.
    result: dict[str, Any] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 — re-raised on caller thread
            result["error"] = exc

    t = threading.Thread(target=_worker, name="llm-translate-bridge", daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


class LLMTranslatorBackend(TranslatorBackend):
    """Translate by prompting an existing :class:`LLMBackend`.

    Construct with an **already-built** ``LLMBackend`` instance — this wrapper
    never loads a model of its own; it reuses the agent's LLM. An optional
    :class:`TranslatorConfig` supplies the default ``src_lang`` / ``tgt_lang``
    (only those two fields are used here; the CT2-specific fields such as
    ``model_path`` / ``device`` are ignored, since there is no second model).
    """

    # The wrapped LLM owns its own resources; we don't manage hot reload here.
    supports_hot_reload = False

    def __init__(
        self,
        llm: LLMBackend,
        config: Optional[TranslatorConfig] = None,
    ) -> None:
        self._llm = llm
        # Only src_lang / tgt_lang are meaningful for this backend; model_path
        # is irrelevant (no model loaded) so a placeholder default is fine.
        self._config = config or TranslatorConfig(model_path="")

    # ── identity / capabilities ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "llm_translator"

    @property
    def capabilities(self) -> set[TranslatorCapability]:
        # TEXT + MULTI_LANGUAGE only. We deliberately do NOT advertise BATCH
        # (no batched decode — would just be a serial loop) nor STREAMING
        # (translate() returns a finished string).
        return {
            TranslatorCapability.TEXT,
            TranslatorCapability.MULTI_LANGUAGE,
        }

    def is_ready(self) -> bool:
        """Delegate readiness to the wrapped LLM when it exposes ``is_ready``.

        :class:`LLMBackend` has no ``is_ready`` in its ABC, so if the concrete
        LLM defines one we honour it; otherwise a constructed wrapper with a
        non-None LLM is considered ready.
        """
        probe = getattr(self._llm, "is_ready", None)
        if callable(probe):
            try:
                return bool(probe())
            except Exception:
                return False
        return self._llm is not None

    # ── lifecycle ───────────────────────────────────────────────────────

    def preload(self) -> None:
        """Delegate to the wrapped LLM's ``preload`` if it has one (else no-op).

        The LLM is expected to be already constructed/loaded by the agent
        layer; this is best-effort so a shared, pre-warmed LLM isn't re-loaded.
        """
        probe = getattr(self._llm, "preload", None)
        if callable(probe):
            probe()

    # ── prompt construction ─────────────────────────────────────────────

    def _build_messages(
        self, text: str, src: str, tgt: str
    ) -> list[dict[str, Any]]:
        """Build the chat messages: strict system prompt + a user instruction
        carrying src_lang / tgt_lang and the source text."""
        user = (
            f"把下面的文本从 {src} 翻译成 {tgt}，只输出译文：\n{text}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _clean(out: str) -> str:
        """Strip leading/trailing whitespace and wrapping quotes the chat model
        may have added around the translation."""
        return out.strip().strip(_STRIP_CHARS).strip()

    # ── translation (async core + sync contract) ────────────────────────

    async def _atranslate_raw(self, messages: list[dict[str, Any]]) -> str:
        """Drive ``LLMBackend.stream_events`` and concatenate text deltas.

        ``stream_events(messages, **kw) -> AsyncIterator[LLMEvent]`` where each
        event has ``kind`` in {"text","tool_call_delta","finish"}; we keep only
        the ``text`` deltas (event.text) and ignore tool-call / finish events.
        """
        parts: list[str] = []
        async for ev in self._llm.stream_events(messages):
            if ev.kind == "text" and ev.text:
                parts.append(ev.text)
        return "".join(parts)

    async def atranslate(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> TranslationResult:
        """Async variant of :meth:`translate` for callers already in a loop."""
        src = src_lang or self._config.src_lang
        tgt = tgt_lang or self._config.tgt_lang
        messages = self._build_messages(text, src, tgt)
        raw = await self._atranslate_raw(messages)
        return TranslationResult(
            text=self._clean(raw), src_lang=src, tgt_lang=tgt
        )

    def translate(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> TranslationResult:
        """Synchronous translate (mirrors :class:`NLLBTranslatorBackend`).

        Bridges the async LLM stream to sync via :func:`_run_sync`.
        """
        src = src_lang or self._config.src_lang
        tgt = tgt_lang or self._config.tgt_lang
        messages = self._build_messages(text, src, tgt)
        raw = _run_sync(self._atranslate_raw(messages))
        return TranslationResult(
            text=self._clean(raw), src_lang=src, tgt_lang=tgt
        )


__all__ = ["LLMTranslatorBackend"]
