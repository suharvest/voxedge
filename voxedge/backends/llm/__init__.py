"""voxedge LLM backend adapters.

The generic, reusable LLM backend slot — symmetric with the ASR / TTS / VAD
backends. :class:`OpenAICompatBackend` speaks the OpenAI-compatible streaming
chat protocol (``/v1/chat/completions`` SSE) over ``httpx`` and yields voxedge
:class:`~voxedge.backends.base.LLMEvent`s (text / tool_call_delta / finish).

Product layers subclass it to add provider-specific request flags (e.g. the
edge-llm ``save_system_prompt_kv_cache`` / ``prefix_cache`` extras) and to
resolve the endpoint from their own env / profile. ``httpx`` is imported lazily
inside the backend module, so ``import voxedge`` stays numpy-only; install the
transport with the ``voxedge[llm]`` extra.
"""
from __future__ import annotations

from voxedge.backends.llm.openai_compat import (
    LLMStreamError,
    OpenAICompatBackend,
)

__all__ = ["OpenAICompatBackend", "LLMStreamError"]
