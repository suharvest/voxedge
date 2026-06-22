"""OpenAI-compatible streaming chat backend over httpx.

The generic LLM backend behind any OpenAI-compatible ``/v1/chat/completions``
SSE endpoint. It is the canonical home for the request-body assembly + SSE →
:class:`~voxedge.backends.base.LLMEvent` parse loop that product layers used to
re-implement per process.

Design notes:
  * ``httpx`` (pure-Python) is the transport — NOT the ``openai`` SDK — so a
    consumer can drive an edge LLM service without carrying the SDK. The import
    is deferred to construction time, keeping ``import voxedge`` numpy-only;
    install with ``voxedge[llm]``.
  * Provider-specific request flags belong in a subclass override of
    :meth:`_build_body` (e.g. edge-llm's ``save_system_prompt_kv_cache`` /
    ``prefix_cache``), and endpoint resolution from env / profile belongs in the
    product layer — this base reads no environment.
  * No transient-retry / prefix-cache session logic here: those are higher-level
    policies layered by the product/agent (kept where their state lives).
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

from voxedge.backends.base import LLMBackend, LLMEvent


class LLMStreamError(RuntimeError):
    """Raised when an OpenAI-compatible upstream signals failure mid-stream
    (an SSE chunk with ``finish_reason="error"``) while still returning HTTP
    200. Subclasses :class:`RuntimeError` so callers that only catch
    ``RuntimeError`` still see it; by the time it fires the caller has already
    received partial tokens, so a transparent retry would emit duplicates."""


class OpenAICompatBackend(LLMBackend):
    """Streaming OpenAI-compatible chat backend over httpx.

    Satisfies :meth:`LLMBackend.stream_events` — yielding ``text`` /
    ``tool_call_delta`` / ``finish`` events — plus the ``stream`` text-only
    filter the ABC also declares. The endpoint is normalized to
    ``.../v1/chat/completions``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "",
        api_key: str = "",
        default_params: Optional[dict[str, Any]] = None,
        request_timeout_s: float = 60.0,
    ) -> None:
        import httpx  # deferred: keeps `import voxedge` numpy-only

        self._httpx = httpx
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        # Forwarded to every chat call (temperature / max_tokens / top_p ...).
        # The engine also forwards its own per-call ``llm_params``; both merge
        # here, per-call winning.
        self.default_params = dict(default_params or {})
        self.request_timeout_s = float(request_timeout_s)
        self._chat_url = self._resolve_chat_url(self.base_url)
        self._client: Optional[Any] = None  # httpx.AsyncClient

    @staticmethod
    def _resolve_chat_url(base: str) -> str:
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def _ensure_client(self):
        if self._client is None:
            # Send the API key as a Bearer header when one is set, so the
            # generic backend works against auth-requiring OpenAI-compatible
            # servers. An empty key (e.g. a dummy "edge-llm") sends no header.
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
            self._client = self._httpx.AsyncClient(
                timeout=self.request_timeout_s, headers=headers
            )
        return self._client

    def _build_body(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the OpenAI-compatible chat request body.

        Top-level OpenAI fields (model / messages / stream / tools) plus any
        caller params (temperature / max_tokens / ...). An ``extra_body`` dict
        in ``params`` is flattened to top-level keys (the contract several
        OpenAI-compatible edge servers expect). Subclasses override to inject
        provider-specific flags via ``super()._build_body(...)``.
        """
        p = dict(params)
        extra_body = p.pop("extra_body", None) or {}
        body: dict[str, Any] = {
            "model": p.pop("model", self.model),
            "messages": messages,
            "stream": True,
        }
        if tools:
            body["tools"] = tools
        for k, v in p.items():
            body[k] = v
        for k, v in extra_body.items():
            body[k] = v
        return body

    async def stream_events(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict[str, Any]]] = None,
        session: Any = None,  # accepted for caller-parity; unused here
        **kw: Any,
    ) -> AsyncIterator[LLMEvent]:
        """Stream OpenAI-compatible SSE chunks as voxedge ``LLMEvent``s.

        ``kind="text"`` for content deltas, ``kind="tool_call_delta"`` for
        streamed ``delta.tool_calls`` (per OpenAI index), ``kind="finish"`` for
        each ``finish_reason``. A ``finish_reason="error"`` frame raises
        :class:`LLMStreamError`.
        """
        params = {**self.default_params, **kw}
        body = self._build_body(messages, tools, params)
        client = self._ensure_client()
        async with client.stream("POST", self._chat_url, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0]
                delta = choice0.get("delta") or {}
                finish_reason = choice0.get("finish_reason")
                if finish_reason == "error":
                    raise LLMStreamError(
                        "OpenAI-compatible upstream emitted finish_reason=error "
                        "mid-stream"
                    )
                content = delta.get("content")
                if content:
                    yield LLMEvent(kind="text", text=content)
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index")
                    if idx is None:
                        idx = 0
                    fn = tc.get("function") or {}
                    yield LLMEvent(
                        kind="tool_call_delta",
                        tool_call_index=idx,
                        tool_call_id=tc.get("id"),
                        name=fn.get("name"),
                        arguments=fn.get("arguments"),
                    )
                if finish_reason:
                    yield LLMEvent(kind="finish", finish_reason=finish_reason)

    async def stream(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        **kw: Any,
    ) -> AsyncIterator[str]:
        """Back-compat text-only iterator (ABC requirement)."""
        async for ev in self.stream_events(messages, **kw):
            if ev.kind == "text" and ev.text:
                yield ev.text

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover - best effort
                pass
            self._client = None
