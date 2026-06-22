"""Canonical OpenAI-compatible LLM backend (httpx) — SSE parse contract.

Mirrors the product server's edge-llm regression cases (mocked httpx SSE →
stream_events yields text + tool_call_delta + finish; body shape; param
forwarding; finish_reason=error raises) so the relocation into voxedge is
behaviour-equivalent. The edge-llm cache-flag layering is tested in the product
repo's test_v2v_server_loop.py against the thin subclass.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from voxedge.backends.llm import LLMStreamError, OpenAICompatBackend


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _mock_backend(body_bytes: bytes, captured: dict) -> OpenAICompatBackend:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, content=body_bytes)

    be = OpenAICompatBackend(base_url="http://test/v1", model="m")
    be._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return be


@run_async
async def test_streams_text_events():
    captured: dict = {}
    body = _sse(
        {"choices": [{"delta": {"content": "Hello "}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    )
    be = _mock_backend(body, captured)
    events = [ev async for ev in be.stream_events([{"role": "user", "content": "hi"}])]
    kinds = [(e.kind, e.text or e.finish_reason) for e in events]
    assert ("text", "Hello ") in kinds
    assert ("text", "world") in kinds
    assert ("finish", "stop") in kinds
    await be.aclose()


@run_async
async def test_streams_tool_call_delta():
    captured: dict = {}
    body = _sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1",
             "function": {"name": "wave", "arguments": '{"x":'}}
        ]}, "finish_reason": None}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "1}"}}
        ]}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    be = _mock_backend(body, captured)
    tools = [{"type": "function", "function": {"name": "wave",
              "parameters": {"type": "object", "properties": {}}}}]
    events = [ev async for ev in be.stream_events(
        [{"role": "user", "content": "wave"}], tools=tools)]
    tcs = [e for e in events if e.kind == "tool_call_delta"]
    assert tcs[0].tool_call_id == "call_1"
    assert tcs[0].name == "wave"
    assert tcs[0].arguments == '{"x":'
    assert tcs[1].arguments == "1}"
    assert any(e.kind == "finish" and e.finish_reason == "tool_calls" for e in events)
    assert captured["json"]["tools"] == tools
    assert captured["json"]["stream"] is True
    # Generic base injects NO provider cache flags.
    assert "save_system_prompt_kv_cache" not in captured["json"]
    await be.aclose()


@run_async
async def test_forwards_params_and_extra_body():
    captured: dict = {}
    body = _sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
    be = _mock_backend(body, captured)
    _ = [ev async for ev in be.stream_events(
        [{"role": "user", "content": "hi"}],
        temperature=0.3, max_tokens=64, extra_body={"prefix_cache": True})]
    assert captured["json"]["temperature"] == 0.3
    assert captured["json"]["max_tokens"] == 64
    # extra_body flattened to top-level.
    assert captured["json"]["prefix_cache"] is True
    await be.aclose()


@run_async
async def test_finish_reason_error_raises():
    captured: dict = {}
    body = _sse({"choices": [{"delta": {}, "finish_reason": "error"}]})
    be = _mock_backend(body, captured)
    raised = False
    try:
        _ = [ev async for ev in be.stream_events([{"role": "user", "content": "x"}])]
    except RuntimeError as e:  # LLMStreamError subclasses RuntimeError
        raised = "finish_reason=error" in str(e)
    assert raised
    assert issubclass(LLMStreamError, RuntimeError)
    await be.aclose()


def test_chat_url_resolution():
    assert OpenAICompatBackend("http://h/v1")._chat_url == "http://h/v1/chat/completions"
    assert OpenAICompatBackend("http://h")._chat_url == "http://h/v1/chat/completions"
    assert OpenAICompatBackend(
        "http://h/v1/chat/completions"
    )._chat_url == "http://h/v1/chat/completions"
