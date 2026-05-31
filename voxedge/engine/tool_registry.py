"""Engine-owned tool registry + ``@tool`` decorator (Phase 1).

Ported from ``agent/openvoicestream_agent/tools/registry.py`` (see
docs/specs/tool-calling-engine-migration.md §2/§3). Builds OpenAI-style
``tools[]`` schemas from Python type hints and dispatches function calls
(sync or async) with per-tool timeout + error isolation. Designed for local,
in-process tools — every entry is trusted code in the same Python process
(no sandboxing, no MCP).

Changes vs the agent source:
  * ``ctx`` injected at dispatch is a :class:`ToolContext` dataclass
    (session_id / conversation / remote_send) instead of the agent's
    ``ToolCallCtx`` — no agent / app coupling.
  * Each :class:`Tool` carries a ``dispatch_mode`` ("local" | "remote").
    Phase 1 implements only the ``local`` path; the ``remote`` wire dispatch
    (``/v2v`` tool_call/tool_result frames) is Phase 2 (spec §4 Mode B).
  * No module-level ``default_registry`` side-effect import of builtins — the
    engine owns its registry instance explicitly.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import types
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)

DispatchMode = Literal["local", "remote"]


@dataclass
class ToolContext:
    """Per-turn context injected into a tool handler that declares ``ctx``.

    Engine-neutral replacement for the agent's ``ToolCallCtx`` (which carried
    app-mode/session objects). Tools that need engine state declare
    ``ctx: ToolContext`` (or just ``ctx``) in their signature; the registry
    injects this at dispatch time and it is NOT part of the LLM-visible schema.

    Fields:
      * ``session_id``   — opaque per-connection id (transport/session key).
      * ``conversation`` — the voxedge ``Session`` driving this turn (or any
        conversation-history object), for tools that inspect/mutate dialog.
      * ``remote_send``  — Phase 2 hook: an awaitable used by remote-dispatch
        proxy handlers to push a ``tool_call`` frame to the device client and
        await a correlated ``tool_result`` (spec §4 Mode B). ``None`` for the
        Phase 1 local path.
    """

    session_id: Optional[str] = None
    conversation: Any = None
    remote_send: Optional[Callable[..., Any]] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema (OpenAI-style)
    fn: Callable[..., Any]
    timeout_s: float = 10.0
    # Short verbal acknowledgement spoken via TTS the moment the tool starts
    # (before its result is appended), for tools whose physical side-effect
    # takes noticeable time. Empty = no preamble. (VoiceArm depends on this.)
    preamble_text: str = ""
    # Fixed verbal acknowledgement spoken on successful completion — used when
    # ``response_mode == "template"`` (skip LLM round 2) or as an optional
    # post-dispatch confirmation. Empty = none.
    completion_text: str = ""
    # How the runner sequences LLM round 2 / TTS after dispatch:
    #   * "await"    — (default) dispatch, wait for result, run LLM round 2.
    #   * "parallel" — dispatch returns fast (~200ms stub), LLM round 2 runs
    #                  while the physical side-effect overlaps the spoken ack.
    #   * "template" — skip LLM round 2; speak ``completion_text`` directly.
    response_mode: str = "await"
    # Phase 1 implements "local" only; "remote" proxies over /v2v (Phase 2).
    dispatch_mode: DispatchMode = "local"


def _py_type_to_schema(t: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON Schema fragment.

    Supports ``str`` / ``int`` / ``float`` / ``bool`` / ``list`` / ``dict`` /
    ``Literal[...]`` / ``Optional[T]`` / ``T | None``. Unknown → string.
    """
    origin = get_origin(t)
    args = get_args(t)

    if origin is Literal:
        sample = args[0]
        if isinstance(sample, bool):
            jtype = "boolean"
        elif isinstance(sample, int):
            jtype = "integer"
        elif isinstance(sample, float):
            jtype = "number"
        else:
            jtype = "string"
        return {"type": jtype, "enum": list(args)}

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _py_type_to_schema(non_none[0])
        return {"type": "string"}

    if origin in (list, tuple, set, frozenset):
        item_schema = _py_type_to_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_schema}
    if origin is dict:
        return {"type": "object"}

    if t is str:
        return {"type": "string"}
    if t is bool:
        return {"type": "boolean"}
    if t is int:
        return {"type": "integer"}
    if t is float:
        return {"type": "number"}
    if t is list:
        return {"type": "array"}
    if t is dict:
        return {"type": "object"}

    return {"type": "string"}


class ToolRegistry:
    """Holds registered tools, exports their OpenAI schemas, and dispatches
    calls. The engine owns one instance; tests construct their own."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(
        self,
        *,
        name: str | None = None,
        description: str = "",
        timeout_s: float = 10.0,
        preamble_text: str = "",
        completion_text: str = "",
        response_mode: str = "await",
        dispatch_mode: DispatchMode = "local",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register ``fn`` as a tool.

        Parameter schema is built from type hints, excluding ``ctx`` (injected
        at dispatch, not LLM-visible). Description defaults to the docstring."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            sig = inspect.signature(fn)
            try:
                hints = get_type_hints(fn)
            except Exception:  # pragma: no cover - defensive
                hints = {}
            props: dict[str, Any] = {}
            required: list[str] = []
            for pname, param in sig.parameters.items():
                if pname == "ctx":
                    continue
                t = hints.get(pname, str)
                props[pname] = _py_type_to_schema(t)
                if param.default is inspect.Parameter.empty:
                    required.append(pname)
            params: dict[str, Any] = {"type": "object", "properties": props}
            if required:
                params["required"] = required
            tname = name or fn.__name__
            self._tools[tname] = Tool(
                name=tname,
                description=description or (fn.__doc__ or "").strip(),
                parameters=params,
                fn=fn,
                timeout_s=timeout_s,
                preamble_text=preamble_text,
                completion_text=completion_text,
                response_mode=response_mode,
                dispatch_mode=dispatch_mode,
            )
            return fn

        return deco

    def register(
        self,
        name: str,
        schema: dict,
        handler: Callable[..., Any],
        *,
        timeout_s: float = 10.0,
        preamble_text: str = "",
        completion_text: str = "",
        response_mode: str = "await",
        dispatch_mode: DispatchMode = "local",
    ) -> None:
        """Programmatic (non-decorator) registration (spec §2 ``register_tool``).

        ``schema`` is the OpenAI-style ``parameters`` JSON Schema for the
        handler (the ``{"type":"object","properties":{...}}`` object — NOT the
        full ``{"type":"function","function":{...}}`` wrapper). ``description``
        is taken from the handler docstring when absent from the schema.
        """
        description = schema.get("description", "") if isinstance(schema, dict) else ""
        if not description:
            description = (getattr(handler, "__doc__", "") or "").strip()
        parameters = schema
        if isinstance(schema, dict) and "properties" not in schema:
            # Tolerate a full function-wrapper or a bare description dict.
            fn_block = schema.get("function") if "function" in schema else None
            if isinstance(fn_block, dict):
                parameters = fn_block.get("parameters", {"type": "object", "properties": {}})
                description = description or fn_block.get("description", "")
            else:
                parameters = {"type": "object", "properties": {}}
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            fn=handler,
            timeout_s=timeout_s,
            preamble_text=preamble_text,
            completion_text=completion_text,
            response_mode=response_mode,
            dispatch_mode=dispatch_mode,
        )

    def list_openai_tools(
        self, allow: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return tools[] in OpenAI's chat-completions format. ``allow``
        filters by name; ``None`` exposes everything registered."""
        out: list[dict[str, Any]] = []
        for tname, t in self._tools.items():
            if allow is not None and tname not in allow:
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            })
        return out

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def unregister(self, name: str) -> bool:
        """Remove a tool. Idempotent (unknown name → False)."""
        return self._tools.pop(name, None) is not None

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any] | None,
        ctx: Any,
    ) -> dict[str, Any]:
        """Invoke the named tool with ``arguments``.

        Always returns a JSON-serialisable dict. On error returns
        ``{"success": False, "error": str}`` so the LLM can self-recover
        rather than crashing the voice loop. Sync and async handlers both
        supported; coroutine results are awaited with the per-tool timeout."""
        t = self._tools.get(name)
        if t is None:
            return {"success": False, "error": f"unknown tool: {name}"}
        allowed = set(t.parameters.get("properties", {}).keys())
        clean: dict[str, Any] = {
            k: v for k, v in (arguments or {}).items() if k in allowed
        }
        try:
            if "ctx" in inspect.signature(t.fn).parameters:
                clean["ctx"] = ctx
            result = t.fn(**clean)
            if inspect.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=t.timeout_s)
            if isinstance(result, dict):
                return result
            return {"value": result}
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"tool {name} timed out after {t.timeout_s}s",
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("tool %s raised %r", name, e)
            return {"success": False, "error": str(e)}


def register_tool(
    registry: ToolRegistry,
    name: str,
    schema: dict,
    handler: Callable[..., Any],
    *,
    timeout_s: float = 10.0,
    preamble_text: str = "",
    completion_text: str = "",
    response_mode: str = "await",
    dispatch_mode: DispatchMode = "local",
) -> None:
    """Module-level convenience wrapper around :meth:`ToolRegistry.register`
    (spec §2 engine-level tool registration API)."""
    registry.register(
        name,
        schema,
        handler,
        timeout_s=timeout_s,
        preamble_text=preamble_text,
        completion_text=completion_text,
        response_mode=response_mode,
        dispatch_mode=dispatch_mode,
    )


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "register_tool",
    "DispatchMode",
]
