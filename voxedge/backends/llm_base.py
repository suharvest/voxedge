"""LLM backend ABC + streaming delta — canonical engine-facing contract.

Phase 1 of the tool-calling migration (docs/specs/tool-calling-engine-migration.md
§2/§3). The voxedge tool runner needs the richer ``stream_events`` shape
(text + tool_call deltas + finish), which already exists on
:class:`voxedge.backends.base.LLMBackend` / :class:`LLMEvent`.

This module is the single import surface the engine's tool loop targets. It
re-exports the canonical ``LLMBackend`` / ``LLMEvent`` (so all existing
backends — Mock, EdgeLLM, etc. — satisfy it unchanged) and adds ``LLMDelta``
as an explicit alias of ``LLMEvent``: one streaming unit carrying either
incremental ``text`` (``content`` in OpenAI terms) or a ``tool_call_delta``.

voxedge does NOT import the agent package — this is a copy of the agent's
``llm/base.py`` contract (agent/openvoicestream_agent/llm/base.py:9-118) kept
local so voxedge core stays free of the agent's openai/httpx deps.
"""
from __future__ import annotations

from voxedge.backends.base import LLMBackend, LLMEvent

# ``LLMDelta`` is the spec §2 name for one streaming output unit. It is the
# same dataclass as ``LLMEvent`` (kind ∈ {"text","tool_call_delta","finish"};
# ``text`` is the assistant ``content`` delta; ``tool_call_*`` carry partial
# tool-call info accumulated per ``tool_call_index``). Aliased rather than
# re-defined so isinstance checks and existing backends interoperate.
LLMDelta = LLMEvent

__all__ = ["LLMBackend", "LLMEvent", "LLMDelta"]
