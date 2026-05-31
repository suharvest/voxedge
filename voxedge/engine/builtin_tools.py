"""Engine-neutral built-in tools (Phase 1).

Ported from ``agent/openvoicestream_agent/tools/builtin.py``. Only the
engine-neutral ``time_now`` is carried over — the agent's ``set_mode`` is
app-mode/ModeManager-specific and is intentionally NOT a voxedge core built-in
(spec §3: "set_mode is agent/app-mode-specific and should not be a voxedge
core built-in").

There is no module-level side-effect registration: call
:func:`register_builtins` against a :class:`ToolRegistry` to opt in.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from voxedge.engine.tool_registry import ToolRegistry


def time_now() -> dict[str, Any]:
    """Return the current local time as ISO 8601."""
    return {"now": datetime.now().isoformat()}


def register_builtins(registry: ToolRegistry) -> None:
    """Register the engine-neutral built-in tools against ``registry``."""
    registry.tool(description="Return the current local time as ISO 8601.")(
        time_now
    )


__all__ = ["time_now", "register_builtins"]
