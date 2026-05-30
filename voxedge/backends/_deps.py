"""Friendly optional-dependency checks for voxedge backend extras.

The heavy backend adapters (jetson / rk / sherpa) import their native runtime
packages LAZILY (inside ``__init__`` / ``preload`` / per-call methods) so the
core stays pip-installable on a CUDA-less, NPU-less dev box (see pyproject
``[project.optional-dependencies]``). The flip side: when a user selects a
backend whose extra was never installed, the lazy import previously surfaced as
a bare, opaque ``ModuleNotFoundError: No module named 'sherpa_onnx'`` — with no
hint about which extra to install.

This module centralises a single ``require()`` helper that turns a missing
package into a clear, actionable ``ImportError`` naming both the missing
distribution and the exact ``pip install voxedge[<extra>]`` incantation. Each
backend's lazy import site wraps its import in ``require(...)`` (or calls the
convenience ``_check_*`` helpers) so the failure mode is explicit.

This file imports nothing heavy itself — it is pure stdlib and safe to import
from the pure-Python core.
"""

from __future__ import annotations

import importlib
from typing import Optional


def require(
    module: str,
    *,
    extra: str,
    package: Optional[str] = None,
):
    """Import ``module`` or raise a clear, actionable ImportError.

    Parameters
    ----------
    module:
        The importable module name (e.g. ``"sherpa_onnx"``).
    extra:
        The voxedge optional-dependency extra that provides it
        (e.g. ``"sherpa"`` → ``pip install voxedge[sherpa]``).
    package:
        The pip distribution name when it differs from ``module``
        (e.g. module ``cuda`` ships in ``cuda-python``). Used only to make the
        error message accurate; defaults to ``module``.

    Returns
    -------
    The imported module object.

    Raises
    ------
    ImportError
        With a message naming the missing package + the install command.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        dist = package or module
        raise ImportError(
            f"voxedge[{extra}] requires {dist!r} (module {module!r}), which is "
            f"not installed. Install the extra with: "
            f"pip install 'voxedge[{extra}]'"
        ) from exc


def require_all(specs, *, extra: str) -> None:
    """Check several modules up front; raise on the first that is missing.

    ``specs`` is an iterable of either ``module`` strings or
    ``(module, package)`` tuples. Use this in a backend ``preload()`` / first
    use to fail fast with a friendly message before any partial work.
    """
    for spec in specs:
        if isinstance(spec, (tuple, list)):
            module, package = spec[0], spec[1]
        else:
            module, package = spec, None
        require(module, extra=extra, package=package)


# ── convenience wrappers per extra ───────────────────────────────────────────


def check_sherpa_deps() -> None:
    """Fail fast with a friendly error if the ``sherpa`` extra is missing."""
    require_all([("sherpa_onnx", "sherpa-onnx")], extra="sherpa")


def check_rk_deps() -> None:
    """Fail fast with a friendly error if the ``rk`` extra is missing."""
    require_all([("rkvoice_stream", "rkvoice-stream")], extra="rk")
