"""voxedge — edge-native real-time voice conversation library.

Phase 1a: pure-Python foundation. Core stays importable with no CUDA / torch /
tensorrt — heavy backends live behind optional extras.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Read from installed package metadata rather than a literal: the literal was
# written once and never maintained, so it reported 0.0.5a0 out of a 0.0.6a0
# install — and a version string that lies is worse than no version string when
# you are trying to work out which fix an image actually has.
try:
    __version__ = _pkg_version("voxedge")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
