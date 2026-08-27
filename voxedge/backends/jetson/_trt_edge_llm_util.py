"""Decoupled helpers for the voxedge TRT-Edge-LLM adapters.

adapted from app/backends/jetson/qwen3_asr.py (deleted; recovered from git
8ef061f~1) + app/core/tts_speakers.py (2026-05-30), dedup after registry
switch.

These are env-free reproductions of the helper functions the production
TRT-Edge-LLM backends imported from ``app.*``. voxedge must not import
``app.*`` (open-core split), so the necessary logic is reproduced here with
ZERO module-scope env reads and ZERO file I/O. The two env-tunable constants
the original energy splitter read (``ASR_ENERGY_SPLIT_RMS`` /
``ASR_ENERGY_MIN_SILENCE_MS``) become explicit parameters with identical
defaults.

The production offline-splitter import path
(``from app.backends.jetson.qwen3_asr import _split_at_silence_vad ...``)
already pointed at a DELETED module — the functions are reproduced here so the
voxedge ASR adapter's long-audio finalize path is self-contained.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# The long-audio splitters used to live here in full. They now live in
# ``voxedge.audio.segment``, which merged this copy with the byte-identical
# algorithm that had grown a second time in ``backends/rk/asr.py``. The names
# below keep this module's existing import surface; the defaults of the shared
# functions are this caller's tuned values, so behaviour is unchanged.
from voxedge.audio.segment import (  # noqa: F401
    DEFAULT_ENERGY_MIN_SILENCE_MS,
    DEFAULT_ENERGY_SPLIT_RMS,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_MS,
    VAD_MAX_SEG_SEC,
    VAD_MIN_SEG_SEC,
    VAD_MIN_SILENCE_MS,
    split_at_silence_energy as _split_at_silence_energy,
    split_at_silence_vad as _split_at_silence_vad,
)


def resolve_speaker_kwargs(
    model_id: str,
    *,
    allow_embedding: bool = True,
    **kwargs: object,
) -> dict[str, object]:
    """Env-free, registry-free speaker kwargs resolver.

    Input priority (first wins), mirroring app/core/tts_speakers.py:
    1. ``speaker_embedding`` — raw float32 bytes (direct voice clone).
    2. ``speaker_id`` — numeric id passed straight through.
    3. ``sid`` — deprecated alias for speaker_id.

    Returns ``{"speaker_embedding": bytes}``, ``{"speaker_id": int}``, or
    ``{}``. ``model_id`` is accepted for signature-compat with the production
    helper but is unused (no registry in voxedge).
    """
    emb = kwargs.get("speaker_embedding")
    if emb is not None:
        if not allow_embedding:
            raise ValueError(
                f"Model {model_id!r} does not support voice clone embeddings"
            )
        return {"speaker_embedding": emb}

    sid = kwargs.get("speaker_id", kwargs.get("sid"))
    if sid is not None:
        return {"speaker_id": int(sid)}

    return {}
