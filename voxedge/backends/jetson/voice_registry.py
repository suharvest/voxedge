"""SparkTTS clone voice-registry — ``voice_id`` → :class:`VoiceProfile`.

A VoiceProfile is the host-enrollment artifact (spec §10) produced by
``enroll_voice.py``: a ``<safe_id>.json`` (routing/metadata, source of truth) plus a
sibling ``<safe_id>.npz`` (``global_ids`` int32[32], ``ref_semantic_ids`` int32[Tr],
``d_vector`` f32[1024]). The registry scans a directory of such pairs and serves the
arrays the SparkTTS clone worker needs (``mode:"clone"`` + ``global_ids`` [+ strategy-B
``ref_semantic_ids`` / ``ref_text``]).

Design (spec §4.3):
  * lives in voxedge (backend-adjacent, reusable across products);
  * lazy-loaded + cached, with an explicit ``reload()`` so an OVS register/delete call
    can re-scan after writing a new profile pair;
  * decoupled from the controllable speaker table — clone voices route by a *string*
    ``voice_id`` (recommended ``clone:`` prefix to avoid colliding with the controllable
    ``gender_pitch_speed`` spec form), controllable voices keep their existing path.

Zero env reads at import / construction (the ``trt_edge_llm_tts_env_staleness`` pitfall):
the voices directory is an explicit argument; the product layer resolves it from env.
``numpy`` is imported lazily (only when a profile is actually loaded) so importing this
module never pulls numpy into a non-clone deployment.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ``voice_id`` values carrying this prefix are unambiguously clone voices. The
# registry does not *require* it (any registered id routes to clone), but the
# product/enrollment layer is encouraged to use it so a clone id can never be
# mistaken for a controllable "female_moderate_high" spec.
CLONE_VOICE_PREFIX = "clone:"


@dataclass(frozen=True)
class VoiceProfile:
    """A loaded, materialized clone voice (spec §10)."""

    voice_id: str
    global_ids: list[int]                 # 32 ints ∈ [0, 4095]
    ref_semantic_ids: list[int] = field(default_factory=list)  # strategy B prefix; [] = strategy A
    ref_text: Optional[str] = None        # strategy B transcript; None = strategy A
    sample_rate: int = 16000
    meta: dict = field(default_factory=dict)
    json_path: Optional[str] = None

    def worker_request_fields(self, *, use_ref_semantic: bool = True) -> dict:
        """Return the SparkTTS clone-worker request fields for this voice.

        ``mode:"clone"`` + ``global_ids`` always; ``ref_semantic_ids`` / ``ref_text``
        only when this profile carries them AND ``use_ref_semantic`` (strategy B). The
        worker treats an absent / empty ``ref_semantic_ids`` as strategy A (global-only).
        """
        fields: dict = {"mode": "clone", "global_ids": list(self.global_ids)}
        if use_ref_semantic and self.ref_semantic_ids:
            fields["ref_semantic_ids"] = list(self.ref_semantic_ids)
            if self.ref_text:
                fields["ref_text"] = self.ref_text
        return fields


def _safe_id(voice_id: str) -> str:
    """Filename-safe form of a voice_id (matches enroll_voice.write_profile)."""
    return voice_id.replace(":", "_").replace("/", "_")


def load_voice_profile(json_path: str) -> VoiceProfile:
    """Load one VoiceProfile from its ``.json`` + sibling ``.npz`` (spec §10 §A.load_profile).

    The json is the source of truth for routing/metadata; numeric arrays come from the
    sibling npz named in ``json["npz_file"]`` (falling back to ``<stem>.npz``).
    """
    import numpy as np  # lazy: only when a profile is actually loaded

    with open(json_path, "r", encoding="utf-8") as f:
        j = json.load(f)

    voice_id = j.get("voice_id") or os.path.splitext(os.path.basename(json_path))[0]
    npz_name = j.get("npz_file") or (os.path.splitext(os.path.basename(json_path))[0] + ".npz")
    npz_path = os.path.join(os.path.dirname(json_path), npz_name)

    global_ids: list[int]
    ref_semantic_ids: list[int] = []
    if os.path.exists(npz_path):
        with np.load(npz_path) as npz:
            global_ids = [int(x) for x in npz["global_ids"].reshape(-1).tolist()]
            if "ref_semantic_ids" in npz:
                ref_semantic_ids = [int(x) for x in npz["ref_semantic_ids"].reshape(-1).tolist()]
    else:
        # Fallback: global_ids may be inlined in the json (the spec keeps a copy there).
        global_ids = [int(x) for x in (j.get("global_ids") or [])]
        if not global_ids:
            raise FileNotFoundError(
                f"VoiceProfile {voice_id!r}: npz {npz_path!r} missing and no inline global_ids"
            )

    if len(global_ids) != 32:
        raise ValueError(
            f"VoiceProfile {voice_id!r}: expected 32 global_ids, got {len(global_ids)}"
        )

    return VoiceProfile(
        voice_id=voice_id,
        global_ids=global_ids,
        ref_semantic_ids=ref_semantic_ids,
        ref_text=j.get("ref_text"),
        sample_rate=int(j.get("sample_rate", 16000)),
        meta={k: v for k, v in j.items() if k not in ("global_ids",)},
        json_path=json_path,
    )


class VoiceRegistry:
    """Directory-backed clone voice registry. Lazy scan + cache, thread-safe.

    The directory holds ``<safe_id>.json`` + ``<safe_id>.npz`` pairs (one per voice).
    A profile that fails to load (corrupt npz, wrong global count) is skipped with a
    warning rather than aborting the whole scan — one bad voice never breaks synthesis
    for the others.
    """

    def __init__(self, voices_dir: Optional[str]):
        self._dir = voices_dir
        self._lock = threading.Lock()
        self._cache: Optional[dict[str, VoiceProfile]] = None

    @property
    def voices_dir(self) -> Optional[str]:
        return self._dir

    def _scan(self) -> dict[str, VoiceProfile]:
        out: dict[str, VoiceProfile] = {}
        if not self._dir or not os.path.isdir(self._dir):
            return out
        for name in sorted(os.listdir(self._dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self._dir, name)
            try:
                prof = load_voice_profile(path)
            except Exception as exc:  # one bad profile must not kill the rest
                logger.warning("VoiceRegistry: skipping %s (%s)", path, exc)
                continue
            out[prof.voice_id] = prof
        return out

    def _map(self) -> dict[str, VoiceProfile]:
        with self._lock:
            if self._cache is None:
                self._cache = self._scan()
            return self._cache

    def reload(self) -> int:
        """Re-scan the voices directory. Returns the number of loaded profiles.

        Call after an OVS register/delete writes/removes a profile pair so the next
        synthesis sees it without a process restart.
        """
        with self._lock:
            self._cache = self._scan()
            return len(self._cache)

    def get(self, voice_id: Optional[str]) -> Optional[VoiceProfile]:
        """Return the profile for ``voice_id`` or ``None`` if not a registered clone voice."""
        if not voice_id or not isinstance(voice_id, str):
            return None
        return self._map().get(voice_id.strip())

    def contains(self, voice_id: Optional[str]) -> bool:
        return self.get(voice_id) is not None

    def list_voices(self) -> list[dict]:
        """List registered clone voices (id + lightweight metadata, no big arrays)."""
        items: list[dict] = []
        for vid, prof in sorted(self._map().items()):
            items.append({
                "voice_id": vid,
                "type": "clone",
                "sample_rate": prof.sample_rate,
                "has_ref_semantic": bool(prof.ref_semantic_ids),
                "ref_semantic_len": len(prof.ref_semantic_ids),
                "ref_text": prof.ref_text,
                "source_meta": prof.meta.get("source_meta"),
            })
        return items

    def profile_paths(self, voice_id: str) -> tuple[Optional[str], Optional[str]]:
        """Return (json_path, npz_path) for ``voice_id`` (whether or not currently loaded)."""
        if not self._dir:
            return None, None
        safe = _safe_id(voice_id)
        jpath = os.path.join(self._dir, safe + ".json")
        npath = os.path.join(self._dir, safe + ".npz")
        return jpath, npath
