"""Speaker diarization clustering kernel (pure numpy, env-free) — voxedge.

Diarization here is **blind clustering over an already-extracted sequence of
per-segment speaker embeddings** (192-d, L2-normalized CAM++ vectors produced
by ``speaker_embedding.SpeakerEmbedder``). This module never touches audio,
sherpa, onnxruntime or any model — it only consumes ``(embedding, start, end)``
triples and emits speaker labels.

Two entry points, mirroring spec §3:

  * ``OnlineDiarizer``  — streaming, incremental cluster-centroid assignment
    (latency ≈ one VAD segment). Optional ``relabel()`` re-clusters the whole
    accumulated session offline to fix "first-come-first-served" splits.
  * ``OfflineDiarizer`` — whole-session agglomerative clustering (AHC, average
    linkage), auto-estimating the speaker count when not given.

Env-free per voxedge convention (cf. ``speaker_embedding.py``): all tuning
(threshold / ema / max_speakers) is injected at construction; flag gating,
session lifecycle and parameter sourcing stay in the product layer. numpy is
the single dependency (scipy intentionally avoided to keep the core install
light — AHC is implemented inline). Never raises: on any internal failure the
methods fall back to a sane default (everything labelled ``spk_0``) and log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """One time-ordered speaker turn.

    ``speaker`` is an anonymous ``spk_N`` label (identification — mapping to a
    real name — is a separate, opt-in consumer responsibility). ``confidence``
    is the cosine similarity of the segment embedding to its cluster centroid.
    ``embedding`` is optional (default omitted to save bandwidth).
    """

    start: float
    end: float
    speaker: str
    confidence: float
    embedding: Optional[np.ndarray] = None


# ── helpers (numpy only) ─────────────────────────────────────────────────────

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit L2 norm (no-op on a zero vector)."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        return vec / norm
    return vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (robust to non-normalized input)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class OnlineDiarizer:
    """Incremental, streaming speaker assignment via running cluster centroids.

    The speaker count is unknown and grows dynamically. Each incoming segment
    is compared (cosine) against every existing centroid; the best match at or
    above ``threshold`` joins that cluster (centroid EMA-updated then
    re-normalized), otherwise a new ``spk_k`` is spawned (until ``max_speakers``
    is reached, after which segments are forced into the nearest cluster).

    Inputs are assumed L2-normalized (CAM++ output is), but centroids are
    re-normalized after every EMA update so similarities stay well-scaled.
    """

    def __init__(self, threshold: float = 0.55, ema: float = 0.7, max_speakers: int = 10):
        self._threshold = float(threshold)
        self._ema = float(ema)
        self._max_speakers = int(max_speakers)
        self._centroids: List[np.ndarray] = []
        # Accumulated (embedding, start, end) for an optional offline relabel.
        self._history: List[Tuple[np.ndarray, float, float]] = []

    @property
    def num_speakers(self) -> int:
        return len(self._centroids)

    def assign(self, emb: np.ndarray, start: float, end: float) -> SpeakerSegment:
        """Assign one segment to a speaker, updating state. Never raises."""
        try:
            emb = _l2_normalize(emb)
            self._history.append((emb.copy(), float(start), float(end)))

            if not self._centroids:
                self._centroids.append(emb.copy())
                return SpeakerSegment(float(start), float(end), "spk_0", 1.0)

            sims = np.array([_cosine(emb, c) for c in self._centroids], dtype=np.float32)
            best = int(np.argmax(sims))
            best_sim = float(sims[best])

            if best_sim >= self._threshold:
                idx = best
            elif len(self._centroids) < self._max_speakers:
                # Spawn a new cluster for an unrecognized speaker.
                self._centroids.append(emb.copy())
                idx = len(self._centroids) - 1
                return SpeakerSegment(float(start), float(end), f"spk_{idx}", 1.0)
            else:
                # Capacity reached — force into the nearest existing cluster.
                idx = best

            # EMA-update the matched centroid, then re-normalize.
            updated = self._ema * self._centroids[idx] + (1.0 - self._ema) * emb
            self._centroids[idx] = _l2_normalize(updated)
            conf = _cosine(emb, self._centroids[idx])
            return SpeakerSegment(float(start), float(end), f"spk_{idx}", conf)
        except Exception:
            logger.exception("OnlineDiarizer.assign failed; defaulting to spk_0.")
            return SpeakerSegment(float(start), float(end), "spk_0", 0.0)

    def relabel(self, num_speakers: Optional[int] = None) -> List[SpeakerSegment]:
        """Re-cluster the full accumulated session offline for stable labels.

        Online assignment is greedy and can split one speaker across multiple
        ``spk_N`` when a mid-stream segment is misjudged. Running the whole
        history through ``OfflineDiarizer`` reconciles those into globally
        consistent labels. Never raises.
        """
        try:
            if not self._history:
                return []
            return OfflineDiarizer(
                min_sim=self._threshold, max_speakers=self._max_speakers
            ).cluster(self._history, num_speakers=num_speakers)
        except Exception:
            logger.exception("OnlineDiarizer.relabel failed; returning spk_0 fallback.")
            return [
                SpeakerSegment(s, e, "spk_0", 0.0) for (_, s, e) in self._history
            ]


class OfflineDiarizer:
    """Whole-session agglomerative clustering (AHC, average linkage).

    Similarity is the pairwise dot product of L2-normalized embeddings
    (cosine). Clustering is plain linear algebra — no model, no sherpa.

    * ``num_speakers`` given → AHC merges until exactly ``k`` clusters remain.
    * ``num_speakers`` is None → threshold-stopping AHC: merge while the closest
      pair's cosine ≥ ``min_sim`` (i.e. cut the dendrogram at distance
      ``1 - min_sim``). This is chosen over spectral-gap estimation because it
      needs a single, physically meaningful knob (the same same/different-speaker
      cosine threshold used online) and degrades gracefully on tiny inputs.

    Labels are ``spk_0..spk_{k-1}`` ordered by each cluster's earliest start so
    output is deterministic. ``confidence`` is the cosine of the segment to its
    cluster centroid.
    """

    def __init__(self, min_sim: float = 0.50, max_speakers: int = 10):
        self._min_sim = float(min_sim)
        self._max_speakers = int(max_speakers)

    def cluster(
        self,
        items: Sequence[Tuple[np.ndarray, float, float]],
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """Cluster ``items`` = list of ``(embedding, start, end)``. Never raises."""
        try:
            items = list(items)
            if not items:
                return []
            if len(items) == 1:
                emb, start, end = items[0]
                return [SpeakerSegment(float(start), float(end), "spk_0", 1.0)]

            embs = np.stack([_l2_normalize(e) for (e, _, _) in items]).astype(np.float32)
            starts = np.array([float(s) for (_, s, _) in items], dtype=np.float64)
            ends = np.array([float(e) for (_, _, e) in items], dtype=np.float64)

            labels = self._ahc(embs, num_speakers)
            return self._finalize(embs, starts, ends, labels)
        except Exception:
            logger.exception("OfflineDiarizer.cluster failed; defaulting to one speaker.")
            return [
                SpeakerSegment(float(s), float(e), "spk_0", 0.0) for (_, s, e) in items
            ]

    # ── internals ────────────────────────────────────────────────────────────

    def _ahc(self, embs: np.ndarray, num_speakers: Optional[int]) -> np.ndarray:
        """Agglomerative average-linkage clustering → integer cluster ids.

        Distance = ``1 - cosine``. Clusters merge via the Lance-Williams update
        for UPGMA (average linkage). Stops at ``k = num_speakers`` clusters when
        given, otherwise when the closest remaining pair is farther apart than
        ``1 - min_sim``.
        """
        n = embs.shape[0]
        sim = embs @ embs.T  # cosine, rows are unit vectors
        dist = 1.0 - sim
        np.fill_diagonal(dist, np.inf)

        sizes = [1] * n
        active = list(range(n))
        # members[c] = list of original indices in cluster c
        members: List[List[int]] = [[i] for i in range(n)]

        if num_speakers is not None:
            target_k = max(1, min(int(num_speakers), n))
        else:
            target_k = 1  # threshold-stop governs the real cut below
        stop_distance = 1.0 - self._min_sim

        while len(active) > target_k:
            # Find the closest active pair.
            best_i = best_j = -1
            best_d = np.inf
            for ai in range(len(active)):
                ci = active[ai]
                for aj in range(ai + 1, len(active)):
                    cj = active[aj]
                    d = dist[ci, cj]
                    if d < best_d:
                        best_d = d
                        best_i, best_j = ci, cj

            if best_i < 0:
                break
            # Threshold stop (only when speaker count is unknown) — but keep
            # merging past the threshold while we still exceed max_speakers, so
            # the hard ceiling is honoured even on a crowded session.
            if (
                num_speakers is None
                and best_d > stop_distance
                and len(active) <= self._max_speakers
            ):
                break

            # Merge cj into ci via Lance-Williams average-linkage update.
            ni, nj = sizes[best_i], sizes[best_j]
            for ck in active:
                if ck == best_i or ck == best_j:
                    continue
                new_d = (ni * dist[best_i, ck] + nj * dist[best_j, ck]) / (ni + nj)
                dist[best_i, ck] = new_d
                dist[ck, best_i] = new_d
            sizes[best_i] = ni + nj
            members[best_i].extend(members[best_j])
            dist[best_i, best_j] = np.inf
            dist[best_j, best_i] = np.inf
            active.remove(best_j)

        labels = np.zeros(n, dtype=np.int64)
        for new_id, c in enumerate(active):
            for idx in members[c]:
                labels[idx] = new_id
        return labels

    def _finalize(
        self,
        embs: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        labels: np.ndarray,
    ) -> List[SpeakerSegment]:
        """Compute centroids, order labels by earliest start, build segments."""
        uniq = np.unique(labels)
        # Centroid per raw cluster (mean embedding, re-normalized).
        centroids = {}
        earliest = {}
        for c in uniq:
            mask = labels == c
            centroids[c] = _l2_normalize(embs[mask].mean(axis=0))
            earliest[c] = float(starts[mask].min())

        # Deterministic naming: spk_0 = cluster with the earliest start.
        order = sorted(uniq.tolist(), key=lambda c: (earliest[c], int(c)))
        remap = {c: i for i, c in enumerate(order)}

        segments: List[SpeakerSegment] = []
        n = embs.shape[0]
        for i in range(n):
            c = int(labels[i])
            conf = _cosine(embs[i], centroids[c])
            segments.append(
                SpeakerSegment(
                    float(starts[i]),
                    float(ends[i]),
                    f"spk_{remap[c]}",
                    conf,
                )
            )
        # Return in time order for a stable, readable transcript.
        segments.sort(key=lambda s: (s.start, s.end))
        return segments
