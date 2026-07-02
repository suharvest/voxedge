"""Diarization clustering kernel tests — synthetic embeddings, no audio/model.

Builds 2-3 random unit "prototype" vectors (one per speaker), then generates
noisy L2-normalized segments around each. Verifies that OfflineDiarizer can
recover the speaker count and label segments correctly, that OnlineDiarizer
assigns sensibly and relabel() merges split clusters, plus edge cases. Fully
deterministic via a fixed RNG seed.
"""
from __future__ import annotations

import numpy as np
import pytest

from voxedge.capabilities.diarization import (
    OfflineDiarizer,
    OnlineDiarizer,
    SpeakerSegment,
)

DIM = 192
SEED = 1234


def _unit(rng: np.random.Generator, dim: int = DIM) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _prototypes(rng: np.random.Generator, k: int) -> list[np.ndarray]:
    """k well-separated unit prototypes (re-draw until mutually low-cosine)."""
    protos: list[np.ndarray] = []
    while len(protos) < k:
        cand = _unit(rng)
        if all(abs(float(np.dot(cand, p))) < 0.15 for p in protos):
            protos.append(cand)
    return protos


def _noisy(rng: np.random.Generator, proto: np.ndarray, sigma: float = 0.12) -> np.ndarray:
    v = proto + sigma * _unit(rng)
    return v / np.linalg.norm(v)


def _make_session(rng, protos, per_speaker, t0=0.0, dt=1.0):
    """Interleaved (emb, start, end) items + the ground-truth speaker index."""
    items = []
    truth = []
    t = t0
    for spk_idx, proto in enumerate(protos):
        for _ in range(per_speaker):
            emb = _noisy(rng, proto)
            items.append((emb, t, t + dt))
            truth.append(spk_idx)
            t += dt
    return items, truth


def _label_accuracy(segments, items, truth):
    """Best-permutation accuracy: map predicted labels to truth by majority."""
    # Map each item (by start time) to its predicted speaker.
    pred_by_start = {round(s.start, 6): s.speaker for s in segments}
    pred = [pred_by_start[round(it[1], 6)] for it in items]
    # Build contingency, greedily assign predicted label -> dominant truth.
    from collections import Counter, defaultdict

    groups = defaultdict(Counter)
    for p, t in zip(pred, truth):
        groups[p][t] += 1
    mapping = {p: c.most_common(1)[0][0] for p, c in groups.items()}
    correct = sum(1 for p, t in zip(pred, truth) if mapping[p] == t)
    return correct / len(truth)


# ── OfflineDiarizer ──────────────────────────────────────────────────────────

def test_offline_estimates_two_speakers():
    rng = np.random.default_rng(SEED)
    protos = _prototypes(rng, 2)
    items, truth = _make_session(rng, protos, per_speaker=6)
    segs = OfflineDiarizer(min_sim=0.50).cluster(items)  # no num_speakers hint
    n_spk = len({s.speaker for s in segs})
    assert n_spk == 2
    assert _label_accuracy(segs, items, truth) >= 0.9


def test_offline_estimates_three_speakers():
    rng = np.random.default_rng(SEED + 1)
    protos = _prototypes(rng, 3)
    items, truth = _make_session(rng, protos, per_speaker=5)
    segs = OfflineDiarizer(min_sim=0.50).cluster(items)
    n_spk = len({s.speaker for s in segs})
    assert n_spk == 3
    assert _label_accuracy(segs, items, truth) >= 0.9


def test_offline_fixed_num_speakers_labels_correct():
    rng = np.random.default_rng(SEED + 2)
    protos = _prototypes(rng, 3)
    items, truth = _make_session(rng, protos, per_speaker=5)
    segs = OfflineDiarizer().cluster(items, num_speakers=3)
    assert len({s.speaker for s in segs}) == 3
    assert _label_accuracy(segs, items, truth) >= 0.95


def test_offline_labels_deterministic_and_time_ordered():
    rng = np.random.default_rng(SEED + 3)
    protos = _prototypes(rng, 2)
    items, _ = _make_session(rng, protos, per_speaker=4)
    a = OfflineDiarizer().cluster(items, num_speakers=2)
    b = OfflineDiarizer().cluster(items, num_speakers=2)
    assert [s.speaker for s in a] == [s.speaker for s in b]
    # earliest segment is always spk_0; segments sorted by start
    assert a[0].speaker == "spk_0"
    assert all(a[i].start <= a[i + 1].start for i in range(len(a) - 1))


def test_offline_confidence_reasonable():
    rng = np.random.default_rng(SEED + 4)
    protos = _prototypes(rng, 2)
    items, _ = _make_session(rng, protos, per_speaker=5)
    segs = OfflineDiarizer().cluster(items, num_speakers=2)
    assert all(0.5 < s.confidence <= 1.0 for s in segs)


# ── OnlineDiarizer ───────────────────────────────────────────────────────────

def test_online_assigns_two_speakers():
    rng = np.random.default_rng(SEED + 5)
    protos = _prototypes(rng, 2)
    items, truth = _make_session(rng, protos, per_speaker=6)
    diar = OnlineDiarizer(threshold=0.55)
    segs = [diar.assign(emb, s, e) for (emb, s, e) in items]
    assert diar.num_speakers == 2
    assert _label_accuracy(segs, items, truth) >= 0.85


def test_online_relabel_merges_split_clusters():
    rng = np.random.default_rng(SEED + 6)
    protos = _prototypes(rng, 2)
    # Single speaker but feed an outlier first so greedy online may over-split.
    diar = OnlineDiarizer(threshold=0.65)
    items = []
    # one noisy outlier from speaker 0, then tight cluster of speaker 0
    items.append((_noisy(rng, protos[0], sigma=0.45), 0.0, 1.0))
    t = 1.0
    for _ in range(8):
        items.append((_noisy(rng, protos[0], sigma=0.08), t, t + 1.0))
        t += 1.0
    for emb, s, e in items:
        diar.assign(emb, s, e)
    relabeled = diar.relabel()
    assert len(relabeled) == len(items)
    # offline reconciliation should collapse to a single speaker
    assert len({s.speaker for s in relabeled}) == 1


def test_online_relabel_recovers_two_speakers():
    rng = np.random.default_rng(SEED + 7)
    protos = _prototypes(rng, 2)
    items, truth = _make_session(rng, protos, per_speaker=6)
    diar = OnlineDiarizer(threshold=0.55)
    for emb, s, e in items:
        diar.assign(emb, s, e)
    relabeled = diar.relabel()
    assert len({s.speaker for s in relabeled}) == 2
    assert _label_accuracy(relabeled, items, truth) >= 0.9


def test_online_max_speakers_cap():
    rng = np.random.default_rng(SEED + 8)
    protos = _prototypes(rng, 4)
    items, _ = _make_session(rng, protos, per_speaker=3)
    diar = OnlineDiarizer(threshold=0.55, max_speakers=2)
    for emb, s, e in items:
        diar.assign(emb, s, e)
    assert diar.num_speakers <= 2


# ── edge cases ───────────────────────────────────────────────────────────────

def test_offline_empty_input():
    assert OfflineDiarizer().cluster([]) == []


def test_offline_single_segment():
    rng = np.random.default_rng(SEED + 9)
    emb = _unit(rng)
    segs = OfflineDiarizer().cluster([(emb, 0.0, 1.0)])
    assert len(segs) == 1
    assert segs[0].speaker == "spk_0"
    assert segs[0].confidence == 1.0


def test_offline_all_same_speaker():
    rng = np.random.default_rng(SEED + 10)
    protos = _prototypes(rng, 1)
    items, _ = _make_session(rng, protos, per_speaker=8)
    segs = OfflineDiarizer(min_sim=0.50).cluster(items)
    assert len({s.speaker for s in segs}) == 1


def test_online_empty_relabel():
    diar = OnlineDiarizer()
    assert diar.relabel() == []


def test_online_single_assign():
    rng = np.random.default_rng(SEED + 11)
    seg = OnlineDiarizer().assign(_unit(rng), 0.0, 1.0)
    assert isinstance(seg, SpeakerSegment)
    assert seg.speaker == "spk_0"


def test_threshold_edge_two_tight_then_split():
    """A high min_sim splits a borderline pair; a low one merges it."""
    rng = np.random.default_rng(SEED + 12)
    protos = _prototypes(rng, 2)
    items, _ = _make_session(rng, protos, per_speaker=4)
    merged = OfflineDiarizer(min_sim=-1.0).cluster(items)  # merges everything
    split = OfflineDiarizer(min_sim=0.50).cluster(items)
    assert len({s.speaker for s in merged}) == 1
    assert len({s.speaker for s in split}) == 2


def test_offline_handles_bad_embedding_gracefully():
    """A zero/NaN embedding must not raise — never-raise contract."""
    rng = np.random.default_rng(SEED + 13)
    items = [
        (np.zeros(DIM, dtype=np.float32), 0.0, 1.0),
        (_unit(rng), 1.0, 2.0),
    ]
    segs = OfflineDiarizer().cluster(items, num_speakers=2)
    assert len(segs) == 2
