"""Split long audio at silence, so a fixed-context decoder never sees more than
it can handle.

Three backends hit the same wall for different reasons and had each grown their
own copy of this: Qwen3-ASR on TRT-Edge-LLM emits a premature ``。``+EOS after
~6.5 s of continuous speech, RK's Qwen3-on-RKLLM has a 512-token decoder context
plus a sliding window that carries garbage from one chunk into the next, and
Whisper's encoder window is fixed at compile time. The two pre-existing copies
(``backends/jetson/_trt_edge_llm_util.py``, ``backends/rk/asr.py``) were the same
algorithm differing only in four tuned constants, which are parameters here.
Every caller passes its own values, so behaviour is unchanged from before the
merge.

``_split_at_silence_vad``'s ``webrtcvad`` import stays lazy; callers are
expected to catch ``ImportError`` and fall back to the energy splitter, which
needs nothing but numpy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Defaults reproduce the TRT-Edge-LLM caller. RK passes its own.
VAD_MAX_SEG_SEC = 4.5        # conservative — leaves margin below the Bug A boundary
VAD_MIN_SEG_SEC = 0.5        # allow finer splits when silence is available
VAD_FRAME_MS = 20            # webrtcvad frame size (10/20/30 supported)
VAD_AGGRESSIVENESS = 2       # 0-3; 2 = balanced for mixed noise conditions
VAD_MIN_SILENCE_MS = 150     # minimum silence run to count as a cut candidate

DEFAULT_ENERGY_SPLIT_RMS = 0.003
DEFAULT_ENERGY_MIN_SILENCE_MS = 80


def _pick_cuts(
    audio: np.ndarray,
    cand: np.ndarray,
    *,
    max_seg: int,
    min_seg: int,
    min_frag: int,
    min_tail: int,
) -> list[np.ndarray]:
    """Greedy cut selection shared by both splitters.

    Walks forward taking the silence point closest to ``max_seg`` from the last
    cut, and falls back to a hard cut at ``max_seg`` when the window holds no
    silence. Fragments shorter than ``min_frag`` are merged back into the
    previous segment: a sub-second fragment is what makes these models bail out
    to their own instruction suffix instead of transcribing.

    That last pass means a returned segment can be LONGER than ``max_seg`` —
    up to ``max_seg + min_tail``. For a decoder that merely degrades on
    over-long input this is the right trade. A caller whose input length is a
    hard constraint (a fixed encoder window) must cap the result itself.
    """
    cuts = [0]
    while len(audio) - cuts[-1] > max_seg:
        target = cuts[-1] + max_seg
        lo = cuts[-1] + min_seg
        mask = (cand >= lo) & (cand <= target)
        cuts.append(int(cand[mask][np.argmax(cand[mask])]) if mask.any() else int(target))
    cuts.append(len(audio))

    i = 1
    while i < len(cuts) - 1:
        if (cuts[i + 1] - cuts[i]) < min_frag:
            cuts.pop(i)
        else:
            i += 1
    while len(cuts) >= 3 and (cuts[-1] - cuts[-2]) < min_tail:
        cuts.pop(-2)
    return [audio[cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)]


def split_at_silence_energy(
    audio: np.ndarray,
    sr: int = 16000,
    *,
    split_rms: float = DEFAULT_ENERGY_SPLIT_RMS,
    min_silence_ms: int = DEFAULT_ENERGY_MIN_SILENCE_MS,
    max_seg_s: float = VAD_MAX_SEG_SEC,
    min_seg_s: float = VAD_MIN_SEG_SEC,
    frame_ms: int = VAD_FRAME_MS,
    min_frag_s: float = 1.0,
    min_tail_s: float = 2.0,
) -> list[np.ndarray]:
    """Dependency-free splitter: frame RMS to find silence gaps."""
    # round(), not int(): a caller computing max_seg_s as a difference of
    # floats (a window minus a boundary cutoff) lands just under the integer,
    # and truncating then splits an utterance exactly one segment long.
    max_seg = round(max_seg_s * sr)
    min_seg = round(min_seg_s * sr)
    if len(audio) <= max_seg:
        return [audio]

    frame_len = int(frame_ms * sr / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [audio]

    framed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    is_silence = rms < split_rms
    min_run = max(1, int(min_silence_ms) // frame_ms)

    cut_candidates: list[int] = []
    run_start: Optional[int] = None
    for i, silent in enumerate(is_silence):
        if silent:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                cut_candidates.append(((run_start + i) // 2) * frame_len)
            run_start = None
    if run_start is not None and n_frames - run_start >= min_run:
        cut_candidates.append(((run_start + n_frames) // 2) * frame_len)

    return _pick_cuts(
        audio,
        np.array(cut_candidates, dtype=np.int64),
        max_seg=max_seg,
        min_seg=min_seg,
        min_frag=round(min_frag_s * sr),
        min_tail=round(min_tail_s * sr),
    )


def split_at_silence_vad(
    audio: np.ndarray,
    sr: int = 16000,
    *,
    max_seg_s: float = VAD_MAX_SEG_SEC,
    min_seg_s: float = VAD_MIN_SEG_SEC,
    frame_ms: int = VAD_FRAME_MS,
    aggressiveness: int = VAD_AGGRESSIVENESS,
    min_silence_ms: int = VAD_MIN_SILENCE_MS,
    min_frag_s: float = 1.0,
    min_tail_s: float = 2.0,
) -> list[np.ndarray]:
    """Same cut selection, with webrtcvad deciding what counts as silence."""
    import webrtcvad

    # round(), not int(): a caller computing max_seg_s as a difference of
    # floats (a window minus a boundary cutoff) lands just under the integer,
    # and truncating then splits an utterance exactly one segment long.
    max_seg = round(max_seg_s * sr)
    min_seg = round(min_seg_s * sr)
    if len(audio) <= max_seg:
        return [audio]

    frame_len = int(frame_ms * sr / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [audio]

    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    vad = webrtcvad.Vad(aggressiveness)
    is_silence = np.array(
        [
            not vad.is_speech(
                pcm16[i * frame_len : (i + 1) * frame_len].tobytes(), sr
            )
            for i in range(n_frames)
        ]
    )
    min_run = max(1, int(min_silence_ms) // frame_ms)

    cut_candidates: list[int] = []
    run_start: Optional[int] = None
    for i, silent in enumerate(is_silence):
        if silent:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= min_run:
                cut_candidates.append(((run_start + i) // 2) * frame_len)
            run_start = None
    if run_start is not None and n_frames - run_start >= min_run:
        cut_candidates.append(((run_start + n_frames) // 2) * frame_len)

    return _pick_cuts(
        audio,
        np.array(cut_candidates, dtype=np.int64),
        max_seg=max_seg,
        min_seg=min_seg,
        min_frag=round(min_frag_s * sr),
        min_tail=round(min_tail_s * sr),
    )
