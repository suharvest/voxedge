"""Whisper mel front end, in numpy.

Kept free of torch and librosa on purpose: the boards this runs on are disk- and
memory-constrained, and torch alone cost 1.2 GB on the Raspberry Pi. Verified
against ``torch.stft`` at max|diff| ~1e-5 / mean ~1e-7, and the filterbank is
bit-identical to ``librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)``
(max|diff| 0.0) — Rockchip's shipped ``mel_80_filters.txt`` is exactly that
matrix, so it can be loaded from disk instead of computed.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80


def _hann(n: int) -> np.ndarray:
    # torch.hann_window defaults to periodic=True
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def load_mel_filters(path: str | Path) -> np.ndarray:
    return np.loadtxt(str(path), dtype=np.float32).reshape((N_MELS, N_FFT // 2 + 1))


def log_mel(
    audio: np.ndarray,
    filters: np.ndarray,
    window_s: float,
    padding_cutoff_s: float = 0.0,
) -> np.ndarray:
    """Mel spectrogram for one encoder window.

    The waveform is padded to the full window *before* the STFT, which is what
    whisper does. Zero-padding the finished mel instead leaves 0.0 in the tail
    while the mel of digital silence is about -0.58 — that shows the encoder a
    constant it never saw in training. Measured cost of getting this wrong: 2.9
    WER points on 10 s-window English long-form, where tail chunks are mostly
    padding.

    ``padding_cutoff_s`` reproduces Hailo's boundary-hallucination guard: crop
    the waveform to (window - cutoff) first, then pad back out to window, so the
    tail of the window is always silence.
    """
    tgt = int(window_s * SAMPLE_RATE)
    if padding_cutoff_s > 0:
        crop = int((window_s - padding_cutoff_s) * SAMPLE_RATE)
        head = np.zeros(crop, dtype=np.float64)
        n = min(len(audio), crop)
        head[:n] = audio[:n]
        buf = np.zeros(tgt, dtype=np.float64)
        buf[:crop] = head
    else:
        buf = np.zeros(tgt, dtype=np.float64)
        n = min(len(audio), tgt)
        buf[:n] = audio[:n]

    pad = N_FFT // 2
    x = np.pad(buf, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
    idx = np.arange(N_FFT)[None, :] + HOP_LENGTH * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(x[idx] * _hann(N_FFT)[None, :], n=N_FFT, axis=-1)
    mag = (np.abs(spec) ** 2).T[:, :-1]          # the reference drops the last frame
    # numpy 1.26 on Accelerate raises divide-by-zero / overflow / invalid out
    # of this matmul when the whole block is zeros — which is exactly what the
    # preload warmup feeds it. The product is correct; only the FPE flag is
    # spurious, so it is silenced here rather than at the call sites.
    with np.errstate(all="ignore"):
        mel = filters @ mag
    ls = np.log10(np.clip(mel, 1e-10, None))
    ls = np.maximum(ls, ls.max() - 8.0)
    ls = ((ls + 4.0) / 4.0).astype(np.float32)
    return np.ascontiguousarray(ls[:, : int(window_s * 100)])
