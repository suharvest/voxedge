"""Speaker-embedding extraction (CAM++ / 3D-Speaker via sherpa-onnx) — voxedge.

Env-free per voxedge convention: ``model_path`` + ``num_threads`` are injected
at construction. Flag gating (OVS_SPEAKER_EMB), path resolution and download
stay in the product layer.

Emits the raw embedding only; identity/matching is the consumer's job. The
kaldi-native-fbank front-end is driven through sherpa-onnx's
``SpeakerEmbeddingExtractor`` so enrollment and query embeddings stay
comparable (only the identical extractor guarantees that). CPU only.
``import sherpa_onnx`` is lazy. The stateless helpers (encode / payload /
decode / resample) use only numpy + stdlib so they work on every image.
"""

from __future__ import annotations

import base64
import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Stable identifier surfaced in payloads so consumers can detect a model swap.
SPEAKER_MODEL_NAME = "campplus_sv_zh_en_3dspeaker"

_TARGET_SR = 16000


class SpeakerEmbedder:
    """Lazy, thread-safe wrapper around ``SpeakerEmbeddingExtractor``.

    Loads once on first use; sticky on hard failure. ``compute`` returns an
    L2-normalized float32 vector (shape [dim]) or ``None``; never raises.
    """

    def __init__(self, model_path: str, num_threads: int = 2):
        self._model_path = model_path
        self._num_threads = num_threads
        self._extractor = None
        self._dim = 0
        self._lock = threading.Lock()
        self._failed = False

    def _ensure(self):
        if self._extractor is not None:
            return self._extractor
        if self._failed:
            return None
        with self._lock:
            if self._extractor is not None:
                return self._extractor
            if self._failed:
                return None
            try:
                import sherpa_onnx

                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=self._model_path,
                    num_threads=self._num_threads,
                    provider="cpu",
                    debug=False,
                )
                ext = sherpa_onnx.SpeakerEmbeddingExtractor(config)
                self._extractor = ext
                self._dim = int(ext.dim)
                logger.info(
                    "SpeakerEmbedder loaded (%s, dim=%d, threads=%d).",
                    self._model_path, self._dim, self._num_threads,
                )
            except Exception:
                self._failed = True
                logger.exception("Failed to load speaker model; disabled.")
                return None
        return self._extractor

    def ready(self) -> bool:
        return self._ensure() is not None

    @property
    def dim(self) -> int:
        self._ensure()
        return self._dim

    def compute(self, samples: np.ndarray, sample_rate: int):
        """Embedding for one utterance. ``samples``: mono float32 in [-1, 1].
        Returns an L2-normalized float32 vector or None. Never raises.
        """
        ext = self._ensure()
        if ext is None:
            return None
        if samples is None or len(samples) == 0:
            return None
        try:
            samples = np.ascontiguousarray(samples, dtype=np.float32)
            stream = ext.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()
            if not ext.is_ready(stream):
                return None  # too little audio for the front-end
            emb = np.array(ext.compute(stream), dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception:
            logger.exception("compute_embedding failed.")
            return None


# ── stateless helpers (numpy + stdlib only) ─────────────────────────────────

def encode_embedding(emb: np.ndarray) -> str:
    """Little-endian float32 bytes, base64 (consumer: np.frombuffer(.., '<f4'))."""
    return base64.b64encode(np.asarray(emb, dtype="<f4").tobytes()).decode("ascii")


def embedding_payload(emb: np.ndarray) -> dict:
    """Cross-service contract fields for a final payload."""
    return {
        "speaker_embedding": encode_embedding(emb),
        "embedding_model": SPEAKER_MODEL_NAME,
        "dim": int(len(emb)),
        "normalized": True,
    }


def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Raw int16 little-endian PCM bytes → float32 in [-1, 1]."""
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0


def resample_linear(samples: np.ndarray, src_sr: int, target_sr: int = _TARGET_SR) -> np.ndarray:
    """Resample mono float32 with linear interpolation (dependency-free).

    Good enough for speaker embedding (CAM++ is robust to mild resampling);
    enrollment and query stay comparable as long as both use this path.
    """
    if src_sr == target_sr or len(samples) == 0:
        return samples
    n_out = int(round(len(samples) * target_sr / src_sr))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def decode_audio_to_16k_mono(data: bytes, fallback_sr: int = _TARGET_SR) -> np.ndarray:
    """Decode an uploaded audio blob to mono float32 @ 16 kHz.

    PCM16 WAV (RIFF, via stdlib ``wave`` — no soundfile/scipy dep) or, if not a
    parseable WAV, raw little-endian int16 PCM at ``fallback_sr``.
    """
    import io
    import wave

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        if sampwidth != 2:
            raise wave.Error("non-pcm16 wav")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        return resample_linear(samples, src_sr)
    except (wave.Error, EOFError, ValueError):
        samples = pcm16_to_float32(data)
        return resample_linear(samples, fallback_sr)
