"""voxedge optional capabilities — auxiliary, opt-in voice features.

These are not ASR/TTS backends but small CPU side-capabilities that enrich a
transcript: punctuation restoration (CT-Transformer) and speaker embedding
(CAM++ / 3D-Speaker), both driven through sherpa-onnx.

voxedge convention: **zero module-scope env reads, no file I/O / downloads**.
The model path + thread count are injected at construction; flag gating,
path resolution and model provisioning live in the product layer.
``import sherpa_onnx`` stays lazy so importing these needs no optional extra.
"""

from __future__ import annotations

from voxedge.capabilities.punctuation import PUNCT_MODEL_NAME, Punctuator
from voxedge.capabilities.speaker_embedding import (
    SPEAKER_MODEL_NAME,
    SpeakerEmbedder,
    decode_audio_to_16k_mono,
    embedding_payload,
    encode_embedding,
    pcm16_to_float32,
    resample_linear,
)

__all__ = [
    "PUNCT_MODEL_NAME",
    "Punctuator",
    "SPEAKER_MODEL_NAME",
    "SpeakerEmbedder",
    "decode_audio_to_16k_mono",
    "embedding_payload",
    "encode_embedding",
    "pcm16_to_float32",
    "resample_linear",
]
