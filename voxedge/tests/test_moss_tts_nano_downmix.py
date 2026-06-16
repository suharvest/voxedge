"""MOSS-TTS-Nano stereo->mono downmix.

The MOSS C++ worker always emits interleaved stereo s16le, but the v2v wire
protocol carries only sample_rate (no channel count), so a mono consumer (the
reachy client) reads stereo bytes as mono -> pitch-doubled/echoey audio.
``channels=1`` must therefore ACTUALLY downmix to mono, not merely relabel
metadata (the prior behaviour).
"""

import numpy as np

from voxedge.backends.jetson.moss_tts_nano import (
    MossTtsNanoBackend,
    MossTtsNanoConfig,
)


def test_stereo_to_mono_averages_lr():
    stereo = np.array(
        [[100, 200], [-100, -200], [32767, 32767], [0, 0]], dtype=np.int16
    )
    mono = np.frombuffer(
        MossTtsNanoBackend._stereo_to_mono_s16le(stereo.tobytes()), dtype=np.int16
    )
    assert list(mono) == [150, -150, 32767, 0]


def test_stereo_to_mono_halves_byte_length():
    pcm = np.zeros((10, 2), dtype=np.int16).tobytes()  # 10 stereo frames = 40 bytes
    assert len(MossTtsNanoBackend._stereo_to_mono_s16le(pcm)) == 20  # 10 mono s16


def test_stereo_to_mono_empty():
    assert MossTtsNanoBackend._stereo_to_mono_s16le(b"") == b""


def test_channels_one_enables_downmix():
    assert MossTtsNanoBackend(MossTtsNanoConfig(channels=1))._downmix_to_mono is True


def test_channels_two_keeps_stereo():
    assert MossTtsNanoBackend(MossTtsNanoConfig(channels=2))._downmix_to_mono is False
