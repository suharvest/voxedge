"""Audio DSP utilities for voxedge (pure-numpy, no ffmpeg/librosa hard dep)."""
from voxedge.audio.rate import (
    TTSRateShifter,
    apply_pcm_rate_pitch,
    apply_wav_rate_pitch,
    pitch_shift_wsola,
    time_stretch_wsola,
)

__all__ = [
    "time_stretch_wsola",
    "pitch_shift_wsola",
    "TTSRateShifter",
    "apply_wav_rate_pitch",
    "apply_pcm_rate_pitch",
]
