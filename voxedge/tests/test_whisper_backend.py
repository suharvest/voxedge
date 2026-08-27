"""Whisper backend: the plumbing around the encoder, with the encoder faked.

The three execution paths need real hardware, so what is checked here is what
is platform-independent and what has actually gone wrong before: the window is
not a free knob, long audio gets cut at silence rather than at a fixed hop, and
a degenerate chunk transcript does not reach the caller.
"""
from __future__ import annotations

import numpy as np
import pytest

from voxedge.backends.base import ASRCapability
from voxedge.backends.whisper import WhisperASR, WhisperASRConfig
from voxedge.backends.whisper.decoder import base64_decode, detokenize
from voxedge.backends.whisper.frontend import log_mel

SR = 16000


class _FakeEncoder:
    def __init__(self, window_s: float = 10.0):
        self.window_s = window_s
        self.calls: list[tuple] = []

    def run(self, mel):
        self.calls.append(mel.shape)
        return np.zeros((1, int(self.window_s * 50), 512), dtype=np.float32)

    def close(self):
        pass


class _FakeDecoder:
    """Returns one canned transcript per call, in order."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.audio_s: list[float] = []

    def decode(self, enc, vocab, language, *, audio_s, max_new=None):
        self.audio_s.append(audio_s)
        return (self._texts.pop(0) if self._texts else ""), [1.0, 1.0]


def _backend(texts, *, window_s=10.0, language="en", padding_cutoff_s=0.0):
    cfg = WhisperASRConfig(
        encoder_kind="rknn", encoder_path="x", decoder_dir="y", vocab_dir="z",
        window_s=window_s, language=language, padding_cutoff_s=padding_cutoff_s,
    )
    be = WhisperASR(cfg)
    be._encoder = _FakeEncoder(window_s)
    be._decoder = _FakeDecoder(texts)
    be._filters = np.zeros((80, 201), dtype=np.float32)
    be._vocab = {}
    return be


def _speech(seconds: float, rng) -> np.ndarray:
    return rng.normal(0, 0.15, int(seconds * SR)).astype(np.float32)


# ── config guards ───────────────────────────────────────────────────────

def test_unsupported_language_is_refused_at_construction():
    # Reporting a language the decoder cannot produce is worse than refusing:
    # the caller gets fluent output in the wrong language and no error.
    with pytest.raises(ValueError, match="not supported"):
        WhisperASRConfig(encoder_kind="rknn", encoder_path="x", decoder_dir="y",
                         vocab_dir="z", language="fr")


def test_padding_cutoff_may_not_consume_the_whole_window():
    with pytest.raises(ValueError, match="no audio"):
        WhisperASRConfig(encoder_kind="hailo", encoder_path="x", decoder_dir="y",
                         vocab_dir="z", window_s=5.0, padding_cutoff_s=5.0)


def test_no_language_id_capability():
    # The language token is forced from config, never detected.
    be = _backend([])
    assert ASRCapability.LANGUAGE_ID not in be.capabilities
    assert {ASRCapability.OFFLINE, ASRCapability.STREAMING} == be.capabilities


# ── segmentation ────────────────────────────────────────────────────────

def test_short_audio_is_one_chunk():
    be = _backend(["hello world"])
    r = be.transcribe_array(_speech(3.0, np.random.default_rng(1)))
    assert r.text == "hello world"
    assert r.meta["chunks"] == 1


def test_long_audio_is_cut_and_no_chunk_exceeds_the_window():
    rng = np.random.default_rng(2)
    # speech / silence / speech / silence / speech — 24 s total
    audio = np.concatenate([
        _speech(6.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32),
        _speech(6.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32),
        _speech(10.8, rng),
    ])
    be = _backend(["a", "b", "c", "d"], window_s=10.0)
    r = be.transcribe_array(audio)
    assert r.meta["chunks"] > 1
    assert max(be._decoder.audio_s) <= 10.0 + 1e-6


def test_cuts_land_in_the_silence():
    # A fixed hop would cut mid-word; the shared splitter prefers the gap.
    rng = np.random.default_rng(3)
    gap_start = 9.0
    audio = np.concatenate([
        _speech(gap_start, rng),
        np.zeros(int(0.8 * SR), dtype=np.float32),
        _speech(9.0, rng),
    ])
    be = _backend(["a", "b", "c"], window_s=10.0)
    be.transcribe_array(audio)
    first = be._decoder.audio_s[0]
    assert gap_start <= first <= gap_start + 0.8


def test_hailo_padding_cutoff_shrinks_the_usable_window():
    # A 5 s HEF with a 1 s boundary guard holds 4 s of audio, not 5.
    be = _backend(["a", "b", "c", "d", "e"], window_s=5.0, padding_cutoff_s=1.0)
    be.transcribe_array(_speech(12.0, np.random.default_rng(4)))
    assert max(be._decoder.audio_s) <= 4.0 + 1e-6


# ── degeneration guards ─────────────────────────────────────────────────

def test_runaway_repetition_inside_one_chunk_is_collapsed():
    be = _backend(["by Llew, " * 40])
    r = be.transcribe_array(_speech(4.0, np.random.default_rng(5)))
    assert r.text.count("Llew") < 5


def test_whole_segments_repeating_are_dropped():
    rng = np.random.default_rng(6)
    audio = np.concatenate([
        _speech(6.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32),
        _speech(6.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32),
        _speech(6.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32),
        _speech(6.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32),
        _speech(6.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32),
        _speech(6.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32),
        _speech(6.0, rng),
    ])
    be = _backend(["6L2s5b2V5LiA5LiL6L+Z5q61"] * 12, window_s=10.0, language="zh")
    r = be.transcribe_array(audio)
    assert r.meta["segments_dropped"] > 0


def test_empty_audio_returns_empty_without_touching_the_encoder():
    be = _backend([])
    r = be.transcribe_array(np.zeros(0, dtype=np.float32))
    assert r.text == "" and r.meta["chunks"] == 0
    assert be._encoder.calls == []


# ── joining ─────────────────────────────────────────────────────────────

def test_chinese_segments_join_without_spaces():
    # The zh path base64-decodes the raw token stream (Rockchip's vocab
    # encoding), so the canned transcripts are base64 too. Both are chosen to
    # be a whole number of 3-byte groups: their decoder returns a single space
    # at the first '=' rather than treating it as padding.
    rng = np.random.default_rng(7)
    audio = np.concatenate([
        _speech(8.0, rng), np.zeros(int(0.6 * SR), dtype=np.float32), _speech(8.0, rng)
    ])
    be = _backend(["5LuK5aSp5aSp5rCU44CC", "5b6I5aW9"], window_s=10.0, language="zh")
    assert be.transcribe_array(audio).text == "今天天气很好"


# ── front end ───────────────────────────────────────────────────────────

def test_mel_pads_the_waveform_not_the_spectrogram():
    # Zero-padding the finished mel leaves 0.0 where the mel of silence is
    # about -0.58 — a constant the encoder never saw in training. Cost of
    # getting this wrong, measured: 2.9 WER points on 10 s-window long-form.
    filters = np.eye(80, 201, dtype=np.float32)
    mel = log_mel(np.zeros(SR, dtype=np.float32), filters, 10.0)
    assert mel.shape == (80, 1000)
    tail = mel[:, 500:]
    assert np.all(tail < -0.1), tail.max()


def test_mel_frame_count_follows_the_window():
    filters = np.eye(80, 201, dtype=np.float32)
    assert log_mel(np.zeros(SR, dtype=np.float32), filters, 5.0).shape[1] == 500
    assert log_mel(np.zeros(SR, dtype=np.float32), filters, 20.0).shape[1] == 2000


# ── tokenizer ───────────────────────────────────────────────────────────

def test_base64_decode_stops_at_padding_and_trims_the_buffer():
    # Rockchip's variant returns a single space at '=' (their zh word break),
    # and the caller must not receive the trailing NULs of the sized buffer.
    assert base64_decode("=") == " "
    assert "\x00" not in base64_decode("5L2g")
    assert base64_decode("") == ""


def test_detokenize_only_base64_decodes_chinese():
    assert detokenize("Ġhello", "en") == " hello"
    assert detokenize("5L2g5aW9", "zh") == "你好"
