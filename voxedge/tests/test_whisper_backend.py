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


def test_legacy_decoder_is_the_default_and_trt_requires_both_plans():
    cfg = WhisperASRConfig(
        encoder_kind="tensorrt", encoder_path="x", decoder_dir="y", vocab_dir="z"
    )
    assert cfg.decoder_kind == "onnx_cpu"
    with pytest.raises(ValueError, match="prefill and step"):
        WhisperASRConfig(
            encoder_kind="tensorrt", encoder_path="x", decoder_dir="y", vocab_dir="z",
            decoder_kind="tensorrt", decoder_prefill_path="prefill.plan",
        )


@pytest.mark.parametrize("cutoff", [5.0, 4.99999, 4.95])
def test_a_cutoff_that_leaves_no_usable_audio_is_refused(cutoff):
    """Checked in SAMPLES, not seconds.

    4.99999 against a 5 s window passes a float "less than" and still leaves
    zero samples, which divides by zero when segments are capped to the window.
    """
    with pytest.raises(ValueError, match="usable audio|no audio"):
        WhisperASRConfig(encoder_kind="hailo", encoder_path="x", decoder_dir="y",
                         vocab_dir="z", window_s=5.0, padding_cutoff_s=cutoff)


@pytest.mark.parametrize("cap", [-1, 0])
def test_a_token_cap_below_one_is_refused(cap):
    """`range(-1)` is empty, so the greedy loop never runs and even a valid
    prefill argmax is discarded — the utterance comes back empty."""
    with pytest.raises(ValueError, match="max_new_tokens"):
        WhisperASRConfig(encoder_kind="rknn", encoder_path="x", decoder_dir="y",
                         vocab_dir="z", max_new_tokens=cap)


def test_the_first_timestamp_token_is_not_text():
    """TIMESTAMP_BEGIN is the FIRST timestamp token; an inclusive comparison
    emitted `<|0.00|>` into the transcript as literal text."""
    from voxedge.backends.whisper.decoder import EOT, TIMESTAMP_BEGIN, OnnxKVDecoder

    class _Session:
        def __init__(self, outs): self._outs = outs
        def get_outputs(self): return [type("O", (), {"name": "logits"})()]
        def run(self, _, feed): return [self._outs.pop(0)]

    def _logits(token):
        row = np.full((1, 1, 51865), -1e9, dtype=np.float32)
        row[0, -1, token] = 1.0
        return row

    dec = OnnxKVDecoder.__new__(OnnxKVDecoder)
    dec._init = _Session([_logits(TIMESTAMP_BEGIN)])
    dec._past = _Session([_logits(EOT)])
    dec._past_inputs = {"input_ids"}
    text, _ = dec.decode(np.zeros((1, 10, 8), dtype=np.float32),
                         {str(TIMESTAMP_BEGIN): "<|0.00|>"}, "en", audio_s=1.0)
    assert text == ""


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


def test_trt_decoder_path_consumes_borrowed_device_hidden_without_d2h():
    cfg = WhisperASRConfig(
        encoder_kind="tensorrt", encoder_path="x", decoder_dir="y", vocab_dir="z",
        decoder_kind="tensorrt", decoder_prefill_path="prefill.plan",
        decoder_step_path="step.plan",
    )
    be = WhisperASR(cfg)

    class _DeviceEncoder:
        def run(self, _mel):
            raise AssertionError("GPU decoder path must not call encoder.run/D2H")

        def run_device(self, _mel):
            return (1234, (1, 1500, 512), np.dtype(np.float32), self)

        def close(self):
            pass

    class _DeviceDecoder(_FakeDecoder):
        def decode(self, enc, vocab, language, *, audio_s, max_new=None):
            assert enc[0:3] == (1234, (1, 1500, 512), np.dtype(np.float32))
            return "device", [1.0]

    be._encoder = _DeviceEncoder()
    be._decoder = _DeviceDecoder(["unused"])
    be._filters = np.zeros((80, 201), dtype=np.float32)
    be._vocab = {}
    assert be.transcribe_array(_speech(1.0, np.random.default_rng(8))).text == "device"


def _fake_trt_decoder_for_validation():
    from voxedge.backends.whisper.decoder import TensorRTKVDecoder

    class _Engine:
        def __init__(self, shapes, inputs):
            self.shapes = shapes
            self.inputs = set(inputs)

        def get_tensor_shape(self, name):
            return self.shapes[name]

    class _Wrapper:
        def __init__(self, shapes, inputs):
            self.names = list(shapes)
            self.is_input = {name: name in inputs for name in shapes}
            self.dtype = {name: (np.int64 if name == "input_ids" else np.float32) for name in shapes}
            self.engine = _Engine(shapes, inputs)

        def profile_shape(self, _name):
            if ".encoder." in _name:
                return ((1, 8, 1500, 64), (1, 8, 1500, 64), (1, 8, 1500, 64))
            return ((1, 8, 4, 64), (1, 8, 128, 64), (1, 8, 447, 64))

    pre = {
        "input_ids": (1, -1),
        "encoder_hidden_states": (1, -1, 512),
        "logits": (1, -1, 51865),
        "present.0.decoder.key": (1, 8, -1, 64),
        "present.0.decoder.value": (1, 8, -1, 64),
        "present.0.encoder.key": (1, 8, -1, 64),
        "present.0.encoder.value": (1, 8, -1, 64),
    }
    step = {
        "input_ids": (1, 1),
        "past_key_values.0.decoder.key": (1, 8, -1, 64),
        "past_key_values.0.decoder.value": (1, 8, -1, 64),
        "past_key_values.0.encoder.key": (1, 8, -1, 64),
        "past_key_values.0.encoder.value": (1, 8, -1, 64),
        "logits": (1, 1, 51865),
        "present.0.decoder.key": (1, 8, -1, 64),
        "present.0.decoder.value": (1, 8, -1, 64),
    }
    dec = TensorRTKVDecoder.__new__(TensorRTKVDecoder)
    dec._prefill = _Wrapper(pre, {"input_ids", "encoder_hidden_states"})
    dec._step = _Wrapper(step, set(step) - {"logits", "present.0.decoder.key", "present.0.decoder.value"})
    dec._layers = [0]
    dec._max_positions = 448
    return dec


def test_trt_decoder_validates_io_dtype_and_dynamic_profile_bounds():
    dec = _fake_trt_decoder_for_validation()
    dec._validate_io()
    with pytest.raises(TypeError, match="dtype"):
        dec._validate_encoder_view((1, 1500, 512), np.float16)
    with pytest.raises(RuntimeError, match="shape"):
        dec._validate_encoder_view((1, 1500, 256), np.float32)
    dec._profile_max = 447
    dec._check_past_length(4)
    dec._check_past_length(447)
    with pytest.raises(RuntimeError, match="overflow"):
        dec._check_past_length(448)
    dec._step.dtype["past_key_values.0.encoder.value"] = np.float16
    with pytest.raises(RuntimeError, match="cross-KV dtype"):
        dec._validate_io()

    dec = _fake_trt_decoder_for_validation()
    dec._step.profile_shape = lambda _name: ((1, 8, 5, 64), (1, 8, 128, 64), (1, 8, 447, 64))
    with pytest.raises(RuntimeError, match="past 4..447"):
        dec._validate_io()


def test_trt_decoder_reuses_kv_buffers_and_frees_each_pointer_once():
    from voxedge.backends.whisper.decoder import TensorRTKVDecoder

    class _Errors:
        cudaSuccess = 0

    class _Cuda:
        cudaError_t = _Errors

        def __init__(self):
            self.next = 100
            self.allocs = []
            self.frees = []

        def cudaMalloc(self, size):
            ptr = self.next
            self.next += 1
            self.allocs.append((ptr, size))
            return 0, ptr

        def cudaFree(self, ptr):
            self.frees.append(ptr)
            return 0

    dec = TensorRTKVDecoder.__new__(TensorRTKVDecoder)
    dec._cudart = _Cuda()
    dec._owned = []
    dec._layers = [0, 1]
    dec._kv = {}
    dec._kv_bytes = 0
    first = dec._ensure_kv_buffers(64)
    allocated = len(dec._owned)
    assert len(first) == 4
    assert dec._ensure_kv_buffers(64) is first
    assert len(dec._owned) == allocated
    second = dec._ensure_kv_buffers(128)
    assert second is dec._kv and len(dec._owned) == allocated
    for ptr in list(dec._owned):
        dec._free_owned(ptr)
    assert sorted(dec._cudart.frees) == sorted(ptr for ptr, _ in dec._cudart.allocs)
    dec._free_owned(100)
    assert dec._cudart.frees.count(100) == 1


def test_trt_encoder_device_view_reports_engine_output_dtype():
    from voxedge.backends.whisper.encoders import TensorRTEncoder

    class _Ctx:
        def set_input_shape(self, *_):
            pass

        def get_tensor_shape(self, _):
            return (1, 1500, 512)

        def set_tensor_address(self, *_):
            pass

        def execute_async_v3(self, *_):
            return True

    class _Pool:
        def stream_handle(self):
            return 1

        def copy_htod(self, *_):
            pass

        def synchronize(self):
            pass

    enc = TensorRTEncoder.__new__(TensorRTEncoder)
    enc._in_name, enc._out_name, enc._in_rank = "mel", "hidden", 3
    enc._in_dtype, enc._out_dtype = np.dtype(np.float32), np.dtype(np.float16)
    enc._ctx, enc._pool, enc._bufs = _Ctx(), _Pool(), {}
    enc._sizes = (0, 0)
    # Allocate calls are not relevant to the dtype contract; use stable fake pointers.
    enc._pool.allocate = lambda size: 11 if not enc._bufs else 22
    ptr, shape, dtype, owner = enc.run_device(np.zeros((80, 3000), dtype=np.float32))
    assert (ptr, shape, dtype, owner) == (22, (1, 1500, 512), np.dtype(np.float16), enc)


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


@pytest.mark.parametrize("kind", ["hailo", "rknn", "tensorrt"])
def test_close_survives_a_half_constructed_encoder(kind):
    """`preload` calls close() from its failure path.

    An encoder whose __init__ raised partway — a bad plan, a missing HEF — must
    still be closable, or the AttributeError replaces the real cause and the
    operator sees a complaint about a buffer dict instead of the file that was
    wrong. Calling it twice must also be safe.
    """
    from voxedge.backends.whisper import encoders

    cls = {"hailo": encoders.HailoEncoder, "rknn": encoders.RknnEncoder,
           "tensorrt": encoders.TensorRTEncoder}[kind]
    obj = cls.__new__(cls)          # __init__ never ran
    obj.close()
    obj.close()


def test_the_tensorrt_runtime_outlives_the_engine():
    """NVIDIA's lifetime contract: the Runtime and Logger must outlive the
    engine and its execution context. Deserializing from a temporary
    `trt.Runtime(...)` destroys both at the end of the statement, and what
    follows is undefined behaviour that works until it does not."""
    import inspect

    from voxedge.backends.whisper.encoders import TensorRTEncoder

    init = inspect.getsource(TensorRTEncoder.__init__)
    assert "self._runtime = trt.Runtime(self._logger)" in init
    assert "trt.Runtime(logger" not in init          # no temporary
    # and teardown order: the runtime is dropped after the context and engine
    close = inspect.getsource(TensorRTEncoder.close)
    assert close.index("_ctx = None") < close.index("_runtime = None")


def test_only_text_ids_reach_the_transcript():
    """Every id from EOT up is a special.

    Bounding at TIMESTAMP_BEGIN excluded only the timestamps, so an argmax of
    SOT / TASK_TRANSCRIBE / NO_TIMESTAMPS put `<|startoftranscript|>` and
    friends into the transcript verbatim.
    """
    from voxedge.backends.whisper.decoder import (
        EOT, NO_TIMESTAMPS, SOT, TASK_TRANSCRIBE, TIMESTAMP_BEGIN,
    )

    assert EOT < SOT < TASK_TRANSCRIBE < NO_TIMESTAMPS < TIMESTAMP_BEGIN
    import inspect

    from voxedge.backends.whisper.decoder import OnnxKVDecoder

    assert "if nxt < EOT:" in inspect.getsource(OnnxKVDecoder.decode)


def test_read_vocab_splits_on_the_first_space_only():
    """A token may contain a space, and may BE one — that is how word
    boundaries are encoded. Splitting on every space truncated it, and
    stripping the line ate a leading space outright."""
    import tempfile
    from pathlib import Path as _P

    from voxedge.backends.whisper.decoder import read_vocab

    p = _P(tempfile.mkstemp(suffix=".txt")[1])
    # The line is `<id><SP><token>`, so a token that is itself a single space
    # is written with TWO spaces; one space means an empty token.
    p.write_text("123 foo bar\n456  leading\n789  \n012 \n", encoding="utf-8")
    vocab = read_vocab(p)
    assert vocab["123"] == "foo bar"
    assert vocab["456"] == " leading"
    assert vocab["789"] == " "
    assert vocab["012"] == ""
    p.unlink()


def test_a_window_of_exactly_100ms_is_accepted():
    """(5.0 - 4.9) * 16000 is 1599.9999999999943; truncating rejected a window
    that is exactly at the limit."""
    WhisperASRConfig(encoder_kind="hailo", encoder_path="x", decoder_dir="y",
                     vocab_dir="z", window_s=5.0, padding_cutoff_s=4.9)
    with pytest.raises(ValueError, match="usable audio"):
        WhisperASRConfig(encoder_kind="hailo", encoder_path="x", decoder_dir="y",
                         vocab_dir="z", window_s=5.0, padding_cutoff_s=4.95)


@pytest.mark.parametrize("cap", [1.5, float("nan"), True])
def test_a_non_integer_token_cap_is_refused(cap):
    """`range()` needs an int: 1.5 passed a ">= 1" check and raised TypeError
    inside the decode loop instead."""
    with pytest.raises(ValueError, match="must be an int"):
        WhisperASRConfig(encoder_kind="rknn", encoder_path="x", decoder_dir="y",
                         vocab_dir="z", max_new_tokens=cap)


def test_the_offline_stream_copies_and_checks_the_rate():
    """It accumulates without resampling, so both matter.

    `np.asarray` returns the SAME object for float32 input, so the buffer held
    the caller's array and a caller reusing it changed what got transcribed.
    And an unmatched rate is heard as a different duration — 8000 samples at
    8 kHz read as half a second at 16 kHz — with no error anywhere.
    """
    from voxedge.backends.base import OfflineAccumulateStream, TranscriptionResult

    class _Backend:
        name, sample_rate = "fake", 16000
        def transcribe_array(self, samples, language="auto"):
            return TranscriptionResult(text=str(int(samples.sum())))

    class _Counting(_Backend):
        def transcribe_array(self, samples, language="auto"):
            return TranscriptionResult(text=str(samples.size))

    stream = OfflineAccumulateStream(_Backend())
    buf = np.ones(100, dtype=np.float32)
    stream.accept_waveform(16000, buf)
    buf[:] = 0
    assert stream.finalize()[0] == "100"

    # A mismatched rate is resampled rather than refused: raising would turn a
    # degraded-but-running deployment into a crash, and SenseVoice-TRT has been
    # on this path all along (it hardcodes 16000 into its own fbank, so the
    # incoming rate was simply ignored).
    stream = OfflineAccumulateStream(_Counting())
    stream.accept_waveform(8000, np.ones(8000, dtype=np.float32))
    assert stream.finalize()[0] == "16000", "one second at 8 kHz is one second"


def test_an_utterance_exactly_one_window_long_is_not_split():
    """The guard and the splitter must convert seconds to samples identically.

    The guard rounded and the splitter truncated, so `(5.0 - 4.9) * 16000 ==
    1599.9999999999943` was accepted as 1600 and enforced as 1599 — cutting an
    utterance exactly one window long into two half-windows.
    """
    from voxedge.backends.whisper.asr import _enforce_window, window_samples

    usable = 5.0 - 4.9
    limit = window_samples(usable)
    assert limit == 1600
    assert [len(c) for c in _enforce_window([np.zeros(limit, np.float32)], usable)] == [limit]
    assert len(_enforce_window([np.zeros(limit + 1, np.float32)], usable)) == 2


def test_a_failed_construction_releases_the_accelerator():
    """`build_encoder` has not returned, so the caller has no object to close.

    An RKNNLite whose init_runtime failed still holds NPU context, and a leaked
    Hailo VDevice blocks every later attempt on the box — HailoRT grants
    /dev/hailo0 to one process.
    """
    import inspect

    from voxedge.backends.whisper import encoders

    for cls in (encoders.HailoEncoder, encoders.RknnEncoder):
        assert "self.close()" in inspect.getsource(cls.__init__), cls.__name__


def test_resampling_happens_once_over_the_whole_utterance():
    """Per-chunk resampling leaves a discontinuity at every chunk boundary.

    Measured on a 440 Hz sine at 8 k -> 16 k: 0.17 of amplitude against a
    signal whose peak is 1.0, repeating at the chunk period. Non-integral
    ratios also drift in length — 22050 -> 16000 over 200 chunks accumulated
    4.6 ms. This stream accumulates and transcribes once, so it can resample
    once, and the result must equal resampling the utterance whole.
    """
    from voxedge.backends.base import (
        OfflineAccumulateStream, TranscriptionResult, _resample_linear,
    )

    class _Backend:
        name, sample_rate = "fake", 16000
        def __init__(self): self.seen = None
        def transcribe_array(self, samples, language="auto"):
            self.seen = samples
            return TranscriptionResult(text=str(samples.size))

    src = 8000
    t = np.arange(src) / src
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    backend = _Backend()
    stream = OfflineAccumulateStream(backend)
    for i in range(0, sine.size, 160):
        stream.accept_waveform(src, sine[i:i + 160])
    assert stream.finalize()[0] == "16000"
    expected = _resample_linear(sine, src, 16000)
    assert np.array_equal(backend.seen, expected), "chunking changed the audio"


def test_the_rate_may_not_change_mid_utterance():
    """One utterance is one rate; a change means the caller is confused, and
    silently resampling two halves differently would hide it."""
    from voxedge.backends.base import OfflineAccumulateStream, TranscriptionResult

    class _Backend:
        name, sample_rate = "fake", 16000
        def transcribe_array(self, samples, language="auto"):
            return TranscriptionResult(text="")

    stream = OfflineAccumulateStream(_Backend())
    stream.accept_waveform(16000, np.ones(10, dtype=np.float32))
    with pytest.raises(ValueError, match="changed mid-utterance"):
        stream.accept_waveform(8000, np.ones(10, dtype=np.float32))


def test_downsampling_attenuates_instead_of_folding():
    """Interpolation alone is fine going up and wrong going down.

    Content above the new Nyquist folds back into the band rather than
    disappearing: a 12 kHz tone in 48 kHz input arrived as a full-amplitude
    4 kHz component after resampling to 16 kHz — inside the speech band, and
    indistinguishable from real audio downstream.
    """
    from voxedge.backends.base import _resample_linear

    src, dst = 48000, 16000
    t = np.arange(src) / src

    def _peak(freq):
        out = _resample_linear(np.sin(2 * np.pi * freq * t).astype(np.float32), src, dst)
        spectrum = np.abs(np.fft.rfft(out))
        return spectrum.max() / len(out) * 2

    assert _peak(12000) < 0.4, "the alias is not attenuated"
    assert _peak(1000) > 0.9, "real speech content was damaged"


def test_an_empty_chunk_is_not_audio():
    """It reached `np.stack([])` in some backends, and letting it pin the
    stream's rate would reject the first real chunk at a different one."""
    from voxedge.backends.base import OfflineAccumulateStream, TranscriptionResult

    class _Backend:
        name, sample_rate = "fake", 16000
        def transcribe_array(self, samples, language="auto"):
            return TranscriptionResult(text=str(samples.size))

    stream = OfflineAccumulateStream(_Backend())
    stream.accept_waveform(16000, np.array([], dtype=np.float32))
    assert stream.finalize() == ("", None)

    stream = OfflineAccumulateStream(_Backend())
    stream.accept_waveform(16000, np.array([], dtype=np.float32))
    stream.accept_waveform(8000, np.ones(80, dtype=np.float32))   # must not raise
    assert stream.finalize()[0] == "160"


def test_every_seconds_to_samples_conversion_rounds():
    """The guard, the splitter, the segmenter and the mel front end must agree.

    A usable window computed as a difference of floats lands just under the
    integer, and any truncating conversion then splits an utterance that is
    exactly one window long.
    """
    from voxedge.audio.segment import split_at_silence_energy
    from voxedge.backends.whisper.frontend import log_mel

    usable = 5.0 - 1.0000000000000004        # 4.0 minus a float's worth
    audio = np.random.RandomState(0).normal(0, 0.15, 128000).astype(np.float32)
    assert [len(c) for c in split_at_silence_energy(
        audio, 16000, max_seg_s=usable)] == [64000, 64000]

    mel = log_mel(np.ones(1600, dtype=np.float32),
                  np.zeros((80, 201), dtype=np.float32), 5.0, 4.9)
    assert mel.shape[1] == 500
