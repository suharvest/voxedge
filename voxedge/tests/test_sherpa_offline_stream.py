"""Offline-only sherpa deployments must still serve a streaming session.

A profile that loads SenseVoice into the offline slot and no Paraformer into
the online slot (``rpi5-sensevoice``) used to have ``create_stream()`` raise
unconditionally, so ``/asr/stream`` and ``/v2v/stream`` accepted the socket,
reported "no streaming ASR available" and closed it — while ``POST /asr``
worked fine on the very same recognizer. These tests pin the three behaviours
that fix depends on:

  * offline-only → ``OfflineAccumulateStream`` and STREAMING advertised,
  * online loaded → the native incremental stream, unchanged,
  * nothing loaded → an error, not a broken stream.

No sherpa_onnx runtime is needed: the recognizers are stubs, and the code path
under test is the dispatch, not the decode.
"""
import numpy as np
import pytest

from voxedge.backends.base import (
    ASRCapability,
    OfflineAccumulateStream,
)
from voxedge.backends.sherpa.asr import (
    SherpaASRBackend,
    SherpaASRConfig,
    SherpaASRStream,
)


class _FakeNativeStream:
    """Stands in for a sherpa_onnx offline stream."""

    def __init__(self, sink):
        self._sink = sink
        self.result = type("R", (), {"text": ""})()

    def accept_waveform(self, sample_rate, samples):
        self._sink.append((sample_rate, np.asarray(samples, dtype=np.float32)))


class _FakeOfflineRecognizer:
    """Records what was fed in and returns a fixed transcript."""

    def __init__(self, text="  你好世界  "):
        self.text = text
        self.fed = []
        self.decoded = 0

    def create_stream(self):
        return _FakeNativeStream(self.fed)

    def decode_stream(self, stream):
        self.decoded += 1
        stream.result.text = self.text


class _FakeOnlineRecognizer:
    def create_stream(self):
        return object()


def _backend(offline=None, online=None) -> SherpaASRBackend:
    be = SherpaASRBackend(config=SherpaASRConfig(streaming_provider="cpu"))
    be._offline_recognizer = offline
    be._online_recognizer = online
    return be


# ── dispatch ────────────────────────────────────────────────────────────────

def test_offline_only_yields_the_accumulate_stream():
    be = _backend(offline=_FakeOfflineRecognizer())
    assert isinstance(be.create_stream(), OfflineAccumulateStream)


def test_offline_only_advertises_streaming():
    be = _backend(offline=_FakeOfflineRecognizer())
    # capabilities() still reports only what natively loaded ...
    assert be.capabilities == {ASRCapability.OFFLINE}
    # ... while has_capability() — what the server gates /asr/stream on —
    # accounts for the adapter.
    assert be.has_capability(ASRCapability.STREAMING) is True
    assert be.supports_offline_streaming is True


def test_online_recognizer_still_wins():
    """Every existing sherpa deployment keeps the native incremental stream."""
    be = _backend(offline=_FakeOfflineRecognizer(), online=_FakeOnlineRecognizer())
    assert isinstance(be.create_stream(), SherpaASRStream)


def test_online_only_is_unchanged():
    be = _backend(online=_FakeOnlineRecognizer())
    assert isinstance(be.create_stream(), SherpaASRStream)
    assert be.supports_offline_streaming is False


def test_nothing_loaded_raises():
    be = _backend()
    assert be.has_capability(ASRCapability.STREAMING) is False
    with pytest.raises(RuntimeError, match="No recognizer loaded"):
        be.create_stream()


# ── transcribe_array (what OfflineAccumulateStream.finalize calls) ──────────

def test_transcribe_array_decodes_and_strips():
    rec = _FakeOfflineRecognizer(text="  hello  ")
    be = _backend(offline=rec)
    out = be.transcribe_array(np.zeros(1600, dtype=np.float32))
    assert out.text == "hello"
    assert rec.decoded == 1
    fed_rate, fed_samples = rec.fed[0]
    assert fed_rate == 16000
    assert fed_samples.shape == (1600,)


def test_transcribe_array_reports_the_pinned_language_not_the_request():
    be = _backend(offline=_FakeOfflineRecognizer())
    be._config.offline_language = "yue"
    assert be.transcribe_array(np.zeros(16, dtype=np.float32), "zh").language == "yue"


def test_transcribe_array_without_an_offline_recognizer_raises():
    with pytest.raises(RuntimeError, match="Offline recognizer not loaded"):
        _backend(online=_FakeOnlineRecognizer()).transcribe_array(
            np.zeros(16, dtype=np.float32)
        )


def _wav_bytes(samples_int16: np.ndarray, rate: int, channels: int = 1) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples_int16.tobytes())
    return buf.getvalue()


def test_transcribe_and_transcribe_array_share_one_decode_path():
    """The file endpoint and the stream endpoint must not drift apart.

    Non-silent audio, checked sample-by-sample: a stub that always returns the
    same text would hide a decode-path swap, so assert on what the recognizer
    was actually handed.
    """
    pytest.importorskip("soundfile")

    rec = _FakeOfflineRecognizer(text="same")
    be = _backend(offline=rec)

    tone = (np.sin(np.linspace(0, 40 * np.pi, 1600)) * 16000).astype(np.int16)
    assert be.transcribe(_wav_bytes(tone, 16000)).text == "same"
    assert be.transcribe_array(tone.astype(np.float32)).text == "same"
    assert rec.decoded == 2

    (rate_file, from_file), (rate_arr, from_arr) = rec.fed
    assert rate_file == rate_arr == 16000
    assert from_file.dtype == from_arr.dtype == np.float32
    # transcribe() reads through soundfile, which scales int16 to [-1, 1);
    # transcribe_array() takes what the stream adapter already produced. Both
    # must describe the same waveform, so compare shapes and normalized shape.
    assert from_file.shape == from_arr.shape == (1600,)
    assert np.allclose(from_file, from_arr / 32768.0, atol=1e-4)
    assert np.abs(from_file).max() > 0.1  # not silence


def test_transcribe_downmixes_stereo_and_resamples_to_16k():
    """The file path's own preprocessing, asserted on the samples handed over."""
    pytest.importorskip("soundfile")

    rec = _FakeOfflineRecognizer()
    be = _backend(offline=rec)

    left = np.full(800, 10000, dtype=np.int16)
    right = np.full(800, -6000, dtype=np.int16)
    stereo = np.empty(1600, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    be.transcribe(_wav_bytes(stereo, 8000, channels=2))

    rate, fed = rec.fed[0]
    assert rate == 16000
    # 800 frames at 8 kHz -> 1600 samples at 16 kHz.
    assert fed.shape == (1600,)
    assert fed.dtype == np.float32
    # Both channels are constant, so the mean is constant too.
    assert np.allclose(fed, (10000 - 6000) / 2 / 32768.0, atol=1e-3)


# ── end-to-end through the adapter ──────────────────────────────────────────

def test_stream_accumulates_then_transcribes_once_at_finalize():
    rec = _FakeOfflineRecognizer(text="一句话")
    be = _backend(offline=rec)
    stream = be.create_stream()

    for _ in range(5):
        stream.accept_waveform(16000, np.full(320, 0.1, dtype=np.float32))
    # No partial before finalize: SenseVoice has no incremental output, and
    # nothing decoded yet.
    assert stream.get_partial() == ("", False)
    assert rec.decoded == 0

    text, _lang = stream.finalize()
    assert text == "一句话"
    assert rec.decoded == 1
    fed_rate, fed_samples = rec.fed[0]
    assert fed_rate == 16000
    assert fed_samples.shape == (5 * 320,)


def test_stream_resamples_a_mismatched_rate_before_decoding():
    rec = _FakeOfflineRecognizer()
    be = _backend(offline=rec)
    stream = be.create_stream()
    stream.accept_waveform(8000, np.zeros(8000, dtype=np.float32))
    stream.finalize()
    fed_rate, fed_samples = rec.fed[0]
    assert fed_rate == 16000
    assert fed_samples.shape == (16000,)


def test_empty_stream_finalizes_without_decoding():
    rec = _FakeOfflineRecognizer()
    stream = _backend(offline=rec).create_stream()
    assert stream.finalize() == ("", None)
    assert rec.decoded == 0
