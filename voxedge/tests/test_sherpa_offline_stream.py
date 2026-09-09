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


def test_transcribe_and_transcribe_array_share_one_decode_path():
    """The file endpoint and the stream endpoint must not drift apart."""
    import io
    import wave

    rec = _FakeOfflineRecognizer(text="same")
    be = _backend(offline=rec)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(1600, dtype=np.int16).tobytes())

    pytest.importorskip("soundfile")
    assert be.transcribe(buf.getvalue()).text == "same"
    assert be.transcribe_array(np.zeros(1600, dtype=np.float32)).text == "same"
    assert rec.decoded == 2


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
