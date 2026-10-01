from __future__ import annotations

import threading
import sys
from types import SimpleNamespace

import numpy as np

import voxedge.backends.jetson.trt_edge_llm_asr as asr_mod
from voxedge.backends.jetson.trt_edge_llm_asr import (
    TRTEdgeLLMASRBackend,
    TranscriptionResult,
)


class _FakeWorkerIO:
    def __init__(self):
        self.requests = []

    def request(self, payload):
        self.requests.append(payload)
        yield {"event": "done", "ok": True, "responses": [{"output_text": "language Englishhello"}]}


def _backend(worker_io):
    backend = object.__new__(TRTEdgeLLMASRBackend)
    backend._worker_lock = threading.Lock()
    backend._wio = worker_io
    backend._worker = object()
    backend._worker_ready_meta = {}
    backend._config = SimpleNamespace(
        temperature=1.0, top_p=1.0, top_k=1, max_generate_length=16,
        collapse_repetition=False,
    )
    backend._ensure_worker = lambda: None
    return backend


def test_worker_language_alias_builds_forced_assistant_request():
    fake = _FakeWorkerIO()
    result = _backend(fake)._transcribe_worker("/tmp/a.wav", 0.0, "en-us")
    request = fake.requests[0]
    assert request["add_generation_prompt"] is False
    assert request["requests"][0]["messages"][-1] == {
        "role": "assistant", "content": "language English<asr_text>"
    }
    assert result.meta["requested_language"] == "en-us"


def test_worker_auto_keeps_legacy_generation_prompt():
    fake = _FakeWorkerIO()
    _backend(fake)._transcribe_worker("/tmp/a.wav", 0.0, "auto")
    request = fake.requests[0]
    assert request["add_generation_prompt"] is True
    assert len(request["requests"][0]["messages"]) == 1


def test_unknown_worker_language_rejected_before_request():
    fake = _FakeWorkerIO()
    try:
        _backend(fake)._transcribe_worker("/tmp/a.wav", 0.0, "xx")
    except ValueError as exc:
        assert "unsupported ASR worker language" in str(exc)
    else:
        raise AssertionError("unknown language was accepted")
    assert fake.requests == []


def test_silence_skip_preserves_pulse():
    quiet = np.zeros(16000, dtype=np.float32)
    pulse = quiet.copy()
    pulse[3200:3520] = 0.02
    fn = TRTEdgeLLMASRBackend._is_effectively_silent_segment
    assert fn(quiet, 16000) is True
    assert fn(pulse, 16000) is False


def test_gain_aware_vad_preserves_quiet_speech():
    peaks = []

    class _FakeVad:
        def __init__(self, mode):
            self.mode = mode

        def is_speech(self, frame, sample_rate):
            peak = max(abs(int(x)) for x in np.frombuffer(frame, dtype=np.int16))
            peaks.append(peak)
            # The raw 0.002 signal is below this threshold; the gain-aware
            # copy is loud enough to cross it. This proves the second VAD
            # branch protects quiet speech rather than making VAD always true.
            return peak >= 1000

    old = sys.modules.get("webrtcvad")
    sys.modules["webrtcvad"] = SimpleNamespace(Vad=_FakeVad)
    try:
        quiet_voice = np.full(16000, 0.002, dtype=np.float32)
        assert TRTEdgeLLMASRBackend._is_effectively_silent_segment(
            quiet_voice, 16000
        ) is False
        assert len(peaks) >= 2 and peaks[0] < 1000 < max(peaks)
    finally:
        if old is None:
            sys.modules.pop("webrtcvad", None)
        else:
            sys.modules["webrtcvad"] = old


def test_gain_aware_vad_receives_padded_tail_frame():
    seen = []

    class _TailVad:
        def __init__(self, mode):
            pass

        def is_speech(self, frame, sample_rate):
            seen.append(frame)
            return any(frame)

    old = sys.modules.get("webrtcvad")
    sys.modules["webrtcvad"] = SimpleNamespace(Vad=_TailVad)
    try:
        audio = np.zeros(321, dtype=np.float32)
        audio[-1] = 0.002
        assert TRTEdgeLLMASRBackend._is_effectively_silent_segment(audio, 16000) is False
        assert len(seen) == 2
        assert all(len(frame) == 640 for frame in seen)
        tail = np.frombuffer(seen[-1], dtype=np.int16)
        assert tail[0] != 0
        assert np.all(tail[1:] == 0)
    finally:
        if old is None:
            sys.modules.pop("webrtcvad", None)
        else:
            sys.modules["webrtcvad"] = old


def test_offline_splitter_passes_max_segment_to_vad_and_energy_and_keeps_tail(monkeypatch):
    calls = []
    source = np.arange(16000 * 2 + 17, dtype=np.float32)

    def fake_vad(audio, sample_rate, *, max_seg_s):
        calls.append(("vad", max_seg_s))
        return [audio]

    def fake_energy(audio, sample_rate, *, max_seg_s):
        calls.append(("energy", max_seg_s))
        return [audio]

    monkeypatch.setattr(asr_mod, "_split_at_silence_vad", fake_vad)
    monkeypatch.setattr(asr_mod, "_split_at_silence_energy", fake_energy)
    got = asr_mod._split_offline_audio(source, 16000, max_segment_s=1.25)
    assert calls == [("vad", 1.25)]
    assert np.array_equal(np.concatenate(got), source)
    assert all(len(seg) <= 20000 for seg in got)

    calls.clear()
    def missing_vad(*args, **kwargs):
        raise ImportError("no webrtcvad")

    monkeypatch.setattr(asr_mod, "_split_at_silence_vad", missing_vad)
    got = asr_mod._split_offline_audio(source, 16000, max_segment_s=1.25)
    assert calls == [("energy", 1.25)]
    assert np.array_equal(np.concatenate(got), source)
    assert all(len(seg) <= 20000 for seg in got)


def test_transcribe_validates_language_before_decoding():
    fake = _FakeWorkerIO()
    backend = _backend(fake)
    backend._ready = True
    backend._config.offline_segment_enabled = True
    backend._config.offline_segment_threshold_s = 6.0
    backend._config.offline_segment_min_s = 0.4
    backend._config.request_audio_wav = True
    try:
        backend.transcribe(b"bad wav", language="xx")
    except ValueError as exc:
        assert "unsupported ASR worker language" in str(exc)
    else:
        raise AssertionError("invalid language bypassed transcribe validation")
    assert fake.requests == []


def test_segment_metadata_separates_silent_empty_and_failed(monkeypatch):
    backend = object.__new__(TRTEdgeLLMASRBackend)
    backend._config = SimpleNamespace(offline_segment_min_s=0.4, offline_segment_threshold_s=6.0)
    backend._worker_ready_meta = {}
    silent = np.zeros(16000, dtype=np.float32)
    voiced = np.ones(16000, dtype=np.float32) * 0.02
    monkeypatch.setattr(asr_mod, "_split_offline_audio", lambda audio, sr, max_segment_s: [silent, voiced, voiced])
    calls = iter((TranscriptionResult("", None, {}), RuntimeError("boom")))

    def fake_transcribe(audio_bytes, language):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    backend.transcribe = fake_transcribe
    result = backend._transcribe_segmented_offline(np.ones(48000, dtype=np.float32), 16000, "English")
    assert result.meta["skipped_silent_segments"] == 1
    assert result.meta["empty_segments"] == 1
    assert result.meta["failed_segments"] == 1
    assert result.meta["requested_language"] == "English"
