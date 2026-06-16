"""Engine-dispatcher pre-speech preroll ring (first-word-drop fix).

Silero clips the ~200-300ms onset while it latches SPEECH_START, so the
dispatcher buffers a short ring of pre-speech frames and replays them as the
FIRST ``accept_audio`` calls of the fresh ASR turn (port of
``server/main.py:3800-3919`` into ``voxedge/engine/audio_dispatcher.py``).

These tests assert the EXACT frames forwarded to ``accept_audio``:
  * with preroll>0: buffered onset frames are injected BEFORE the post-start
    chunk, in chronological order, fed exactly once (no double-count);
  * with preroll=0: byte-identical to the no-ring path (no injection).
"""
from __future__ import annotations

import asyncio

import numpy as np

from voxedge.engine.audio_dispatcher import _AudioDispatcher


# ── minimal fakes ─────────────────────────────────────────────────────────────

class _FakeVAD:
    SPEECH_START = "start"
    SPEECH_END = "end"

    def __init__(self, script):
        # script: list of events (one per chunk), None = no event
        self._script = list(script)
        self._i = 0

    def process(self, samples):
        ev = self._script[self._i] if self._i < len(self._script) else None
        self._i += 1
        return ev


class _FakeState:
    def __init__(self):
        self.client_closed = False
        self.asr_session_closed = False
        self.asr_active = False

    def stamp_endpoint(self, *_a, **_k):
        pass


class _FakeASRLoop:
    """Mimics _ASRLoop.open_turn() which sets state.asr_active = True."""

    def __init__(self, state):
        self._state = state

    async def open_turn(self):
        self._state.asr_active = True
        return True


class _FakeASRMgr:
    def __init__(self):
        self.accepted: list[np.ndarray] = []

    async def accept_audio(self, samples):
        self.accepted.append(np.asarray(samples).copy())


class _FakeBackend:
    sample_rate = 16000


class _FakeEngine:
    def __init__(self, vad_preroll_ms):
        self.multi_utterance = False
        self.vad_preroll_ms = vad_preroll_ms


class _FakeTransport:
    def __init__(self, chunks):
        self._chunks = chunks

    async def recv_audio(self):
        for c in self._chunks:
            yield c


class _FakeSession:
    def __init__(self, chunks, vad_script, *, preroll_ms):
        self.state = _FakeState()
        self.engine = _FakeEngine(preroll_ms)
        self.asr_enabled = True
        self._vad = _FakeVAD(vad_script)
        self._asr = _FakeASRLoop(self.state)
        self._asr_mgr = _FakeASRMgr()
        self._asr_be = _FakeBackend()
        self.transport = _FakeTransport(chunks)
        self.events: list[dict] = []

    async def _send_event(self, payload):
        self.events.append(payload)

    async def _bargein_tts(self):
        pass


def _pcm(value: int, n: int = 160) -> bytes:
    """A distinct constant-valued int16 chunk so we can identify it later."""
    return np.full(n, value, dtype=np.int16).tobytes()


def _f32(value: int, n: int = 160) -> np.ndarray:
    return np.full(n, value, dtype=np.int16).astype(np.float32) / 32768.0


def _run(sess):
    asyncio.run(_AudioDispatcher(sess).run())


# ── tests ─────────────────────────────────────────────────────────────────────

def test_preroll_injects_onset_frames_before_trigger_chunk():
    # 2 idle chunks (buffered), then SPEECH_START on the 3rd (trigger) chunk.
    # 300ms @ 16k = 4800 samples; chunks are 160 samples → ring holds all idle.
    chunks = [_pcm(11), _pcm(22), _pcm(33)]
    vad_script = [None, None, _FakeVAD.SPEECH_START]
    sess = _FakeSession(chunks, vad_script, preroll_ms=300)

    _run(sess)

    accepted = sess._asr_mgr.accepted
    # Expected order: [idle1, idle2] injected as ONE concatenated preroll,
    # then the trigger chunk fed via the normal accept_audio path.
    assert len(accepted) == 2
    np.testing.assert_array_equal(accepted[0], np.concatenate([_f32(11), _f32(22)]))
    np.testing.assert_array_equal(accepted[1], _f32(33))


def test_preroll_zero_is_byte_identical_no_injection():
    chunks = [_pcm(11), _pcm(22), _pcm(33)]
    vad_script = [None, None, _FakeVAD.SPEECH_START]
    sess = _FakeSession(chunks, vad_script, preroll_ms=0)

    _run(sess)

    accepted = sess._asr_mgr.accepted
    # No ring → only the trigger chunk reaches accept_audio (legacy behaviour).
    assert len(accepted) == 1
    np.testing.assert_array_equal(accepted[0], _f32(33))


def test_preroll_ring_capped_drops_oldest():
    # preroll cap = 16000 * 20 / 1000 = 320 samples → holds ~2 chunks (320),
    # so with 3 idle chunks the OLDEST is dropped before SPEECH_START.
    chunks = [_pcm(11), _pcm(22), _pcm(33), _pcm(44)]
    vad_script = [None, None, None, _FakeVAD.SPEECH_START]
    sess = _FakeSession(chunks, vad_script, preroll_ms=20)

    _run(sess)

    accepted = sess._asr_mgr.accepted
    # Ring kept only the last frames within cap: {22,33} (11 evicted), then 44.
    assert len(accepted) == 2
    np.testing.assert_array_equal(accepted[0], np.concatenate([_f32(22), _f32(33)]))
    np.testing.assert_array_equal(accepted[1], _f32(44))


def test_no_preroll_when_speech_starts_on_first_chunk():
    # SPEECH_START immediately: nothing buffered → no injection, just the chunk.
    chunks = [_pcm(55)]
    vad_script = [_FakeVAD.SPEECH_START]
    sess = _FakeSession(chunks, vad_script, preroll_ms=300)

    _run(sess)

    accepted = sess._asr_mgr.accepted
    assert len(accepted) == 1
    np.testing.assert_array_equal(accepted[0], _f32(55))
