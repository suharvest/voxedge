"""TRT-Edge-LLM TTS config → worker-env / worker-request behavior (migration gap).

The env/profile → ``TRTEdgeLLMTTSConfig`` mapping (perf_profile, seed, talker_*,
chunk frames, stateful_code2wav) is covered product-side in
``app/tests/test_voxedge_backend_config.py``. This locks the voxedge-side
*config → behavior* translation that the stale product tests asserted against the
old env-reading backend:

  * ``_worker_env()`` emits the stateful / non-stateful (official) env dict
    derived from ``config.stateful_code2wav`` + ``highperf_enabled``;
  * the streaming worker request derives ``first_chunk_frames`` from the perf
    profile (quality=7 / balanced=6 / fast=4 under stateful code2wav) and from
    the ``streaming_profile`` (v2v fast window), with chunk_frames=10 under
    stateful;
  * text segmentation reuses one fixed ``seed`` across all segments;
  * base64 chunk transport is decoded to raw PCM bytes.

NOTE intentionally NOT ported (dropped product behavior, not voxedge gaps):
  * ``OVS_TTS_SPEAKERS_JSON`` registry lookups (speaker-name / embedding-by-id) —
    voxedge ``resolve_speaker_kwargs`` is registry-free; plain speaker_id
    pass-through is covered in ``test_engine_tts_parity.py``.

Mac-safe: backend via ``__new__`` + a fake subprocess driven through voxedge's
own ``WorkerIO`` (no CUDA, no real worker binary).
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time

from voxedge.backends.jetson.trt_edge_llm_tts import (
    TRTEdgeLLMTTSBackend,
    TRTEdgeLLMTTSConfig,
    _split_tts_text,
)


# ── _worker_env(): pure config → env dict (no worker needed) ──────────────────


def _env_backend(**config_kwargs):
    backend = TRTEdgeLLMTTSBackend.__new__(TRTEdgeLLMTTSBackend)
    backend._config = TRTEdgeLLMTTSConfig(**config_kwargs)
    return backend


def test_worker_env_stateful_defaults():
    be = _env_backend(stateful_code2wav=True)
    env = be._worker_env()
    assert env["EDGE_LLM_TTS_CUDA_GRAPH"] == "0"
    assert env["EDGE_LLM_TTS_STATEFUL_CODE2WAV"] == "1"
    assert env["EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES"] == "0"
    assert env["QWEN3_TTS_CP_DECODE_CUDA_GRAPH"] == "1"
    assert env["QWEN3_TTS_ACTIVE_CP_GROUPS"] == "13"


def test_worker_env_official_non_stateful():
    # Non-stateful (official-like) path: context frames = 3, no CP-decode keys.
    be = _env_backend(stateful_code2wav=False, qwen3_runtime_profile="official")
    env = be._worker_env()
    assert env["EDGE_LLM_TTS_STATEFUL_CODE2WAV"] == "0"
    assert env["EDGE_LLM_TTS_CODE2WAV_CONTEXT_FRAMES"] == "3"
    assert "QWEN3_TTS_CP_DECODE_CUDA_GRAPH" not in env
    assert "QWEN3_TTS_ACTIVE_CP_GROUPS" not in env


# ── streaming worker request: config → payload ───────────────────────────────


class _FakeStdin:
    def __init__(self):
        self.writes = []
        self._lock = threading.Lock()

    def write(self, s):
        with self._lock:
            self.writes.append(s)
        return len(s)

    def flush(self):
        pass


class _FakeStdoutQueue:
    def __init__(self):
        self._q = queue.Queue()

    def feed(self, line):
        self._q.put(line if line.endswith("\n") else line + "\n")

    def eof(self):
        self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item


class _FakeProc:
    def __init__(self):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdoutQueue()
        self.stderr = None

    def poll(self):
        return None


def _make_streaming_backend(**config_kwargs):
    """Backend wired to a fake proc + voxedge WorkerIO, capturing every request.

    A daemon feeder echoes a terminal ``done`` (and optional pre-chunks) keyed by
    each request id, so ``generate_streaming`` drains cleanly.
    """
    from voxedge.backends.jetson.worker_io import WorkerIO

    backend = TRTEdgeLLMTTSBackend.__new__(TRTEdgeLLMTTSBackend)
    backend._config = TRTEdgeLLMTTSConfig(**config_kwargs)
    backend._product_backend = None
    backend._ready = True
    backend._worker_lock = threading.Lock()
    backend._worker_stderr_tail = []
    backend._worker_concurrency = 1

    proc = _FakeProc()
    wio = WorkerIO(proc, concurrency=1)
    backend._worker = proc
    backend._worker_io = wio
    backend._ensure_worker = lambda: None

    requests: list[dict] = []

    def _feeder():
        seen = 0
        while True:
            for _ in range(200):
                if len(proc.stdin.writes) > seen:
                    break
                time.sleep(0.005)
            else:
                return  # no more requests arriving
            line = proc.stdin.writes[seen]
            seen += 1
            req = json.loads(line)
            requests.append(req)
            rid = req["id"]
            for ch in backend._feed_chunks:
                proc.stdout.feed(json.dumps({**ch, "id": rid}))
            proc.stdout.feed(json.dumps({"id": rid, "event": "done", "ok": True}))

    backend._feed_chunks = []
    backend._requests = requests
    threading.Thread(target=_feeder, daemon=True).start()
    return backend


def test_stateful_quality_profile_first_chunk_7():
    be = _make_streaming_backend(stateful_code2wav=True, perf_profile="quality")
    assert list(be.generate_streaming("你好", segment_text=False, _retry_empty=False)) == []
    req = be._requests[0]
    assert req["first_chunk_frames"] == 7
    assert req["chunk_frames"] == 10
    assert req["max_chunk_frames"] == 10
    assert req["adaptive_chunks"] is False


def test_stateful_balanced_profile_first_chunk_6():
    be = _make_streaming_backend(stateful_code2wav=True, perf_profile="balanced")
    assert list(be.generate_streaming("你好", segment_text=False, _retry_empty=False)) == []
    assert be._requests[0]["first_chunk_frames"] == 6


def test_stateful_fast_profile_first_chunk_4():
    be = _make_streaming_backend(stateful_code2wav=True, perf_profile="fast")
    assert list(be.generate_streaming("你好", segment_text=False, _retry_empty=False)) == []
    assert be._requests[0]["first_chunk_frames"] == 4


def test_v2v_streaming_profile_uses_first_frame_fast_window():
    be = _make_streaming_backend(stateful_code2wav=False)
    assert list(
        be.generate_streaming("你好", streaming_profile="v2v", segment_text=False, _retry_empty=False)
    ) == []
    req = be._requests[0]
    assert req["first_chunk_frames"] == 1
    assert req["chunk_frames"] == 97
    assert req["max_chunk_frames"] == 97
    assert req["adaptive_chunks"] is False


def test_segments_reuse_fixed_seed():
    be = _make_streaming_backend(
        stateful_code2wav=True, seed=123, segment_max_chars_cjk=8
    )
    text = "这是一个没有任何标点符号的长中文单句"
    assert list(be.generate_streaming(text, _retry_empty=False)) == []
    assert len(be._requests) > 1
    assert [r["text"] for r in be._requests] == _split_tts_text(text, 8, max_chars_cjk=8)
    assert {r["seed"] for r in be._requests} == {123}


def test_base64_chunk_decoded_to_pcm():
    be = _make_streaming_backend(stateful_code2wav=True)
    be._feed_chunks = [
        {
            "event": "chunk",
            "ok": True,
            "chunk_transport": "base64",
            "audio_b64": base64.b64encode(b"pcm").decode("ascii"),
        }
    ]
    out = list(be.generate_streaming("你好", segment_text=False, _retry_empty=False))
    assert out == [b"pcm"]
