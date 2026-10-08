"""ASR backend via TRT-Edge-LLM C++ worker (qwen3_asr_worker / llm_inference).

adapted from app/backends/jetson/trt_edge_llm_asr.py + app/core/worker_io.py
(2026-05-30), dedup after registry switch.

Audio is converted to a Whisper-compatible log-mel spectrogram in Python
(numpy-only), saved as a safetensors file, and passed to the LLM binary via
``--multimodalEngineDir`` for the audio encoder. The Python side spawns a C++
worker subprocess and talks JSON-line IPC, so it imports cleanly on a machine
with no CUDA / tensorrt.

Differences from the production copy (decoupling per spec §3.1 / §10):
  * ABCs imported from ``voxedge.backends.base`` (ASRBackend / ASRCapability /
    ASRStream / TranscriptionResult) and ``ConcurrencyCapability`` from
    ``voxedge.engine.concurrency_capability``.
  * ALL ~30 ``os.environ.get(...)`` reads (EDGE_LLM_* paths, ASR_* sampling,
    OVS_VAD_* offline-split, manifest path) are replaced by an explicit
    ``TRTEdgeLLMASRConfig`` dataclass injected at construction time. voxedge
    has ZERO module-scope or hardcoded env reads.
  * ``WorkerIO`` imported from the sibling ``voxedge.backends.jetson.worker_io``
    (not ``app.core.worker_io``).
  * The production offline-split path imported a DELETED module
    (``app.backends.jetson.qwen3_asr``) and ``app.core.vad`` /
    ``app.core.qwen3_artifact_downloader``. The silence splitters are
    reproduced env-free in ``._util``; the optional VAD-backend splitter and
    artifact auto-download are dropped (voxedge ships neither), so the long
    audio path uses the webrtcvad→energy splitter cascade only.
  * ``concurrency_capability`` is an instance method (voxedge base contract)
    reading ``config.max_slots`` instead of env/profile; the N>1
    ``--max_slots`` conditional (main fix b1cb1a5) is preserved.

Supports: OFFLINE, MULTI_LANGUAGE, STREAMING
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from voxedge.text.degenerate import collapse_repetition, collapse_segment_repeats
from voxedge.backends.base import (
    ASRBackend,
    ASRCapability,
    ASRStream,
    TranscriptionResult,
)
from voxedge.engine.concurrency_capability import ConcurrencyCapability

from ._trt_edge_llm_util import (
    DEFAULT_ENERGY_SPLIT_RMS,
    VAD_MAX_SEG_SEC,
    _split_at_silence_energy,
    _split_at_silence_vad,
)
from .trt_edge_llm_ipc import audio_bytes_to_mel, run_binary, write_safetensors
from .worker_io import WorkerExitError as _WIOExitError
from .worker_io import WorkerIO

logger = logging.getLogger(__name__)


# ── env → config mapping (defaults byte-equal to production env defaults) ────
# Original env var                       → TRTEdgeLLMASRConfig field
#   EDGE_LLM_ASR_BIN                      → asr_binary
#   EDGE_LLM_ASR_WORKER_BIN              → worker_binary
#   EDGE_LLM_ASR_PLUGIN_PATH/EDGELLM_ASR_PLUGIN_PATH → plugin_path
#   EDGE_LLM_ASR_ENGINE_DIR             → engine_dir
#   EDGE_LLM_ASR_AUDIO_ENC_DIR          → audio_encoder_dir
#   EDGE_LLM_ASR_WORKER                  → use_worker (default True)
#   EDGE_LLM_ASR_MEL_TENSOR_NAME        → mel_tensor_name ("mel")
#   EDGE_LLM_ASR_MAX_MEL_FRAMES         → max_mel_frames (6000)
#   EDGE_LLM_ASR_MAX_CONCURRENT          → max_slots (1)   ← N>1 gates --max_slots
#   EDGE_LLM_ASR_REQUIRE_IFB            → require_ifb (False) ← v011 IFB ready provenance
#   EDGE_LLM_ASR_STREAM_MODE            → stream_mode ("accumulate")
#   EDGE_LLM_ASR_STREAM_CHUNK_SEC       → stream_chunk_sec (0.5)
#   EDGE_LLM_ASR_STREAM_UNFIXED_CHUNKS  → stream_unfixed_chunks (2)
#   EDGE_LLM_ASR_STREAM_UNFIXED_TOKENS  → stream_unfixed_tokens (5)
#   EDGE_LLM_ASR_MEL_SETTINGS           → mel_settings_path ("")
#   EDGE_LLM_ASR_MEL_FILTERS            → mel_filters_path ("")
#   ASR_TEMPERATURE                      → temperature (1.0)
#   ASR_TOP_P                            → top_p (1.0)
#   ASR_TOP_K                            → top_k (1)
#   ASR_MAX_GENERATE_LENGTH             → max_generate_length (200)
#   EDGE_LLM_ASR_MIN_AUDIO_FRAMES       → min_audio_frames (100)
#   EDGE_LLM_ASR_OFFLINE_SEGMENT        → offline_segment_enabled (True)
#   EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC    → offline_segment_threshold_s (6.0)
#   EDGE_LLM_ASR_OFFLINE_MIN_SEGMENT_SEC → offline_segment_min_s (0.4)
#   EDGE_LLM_ASR_WORKER_WARMUP/SKIP_ASR_WARMUP → worker_warmup (True)
#   EDGE_LLM_ASR_PREWARM_MAX            → prewarm_max (6)
#   EDGE_LLM_ASR_CUDA_GRAPH             → worker_cuda_graph ("0")


@dataclass
class TRTEdgeLLMASRConfig:
    """Explicit construction-time config for :class:`TRTEdgeLLMASRBackend`.

    Every field default is identical to the production env default. Nothing
    here reads ``os.environ``: the path/engine fields have NO usable default
    (production resolved them from ``~/...`` artifact trees via env at module
    import) and MUST be supplied by the caller for a working backend — they
    default to empty strings so the module imports without CUDA/artifacts.
    """

    # Binaries / engines / plugin (no usable default — supply at construction).
    asr_binary: str = ""
    worker_binary: str = ""
    plugin_path: str = ""
    engine_dir: str = ""
    audio_encoder_dir: str = ""

    use_worker: bool = True
    mel_tensor_name: str = "mel"
    max_mel_frames: int = 6000
    # Slot-pool admission ceiling. Default 1 == legacy single-session. N>1
    # gates ``--max_slots`` (main fix b1cb1a5 — preserved).
    max_slots: int = 1
    # v011 IFB ready-provenance gate. When True, after the worker's actual
    # ``ready`` event the backend additionally requires ``ifb is True``,
    # integer (bool NOT accepted) ``max_slots``/``gpu_slots`` equal to the
    # configured slots, and ``engine_max_batch_size >= configured slots``.
    # Missing/false/mismatch rejects startup with detail — the backend never
    # marks itself ready nor claims capacity on failure. Deliberately NOT
    # gated on ``max_slots > 1``: a v010 legacy IFB-less pool with
    # max_slots=2 remains valid when this is False (default), which
    # preserves every legacy B1/B2 pool path.
    require_ifb: bool = False

    stream_mode: str = "accumulate"
    stream_chunk_sec: float = 0.5
    stream_unfixed_chunks: int = 2
    stream_unfixed_tokens: int = 5
    # Proactive long-audio segment cap (seconds). The qwen3_asr_worker prefills
    # the cumulative audio every chunk; the engine KV cache overflows at ~6.2s
    # (prefill_failed, max_kv 256/128). Rotate to a fresh worker segment once the
    # accumulated audio reaches this length so each prefill stays under the cap.
    # Clean cut (no audio carryover) -> no boundary re-transcription/duplication;
    # the trade-off is a possible word split at the boundary (far better than the
    # current total failure on >6.2s audio). 0 / negative disables it (legacy
    # single-segment behaviour). Only audio LONGER than the cap is affected —
    # short utterances take the unchanged single-segment path.
    # NEEDS on-device verification (7.5 / 12.9 / 20s + short-audio latency
    # unchanged) before being relied on in production.
    segment_cap_sec: float = 5.5
    mel_settings_path: str = ""
    mel_filters_path: str = ""
    # WAV-ingest mode (TensorRT-Edge-LLM v0.9.0+): the worker writes the
    # received PCM to a temp WAV and lets the runtime's audio front-end extract
    # mel internally, so the host-side mel_settings/mel_filters assets are not
    # required. Defaults False to preserve the v0.8.0 mel-asset contract; the
    # v090 profiles set EDGELLM_REQUEST_AUDIO_WAV=1 to opt in.
    request_audio_wav: bool = False

    # Sampling.
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 1
    max_generate_length: int = 200

    min_audio_frames: int = 100
    # 关掉后退化输出原样透传（排查用）。见 voxedge/text/degenerate.py
    collapse_repetition: bool = True

    # Offline long-audio segmentation.
    offline_segment_enabled: bool = True
    offline_segment_threshold_s: float = 6.0
    offline_segment_min_s: float = 0.4
    # Empty/clipped-final offline rescue: a streaming final with this many
    # whitespace-separated tokens OR fewer (and ≥ offline_segment_min_s of
    # buffered audio) is re-transcribed offline. 1 catches the single-token
    # "drop" → multi-word case while leaving every normal multi-token final on
    # the zero-new-work fast path. 0 restricts the rescue to fully-empty finals.
    offline_rescue_max_words: int = 1

    # Worker warmup.
    worker_warmup: bool = True
    prewarm_max: int = 6
    worker_cuda_graph: str = "0"

    # Extra env to pass through to the worker subprocess (e.g. profile vars).
    # voxedge does not read process env; callers inject what the worker needs.
    extra_worker_env: dict = field(default_factory=dict)

    # Optional stable artifact name for the runtime-artifact manifest
    # (voxedge.artifacts). None preserves the existing host-mounted behaviour.
    artifact_ref: Optional[str] = None

    def __post_init__(self) -> None:
        self.max_slots = max(1, int(self.max_slots))
        self.stream_mode = (self.stream_mode or "accumulate").strip().lower()


class WorkerProtocolError(RuntimeError):
    """Base class for ASR worker protocol-level errors."""


class NoActiveSessionError(WorkerProtocolError):
    """Worker reported there is no active session (stale id / double-end)."""


class SessionAlreadyActiveError(WorkerProtocolError):
    """Worker reported a session is already active for the given id."""


class WorkerExitError(WorkerProtocolError):
    """Worker subprocess exited or didn't respond before the ack deadline."""


class PoolSaturatedError(RuntimeError):
    """Worker rejected a begin because every slot in its pool is busy.

    The C++ ``qwen3_asr_worker`` with ``--max_slots N`` returns
    ``{"error":"pool_saturated","status":4429,"max_slots":N}`` when an N+1st
    distinct session id tries to begin. Intentionally NOT a
    ``WorkerProtocolError`` subclass: a saturation is a fast-fail busy reject,
    not a worker fault that should trigger a destructive worker restart.
    """

    status: int = 4429

    def __init__(self, message: str, max_slots: Optional[int] = None) -> None:
        super().__init__(message)
        self.max_slots = max_slots


def _classify_worker_response(
    output_data: dict, *, request_event: str | None = None
) -> WorkerProtocolError | None:
    """Map a worker error JSON payload to a typed exception (or None)."""
    if not isinstance(output_data, dict):
        return None
    if output_data.get("event") != "error" and output_data.get("ok") is not False:
        return None
    msg = ""
    for key in ("error", "message", "reason", "detail"):
        v = output_data.get(key)
        if isinstance(v, str) and v:
            msg = v
            break
    if not msg:
        msg = str(output_data)
    low = msg.lower()
    if (
        output_data.get("status") == 4429
        or "pool_saturated" in low
        or "too_many_asr_sessions" in low
        or "too many asr sessions" in low
    ):
        return PoolSaturatedError(msg, max_slots=output_data.get("max_slots"))
    if "no active session" in low or "no_active_session" in low or "unknown session" in low:
        return NoActiveSessionError(msg)
    if "already active" in low or "session_already_active" in low or "already exists" in low:
        return SessionAlreadyActiveError(msg)
    if "exit" in low or "terminated" in low or "worker dead" in low:
        return WorkerExitError(msg)
    return None


class TRTEdgeLLMASRBackend(ASRBackend):
    """ASR via TRT-Edge-LLM qwen3_asr_worker subprocess."""

    @property
    def supports_hot_reload(self) -> bool:  # type: ignore[override]
        return self._use_worker()

    def concurrency_capability(self) -> ConcurrencyCapability:
        """Declare the ASR slot-pool ceiling.

        The backend multiplexes N distinct ASR sessions over one C++ worker
        subprocess (``--max_slots N`` + WorkerIO Semaphore(N)). Reads
        ``config.max_slots`` (was env ``EDGE_LLM_ASR_MAX_CONCURRENT`` / profile
        ``asr_max_slots``). N>1 enables ``supports_parallel``.
        """
        n = max(1, int(self._config.max_slots))
        return ConcurrencyCapability(
            supports_parallel=n > 1,
            max_concurrent=n,
            is_stateful=True,
            requires_exclusive_device=True,
            scaling_mode="single_runtime_multiplex",
        )

    def __init__(self, config: Optional[TRTEdgeLLMASRConfig] = None):
        self._config = config or TRTEdgeLLMASRConfig()
        self._ready = False
        self._worker: Optional[subprocess.Popen] = None
        self._worker_lock = threading.Lock()
        self._restart_lock = threading.Lock()
        self._worker_ready_meta: dict = {}
        self._worker_stderr_tail: deque[str] = deque(maxlen=80)
        self._max_slots: int = max(1, int(self._config.max_slots))
        self._wio: Optional[WorkerIO] = None
        # IFB telemetry provenance (v011): latest worker health snapshot plus a
        # bounded trace ring, both guarded by one lock. Never contains audio
        # payloads — only small worker-emitted dicts. Identity counters here are
        # observational provenance, NOT the official in-window mid-flight proof.
        self._diag_lock = threading.Lock()
        self._latest_ifb_health: Optional[dict] = None
        self._ifb_traces: deque = deque(maxlen=64)
        self._last_health_log_at = 0.0
        # PIDs whose stderr has a dedicated drainer thread (launch path) — a
        # second reader on the same TextIO would race and split lines.
        self._stderr_drain_pids: set = set()
        # Per-worker failure marker (v2): set when a request failed against a
        # worker that is STILL PROVEN LIVE (`poll() is None`) — a broken pipe /
        # missing response is not proof the process exited. While set, requests
        # reject explicitly and no respawn is attempted over the live process.
        # Reset only on an owned successful fresh launch or a confirmed retired
        # exit. ``_worker_failed_reason`` is diagnostic only.
        self._worker_failed = False
        self._worker_failed_reason: Optional[str] = None
        # Pending stdin-close bookkeeping (v2/v3): keyed by id(stdin), maps to
        # the SHARED operation record (retained stdin, done Event, outcome
        # state) set when the ONE owned close thread for that stdin completes.
        # Guarantees a repeated survivor restart never starts a second close
        # thread against the same stdin (no duplicate close ownership), and
        # every caller reads the SAME actual close outcome (exception with
        # closed=False is EOF=False, never claimed from thread completion).
        self._stdin_close_lock = threading.Lock()
        self._stdin_close_pending: dict[int, tuple] = {}
        # Survivor reader-ownership markers (v3). Both maps are keyed by the
        # id() of the ACTUAL object and retain that object as (part of) the
        # value, so the id cannot be reused by another object while the
        # operation is active. ``_stdout_drain_pending`` guarantees exactly
        # ONE reader of each actual stdout stream across ready-rejection
        # teardowns and repeated failed restarts (a survivor keeps its WIO,
        # and a stream with a pending teardown drainer never gets a second
        # one). ``_wio_close_pending`` guarantees at most ONE WorkerIO.close
        # thread per ACTUAL WIO object across repeated failed restarts.
        self._ownership_lock = threading.Lock()
        self._stdout_drain_pending: dict[int, object] = {}
        self._wio_close_pending: dict[int, tuple] = {}

    # -- worker teardown (no-KILL contract) ----------------------------------

    @staticmethod
    def _drain_to_void(stream) -> None:
        """Actively consume a pipe until EOF so a child blocked writing to a
        FULL pipe can still exit — without this, the EOF wait below can block
        forever on a worker that fills its stdout/stderr buffer."""
        try:
            for _ in stream:
                pass
        except Exception:
            pass

    def _start_stdout_drain_once(
        self, worker: subprocess.Popen, *, deadline_monotonic: Optional[float] = None
    ) -> None:
        """Start the ONE owned stdout drainer for this ACTUAL stdout stream, or
        do nothing if an earlier drainer still owns it. The actual stream is
        retained in the pending map so its id() cannot be reused while the
        operation is active; ownership is released only when the drain thread
        finishes (EOF or stream error). If a shared absolute deadline is given
        and has already expired, NO thread is started and NO pending marker is
        left behind (no permanently-pending phantom operation)."""
        stream = worker.stdout
        if stream is None or getattr(stream, "closed", False):
            return
        key = id(stream)
        with self._ownership_lock:
            if key in self._stdout_drain_pending:
                # An earlier drainer still owns this actual stream; a second
                # reader would race and split lines.
                return
            # Deadline recheck AFTER the leaf lock, BEFORE starting any new
            # thread: expiry adds no new thread and leaves no pending marker.
            if (
                deadline_monotonic is not None
                and float(deadline_monotonic) - time.monotonic() <= 0.0
            ):
                return
            self._stdout_drain_pending[key] = stream  # retain: no id reuse
        threading.Thread(
            target=self._drain_stdout_owned,
            args=(stream, key),
            name="asr-worker-teardown-drain-stdout",
            daemon=True,
        ).start()

    def _drain_stdout_owned(self, stream, key: int) -> None:
        try:
            self._drain_to_void(stream)
        finally:
            with self._ownership_lock:
                if self._stdout_drain_pending.get(key) is stream:
                    del self._stdout_drain_pending[key]

    def _wio_close_bounded(
        self,
        wio,
        *,
        budget_s: float,
        deadline_monotonic: Optional[float] = None,
    ) -> bool:
        """Bounded WorkerIO.close() ATTEMPT. At most ONE close operation exists
        per ACTUAL WIO object: the pending map stores the retained WIO, the
        done Event AND the SHARED outcome state together, so every caller —
        including one that reuses an operation whose thread raised AFTER an
        earlier caller's wait already timed out — consults the SAME actual
        outcome instead of a per-caller error flag. The actual WIO is retained
        so its id() cannot be reused while the operation is active. A completed
        FAILED attempt may be retried (the entry is removed on completion), but
        while pending it is never duplicated.

        Returns True iff close() COMPLETED without raising. WorkerIO.close()
        success is the close ATTEMPT completing — it is NOT stdin-EOF proof
        (stdin EOF is separately attempted on the wrapper); thread completion
        alone NEVER proves anything. A shared absolute ``deadline_monotonic``
        (from restart) caps the relative ``budget_s``: the allowed end is
        min(entry+budget, shared deadline) and expiry adds no new thread/wait.
        The wrapper's stdin.close() remains the sole stdin closer (v2)."""
        budget = float(budget_s)
        if budget <= 0.0:
            return False
        key = id(wio)
        allowed_end = time.monotonic() + budget
        if deadline_monotonic is not None:
            allowed_end = min(allowed_end, float(deadline_monotonic))
        with self._ownership_lock:
            entry = self._wio_close_pending.get(key)
            first = entry is None
            if first:
                # Deadline recheck AFTER the leaf lock, BEFORE starting any
                # new thread: expiry starts nothing and leaves NO pending
                # marker behind (no phantom operation).
                if allowed_end - time.monotonic() <= 0.0:
                    return False
                done: threading.Event = threading.Event()
                outcome = {"state": "pending"}  # SHARED operation outcome
                self._wio_close_pending[key] = (wio, done, outcome)
            else:
                done, outcome = entry[1], entry[2]
        if first:

            def _close():
                try:
                    wio.close()
                    outcome["state"] = "ok"
                except Exception:
                    logger.debug(
                        "WorkerIO.close() during restart raised", exc_info=True
                    )
                    outcome["state"] = "error"
                finally:
                    done.set()

            threading.Thread(
                target=_close,
                name="asr-restart-wio-close",
                daemon=True,
            ).start()
        remaining = allowed_end - time.monotonic()
        completed = bool(remaining > 0.0) and done.wait(remaining)
        if completed:
            # The close operation ENDED (completed cleanly or raised) — it can
            # no longer be pending against this WIO.
            with self._ownership_lock:
                pending = self._wio_close_pending.get(key)
                if pending is not None and pending[0] is wio:
                    del self._wio_close_pending[key]
        if not completed:
            # Still pending when the allowed end passed (or no time left): no
            # claim either way about the outcome beyond "not completed here".
            return False
        if outcome["state"] != "ok":
            logger.warning(
                "ASR WorkerIO.close() RAISED during restart (shared operation "
                "outcome); the close thread ended but this is the close "
                "ATTEMPT failing, NOT proof or disproof of stdin EOF"
            )
            return False
        return True

    def _close_stdin_eof(
        self,
        worker: subprocess.Popen,
        *,
        budget_s: float = 1.0,
        deadline_monotonic: Optional[float] = None,
    ) -> bool:
        """Bounded stdin EOF **attempt** (no raw-fd close, no dup2 parking).

        Lock-chain diagnosis (2026-10-03): ``WorkerIO.request``/``send_request``
        write+flush INSIDE ``_stdin_lock`` — a flush blocked on a full pipe
        (native not reading) holds that lock; ``WorkerIO.close`` acquires the
        same lock WITHOUT a timeout; and ``TextIOWrapper.close()`` itself
        flushes buffered bytes, which is a blocking write into a full pipe.
        So a bare ``stdin.close()`` (or a synchronous ``wio.close()``) can hang
        forever and ``Popen.wait`` timeouts never bound it.

        v2 ownership rule: the ``TextIOWrapper`` is the ONLY owner allowed to
        close its own fd. We NEVER call ``os.close(fd)`` on it and NEVER
        ``dup2`` over it — either would let a later (blocked) wrapper flush or
        close touch an unrelated fd number that the OS reused. Instead we run
        the wrapper's own ``close()`` in a single bounded daemon thread and
        report honestly:

          * return True  — the wrapper close completed inside the allowed end
            (min(entry + ``budget_s``, shared deadline)) AND the wrapper's own
            ``closed`` flag is actually set (EOF delivered);
          * return False — close was still blocked when the allowed end passed,
            OR the wrapper close() RAISED (thread completion alone NEVER
            proves EOF: an exception with ``closed`` still False means NO EOF,
            and that failure outcome belongs to the SHARED operation, so
            reused callers see it too). The pending close keeps running and
            unwinds only when the child exits and drains the pipe; the caller
            proceeds to the bounded OWN TERM using the reserved budget and does
            NOT claim EOF was delivered. A completed FAILED attempt may be
            retried; while pending it is never duplicated.

        At most ONE close thread exists per stdin: a repeated survivor restart
        reuses the pending event. If the budget is already exhausted (<=0) no
        thread is started and no wait is added — the current closed state is
        reported as-is.
        """
        stdin = worker.stdin
        if stdin is None or getattr(stdin, "closed", False):
            return True
        budget = float(budget_s)
        if budget <= 0.0:
            # Expired budget: add NO thread and NO wait.
            return False
        allowed_end = time.monotonic() + budget
        if deadline_monotonic is not None:
            allowed_end = min(allowed_end, float(deadline_monotonic))
        key = id(stdin)
        with self._stdin_close_lock:
            entry = self._stdin_close_pending.get(key)
            first = entry is None
            if first:
                # Deadline recheck AFTER the leaf lock, BEFORE starting any
                # new thread: expiry starts nothing and leaves NO pending
                # marker behind (no phantom operation).
                if allowed_end - time.monotonic() <= 0.0:
                    return False
                done: threading.Event = threading.Event()
                outcome = {"state": "pending"}  # SHARED operation outcome
                # Retain the actual stdin wrapper with the operation so its
                # id() cannot be reused while the close is active.
                self._stdin_close_pending[key] = (stdin, done, outcome)
            else:
                done, outcome = entry[1], entry[2]
        if first:

            def _close():
                try:
                    stdin.close()
                    # Actual closed-flag proof, not thread completion: only a
                    # wrapper that reports itself closed delivered EOF.
                    outcome["state"] = (
                        "ok" if getattr(stdin, "closed", False) else "error"
                    )
                except Exception:
                    logger.debug(
                        "owned-worker stdin close raised", exc_info=True
                    )
                    outcome["state"] = "error"
                finally:
                    done.set()
                    with self._stdin_close_lock:
                        pending = self._stdin_close_pending.get(key)
                        if pending is not None and pending[0] is stdin:
                            del self._stdin_close_pending[key]

            threading.Thread(
                target=_close,
                name="asr-owned-stdin-close",
                daemon=True,
            ).start()
        remaining = allowed_end - time.monotonic()
        completed = bool(remaining > 0.0) and done.wait(remaining)
        if not completed:
            # Blocked flush (or expired shared slice): the ONE owned wrapper
            # close() thread is stuck writing to a full pipe. We do NOT
            # fabricate EOF via raw fd manipulation; we report the attempt as
            # NOT completed and let the caller proceed to its bounded owned
            # TERM. The pending close thread stays the sole owner of that
            # stdin and will complete when the child exits and drains.
            logger.warning(
                "ASR owned-worker stdin close() not completed within the "
                "allowed %.2fs; EOF ATTEMPT NOT completed (no raw-fd close "
                "performed); continuing to bounded owned TERM",
                max(0.0, time.monotonic() - (allowed_end - budget)),
            )
            return False
        if outcome["state"] == "ok":
            return True
        # Thread ENDED but the actual result is failure (close raised, or the
        # wrapper does not report itself closed). Report the ACTUAL outcome of
        # the SHARED operation: EOF is NOT claimed.
        logger.warning(
            "ASR owned-worker stdin close attempt FAILED (exception or "
            "closed flag not set); EOF NOT claimed from thread completion alone"
        )
        return False

    def _shutdown_owned_worker(
        self,
        worker: subprocess.Popen,
        *,
        deadline_s: float = 15.0,
        deadline_monotonic: Optional[float] = None,
        stdout_has_reader: bool = False,
    ) -> bool:
        """Bounded no-KILL teardown of a Popen THIS call owns.

        Sequence: stdin EOF ATTEMPT → bounded true-wait → terminate (SIGTERM to
        the child only, never kill/killpg, never a group signal to the
        inherited parent PGID) → bounded wait. ONE shared budget (default 15 s,
        not 15+15): the EOF phase gets at most 1 s or the remaining budget,
        whichever is smaller, and the EOF true-wait gets the first 2/3 of the
        budget so the owned-terminate wait always has the remaining ~1/3
        reserve. NO minimum floor is applied: once the (absolute) deadline has
        expired, this returns the CURRENT poll result immediately, adding no
        thread, wait, or signal. Returns True iff the process is observed
        exited within the deadline; on False the caller keeps the Popen
        reference (survivor) so a respawn cannot be layered over a live worker.

        BUDGET KINDS: ``deadline_monotonic`` (absolute, preferred — passed by
        ``restart_worker`` so the budget covers lock acquisition too) or the
        legacy relative ``deadline_s`` (default 15 s) from which an absolute
        deadline is derived. Public/legacy callers stay compatible.

        Drain ownership: stdout is drained here ONLY when no WorkerIO reader
        thread owns it (restart passes ``stdout_has_reader=True`` — that
        thread keeps consuming stdout until EOF even after ``close()``), and
        stderr only when the launch path's dedicated drainer is not already
        consuming it (``_stderr_drain_pids``). Two readers on the same TextIO
        would race and split lines. Only this worker's stdin is closed (by its
        own wrapper, in a bounded thread) — never an unbounded close on a pipe
        another thread may hold.
        """
        now = time.monotonic()
        deadline = (
            float(deadline_monotonic)
            if deadline_monotonic is not None
            else now + float(deadline_s)
        )
        if deadline - now <= 0.0:
            # Expired budget: report the CURRENT liveness without adding any
            # thread, wait, or signal.
            return worker.poll() is not None
        total = deadline - now
        eof_deadline = now + total * (2.0 / 3.0)
        # Budget recheck immediately before creating ANY helper thread: if the
        # deadline expired getting here, no drain/stdin thread is started.
        if deadline - time.monotonic() <= 0.0:
            return worker.poll() is not None
        # Active draining first: prevents an EOF-wait block on a full pipe —
        # but only for streams this call uniquely owns. stdout ownership is
        # tracked per ACTUAL stream (_stdout_drain_pending), so exactly ONE
        # reader exists across ready-rejection teardowns and repeated failed
        # restarts of the same survivor. The shared absolute deadline is
        # passed down so no helper thread is created after expiry.
        if not stdout_has_reader:
            self._start_stdout_drain_once(worker, deadline_monotonic=deadline)
        if (
            worker.pid not in self._stderr_drain_pids
            and worker.stderr is not None
            and not getattr(worker.stderr, "closed", False)
        ):
            # Deadline recheck immediately before creating the stderr helper:
            # expiry adds no new thread.
            if deadline - time.monotonic() <= 0.0:
                return worker.poll() is not None
            threading.Thread(
                target=self._drain_to_void,
                args=(worker.stderr,),
                name="asr-worker-teardown-drain-stderr",
                daemon=True,
            ).start()
        # Prompt the wrapper to close stdin; the wait is bounded by the SAME
        # deadline. When stdin is not already closed, the wrapper's own
        # close() thread is the sole closer — no raw fd / dup2 here.
        if worker.stdin is not None and not getattr(worker.stdin, "closed", False):
            eof_budget = min(1.0, max(0.0, eof_deadline - time.monotonic()))
            try:
                self._close_stdin_eof(
                    worker, budget_s=eof_budget, deadline_monotonic=deadline
                )
            except Exception:
                logger.debug("owned stdin EOF attempt raised", exc_info=True)
        try:
            worker.wait(timeout=max(0.0, eof_deadline - time.monotonic()))
            return True
        except subprocess.TimeoutExpired:
            pass
        # Budget recheck immediately before OWN TERM: if the EOF wait returned
        # at/after the shared deadline, do NOT send TERM after expiry — report
        # the current liveness and keep the Popen owned fail-closed.
        if deadline - time.monotonic() <= 0.0:
            return worker.poll() is not None
        try:
            worker.terminate()
        except Exception:
            logger.debug("owned-worker terminate raised", exc_info=True)
        try:
            worker.wait(timeout=max(0.0, deadline - time.monotonic()))
            return True
        except subprocess.TimeoutExpired:
            return False

    # -- ASRBackend interface ------------------------------------------------

    @property
    def name(self) -> str:
        return "trt_edgellm"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {
            ASRCapability.OFFLINE,
            ASRCapability.MULTI_LANGUAGE,
            ASRCapability.STREAMING,
        }

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        """Verify all required files exist."""
        cfg = self._config
        worker_binary = cfg.worker_binary
        asr_binary = cfg.asr_binary
        plugin_path = cfg.plugin_path
        engine_dir = cfg.engine_dir
        audio_encoder_dir = cfg.audio_encoder_dir
        required = [
            (worker_binary if self._use_worker() else asr_binary, "ASR binary"),
            (plugin_path, "TRT-Edge-LLM plugin"),
            (os.path.join(engine_dir, "config.json"), "LLM config"),
            (os.path.join(engine_dir, "llm.engine"), "LLM engine"),
            (os.path.join(audio_encoder_dir, "audio", "config.json"), "audio encoder config"),
            (os.path.join(audio_encoder_dir, "audio", "audio_encoder.engine"), "audio encoder engine"),
        ]
        missing = [(path, label) for path, label in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "ASR preload failed — missing:\n  "
                + "\n  ".join(f"{l}: {p}" for p, l in missing)
            )
        self._require_streaming_worker_assets()

        logger.info("ASR backend preload OK (config=%s)", self._config)
        if self._use_worker():
            with self._worker_lock:
                self._ensure_worker()
        self._ready = True
        if self._use_worker():
            self._warm_worker()

    def unload(self) -> None:
        """Kill the resident ASR worker subprocess to fully release GPU memory."""
        if not self._ready and self._worker is None:
            return
        try:
            self.restart_worker()
        except Exception:
            logger.exception("TRTEdgeLLMASRBackend.unload failed; continuing")
        finally:
            self._ready = False

    def _warm_worker(self) -> None:
        """Pre-warm TRT audio_encoder optimization profile for batch shapes 1..N."""
        if not self._config.worker_warmup:
            logger.info("TRT-EdgeLLM ASR worker warmup skipped.")
            return
        prewarm_max = max(1, min(int(self._config.prewarm_max), 60))
        import time as _time

        t0 = _time.monotonic()
        warmed = 0
        for seconds in range(1, prewarm_max + 1):
            try:
                silence = np.zeros(16000 * seconds, dtype=np.float32)
                self.transcribe(_float_audio_to_wav_bytes(silence, 16000))
                warmed += 1
            except Exception as exc:
                msg = str(exc)
                if "cannot handle" in msg or "TensorRT Edge LLM" in msg:
                    logger.info(
                        "TRT-EdgeLLM ASR pre-warm: engine boundary at batch=%d "
                        "(expected, stopping)", seconds,
                    )
                else:
                    logger.warning(
                        "TRT-EdgeLLM ASR pre-warm batch=%d failed: %s", seconds, exc
                    )
                break
        elapsed = _time.monotonic() - t0
        logger.info(
            "TRT-EdgeLLM ASR worker pre-warmed shapes 1..%d in %.1fs", warmed, elapsed
        )

    def _use_worker(self) -> bool:
        return bool(self._config.use_worker)

    def _use_streaming_worker(self) -> bool:
        return self._config.stream_mode in (
            "worker", "stream", "streaming", "chunk_confirm", "prefix"
        )

    def _require_streaming_worker_assets(self) -> None:
        if not self._use_streaming_worker():
            return
        if self._config.request_audio_wav:
            # WAV-ingest mode (v0.9.0+): the worker extracts mel internally from
            # a temp WAV, so host-side mel assets are not needed. _ensure_worker
            # already omits --melSettings/--melFilters when they are unset.
            logger.info(
                "ASR WAV-ingest mode (EDGELLM_REQUEST_AUDIO_WAV=1): "
                "skipping host-side mel-asset requirement."
            )
            return
        missing = []
        if not self._use_worker():
            missing.append("use_worker=True is required for streaming worker mode")
        for value, label in (
            (self._config.mel_settings_path, "mel_settings_path"),
            (self._config.mel_filters_path, "mel_filters_path"),
        ):
            if not value or not os.path.exists(value):
                missing.append(f"{label}: {value or '(unset)'}")
        if missing:
            raise FileNotFoundError(
                "stream_mode=worker requires PCM mel assets:\n  "
                + "\n  ".join(missing)
            )

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        env.update(self._config.extra_worker_env)
        env["EDGELLM_PLUGIN_PATH"] = self._config.plugin_path
        env.setdefault("EDGE_LLM_ASR_CUDA_GRAPH", self._config.worker_cuda_graph)
        # Keep the worker's audio-ingest mode consistent with the preload guard
        # (setdefault so an explicit profile/env value still wins).
        env.setdefault(
            "EDGELLM_REQUEST_AUDIO_WAV",
            "1" if self._config.request_audio_wav else "0",
        )
        return env

    def _drain_worker_stderr(self, worker: subprocess.Popen) -> None:
        if worker.stderr is None:
            return
        for line in worker.stderr:
            text = line.rstrip()
            self._worker_stderr_tail.append(text)
            if "[JV_MEM]" in text:
                logger.info("ASR worker: %s", text)
            else:
                logger.debug("ASR worker stderr: %s", text)

    def _stderr_tail_text(self) -> str:
        return "\n".join(self._worker_stderr_tail)

    def _ensure_worker(self) -> None:
        """Launch the resident worker if none is live.

        LIFECYCLE CONTRACT: the caller MUST hold ``self._worker_lock``. This
        method mutates ``self._worker`` / ``self._wio`` and must never be
        invoked concurrently with ``restart_worker`` (which also holds
        ``_worker_lock`` for its whole teardown). Keeping the launch under the
        same lock that restart uses is what prevents a second process from
        being spawned during a teardown.
        """
        if self._worker is not None and self._worker.poll() is None:
            # A live Popen stays referenced. If it is a failed-ready survivor
            # (_wio is None) we still must NOT respawn over it; the request
            # paths reject explicitly when the pair is unusable.
            return
        cmd = [
            self._config.worker_binary,
            "--engineDir",
            self._config.engine_dir,
            "--multimodalEngineDir",
            self._config.audio_encoder_dir,
        ]
        # Only emit --max_slots when N>1 (main fix b1cb1a5): at N=1 we omit it
        # for byte-equivalent legacy behavior and back-compat with worker
        # binaries built before --max_slots existed.
        if self._max_slots and self._max_slots > 1:
            cmd += ["--max_slots", str(self._max_slots)]
        mel_settings = self._config.mel_settings_path or ""
        mel_filters = self._config.mel_filters_path or ""
        if mel_settings and mel_filters:
            cmd += ["--melSettings", mel_settings, "--melFilters", mel_filters]
        self._worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
        )
        self._worker_stderr_tail.clear()
        threading.Thread(
            target=self._drain_worker_stderr,
            args=(self._worker,),
            name="trt-edgellm-asr-stderr",
            daemon=True,
        ).start()
        self._stderr_drain_pids.add(self._worker.pid)
        assert self._worker.stdout is not None
        ready_line = self._worker.stdout.readline()
        worker = self._worker
        failure: Optional[str] = None
        ready: dict = {}
        if not ready_line:
            failure = (
                "ASR worker failed to start (EOF before ready): "
                + self._stderr_tail_text()
            )
        else:
            try:
                ready = json.loads(ready_line)
            except Exception as exc:
                failure = (
                    f"ASR worker ready line is not valid JSON: "
                    f"{ready_line[:200]!r} ({exc})"
                )
            if failure is None and (
                not isinstance(ready, dict) or ready.get("event") != "ready"
            ):
                failure = f"ASR worker did not become ready: {ready}"
            if failure is None and self._config.require_ifb:
                rejection = self._ifb_ready_rejection(
                    ready, required_slots=max(1, int(self._config.max_slots))
                )
                if rejection is not None:
                    failure = f"ASR worker ready IFB provenance rejected: {rejection}"
        if failure is not None:
            # Bounded OWN cleanup of the worker this call just launched:
            # stdin EOF → bounded true-wait → owned terminate → bounded wait.
            # Never kill/killpg; no signals to the inherited parent PGID.
            # stdout has NO WorkerIO reader yet (backend owns it); the launch
            # path's stderr drainer already owns stderr.
            exited = self._shutdown_owned_worker(
                worker, stdout_has_reader=False
            )
            # Previous ready state must NEVER survive a launch/ready failure.
            self._ready = False
            if exited:
                # Confirmed retired: clear the Popen AND any stale WIO so the
                # pair invariant is local.
                self._worker = None
                self._wio = None
                self._worker_failed = False
                self._worker_failed_reason = None
            else:
                # Survivor: keep the Popen reference so _ensure_worker cannot
                # respawn a second worker over the still-live one; there is no
                # WIO here (launch path), but the teardown's stdout drain
                # ownership is retained in _stdout_drain_pending so a later
                # restart of this survivor cannot start a second stdout
                # reader. Mark unusable so requests reject explicitly.
                self._wio = None
                self._worker_failed = True
                self._worker_failed_reason = f"ready handshake failed: {failure}"
                logger.error(
                    "ASR worker ready handshake failed and worker pid=%s "
                    "survived EOF+TERM within the shared deadline; keeping the "
                    "reference to prevent respawn",
                    worker.pid,
                )
            raise RuntimeError(failure)
        # Successful owned launch: reset stale engine diagnostics so no health
        # from a previous worker process leaks into this one's provenance.
        with self._diag_lock:
            self._latest_ifb_health = None
            self._ifb_traces.clear()
            self._last_health_log_at = 0.0
        self._worker_ready_meta = ready
        self._max_slots = max(1, int(self._config.max_slots))
        # A successful owned fresh launch is the ONLY place (besides a
        # confirmed retired exit) that clears the per-worker failure marker.
        self._worker_failed = False
        self._worker_failed_reason = None
        # NB: ``_ensure_worker`` reads the worker's initial ``ready`` line
        # itself (above) BEFORE handing stdout to the WorkerIO reader thread.
        self._wio = WorkerIO(
            self._worker,
            concurrency=self._max_slots,
            telemetry_callback=self._on_worker_telemetry,
        )

    @staticmethod
    def _ifb_ready_rejection(
        ready: dict, *, required_slots: int
    ) -> Optional[str]:
        """Opt-in v011 ready-provenance validation. Returns a rejection detail
        string, or None when the ready event proves IFB capability for the
        configured slot count. Bool is NOT accepted as an integer value."""
        if ready.get("ifb") is not True:
            return f"ready.ifb must be true, got {ready.get('ifb')!r}"
        for key in ("max_slots", "gpu_slots"):
            value = ready.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                return (
                    f"ready.{key} must be an integer (bool not accepted), "
                    f"got {value!r}"
                )
            if value != required_slots:
                return f"ready.{key}={value} != configured slots {required_slots}"
        batch = ready.get("engine_max_batch_size")
        if isinstance(batch, bool) or not isinstance(batch, int):
            return (
                "ready.engine_max_batch_size must be an integer, "
                f"got {batch!r}"
            )
        if batch < required_slots:
            return (
                f"ready.engine_max_batch_size={batch} < configured slots "
                f"{required_slots}"
            )
        return None

    def _on_worker_telemetry(self, event: dict) -> None:
        """WorkerIO telemetry sink: cache latest IFB health + bounded traces.

        The NATIVE worker emits telemetry with the canonical ``type`` key
        (qwen3_asr_worker lifecycle health_final:
        ``{"type":"asr_ifb_health",...}``); ``event`` is a compat alias.
        Rate-bounded, audio-free logging carries the ACTUAL native stall
        counter keys (stalls_founder_only / stalls_guided /
        stalls_incompatible / stalls_no_capacity) plus admitted_mid_flight —
        there is no single ``stalls`` key in the source. Exceptions never
        propagate into the WorkerIO reader (they are isolated there too).
        """
        kind = event.get("type") or event.get("event")
        with self._diag_lock:
            if kind == "asr_ifb_health":
                self._latest_ifb_health = dict(event)
            elif kind == "asr_ifb_trace":
                self._ifb_traces.append(dict(event))
        if kind == "asr_ifb_health":
            now = time.monotonic()
            with self._diag_lock:
                due = now - self._last_health_log_at >= 10.0
                if due:
                    self._last_health_log_at = now
            if due:
                stall_counters = {
                    key: value
                    for key, value in sorted(event.items())
                    if key.startswith("stalls_")
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                }
                logger.info(
                    "ASR worker asr_ifb_health: stall_counters=%r "
                    "admitted_mid_flight=%r resident=%r queued=%r",
                    stall_counters,
                    event.get("admitted_mid_flight"),
                    event.get("resident"),
                    event.get("queued"),
                )

    def runtime_diagnostics(self) -> dict:
        """Additive runtime provenance snapshot (validated ready + latest IFB
        health + bounded traces). Copies only; no capability override, no HTTP
        exposure claim — HTTP surfacing is a separate integration task."""
        with self._diag_lock:
            health = (
                dict(self._latest_ifb_health) if self._latest_ifb_health else None
            )
            traces = [dict(t) for t in self._ifb_traces]
        return {
            "require_ifb": bool(getattr(self._config, "require_ifb", False)),
            "ready": self._ready,
            # Live snapshot from the actual owned Popen: a cached ready=True
            # after the worker died must NOT qualify IFB provenance.
            "worker_alive": (
                self._worker.poll() is None
                if self._worker is not None
                else None
            ),
            "worker_pid": self._worker.pid if self._worker is not None else None,
            "validated_ready": dict(self._worker_ready_meta),
            "latest_asr_ifb_health": health,
            "asr_ifb_traces": traces,
            "asr_ifb_trace_capacity": 64,
        }

    def _clear_worker_if_current(self, worker, wio, *, reason: str = "") -> None:
        """Compare-and-handle the captured ownership pair under the lock.

        A request error (``WorkerExitError`` / broken pipe / no response) is NOT
        proof that the process exited. So:

          * if the current pair still matches the captured pair AND the captured
            Popen is PROVEN EXITED (``poll() is not None``), clear BOTH the
            Popen and its own WorkerIO atomically and reset the failure marker;
          * if it matches but the Popen is STILL LIVE, retain the Popen AND its
            WorkerIO reader ownership and mark the worker unusable / not-ready.
            No respawn is attempted; later requests reject explicitly.

        If the current pair does not match (a newer worker was installed), this
        is a no-op — an OLDER failure must never mark or discard a newer worker.
        """
        with self._worker_lock:
            if self._worker is worker and self._wio is wio:
                if worker is not None and worker.poll() is None:
                    # Still live: keep the pair, fail closed.
                    self._worker_failed = True
                    self._worker_failed_reason = reason or "request failed"
                    self._ready = False
                else:
                    # Proven exited (or no worker): drop the pair together.
                    self._worker = None
                    self._wio = None
                    self._worker_ready_meta = {}
                    self._worker_failed = False
                    self._worker_failed_reason = None

    def _worker_request(
        self,
        input_data: dict,
        *,
        expected_cancel: Optional[threading.Event] = None,
        on_written: Optional["Callable[[], None]"] = None,
    ) -> dict:
        """Send one streaming protocol line to the worker, return its single reply.

        ``on_written`` is forwarded to the underlying ``WorkerIO.request`` ONLY
        when supplied (kept default-off so every existing caller/request is
        byte-for-byte unchanged). ``expected_cancel`` is a RESPONSE-classification
        marker only and is deliberately NOT passed to ``WorkerIO.request`` as a
        ``cancel_event`` — doing so would make WorkerIO emit its own second,
        autonomous cancel, duplicating the explicit ``_worker_cancel_and_wait``
        control write.
        """
        req_event = input_data.get("event") if isinstance(input_data, dict) else None
        req_id = input_data.get("id") if isinstance(input_data, dict) else None
        with self._worker_lock:
            self._ensure_worker()
            worker = self._worker
            wio = self._wio
            failed = self._worker_failed
            failed_reason = self._worker_failed_reason
        if worker is None or wio is None or failed:
            # Explicit rejection, NOT an assert: a failed-ready survivor keeps
            # a live Popen with no usable WorkerIO, and a failed request against
            # a still-live worker marks it unusable. Either way no request may
            # be sent and no respawn is attempted over the live process.
            stderr = self._stderr_tail_text()
            raise WorkerExitError(
                "ASR worker unavailable ("
                f"{'failed: ' + failed_reason if failed else ('live process without WorkerIO' if worker is not None else 'no worker')}"
                f"): {stderr}"
            )
        try:
            output_data: Optional[dict] = None
            if on_written is None:
                gen = wio.request(input_data)
            else:
                gen = wio.request(input_data, on_written=on_written)
            try:
                for ev in gen:
                    output_data = ev
                    break
            finally:
                gen.close()
        except _WIOExitError as exc:
            stderr = self._stderr_tail_text()
            self._clear_worker_if_current(
                worker, wio, reason=f"worker exited mid-request: {exc}"
            )
            raise WorkerExitError(
                f"ASR worker exited before response: {exc}: {stderr}"
            ) from exc
        except (BrokenPipeError, OSError) as exc:
            stderr = self._stderr_tail_text()
            self._clear_worker_if_current(
                worker, wio, reason=f"worker stdin broken: {exc}"
            )
            raise WorkerExitError(
                f"ASR worker stdin broken (likely killed): {exc}: {stderr}"
            ) from exc
        if output_data is None:
            stderr = self._stderr_tail_text()
            self._clear_worker_if_current(
                worker, wio, reason="worker produced no response"
            )
            raise WorkerExitError(f"ASR worker exited before response: {stderr}")
        if (
            expected_cancel is not None
            and expected_cancel.is_set()
            and isinstance(req_id, str)
            and req_id
            and output_data.get("event") == "cancelled"
            and output_data.get("ok") is False
            and output_data.get("id") == req_id
        ):
            # Opt-in expected-cancel arm: while a cancel is genuinely requested
            # for this very request, return the ACTUAL dict to the ordinary
            # consumer instead of the generic ok=false WorkerProtocolError.
            # No marker clear, no SID release, no receipt logic beyond this.
            return output_data
        typed = _classify_worker_response(output_data, request_event=req_event)
        if typed is not None:
            raise typed
        # Native handleEnd uses this error after releasing an empty session's
        # lane and erasing its session.  Treat only the matching end request as
        # a normal terminal response; all other errors retain their contract.
        if (
            req_event == "end"
            and output_data.get("event") == "error"
            and output_data.get("ok") is False
            and output_data.get("error") == "no_audio_accumulated"
            and req_id is not None
            and output_data.get("id") == req_id
        ):
            return output_data
        if output_data.get("event") == "error" or output_data.get("ok") is False:
            raise WorkerProtocolError(f"ASR worker error: {output_data}")
        return output_data

    def _worker_cancel_and_wait(self, sid: str, timeout_s: float) -> dict:
        """Send a native cancel for ``sid`` and return the ACTUAL receipt dict.

        Transport-only foundation (NO stream-state / legacy-cancel migration):
        this is the backend's thin, ownership-safe wrapper over
        ``WorkerIO.cancel_and_wait``. It never routes through the ordinary
        ``_worker_request`` line discipline and never touches the consumer
        cancellation carve-out (none exists here yet).

        Validation happens BEFORE any stdin write and BEFORE the worker
        snapshot/lookup, using the SAME canonical rules WorkerIO enforces
        (non-empty str id; finite, positive, non-bool timeout). The snapshot of
        ``(_worker, _wio, _worker_failed, _worker_failed_reason)`` is taken
        under ``self._worker_lock`` and the lock is RELEASED before the WIO
        call — the blocking cancel/receipt wait must never hold the backend
        lock. This method NEVER calls ``_ensure_worker`` and NEVER spawns or
        preloads a worker; an absent/failed/unusable pair is an explicit
        rejection using the EXISTING ``WorkerExitError`` message pattern.

        The captured pair is passed straight to the ACTUAL ``wio.cancel_and_wait``
        and the matching native dict is returned verbatim (id/epoch/ok=false) —
        no dummy ACK and no shape assumptions; WIO guarantees the receipt
        shape. ``TimeoutError`` / ``ValueError`` / ``RuntimeError``
        (capacity/duplicate) propagate UNCHANGED and must NOT clear the pair,
        restart, or mark the worker failed: the worker and its handles remain
        caller-owned. A genuine worker exit (``WorkerIO`` ``WorkerExitError``)
        maps to the existing backend ``WorkerExitError`` and is the only case
        that runs the existing ``_clear_worker_if_current`` compare-and-handle
        (which itself only clears a PROVEN-exited captured pair); broken stdin
        (``BrokenPipeError``/``OSError``) follows the same broken-stdin policy
        as ``_worker_request``. No blanket ``except`` clause is used.

        KNOWN LIMITATION (deliberate, do not overclaim): the underlying
        ``TextIO.write`` + ``flush`` inside WorkerIO may block for an opaque,
        OS/pipe-bounded duration. This helper therefore carries NO hard-deadline
        guarantee for the whole call and MUST be run off the event loop by a
        caller/supervisor that retains live writers on timeout.
        """
        # --- Validate BEFORE any write or worker lookup (reuse WIO rules). ---
        if not isinstance(sid, str) or not sid:
            raise ValueError(f"_worker_cancel_and_wait: invalid sid={sid!r}")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError(
                f"_worker_cancel_and_wait: invalid timeout_s={timeout_s!r}"
            )

        # --- Snapshot ownership under the lock, then RELEASE it. ---
        with self._worker_lock:
            worker = self._worker
            wio = self._wio
            failed = self._worker_failed
            failed_reason = self._worker_failed_reason

        if worker is None or wio is None or failed:
            # Explicit rejection, NOT an assert and NOT a respawn: mirrors
            # ``_worker_request``; no request is sent over a live-but-unusable
            # survivor, and no ``_ensure_worker`` is attempted here.
            stderr = self._stderr_tail_text()
            raise WorkerExitError(
                "ASR worker unavailable ("
                f"{'failed: ' + failed_reason if failed else ('live process without WorkerIO' if worker is not None else 'no worker')}"
                f"): {stderr}"
            )

        # Lock is released above; the blocking wait runs uncoupled from it.
        try:
            return wio.cancel_and_wait(sid, timeout_s)
        except _WIOExitError as exc:
            stderr = self._stderr_tail_text()
            self._clear_worker_if_current(
                worker, wio, reason=f"worker exited during cancel: {exc}"
            )
            raise WorkerExitError(
                f"ASR worker exited before cancel receipt: {exc}: {stderr}"
            ) from exc
        except TimeoutError:
            # CRITICAL ORDERING: the builtin ``TimeoutError`` subclasses
            # ``OSError``, so this arm MUST precede the ``(BrokenPipeError,
            # OSError)`` arm. A receipt timeout is NOT a broken stdin: it
            # propagates UNCHANGED and must NEVER clear the pair, mark the
            # worker failed, or trigger a restart — the worker and its handles
            # stay caller-owned.
            raise
        except (BrokenPipeError, OSError) as exc:
            stderr = self._stderr_tail_text()
            self._clear_worker_if_current(
                worker, wio, reason=f"worker stdin broken during cancel: {exc}"
            )
            raise WorkerExitError(
                f"ASR worker stdin broken during cancel (likely killed): {exc}: {stderr}"
            ) from exc

    def restart_worker(self) -> None:
        """Gracefully retire the resident worker (stdin EOF attempt → bounded
        true-wait → owned terminate → bounded wait; NEVER kill/killpg) so the
        next request launches a fresh one.

        LOCK ORDER / BUDGET: ONE absolute monotonic 15 s deadline is set BEFORE
        either lock is acquired. BOTH ``_restart_lock`` and ``_worker_lock`` are
        acquired with ``acquire(timeout=remaining)`` from that same deadline, so
        a wedged holder of either lock produces a bounded explicit error instead
        of an unbounded wait past the budget; on timeout NOTHING about the
        Popen/WIO ownership is changed. The whole teardown then runs under
        ``_worker_lock`` (so ``_ensure_worker`` — the only launch site — cannot
        spawn a second process while the old one terminates), and every close /
        helper wait shares the SAME absolute deadline (``deadline_monotonic``).
        ``_worker``/``_wio`` stay referenced until exit is CONFIRMED; the pair
        is cleared atomically only then.
        """
        deadline = time.monotonic() + 15.0
        acquired_restart = self._restart_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        )
        if not acquired_restart:
            raise RuntimeError(
                "ASR worker restart could not acquire lifecycle restart lock "
                "within the shared 15s budget; Popen/WorkerIO ownership "
                "unchanged"
            )
        try:
            acquired_worker = self._worker_lock.acquire(
                timeout=max(0.0, deadline - time.monotonic())
            )
            if not acquired_worker:
                raise RuntimeError(
                    "ASR worker restart could not acquire lifecycle worker lock "
                    "within the shared 15s budget; Popen/WorkerIO ownership "
                    "unchanged"
                )
            try:
                worker = self._worker
                if worker is None:
                    return
                # Ownership markers stay in place through teardown; only
                # _worker_ready_meta is invalidated up front.
                self._worker_ready_meta = {}
                wio = self._wio
                # If the initial locks consumed the entire shared budget, no
                # WIO close thread is started at all.
                if wio is not None and deadline - time.monotonic() > 0.0:
                    # Bounded by min(1 s, remaining shared budget): the 1 s wait
                    # is INSIDE the one shutdown budget, not additive. The close
                    # operation is bound to the ACTUAL WIO object, so a repeated
                    # failed restart reuses the pending close instead of
                    # starting another blocked close thread for the same WIO.
                    wio_wait = min(1.0, deadline - time.monotonic())
                    if not self._wio_close_bounded(
                        wio, budget_s=wio_wait, deadline_monotonic=deadline
                    ):
                        # A request thread blocked in stdin flush holds
                        # _stdin_lock; WorkerIO.close() can never finish until
                        # the pipe drains/child exits. Do NOT wait unboundedly:
                        # proceed with the bounded owned teardown, which asks the
                        # wrapper to close stdin (sole closer) and then TERMs.
                        logger.warning(
                            "ASR worker restart: WorkerIO.close() did not "
                            "complete within %.2fs (writer blocked on "
                            "_stdin_lock with an unread pipe); proceeding with "
                            "bounded owned teardown",
                            wio_wait,
                        )
                if worker.poll() is None:
                    # No-KILL teardown sharing the absolute deadline from
                    # restart entry (covers lock acquisition, wio close, EOF
                    # attempt, true-wait and TERM). Drain ownership: wio's
                    # reader thread keeps consuming stdout until EOF after
                    # close() — do NOT start a second reader; the launch path's
                    # stderr drainer likewise still owns stderr
                    # (_stderr_drain_pids).
                    exited = self._shutdown_owned_worker(
                        worker,
                        deadline_monotonic=deadline,
                        stdout_has_reader=wio is not None,
                    )
                    if not exited:
                        # Survivor: keep the Popen referenced (so no respawn can
                        # be layered over it) and RETAIN the actual WorkerIO:
                        # its reader thread still owns stdout until EOF, and
                        # dropping the reference here would let a later restart
                        # pass stdout_has_reader=False and start a SECOND reader
                        # on the same live pipe. The WIO is merely unusable for
                        # requests; the failure marker (below) is what makes
                        # subsequent requests reject explicitly, and a later
                        # restart reuses the pending WIO-close ownership.
                        self._worker = worker
                        self._wio = wio
                        self._ready = False
                        self._worker_failed = True
                        self._worker_failed_reason = (
                            "restart survivor (EOF attempt + TERM exceeded "
                            "shared deadline)"
                        )
                        logger.error(
                            "ASR worker restart: pid=%s survived EOF attempt + "
                            "TERM within the shared deadline; reference kept, "
                            "ready=False",
                            worker.pid,
                        )
                        raise RuntimeError(
                            "ASR worker restart failed: worker pid="
                            f"{worker.pid} survived stdin-EOF attempt + "
                            "SIGTERM within the shared deadline; reference kept, "
                            "ready=False"
                        )
                # Confirmed exited (or already dead): clear the pair atomically
                # and reset the failure marker. Stdin EOF was attempted by
                # _shutdown_owned_worker for a live process; for an already-dead
                # process there is nothing left to flush. No redundant second
                # EOF pass — that would add budget outside the shared deadline.
                # stdout/stderr stay with their drainer threads until EOF.
                self._worker = None
                self._wio = None
                self._worker_failed = False
                self._worker_failed_reason = None
            finally:
                self._worker_lock.release()
            # Successful owned restart: drop any diagnostics from the discarded
            # worker so no stale engine health/trace survives into the next
            # launch.
            with self._diag_lock:
                self._latest_ifb_health = None
                self._ifb_traces.clear()
                self._last_health_log_at = 0.0
        finally:
            self._restart_lock.release()
        logger.info("ASR worker restarted (will respawn on next request)")

    def _postprocess_text(self, text: str) -> tuple[str, Optional[str]]:
        """所有 ASR 文本的公共出口：剥语言前缀 + 收掉自回归退化的整段复读。

        离线与流式都经过这里 —— 两条路径在短音频上产出逐字相同的退化输出
        （2026-08-08 orin-nx 实测），所以守卫必须放在共用位置而非流式那一侧。
        """
        text, language_detected = self._strip_language_prefix(text)
        if text and self._config.collapse_repetition:
            collapsed, did = collapse_repetition(text)
            if did:
                logger.warning(
                    "ASR 输出疑似解码退化，已塌缩: %r -> %r",
                    text[:80], collapsed,
                )
                text = collapsed
        return text, language_detected

    @staticmethod
    def _strip_language_prefix(text: str) -> tuple[str, Optional[str]]:
        language_detected = None
        if text and len(text) >= 9 and text[:9] == "language ":
            known_languages = (
                "Chinese", "English", "Cantonese", "Japanese", "Korean",
                "French", "German", "Italian", "Portuguese", "Russian",
                "Spanish",
            )
            # 显式 native 解码头标记 '<asr_text>' 是语言标签的硬边界，
            # 且必须先于 legacy 已知前缀循环判定：完整标记前的头部为空/
            # 纯空白 => 不发明语言，保留整段输入；头部为非空且无空白的
            # 单词标签 => 整个 native 标签（已知或未知），正文从精确标记
            # 处开始。多词头部不当作标签，回退 legacy 处理。
            marker_pos = text.find("<asr_text>", 9)
            if marker_pos != -1:
                header = text[9:marker_pos].strip()
                if not header:
                    return text, None
                if len(header.split()) == 1:
                    language_detected = header
                    text = text[marker_pos:]
            if language_detected is None:
                for name in known_languages:
                    prefix = f"language {name}"
                    if text.startswith(prefix):
                        language_detected = name
                        text = text[len(prefix):].lstrip()
                        break
                else:
                    space = text.find(" ", 9)
                    if space > 0:
                        language_detected = text[9:space]
                        text = text[space + 1:].lstrip()
                    else:
                        label = text[9:]
                        if label.strip():
                            language_detected = label
                            text = ""
                        else:
                            # 空头标签：不发明语言，保留输入而不是删掉正文。
                            return text, None
        # native 解码可能把控制标记原样吐在正文最前面（NX 实测
        # '{"text":"<asr_text>Concord returned ..."}'）。只剥这一个
        # 完整的引导标记及其周围的定界空白；正文中间出现的字面
        # '<asr_text>' 和不完整的 '<asr_' 前缀一律保留。
        if text:
            stripped = text.lstrip()
            if stripped.startswith("<asr_text>"):
                text = stripped[len("<asr_text>"):].lstrip()
        return text, language_detected

    @staticmethod
    def _canonical_worker_language(language: str) -> Optional[str]:
        aliases = {
            "zh": "Chinese", "zh-cn": "Chinese", "chinese": "Chinese",
            "en": "English", "en-us": "English", "english": "English",
            "yue": "Cantonese", "cantonese": "Cantonese",
            "ja": "Japanese", "japanese": "Japanese", "ko": "Korean",
            "korean": "Korean", "fr": "French", "french": "French",
            "de": "German", "german": "German", "it": "Italian",
            "italian": "Italian", "pt": "Portuguese", "portuguese": "Portuguese",
            "ru": "Russian", "russian": "Russian", "es": "Spanish",
            "spanish": "Spanish", "ar": "Arabic", "arabic": "Arabic",
            "hi": "Hindi", "hindi": "Hindi", "bn": "Bengali", "bengali": "Bengali",
            "ur": "Urdu", "urdu": "Urdu", "id": "Indonesian", "indonesian": "Indonesian",
            "ms": "Malay", "malay": "Malay", "vi": "Vietnamese", "vietnamese": "Vietnamese",
            "th": "Thai", "thai": "Thai", "tr": "Turkish", "turkish": "Turkish",
            "nl": "Dutch", "dutch": "Dutch", "pl": "Polish", "polish": "Polish",
            "uk": "Ukrainian", "ukrainian": "Ukrainian", "sv": "Swedish", "swedish": "Swedish",
            "no": "Norwegian", "norwegian": "Norwegian", "da": "Danish", "danish": "Danish",
            "fi": "Finnish", "finnish": "Finnish", "el": "Greek", "greek": "Greek",
            "he": "Hebrew", "hebrew": "Hebrew", "fa": "Persian", "persian": "Persian",
            "cs": "Czech", "czech": "Czech", "sk": "Slovak", "slovak": "Slovak",
            "hu": "Hungarian", "hungarian": "Hungarian", "ro": "Romanian", "romanian": "Romanian",
            "bg": "Bulgarian", "bulgarian": "Bulgarian", "hr": "Croatian", "croatian": "Croatian",
            "sr": "Serbian", "serbian": "Serbian", "ta": "Tamil", "tamil": "Tamil",
            "te": "Telugu", "telugu": "Telugu", "mr": "Marathi", "marathi": "Marathi",
            "gu": "Gujarati", "gujarati": "Gujarati", "kn": "Kannada", "kannada": "Kannada",
            "ml": "Malayalam", "malayalam": "Malayalam", "pa": "Punjabi", "punjabi": "Punjabi",
            "ne": "Nepali", "nepali": "Nepali", "si": "Sinhala", "sinhala": "Sinhala",
            "my": "Burmese", "burmese": "Burmese", "km": "Khmer", "khmer": "Khmer",
            "lo": "Lao", "lao": "Lao", "mn": "Mongolian", "mongolian": "Mongolian",
            "bo": "Tibetan", "tibetan": "Tibetan", "ug": "Uyghur", "uyghur": "Uyghur",
        }
        requested = str(language or "auto").strip()
        if not requested or requested.lower() == "auto":
            return None
        if not requested.isascii() or any(ord(ch) < 32 for ch in requested):
            raise ValueError(f"unsupported ASR worker language: {language!r}")
        canonical = aliases.get(requested.lower())
        if canonical is None:
            raise ValueError(f"unsupported ASR worker language: {language!r}")
        return canonical

    def _transcribe_worker(
        self, mel_path: str, elapsed_mel_s: float, language: str = "auto"
    ) -> TranscriptionResult:
        req_id = uuid.uuid4().hex
        canonical_language = self._canonical_worker_language(language)
        input_data = {
            "id": req_id,
            "requests": [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "audio", "audio": mel_path}],
                        }
                    ],
                }
            ],
            "batch_size": 1,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "top_k": self._config.top_k,
            "max_generate_length": self._config.max_generate_length,
            "apply_chat_template": True,
            "add_generation_prompt": canonical_language is None,
        }
        if canonical_language is not None:
            input_data["requests"][0]["messages"].append({
                "role": "assistant",
                "content": f"language {canonical_language}<asr_text>",
            })
        with self._worker_lock:
            self._ensure_worker()
            worker = self._worker
            wio = self._wio
            failed = self._worker_failed
            failed_reason = self._worker_failed_reason
        if worker is None or wio is None or failed:
            # Explicit rejection (see _worker_request): a failed-ready survivor,
            # a failure-marked live worker, or a concurrent teardown leaves no
            # usable WorkerIO. Never respawn over the live process.
            raise RuntimeError(
                "ASR worker unavailable ("
                f"{'failed: ' + failed_reason if failed else ('live process without WorkerIO' if worker is not None else 'no worker')}"
                f"): {self._stderr_tail_text()}"
            )
        t0 = time.time()
        gen = wio.request(input_data)
        try:
            output_data: dict = {}
            for ev in gen:
                output_data = ev
                typed = _classify_worker_response(ev)
                if typed is not None:
                    raise typed
                if ev.get("event") not in ("done", "cancelled") and (
                    ev.get("event") == "error" or ev.get("ok") is False
                ):
                    raise WorkerProtocolError(f"ASR worker error: {ev}")
        except _WIOExitError as exc:
            stderr = self._stderr_tail_text()
            self._clear_worker_if_current(
                worker, wio, reason=f"worker exited mid-request: {exc}"
            )
            raise RuntimeError(
                f"ASR worker exited before response: {exc}: {stderr}"
            ) from exc
        finally:
            gen.close()
        elapsed_worker = time.time() - t0

        if not output_data.get("ok"):
            raise RuntimeError(f"ASR worker failed: {output_data}")

        responses = output_data.get("responses", [])
        if not responses:
            raise RuntimeError(f"ASR produced no responses: {output_data}")
        text = responses[0].get("output_text", "")
        if text == "TensorRT Edge LLM cannot handle this request. Fails.":
            raise RuntimeError(f"ASR inference failed (model returned error): {responses[0]}")
        text, language_detected = self._postprocess_text(text)
        total_s = elapsed_mel_s + elapsed_worker
        return TranscriptionResult(
            text=text,
            language=language_detected,
            meta={
                "requested_language": language,
                "inference_time_s": round(total_s, 3),
                "mel_time_s": round(elapsed_mel_s, 3),
                "worker_time_s": round(elapsed_worker, 3),
                "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
            },
        )

    @staticmethod
    def _is_effectively_silent_segment(
        audio: np.ndarray, sample_rate: int, *, split_rms: float = DEFAULT_ENERGY_SPLIT_RMS
    ) -> bool:
        """Skip only uniformly quiet segments, preserving low-volume speech/noise pulses."""
        if audio.ndim != 1 or not np.isfinite(audio).all():
            return False
        frame_len = max(1, int(sample_rate * 20 / 1000))
        if len(audio) == 0:
            return True
        n = (len(audio) + frame_len - 1) // frame_len
        padded = np.pad(audio, (0, n * frame_len - len(audio)))
        frames = padded.reshape(n, frame_len)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        overall_rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        if overall_rms < split_rms:
            source = padded
            peak = float(np.max(np.abs(source)))
            if peak <= 1e-8:
                return True
            try:
                import webrtcvad
                gain_source = np.clip(source * (0.5 / peak), -1.0, 1.0)
                for candidate in (source, gain_source):
                    vad = webrtcvad.Vad(0)
                    pcm = (candidate * 32767).astype(np.int16)
                    if any(
                        vad.is_speech(
                            pcm[i * frame_len : (i + 1) * frame_len].tobytes(), sample_rate
                        )
                        for i in range(n)
                    ):
                        return False
            except Exception:
                # Unknown VAD capability is not evidence of silence. Preserve
                # non-zero audio rather than dropping a quiet real utterance.
                return False
        return overall_rms < split_rms and float(frame_rms.max()) <= 2 * split_rms

    def _prepare_worker_audio(
        self, audio_bytes: bytes, tmpdir: str
    ) -> "tuple[str, float]":
        """Write the per-request audio file the worker ingests.

        WAV-ingest mode (v0.9.0+): a raw 16 kHz mono WAV — the worker extracts
        mel internally. Legacy mode: a host-computed mel safetensors tensor.
        Returns (path, elapsed_prep_s). Both the resident-worker and one-shot
        paths just forward the path as ``{"type": "audio", "audio": <path>}``.
        """
        t0 = time.time()
        if self._config.request_audio_wav:
            audio, sr = _wav_bytes_to_float_audio(audio_bytes)
            if sr != 16000:
                ratio = 16000 / sr
                new_len = max(1, int(round(len(audio) * ratio)))
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
            wav_bytes = _float_audio_to_wav_bytes(audio, 16000)
            wav_path = os.path.join(tmpdir, "audio.wav")
            with open(wav_path, "wb") as f:
                f.write(wav_bytes)
            elapsed = time.time() - t0
            logger.info(
                "WAV-ingest audio written: %d bytes -> %s", len(wav_bytes), wav_path
            )
            return wav_path, elapsed

        mel = audio_bytes_to_mel(
            audio_bytes, min_audio_frames=self._config.min_audio_frames
        )  # [1, 128, T] float32
        max_mel_frames = int(self._config.max_mel_frames)
        if mel.shape[2] > max_mel_frames:
            raise ValueError(
                f"Audio too long: {mel.shape[2]} frames (~{mel.shape[2]*0.01:.0f}s). "
                f"Max {max_mel_frames} frames (~{max_mel_frames*0.01:.0f}s). Split into smaller chunks."
            )
        mel_fp16 = mel.astype(np.float16)
        mel_path = os.path.join(tmpdir, "mel.safetensors")
        write_safetensors(mel_fp16, self._config.mel_tensor_name, mel_path)
        elapsed = time.time() - t0
        logger.info(
            "Mel computed: shape=%s size=%s -> %s",
            list(mel_fp16.shape),
            mel_fp16.nbytes,
            mel_path,
        )
        return mel_path, elapsed

    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "auto",
    ) -> TranscriptionResult:
        """Transcribe audio via the resident C++ worker (or one-shot binary)."""
        if not self._ready:
            raise RuntimeError("ASR backend not preloaded")

        # Validate the API language before any early-return path (including
        # silent short audio), so silence handling cannot weaken the worker
        # language contract.
        self._canonical_worker_language(language)

        if self._config.offline_segment_enabled:
            try:
                audio, sample_rate = _wav_bytes_to_float_audio(audio_bytes)
                duration_s = len(audio) / max(sample_rate, 1)
            except Exception:
                audio = None
                sample_rate = 16000
                duration_s = 0.0
            if audio is not None and duration_s > self._config.offline_segment_threshold_s:
                return self._transcribe_segmented_offline(audio, sample_rate, language)
            if audio is not None and self._is_effectively_silent_segment(audio, sample_rate):
                return TranscriptionResult(
                    text="",
                    language=None,
                    meta={
                        "segmented": False,
                        "requested_language": language,
                        "original_duration_s": round(duration_s, 3),
                        "skipped_silent_segments": 1,
                        "skipped_silent_durations_s": [round(duration_s, 3)],
                        "empty_segments": 0,
                        "failed_segments": 0,
                    },
                )

        with tempfile.TemporaryDirectory(prefix="trt_edgellm_asr_") as tmpdir:
            audio_path, elapsed_prep_s = self._prepare_worker_audio(audio_bytes, tmpdir)

            if self._use_worker():
                return self._transcribe_worker(audio_path, elapsed_prep_s, language)

            input_data = {
                "requests": [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "audio", "audio": audio_path}],
                            }
                        ],
                    }
                ],
                "batch_size": 1,
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
                "max_generate_length": self._config.max_generate_length,
                "apply_chat_template": True,
                "add_generation_prompt": True,
            }

            input_path = os.path.join(tmpdir, "input.json")
            with open(input_path, "w") as f:
                json.dump(input_data, f)

            output_path = os.path.join(tmpdir, "output.json")
            cli_args = [
                "--engineDir", self._config.engine_dir,
                "--multimodalEngineDir", self._config.audio_encoder_dir,
                "--inputFile", input_path,
                "--outputFile", output_path,
            ]

            t0 = time.time()
            result = run_binary(
                self._config.asr_binary, cli_args, timeout=60,
                plugin_path=self._config.plugin_path,
            )
            elapsed = time.time() - t0

            if result.returncode != 0 or not os.path.exists(output_path):
                raise RuntimeError(
                    f"ASR subprocess failed (exit={result.returncode}): "
                    f"stdout={result.stdout[-300:]}, stderr={result.stderr[-300:]}"
                )

            with open(output_path) as f:
                output_data = json.load(f)

            responses = output_data.get("responses", [])
            if not responses:
                raise RuntimeError(f"ASR produced no responses: {output_data}")

            r = responses[0]
            text = r.get("output_text", "")
            if text == "TensorRT Edge LLM cannot handle this request. Fails.":
                raise RuntimeError(f"ASR inference failed (model returned error): {r}")

            text, language_detected = self._postprocess_text(text)
            return TranscriptionResult(
                text=text,
                language=language_detected,
                meta={"inference_time_s": round(elapsed, 3)},
            )

    def _transcribe_segmented_offline(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str,
    ) -> TranscriptionResult:
        """Split long offline WAV uploads before sending them to the worker."""
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = max(1, int(round(len(audio) * ratio)))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
            sample_rate = 16000

        original_duration_s = len(audio) / sample_rate
        segments = _split_offline_audio(
            audio,
            sample_rate,
            max_segment_s=self._config.offline_segment_threshold_s,
        )
        # (text, language) per segment. Language is filled in after the loop for
        # segments the model never labelled — see the unlabelled-segment note below.
        parts: list[tuple[str, Optional[str]]] = []
        total_inference_s = 0.0
        total_mel_s = 0.0
        total_worker_s = 0.0
        failed_segments = 0
        skipped_silent_segments = 0
        skipped_silent_durations_s: list[float] = []
        empty_segments = 0
        empty_segment_durations_s: list[float] = []
        min_seg_s = self._config.offline_segment_min_s

        for seg in segments:
            seg_duration_s = len(seg) / sample_rate
            if seg_duration_s < min_seg_s:
                continue
            if self._is_effectively_silent_segment(seg, sample_rate):
                skipped_silent_segments += 1
                skipped_silent_durations_s.append(round(seg_duration_s, 3))
                continue
            wav_bytes = _float_audio_to_wav_bytes(seg, sample_rate)
            try:
                result = self.transcribe(wav_bytes, language=language)
            except Exception as exc:
                failed_segments += 1
                logger.warning(
                    "TRT-EdgeLLM ASR offline segment failed (%.1fs): %s",
                    seg_duration_s,
                    exc,
                )
                continue
            if result.text:
                parts.append((result.text, result.language))
            else:
                empty_segments += 1
                empty_segment_durations_s.append(round(seg_duration_s, 3))
            meta = result.meta or {}
            total_inference_s += float(meta.get("inference_time_s", 0.0) or 0.0)
            total_mel_s += float(meta.get("mel_time_s", 0.0) or 0.0)
            total_worker_s += float(meta.get("worker_time_s", 0.0) or 0.0)

        # The ASR head prepends a "language <Lang>" tag to every well-formed
        # transcript, and _strip_language_prefix turns that into `language`. A
        # segment that comes back with `language=None` therefore never entered
        # the decode contract the int4 recipe validates against — and in practice
        # that is exactly when the output is degenerate: across a 35-minute
        # episode every clean segment reported a language and the only looping
        # one reported None. Treat the missing tag as the failure signal, label
        # the segment from the rest of the file, and report the count so callers
        # can see how much of the transcript is suspect.
        texts, unlabelled, majority_language = _split_segment_parts(parts)
        # collapse_repetition only sees one segment at a time, so a hallucination
        # that repeats once per segment slips through it and only piles up at the
        # join. Drop those runs here.
        texts, repeated = collapse_segment_repeats(texts)
        if repeated:
            logger.warning(
                "ASR: dropped %d segment(s) that repeated the previous segment verbatim",
                repeated,
            )

        return TranscriptionResult(
            text=_join_segment_texts(texts, majority_language or language),
            language=majority_language,
            meta={
                "segmented": True,
                "segment_count": len(segments),
                "unlabelled_segments": unlabelled,
                "repeated_segments": repeated,
                "failed_segments": failed_segments,
                "empty_segments": empty_segments,
                "empty_segment_durations_s": empty_segment_durations_s,
                "skipped_silent_segments": skipped_silent_segments,
                "skipped_silent_durations_s": skipped_silent_durations_s,
                "requested_language": language,
                "original_duration_s": round(original_duration_s, 3),
                "inference_time_s": round(total_inference_s, 3),
                "mel_time_s": round(total_mel_s, 3),
                "worker_time_s": round(total_worker_s, 3),
                "worker_init_ms": round(float(self._worker_ready_meta.get("init_ms", 0.0)), 1),
            },
        )

    def create_stream(self, language: str = "auto") -> ASRStream:
        """Accumulate stream audio and run the resident worker on finalize."""
        if not self._ready:
            raise RuntimeError("ASR backend not preloaded")
        if self._use_streaming_worker():
            return _TRTEdgeLLMStreamingASRStream(self, language=language)
        return _TRTEdgeLLMAccumulatingASRStream(self, language=language)


def _float_audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return out.getvalue()


def _wav_bytes_to_float_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate


def _split_offline_audio(
    audio: np.ndarray,
    sample_rate: int,
    *,
    max_segment_s: float,
) -> list[np.ndarray]:
    """Split long offline audio at silence (webrtcvad → energy fallback).

    The production version first tried a profile-configured VAD backend
    (``app.core.vad``); voxedge ships no VAD backend so it goes straight to
    the env-free webrtcvad→energy splitter cascade.
    """
    try:
        segments = _split_at_silence_vad(
            audio, sample_rate, max_seg_s=max_segment_s
        )
    except ImportError:
        segments = _split_at_silence_energy(
            audio, sample_rate, max_seg_s=max_segment_s
        )
    except Exception as exc:
        logger.warning("TRT-EdgeLLM ASR offline splitter failed: %s", exc)
        segments = [audio]

    max_samples = max(1, int(max_segment_s * sample_rate))
    bounded: list[np.ndarray] = []
    for seg in segments:
        if len(seg) <= max_samples:
            bounded.append(seg)
            continue
        for start in range(0, len(seg), max_samples):
            bounded.append(seg[start:start + max_samples])
    return [seg for seg in bounded if len(seg) > 0]


def _split_segment_parts(
    parts: list[tuple[str, Optional[str]]]
) -> tuple[list[str], int, Optional[str]]:
    """Return (texts, unlabelled_count, majority_language) for the segment results.

    A segment whose ``language`` is None never emitted the "language <Lang>" head
    tag, which means its decode left the contract the int4 recipe validates
    against — in practice that is exactly when the text is degenerate. The count
    goes into meta so callers can see how much of the transcript is suspect.

    The majority language is derived from the segments that *did* report one, and
    is used only to pick the join style and to report a file-level language. It is
    deliberately not taken from the caller's request: that value never reaches the
    decoder (the worker only ever *strips* the head tag, it never primes one), so
    echoing it back would claim a detection that never happened.
    """
    labelled = [lang for _, lang in parts if lang]
    majority = max(set(labelled), key=labelled.count) if labelled else None
    unlabelled = sum(1 for _, lang in parts if not lang)
    if unlabelled:
        logger.warning(
            "ASR: %d/%d offline segment(s) carried no language tag — their decode "
            "left the expected contract, so that text is unreliable.",
            unlabelled, len(parts),
        )
    return [text for text, _ in parts], unlabelled, majority


from voxedge.text.join import join_segments as _join_segment_texts  # noqa: E402


class _TRTEdgeLLMAccumulatingASRStream(ASRStream):

    #: 只持有 Python 侧的 _chunks 列表，没有 worker 会话/槽位（不发 begin），
    #: 随对象回收即可。
    OWNS_RESOURCES = False
    def __init__(self, backend: TRTEdgeLLMASRBackend, language: str = "auto"):
        self._backend = backend
        self._language = language
        self._chunks: list[np.ndarray] = []
        self._cancelled = False
        self._final_text_cache = ""

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if self._cancelled:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        self._chunks.append(samples.copy())

    def cancel_and_finalize(self) -> None:
        if self._cancelled:
            return
        self._final_text_cache = ""
        self._cancelled = True
        self._chunks = []

    def finalize(self) -> tuple[str, Optional[str]]:
        if self._cancelled:
            return self._final_text_cache, None
        if not self._chunks:
            return "", None
        audio = np.concatenate(self._chunks)
        # The TRT audio_encoder engine is built with a fixed optimization
        # profile; forwarding >10s in one shot makes the worker reject the
        # request. Split at natural silence (webrtcvad → energy fallback),
        # then concatenate per-segment transcripts.
        try:
            segments = _split_at_silence_vad(audio)
        except ImportError:
            segments = _split_at_silence_energy(audio)
        except Exception:
            segments = [audio]

        MIN_SEG_S = 0.4
        texts: list[str] = []
        detected_language: Optional[str] = None
        for seg in segments:
            if len(seg) / 16000 < MIN_SEG_S:
                continue
            wav_bytes = _float_audio_to_wav_bytes(seg, 16000)
            try:
                result = self._backend.transcribe(wav_bytes, language=self._language)
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "TRT-EdgeLLM ASR segment failed (%.1fs): %s",
                    len(seg) / 16000, e,
                )
                continue
            if result.text:
                texts.append(result.text)
                if detected_language is None and getattr(result, "language", None):
                    detected_language = result.language

        cjk_langs = {"Chinese", "Japanese", "Korean", "Cantonese", "zh", "ja", "ko"}
        is_cjk = self._language in cjk_langs
        if len(texts) > 1:
            trail_punct = "。，、！？；,.!?;"
            cleaned: list[str] = []
            for i, t in enumerate(texts):
                if i < len(texts) - 1:
                    cleaned.append(t.rstrip(trail_punct).rstrip())
                else:
                    cleaned.append(t)
            texts = cleaned
        separator = "" if is_cjk else " "
        return separator.join(texts).strip(), detected_language

    def get_partial(self) -> tuple[str, bool]:
        return "", False


class _TRTEdgeLLMStreamingASRStream(ASRStream):
    """TRT-EdgeLLM qwen3_asr_worker streaming protocol adapter.

    Enabled only with stream_mode=worker. The worker receives cumulative
    float32 PCM via ``pcm_b64`` and emits partial/final JSON events.
    """

    #: close() 必须向 worker 发 end 归还槽位（max_slots=1，漏一个 ASR 就死）。
    OWNS_RESOURCES = True

    def __init__(self, backend: TRTEdgeLLMASRBackend, language: str = "auto"):
        self._backend = backend
        self._language = language
        self._session_id = uuid.uuid4().hex
        self._sample_rate = 16000
        self._hop_samples = max(
            1, int(float(backend._config.stream_chunk_sec) * self._sample_rate)
        )
        self._audio_accum = np.zeros(0, dtype=np.float32)
        self._samples_since_hop = 0
        self._partial_text = ""
        self._final_text = ""
        # Text committed from earlier segments after a proactive long-audio
        # rotation (see _maybe_rotate_segment). Empty for the common short-audio
        # single-segment case.
        self._committed_text = ""
        _cap = float(getattr(backend._config, "segment_cap_sec", 0.0) or 0.0)
        self._segment_cap_samples = int(_cap * self._sample_rate) if _cap > 0 else 0
        self._detected_language: Optional[str] = None
        self._cancelled = False
        self._closed = False
        # --- Native stream-cancel state (U2 attempt2) ----------------------
        # ``_state_lock`` guards ONLY the short cancel/rotation bookkeeping
        # below. It is NEVER held across worker/stdin IO, waits, transcribe, or
        # text postprocess.
        self._state_lock = threading.Lock()
        # Per-SID BEGIN-WRITTEN barrier. Created by ``_begin_written_event`` and
        # set EXACTLY once by that SID's ``_begin`` on_written callback (i.e. only
        # after the native ``begin`` line is actually written+flushed). It is
        # used ONLY to gate a cancel write behind a provably-known native SID.
        self._begin_written: dict[str, threading.Event] = {}
        # Per-SID CANCEL-REQUESTED marker. Created by
        # ``_cancel_requested_event`` and set ONLY by ``arm_cancel`` /
        # ``request_cancel``. It is the ``expected_cancel`` marker for ordinary
        # requests and is NEVER set by a begin. It starts UNSET and is NEVER
        # cleared once armed (stays set while consumers drain / after receipt).
        self._cancel_requested: dict[str, threading.Event] = {}
        # SIDs with an armed cancellation intent.
        self._cancel_armed_sids: set[str] = set()
        # Stable per-SID intent record: {sid, requested_at}. Never cleared while
        # the old SID may still be draining; the old record survives rotation.
        self._cancel_intent: Optional[dict] = None
        # ACTUAL matching native receipt applied to exactly one SID.
        self._cancel_confirm: Optional[dict] = None
        # Worker exited while an intent was unresolved (distinct from timeout).
        self._cancel_exit = False
        # Duplicate-control ownership: True while exactly one request_cancel is
        # inside the actual helper. Prevents two controls issuing two native
        # cancels for the same intent. Released when the helper RETURNS
        # (success or TimeoutError); the intent itself is retained.
        self._control_inflight = False
        # Per-SID count of ordinary requests currently in flight (begin/chunk/
        # end). The old SID's marker/intent is NOT dropped while count != 0.
        self._inflight_count: dict[str, int] = {}
        self._begin()

    # ------------------------------------------------------------------
    # Native cancel state helpers (no IO, short-lock only)
    # ------------------------------------------------------------------
    def _begin_written_event(self, sid: str) -> threading.Event:
        """Return (creating if needed) the begin-written barrier for ``sid``."""
        with self._state_lock:
            ev = self._begin_written.get(sid)
            if ev is None:
                ev = threading.Event()
                self._begin_written[sid] = ev
            return ev

    def _cancel_requested_event(self, sid: str) -> threading.Event:
        """Return (creating if needed) the cancel-requested marker for ``sid``."""
        with self._state_lock:
            ev = self._cancel_requested.get(sid)
            if ev is None:
                ev = threading.Event()
                self._cancel_requested[sid] = ev
            return ev

    def _cancel_armed_for(self, sid: str) -> bool:
        with self._state_lock:
            ev = self._cancel_requested.get(sid)
            return bool(ev is not None and ev.is_set())

    def _inflight_inc(self, sid: str) -> None:
        with self._state_lock:
            self._inflight_count[sid] = self._inflight_count.get(sid, 0) + 1

    def _inflight_dec(self, sid: str) -> None:
        with self._state_lock:
            n = self._inflight_count.get(sid, 0) - 1
            if n <= 0:
                self._inflight_count.pop(sid, None)
            else:
                self._inflight_count[sid] = n

    def _inflight_for(self, sid: str) -> int:
        with self._state_lock:
            return self._inflight_count.get(sid, 0)

    def _ordinary_request(
        self, payload: dict, *, on_written=None
    ) -> dict:
        """Send one ordinary begin/chunk/end request with in-flight accounting.

        Uses the SID's CANCEL-REQUESTED event (unset until a cancel is armed) as
        ``expected_cancel``. The in-flight count is incremented before the IO
        and decremented in ``finally``; NO lock is held over the IO. A cancelled
        terminal is returned to the caller unchanged (which must suppress normal
        state writes). ``on_written`` is forwarded to the backend ONLY for the
        begin request (sets the begin-written barrier).
        """
        sid = payload.get("id")
        if isinstance(sid, str) and sid:
            self._inflight_inc(sid)
            marker = self._cancel_requested_event(sid)
        else:
            marker = None
        try:
            if on_written is None:
                return self._backend._worker_request(
                    payload, expected_cancel=marker
                )
            return self._backend._worker_request(
                payload, expected_cancel=marker, on_written=on_written
            )
        finally:
            if isinstance(sid, str) and sid:
                self._inflight_dec(sid)

    def _record_cancel_receipt(self, sid: str, receipt: dict) -> None:
        """Record an ACTUAL matching cancelled terminal for an ARMED ``sid`` only.

        An unarmed / non-matching ``cancelled`` event must NOT fabricate an
        intent or a confirmation — it propagates ``WorkerProtocolError`` instead.
        """
        if not (
            isinstance(receipt, dict)
            and receipt.get("event") == "cancelled"
            and receipt.get("ok") is False
            and receipt.get("id") == sid
        ):
            raise WorkerProtocolError(
                f"non-matching cancelled receipt for sid={sid!r}: {receipt!r}"
            )
        with self._state_lock:
            ev = self._cancel_requested.get(sid)
            if ev is None or not ev.is_set():
                raise WorkerProtocolError(
                    f"cancelled receipt for UNArmED sid={sid!r}: {receipt!r}"
                )
            self._cancel_armed_sids.add(sid)
            if self._cancel_intent is None or self._cancel_intent.get("sid") != sid:
                # Do NOT overwrite a live intent for another SID.
                if self._cancel_intent is None:
                    self._cancel_intent = {
                        "sid": sid,
                        "requested_at": time.monotonic(),
                    }
            if (
                self._cancel_confirm is None
                or self._cancel_confirm.get("sid") != sid
            ):
                self._cancel_confirm = {
                    "sid": sid,
                    "epoch": receipt.get("epoch"),
                    "receipt": receipt,
                }

    def arm_cancel(self) -> str:
        """Non-blocking, IO-free: capture the CURRENT SID and arm its cancel.

        Sets ONLY the SID's cancel-requested marker (never the begin-written
        barrier). Called by the caller BEFORE it schedules the blocking control
        job, so the intent is visible to the ordinary consumer immediately.
        Returns the captured SID. Never clears an existing intent.
        """
        with self._state_lock:
            sid = self._session_id
            ev = self._cancel_requested.get(sid)
            if ev is None:
                ev = threading.Event()
                self._cancel_requested[sid] = ev
            ev.set()
            self._cancel_armed_sids.add(sid)
            if self._cancel_intent is None:
                self._cancel_intent = {
                    "sid": sid,
                    "requested_at": time.monotonic(),
                }
            return sid

    def request_cancel(self, timeout_s: float) -> dict:
        """Actually cancel the captured SID; return the ACTUAL native receipt.

        Validation of ``timeout_s`` happens BEFORE any state change or write.
        Arms the current SID if not already armed, waits (inclusive, one
        absolute monotonic deadline) for that SID's BEGIN-WRITTEN barrier, then
        runs the ACTUAL backend ``_worker_cancel_and_wait`` with the REMAINING
        budget. If a matching receipt was already confirmed for this SID, the
        retained actual receipt is returned WITHOUT writing a duplicate cancel.

        Failure semantics (deliberately distinct):
          * ``TimeoutError``: control-inflight released, intent stays armed, NO
            confirmed receipt, stream NOT closed, NO restart/pool.
          * ``WorkerExitError``: ``_cancel_exit`` set, NO ACK.
        """
        # --- Validate BEFORE changing state / writing anything. ---
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError(
                f"request_cancel: invalid timeout_s={timeout_s!r}"
            )
        deadline = time.monotonic() + float(timeout_s)

        # --- Capture/arm the intent under the short lock, then RELEASE it. ---
        with self._state_lock:
            intent = self._cancel_intent
            if intent is None:
                sid = self._session_id
                ev = self._cancel_requested.get(sid)
                if ev is None:
                    ev = threading.Event()
                    self._cancel_requested[sid] = ev
                ev.set()
                self._cancel_armed_sids.add(sid)
                intent = {"sid": sid, "requested_at": time.monotonic()}
                self._cancel_intent = intent
            captured_sid = intent["sid"]
            # Idempotence: an already-confirmed SID returns the retained receipt
            # (never a duplicate cancel write to an already-released SID).
            if (
                self._cancel_confirm is not None
                and self._cancel_confirm.get("sid") == captured_sid
            ):
                return self._cancel_confirm["receipt"]
            if self._control_inflight:
                raise RuntimeError(
                    "request_cancel: duplicate control for "
                    f"sid={captured_sid!r} already in flight"
                )
            self._control_inflight = True
            begin_ev = self._begin_written.get(captured_sid)
            if begin_ev is None:
                begin_ev = threading.Event()
                self._begin_written[captured_sid] = begin_ev

        # --- Begin-written barrier: one absolute deadline, remaining budget. ---
        remaining = deadline - time.monotonic()
        if remaining > 0 and not begin_ev.is_set():
            begin_ev.wait(remaining)
        remaining = deadline - time.monotonic()
        # No assumed wake / no early false: re-check the ACTUAL barrier and the
        # ACTUAL remaining budget after the wait.
        if not begin_ev.is_set() or remaining <= 0:
            with self._state_lock:
                self._control_inflight = False
            raise TimeoutError(
                f"request_cancel: begin-write barrier not satisfied for "
                f"sid={captured_sid!r} within budget"
            )

        # --- ACTUAL native cancel for the CAPTURED SID (never current SID). ---
        try:
            receipt = self._backend._worker_cancel_and_wait(captured_sid, remaining)
        except TimeoutError:
            # The helper RETURNED: release duplicate-control ownership, but
            # keep the intent armed (no confirm, no closed, no restart).
            with self._state_lock:
                self._control_inflight = False
            raise
        except WorkerExitError:
            with self._state_lock:
                self._control_inflight = False
                self._cancel_exit = True
            raise
        except BaseException:
            with self._state_lock:
                self._control_inflight = False
            raise

        # --- Validate the receipt shape even for a fake backend. ---
        if not isinstance(receipt, dict) or not (
            receipt.get("event") == "cancelled"
            and receipt.get("ok") is False
            and receipt.get("id") == captured_sid
        ):
            with self._state_lock:
                self._control_inflight = False
            raise WorkerProtocolError(
                "request_cancel: non-matching cancelled receipt for "
                f"sid={captured_sid!r}: {receipt!r}"
            )
        self._record_cancel_receipt(captured_sid, receipt)
        with self._state_lock:
            self._control_inflight = False
        return receipt

    def _begin(self) -> None:
        # Capture the SID for THIS begin; the on_written callback must set the
        # captured SID's BEGIN-WRITTEN barrier, and the ordinary request uses the
        # SEPARATE cancel-requested marker (unset until a cancel is armed).
        sid = self._session_id
        begin_ev = self._begin_written_event(sid)
        ev = {
            "event": "begin",
            "id": sid,
            "sample_rate": self._sample_rate,
            "chunk_size_sec": float(self._backend._config.stream_chunk_sec),
            "unfixed_chunk_num": int(self._backend._config.stream_unfixed_chunks),
            "unfixed_token_num": int(self._backend._config.stream_unfixed_tokens),
            "context": "",
        }
        if self._language and self._language != "auto":
            ev["force_language"] = self._language
        resp = self._ordinary_request(ev, on_written=begin_ev.set)
        # The begin gate only needs the write barrier, which on_written sets.
        if resp.get("event") == "cancelled":
            # A begin can itself observe an armed cancel; record it (this
            # validates the arm), mutate nothing, and let callers see it.
            self._record_cancel_receipt(sid, resp)
            return
        if resp.get("event") != "begin_ack":
            raise RuntimeError(f"ASR streaming worker begin failed: {resp}")

    def _send_chunk(self, *, last: bool) -> dict:
        sid = self._session_id
        pcm = np.asarray(self._audio_accum, dtype="<f4")
        pcm_b64 = base64.b64encode(pcm.tobytes()).decode("ascii")
        resp = self._ordinary_request(
            {
                "event": "chunk",
                "id": sid,
                "pcm_b64": pcm_b64,
                "audio_sec": len(self._audio_accum) / self._sample_rate,
                "last": last,
            }
        )
        event = resp.get("event")
        if event == "cancelled":
            # Control outcome, NEVER a normal partial/final write. Signal the
            # state machine only; mutate NO text/_closed/_audio state.
            self._record_cancel_receipt(sid, resp)
            return resp
        # Prepare any expensive text computation OUTSIDE the lock.
        prepared = None
        if event == "partial":
            stripped, lang = self._backend._strip_language_prefix(
                resp.get("text", "") or ""
            )
            prepared = (stripped.strip(), lang)
        elif event == "final":
            stripped, lang = self._backend._postprocess_text(resp.get("text", "") or "")
            prepared = (stripped.strip(), lang)
        elif event == "segment_rotation":
            carry_samples = int(float(resp.get("carryover_sec", 1.0)) * self._sample_rate)
            prepared = carry_samples
        else:
            raise RuntimeError(f"unexpected ASR streaming worker event: {resp}")

        # Guard AND normal state mutation share the SAME short state lock, and
        # are linearized against arm_cancel. No IO / waits / postprocess below.
        with self._state_lock:
            if self._session_id != sid:
                return resp
            ev = self._cancel_requested.get(sid)
            if ev is not None and ev.is_set():
                return resp
            if event == "segment_rotation":
                carry_samples = prepared
                if carry_samples > 0 and len(self._audio_accum) > carry_samples:
                    self._audio_accum = self._audio_accum[-carry_samples:].copy()
            elif event == "partial":
                stripped, lang = prepared
                self._partial_text = stripped
                if lang:
                    self._detected_language = lang
            elif event == "final":
                stripped, lang = prepared
                self._final_text = stripped
                if lang:
                    self._detected_language = lang
                self._closed = True
        return resp

    def _join_committed(self, tail: str) -> str:
        """Concatenate text committed from earlier rotated segments with the
        current segment's text. For the common single-segment case
        (_committed_text == "") this just returns ``tail`` stripped."""
        tail = (tail or "").strip()
        if self._committed_text and tail:
            return (self._committed_text + " " + tail).strip()
        return (self._committed_text or tail).strip()

    def _rotate_segment(self) -> None:
        """Proactive long-audio cap: finalize the current segment, commit its
        text, then start a fresh worker session so the next prefill stays under
        the engine KV cap. Clean cut — no audio carryover — so there is no
        boundary re-transcription / duplication (the trade-off is a possible
        word split at the cut, far better than the current total failure)."""
        old_sid = self._session_id
        resp = self._send_chunk(last=True)  # normally 'final' -> sets _final_text
        # Robustness: if the worker returns 'segment_rotation' for this forced
        # finalize (instead of 'final'), _final_text is NOT set — fall back to
        # the latest partial so this segment's text is not silently dropped.
        # Compute the text OUTSIDE the lock (collapse_repetition is not trivial).
        if resp.get("event") == "final":
            seg = (self._final_text or "").strip()
        else:
            # 这条兜底路径把 partial 当成 final 提交，而 partial 是刻意不做
            # 退化塌缩的（见 partial 分支注释）。既然它要被当作定稿写进
            # _committed_text，就得在这里补上塌缩，否则长音频轮转时退化文本
            # 会绕过守卫。
            seg = self._collapse_if_promoted((self._partial_text or "").strip())

        # Linearize the commit/reset/SID switch under the lock, checking the
        # CAPTURED old SID and the cancel marker atomically. If a cancel was
        # armed during the blocked final, leave committed text UNCHANGED and do
        # NOT launch a new SID. No IO below.
        new_sid: Optional[str] = None
        with self._state_lock:
            if self._session_id != old_sid:
                return
            ev = self._cancel_requested.get(old_sid)
            if ev is not None and ev.is_set():
                return
            if seg:
                self._committed_text = (
                    (self._committed_text + " " + seg).strip()
                    if self._committed_text else seg
                )
            self._audio_accum = np.zeros(0, dtype=np.float32)
            self._samples_since_hop = 0
            self._partial_text = ""
            self._final_text = ""
            self._closed = False
            new_sid = uuid.uuid4().hex
            self._session_id = new_sid
        # IO (begin) is issued OUTSIDE the lock, for the SID we committed.
        self._begin()

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if self._cancelled or self._closed:
            return
        if self._cancel_armed_for(self._session_id):
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if sample_rate != self._sample_rate:
            ratio = self._sample_rate / sample_rate
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        self._audio_accum = np.concatenate([self._audio_accum, samples])
        self._samples_since_hop += len(samples)
        # Proactive long-audio rotation: cut to a fresh segment before the worker
        # prefill overflows the engine KV cache (~6.2s). Only fires for audio
        # longer than the cap; short utterances never reach this branch and take
        # the unchanged single-segment path below.
        if self._segment_cap_samples and len(self._audio_accum) >= self._segment_cap_samples:
            self._rotate_segment()
            return
        while self._samples_since_hop >= self._hop_samples:
            self._send_chunk(last=False)
            self._samples_since_hop -= self._hop_samples

    def prepare_finalize(self) -> None:
        pass

    def finalize(self) -> tuple[str, Optional[str]]:
        if self._cancelled or self._closed:
            return self._join_committed(self._final_text), self._detected_language
        if self._cancel_armed_for(self._session_id):
            return self._join_committed(self._final_text), self._detected_language
        try:
            return self._finalize_inner()
        finally:
            # 正常路径以前从不发 ``end``：只有「零音频」分支和 cancel_and_finalize()
            # 会发。于是每一次成功的识别都漏掉一个 worker 槽位，而 max_slots=1 ——
            # 第一轮之后所有 create_stream 都抛 PoolSaturatedError，设备永远停在
            # 「聆听中」。实测：05:58:52 opened → 05:58:53 closed → 之后每次连接
            # 都 pool_saturated，直到重启容器。
            self.close()

    def _finalize_inner(self) -> tuple[str, Optional[str]]:
        sid = self._session_id
        if len(self._audio_accum) == 0:
            if self._cancel_armed_for(sid):
                return self._join_committed(self._final_text), self._detected_language
            resp = self._ordinary_request({"event": "end", "id": sid})
            if resp.get("event") == "cancelled":
                self._record_cancel_receipt(sid, resp)
                return self._join_committed(self._final_text), self._detected_language
            with self._state_lock:
                if self._session_id == sid:
                    self._closed = True
            return self._join_committed(""), self._detected_language
        self._send_chunk(last=True)
        # Late final after an armed cancel must suppress text writes AND the
        # offline rescue: _send_chunk already refused to mutate _final_text, so
        # return whatever was committed before the cancel without rescuing.
        if self._cancel_armed_for(sid):
            return self._join_committed(self._final_text), self._detected_language
        text = self._join_committed(self._final_text)
        # Empty-final rescue (2026-06-14): the streaming worker withholds up to
        # ``unfixed_token_num`` trailing tokens; a SHORT utterance whose entire
        # output fits inside that hold emits NO final text (observed real-machine:
        # short English commands → empty asr_final, while the offline transcribe()
        # path — what POST /asr uses — transcribes the very same audio cleanly;
        # short Chinese commits enough tokens to escape the hold). When the worker
        # returned empty but we buffered ≥ offline_segment_min_s of audio, fall
        # back to the offline path. Non-empty finals (Chinese / long English) take
        # zero new code.
        #
        # Extended (2026-06-15): the worker can also commit a SINGLE token of a
        # multi-word utterance ("drop" out of "drop it down" — real-machine)
        # while withholding the rest, which the bare empty check misses. Also
        # rescue a suspiciously SHORT streaming final (≤ offline_rescue_max_words
        # tokens) when enough audio was buffered. The normal multi-token fast
        # path is unchanged: zero new work, no offline call. If offline comes
        # back empty or NOT longer, keep the streaming text (never regress).
        stripped = text.strip()
        word_count = len(stripped.split())
        max_words = int(getattr(self._backend._config, "offline_rescue_max_words", 0))
        enough_audio = (
            len(self._audio_accum) / self._sample_rate
            >= self._backend._config.offline_segment_min_s
        )
        # CJK guard: the short-final rescue targets space-delimited languages
        # (English "drop"). Chinese has no word spaces — a real one-character
        # commit like "回家" is word_count==1 but NOT clipped (the worker
        # commits enough CJK tokens to escape the hold), so never re-transcribe
        # non-empty CJK finals; that keeps the existing Chinese fast path exact.
        has_cjk = any("一" <= ch <= "鿿" for ch in stripped)
        short_trigger = bool(stripped) and not has_cjk and word_count <= max_words
        if (not stripped or short_trigger) and enough_audio:
            try:
                wav_bytes = _float_audio_to_wav_bytes(self._audio_accum, self._sample_rate)
                result = self._backend.transcribe(wav_bytes, language=self._language)
                offline = (result.text or "").strip()
                # For empty finals: accept any non-empty offline text. For a
                # short non-empty final: only replace when offline is strictly
                # longer (more words) — never regress a real streaming token.
                if offline and (
                    not stripped or len(offline.split()) > word_count
                ):
                    logging.getLogger(__name__).info(
                        "streaming finalize %s → offline fallback recovered %d chars",
                        "empty" if not stripped else f"short({word_count}w)",
                        len(offline),
                    )
                    return offline, (
                        getattr(result, "language", None) or self._detected_language
                    )
            except Exception:
                logging.getLogger(__name__).warning(
                    "streaming finalize offline fallback failed", exc_info=True
                )
        return text, self._detected_language

    def close(self) -> None:
        """释放 worker 槽位。幂等，finalize()/cancel 之后再调也安全。

        server 的 ASR handler 在 finally 里做 ``getattr(stream, "close", None)``
        并调用它 —— 但这个流类此前没有 ``close``，getattr 拿到 None，整段兜底
        形同虚设。ws 在 finalize 之外的任何路径结束（客户端先断、异常、取消）
        都会让槽位永久泄漏。

        U2 attempt2 收尾语义:
          * 已收到 ACTUAL matching cancelled receipt 的 SID: native 已同步
            releaseSession — 直接标记逻辑 native 关闭，**不发 end**（未知 SID
            的 end 是白白阻塞 ordinary consumer 等待一个永不到来的 terminal）。
            这只是逻辑 native 关闭，不声称 writer/thread/GPU 已完成。
          * 已武装但未确认的取消: 不发 end，也不置 ``_closed``。
          * 未取消: 发带该 SID cancel 标记的 end；end 失败不得在 finally 里
            假装已关闭，保留 ``_closed=False``。
        """
        if self._closed:
            return
        with self._state_lock:
            sid = self._session_id
            intent = self._cancel_intent
            confirm = self._cancel_confirm
            exit_seen = self._cancel_exit
        if confirm is not None and confirm.get("sid") == sid:
            # Confirmed native receipt == native SID release. No redundant end.
            with self._state_lock:
                if self._session_id == sid:
                    self._closed = True
            return
        if (
            intent is not None
            and intent.get("sid") == sid
            and not exit_seen
        ):
            logger.warning(
                "ASR stream close: cancellation for session %s unconfirmed; "
                "NOT marking closed / NOT sending end (slot release left to "
                "the retained caller/control job)",
                sid,
            )
            return
        try:
            resp = self._ordinary_request({"event": "end", "id": sid})
            if resp.get("event") == "cancelled":
                # Actual cancelled terminal: control outcome, no normal closing.
                self._record_cancel_receipt(sid, resp)
                return
        except Exception:
            logging.getLogger(__name__).warning(
                "ASR stream close: end event failed; worker slot may leak",
                exc_info=True,
            )
            # Do NOT mark closed falsely; the slot is NOT proven released.
            return
        with self._state_lock:
            if self._session_id == sid:
                self._closed = True

    def _collapse_if_promoted(self, seg: str) -> str:
        """partial 被晋升为 final / 提交为定稿时补做退化塌缩。

        partial 本身刻意不塌缩（会随音频增长反复重算，塌缩会让字幕抖动），
        所以每一处把 partial 当定稿用的地方都必须过这里，否则退化文本绕过守卫。
        当前调用点：轮转兜底、cancel_and_finalize。
        """
        if not seg or not self._backend._config.collapse_repetition:
            return seg
        collapsed, did = collapse_repetition(seg)
        if did:
            logger.warning(
                "ASR 被晋升为定稿的 partial 疑似退化，已塌缩: %r -> %r",
                seg[:80], collapsed,
            )
            return collapsed
        return seg

    def cancel_and_finalize(self) -> None:
        """Legacy hard-cancel entrypoint — delegate to ACTUAL native cancel.

        This is the ``ASRStream.cancel()`` alias target used by the session
        manager. It NO LONGER promotes the partial into a fake final, NO LONGER
        creates an unretained one-shot executor, and NO LONGER maps a wait
        timeout to ``WorkerExitError`` (a ``TimeoutError`` here is exactly a
        timeout: intent armed, no confirmation, no closed, no restart). It
        delegates to ``request_cancel`` with the historical 0.5s bound.
        """
        self.request_cancel(0.5)

    def get_partial(self) -> tuple[str, bool]:
        if self._closed:
            return self._join_committed(self._final_text), True
        return self._join_committed(self._partial_text), False


def build_config_from_env(env: "dict | None" = None) -> TRTEdgeLLMASRConfig:
    """Build TRTEdgeLLMASRConfig from environment variables.

    All path fields are resolved from the passed ``env`` dict (or
    ``os.environ`` when None). Resolution logic mirrors the ``_deploy_paths``
    module-level constants but uses the supplied dict so callers can pass an
    explicit env override without touching the real process environment.
    Mirrors ``server.core.voxedge_backend_config.build_trt_edge_llm_asr_config``
    field-for-field (env → manifest → default precedence; manifest support
    is retained: set EDGE_LLM_ASR_MANIFEST to a JSON path).
    """
    import json as _json
    import os as _os

    if env is None:
        env = _os.environ

    # ---------------------------------------------------------------------------
    # Inline env-dict-aware path resolvers (mirror _deploy_paths but use ``env``)
    # ---------------------------------------------------------------------------

    def _prefer_existing_e(primary: str, fallback: str) -> str:
        if primary and _os.path.exists(primary):
            return primary
        return fallback

    def _first_existing_dir_e(*paths: str) -> str:
        for p in paths:
            if p and _os.path.exists(p):
                return p
        return paths[-1] if paths else ""

    edge_base = env.get("EDGE_LLM_BASE", _os.path.expanduser("~/project/tensorrt-edge-llm"))
    edge_build = _os.path.join(edge_base, env.get("EDGE_LLM_BUILD_DIR", "build_sm87"))
    ovs_base = env.get("OVS_BASE", "")
    ovs_build = env.get(
        "OVS_WORKER_BUILD",
        _os.path.join(ovs_base, "build", "edgellm_voice_worker", "workers") if ovs_base else "",
    )

    def _asr_binary_e() -> str:
        explicit = env.get("EDGE_LLM_ASR_BIN")
        if explicit:
            return explicit
        return _os.path.join(edge_build, "examples/llm/llm_inference")

    def _asr_worker_binary_e() -> str:
        explicit = env.get("EDGE_LLM_ASR_WORKER_BIN")
        if explicit:
            return explicit
        return _prefer_existing_e(
            _os.path.join(ovs_build, "qwen3_asr_worker") if ovs_build else "",
            _os.path.join(edge_build, "examples/llm/qwen3_asr_worker"),
        )

    def _asr_plugin_path_e() -> str:
        for key in ("EDGE_LLM_ASR_PLUGIN_PATH", "EDGELLM_ASR_PLUGIN_PATH", "EDGELLM_PLUGIN_PATH"):
            v = env.get(key)
            if v:
                return v
        return _os.path.join(edge_build, "libNvInfer_edgellm_plugin.so")

    # ASR engine dir resolution (mirrors _deploy_paths module-level logic using env)
    def _asr_engine_dir_e() -> str:
        explicit = env.get("EDGE_LLM_ASR_ENGINE_DIR")
        if explicit:
            return explicit
        # Full engine dir
        _fp8 = _os.path.expanduser("~/qwen3-asr-edgellm-runtime/engines/thinker_full_in128_kv256_fp8embed_0510")
        _small = _os.path.expanduser("~/qwen3-asr-edgellm-runtime/engines/thinker_full_in128_kv256_0510")
        _dialog = _os.path.expanduser("~/qwen3-asr-edgellm-runtime/engines/thinker_kv512")
        _export = _os.path.expanduser("~/qwen3-asr-trt-edge-llm-export/engines/thinker")
        full_dir = env.get(
            "EDGE_LLM_ASR_FULL_ENGINE_DIR",
            _fp8 if _os.path.exists(_os.path.join(_fp8, "llm.engine"))
            else _small if _os.path.exists(_os.path.join(_small, "llm.engine"))
            else _dialog if _os.path.exists(_os.path.join(_dialog, "llm.engine"))
            else _export,
        )
        # Pruned engine dir
        _pruned = _os.path.expanduser("~/qwen3-asr-edgellm-runtime/engines/thinker_prunedembed35k_kv512")
        _off_pruned = _os.path.expanduser("~/qwen3-asr-edgellm-runtime/engines/thinker_pruned35k_kv512")
        pruned_dir = env.get(
            "EDGE_LLM_ASR_PRUNED_ENGINE_DIR",
            _pruned if _os.path.exists(_os.path.join(_pruned, "llm.engine")) else _off_pruned,
        )
        vocab = env.get("EDGE_LLM_ASR_VOCAB_PRUNED", "0").lower()
        if vocab in ("1", "true", "yes"):
            return pruned_dir
        if vocab in ("0", "false", "no"):
            return full_dir
        # unrecognized: probe order
        if _os.path.exists(_os.path.join(_pruned, "llm.engine")):
            return _pruned
        if _os.path.exists(_os.path.join(_off_pruned, "llm.engine")):
            return _off_pruned
        return full_dir

    def _asr_audio_enc_dir_e() -> str:
        explicit = env.get("EDGE_LLM_ASR_AUDIO_ENC_DIR")
        if explicit:
            return explicit
        return _os.path.expanduser("~/qwen3-asr-trt-edge-llm-export/engines/audio_encoder")

    # ---------------------------------------------------------------------------

    def _env_bool(name: str, default: bool) -> bool:
        value = env.get(name)
        if value is None:
            return default
        return value.lower() not in ("0", "false", "no")

    # -- manifest (EDGE_LLM_ASR_MANIFEST) --
    manifest: dict = {}
    manifest_path = env.get("EDGE_LLM_ASR_MANIFEST")
    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = _json.load(f)
    use_worker_default = bool(manifest.get("use_worker", True))

    # -- slot ceiling: env → manifest → 1 --
    max_slots_raw = env.get(
        "EDGE_LLM_ASR_MAX_CONCURRENT",
        str(manifest.get("asr_max_slots", manifest.get("max_concurrent", 1))),
    )

    # -- warmup: legacy gates on SKIP_ASR_WARMUP OR EDGE_LLM_ASR_WORKER_WARMUP --
    skip_warmup = env.get("SKIP_ASR_WARMUP", "").lower() in ("1", "true", "yes")
    warmup_disabled = env.get("EDGE_LLM_ASR_WORKER_WARMUP", "1").lower() in ("0", "false", "no")
    worker_warmup = not (skip_warmup or warmup_disabled)

    try:
        prewarm_max = int(env.get("EDGE_LLM_ASR_PREWARM_MAX", "6"))
    except ValueError:
        prewarm_max = 6

    try:
        min_audio_frames = int(env.get("EDGE_LLM_ASR_MIN_AUDIO_FRAMES", "100"))
    except ValueError:
        min_audio_frames = 100

    return TRTEdgeLLMASRConfig(
        # --- paths (all resolved from passed env dict) ---
        asr_binary=env.get("EDGE_LLM_ASR_BIN", manifest.get("asr_binary", _asr_binary_e())),
        worker_binary=env.get(
            "EDGE_LLM_ASR_WORKER_BIN", manifest.get("worker_binary", _asr_worker_binary_e())
        ),
        plugin_path=env.get(
            "EDGE_LLM_ASR_PLUGIN_PATH",
            env.get(
                "EDGELLM_ASR_PLUGIN_PATH",
                manifest.get("asr_plugin_path", manifest.get("plugin_path", _asr_plugin_path_e())),
            ),
        ),
        engine_dir=env.get("EDGE_LLM_ASR_ENGINE_DIR", manifest.get("engine_dir", _asr_engine_dir_e())),
        audio_encoder_dir=env.get(
            "EDGE_LLM_ASR_AUDIO_ENC_DIR", manifest.get("audio_encoder_dir", _asr_audio_enc_dir_e())
        ),
        # --- flags ---
        use_worker=_env_bool("EDGE_LLM_ASR_WORKER", use_worker_default),
        mel_tensor_name=env.get(
            "EDGE_LLM_ASR_MEL_TENSOR_NAME", manifest.get("mel_tensor_name", "mel")
        ),
        max_mel_frames=int(
            env.get("EDGE_LLM_ASR_MAX_MEL_FRAMES", str(manifest.get("max_mel_frames", 6000)))
        ),
        max_slots=max(1, int(max_slots_raw)),
        require_ifb=_env_bool("EDGE_LLM_ASR_REQUIRE_IFB", False),
        stream_mode=env.get(
            "EDGE_LLM_ASR_STREAM_MODE", manifest.get("stream_mode", "accumulate")
        ),
        stream_chunk_sec=float(
            env.get("EDGE_LLM_ASR_STREAM_CHUNK_SEC", str(manifest.get("stream_chunk_sec", 0.5)))
        ),
        stream_unfixed_chunks=int(
            env.get(
                "EDGE_LLM_ASR_STREAM_UNFIXED_CHUNKS",
                str(manifest.get("stream_unfixed_chunks", 2)),
            )
        ),
        stream_unfixed_tokens=int(
            env.get(
                "EDGE_LLM_ASR_STREAM_UNFIXED_TOKENS",
                str(manifest.get("stream_unfixed_tokens", 5)),
            )
        ),
        segment_cap_sec=float(
            env.get(
                "EDGE_LLM_ASR_SEGMENT_CAP_SEC",
                str(manifest.get("segment_cap_sec", 5.5)),
            )
        ),
        mel_settings_path=env.get(
            "EDGE_LLM_ASR_MEL_SETTINGS", manifest.get("mel_settings_path", "")
        ),
        mel_filters_path=env.get(
            "EDGE_LLM_ASR_MEL_FILTERS", manifest.get("mel_filters_path", "")
        ),
        request_audio_wav=_env_bool("EDGELLM_REQUEST_AUDIO_WAV", False),
        # --- sampling ---
        temperature=float(env.get("ASR_TEMPERATURE", "1.0")),
        top_p=float(env.get("ASR_TOP_P", "1.0")),
        top_k=int(env.get("ASR_TOP_K", "1")),
        max_generate_length=int(env.get("ASR_MAX_GENERATE_LENGTH", "200")),
        min_audio_frames=min_audio_frames,
        # --- offline segmentation ---
        offline_segment_enabled=_env_bool("EDGE_LLM_ASR_OFFLINE_SEGMENT", True),
        offline_segment_threshold_s=float(
            env.get("EDGE_LLM_ASR_OFFLINE_SEGMENT_SEC", "6.0")
        ),
        offline_segment_min_s=float(
            env.get("EDGE_LLM_ASR_OFFLINE_MIN_SEGMENT_SEC", "0.4")
        ),
        # --- worker warmup ---
        worker_warmup=worker_warmup,
        prewarm_max=prewarm_max,
        worker_cuda_graph=env.get("EDGE_LLM_ASR_CUDA_GRAPH", "0"),
    )
