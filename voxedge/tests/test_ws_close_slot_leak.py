"""Regression tests for the WS-close slot-leak bug (engine path, blocker 3b-ii).

Root cause (codex): on the ASR+TTS V2V path a client WS close drained the recv
loops and ``_watch_input_end`` only set ``asr_session_closed`` — NOT
``client_closed`` / ``tts_flush`` (the flush was gated behind
``if not self.asr_enabled``). So ``_tts_out_task`` spun forever in
``while not client_closed`` → ``Session.run()``'s gather never resolved →
``Session.run()`` never returned → the product ``finally`` that releases the
limiter slot never ran → the session-limiter slot leaked, locking out SLV.

These tests pin the fix WITHOUT any CUDA: mock backends + in-memory / fake-ws
transports, driven through the public Transport API.

Critical invariant under test: a quiet-but-connected client (no WS close, no
disconnect frame) must NOT terminate the session — only a real input-end
(``_CLOSE`` enqueued by the transport) does.
"""
from __future__ import annotations

import asyncio

import numpy as np

from voxedge.backends.mock import MockASR, MockLLM, MockTTS, MockVAD
from voxedge.engine.conversation import Session
from voxedge.engine import ConversationEngine
from voxedge.engine.tool_registry import ToolRegistry
from voxedge.transport import InProcessTransport
from voxedge.transport.base import WebSocketTransport, _CLOSE


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())

    wrapper.__name__ = coro_fn.__name__
    return wrapper


def _pcm(loud: bool, n: int = 512) -> bytes:
    arr = (np.ones(n, dtype=np.int16) * 8000) if loud else np.zeros(n, dtype=np.int16)
    return arr.tobytes()


def _v2v_engine(**kwargs) -> ConversationEngine:
    """Full ASR+TTS V2V loop (the path that previously hung on WS close)."""
    return ConversationEngine(
        backends={
            "asr": MockASR(transcript="hello world", language="English"),
            "tts": MockTTS(sample_rate=24000),
            "vad": MockVAD(silence_chunks=2),
            "llm": MockLLM(reply="Sure."),
        },
        multi_utterance=True,
        **kwargs,
    )


# ───────────────────────────────────────────────────────────────────────
# Test 1: engine-path client disconnect → Session.run() returns + flags set
# ───────────────────────────────────────────────────────────────────────


@run_async
async def test_v2v_client_close_no_speech_returns_and_sets_flags():
    """ASR+TTS V2V: a client that connects then closes WITHOUT producing a
    final transcript (so no LLM turn ever sets tts_flush) is the exact leak
    case — pre-fix _tts_out_task spins in `while not client_closed` and the
    gather never resolves, so Session.run() only unwinds when the OUTER
    wait_for cancels it (a 10s+ hang that leaks the slot for that whole time).

    Post-fix: _watch_input_end sets client_closed + tts_flush, both work tasks
    break, the gather resolves, run() returns promptly, and the product finally
    releases the slot. The tight 4s timeout converts the regression into a hard
    failure (pre-fix this body takes the full outer timeout to unwind)."""
    engine = _v2v_engine()
    transport = InProcessTransport()
    session = Session(engine, transport)

    # No audio at all — client opens the socket then immediately closes it.
    transport.end_input()  # enqueues _CLOSE on both recv queues

    await asyncio.wait_for(session.run(), timeout=4.0)

    assert session.state["client_closed"] is True
    assert session.state["tts_flush"] is True
    assert session.state["asr_session_closed"] is True


@run_async
async def test_v2v_client_close_after_turn_returns():
    """Companion to the no-speech case: a full spoken turn (asr_final → LLM →
    TTS) followed by a client close must also return cleanly. This path sets
    tts_flush via the LLM turn even pre-fix, so it guards the fix doesn't
    regress the normal close-out."""
    engine = _v2v_engine()
    transport = InProcessTransport()
    session = Session(engine, transport)

    for _ in range(3):
        await transport.feed_audio(_pcm(loud=True))
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=False))
    transport.end_input()

    await asyncio.wait_for(session.run(), timeout=5.0)

    assert session.state["client_closed"] is True
    assert session.state["tts_flush"] is True
    assert session.state["asr_session_closed"] is True


# ───────────────────────────────────────────────────────────────────────
# Test 2: quiet-but-connected client → session does NOT terminate
# ───────────────────────────────────────────────────────────────────────


@run_async
async def test_quiet_client_does_not_terminate_session():
    """The critical invariant: a connected client that simply goes silent (no
    _CLOSE / no disconnect) must NOT trip client_closed — the recv loops stay
    blocked on the queue, _watch_input_end never fires, run() keeps going.

    We feed one utterance, never call end_input(), and assert run() is still
    pending after a grace period with client_closed still False."""
    engine = _v2v_engine()
    transport = InProcessTransport()
    session = Session(engine, transport)

    for _ in range(3):
        await transport.feed_audio(_pcm(loud=True))
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=False))
    # NOTE: deliberately NO transport.end_input() — client is quiet, not gone.

    run_task = asyncio.create_task(session.run())
    # Let the engine process the utterance and then idle on the open recv queue.
    await asyncio.sleep(0.5)

    assert not run_task.done(), "quiet client wrongly terminated the session"
    assert session.state["client_closed"] is False
    assert session.state["asr_session_closed"] is False

    # Now actually close → it must wind down promptly.
    transport.end_input()
    await asyncio.wait_for(run_task, timeout=5.0)
    assert session.state["client_closed"] is True


# ───────────────────────────────────────────────────────────────────────
# Test 3: in-flight remote tool future is cancelled on disconnect
# ───────────────────────────────────────────────────────────────────────


@run_async
async def test_inflight_remote_tool_future_cancelled_on_close():
    """A pending remote tool call (awaiting a client tool_result) must be
    cancelled when the client disconnects, so its _dispatch_remote await
    returns a recoverable abort instead of hanging the session."""
    engine = _v2v_engine(tool_registry=ToolRegistry())
    transport = InProcessTransport()
    session = Session(engine, transport)

    # Simulate an in-flight remote dispatch by registering a pending future
    # directly on the registry (same structure _dispatch_remote builds).
    registry = engine.tool_registry
    assert registry is not None
    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    registry._pending_remote["call-123"] = fut

    # Drive a turn then close.
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=True))
    for _ in range(3):
        await transport.feed_audio(_pcm(loud=False))
    transport.end_input()

    await asyncio.wait_for(session.run(), timeout=5.0)

    # The future was cancelled and removed by cancel_pending_remote().
    assert fut.cancelled()
    assert "call-123" not in registry._pending_remote


# ───────────────────────────────────────────────────────────────────────
# Test 4: WebSocketTransport._pump routes WebSocketDisconnect → _CLOSE
# ───────────────────────────────────────────────────────────────────────


class _WebSocketDisconnect(Exception):
    """Duck-typed stand-in for starlette.websockets.WebSocketDisconnect.

    The transport recognizes the disconnect by class NAME (to stay
    FastAPI-import-free), so the name must match exactly.
    """


class _FakeWS:
    """Minimal Starlette-like ws: a scripted receive() that can either yield a
    clean disconnect frame or RAISE WebSocketDisconnect, plus no-op sends."""

    def __init__(self, script):
        self._script = list(script)
        self.closed = False

    async def receive(self):
        if not self._script:
            # Nothing left scripted — emulate a hung-then-cancelled socket.
            await asyncio.sleep(3600)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def send_json(self, obj):  # pragma: no cover - not exercised here
        pass

    async def send_bytes(self, data):  # pragma: no cover
        pass

    async def close(self, code=None):
        self.closed = True


async def _drain_iter(aiter):
    out = []
    async for x in aiter:
        out.append(x)
    return out


@run_async
async def test_ws_pump_clean_disconnect_frame_enqueues_close():
    """A clean {'type':'websocket.disconnect'} frame ends both recv iterators
    (each returns at _CLOSE) and does not raise."""
    ws = _FakeWS([
        {"type": "websocket.receive", "bytes": b"\x00\x01"},
        {"type": "websocket.disconnect"},
    ])
    t = WebSocketTransport(ws)
    audio = await asyncio.wait_for(_drain_iter(t.recv_audio()), timeout=2.0)
    events = await asyncio.wait_for(_drain_iter(t.recv_event()), timeout=2.0)
    assert audio == [b"\x00\x01"]
    assert events == []
    assert t._closed is True


@run_async
async def test_ws_pump_raised_disconnect_enqueues_close_not_swallowed():
    """A RAISED WebSocketDisconnect (Starlette's behaviour once the socket is
    gone) must NOT be silently swallowed into a wedge — the pump must still
    enqueue _CLOSE so both recv iterators return and the engine can tear down.
    """
    ws = _FakeWS([
        {"type": "websocket.receive", "bytes": b"\x02\x03"},
        _WebSocketDisconnect("client gone"),
    ])
    t = WebSocketTransport(ws)
    # If the disconnect were swallowed without enqueueing _CLOSE, these would
    # hang; the timeout converts that regression into a hard failure.
    audio = await asyncio.wait_for(_drain_iter(t.recv_audio()), timeout=2.0)
    events = await asyncio.wait_for(_drain_iter(t.recv_event()), timeout=2.0)
    assert audio == [b"\x02\x03"]
    assert events == []
    assert t._closed is True


# ───────────────────────────────────────────────────────────────────────
# Test 5: full WebSocketTransport + V2V engine, raised disconnect → run returns
# ───────────────────────────────────────────────────────────────────────


@run_async
async def test_v2v_over_ws_raised_disconnect_returns():
    """End-to-end over WebSocketTransport: a raised WebSocketDisconnect mid
    audio stream must make Session.run() return (the leak scenario over the
    real ws adapter, not just the in-process queue)."""
    engine = _v2v_engine()
    audio_frame = _pcm(loud=True)
    silent_frame = _pcm(loud=False)
    script = (
        [{"type": "websocket.receive", "bytes": audio_frame} for _ in range(3)]
        + [{"type": "websocket.receive", "bytes": silent_frame} for _ in range(3)]
        + [_WebSocketDisconnect("client gone")]
    )
    ws = _FakeWS(script)
    transport = WebSocketTransport(ws)
    session = Session(engine, transport)

    await asyncio.wait_for(session.run(), timeout=5.0)

    assert session.state["client_closed"] is True
    assert session.state["tts_flush"] is True
    assert ws.closed is True
