"""Server-loop INTEGRATION regression tests — drive the real ConversationEngine
over InProcessTransport through the REAL remote-tool path (engine emits a
SERVER_TOOL_CALL; a fake agent replies with a CLIENT_TOOL_RESULT), with a
scriptable LLM, to lock in the end-to-end sequencing the unit tests
(test_barge_in_and_template.py) only cover white-box:

  S1  preamble ("好的。") is spoken BEFORE the round-2 reply (remote tool).
  S4  multi-turn (x3): each turn calls the tool, fires the preamble, replies —
      no wedge / deadlock across turns on one session.
  S3  barge-in WHILE the remote tool is in flight (arm 'moving') cancels the
      turn: no stale reply leaks, session not wedged.
  S2  barge-in mid round-2 reply stops the remaining sentences (queue drained,
      LLM turn cancelled cooperatively).

Synchronisation is event-driven (wait for the actual emitted event / gate the
LLM between sentences) rather than wall-clock sleeps, so the timing assertions
are deterministic, not flaky. No pytest-asyncio (asyncio.run per test).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

import numpy as np

from voxedge.backends.base import LLMBackend, LLMEvent
from voxedge.backends.mock import MockASR, MockTTS, MockVAD
from voxedge.engine import ConversationEngine
from voxedge.engine.tool_registry import ToolRegistry
from voxedge.transport import InProcessTransport


def run_async(coro_fn):
    def wrapper():
        asyncio.run(coro_fn())
    wrapper.__name__ = coro_fn.__name__
    return wrapper


def _pcm(loud: bool, n: int = 512) -> bytes:
    arr = np.ones(n, dtype=np.int16) * 8000 if loud else np.zeros(n, dtype=np.int16)
    return arr.tobytes()


class ScriptedToolLLM(LLMBackend):
    """A turn whose last message is a tool result is round 2 (reply); otherwise
    round 1 (call the tool). round 2 optionally awaits ``gate`` before each
    sentence after the first, so a test can deterministically interrupt the
    reply mid-stream."""

    def __init__(self, tool_name: str, round2_sentences: list[str],
                 gate: Optional[asyncio.Event] = None):
        self.tool_name = tool_name
        self.round2_sentences = round2_sentences
        self.gate = gate
        self.calls = 0
        self.system_prompts: list[str] = []

    @property
    def name(self) -> str:
        return "scripted_tool_llm"

    async def stream(self, messages, **kw) -> AsyncIterator[str]:  # pragma: no cover
        if False:
            yield ""

    async def stream_events(self, messages, *, tools=None, **kw) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        self.system_prompts.append(
            next(
                (str(m.get("content") or "") for m in messages if m.get("role") == "system"),
                "",
            )
        )
        last_role = messages[-1].get("role") if messages else None
        if last_role != "tool":
            yield LLMEvent(kind="tool_call_delta", tool_call_index=0,
                           tool_call_id="call_0", name=self.tool_name, arguments="{}")
            yield LLMEvent(kind="finish", finish_reason="tool_calls")
        else:
            for i, s in enumerate(self.round2_sentences):
                if i and self.gate is not None:
                    await self.gate.wait()
                yield LLMEvent(kind="text", text=s)
            yield LLMEvent(kind="finish", finish_reason="stop")


def _remote_registry(tool_name: str, preamble: str = "好的。") -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        tool_name, {"type": "object", "properties": {}}, None,
        preamble_text=preamble, response_mode="parallel", dispatch_mode="remote",
        description=f"do {tool_name}",
    )
    return reg


class _Sim:
    def __init__(self, llm, registry, *, auto_create_response: bool = True):
        self.transport = InProcessTransport()
        self.engine = ConversationEngine(
            backends={"asr": MockASR(transcript="挥手"), "vad": MockVAD(silence_chunks=2),
                      "llm": llm, "tts": MockTTS()},
            tool_registry=registry, system_prompt="SYS /no_think", multi_utterance=True,
            auto_create_response=auto_create_response,
        )
        self.timeline: list[tuple[str, str]] = []
        self.tool_calls_seen = 0
        self.tool_release = asyncio.Event()
        self.tool_release.set()
        self._agent_task = None
        self._run_task = None

    async def _fake_agent(self):
        try:
            while True:
                ev = await self.transport.events_out()
                typ = ev.get("type")
                detail = ev.get("sentence") or ev.get("name") or ev.get("event") or ""
                self.timeline.append((typ, str(detail)))
                if typ == "tool_call":
                    self.tool_calls_seen += 1
                    asyncio.create_task(self._answer(ev.get("call_id"), ev.get("name")))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _answer(self, cid, name):
        await self.tool_release.wait()
        await self.transport.feed_event(
            {"type": "tool_result", "call_id": cid, "name": name,
             "ok": True, "result": {"started": True, "success": True}})

    def start(self):
        self._agent_task = asyncio.create_task(self._fake_agent())
        self._run_task = asyncio.create_task(self.engine.run(self.transport))

    async def utter(self):
        for _ in range(3):
            await self.transport.feed_audio(_pcm(True))
        for _ in range(3):
            await self.transport.feed_audio(_pcm(False))

    async def barge_in(self):
        for _ in range(2):
            await self.transport.feed_audio(_pcm(True))

    async def wait_for(self, typ, detail_substr=None, timeout=5.0):
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            for t, d in self.timeline:
                if t == typ and (detail_substr is None or detail_substr in d):
                    return
            await asyncio.sleep(0.005)
        raise AssertionError(f"timed out waiting for {typ}/{detail_substr}; timeline={self.timeline}")

    async def count_eventually(self, typ, target, timeout=5.0):
        """Wait until at least `target` events of `typ` are seen; return count."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            n = sum(1 for t, _ in self.timeline if t == typ)
            if n >= target:
                return n
            await asyncio.sleep(0.005)
        return sum(1 for t, _ in self.timeline if t == typ)

    async def finish(self, timeout=10.0):
        self.transport.end_input()
        deadlocked = False
        try:
            await asyncio.wait_for(self._run_task, timeout=timeout)
        except asyncio.TimeoutError:
            deadlocked = True
            self._run_task.cancel()
        if self._agent_task:
            self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass
        return deadlocked

    def sentences(self):
        return [d for t, d in self.timeline if t == "tts_sentence_done"]


@run_async
async def test_manual_response_waits_for_prompt_update_then_create():
    llm = ScriptedToolLLM("wave", ["更新后的回复。"])
    sim = _Sim(
        llm,
        _remote_registry("wave"),
        auto_create_response=False,
    )
    sim.start()
    await sim.utter()
    await sim.wait_for("asr_final")
    await asyncio.sleep(0)
    assert llm.calls == 0, "LLM must not start before response.create"

    await sim.transport.feed_event({
        "type": "tool_advertise",
        "tools": [],
        "system_prompt": "UPDATED [Faces: Alice] /no_think",
        "warm_prefix": False,
    })
    await sim.transport.feed_event({"type": "response.create"})
    await sim.wait_for("tts_sentence_done", "更新后的回复")
    await sim.finish()

    assert llm.calls == 2
    assert llm.system_prompts[0] == "UPDATED [Faces: Alice] /no_think"


# ── S1: preamble before reply ─────────────────────────────────────────

@run_async
async def test_s1_preamble_before_reply():
    llm = ScriptedToolLLM("wave", ["这是动作后的回复。"])
    sim = _Sim(llm, _remote_registry("wave"))
    sim.start()
    await sim.utter()
    await sim.wait_for("tts_sentence_done", "回复")  # turn produced its reply
    await sim.finish()
    sents = sim.sentences()
    assert any("好的" in s for s in sents), f"no preamble: {sents}"
    assert any("回复" in s for s in sents), f"no reply: {sents}"
    pre_i = min(i for i, s in enumerate(sents) if "好的" in s)
    rep_i = min(i for i, s in enumerate(sents) if "回复" in s)
    assert pre_i < rep_i, f"preamble must precede reply: {sents}"
    assert llm.calls == 2, f"expected round1(tool)+round2(reply), got {llm.calls}"


# ── S4: multi-turn, no wedge ──────────────────────────────────────────

@run_async
async def test_s4_multi_turn_no_wedge():
    llm = ScriptedToolLLM("wave", ["回复。"])
    sim = _Sim(llm, _remote_registry("wave"))
    sim.start()
    for n in range(1, 4):
        await sim.utter()
        await sim.count_eventually("asr_final", n)
        await sim.count_eventually("tts_sentence_done", n * 2)  # preamble + reply each
    deadlocked = await sim.finish()
    assert not deadlocked, "session wedged across turns"
    assert sim.tool_calls_seen == 3, f"each turn should call the tool, got {sim.tool_calls_seen}"
    assert sum(1 for s in sim.sentences() if "好的" in s) == 3, "preamble should fire every turn"


# ── S3: barge-in while the remote tool is in flight ───────────────────

@run_async
async def test_s3_bargein_during_tool_dispatch():
    llm = ScriptedToolLLM("wave", ["不该被听到的回复。"])
    sim = _Sim(llm, _remote_registry("wave"))
    sim.tool_release.clear()           # HOLD the tool result (arm 'moving')
    sim.start()
    await sim.utter()
    await sim.wait_for("tts_sentence_done", "好的")  # preamble out, turn now awaiting tool
    await sim.wait_for("tool_call", "wave")
    await sim.barge_in()               # user interrupts mid-action
    await sim.wait_for("vad_event", "speech_start")
    sim.tool_release.set()             # arm finishes late; result arrives post-barge
    deadlocked = await sim.finish()
    assert not deadlocked, "barge-in during tool dispatch wedged the session"
    assert not any("不该" in s for s in sim.sentences()), \
        f"stale reply leaked after barge-in: {sim.sentences()}"


# ── S2: barge-in mid round-2 reply stops the rest ─────────────────────

@run_async
async def test_s2_bargein_mid_reply_stops_remaining():
    gate = asyncio.Event()  # round2 waits on this before sentence 2+
    llm = ScriptedToolLLM("wave", ["第一句。", "第二句。", "第三句。"], gate=gate)
    sim = _Sim(llm, _remote_registry("wave"))
    sim.start()
    await sim.utter()
    await sim.wait_for("tts_sentence_done", "第一句")  # reply started (sentence 1 out)
    await sim.barge_in()                                # interrupt
    await sim.wait_for("vad_event", "speech_start")
    gate.set()                                          # let the LLM try sentence 2+
    deadlocked = await sim.finish()
    assert not deadlocked
    played = [s for s in sim.sentences() if "句" in s]
    assert played == ["第一句。"], f"barge-in must stop sentences 2+: played={played}"
