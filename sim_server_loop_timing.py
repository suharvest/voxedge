"""Server-loop timing/sequencing simulation — drives the REAL voxedge
ConversationEngine over InProcessTransport through the REAL remote-tool path
(engine emits SERVER_TOOL_CALL; a fake agent replies with CLIENT_TOOL_RESULT),
with a scriptable LLM, to probe sequencing defects WITHOUT a device or voice.

Run: cd ~/project/voxedge && .venv/bin/python sim_server_loop_timing.py
Prints a per-scenario event timeline + PASS/DEFECT verdicts.
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


def _pcm(loud: bool, n: int = 512) -> bytes:
    arr = np.ones(n, dtype=np.int16) * 8000 if loud else np.zeros(n, dtype=np.int16)
    return arr.tobytes()


class ScriptedToolLLM(LLMBackend):
    """round 1 → tool_call(`tool_name`); round 2 → `round2_sentences` (one text
    event per sentence, optional inter-sentence delay so a barge-in can land
    mid-reply). `calls` counts rounds entered."""

    def __init__(self, tool_name: str, round2_sentences: list[str],
                 round2_gap_s: float = 0.0):
        self.tool_name = tool_name
        self.round2_sentences = round2_sentences
        self.round2_gap_s = round2_gap_s
        self.calls = 0

    @property
    def name(self) -> str:
        return "scripted_tool_llm"

    async def stream(self, messages, **kw) -> AsyncIterator[str]:  # pragma: no cover
        if False:
            yield ""

    async def stream_events(self, messages, *, tools=None, **kw) -> AsyncIterator[LLMEvent]:
        self.calls += 1
        # Per-turn correct: a turn whose last message is a tool result is
        # round 2 (reply); otherwise it's round 1 (decide to call the tool).
        last_role = messages[-1].get("role") if messages else None
        if last_role != "tool":
            yield LLMEvent(kind="tool_call_delta", tool_call_index=0,
                           tool_call_id="call_0", name=self.tool_name, arguments="{}")
            yield LLMEvent(kind="finish", finish_reason="tool_calls")
        else:
            for i, s in enumerate(self.round2_sentences):
                if i and self.round2_gap_s:
                    await asyncio.sleep(self.round2_gap_s)
                yield LLMEvent(kind="text", text=s)
            yield LLMEvent(kind="finish", finish_reason="stop")


def make_remote_registry(tool_name: str, preamble: str = "好的。") -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        tool_name, {"type": "object", "properties": {}},
        None,  # remote noop handler; dispatch_mode="remote" proxies over the wire
        preamble_text=preamble, response_mode="parallel", dispatch_mode="remote",
        description=f"do {tool_name}",
    )
    return reg


class Sim:
    """Drives engine.run + a fake agent that answers tool_calls, recording a
    timestamped event timeline."""

    def __init__(self, llm, registry, tool_result_delay_s: float = 0.0):
        self.transport = InProcessTransport()
        self.engine = ConversationEngine(
            backends={"asr": MockASR(transcript="挥手"), "vad": MockVAD(silence_chunks=2),
                      "llm": llm, "tts": MockTTS()},
            tool_registry=registry, system_prompt="SYS /no_think", multi_utterance=True,
        )
        self.tool_result_delay_s = tool_result_delay_s
        self.timeline: list[tuple[float, str, str]] = []  # (t, type, detail)
        self._t0 = None
        self._agent_task = None
        self._run_task = None
        self.tool_release = asyncio.Event()
        self.tool_release.set()  # default: respond immediately
        self.tool_calls_seen = 0

    def _now(self):
        loop = asyncio.get_event_loop()
        if self._t0 is None:
            self._t0 = loop.time()
        return loop.time() - self._t0

    async def _fake_agent(self):
        """Sole consumer of outbound events; records timeline + answers tool_calls."""
        try:
            while True:
                ev = await self.transport.events_out()
                typ = ev.get("type")
                detail = ev.get("sentence") or ev.get("name") or ev.get("event") or ""
                self.timeline.append((self._now(), typ, str(detail)[:40]))
                if typ == "tool_call":
                    self.tool_calls_seen += 1
                    cid = ev.get("call_id")
                    name = ev.get("name")
                    asyncio.create_task(self._answer(cid, name))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # transport closed
            self.timeline.append((self._now(), "_agent_exit", repr(e)[:40]))

    async def _answer(self, cid, name):
        if self.tool_result_delay_s:
            await asyncio.sleep(self.tool_result_delay_s)
        await self.tool_release.wait()
        await self.transport.feed_event(
            {"type": "tool_result", "call_id": cid, "name": name,
             "ok": True, "result": {"started": True, "success": True}})
        self.timeline.append((self._now(), "_agent_sent_result", name or ""))

    def start(self):
        self._agent_task = asyncio.create_task(self._fake_agent())
        self._run_task = asyncio.create_task(self.engine.run(self.transport))

    async def utter(self, loud=3, silent=3):
        for _ in range(loud):
            await self.transport.feed_audio(_pcm(True))
        for _ in range(silent):
            await self.transport.feed_audio(_pcm(False))

    async def barge_in(self, loud=2):
        """Inject speech (VAD SPEECH_START) to interrupt an in-flight turn."""
        for _ in range(loud):
            await self.transport.feed_audio(_pcm(True))

    async def finish(self, timeout=10.0):
        self.transport.end_input()
        try:
            await asyncio.wait_for(self._run_task, timeout=timeout)
        except asyncio.TimeoutError:
            self.timeline.append((self._now(), "_DEADLOCK", "run did not finish"))
            self._run_task.cancel()
        if self._agent_task:
            self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass

    def tts_sentences(self):
        return [d for _, t, d in self.timeline if t == "tts_sentence_done"]

    def print_timeline(self, title):
        print(f"\n=== {title} ===")
        for t, typ, d in self.timeline:
            print(f"  {t:6.2f}s  {typ:22s} {d}")


# ──────────────────────── scenarios ────────────────────────

def verdict(name, ok, msg):
    print(f"  [{'PASS' if ok else 'DEFECT'}] {name}: {msg}")
    return ok


async def s1_preamble_ordering():
    llm = ScriptedToolLLM("wave", ["这是动作后的回复。"])
    sim = Sim(llm, make_remote_registry("wave"))
    sim.start()
    await sim.utter()
    await asyncio.sleep(2.0)
    await sim.finish()
    sim.print_timeline("S1 preamble ordering (single turn, remote tool)")
    sents = sim.tts_sentences()
    has_pre = any("好的" in s for s in sents)
    has_reply = any("回复" in s for s in sents)
    pre_first = has_pre and has_reply and (
        min(i for i, s in enumerate(sents) if "好的" in s) <
        min(i for i, s in enumerate(sents) if "回复" in s))
    ok = verdict("S1", has_pre and has_reply and pre_first and llm.calls == 2,
                 f"sents={sents} calls={llm.calls} (preamble before reply={pre_first})")
    return ok


async def s4_multi_turn():
    llm = ScriptedToolLLM("wave", ["回复。"])
    sim = Sim(llm, make_remote_registry("wave"))
    sim.start()
    for _ in range(3):
        await sim.utter()
        await asyncio.sleep(1.2)
    await sim.finish()
    sim.print_timeline("S4 multi-turn x3")
    finals = [1 for _, t, _ in sim.timeline if t == "asr_final"]
    pre = sum(1 for s in sim.tts_sentences() if "好的" in s)
    deadlock = any(t == "_DEADLOCK" for _, t, _ in sim.timeline)
    ok = verdict("S4", len(finals) == 3 and not deadlock and pre >= 3,
                 f"asr_finals={len(finals)} preambles={pre} tool_calls={sim.tool_calls_seen} deadlock={deadlock}")
    return ok


async def s3_bargein_during_tool():
    """Hold the tool_result (arm 'moving'); barge-in while held → in-flight
    turn must cancel, no round2 reply, session not wedged."""
    llm = ScriptedToolLLM("wave", ["不该被听到的回复。"])
    sim = Sim(llm, make_remote_registry("wave"))
    sim.tool_release.clear()  # HOLD tool result
    sim.start()
    await sim.utter()
    await asyncio.sleep(1.0)  # turn now blocked awaiting tool result
    await sim.barge_in()      # user interrupts mid-action
    await asyncio.sleep(0.8)
    sim.tool_release.set()    # arm finishes late (result arrives post-barge)
    await asyncio.sleep(0.8)
    await sim.finish()
    sim.print_timeline("S3 barge-in during tool dispatch")
    sents = sim.tts_sentences()
    leaked = any("不该" in s for s in sents)
    barged = any(t == "vad_event" and d == "speech_start" for _, t, d in sim.timeline)
    deadlock = any(t == "_DEADLOCK" for _, t, _ in sim.timeline)
    ok = verdict("S3", barged and not leaked and not deadlock,
                 f"barge_seen={barged} stale_reply_leaked={leaked} deadlock={deadlock} sents={sents}")
    return ok


async def s2_bargein_mid_reply():
    """round2 emits 3 sentences with gaps; barge-in after the first → the rest
    must NOT reach TTS (queue drained, turn cancelled)."""
    llm = ScriptedToolLLM("wave", ["第一句。", "第二句。", "第三句。"], round2_gap_s=1.0)
    sim = Sim(llm, make_remote_registry("wave"))
    sim.start()
    await sim.utter()
    await asyncio.sleep(0.3)    # round1+tool done, round2 emitted sentence 1, now in gap
    await sim.barge_in()        # interrupt mid-reply (during the inter-sentence gap)
    await asyncio.sleep(2.5)
    await sim.finish()
    sim.print_timeline("S2 barge-in mid round2 reply")
    sents = sim.tts_sentences()
    n_reply = sum(1 for s in sents if "句" in s)
    barged = any(t == "vad_event" and d == "speech_start" for _, t, d in sim.timeline)
    # Defect if all 3 reply sentences played despite the barge-in.
    ok = verdict("S2", barged and n_reply < 3,
                 f"reply_sentences_played={n_reply}/3 barge_seen={barged} sents={sents}")
    return ok


async def main():
    results = {}
    for fn in (s1_preamble_ordering, s4_multi_turn, s3_bargein_during_tool, s2_bargein_mid_reply):
        try:
            results[fn.__name__] = await fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            results[fn.__name__] = False
    print("\n==================== SUMMARY ====================")
    for k, v in results.items():
        print(f"  {'PASS  ' if v else 'DEFECT'}  {k}")


if __name__ == "__main__":
    asyncio.run(main())
