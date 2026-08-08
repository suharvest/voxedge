"""cancel 超时路径必须回收 stream —— 即使后端没有 restart_worker。

背景：超时分支刻意不同步 close（_cancel_call 线程可能仍持有那个 C++ stream，
同步 close 会和它竞争），改为交给 backend.restart_worker() 回收 worker 侧资源。
但只有 trt_edge_llm 实现了 restart_worker，其他后端走到这条路径就永久漏掉这一段。

修复：无论后端有没有 restart_worker，都挂一个延迟 close —— 等 _cancel_call
线程真正返回后再回收，那时已无并发访问。

破坏链路验证（2026-08-07）：去掉 _schedule_deferred_close 调用后，
test_cancel_timeout_closes_stream_without_restart_worker 会因 closed=False 失败。
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from voxedge.engine.asr_session_manager import ASRSessionManager, SessionState


class _WedgedStream:
    """cancel() 卡住一段时间才返回，模拟 worker 无响应。

    记录事件**顺序**而不只是最终状态：close 必须发生在 cancel 返回之后。
    只断言「最终 closed 为真」是不够的 —— wait_for 超时会把 future 置为
    CANCELLED 从而让 done_callback 立刻触发，这种竞态实现同样能让
    closed 变成真，却是在 cancel 仍在跑的时候关的。
    """

    def __init__(self, block_s: float) -> None:
        self._block_s = block_s
        self.closed = False
        self.cancel_returned = threading.Event()
        self.events: list[str] = []
        self._lock = threading.Lock()

    def _record(self, name: str) -> None:
        with self._lock:
            self.events.append(name)

    def cancel(self) -> None:
        threading.Event().wait(self._block_s)
        self._record("cancel_returned")
        self.cancel_returned.set()

    def close(self) -> None:
        self._record("close")
        self.closed = True


class _BackendNoRestart:
    """没有 restart_worker 的后端 —— 除 trt_edge_llm 外都是这种。"""

    def create_stream(self, **_kw):
        raise AssertionError("测试不应触发新建 stream")


def _mgr_with(stream, backend) -> ASRSessionManager:
    mgr = ASRSessionManager(backend)
    mgr._stream = stream
    mgr._state = SessionState.ACTIVE
    return mgr


def test_cancel_timeout_closes_stream_without_restart_worker() -> None:
    """超时 + 后端无 restart_worker → stream 仍须在线程返回后被 close。"""

    async def scenario() -> _WedgedStream:
        # cancel 阻塞时间要长于 _CANCEL_ACK_TIMEOUT_S，才会走超时分支
        block = ASRSessionManager._CANCEL_ACK_TIMEOUT_S + 0.5
        stream = _WedgedStream(block_s=block)
        mgr = _mgr_with(stream, _BackendNoRestart())

        await mgr.cancel(reason="test-timeout")
        assert mgr._state is SessionState.IDLE
        # 超时分支不得同步 close（那样会与仍在跑的 _cancel_call 线程竞争）
        assert stream.closed is False, "超时分支不应同步 close"

        # 等 _cancel_call 线程自己返回，延迟回收此时才应发生
        deadline = asyncio.get_event_loop().time() + block + 3.0
        while asyncio.get_event_loop().time() < deadline:
            if stream.closed:
                break
            await asyncio.sleep(0.05)
        return stream

    stream = asyncio.run(scenario())
    assert stream.cancel_returned.is_set(), "_cancel_call 线程应已返回"
    assert stream.closed is True, "线程返回后 stream 必须被回收（延迟 close 未生效）"
    # 核心断言：顺序。close 早于 cancel 返回 = 与执行线程并发访问同一个
    # C++ stream，正是这条路径当初拒绝同步 close 的原因。
    assert stream.events == ["cancel_returned", "close"], (
        f"close 必须发生在 cancel 返回之后，实际顺序: {stream.events}"
    )


def test_outer_cancellation_still_reclaims_and_resets() -> None:
    """外层 wait_for 取消这个协程时，仍须挂延迟回收并复位状态。

    真实路径：asr_loop.py 用 wait_for(mgr.cancel(...), timeout=2.0) 包裹本协程。
    CancelledError 继承 BaseException，不会落进 `except Exception`。
    """

    async def scenario():
        block = ASRSessionManager._CANCEL_ACK_TIMEOUT_S + 2.0
        stream = _WedgedStream(block_s=block)
        mgr = _mgr_with(stream, _BackendNoRestart())

        # 外层超时必须早于内层 _CANCEL_ACK_TIMEOUT_S，才能走到 CancelledError
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(mgr.cancel(reason="outer"), timeout=0.2)

        assert mgr._state is SessionState.IDLE, "被外层取消后状态未复位，会卡在 CANCELLING"

        deadline = asyncio.get_event_loop().time() + block + 3.0
        while asyncio.get_event_loop().time() < deadline:
            if stream.closed:
                break
            await asyncio.sleep(0.05)
        return stream

    stream = asyncio.run(scenario())
    assert stream.closed is True, "被外层取消后 stream 未被回收"
    assert stream.events == ["cancel_returned", "close"], (
        f"close 仍须在 cancel 返回之后，实际: {stream.events}"
    )


class _BackendSlowRestart:
    """有 restart_worker，但它很慢 —— 制造「内层已超时、正在 restart 时被外层
    取消」这个窗口。CancelledError 从 except 块内部抛出，同级 handler 接不住。"""

    def __init__(self, block_s: float) -> None:
        self._block_s = block_s

    def restart_worker(self) -> None:
        threading.Event().wait(self._block_s)

    def create_stream(self, **_kw):
        raise AssertionError("测试不应触发新建 stream")


def test_cancelled_during_restart_worker_still_resets() -> None:
    """内层超时 → 正在 restart_worker → 外层取消。状态仍须复位，stream 仍须回收。"""

    async def scenario():
        inner = ASRSessionManager._CANCEL_ACK_TIMEOUT_S
        stream = _WedgedStream(block_s=inner + 2.0)
        mgr = _mgr_with(stream, _BackendSlowRestart(block_s=5.0))

        # 外层超时落在「内层已超时、restart_worker 仍在跑」的窗口内
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(mgr.cancel(reason="during-restart"), timeout=inner + 0.5)

        assert mgr._state is SessionState.IDLE, (
            "restart_worker 期间被取消后状态未复位，manager 会永久卡在 CANCELLING"
        )
        deadline = asyncio.get_event_loop().time() + 8.0
        while asyncio.get_event_loop().time() < deadline:
            if stream.closed:
                break
            await asyncio.sleep(0.05)
        return stream

    stream = asyncio.run(scenario())
    assert stream.closed is True, "该窗口下 stream 未被回收"
    assert stream.events == ["cancel_returned", "close"]


def test_cancel_fast_path_still_closes_synchronously() -> None:
    """未超时时维持原行为：executor 调用返回后立即同步 close。"""

    async def scenario() -> _WedgedStream:
        stream = _WedgedStream(block_s=0.0)
        mgr = _mgr_with(stream, _BackendNoRestart())
        await mgr.cancel(reason="test-fast")
        return stream

    stream = asyncio.run(scenario())
    assert stream.closed is True


class _RestartTrackingBackend:
    """记录 restart_worker 与 create_stream 的相对顺序。

    codex review 第四轮：外层取消 _maybe_restart_worker 后 finally 会复位 IDLE
    并放开锁，而 executor 线程仍在杀 worker。此时若新会话立刻建流并发 begin，
    就会与重启撞车（两者用不同的锁）。
    """

    def __init__(self, restart_block_s: float) -> None:
        self._block = restart_block_s
        self.events: list[str] = []
        self._lock = threading.Lock()

    def _record(self, name: str) -> None:
        with self._lock:
            self.events.append(name)

    def restart_worker(self) -> None:
        self._record("restart_begin")
        threading.Event().wait(self._block)
        self._record("restart_end")

    def create_stream(self, **_kw):
        self._record("create_stream")
        return _WedgedStream(block_s=0.0)


def test_new_stream_waits_for_pending_restart() -> None:
    """restart 仍在跑时不得建流 —— create_stream 必须排在 restart_end 之后。"""

    async def scenario():
        be = _RestartTrackingBackend(restart_block_s=1.0)
        stream = _WedgedStream(block_s=ASRSessionManager._CANCEL_ACK_TIMEOUT_S + 2.0)
        mgr = _mgr_with(stream, be)

        # 外层在 restart 执行途中取消 cancel()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                mgr.cancel(reason="during-restart"),
                timeout=ASRSessionManager._CANCEL_ACK_TIMEOUT_S + 0.3,
            )
        assert mgr._state is SessionState.IDLE
        assert "restart_begin" in be.events and "restart_end" not in be.events, (
            "前置条件不成立：restart 应仍在进行中"
        )

        await mgr.on_speech_start()
        return be

    be = asyncio.run(scenario())
    assert "create_stream" in be.events, "未建流，测试前提不成立"
    assert be.events.index("restart_end") < be.events.index("create_stream"), (
        f"新流在 restart 结束前就建了，会与重启撞车: {be.events}"
    )
