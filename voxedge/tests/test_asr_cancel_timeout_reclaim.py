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

from voxedge.engine.asr_session_manager import ASRSessionManager, SessionState


class _WedgedStream:
    """cancel() 卡住一段时间才返回，模拟 worker 无响应。"""

    def __init__(self, block_s: float) -> None:
        self._block_s = block_s
        self.closed = False
        self.cancel_returned = threading.Event()

    def cancel(self) -> None:
        threading.Event().wait(self._block_s)
        self.cancel_returned.set()

    def close(self) -> None:
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


def test_cancel_fast_path_still_closes_synchronously() -> None:
    """未超时时维持原行为：executor 调用返回后立即同步 close。"""

    async def scenario() -> _WedgedStream:
        stream = _WedgedStream(block_s=0.0)
        mgr = _mgr_with(stream, _BackendNoRestart())
        await mgr.cancel(reason="test-fast")
        return stream

    stream = asyncio.run(scenario())
    assert stream.closed is True
