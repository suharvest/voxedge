"""锁住 ASRStream 的资源释放契约。

背景：``close()`` 曾是基类里的空实现，于是服务端那句
``close = getattr(stream, "close", None); if close is not None: close()``
永远拿得到一个什么都不做的方法 —— 判断从来拦不住任何东西。子类忘了实现 close
会静默继承空壳：调用"成功"、资源不释放、零报错。

2026-08-06 的 ASR worker 槽位泄漏正是这么活下来的：
``_TRTEdgeLLMStreamingASRStream`` 从没实现 close，而 worker 侧 max_slots=1，
每次成功识别漏一个槽位，第二轮起 create_stream 必抛 PoolSaturatedError，
设备永远停在「聆听中」。两边日志都看不出异常。

所以这里锁的不是"close 实现得对"，而是"漏写会立刻炸"。
"""
import pytest

from voxedge.backends.base import ASRStream


def _stub_body(ns):
    """填上 ASRStream 的抽象方法，让类能被实例化以外的检查通过。"""
    ns["accept_waveform"] = lambda self, sample_rate, samples: None
    ns["finalize"] = lambda self: ("", None)
    return ns


def _make(name, **attrs):
    return type(name, (ASRStream,), _stub_body(dict(attrs)))


def test_missing_declaration_fails_at_class_definition():
    """不声明 OWNS_RESOURCES —— 定义类时就抛，不用等运行到 close。"""
    with pytest.raises(TypeError, match="必须显式声明 OWNS_RESOURCES"):
        _make("NoDeclaration")


def test_owning_without_close_fails_at_class_definition():
    """声明了拥有资源却没实现 close —— 正是当初那个泄漏的形状。"""
    with pytest.raises(TypeError, match="没有实现 close"):
        _make("OwnsButNoClose", OWNS_RESOURCES=True)


def test_owning_with_close_is_accepted():
    cls = _make("OwnsWithClose", OWNS_RESOURCES=True,
                close=lambda self: setattr(self, "closed", True))
    assert cls.OWNS_RESOURCES is True


def test_explicit_false_is_accepted_and_close_is_a_safe_noop():
    """声明无资源时，基类空实现就是正确语义，调用方可以无条件调。"""
    cls = _make("OwnsNothing", OWNS_RESOURCES=False)
    obj = cls()
    obj.close()
    obj.close()   # 幂等


def test_declaration_is_inherited():
    """从已声明的类再派生，不必重复声明。"""
    parent = _make("Parent", OWNS_RESOURCES=True, close=lambda self: None)
    child = type("Child", (parent,), {})
    assert child.OWNS_RESOURCES is True


# 仓库内所有真实子类都必须已声明；新增后端忘了声明会在收集阶段就失败。
_REAL_SUBCLASSES = [
    ("voxedge.backends.base", "OfflineAccumulateStream"),
    ("voxedge.backends.mock", "MockASRStream"),
    ("voxedge.backends.sherpa.asr", "SherpaASRStream"),
]


@pytest.mark.parametrize("module,name", _REAL_SUBCLASSES)
def test_real_subclasses_declare_ownership(module, name):
    mod = pytest.importorskip(module)
    cls = getattr(mod, name)
    assert cls.OWNS_RESOURCES is not None, f"{name} 未声明 OWNS_RESOURCES"
    if cls.OWNS_RESOURCES:
        assert cls.close is not ASRStream.close, f"{name} 声明拥有资源却未实现 close"


def test_prepare_finalize_stays_optional():
    """prepare_finalize 与 close 性质不同，不该被一起收紧。

    它是可选优化（预编码剩余音频，让 finalize 只跑解码器），空实现是正确语义，
    漏实现不会泄漏任何东西。把它也做成强制只会制造无意义的样板。
    """
    cls = _make("NoPrepare", OWNS_RESOURCES=False)
    cls().prepare_finalize()
