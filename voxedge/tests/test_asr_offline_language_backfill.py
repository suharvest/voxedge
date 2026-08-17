"""缺语言标签的离线段：回填 + 计数。

ASR head 会给每份正常转写加上 "language <Lang>" 前缀，_strip_language_prefix
把它变成 TranscriptionResult.language。一段返回 language=None 就说明它没进入
int4 recipe 验证时用的解码契约 —— 实测一段 35 分钟音频里，每个干净段都报了
语言，唯一复读的那段报 None。
"""
from voxedge.backends.jetson.trt_edge_llm_asr import _label_unlabelled_segments


def test_all_labelled_reports_zero() -> None:
    texts, unlabelled = _label_unlabelled_segments(
        [("你好", "Chinese"), ("世界", "Chinese")], "Chinese"
    )
    assert unlabelled == 0
    assert texts == ["你好", "世界"]


def test_unlabelled_segment_counted_and_text_kept() -> None:
    """文本仍然交出去（塌缩守卫已经处理过），但计数要暴露给调用方。"""
    texts, unlabelled = _label_unlabelled_segments(
        [("你好", "Chinese"), ("一方面我们看到", None), ("世界", "Chinese")], "Chinese"
    )
    assert unlabelled == 1
    assert texts == ["你好", "一方面我们看到", "世界"]


def test_majority_language_wins_over_request() -> None:
    """回填取同文件其他段的多数语言，不取请求参数 —— 请求里的 language 从来
    没有到达解码器，worker 只会剥标签、不会预置。"""
    texts, unlabelled = _label_unlabelled_segments(
        [("hello", "English"), ("loop", None), ("world", "English")], "Chinese"
    )
    assert unlabelled == 1
    assert texts == ["hello", "loop", "world"]


def test_all_unlabelled_falls_back_to_caller() -> None:
    texts, unlabelled = _label_unlabelled_segments([("a", None), ("b", None)], "Chinese")
    assert unlabelled == 2
    assert texts == ["a", "b"]


def test_empty_input() -> None:
    texts, unlabelled = _label_unlabelled_segments([], "Chinese")
    assert unlabelled == 0
    assert texts == []
