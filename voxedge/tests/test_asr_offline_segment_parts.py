"""离线分段结果的拆解：缺语言标签的计数 + 多数语言。

ASR head 会给每份正常转写加上 "language <Lang>" 前缀，_strip_language_prefix
把它变成 TranscriptionResult.language。一段返回 language=None 就说明它没进入
int4 recipe 验证时用的解码契约 —— 实测一段 35 分钟音频里，每个干净段都报了
语言，唯一复读的那段报 None。
"""
from voxedge.backends.jetson.trt_edge_llm_asr import _split_segment_parts


def test_all_labelled_reports_zero() -> None:
    texts, unlabelled, majority = _split_segment_parts(
        [("你好", "Chinese"), ("世界", "Chinese")]
    )
    assert (texts, unlabelled, majority) == (["你好", "世界"], 0, "Chinese")


def test_unlabelled_segment_counted_and_text_kept() -> None:
    """文本仍然交出去（塌缩守卫已经处理过），但计数要暴露给调用方。"""
    texts, unlabelled, majority = _split_segment_parts(
        [("你好", "Chinese"), ("一方面我们看到", None), ("世界", "Chinese")]
    )
    assert unlabelled == 1
    assert majority == "Chinese"
    assert texts == ["你好", "一方面我们看到", "世界"]


def test_majority_language_from_labelled_segments() -> None:
    texts, unlabelled, majority = _split_segment_parts(
        [("hello", "English"), ("loop", None), ("world", "English")]
    )
    assert (unlabelled, majority) == (1, "English")
    assert texts == ["hello", "loop", "world"]


def test_all_unlabelled_reports_no_language() -> None:
    """一段都没报语言时不能凭空造一个 —— 调用方传进来的 language 从来没有
    到达解码器，回显它等于谎称检测到了。"""
    texts, unlabelled, majority = _split_segment_parts([("a", None), ("b", None)])
    assert (unlabelled, majority) == (2, None)
    assert texts == ["a", "b"]


def test_empty_input() -> None:
    assert _split_segment_parts([]) == ([], 0, None)
