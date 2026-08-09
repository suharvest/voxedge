"""TranscriptionResult.language must describe what happened, not what was asked.

Backends honour a per-request language to wildly different degrees:

  * SenseVoice (sherpa)  — binds language when the recognizer is constructed,
                           so only a deployment-level pin is possible
  * SenseVoice-TRT       — maps and applies it per call
  * Paraformer-TRT       — has no language switch at all

All three previously got this wrong in one of two directions: sherpa and
Paraformer echoed the caller's request back untouched (a caller asking for
`zh` got a reply claiming `zh` from a decoder that never saw it), while
SenseVoice-TRT applied the language and then reported None. These tests pin
the single contract they now share via resolve_reported_language().
"""
import logging

import pytest

from voxedge.backends.base import resolve_reported_language


def _resolve(requested, honoured, warned=None):
    return resolve_reported_language(
        requested, honoured=honoured, backend="test_backend",
        warned=warned if warned is not None else set(),
    )


# ── the three backend shapes ────────────────────────────────────────────────

def test_pinned_backend_reports_the_pin_not_the_request(caplog):
    """sherpa shape: config-level pin, request cannot override it."""
    with caplog.at_level(logging.WARNING):
        assert _resolve("zh", "yue") == "yue"
    assert len(caplog.records) == 1


def test_per_call_backend_reports_what_it_applied():
    """SenseVoice-TRT shape: the request is honoured, so it is also reported."""
    assert _resolve("ja", "ja") == "ja"


def test_selectionless_backend_reports_none(caplog):
    """Paraformer shape: no language switch exists; None, never an echo."""
    with caplog.at_level(logging.WARNING):
        assert _resolve("zh", None) is None
    assert len(caplog.records) == 1
    assert "selects no language" in caplog.records[0].getMessage()


# ── request normalisation ───────────────────────────────────────────────────

@pytest.mark.parametrize("requested", ["auto", "", "   ", None, "AUTO", " Auto "])
def test_unset_or_auto_request_is_silent(requested, caplog):
    with caplog.at_level(logging.WARNING):
        assert _resolve(requested, "yue") == "yue"
        assert _resolve(requested, None) is None
    assert caplog.records == []


def test_case_and_whitespace_do_not_trigger_a_false_conflict(caplog):
    with caplog.at_level(logging.WARNING):
        assert _resolve("  ZH  ", "zh") == "zh"
    assert caplog.records == []


# ── log hygiene ─────────────────────────────────────────────────────────────

def test_warns_once_per_distinct_language(caplog):
    warned = set()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            _resolve("zh", "yue", warned)
        _resolve("ja", "yue", warned)
        _resolve("zh", "yue", warned)
    assert len(caplog.records) == 2


def test_warned_set_is_not_shared_between_backends(caplog):
    a, b = set(), set()
    with caplog.at_level(logging.WARNING):
        _resolve("zh", "yue", a)
        _resolve("zh", "yue", b)
    # Two independent backends each get to report the problem once.
    assert len(caplog.records) == 2
