"""SenseVoice binds its language at load time — the backend must not pretend otherwise.

Before this, ``transcribe(audio, language=...)`` accepted a per-request language,
threw it away, and echoed it back in ``TranscriptionResult.language`` — so a
caller asking for ``zh`` got a reply claiming ``zh`` from a recognizer running
in ``auto``. These tests pin the honest behaviour: report what the recognizer
was actually built with, and say once that the request was ignored.

No sherpa_onnx runtime is needed: ``_effective_language`` is pure config logic.
"""
import logging

import pytest

from voxedge.backends.sherpa.asr import SherpaASRBackend, SherpaASRConfig


def _backend(language: str = "") -> SherpaASRBackend:
    return SherpaASRBackend(config=SherpaASRConfig(offline_language=language))


def test_defaults_preserve_historical_behaviour():
    cfg = SherpaASRConfig()
    assert cfg.offline_use_itn is True   # was hardcoded True in _load_offline_recognizer
    assert cfg.offline_language == ""    # was never passed → sherpa default (auto)


@pytest.mark.parametrize("requested", ["auto", "", "   ", None])
def test_unset_request_resolves_to_auto_without_warning(requested, caplog):
    with caplog.at_level(logging.WARNING):
        assert _backend()._effective_language(requested) == "auto"
    assert caplog.records == []


def test_request_matching_the_pin_is_not_warned(caplog):
    with caplog.at_level(logging.WARNING):
        assert _backend("yue")._effective_language("yue") == "yue"
    assert caplog.records == []


def test_conflicting_request_reports_effective_language_and_warns(caplog):
    be = _backend("yue")
    with caplog.at_level(logging.WARNING):
        # The caller asked for zh; the recognizer is built for yue. Returning
        # "zh" here would be the original lie.
        assert be._effective_language("zh") == "yue"
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "'zh'" in msg and "'yue'" in msg


def test_repeated_conflicts_warn_once_per_distinct_language(caplog):
    be = _backend()
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            be._effective_language("zh")
        be._effective_language("ja")
        be._effective_language("zh")
    # One notice per distinct value — a per-utterance caller must not flood logs.
    assert len(caplog.records) == 2
