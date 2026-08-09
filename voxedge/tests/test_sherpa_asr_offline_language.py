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


# ── the config must actually reach sherpa, not just exist ───────────────────

def _fake_sherpa(captured: dict):
    """Minimal stand-in for the sherpa_onnx module, capturing loader kwargs."""
    import types

    mod = types.ModuleType("sherpa_onnx")

    class _OfflineRecognizer:
        @staticmethod
        def from_sense_voice(**kwargs):
            captured.update(kwargs)
            return object()

    mod.OfflineRecognizer = _OfflineRecognizer
    return mod


def _load_with(monkeypatch, tmp_path, **cfg_kwargs) -> dict:
    import sys

    captured: dict = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", _fake_sherpa(captured))
    model_dir = tmp_path / "sensevoice" / "sherpa-onnx-sense-voice-x"
    model_dir.mkdir(parents=True)
    (model_dir / "model.int8.onnx").write_bytes(b"\x00")
    (model_dir / "tokens.txt").write_bytes(b"\x00")

    be = SherpaASRBackend(
        config=SherpaASRConfig(model_root=str(tmp_path), **cfg_kwargs)
    )
    be._load_offline_recognizer()
    return captured


def test_config_is_passed_through_to_from_sense_voice(monkeypatch, tmp_path):
    """Guards the wiring, not just the dataclass.

    Asserting only on SherpaASRConfig leaves the loader free to drift: someone
    re-hardcoding use_itn=True, or dropping the language kwarg, would not fail
    a single test. So assert on what sherpa actually receives.
    """
    captured = _load_with(
        monkeypatch, tmp_path, offline_use_itn=False, offline_language="yue",
    )
    assert captured["use_itn"] is False
    assert captured["language"] == "yue"


def test_defaults_reach_sherpa_unchanged(monkeypatch, tmp_path):
    """The historical hardcoded behaviour: ITN on, language unset (= auto)."""
    captured = _load_with(monkeypatch, tmp_path)
    assert captured["use_itn"] is True
    assert captured["language"] == ""
