"""Session-scoped ASR stream options must reach backend.create_stream.

The RK Qwen3 backend owns its own endpoint VAD and reads
``stream_options["vad_endpoint_silence_ms"]`` (rkvoice_stream
``backends/asr/qwen3_rk.py``). Before this pass-through the option was
unreachable: the manager always called ``create_stream(language=...)``, so only
the profile's env value (stamped at startup, not overridable per session) could
tune the endpoint threshold. Measured cost of the missing channel: a 400 ms
backend threshold cut a 1.5 s mid-sentence pause into two turns
(see seeed-local-voice docs/CONFIGURATION.md).

Contract under test:
  * default (no options) → the legacy one-argument call is byte-unchanged, so
    backends whose create_stream does not accept stream_options keep working;
  * options present → forwarded on EVERY stream (re)create, not just the first.
"""

from __future__ import annotations

import asyncio
from typing import Any

from voxedge.engine.asr_session_manager import ASRSessionManager


class _RecordingBackend:
    """create_stream(language=...) only — the legacy backend shape."""

    sample_rate = 16000

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create_stream(self, language: str = "auto") -> str:
        self.calls.append({"language": language})
        return f"stream-{len(self.calls)}"


class _OptionsBackend(_RecordingBackend):
    """create_stream(language=..., stream_options=...) — the RK shape."""

    def create_stream(self, language: str = "auto", stream_options=None) -> str:
        self.calls.append({"language": language, "stream_options": stream_options})
        return f"stream-{len(self.calls)}"


def _manager(backend, **kwargs) -> ASRSessionManager:
    return ASRSessionManager(backend, language="zh", **kwargs)


def test_default_keeps_the_single_argument_call():
    be = _RecordingBackend()
    mgr = _manager(be)
    assert mgr._new_stream_sync() == "stream-1"
    assert be.calls == [{"language": "zh"}]


def test_options_are_forwarded_to_create_stream():
    be = _OptionsBackend()
    mgr = _manager(be, stream_options={"vad_endpoint_silence_ms": 1500})
    assert mgr._new_stream_sync() == "stream-1"
    assert be.calls == [
        {"language": "zh", "stream_options": {"vad_endpoint_silence_ms": 1500}}
    ]


def test_options_are_forwarded_on_every_recreate():
    be = _OptionsBackend()
    mgr = _manager(be, stream_options={"vad_endpoint_silence_ms": 1500})
    mgr._new_stream_sync()
    mgr._new_stream_sync()
    assert [c["stream_options"] for c in be.calls] == [
        {"vad_endpoint_silence_ms": 1500},
        {"vad_endpoint_silence_ms": 1500},
    ]


def test_options_dict_is_copied_not_aliased():
    opts = {"vad_endpoint_silence_ms": 1500}
    be = _OptionsBackend()
    mgr = _manager(be, stream_options=opts)
    opts["vad_endpoint_silence_ms"] = 50  # caller mutates afterwards
    mgr._new_stream_sync()
    assert be.calls[0]["stream_options"] == {"vad_endpoint_silence_ms": 1500}


def test_empty_options_behave_like_absent():
    be = _RecordingBackend()
    mgr = _manager(be, stream_options={})
    assert mgr._new_stream_sync() == "stream-1"
    assert be.calls == [{"language": "zh"}]


def test_legacy_backend_without_the_parameter_is_not_broken(caplog):
    """Jetson (TRT Edge-LLM / Paraformer) and Sherpa take `language` only.

    An operator default or session override must not turn into a TypeError that
    fails ASR: the hint is dropped and the profile threshold applies.
    """
    be = _RecordingBackend()  # create_stream(language) only
    mgr = _manager(be, stream_options={"vad_endpoint_silence_ms": 1500})
    with caplog.at_level("WARNING"):
        assert mgr._new_stream_sync() == "stream-1"
        # Second call must reuse the cached capability (no second warning).
        assert mgr._new_stream_sync() == "stream-2"
    assert be.calls == [{"language": "zh"}, {"language": "zh"}]
    warnings = [r for r in caplog.records if "does not accept stream_options" in r.message]
    assert len(warnings) == 1


def test_var_keyword_backend_counts_as_supporting_options():
    class _KwargsBackend:
        sample_rate = 16000

        def __init__(self) -> None:
            self.calls = []

        def create_stream(self, language="auto", **kwargs):
            self.calls.append({"language": language, **kwargs})
            return "stream"

    be = _KwargsBackend()
    mgr = _manager(be, stream_options={"vad_endpoint_silence_ms": 1500})
    assert mgr._new_stream_sync() == "stream"
    assert be.calls == [{"language": "zh", "stream_options": {"vad_endpoint_silence_ms": 1500}}]
