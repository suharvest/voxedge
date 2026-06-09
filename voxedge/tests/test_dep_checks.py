"""G6: friendly optional-dependency errors for backend extras.

When a backend whose extra was never installed is selected, the lazy native
import previously surfaced as a bare ``ModuleNotFoundError: No module named
'sherpa_onnx'`` with no hint. ``voxedge.backends._deps.require`` now turns that
into a clear ImportError naming the missing distribution + the
``pip install 'voxedge[<extra>]'`` command.

These tests force the import to fail (via ``sys.modules`` sentinel /
``builtins.__import__`` patch) and assert the message content. They never
require the real heavy packages, so they run on any dev box.
"""

from __future__ import annotations

import builtins

import pytest

from voxedge.backends import _deps


def test_require_passes_through_present_module():
    # a stdlib module that always exists
    mod = _deps.require("json", extra="whatever")
    import json

    assert mod is json


def test_require_missing_names_package_and_extra():
    with pytest.raises(ImportError) as ei:
        _deps.require("definitely_not_a_real_module_xyz", extra="sherpa")
    msg = str(ei.value)
    assert "voxedge[sherpa]" in msg
    assert "definitely_not_a_real_module_xyz" in msg
    assert "pip install" in msg


def test_require_uses_package_alias_in_message():
    with pytest.raises(ImportError) as ei:
        _deps.require("cuda", extra="jetson", package="cuda-python")
    msg = str(ei.value)
    # the pip distribution name (not just the import name) is surfaced
    assert "cuda-python" in msg
    assert "voxedge[jetson]" in msg


def _patch_import_fail(monkeypatch, missing: str):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing or name.startswith(missing + "."):
            raise ImportError(f"No module named {missing!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_check_sherpa_deps_friendly_when_missing(monkeypatch):
    _patch_import_fail(monkeypatch, "sherpa_onnx")
    with pytest.raises(ImportError) as ei:
        _deps.check_sherpa_deps()
    msg = str(ei.value)
    assert "sherpa-onnx" in msg
    assert "voxedge[sherpa]" in msg


def test_check_rk_deps_friendly_when_missing(monkeypatch):
    _patch_import_fail(monkeypatch, "rkvoice_stream")
    with pytest.raises(ImportError) as ei:
        _deps.check_rk_deps()
    msg = str(ei.value)
    assert "rkvoice-stream" in msg
    assert "voxedge[rk]" in msg


def test_sherpa_asr_backend_preload_friendly_error(monkeypatch):
    """End-to-end: SherpaASRBackend.preload() raises the friendly error."""
    from voxedge.backends.sherpa.asr import SherpaASRBackend, SherpaASRConfig

    _patch_import_fail(monkeypatch, "sherpa_onnx")
    backend = SherpaASRBackend(SherpaASRConfig())
    with pytest.raises(ImportError) as ei:
        backend.preload()
    assert "voxedge[sherpa]" in str(ei.value)


def test_sherpa_tts_backend_preload_friendly_error(monkeypatch):
    from voxedge.backends.sherpa.tts import SherpaTTSBackend, SherpaTTSConfig

    _patch_import_fail(monkeypatch, "sherpa_onnx")
    backend = SherpaTTSBackend(SherpaTTSConfig())
    with pytest.raises(ImportError) as ei:
        backend.preload()
    assert "voxedge[sherpa]" in str(ei.value)


def test_rk_asr_backend_preload_friendly_error(monkeypatch):
    # Lazy lifecycle (consistent with sherpa/jetson above): construction is
    # cheap and does NOT require the rk runtime; the friendly error surfaces at
    # preload() when NPU init is actually attempted.
    from voxedge.backends.rk.asr import RKASRBackend, RKASRConfig

    _patch_import_fail(monkeypatch, "rkvoice_stream")
    backend = RKASRBackend(RKASRConfig())  # cheap; no rkvoice_stream needed
    with pytest.raises(ImportError) as ei:
        backend.preload()
    assert "voxedge[rk]" in str(ei.value)


def test_rk_asr_stream_adapter_forwards_stream_flags_and_options():
    from voxedge.backends.rk.asr import RKASRBackend, RKASRConfig

    class InnerStream:
        immediate_client_eos_cancel_safe = True
        prefer_backend_endpoint_vad = True

    class InnerBackend:
        def __init__(self):
            self.seen = None

        def create_stream(self, language="auto", stream_options=None):
            self.seen = (language, dict(stream_options or {}))
            return InnerStream()

    backend = RKASRBackend(RKASRConfig())
    backend._inner = InnerBackend()

    stream = backend.create_stream(
        language="Chinese",
        stream_options={"vad_endpoint_silence_ms": 800},
    )

    assert backend._inner.seen == (
        "Chinese",
        {"vad_endpoint_silence_ms": 800},
    )
    assert stream.immediate_client_eos_cancel_safe is True
    assert stream.prefer_backend_endpoint_vad is True


def test_rk_tts_backend_preload_friendly_error(monkeypatch):
    from voxedge.backends.rk.tts import RKTTSBackend, RKTTSConfig

    _patch_import_fail(monkeypatch, "rkvoice_stream")
    backend = RKTTSBackend(RKTTSConfig())  # cheap; no rkvoice_stream needed
    with pytest.raises(ImportError) as ei:
        backend.preload()
    assert "voxedge[rk]" in str(ei.value)


def test_paraformer_preload_friendly_error_when_trt_missing(monkeypatch):
    from voxedge.backends.jetson.paraformer_trt import (
        ParaformerTRTBackend,
        ParaformerTRTConfig,
    )

    _patch_import_fail(monkeypatch, "tensorrt")
    backend = ParaformerTRTBackend(ParaformerTRTConfig())
    with pytest.raises(ImportError) as ei:
        backend.preload()
    assert "voxedge[jetson]" in str(ei.value)
    assert "tensorrt" in str(ei.value)
