"""Regression: backends that do NOT override ``concurrency_capability`` must
resolve to a typed ``ConcurrencyCapability`` (serialized N=1), NOT a raw dict.

The base class used to return ``{"max_concurrency": 1, "mode": "serialized"}``,
so a backend that didn't override (e.g. moss_tts_nano) crashed the capability
resolver at startup with ``'dict' object has no attribute 'max_concurrent'``
— breaking every TTS-only profile on the device. See profiling sweep #39
(orin-nx + orin-nano both hit it).
"""

from voxedge.backends.base import ASRBackend, TTSBackend
from voxedge.engine.concurrency_capability import ConcurrencyCapability


class _MossLikeTTS(TTSBackend):
    """Concrete TTS backend that does NOT override concurrency_capability."""

    def __init__(self):
        pass

    @property
    def name(self):
        return "mosslike"

    @property
    def sample_rate(self):
        return 24000

    def capabilities(self):
        return set()

    def is_ready(self):
        return True

    def preload(self):
        pass

    async def synthesize(self, *a, **k):
        yield b""


class _BareASR(ASRBackend):
    """Concrete ASR backend that does NOT override concurrency_capability."""

    def __init__(self):
        pass

    @property
    def name(self):
        return "bareasr"

    @property
    def sample_rate(self):
        return 16000

    def capabilities(self):
        return set()

    def is_ready(self):
        return True

    def preload(self):
        pass

    def transcribe(self, *a, **k):
        raise NotImplementedError


def test_tts_base_default_is_concurrency_capability():
    # Resolver path: cls.__new__(cls).concurrency_capability() (no __init__).
    cap = _MossLikeTTS.__new__(_MossLikeTTS).concurrency_capability()
    assert isinstance(cap, ConcurrencyCapability)
    assert cap.max_concurrent == 1
    assert cap.supports_parallel is False
    # The crash was on attribute access — guard it explicitly.
    assert hasattr(cap, "max_concurrent")


def test_asr_base_default_is_concurrency_capability():
    cap = _BareASR.__new__(_BareASR).concurrency_capability()
    assert isinstance(cap, ConcurrencyCapability)
    assert cap.max_concurrent == 1
