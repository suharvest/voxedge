"""SenseVoice TRT: resident buffers + valid-frame D2H + admission-only capability.

Two device measurements (orin-nano) shaped the code this file locks down:

  * An execution-context pool was rejected — GR3D is 98% at 1 stream,
    ``--streams=2/4`` bought 1.11x/1.13x for +216/+302 MB, throughput is
    enqueue-bound (CPU 36.98 ms vs GPU 37.02 ms).
  * A pinned host buffer was rejected — pinned D2H is faster (1.24 vs 4.77 ms)
    but the copy out of the shared block costs 7.71 ms, which is host bandwidth
    (~4.5 GB/s), not a pinned artefact: a pageable numpy->numpy copy of the same
    34.5 MB measured 7.70 ms. Net 0.7 ms slower, plus 32.9 MB page-locked.

What survives, and what these tests assert:

  * allocations are made once (2x cudaMalloc + 1x cudaStreamCreate) and reused —
    zero malloc/free/stream churn per request (measured 5.36 ms of the old path)
  * the D2H moves only the ``valid`` leading frames, not all ``T_FIXED``; on a
    3 s clip that is 54 of 344 rows, 84% of the transfer skipped
  * ``max_concurrent`` is an admission ceiling: ``supports_parallel`` stays False
  * ``unload()`` releases exactly what was taken, and is idempotent

No GPU here: a fake ``cuda.cudart`` is injected into ``sys.modules``. It records
every call, asserts on double free / unknown pointer, and services D2H copies out
of a caller-registered "device" array so the valid-prefix semantics are exercised
for real rather than merely counted.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys
import textwrap
import threading
import types

import numpy as np
import pytest

from voxedge.backends.jetson.sensevoice_trt import (
    BLANK_ID,
    LFR_DIM,
    T_FIXED,
    SenseVoiceTRTBackend,
    SenseVoiceTRTConfig,
)

# Small stand-in vocabulary keeps the fake logits readable; the frame count is
# the real T_FIXED so the valid-prefix arithmetic is the production arithmetic.
V = 8
OUT_SHAPE = (1, T_FIXED, V)
ROW_BYTES = V * 4
PIECES = "abcdefgh"  # id -> piece, id 0 is BLANK_ID


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCudart:
    """Minimal cudart stand-in: records calls, catches double frees, moves data."""

    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    def __init__(self):
        self.calls: list[tuple] = []
        self.malloc_n = 0
        self.free_n = 0
        self.stream_create_n = 0
        self.stream_destroy_n = 0
        self.live_ptrs: set[int] = set()
        self.live_streams: set[int] = set()
        # ptr -> numpy array standing in for device memory contents (D2H source)
        self.device_data: dict[int, np.ndarray] = {}
        self._next = 0x1000

    def _ptr(self) -> int:
        self._next += 0x1000
        return self._next

    def cudaMalloc(self, nbytes):
        self.malloc_n += 1
        ptr = self._ptr()
        self.live_ptrs.add(ptr)
        self.calls.append(("malloc", nbytes, ptr))
        return (0, ptr)

    def cudaFree(self, ptr):
        self.free_n += 1
        assert ptr in self.live_ptrs, f"double free / unknown device ptr {ptr}"
        self.live_ptrs.discard(ptr)
        self.device_data.pop(ptr, None)
        self.calls.append(("free", ptr))
        return (0,)

    def cudaStreamCreate(self):
        self.stream_create_n += 1
        s = self._ptr()
        self.live_streams.add(s)
        self.calls.append(("stream_create", s))
        return (0, s)

    def cudaStreamDestroy(self, stream):
        self.stream_destroy_n += 1
        assert stream in self.live_streams, f"double destroy stream {stream}"
        self.live_streams.discard(stream)
        self.calls.append(("stream_destroy", stream))
        return (0,)

    def cudaStreamSynchronize(self, stream):
        self.calls.append(("stream_sync", stream))
        return (0,)

    def cudaDeviceSynchronize(self):
        return (0,)

    def cudaMemcpy(self, dst, src, nbytes, kind):
        self.calls.append(("memcpy", int(dst), int(src), int(nbytes), kind))
        if kind == self.cudaMemcpyKind.cudaMemcpyDeviceToHost:
            source = self.device_data.get(int(src))
            if source is not None:
                assert nbytes <= source.nbytes, "D2H would read past device buffer"
                # Copy exactly the requested prefix — anything the backend did
                # not ask for must NOT reach the host array.
                ctypes.memmove(int(dst), source.ctypes.data, int(nbytes))
        return (0,)

    # helpers for the tests -------------------------------------------------

    def memcpys(self, kind=None):
        return [c for c in self.calls if c[0] == "memcpy" and (kind is None or c[4] == kind)]

    def d2h(self):
        return self.memcpys(self.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    def h2d(self):
        return self.memcpys(self.cudaMemcpyKind.cudaMemcpyHostToDevice)


class FakeContext:
    def __init__(self):
        self.input_shapes: dict = {}
        self.addresses: dict = {}
        self.exec_calls: list = []
        self.ok = True

    def set_input_shape(self, name, shape):
        self.input_shapes[name] = tuple(shape)

    def get_tensor_shape(self, name):
        return OUT_SHAPE

    def set_tensor_address(self, name, ptr):
        self.addresses[name] = int(ptr)

    def execute_async_v3(self, stream):
        self.exec_calls.append(stream)
        return self.ok


class FakeSentencePiece:
    def get_piece_size(self):
        return V

    def id_to_piece(self, i):
        return PIECES[i]


@pytest.fixture()
def cudart(monkeypatch):
    """Inject a fake ``cuda`` module so the method-local imports resolve."""
    fake = FakeCudart()
    mod = types.ModuleType("cuda")
    mod.cudart = fake
    monkeypatch.setitem(sys.modules, "cuda", mod)
    return fake


def _backend(cudart_, max_concurrent: int = 1) -> SenseVoiceTRTBackend:
    """Backend past preload()'s engine phase: context + resident buffers."""
    b = SenseVoiceTRTBackend(SenseVoiceTRTConfig(max_concurrent=max_concurrent))
    b._engine = object()
    b._ctx = FakeContext()
    b._in_name = "speech"
    b._out_name = "logits"
    b._out_shape = OUT_SHAPE
    b._alloc_buffers()
    b._ready = True
    return b


def _speech():
    return np.zeros((1, T_FIXED, LFR_DIM), dtype=np.float32)


def _device_logits(cudart_, backend, ids):
    """Publish (T_FIXED, V) logits as the engine output; row i argmaxes to ids[i]."""
    logits = np.zeros((T_FIXED, V), dtype=np.float32)
    logits[np.arange(T_FIXED), np.asarray(ids)] = 1.0
    cudart_.device_data[backend._d_out] = np.ascontiguousarray(logits)
    return logits


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_default_max_concurrent_is_one():
    assert SenseVoiceTRTConfig().max_concurrent == 1


@pytest.mark.parametrize("raw,expected", [(0, 1), (-5, 1), (1, 1), (2, 2), (4, 4), ("3", 3)])
def test_max_concurrent_clamped(raw, expected):
    assert SenseVoiceTRTConfig(max_concurrent=raw).max_concurrent == expected


def test_bpe_default_still_derived_from_model_dir():
    cfg = SenseVoiceTRTConfig(model_dir="/m")
    assert cfg.bpe_model == "/m/chn_jpn_yue_eng_ko_spectok.bpe.model"


# ---------------------------------------------------------------------------
# capability: admission ceiling, execution still serialized
# ---------------------------------------------------------------------------


def test_capability_default_matches_pre_change_behaviour():
    from voxedge.engine.concurrency_capability import ConcurrencyCapability

    cap = SenseVoiceTRTBackend(SenseVoiceTRTConfig()).concurrency_capability()
    assert cap == ConcurrencyCapability.default()


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_capability_reports_admission_ceiling_but_never_parallel(n):
    cap = SenseVoiceTRTBackend(SenseVoiceTRTConfig(max_concurrent=n)).concurrency_capability()
    assert cap.max_concurrent == n
    # DELIBERATE pairing: admit N, execute serialized on the single context.
    assert cap.supports_parallel is False


def test_capability_keeps_conservative_defaults_for_other_fields():
    cap = SenseVoiceTRTBackend(SenseVoiceTRTConfig(max_concurrent=4)).concurrency_capability()
    assert cap.is_stateful is True
    assert cap.requires_exclusive_device is True
    assert cap.scaling_mode == "single_runtime_multiplex"


def test_capability_accepts_profile_arg():
    b = SenseVoiceTRTBackend(SenseVoiceTRTConfig(max_concurrent=2))
    assert b.concurrency_capability(None).max_concurrent == 2


def test_capability_resolver_reads_it():
    from voxedge.engine.capability_resolver import capability_of

    b = SenseVoiceTRTBackend(SenseVoiceTRTConfig(max_concurrent=4))
    cap = capability_of(b)
    assert (cap.max_concurrent, cap.supports_parallel) == (4, False)


# ---------------------------------------------------------------------------
# allocation happens once — device only, no pinned host block
# ---------------------------------------------------------------------------


def test_alloc_buffers_allocates_once(cudart):
    b = _backend(cudart)
    assert cudart.malloc_n == 2          # d_in + d_out, nothing else
    assert cudart.stream_create_n == 1
    assert cudart.free_n == 0 and cudart.stream_destroy_n == 0
    sizes = [c[1] for c in cudart.calls if c[0] == "malloc"]
    assert sizes == [T_FIXED * LFR_DIM * 4, T_FIXED * V * 4]
    # addresses bound once, at allocation time
    assert b._ctx.addresses == {"speech": b._d_in, "logits": b._d_out}


def test_backend_holds_no_pinned_host_buffer(cudart):
    """The pinned variant lost 0.7 ms net — it must not creep back."""
    b = _backend(cudart)
    assert not hasattr(b, "_h_out")
    assert not hasattr(b, "_h_out_view")
    assert not any(c[0].startswith("host_") for c in cudart.calls)
    # The fake deliberately does NOT implement cudaHostAlloc/cudaFreeHost, so a
    # reintroduced pinned path would fail here with AttributeError, not silently.
    assert not hasattr(cudart, "cudaHostAlloc")
    assert not hasattr(cudart, "cudaFreeHost")


def test_alloc_failure_releases_partial_allocations(cudart):
    class Boom(FakeCudart):
        def cudaStreamCreate(self):
            raise RuntimeError("boom")

    boom = Boom()
    sys.modules["cuda"].cudart = boom
    b = SenseVoiceTRTBackend(SenseVoiceTRTConfig())
    b._engine, b._ctx = object(), FakeContext()
    b._in_name, b._out_name, b._out_shape = "speech", "logits", OUT_SHAPE
    with pytest.raises(RuntimeError, match="boom"):
        b._alloc_buffers()
    assert boom.live_ptrs == set()
    assert (b._d_in, b._d_out, b._stream) == (0, 0, None)


def test_alloc_error_code_is_checked(cudart):
    class Failing(FakeCudart):
        def cudaMalloc(self, nbytes):
            return (2, 0)  # cudaErrorMemoryAllocation

    sys.modules["cuda"].cudart = Failing()
    b = SenseVoiceTRTBackend(SenseVoiceTRTConfig())
    b._engine, b._ctx = object(), FakeContext()
    b._in_name, b._out_name, b._out_shape = "speech", "logits", OUT_SHAPE
    with pytest.raises(RuntimeError, match="cudaMalloc"):
        b._alloc_buffers()


# ---------------------------------------------------------------------------
# _infer reuses the resident buffers
# ---------------------------------------------------------------------------


def test_infer_does_not_allocate_per_call(cudart):
    b = _backend(cudart)
    base = (cudart.malloc_n, cudart.stream_create_n)

    for _ in range(5):
        out = b._infer(_speech())
        assert out.shape == (T_FIXED, V)

    assert (cudart.malloc_n, cudart.stream_create_n) == base
    assert cudart.free_n == 0
    assert cudart.stream_destroy_n == 0
    # one H2D + one D2H per call, on the one resident stream
    assert [c[4] for c in cudart.memcpys()] == [1, 2] * 5
    assert b._ctx.exec_calls == [b._stream] * 5


def test_infer_h2d_and_d2h_target_resident_buffers(cudart):
    b = _backend(cudart)
    b._infer(_speech())
    h2d = cudart.h2d()[0]
    d2h = cudart.d2h()[0]
    assert h2d[1] == b._d_in                  # H2D dst = resident device input
    assert h2d[3] == T_FIXED * LFR_DIM * 4
    assert d2h[2] == b._d_out                 # D2H src = resident device output
    assert d2h[3] == T_FIXED * V * 4          # valid=None -> full transfer


# ---------------------------------------------------------------------------
# valid-frame D2H trimming (the core of this change)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("valid", [1, 54, 171, 344])
def test_d2h_moves_only_valid_rows(cudart, valid):
    b = _backend(cudart)
    out = b._infer(_speech(), valid)
    assert out.shape == (valid, V)
    d2h = cudart.d2h()[0]
    assert d2h[3] == valid * ROW_BYTES, "D2H byte count must follow valid, not T_FIXED"
    # H2D is unaffected — the whole padded input still has to go up
    assert cudart.h2d()[0][3] == T_FIXED * LFR_DIM * 4


def test_d2h_saving_on_a_realistic_clip(cudart):
    """3 s clip: 54 of 344 frames, i.e. 84% of the output transfer skipped."""
    b = _backend(cudart)
    b._infer(_speech(), 54)
    moved = cudart.d2h()[0][3]
    full = T_FIXED * ROW_BYTES
    assert moved == 54 * ROW_BYTES
    assert moved / full < 0.16


@pytest.mark.parametrize("valid,rows", [(None, T_FIXED), (T_FIXED + 1, T_FIXED), (10**6, T_FIXED)])
def test_valid_none_or_too_large_falls_back_to_t_fixed(cudart, valid, rows):
    b = _backend(cudart)
    out = b._infer(_speech(), valid)
    assert out.shape == (rows, V)
    assert cudart.d2h()[0][3] == rows * ROW_BYTES


@pytest.mark.parametrize("valid", [0, -1, -999])
def test_valid_below_one_is_clamped_to_one(cudart, valid):
    b = _backend(cudart)
    out = b._infer(_speech(), valid)
    assert out.shape == (1, V)
    assert cudart.d2h()[0][3] == ROW_BYTES


def test_returned_rows_are_the_leading_frames(cudart):
    """The rows handed back must be the engine's first `valid` rows, in order."""
    b = _backend(cudart)
    ids = [(i % (V - 1)) + 1 for i in range(T_FIXED)]  # never BLANK, all distinct-ish
    full = _device_logits(cudart, b, ids)
    out = b._infer(_speech(), 7)
    assert out.shape == (7, V)
    assert np.array_equal(out, full[:7])


def test_padding_rows_never_reach_the_caller(cudart):
    b = _backend(cudart)
    ids = [1] * 5 + [7] * (T_FIXED - 5)  # everything past row 5 is the pad marker
    _device_logits(cudart, b, ids)
    out = b._infer(_speech(), 5)
    assert out.argmax(-1).tolist() == [1] * 5
    assert 7 not in out.argmax(-1).tolist()


def test_consecutive_calls_return_independent_arrays(cudart):
    b = _backend(cudart)
    _device_logits(cudart, b, [1] * T_FIXED)
    first = b._infer(_speech(), 4)
    snapshot = first.copy()
    _device_logits(cudart, b, [5] * T_FIXED)
    second = b._infer(_speech(), 4)
    assert not np.shares_memory(first, second)
    assert np.array_equal(first, snapshot), "an earlier result was mutated"
    assert second.argmax(-1).tolist() == [5] * 4


# ---------------------------------------------------------------------------
# end-to-end decode through transcribe_array
# ---------------------------------------------------------------------------


def _decoding_backend(cudart_, ids, valid):
    b = _backend(cudart_)
    b._sp = FakeSentencePiece()
    speech = _speech()
    b._build_speech = lambda audio, lang="auto", textnorm="withitn": (speech, valid)
    _device_logits(cudart_, b, ids)
    return b


def test_transcribe_array_decodes_only_valid_frames(cudart):
    # rows 0..4 spell b,b,c,<blank>,d -> collapsed "bcd"; the pad rows all argmax
    # to piece "h", which must never appear.
    ids = [1, 1, 2, BLANK_ID, 3] + [7] * (T_FIXED - 5)
    b = _decoding_backend(cudart, ids, valid=5)
    res = b.transcribe_array(np.zeros(16000, dtype=np.float32))
    assert res.text == "bcd"
    assert "h" not in res.text
    assert cudart.d2h()[0][3] == 5 * ROW_BYTES


def test_transcribe_array_text_grows_with_valid(cudart):
    ids = [1, 2, 3, 4, 5] + [7] * (T_FIXED - 5)
    for valid, expected in [(1, "b"), (3, "bcd"), (5, "bcdef")]:
        b = _decoding_backend(cudart, ids, valid)
        assert b.transcribe_array(np.zeros(16000, dtype=np.float32)).text == expected


def test_transcribe_array_full_length_still_works(cudart):
    ids = [1] * T_FIXED
    b = _decoding_backend(cudart, ids, valid=T_FIXED)
    res = b.transcribe_array(np.zeros(16000, dtype=np.float32))
    assert res.text == "b"  # CTC collapse of 344 identical frames
    assert cudart.d2h()[0][3] == T_FIXED * ROW_BYTES


# ---------------------------------------------------------------------------
# failure / concurrency behaviour
# ---------------------------------------------------------------------------


def test_infer_rejects_wrong_shape_without_allocating(cudart):
    b = _backend(cudart)
    with pytest.raises(ValueError, match="expects speech shape"):
        b._infer(np.zeros((1, 8, LFR_DIM), dtype=np.float32))
    assert cudart.malloc_n == 2  # unchanged


def test_infer_execute_failure_returns_none_and_releases_lock(cudart):
    b = _backend(cudart)
    b._ctx.ok = False
    assert b._infer(_speech(), 54) is None
    assert not b._lock.locked()
    assert cudart.d2h() == [], "no D2H after a failed execute"
    assert cudart.free_n == 0 and b._d_in and b._d_out


def test_transcribe_array_survives_execute_failure(cudart):
    b = _decoding_backend(cudart, [1] * T_FIXED, valid=5)
    b._ctx.ok = False
    res = b.transcribe_array(np.zeros(16000, dtype=np.float32))
    assert res.text == ""


def test_infer_is_serialized(cudart):
    """One context: concurrent callers must never overlap inside the lock."""
    b = _backend(cudart)
    overlap: list[int] = []
    active: list[int] = []
    guard = threading.Lock()
    real_exec = b._ctx.execute_async_v3

    def watched(stream):
        with guard:
            active.append(1)
            overlap.append(len(active))
        try:
            return real_exec(stream)
        finally:
            with guard:
                active.pop()

    b._ctx.execute_async_v3 = watched
    ts = [threading.Thread(target=lambda: b._infer(_speech(), 54)) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
        assert not t.is_alive()
    assert max(overlap) == 1
    assert len(b._ctx.exec_calls) == 8


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------


def test_unload_frees_everything(cudart):
    b = _backend(cudart)
    b.unload()
    assert b.is_ready() is False
    assert cudart.free_n == 2 and cudart.stream_destroy_n == 1
    assert cudart.live_ptrs == set()
    assert cudart.live_streams == set()
    assert (b._d_in, b._d_out, b._stream) == (0, 0, None)
    assert b._ctx is None and b._engine is None


def test_unload_is_idempotent(cudart):
    b = _backend(cudart)
    b.unload()
    b.unload()
    b.unload()
    # FakeCudart asserts on double free, so these counts prove single release
    assert cudart.free_n == 2 and cudart.stream_destroy_n == 1


def test_unload_without_preload_is_safe(cudart):
    b = SenseVoiceTRTBackend(SenseVoiceTRTConfig())
    b.unload()
    assert b.is_ready() is False
    assert cudart.free_n == 0


def test_transcribe_rejects_after_unload(cudart):
    b = _backend(cudart)
    b.unload()
    with pytest.raises(RuntimeError, match="not ready"):
        b.transcribe_array(np.zeros(16000, dtype=np.float32))


def test_realloc_after_unload(cudart):
    b = _backend(cudart)
    b.unload()
    b._ctx = FakeContext()
    b._alloc_buffers()
    b._ready = True
    assert b.is_ready() is True
    assert b._infer(_speech(), 54).shape == (54, V)
    assert cudart.malloc_n == 4 and cudart.stream_create_n == 2  # 2 rounds, no leak
    assert len(cudart.live_ptrs) == 2 and len(cudart.live_streams) == 1


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------


def test_module_imports_without_cuda_or_tensorrt():
    """Import must work with no tensorrt/cuda present — the imports are lazy.

    Runs in a subprocess with ``tensorrt``/``cuda`` blocked at the finder level.
    The earlier version asserted ``"tensorrt" not in sys.modules``, which tests
    the ambient state of the whole test process rather than this module: on a
    real Jetson, or simply after any other test imported TensorRT, it failed
    even though the module is still perfectly lazy. Tests share sys.modules and
    ordering is not guaranteed, so the check has to be isolated.
    """
    import subprocess
    import sys as _sys

    script = textwrap.dedent(
        """
        import sys

        class _Blocker:
            def find_module(self, name, path=None):
                return self if name.split(".")[0] in ("tensorrt", "cuda") else None
            def load_module(self, name):
                raise ImportError(f"{name} blocked for this test")
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("tensorrt", "cuda"):
                    raise ImportError(f"{name} blocked for this test")
                return None

        sys.meta_path.insert(0, _Blocker())
        for mod in [m for m in sys.modules if m.split(".")[0] in ("tensorrt", "cuda")]:
            del sys.modules[mod]

        from voxedge.backends.jetson.sensevoice_trt import SenseVoiceTRTConfig
        assert SenseVoiceTRTConfig().max_concurrent == 1
        assert "tensorrt" not in sys.modules, "import must stay lazy"
        print("OK")
        """
    )
    proc = subprocess.run(
        [_sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout
