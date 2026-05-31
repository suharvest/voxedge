"""Env-free arena helper + arena-backed CudaMemoryPool (migration gap).

Background: production ``app/backends/jetson/matcha_trt.py`` exposed
``_read_arena_size_bytes`` (read ``OVS_*_ARENA_SIZE_MB`` from os.environ) plus a
``CudaMemoryPool`` bump-allocator. The env-free voxedge migration split these:

  * env → ``arena_size_mb`` now happens in the product config-builder
    (covered: ``app/tests/test_voxedge_backend_config.py`` — ``OVS_MATCHA_ARENA_SIZE_MB``
    → ``cfg.arena_size_mb``);
  * the pure MB→bytes conversion is ``voxedge...._util.arena_size_bytes``;
  * the bump-pointer sub-allocator is ``voxedge...._util.CudaMemoryPool``.

The conversion + pool mechanics had no voxedge coverage after the move. These
tests lock both, entirely with a mocked ``cuda.cudart`` so they run on any host
(Mac dev, CI, Jetson) — no CUDA required.
"""

from __future__ import annotations

import sys
import types

import pytest

from voxedge.backends.jetson._util import CudaMemoryPool, arena_size_bytes


# ── arena_size_bytes (env-free MB→bytes) ──────────────────────────────────────


def test_arena_size_bytes_explicit_mb():
    assert arena_size_bytes(8) == 8 * 1024 * 1024


def test_arena_size_bytes_none_uses_default():
    assert arena_size_bytes(None) == 16 * 1024 * 1024
    assert arena_size_bytes(None, default_mb=24) == 24 * 1024 * 1024


def test_arena_size_bytes_clamps_sub_one_mb():
    assert arena_size_bytes(0) == 1 * 1024 * 1024
    assert arena_size_bytes(-5) == 1 * 1024 * 1024


# ── CudaMemoryPool arena mechanics (mocked cudart) ────────────────────────────


@pytest.fixture
def fake_cudart(monkeypatch):
    """Install a stub ``cuda.cudart`` that tracks malloc/free + stream calls."""

    class _Enum:
        cudaSuccess = 0

    class _Kind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    state = {
        "next_ptr": 0x10000,
        "mallocs": [],  # list[(ptr, size)]
        "frees": [],
        "stream_creates": 0,
        "stream_destroys": [],
    }

    def cudaMalloc(size):
        ptr = state["next_ptr"]
        state["next_ptr"] += max(size, 1)
        state["mallocs"].append((ptr, int(size)))
        return _Enum.cudaSuccess, ptr

    def cudaFree(ptr):
        state["frees"].append(int(ptr))
        return _Enum.cudaSuccess

    def cudaStreamCreate():
        state["stream_creates"] += 1
        return _Enum.cudaSuccess, 0xABCD0000

    def cudaStreamDestroy(handle):
        state["stream_destroys"].append(int(handle))
        return _Enum.cudaSuccess

    def cudaStreamSynchronize(_handle):
        return _Enum.cudaSuccess

    fake = types.SimpleNamespace(
        cudaMalloc=cudaMalloc,
        cudaFree=cudaFree,
        cudaStreamCreate=cudaStreamCreate,
        cudaStreamDestroy=cudaStreamDestroy,
        cudaStreamSynchronize=cudaStreamSynchronize,
        cudaError_t=_Enum,
        cudaMemcpyKind=_Kind,
    )
    mod = types.ModuleType("cuda")
    mod.cudart = fake  # type: ignore[attr-defined]
    cudart_mod = types.ModuleType("cuda.cudart")
    for attr in (
        "cudaMalloc",
        "cudaFree",
        "cudaStreamCreate",
        "cudaStreamDestroy",
        "cudaStreamSynchronize",
        "cudaError_t",
        "cudaMemcpyKind",
    ):
        setattr(cudart_mod, attr, getattr(fake, attr))
    monkeypatch.setitem(sys.modules, "cuda", mod)
    monkeypatch.setitem(sys.modules, "cuda.cudart", cudart_mod)
    return state


def test_arena_single_cudaMalloc(fake_cudart):
    """Arena should call cudaMalloc once regardless of sub-allocation count."""
    pool = CudaMemoryPool(arena_size_bytes=64 * 1024)
    p1 = pool.allocate(1000)
    p2 = pool.allocate(2000)
    p3 = pool.allocate(500)
    assert len(fake_cudart["mallocs"]) == 1
    assert fake_cudart["mallocs"][0][1] == 64 * 1024
    assert p1 != p2 != p3
    arena_base = fake_cudart["mallocs"][0][0]
    for p in (p1, p2, p3):
        assert (p - arena_base) % 256 == 0


def test_arena_reuse_across_free_all_cycles(fake_cudart):
    """free_all() resets the bump offset but does NOT free the arena."""
    pool = CudaMemoryPool(arena_size_bytes=64 * 1024)
    p1 = pool.allocate(4096)
    pool.free_all()
    p2 = pool.allocate(4096)
    assert len(fake_cudart["mallocs"]) == 1
    assert fake_cudart["frees"] == []
    assert p1 == p2


def test_arena_overflow_falls_back_to_cuda_malloc(fake_cudart):
    """Allocations that don't fit the arena route to per-call cudaMalloc."""
    pool = CudaMemoryPool(arena_size_bytes=4 * 1024)
    p_small = pool.allocate(1024)
    p_big = pool.allocate(16 * 1024)  # > arena → overflow
    assert len(fake_cudart["mallocs"]) == 2
    assert fake_cudart["mallocs"][1][1] == 16 * 1024
    assert pool._overflow_count == 1
    assert pool._overflow_bytes == 16 * 1024
    assert p_small != p_big
    pool.free_all()
    assert fake_cudart["frees"] == [p_big]
    assert pool._overflow_allocs == []


def test_destroy_frees_arena_plus_overflow_plus_stream(fake_cudart):
    pool = CudaMemoryPool(arena_size_bytes=4 * 1024)
    pool.allocate(1024)               # arena
    p_overflow = pool.allocate(8192)  # overflow
    arena_base = fake_cudart["mallocs"][0][0]
    pool.destroy()
    assert p_overflow in fake_cudart["frees"]
    assert arena_base in fake_cudart["frees"]
    assert fake_cudart["stream_destroys"] == [0xABCD0000]
    assert pool._arena_ptr is None
    assert pool._stream is None
    assert pool._initialized is False
    pool.destroy()  # idempotent


def test_peak_telemetry_updates(fake_cudart):
    pool = CudaMemoryPool(arena_size_bytes=1024 * 1024)
    pool.allocate(1000)
    pool.allocate(2000)
    peak_after_2 = pool._peak_offset
    pool.free_all()
    assert pool._peak_offset == peak_after_2  # high-water mark not reset
    pool.allocate(500)
    assert pool._peak_offset == peak_after_2  # smaller alloc doesn't lower peak


def test_legacy_mode_without_arena(fake_cudart):
    """arena_size_bytes=None preserves the original cudaMalloc-per-call path."""
    pool = CudaMemoryPool()  # no arena
    p1 = pool.allocate(1024)
    p2 = pool.allocate(2048)
    assert len(fake_cudart["mallocs"]) == 2
    assert p1 in pool._allocations and p2 in pool._allocations
    pool.free_all()
    assert sorted(fake_cudart["frees"]) == sorted([p1, p2])
    assert pool._allocations == []
