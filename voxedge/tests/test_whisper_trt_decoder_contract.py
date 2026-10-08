"""Contract tests for the TensorRT Whisper decoder without CUDA/TensorRT."""

import numpy as np
import pytest

from voxedge.backends.whisper.decoder import EOT, TensorRTKVDecoder


class _Cuda:
    class cudaError_t:
        cudaSuccess = 0

    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2
        cudaMemcpyDeviceToDevice = 3

    def __init__(self):
        self.next = 100
        self.mem = {}
        self.frees = []
        self.syncs = 0
        self.h2d_copies = []

    def cudaMalloc(self, n):
        p = self.next
        self.next += 1
        self.mem[p] = bytearray(n)
        return 0, p

    def cudaFree(self, p):
        self.frees.append(p)
        self.mem.pop(p, None)
        return 0

    def cudaMemcpyAsync(self, dst, src, n, kind, stream):
        if kind == self.cudaMemcpyKind.cudaMemcpyHostToDevice:
            import ctypes
            ctypes.memmove((ctypes.c_char * n).from_buffer(self.mem[dst]), src, n)
            self.h2d_copies.append((dst, n, bytes(self.mem[dst][:n])))
        if kind == self.cudaMemcpyKind.cudaMemcpyDeviceToHost:
            import ctypes
            ctypes.memmove(dst, (ctypes.c_char * n).from_buffer(self.mem[src]), n)
        return 0

    def cudaStreamSynchronize(self, stream):
        self.syncs += 1
        return 0


class _Ctx:
    def __init__(self, shapes):
        self.shapes = shapes
        self.bound = {}
        self.output_queries = []
        self.required_inputs = set()
        self.set_inputs = set()
        self.output_query_ready = []

    def set_input_shape(self, name, shape):
        self.shapes[name] = tuple(shape)
        self.set_inputs.add(name)
        return True

    def get_tensor_shape(self, name):
        self.output_queries.append(name)
        if name.startswith("present.") and self.required_inputs:
            self.output_query_ready.append(self.required_inputs <= self.set_inputs)
        return self.shapes[name]

    def set_tensor_address(self, name, ptr):
        self.bound[name] = ptr
        return True


class _Engine:
    def __init__(self, shapes, inputs, profiles):
        self.engine = type("E", (), {"get_tensor_shape": lambda s, n: s.shapes[n]})()
        self.engine.shapes = shapes
        self.names = list(shapes)
        self.is_input = {n: n in inputs for n in shapes}
        self.dtype = {n: (np.int64 if n == "input_ids" else np.float32) for n in shapes}
        self.ctx = _Ctx(dict(shapes))
        self.ptr = {}
        self.shape = {}
        self._profiles = profiles
        self._required_inputs = {
            n for n in inputs if n.startswith("past_key_values.") and ".decoder." in n
        }

    def profile_shape(self, name):
        return self._profiles[name]

    def set_shape(self, name, shape):
        if not self.ctx.set_input_shape(name, shape):
            raise RuntimeError("set shape")
        self.shape[name] = tuple(shape)
        if name == "encoder_hidden_states":
            self.shape["logits"] = (1, 4, 51865); self.ctx.shapes["logits"] = self.shape["logits"]
            for n in self.names:
                if n.startswith("present.0.decoder."):
                    self.shape[n] = (1, 8, 4, 64); self.ctx.shapes[n] = self.shape[n]
                if n.startswith("present.0.encoder."):
                    self.shape[n] = (1, 8, shape[1], 64); self.ctx.shapes[n] = self.shape[n]
        if (
            name.startswith("past_key_values.0.decoder.")
            and self._required_inputs <= (self.shape.keys() | {name})
        ):
            for kind in ("key", "value"):
                out = "present.0.decoder." + kind
                self.ctx.shapes[out] = (shape[0], shape[1], shape[2] + 1, shape[3])

    def bind(self, name, ptr):
        if not self.ctx.set_tensor_address(name, ptr):
            raise RuntimeError("bind")
        self.ptr[name] = ptr

    def allocate_outputs(self, malloc, only=None):
        for n in self.names:
            if self.is_input[n] or (only is not None and n not in only):
                continue
            shape = self.ctx.get_tensor_shape(n)
            self.shape[n] = tuple(shape)
            self.ptr[n] = malloc(int(np.prod(shape)) * np.dtype(self.dtype[n]).itemsize)

    def run(self, stream):
        if "encoder_hidden_states" in self.names:
            self.shape["logits"] = (1, 4, 51865)
            self.ctx.shapes["logits"] = self.shape["logits"]
            for n in self.names:
                if n.startswith("present.0.decoder."):
                    self.shape[n] = (1, 8, 4, 64); self.ctx.shapes[n] = self.shape[n]
                if n.startswith("present.0.encoder."):
                    self.shape[n] = (1, 8, 1500, 64); self.ctx.shapes[n] = self.shape[n]
        return None

    def close(self):
        return None


def _decoder():
    names = {
        "input_ids": (1, -1), "encoder_hidden_states": (1, -1, 512),
        "logits": (1, -1, 51865),
        "present.0.decoder.key": (1, 8, -1, 64), "present.0.decoder.value": (1, 8, -1, 64),
        "present.0.encoder.key": (1, 8, -1, 64), "present.0.encoder.value": (1, 8, -1, 64),
    }
    step = {
        "input_ids": (1, 1), "logits": (1, 1, 51865),
        "past_key_values.0.decoder.key": (1, 8, -1, 64), "past_key_values.0.decoder.value": (1, 8, -1, 64),
        "past_key_values.0.encoder.key": (1, 8, -1, 64), "past_key_values.0.encoder.value": (1, 8, -1, 64),
        "present.0.decoder.key": (1, 8, -1, 64), "present.0.decoder.value": (1, 8, -1, 64),
    }
    prof = {n: ((1, 8, 4 if "decoder" in n else 1500, 64), (1, 8, 128 if "decoder" in n else 1500, 64), (1, 8, 1500 if "encoder" in n else 447, 64))
            for n in step if "past_key_values" in n}
    prof["encoder_hidden_states"] = ((1, 1, 512), (1, 750, 512), (1, 1500, 512))
    d = TensorRTKVDecoder.__new__(TensorRTKVDecoder)
    d._cudart = _Cuda(); d._stream = 1; d._owned = []; d._staged = {}; d._staged_ptr = {}
    d._kv = {}; d._kv_bytes = 0; d._max_positions = 448; d._profile_max = 447
    d._failed = False; d._closed = False; d._layers = [0]; d._hidden_dtype = np.dtype(np.float32)
    d._prefill = _Engine(names, {"input_ids", "encoder_hidden_states"}, prof)
    d._step = _Engine(step, {n for n in step if n not in {"logits", "present.0.decoder.key", "present.0.decoder.value"}}, prof)
    d._step.ctx.required_inputs = d._step._required_inputs
    return d


def test_set_shape_and_bind_false_fail_closed():
    d = _decoder()
    d._prefill.ctx.set_input_shape = lambda *a: False
    with pytest.raises(RuntimeError):
        d._prefill.set_shape("input_ids", (1, 4))


def test_capacity_includes_present_token_and_close_is_idempotent():
    d = _decoder()
    d._ensure_kv_buffers(1 * 8 * 448 * 64 * 4)
    assert all(len(buf) == 1 * 8 * 448 * 64 * 4 for buf in d._cudart.mem.values())
    d.close(); d.close()
    assert len(d._cudart.frees) == 4


def test_copy_h2d_reuses_capacity_after_narrow_transfer():
    d = _decoder()
    wide = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    narrow = np.asarray([[9]], dtype=np.int64)
    ptrs = []
    for arr in (wide, narrow, wide, narrow, wide, narrow):
        ptrs.append(d._copy_h2d("input_ids", arr, d._step))

    assert ptrs == [ptrs[0]] * len(ptrs)
    assert d._staged_ptr["input_ids"] == (ptrs[0], wide.nbytes)
    assert len(d._owned) == 1
    assert [n for _, n, _ in d._cudart.h2d_copies] == [32, 8, 32, 8, 32, 8]
    assert [payload for _, _, payload in d._cudart.h2d_copies] == [
        wide.tobytes(), narrow.tobytes(), wide.tobytes(),
        narrow.tobytes(), wide.tobytes(), narrow.tobytes(),
    ]
    assert d._step.ptr["input_ids"] == ptrs[0]
    d.close()
    assert len(d._cudart.frees) == 1


def test_copy_h2d_growth_keeps_old_staging_owned_until_close():
    d = _decoder()
    narrow = np.asarray([[9]], dtype=np.int64)
    wide = np.asarray([[1, 2, 3, 4]], dtype=np.int64)
    ptrs = [
        d._copy_h2d("input_ids", arr, d._step)
        for arr in (narrow, wide, narrow, wide)
    ]

    assert ptrs[0] != ptrs[1]
    assert ptrs[1:] == [ptrs[1]] * 3
    assert len(d._owned) == 2
    assert d._staged_ptr["input_ids"] == (ptrs[1], wide.nbytes)
    d.close()
    assert sorted(d._cudart.frees) == sorted(set(ptrs))
    assert len(d._cudart.frees) == 2


def test_decode_runs_prefill_and_step_with_borrowed_hidden():
    d = _decoder()
    # The fake device returns zero logits, which selects token 0 and exercises
    # one complete prefill -> KV copy -> step -> logits read cycle.
    text, timings = d.decode((77, (1, 1500, 512), np.float32, object()), {}, "en", audio_s=1.0, max_new=1)
    assert text == ""
    assert len(timings) == 2
    assert d._cudart.syncs >= 4


def test_step_waits_for_all_dynamic_inputs_before_resolving_outputs():
    d = _decoder()
    text, timings = d.decode(
        (77, (1, 1500, 512), np.float32, object()), {}, "en", audio_s=1.0, max_new=2
    )
    assert text == ""
    assert len(timings) == 3
    assert d._step.ctx.output_queries
    assert all(d._step.ctx.output_query_ready)
    assert all(
        d._step.shape[name] == (1, 8, 5, 64)
        for name in ("past_key_values.0.decoder.key", "past_key_values.0.decoder.value")
    )
    assert d._failed is False
    d.close()
    assert d._owned == []


def _run_fake_step_at(d, past_len):
    """Exercise the same shape, bind, output allocation and execute path as decode."""
    ids = np.asarray([[0]], dtype=np.int64)
    d._step.set_shape("input_ids", ids.shape)
    kv_bytes = 1 * 8 * 448 * 64 * 4
    kv = d._ensure_kv_buffers(kv_bytes)
    for layer in d._layers:
        for kind in ("key", "value"):
            name = f"past_key_values.{layer}.decoder.{kind}"
            out = f"present.{layer}.decoder.{kind}"
            d._step.set_shape(name, (1, 8, past_len, 64))
    for layer in d._layers:
        for kind in ("key", "value"):
            name = f"past_key_values.{layer}.decoder.{kind}"
            out = f"present.{layer}.decoder.{kind}"
            assert d._step.ctx.get_tensor_shape(out) == (1, 8, past_len + 1, 64)
            past, present = kv[(layer, kind)]
            d._step.bind(name, kv[(layer, kind)][0])
            d._step.bind(out, kv[(layer, kind)][1])
    d._copy_h2d("input_ids", ids, d._step)
    d._step.allocate_outputs(d._malloc, only=("logits",))
    d._step.run(d._stream)
    return d._step.shape["logits"]


def test_fake_step_executes_at_profile_edges_and_rejects_448():
    d = _decoder()
    d._validate_io()
    assert _run_fake_step_at(d, 64) == (1, 1, 51865)
    assert _run_fake_step_at(d, 447) == (1, 1, 51865)
    with pytest.raises(RuntimeError, match="overflow"):
        d._check_past_length(448)
    d.close()
    assert d._owned == []


def test_same_decoder_reuses_kv_buffers_across_three_decodes_then_closes():
    d = _decoder()
    kv_ids = None
    for _ in range(3):
        d.decode((77, (1, 1500, 512), np.float32, object()), {}, "en", audio_s=1.0, max_new=1)
        current = tuple(frozenset(pair) for pair in d._kv.values())
        if kv_ids is None:
            kv_ids = current
        else:
            assert current == kv_ids
    d.close()
    assert d._owned == []
    assert d._kv == {}
    assert d._kv_bytes == 0
    assert d._prefill is None and d._step is None
    assert d._stream is None
    assert d._runtime is None and d._logger is None


def test_validate_io_accepts_fixed_prefill_and_dynamic_step_geometry():
    d = _decoder()
    d._validate_io()
    assert d._self_geometry == (1, 8, 64)
    assert d._cross_geometry == (1, 8, 64)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("past_key_values.0.decoder.key", (1, 7, -1, 64)),
        ("past_key_values.0.decoder.value", (1, 8, -1, 32)),
        ("past_key_values.0.decoder.key", (1, 8, -1)),
        ("past_key_values.0.decoder.key", (2, 8, -1, 64)),
    ],
)
def test_validate_io_rejects_unsafe_nonsequence_geometry(field, replacement):
    d = _decoder()
    d._step.engine.shapes[field] = replacement
    with pytest.raises(RuntimeError):
        d._validate_io()


def test_validate_io_rejects_cross_source_profile_not_fixed_at_1500():
    d = _decoder()
    profile = d._step._profiles["past_key_values.0.encoder.key"]
    d._step._profiles["past_key_values.0.encoder.key"] = (
        profile[0], profile[1], (1, 8, 1499, 64)
    )
    with pytest.raises(RuntimeError):
        d._validate_io()


def test_validate_io_rejects_kv_dtype_mismatch():
    d = _decoder()
    d._step.dtype["past_key_values.0.decoder.key"] = np.float16
    with pytest.raises(RuntimeError):
        d._validate_io()
