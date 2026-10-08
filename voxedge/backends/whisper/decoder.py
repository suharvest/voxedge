"""Whisper decoder on CPU, ONNX with a real KV cache.

This is deliberately *not* on the accelerator. Neither vendor's NPU decoder has
a KV cache — Hailo compiles a fixed 32-token sequence, Rockchip a 12-slot
sliding window — so both recompute the whole sequence every autoregressive
step. Measured on the same audio, moving the decoder here made every board both
faster and more accurate (RK3588 English long-form 10.44% -> 7.58% WER while RTF
went 0.149 -> 0.061).
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import numpy as np

# Whisper special tokens.
EOT = 50257
SOT = 50258
TASK_TRANSCRIBE = 50359
NO_TIMESTAMPS = 50363
TIMESTAMP_BEGIN = 50364
LANG_TOKEN = {"en": 50259, "zh": 50260}

# The decoder's position table holds 448 entries. Going past it is not a quality
# degradation, it is an onnxruntime error ("idx=448 ... out of data bounds").
MAX_POSITIONS = 448


class OnnxKVDecoder:
    """optimum's two-graph export: decoder_model (prefill) + decoder_with_past.

    Prefill also emits the cross-attention K/V for the whole utterance, so those
    are computed once and reused for every subsequent step. ``encoder_sequence
    _length`` is a dynamic axis, which is why a 10 s or 20 s encoder feeds a
    decoder exported at 30 s without re-export.
    """

    def __init__(self, onnx_dir: str | Path, intra_op_threads: int = 0) -> None:
        import onnxruntime as ort

        d = Path(onnx_dir)
        opts = ort.SessionOptions()
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
        self._init = ort.InferenceSession(
            str(d / "decoder_model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._past = ort.InferenceSession(
            str(d / "decoder_with_past_model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._past_inputs = {i.name for i in self._past.get_inputs()}

    def decode(
        self,
        encoder_out: np.ndarray,
        vocab: dict[str, str],
        language: str,
        *,
        audio_s: float,
        max_new: Optional[int] = None,
    ) -> tuple[str, list[float]]:
        """Greedy decode. Returns (raw text, per-token wall times in ms)."""
        import time

        enc = np.ascontiguousarray(encoder_out.astype(np.float32))
        if enc.ndim == 2:
            enc = enc[None]

        forced = [SOT, LANG_TOKEN[language], TASK_TRANSCRIBE, NO_TIMESTAMPS]

        # Bound tokens by how much audio there actually is. A fixed cap only
        # guards the crash; it does not guard the runaway. Whisper skips EOS on
        # audio without enough content — a short utterance zero-padded to fill a
        # fixed window, or a near-silent tail chunk — and will happily generate
        # to whatever limit it is given.
        budget = int(max(16, min(220, audio_s * 8 + 12)))
        hard_cap = MAX_POSITIONS - len(forced) - 1
        if max_new is not None and max_new < 1:
            # range(-1) is empty, so the loop never runs and even a valid
            # prefill argmax is discarded — the utterance comes back "".
            raise ValueError(f"max_new must be >= 1, got {max_new}")
        cap = min(budget, hard_cap) if max_new is None else min(max_new, hard_cap)

        token_times: list[float] = []
        t0 = time.perf_counter()
        outs = self._init.run(
            None,
            {"input_ids": np.asarray([forced], dtype=np.int64), "encoder_hidden_states": enc},
        )
        token_times.append((time.perf_counter() - t0) * 1000)

        names = [o.name for o in self._init.get_outputs()]
        logits = outs[0]
        kv = {
            n.replace("present", "past_key_values"): v
            for n, v in zip(names[1:], outs[1:])
        }
        kv = {k: v for k, v in kv.items() if k in self._past_inputs}
        nxt = int(logits[0, -1].argmax())

        text = ""
        for _ in range(cap):
            if nxt == EOT:
                break
            # Text ids are strictly below EOT; everything from EOT up is a
            # special — EOT, SOT, the language tags, the task tags,
            # NO_TIMESTAMPS, then the timestamps. Bounding at TIMESTAMP_BEGIN
            # only excluded the last group, so a `<|startoftranscript|>` or
            # `<|transcribe|>` argmax still landed in the transcript verbatim.
            if nxt < EOT:
                text += vocab.get(str(nxt), "")
            t = time.perf_counter()
            feed: dict[str, np.ndarray] = {
                "input_ids": np.asarray([[nxt]], dtype=np.int64)
            }
            feed.update(kv)
            if "encoder_hidden_states" in self._past_inputs:
                feed["encoder_hidden_states"] = enc
            outs = self._past.run(None, feed)
            token_times.append((time.perf_counter() - t) * 1000)
            names = [o.name for o in self._past.get_outputs()]
            logits = outs[0]
            for n, v in zip(names[1:], outs[1:]):
                k = n.replace("present", "past_key_values")
                if k in self._past_inputs:
                    kv[k] = v
            nxt = int(logits[0, -1].argmax())
        return text, token_times


class TensorRTKVDecoder:
    """Whisper decoder with device-resident cross/self attention KV.

    The decoder accepts the borrowed ``(ptr, shape, dtype, owner)`` tuple from
    ``TensorRTEncoder.run_device``.  The owner is retained for the duration of
    decode; this class never frees that pointer.  Argmax is intentionally a
    small host read of logits, while hidden states and all KV stay on device.
    """

    def __init__(
        self,
        prefill_path: str | Path,
        step_path: str | Path,
        *,
        max_positions: int = MAX_POSITIONS,
    ) -> None:
        self._logger = self._runtime = None
        self._prefill = self._step = None
        self._stream = None
        self._owned: list[int] = []
        self._staged: dict[str, np.ndarray] = {}
        self._staged_ptr: dict[str, tuple[int, int]] = {}
        self._kv: dict[tuple[int, str], list[int]] = {}
        self._kv_bytes = 0
        self._max_positions = max_positions
        self._failed = False
        self._closed = False
        try:
            import tensorrt as trt
            from cuda import cudart

            self._cudart = cudart
            self._logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._logger)
            err, self._stream = cudart.cudaStreamCreate()
            self._check(err, "cudaStreamCreate")
            self._prefill = _TRTEngine(self._runtime, prefill_path, self._stream, cudart)
            self._step = _TRTEngine(self._runtime, step_path, self._stream, cudart)
            self._layers = sorted(
                int(name.split(".")[1])
                for name in self._prefill.names
                if name.startswith("present.") and ".decoder.key" in name
            )
            if not self._layers or self._layers != list(range(len(self._layers))):
                raise RuntimeError(f"unexpected Whisper TRT decoder layer names: {self._layers}")
            self._validate_io()
            self._profile_max = self._step.profile_max(
                "past_key_values.0.decoder.key"
            )[2]
            if self._profile_max < max_positions - 1:
                raise RuntimeError(
                    f"Whisper TRT step profile past max {self._profile_max} "
                    f"is below required {max_positions - 1}"
                )
        except Exception:
            self.close()
            raise

    def _check(self, result, what: str):
        err = result[0] if isinstance(result, tuple) else result
        if err != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{what} failed: {err}")

    def _malloc(self, nbytes: int) -> int:
        ptr = self._cudart.cudaMalloc(nbytes)
        err, value = ptr if isinstance(ptr, tuple) else (ptr, None)
        self._check(err, f"cudaMalloc({nbytes})")
        value = int(value)
        self._owned.append(value)
        return value

    def _free_owned(self, ptr: int) -> None:
        if ptr not in self._owned:
            return
        self._owned.remove(ptr)
        try:
            self._cudart.cudaFree(ptr)
        except Exception:
            pass

    @staticmethod
    def _static_compatible(left, right, *, label: str, dynamic_axes=()) -> None:
        if len(left) != len(right):
            raise RuntimeError(f"Whisper TRT {label} rank mismatch: {left} vs {right}")
        for axis, (a, b) in enumerate(zip(left, right)):
            if axis in dynamic_axes:
                continue
            if a != b:
                raise RuntimeError(f"Whisper TRT {label} shape mismatch: {left} vs {right}")

    def _validate_io(self) -> None:
        pre_inputs = {n for n in self._prefill.names if self._prefill.is_input[n]}
        pre_outputs = set(self._prefill.names) - pre_inputs
        step_inputs = {n for n in self._step.names if self._step.is_input[n]}
        step_outputs = set(self._step.names) - step_inputs
        expected_pre_in = {"input_ids", "encoder_hidden_states"}
        expected_step_in = {"input_ids"}
        expected_pre_out = {"logits"}
        expected_step_out = {"logits"}
        for layer in self._layers:
            for kind in ("key", "value"):
                expected_step_in.add(f"past_key_values.{layer}.decoder.{kind}")
                expected_step_in.add(f"past_key_values.{layer}.encoder.{kind}")
                expected_pre_out.add(f"present.{layer}.decoder.{kind}")
                expected_pre_out.add(f"present.{layer}.encoder.{kind}")
                expected_step_out.add(f"present.{layer}.decoder.{kind}")
        if pre_inputs != expected_pre_in or pre_outputs != expected_pre_out:
            raise RuntimeError(
                f"Whisper TRT prefill IO mismatch: inputs={sorted(pre_inputs)} outputs={sorted(pre_outputs)}"
            )
        if step_inputs != expected_step_in or step_outputs != expected_step_out:
            raise RuntimeError(
                f"Whisper TRT step IO mismatch: inputs={sorted(step_inputs)} outputs={sorted(step_outputs)}"
            )
        if np.dtype(self._prefill.dtype["input_ids"]) != np.dtype(np.int64):
            raise RuntimeError("Whisper TRT prefill input_ids must be int64")
        if np.dtype(self._step.dtype["input_ids"]) != np.dtype(np.int64):
            raise RuntimeError("Whisper TRT step input_ids must be int64")
        self._hidden_dtype = np.dtype(self._prefill.dtype["encoder_hidden_states"])
        if self._hidden_dtype != np.dtype(np.float32):
            raise RuntimeError(f"Whisper TRT decoder hidden dtype must be float32, got {self._hidden_dtype}")
        if np.dtype(self._prefill.dtype["logits"]) != np.dtype(np.float32):
            raise RuntimeError("Whisper TRT prefill logits dtype must be float32")
        if np.dtype(self._step.dtype["logits"]) != np.dtype(np.float32):
            raise RuntimeError("Whisper TRT step logits dtype must be float32")
        self._self_geometry = None
        self._cross_geometry = None
        for layer in self._layers:
            for kind in ("key", "value"):
                pre_dec = f"present.{layer}.decoder.{kind}"
                step_dec_in = f"past_key_values.{layer}.decoder.{kind}"
                step_dec_out = f"present.{layer}.decoder.{kind}"
                pre_cross = f"present.{layer}.encoder.{kind}"
                step_cross = f"past_key_values.{layer}.encoder.{kind}"
                io_shapes = {
                    pre_dec: self._prefill.engine.get_tensor_shape(pre_dec),
                    step_dec_in: self._step.engine.get_tensor_shape(step_dec_in),
                    step_dec_out: self._step.engine.get_tensor_shape(step_dec_out),
                    pre_cross: self._prefill.engine.get_tensor_shape(pre_cross),
                    step_cross: self._step.engine.get_tensor_shape(step_cross),
                }
                if any(len(shape) != 4 for shape in io_shapes.values()):
                    raise RuntimeError(
                        f"Whisper TRT KV tensors must be rank 4 at layer {layer}/{kind}: {io_shapes}"
                    )
                for a, b, label in (
                    (self._prefill.dtype[pre_dec], self._step.dtype[step_dec_in], "self-KV dtype"),
                    (self._prefill.dtype[pre_cross], self._step.dtype[step_cross], "cross-KV dtype"),
                    (self._step.dtype[step_dec_in], self._step.dtype[step_dec_out], "step KV dtype"),
                ):
                    if np.dtype(a) != np.dtype(b):
                        raise RuntimeError(f"Whisper TRT {label} mismatch at layer {layer}/{kind}")
                self._static_compatible(
                    io_shapes[pre_dec],
                    io_shapes[step_dec_in],
                    label=f"self-KV layer {layer}/{kind}", dynamic_axes=(2,),
                )
                self._static_compatible(
                    io_shapes[pre_cross],
                    io_shapes[step_cross],
                    label=f"cross-KV layer {layer}/{kind}", dynamic_axes=(2,),
                )
                self._static_compatible(
                    io_shapes[step_dec_in],
                    io_shapes[step_dec_out],
                    label=f"step KV layer {layer}/{kind}", dynamic_axes=(2,),
                )
                profile = self._step.profile_shape(step_dec_in)
                if (
                    len(profile) != 3
                    or any(len(shape) != 4 for shape in profile)
                    or any(int(shape[axis]) <= 0 for shape in profile for axis in (0, 1, 3))
                    or any(profile[0][axis] != profile[2][axis] for axis in (0, 1, 3))
                    or profile[0][2] > 4
                    or profile[2][2] < self._max_positions - 1
                ):
                    raise RuntimeError(f"Whisper TRT {step_dec_in} profile must cover past 4..447: {profile}")
                cross_profile = self._step.profile_shape(step_cross)
                if (
                    len(cross_profile) != 3
                    or any(len(shape) != 4 for shape in cross_profile)
                    or any(int(shape[axis]) <= 0 for shape in cross_profile for axis in (0, 1, 2, 3))
                    or any(cross_profile[0][axis] != cross_profile[2][axis] for axis in (0, 1, 2, 3))
                    or cross_profile[0][2] != 1500
                ):
                    raise RuntimeError(
                        f"Whisper TRT {step_cross} profile must fix cross-KV source length at 1500: "
                        f"{cross_profile}"
                    )
                pre_shape = io_shapes[pre_dec]
                step_shape = io_shapes[step_dec_in]
                geometry = (pre_shape[0], pre_shape[1], pre_shape[3])
                step_geometry = (step_shape[0], step_shape[1], step_shape[3])
                if geometry != step_geometry:
                    raise RuntimeError(f"Whisper TRT self-KV geometry mismatch at layer {layer}/{kind}")
                if self._self_geometry is None:
                    self._self_geometry = geometry
                elif geometry != self._self_geometry:
                    raise RuntimeError(f"Whisper TRT self-KV layer geometry mismatch at layer {layer}/{kind}")
                cross_shape = io_shapes[pre_cross]
                cross_geometry = (cross_shape[0], cross_shape[1], cross_shape[3])
                if self._cross_geometry is None:
                    self._cross_geometry = cross_geometry
                elif cross_geometry != self._cross_geometry:
                    raise RuntimeError(f"Whisper TRT cross-KV layer geometry mismatch at layer {layer}/{kind}")

    def _copy_h2d(self, name: str, arr: np.ndarray, engine: "_TRTEngine") -> int:
        arr = np.ascontiguousarray(arr, dtype=engine.dtype[name])
        self._staged[name] = arr
        previous = self._staged_ptr.get(name)
        if previous is not None and previous[1] >= arr.nbytes:
            # Keep the allocation's capacity when a narrower tensor is copied
            # into it.  Recording the current transfer length would make a
            # later wider tensor allocate again despite the original buffer
            # still being large enough.
            ptr, capacity = previous
        else:
            ptr = self._malloc(arr.nbytes)
            capacity = arr.nbytes
        self._staged_ptr[name] = (ptr, capacity)
        self._check(
            self._cudart.cudaMemcpyAsync(
                ptr,
                arr.ctypes.data,
                arr.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            ),
            f"H2D {name}",
        )
        engine.bind(name, ptr)
        return ptr

    def _copy_d2h(self, ptr: int, shape, dtype) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        self._check(
            self._cudart.cudaMemcpyAsync(
                out.ctypes.data,
                ptr,
                out.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            ),
            "D2H logits",
        )
        self._check(self._cudart.cudaStreamSynchronize(self._stream), "logits sync")
        return out

    def decode(
        self,
        encoder_out,
        vocab: dict[str, str],
        language: str,
        *,
        audio_s: float,
        max_new: Optional[int] = None,
    ) -> tuple[str, list[float]]:
        if self._closed:
            raise RuntimeError("Whisper TRT decoder is closed")
        if self._failed:
            raise RuntimeError("Whisper TRT decoder is unusable after a failed decode")
        try:
            return self._decode_impl(encoder_out, vocab, language, audio_s=audio_s, max_new=max_new)
        except Exception:
            self._failed = True
            raise

    def _decode_impl(
        self, encoder_out, vocab: dict[str, str], language: str, *, audio_s: float,
        max_new: Optional[int] = None,
    ) -> tuple[str, list[float]]:
        if not isinstance(encoder_out, tuple) or len(encoder_out) != 4:
            raise TypeError("TensorRT Whisper decoder requires a borrowed device output tuple")
        enc_ptr, enc_shape, enc_dtype, owner = encoder_out
        if owner is None:
            raise TypeError("Whisper TRT encoder output must be borrowed device memory")
        self._validate_encoder_view(enc_shape, enc_dtype)
        forced = np.asarray(
            [[SOT, LANG_TOKEN[language], TASK_TRANSCRIBE, NO_TIMESTAMPS]],
            dtype=np.int64,
        )
        budget = int(max(16, min(220, audio_s * 8 + 12)))
        hard_cap = self._max_positions - len(forced[0]) - 1
        cap = min(budget, hard_cap) if max_new is None else min(max_new, hard_cap)
        if max_new is not None and max_new < 1:
            raise ValueError(f"max_new must be >= 1, got {max_new}")

        t0 = time.perf_counter()
        self._prefill.set_shape("input_ids", forced.shape)
        self._prefill.set_shape("encoder_hidden_states", tuple(enc_shape))
        self._copy_h2d("input_ids", forced, self._prefill)
        self._prefill.bind("encoder_hidden_states", int(enc_ptr))
        self._prefill.allocate_outputs(self._malloc)
        self._prefill.run(self._stream)
        self._check(self._cudart.cudaStreamSynchronize(self._stream), "prefill sync")
        self._validate_resolved_prefill_shapes(tuple(enc_shape), forced.shape)
        token_times = [(time.perf_counter() - t0) * 1000]

        logits = self._copy_d2h(
            self._prefill.ptr["logits"],
            self._prefill.shape["logits"],
            self._prefill.dtype["logits"],
        )
        nxt = int(logits[0, -1].argmax())

        # Prefill cross-KV is borrowed by the step context; it is owned by the
        # prefill engine and remains valid until close().
        for layer in self._layers:
            for kind in ("key", "value"):
                src = f"present.{layer}.encoder.{kind}"
                dst = f"past_key_values.{layer}.encoder.{kind}"
                self._step.set_shape(dst, self._prefill.shape[src])
                self._step.bind(dst, self._prefill.ptr[src])

        b, _, past_len, d = self._prefill.shape["present.0.decoder.key"]
        h = self._prefill.shape["present.0.decoder.key"][1]
        item = np.dtype(self._step.dtype["past_key_values.0.decoder.key"]).itemsize
        kv_bytes = int(b * h * (self._profile_max + 1) * d * item)
        kv = self._ensure_kv_buffers(kv_bytes)
        for layer in self._layers:
            for kind in ("key", "value"):
                src = self._prefill.ptr[f"present.{layer}.decoder.{kind}"]
                n = int(np.prod(self._prefill.shape[f"present.{layer}.decoder.{kind}"])) * item
                self._check(
                    self._cudart.cudaMemcpyAsync(
                        kv[(layer, kind)][0], src, n,
                        self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        self._stream,
                    ),
                    "prefill self-KV copy",
                )
        self._check(self._cudart.cudaStreamSynchronize(self._stream), "KV copy sync")

        text = ""
        for _ in range(cap):
            if nxt == EOT:
                break
            if nxt < EOT:
                text += vocab.get(str(nxt), "")
            self._check_past_length(past_len)
            ids = np.asarray([[nxt]], dtype=np.int64)
            self._step.set_shape("input_ids", ids.shape)
            # TensorRT resolves dynamic outputs only after every dynamic input
            # shape in the context is set.  Querying a present tensor inside
            # the per-layer loop leaves the remaining self-KV inputs
            # unresolved and can return (1, 8, -1, 64), even though the
            # requested shape is valid.  Complete the input-shape phase first.
            for layer in self._layers:
                for kind in ("key", "value"):
                    self._step.set_shape(
                        f"past_key_values.{layer}.decoder.{kind}",
                        (b, h, past_len, d),
                    )
            for layer in self._layers:
                for kind in ("key", "value"):
                    out_name = f"present.{layer}.decoder.{kind}"
                    resolved = tuple(self._step.ctx.get_tensor_shape(
                        out_name
                    ))
                    expected = (b, h, past_len + 1, d)
                    if resolved != expected:
                        raise RuntimeError(
                            f"Whisper TRT step self-KV output shape mismatch: {resolved} vs {expected}"
                        )
            for layer in self._layers:
                for kind in ("key", "value"):
                    past, present = kv[(layer, kind)]
                    self._step.bind(f"past_key_values.{layer}.decoder.{kind}", past)
                    self._step.bind(
                        f"present.{layer}.decoder.{kind}", present
                    )
            self._copy_h2d("input_ids", ids, self._step)
            self._step.allocate_outputs(self._malloc, only=("logits",))
            t = time.perf_counter()
            self._step.run(self._stream)
            self._check(self._cudart.cudaStreamSynchronize(self._stream), "step sync")
            logits = self._copy_d2h(
                self._step.ptr["logits"], self._step.shape["logits"], self._step.dtype["logits"]
            )
            token_times.append((time.perf_counter() - t) * 1000)
            for pair in kv.values():
                pair.reverse()
            past_len += 1
            nxt = int(logits[0, 0].argmax())
        return text, token_times

    def _validate_resolved_prefill_shapes(self, enc_shape, forced_shape) -> None:
        if tuple(forced_shape) != (1, 4):
            raise RuntimeError(f"Whisper TRT forced input must be [1,4], got {forced_shape}")
        if len(enc_shape) != 3 or any(int(x) <= 0 for x in enc_shape):
            raise RuntimeError(f"Whisper TRT encoder shape must be positive rank-3, got {enc_shape}")
        if tuple(self._prefill.shape["encoder_hidden_states"]) != tuple(enc_shape):
            raise RuntimeError("Whisper TRT encoder hidden shape was not resolved as requested")
        logits = tuple(self._prefill.shape["logits"])
        if logits != (1, 4, 51865):
            raise RuntimeError(f"Whisper TRT prefill logits shape mismatch: {logits}")
        self_shape = None
        cross_shape = None
        for layer in self._layers:
            for kind in ("key", "value"):
                s = tuple(self._prefill.shape[f"present.{layer}.decoder.{kind}"])
                c = tuple(self._prefill.shape[f"present.{layer}.encoder.{kind}"])
                if len(s) != 4 or any(x <= 0 for x in s) or s[0] != enc_shape[0] or s[2] != 4:
                    raise RuntimeError(f"Whisper TRT resolved self-KV shape invalid: {s}")
                if len(c) != 4 or any(x <= 0 for x in c) or c[0] != enc_shape[0] or c[2] != enc_shape[1]:
                    raise RuntimeError(f"Whisper TRT resolved cross-KV shape invalid: {c}")
                if self_shape is None:
                    self_shape = s
                elif (s[0], s[1], s[3]) != (self_shape[0], self_shape[1], self_shape[3]):
                    raise RuntimeError(f"Whisper TRT resolved self-KV geometry mismatch: {s} vs {self_shape}")
                if cross_shape is None:
                    cross_shape = c
                elif (c[0], c[1], c[3]) != (cross_shape[0], cross_shape[1], cross_shape[3]):
                    raise RuntimeError(f"Whisper TRT resolved cross-KV geometry mismatch: {c} vs {cross_shape}")
                profile = self._step.profile_shape(f"past_key_values.{layer}.encoder.{kind}")
                if not self._profile_contains(profile, c):
                    raise RuntimeError(f"Whisper TRT cross-KV profile does not accept {c}: {profile}")
        if self_shape is None or cross_shape is None:
            raise RuntimeError("Whisper TRT decoder has no KV outputs")
        b, h, _, d = self_shape
        for layer in self._layers:
            for kind in ("key", "value"):
                profile = self._step.profile_shape(f"past_key_values.{layer}.decoder.{kind}")
                if not self._profile_contains(profile, (b, h, 4, d)):
                    raise RuntimeError(f"Whisper TRT self-KV profile rejects prefill shape: {profile}")
                if profile[2][2] < self._max_positions - 1:
                    raise RuntimeError(f"Whisper TRT self-KV profile max is too short: {profile}")
        hidden_profile = self._prefill.profile_shape("encoder_hidden_states")
        if not self._profile_contains(hidden_profile, tuple(enc_shape)):
            raise RuntimeError(f"Whisper TRT hidden profile rejects {enc_shape}: {hidden_profile}")

    @staticmethod
    def _profile_contains(profile, shape) -> bool:
        return all(int(lo) <= int(v) <= int(hi) for v, lo, hi in zip(shape, profile[0], profile[2]))

    def _validate_encoder_view(self, shape, dtype) -> None:
        if np.dtype(dtype) != self._hidden_dtype:
            raise TypeError(
                f"Whisper TRT encoder output dtype {np.dtype(dtype)} does not match "
                f"decoder hidden dtype {self._hidden_dtype}"
            )
        expected = self._prefill.engine.get_tensor_shape("encoder_hidden_states")
        self._static_compatible(tuple(shape), expected, label="encoder hidden", dynamic_axes=(1,))

    def _check_past_length(self, past_len: int) -> None:
        if past_len > self._max_positions - 1 or past_len > self._profile_max:
            raise RuntimeError(f"Whisper TRT decoder past length overflow: {past_len}")

    def _ensure_kv_buffers(self, kv_bytes: int):
        if self._kv and self._kv_bytes >= kv_bytes:
            return self._kv
        old = self._kv
        fresh: dict[tuple[int, str], list[int]] = {}
        allocated: list[int] = []
        try:
            for layer in self._layers:
                for kind in ("key", "value"):
                    pair = [self._malloc(kv_bytes), self._malloc(kv_bytes)]
                    allocated.extend(pair)
                    fresh[(layer, kind)] = pair
        except Exception:
            for ptr in allocated:
                self._free_owned(ptr)
            raise
        self._kv = fresh
        self._kv_bytes = kv_bytes
        for pair in old.values():
            for ptr in pair:
                self._free_owned(ptr)
        return fresh

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        cudart = getattr(self, "_cudart", None)
        if cudart is not None:
            for ptr in reversed(getattr(self, "_owned", [])):
                try:
                    cudart.cudaFree(ptr)
                except Exception:
                    pass
        self._owned = []
        self._kv = {}
        self._kv_bytes = 0
        for name in ("_step", "_prefill"):
            engine = getattr(self, name, None)
            if engine is not None:
                engine.close()
                setattr(self, name, None)
        if cudart is not None and getattr(self, "_stream", None) is not None:
            try:
                cudart.cudaStreamDestroy(self._stream)
            except Exception:
                pass
        self._stream = None
        self._runtime = None
        self._logger = None


class _TRTEngine:
    """Small TensorRT engine wrapper used only by ``TensorRTKVDecoder``."""

    def __init__(self, runtime, path, stream, cudart):
        import tensorrt as trt

        self._runtime = runtime
        self._cudart = cudart
        self._stream = stream
        self.engine = None
        self.ctx = None
        try:
            self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self.engine is None:
                raise RuntimeError(f"failed to deserialize {path}")
            self.ctx = self.engine.create_execution_context()
            if self.ctx is None:
                raise RuntimeError(f"failed to create execution context for {path}")
            self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.is_input = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in self.names}
            self.dtype = {n: trt.nptype(self.engine.get_tensor_dtype(n)) for n in self.names}
            self.ptr: dict[str, int] = {}
            self.shape: dict[str, tuple[int, ...]] = {}
        except Exception:
            self.close()
            raise

    def set_shape(self, name, shape):
        if not self.ctx.set_input_shape(name, tuple(shape)):
            raise RuntimeError(f"TensorRT set_input_shape failed for {name}: {shape}")
        self.shape[name] = tuple(shape)

    def bind(self, name, ptr):
        if not self.ctx.set_tensor_address(name, int(ptr)):
            raise RuntimeError(f"TensorRT set_tensor_address failed for {name}")
        self.ptr[name] = int(ptr)

    def allocate_outputs(self, malloc, only=None):
        allowed = set(only) if only is not None else None
        for name in self.names:
            if self.is_input[name] or (allowed is not None and name not in allowed):
                continue
            shape = tuple(self.ctx.get_tensor_shape(name))
            if any(x < 0 for x in shape):
                raise RuntimeError(f"unresolved TensorRT output shape for {name}: {shape}")
            old_shape = self.shape.get(name)
            if name in self.ptr and old_shape == shape:
                self.bind(name, self.ptr[name])
                continue
            self.shape[name] = shape
            self.bind(name, malloc(int(np.prod(shape)) * np.dtype(self.dtype[name]).itemsize))

    def run(self, stream):
        if not self.ctx.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def profile_max(self, name):
        return tuple(self.engine.get_tensor_profile_shape(name, 0)[2])

    def profile_shape(self, name):
        return tuple(tuple(x) for x in self.engine.get_tensor_profile_shape(name, 0))

    def close(self):
        self.ctx = None
        self.engine = None
        self._runtime = None


def read_vocab(path: str | Path) -> dict[str, str]:
    """Rockchip's id->token table.

    Splits on the FIRST space, matching their reader. Using rpartition instead
    silently mangles every token that contains a space.
    """
    vocab: dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            # Only the newline goes. `.strip()` would eat a token that IS a
            # space or begins with one — that is how word boundaries are
            # encoded — and splitting on every space truncated any token
            # containing one ("123 foo bar" mapped 123 to "foo").
            key, _, value = line.rstrip("\n").partition(" ")
            if key:
                vocab[key] = value
    return vocab


def _b64_index(c: str) -> int:
    if "A" <= c <= "Z":
        return ord(c) - 65
    if "a" <= c <= "z":
        return ord(c) - 97 + 26
    if "0" <= c <= "9":
        return ord(c) - 48 + 52
    return 62 if c == "+" else 63


def base64_decode(s: str) -> str:
    """Rockchip's hand-rolled decoder, not a stdlib drop-in.

    It returns a single space the moment it meets '=', which is how their zh
    vocab encodes a word break — ``base64.b64decode`` has different semantics
    here. Upstream also returns the whole pre-sized buffer, so short decodes
    carry trailing NULs; invisible on a terminal, scored as insertions. Hence
    the ``[:oi]``.
    """
    if not s:
        return ""
    # Upstream reads s[i+1] unconditionally and sizes the buffer for a length
    # that is a multiple of 4. A truncated or non-base64 token stream — a
    # decoder cut short, a vocab mismatch — then indexes past the end instead
    # of degrading. Pad the input and size the buffer for the padded length.
    if len(s) % 4:
        s = s + "=" * (4 - len(s) % 4)
    out = bytearray(len(s) // 4 * 3 + 3)
    i = oi = 0
    while i < len(s):
        if s[i] == "=":
            return " "
        out[oi] = (_b64_index(s[i]) << 2) + ((_b64_index(s[i + 1]) & 0x30) >> 4)
        if i + 2 < len(s) and s[i + 2] != "=":
            out[oi + 1] = ((_b64_index(s[i + 1]) & 0x0F) << 4) + (
                (_b64_index(s[i + 2]) & 0x3C) >> 2
            )
            if i + 3 < len(s) and s[i + 3] != "=":
                out[oi + 2] = ((_b64_index(s[i + 2]) & 0x03) << 6) + _b64_index(s[i + 3])
                oi += 3
            else:
                oi += 2
        else:
            oi += 1
        i += 4
    return out[:oi].decode("utf-8", errors="replace")


def detokenize(raw: str, language: str) -> str:
    text = raw.replace("Ġ", " ").replace("<|endoftext|>", "").replace("\n", "")
    return base64_decode(text) if language == "zh" else text
