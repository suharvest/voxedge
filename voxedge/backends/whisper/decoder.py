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
