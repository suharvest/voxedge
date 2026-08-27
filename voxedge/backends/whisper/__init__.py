"""OpenAI Whisper on edge accelerators.

One backend, three encoder execution paths (Hailo HEF / Rockchip RKNN /
TensorRT), all sharing the same mel front end and the same CPU KV-cache
decoder. Measurements behind the design are in the seeed-local-voice repo at
``docs/perf/whisper-cross-device-20260827.md``.

Two things drive the shape of this backend:

* **The NPU decoders are not usable.** Neither Hailo's fixed-32-token HEF
  decoder nor Rockchip's 12-slot sliding-window one has a KV cache, so both
  recompute the whole sequence every step. Running the encoder on the
  accelerator and the decoder as a CPU ONNX graph with a real KV cache measured
  both faster and more accurate on every board. The all-NPU path is therefore
  not offered here.

* **Whisper has no streaming state.** The encoder window is fixed at build
  time; there is nothing to carry across chunks. This backend is offline and
  sets ``supports_offline_streaming``, so the framework's
  ``OfflineAccumulateStream`` provides a session that accumulates and
  transcribes at finalize. It does not emit partials — see ``capabilities``.
"""

from .asr import WhisperASR, WhisperASRConfig

__all__ = ["WhisperASR", "WhisperASRConfig"]
