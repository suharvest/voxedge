"""Matcha mel-time padding to the TRT profile minimum (migration gap).

The Matcha decoder/vocos TRT engines have a minimum mel-time profile; short
utterances produce mel tensors below it and must be zero-padded on axis 2 before
the engine accepts them. Production captured the minimum from
``MATCHA_MIN_MEL_FRAMES`` via a module constant; the env-free voxedge port turns
it into an explicit ``min_frames`` argument (``_pad_mel_axis(arr, min_frames)``),
sourced from ``config.min_mel_frames``. The env → ``min_mel_frames`` mapping is
covered product-side; this locks the padding algorithm itself, which had no
voxedge coverage after the rewrite. NumPy-only, no CUDA.
"""

from __future__ import annotations

import numpy as np

from voxedge.backends.jetson.matcha_trt import MEL_DIM, _pad_mel_axis


def test_pad_extends_short_tensor_to_min_frames():
    arr = np.ones((1, MEL_DIM, 64), dtype=np.float32)
    padded = _pad_mel_axis(arr, min_frames=72)
    assert padded.shape == (1, MEL_DIM, 72)
    np.testing.assert_array_equal(padded[:, :, :64], arr)
    assert np.all(padded[:, :, 64:] == 0)


def test_pad_leaves_at_or_above_min_unchanged():
    arr = np.ones((1, MEL_DIM, 72), dtype=np.float32)
    assert _pad_mel_axis(arr, min_frames=72) is arr
    longer = np.ones((1, MEL_DIM, 128), dtype=np.float32)
    assert _pad_mel_axis(longer, min_frames=72) is longer


def test_pad_respects_configurable_min_frames():
    arr = np.ones((1, MEL_DIM, 64), dtype=np.float32)
    assert _pad_mel_axis(arr, min_frames=96).shape == (1, MEL_DIM, 96)
    assert _pad_mel_axis(arr, min_frames=128).shape == (1, MEL_DIM, 128)
