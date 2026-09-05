# VoxEdge Kokoro/RK productization evidence

## Baseline

- Worktree: `voxedge-kokoro-productization`
- Branch: `feat/kokoro-rk-productization`
- Baseline: `origin/main` at `3481a408a5aae5671003fca25b1817f91398a846`
- Release version: `0.0.13a0`
- RK extra: `rkvoice-stream>=0.2.0` on aarch64

## Source and documentation

- `voxedge/backends/base.py`: closes delegated iterators on consumer exit.
- `voxedge/backends/rk/tts.py`: lifecycle serialization, retryable teardown ownership, cancellation forwarding, runtime diagnostics, Kokoro streaming capability, and kana language routing.
- `voxedge/tests/test_rk_tts_lifecycle.py`
- `voxedge/tests/test_rk_tts_kokoro_language_routing.py`
- `README.md`, `README.zh-CN.md`: OVS/RKVoice integration, external read-only bundles, language routing, cancellation and streaming boundaries, and pinned install command.
- `uv.lock`: resolved RKVoice Stream `0.2.0` from the current productization commit `32b4694e5946eb8bed63db6ed8116aa4b146aa94`.

## Verification

```
39 passed in 0.18s
33 passed, 588 deselected in 0.23s
uv pip check: All installed packages are compatible
voxedge.__version__: 0.0.13a0
```

The first command covered the two new tests, dependency/factory integration checks, and packaging tests. The second covered all RK TTS and Kokoro tests, including the concurrent unload check.

## Distributions

- `/tmp/voxedge-build-v3/voxedge-0.0.13a0-py3-none-any.whl`
  - SHA-256: `26ad2bfaf407f8a0d1e065139e37cf9c1f39b40b9391121ad4b7a2e72ec3d7ae`
  - Contains `voxedge/backends/rk/tts.py`; metadata version is `0.0.13a0` and declares `rkvoice-stream>=0.2.0` for the RK extra.
- `/tmp/voxedge-build-v3/voxedge-0.0.13a0.tar.gz`
  - SHA-256: `538bff41e50affc4a5aef8dc8f525e9e2e56d58a52cce9dd3aacc770275918f6`
  - Contains `voxedge/backends/rk/tts.py`.

## Repository checks

`git diff --check` passed. `voxedge==0.0.13a0` is published on PyPI; GitHub
publication is pending a repository-scoped write credential. `pyproject.toml`
and `uv.lock` both record the exact RKVoice commit
`32b4694e5946eb8bed63db6ed8116aa4b146aa94`.
