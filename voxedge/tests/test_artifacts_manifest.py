"""Tests for the backend-agnostic artifact manifest + env-free download helper.

All tests are offline: a local directory stands in for the "HF source" and a
local ``download_fn`` copies files from it, so the manifest/SHA logic is
exercised without network or ``huggingface_hub``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from voxedge.artifacts import (
    SCHEMA_VERSION,
    ArtifactError,
    ManifestError,
    default_manifest_path,
    load_manifest,
    parse_manifest,
    resolve_artifact,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_source_tree(root: Path, files: dict[str, bytes]) -> None:
    """Lay out a fake HF source tree at ``root`` keyed by source_path."""
    for source_path, data in files.items():
        dest = root / source_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


def _local_download_fn(source_root: Path):
    """A download_fn that copies from a local source tree (no network)."""

    def _dl(repo_id, revision, source_path, dest, endpoint):
        src = source_root / source_path
        if not src.exists():
            raise FileNotFoundError(f"local source missing: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    return _dl


def _manifest_dict(file_list: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "demo-artifact": {
                "artifact_ref": "demo-artifact",
                "backend_key": "rk.tts",
                "device": "rk3588",
                "precision": "int8",
                "hf_repo": "org/demo-repo",
                "revision": "main",
                "runtime_contract": {"env": {"TTS_BACKEND": "kokoro_rknn"}},
                "file_list": file_list,
            }
        },
    }


def _write_manifest(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


# ── manifest parsing / schema validation ─────────────────────────────────────


def test_bundled_sample_manifest_parses_and_includes_style_npy():
    manifest = load_manifest(default_manifest_path())
    spec = manifest.get("rk3588-kokoro-hybrid-2026-05-23")
    assert spec.backend_key == "rk.tts"
    assert spec.device == "rk3588"
    assert spec.hf_repo == "example-org/example-artifacts"
    rels = [f.rel_path for f in spec.file_list]
    # style.npy MUST be present — demonstrates voice-pack preflight coverage.
    assert any(p.endswith("style.npy") for p in rels), rels


def test_parse_rejects_wrong_schema_version():
    with pytest.raises(ManifestError, match="schema_version"):
        parse_manifest({"schema_version": 99, "artifacts": {}})


def test_parse_rejects_empty_artifacts():
    with pytest.raises(ManifestError, match="artifacts"):
        parse_manifest({"schema_version": SCHEMA_VERSION, "artifacts": {}})


def test_parse_rejects_empty_file_list():
    data = _manifest_dict([])
    with pytest.raises(ManifestError, match="file_list"):
        parse_manifest(data)


def test_parse_rejects_file_missing_sha():
    data = _manifest_dict([{"path": "a/b.bin"}])
    with pytest.raises(ManifestError, match="sha256"):
        parse_manifest(data)


def test_parse_rejects_key_ref_mismatch():
    data = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "key-a": {
                "artifact_ref": "key-b",
                "file_list": [{"path": "x", "sha256": "00"}],
            }
        },
    }
    with pytest.raises(ManifestError, match="artifact_ref"):
        parse_manifest(data)


def test_source_path_defaults_to_path():
    data = _manifest_dict([{"path": "a/b.bin", "sha256": "abc"}])
    spec = parse_manifest(data).get("demo-artifact")
    f = spec.file_list[0]
    assert f.source_rel == "a/b.bin"


def test_load_manifest_missing_file(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.json")


def test_load_manifest_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(p)


def test_get_unknown_artifact_ref():
    data = _manifest_dict([{"path": "a", "sha256": "00"}])
    manifest = parse_manifest(data)
    with pytest.raises(ManifestError, match="not found"):
        manifest.get("does-not-exist")


# ── resolve_artifact: download + install + SHA ───────────────────────────────


def test_resolve_installs_files_to_install_root(tmp_path):
    src_root = tmp_path / "src"
    files = {
        "src/foo.bin": b"hello-foo",
        "src/sub/bar.npy": b"hello-bar",
    }
    _make_source_tree(src_root, files)

    data = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "demo-artifact": {
                "artifact_ref": "demo-artifact",
                "hf_repo": "org/demo",
                "file_list": [
                    {
                        "path": "models/foo.bin",
                        "source_path": "src/foo.bin",
                        "sha256": _sha256_bytes(b"hello-foo"),
                    },
                    {
                        "path": "models/sub/bar.npy",
                        "source_path": "src/sub/bar.npy",
                        "sha256": _sha256_bytes(b"hello-bar"),
                    },
                ],
            }
        },
    }
    manifest_path = _write_manifest(tmp_path / "m.json", data)
    install_root = tmp_path / "install"

    result = resolve_artifact(
        "demo-artifact",
        install_root,
        manifest_path=manifest_path,
        download_fn=_local_download_fn(src_root),
    )

    assert (install_root / "models/foo.bin").read_bytes() == b"hello-foo"
    assert (install_root / "models/sub/bar.npy").read_bytes() == b"hello-bar"
    assert sorted(result.installed) == ["models/foo.bin", "models/sub/bar.npy"]
    assert result.skipped == []
    assert result.ready is True


def test_resolve_skips_already_present_and_correct(tmp_path):
    src_root = tmp_path / "src"
    _make_source_tree(src_root, {"foo.bin": b"data"})
    sha = _sha256_bytes(b"data")
    data = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "demo-artifact": {
                "artifact_ref": "demo-artifact",
                "hf_repo": "org/demo",
                "file_list": [
                    {"path": "m/foo.bin", "source_path": "foo.bin", "sha256": sha}
                ],
            }
        },
    }
    manifest_path = _write_manifest(tmp_path / "m.json", data)
    install_root = tmp_path / "install"
    # Pre-place the correct file.
    (install_root / "m").mkdir(parents=True)
    (install_root / "m/foo.bin").write_bytes(b"data")

    calls = []

    def _counting_dl(*a):
        calls.append(a)

    result = resolve_artifact(
        "demo-artifact",
        install_root,
        manifest_path=manifest_path,
        download_fn=_counting_dl,
    )
    assert calls == []  # never downloaded
    assert result.skipped == ["m/foo.bin"]
    assert result.installed == []
    assert result.ready is True


def test_resolve_sha_mismatch_raises_and_removes(tmp_path):
    src_root = tmp_path / "src"
    _make_source_tree(src_root, {"foo.bin": b"actual-bytes"})
    data = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "demo-artifact": {
                "artifact_ref": "demo-artifact",
                "hf_repo": "org/demo",
                "file_list": [
                    {
                        "path": "m/foo.bin",
                        "source_path": "foo.bin",
                        "sha256": _sha256_bytes(b"WRONG-expected"),
                    }
                ],
            }
        },
    }
    manifest_path = _write_manifest(tmp_path / "m.json", data)
    install_root = tmp_path / "install"

    with pytest.raises(ArtifactError, match="sha256 mismatch"):
        resolve_artifact(
            "demo-artifact",
            install_root,
            manifest_path=manifest_path,
            download_fn=_local_download_fn(src_root),
        )
    # corrupt file removed
    assert not (install_root / "m/foo.bin").exists()


def test_resolve_missing_source_raises(tmp_path):
    data = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "demo-artifact": {
                "artifact_ref": "demo-artifact",
                "hf_repo": "org/demo",
                "file_list": [
                    {"path": "m/foo.bin", "source_path": "absent.bin", "sha256": "ab"}
                ],
            }
        },
    }
    manifest_path = _write_manifest(tmp_path / "m.json", data)
    src_root = tmp_path / "src"
    src_root.mkdir()

    with pytest.raises(ArtifactError, match="download failed"):
        resolve_artifact(
            "demo-artifact",
            tmp_path / "install",
            manifest_path=manifest_path,
            download_fn=_local_download_fn(src_root),
        )


def test_resolve_missing_file_no_repo_is_preflight_fail(tmp_path):
    """A required file absent with no hf_repo to fetch it → ArtifactError."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "demo-artifact": {
                "artifact_ref": "demo-artifact",
                "hf_repo": "",  # nothing to download from
                "file_list": [
                    {"path": "m/foo.bin", "sha256": _sha256_bytes(b"x")}
                ],
            }
        },
    }
    manifest_path = _write_manifest(tmp_path / "m.json", data)
    with pytest.raises(ArtifactError, match="preflight fail"):
        resolve_artifact(
            "demo-artifact", tmp_path / "install", manifest_path=manifest_path
        )


def test_resolve_unknown_artifact_ref(tmp_path):
    data = _manifest_dict([{"path": "a", "sha256": _sha256_bytes(b"x")}])
    manifest_path = _write_manifest(tmp_path / "m.json", data)
    with pytest.raises(ManifestError, match="not found"):
        resolve_artifact(
            "nope", tmp_path / "install", manifest_path=manifest_path
        )


# ── style.npy preflight semantics ────────────────────────────────────────────


def test_missing_style_npy_fails_preflight(tmp_path):
    """A kokoro voice pack manifest whose style.npy is absent at the source
    must fail loudly (root cause of the silent-audio voice-pack bug)."""
    src_root = tmp_path / "src"
    # decoder present, style.npy intentionally NOT created at source
    _make_source_tree(src_root, {"decoder.rknn": b"engine-bytes"})

    data = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "kokoro": {
                "artifact_ref": "kokoro",
                "backend_key": "rk.tts",
                "hf_repo": "org/kokoro",
                "file_list": [
                    {
                        "path": "opt/kokoro/decoder.rknn",
                        "source_path": "decoder.rknn",
                        "sha256": _sha256_bytes(b"engine-bytes"),
                    },
                    {
                        "path": "opt/kokoro/voices/style.npy",
                        "source_path": "voices/style.npy",
                        "sha256": _sha256_bytes(b"style-bytes"),
                    },
                ],
            }
        },
    }
    manifest_path = _write_manifest(tmp_path / "m.json", data)

    with pytest.raises(ArtifactError, match="style.npy"):
        resolve_artifact(
            "kokoro",
            tmp_path / "install",
            manifest_path=manifest_path,
            download_fn=_local_download_fn(src_root),
        )


def test_style_npy_installs_when_present(tmp_path):
    src_root = tmp_path / "src"
    _make_source_tree(
        src_root,
        {"decoder.rknn": b"engine-bytes", "voices/style.npy": b"style-bytes"},
    )
    data = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": {
            "kokoro": {
                "artifact_ref": "kokoro",
                "hf_repo": "org/kokoro",
                "file_list": [
                    {
                        "path": "opt/kokoro/decoder.rknn",
                        "source_path": "decoder.rknn",
                        "sha256": _sha256_bytes(b"engine-bytes"),
                    },
                    {
                        "path": "opt/kokoro/voices/style.npy",
                        "source_path": "voices/style.npy",
                        "sha256": _sha256_bytes(b"style-bytes"),
                    },
                ],
            }
        },
    }
    manifest_path = _write_manifest(tmp_path / "m.json", data)
    install_root = tmp_path / "install"
    result = resolve_artifact(
        "kokoro",
        install_root,
        manifest_path=manifest_path,
        download_fn=_local_download_fn(src_root),
    )
    assert (install_root / "opt/kokoro/voices/style.npy").read_bytes() == b"style-bytes"
    assert result.ready is True


# ── config wiring ────────────────────────────────────────────────────────────


def test_config_dataclasses_have_artifact_ref():
    from voxedge.backends.rk.tts import RKTTSConfig
    from voxedge.backends.rk.asr import RKASRConfig
    from voxedge.backends.jetson.trt_edge_llm_asr import TRTEdgeLLMASRConfig
    from voxedge.backends.jetson.trt_edge_llm_tts import TRTEdgeLLMTTSConfig
    from voxedge.backends.jetson.kokoro_trt import KokoroTRTConfig

    # default None — does not change existing behaviour
    assert RKTTSConfig().artifact_ref is None
    assert RKASRConfig().artifact_ref is None
    assert TRTEdgeLLMASRConfig().artifact_ref is None
    assert TRTEdgeLLMTTSConfig().artifact_ref is None
    assert KokoroTRTConfig().artifact_ref is None
    # settable
    assert RKTTSConfig(artifact_ref="rk3588-kokoro-hybrid-2026-05-23").artifact_ref
