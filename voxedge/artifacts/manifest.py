"""Backend-agnostic runtime artifact manifest — schema + loader.

This module is **env-free** and **pure-Python** (stdlib + dataclasses only):
no ``os.environ`` reads, no ``huggingface_hub`` import, no network. It only
parses and validates a JSON manifest that describes the runtime artifacts a
backend needs (engines / plugins / model weights / voice-pack sidecars), each
pinned by SHA-256 and referenced by a stable ``artifact_ref`` name.

Schema (a backend-agnostic generalisation of a per-device artifact manifest):

    {
      "schema_version": 1,
      "artifacts": {
        "<artifact_ref>": {
          "backend_key": "rk.tts" | "jetson.trt_edge_llm" | ...,
          "artifact_ref": "<stable name>",         # echoed; must match the key
          "device":      "rk3588" | "jetson-orin-sm87" | ...,
          "precision":   "fp16" | "int8" | "w4a16" | ...,
          "hf_repo":     "<org>/<repo>",
          "revision":    "main" | "<commit-sha>",
          "sha256":      "<top-level digest, optional>",
          "runtime_contract": { ... },             # env/profile constraints
          "file_list": [
            { "path": "<artifact-relative path, NO install root>",
              "source_path": "<HF source path, defaults to path>",
              "sha256": "<per-file digest>",
              "size_bytes": <int, optional> },
            ...
          ]
        }
      }
    }

``file_list[].path`` is the artifact-relative install path; the install root is
chosen by the caller (product / profile) at :func:`resolve_artifact` time — the
manifest never encodes an absolute install root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """Raised when a manifest is malformed or an artifact_ref is missing."""


@dataclass(frozen=True)
class ArtifactFile:
    """One file in an artifact's ``file_list``.

    ``path`` is the artifact-relative destination (no install root). ``source_path``
    is the path inside the HF repo to download from; it defaults to ``path``.
    """

    path: str
    sha256: str
    source_path: str = ""
    size_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.path or not str(self.path).strip():
            raise ManifestError("file_list entry missing 'path'")
        if not self.sha256 or not str(self.sha256).strip():
            raise ManifestError(f"file_list entry {self.path!r} missing 'sha256'")
        # Normalise source_path to default to path (frozen → object.__setattr__).
        if not self.source_path:
            object.__setattr__(self, "source_path", self.path)

    @property
    def rel_path(self) -> str:
        """Artifact-relative install path, leading slash stripped."""
        return self.path.lstrip("/")

    @property
    def source_rel(self) -> str:
        """HF-source-relative path, leading slash stripped."""
        return self.source_path.lstrip("/")


@dataclass(frozen=True)
class ArtifactSpec:
    """A single named runtime artifact (resolved by ``artifact_ref``)."""

    artifact_ref: str
    file_list: tuple[ArtifactFile, ...]
    backend_key: str = ""
    device: str = ""
    precision: str = ""
    hf_repo: str = ""
    revision: str = "main"
    sha256: str = ""
    runtime_contract: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifact_ref:
            raise ManifestError("artifact spec missing 'artifact_ref'")
        if not self.file_list:
            raise ManifestError(
                f"artifact {self.artifact_ref!r} has an empty file_list"
            )


@dataclass(frozen=True)
class ArtifactManifest:
    """Parsed, validated manifest. Keyed by ``artifact_ref``."""

    schema_version: int
    artifacts: dict[str, ArtifactSpec]
    hf_endpoint: str = ""

    def get(self, artifact_ref: str) -> ArtifactSpec:
        """Return the :class:`ArtifactSpec` for ``artifact_ref`` or raise."""
        spec = self.artifacts.get(artifact_ref)
        if spec is None:
            raise ManifestError(
                f"artifact_ref {artifact_ref!r} not found; "
                f"available={sorted(self.artifacts)}"
            )
        return spec

    def __contains__(self, artifact_ref: str) -> bool:
        return artifact_ref in self.artifacts


def _parse_file(raw: dict[str, Any]) -> ArtifactFile:
    if not isinstance(raw, dict):
        raise ManifestError(f"file_list entry must be an object, got {type(raw)!r}")
    return ArtifactFile(
        path=str(raw.get("path", "")),
        sha256=str(raw.get("sha256", "")),
        source_path=str(raw.get("source_path", "") or ""),
        size_bytes=raw.get("size_bytes"),
    )


def _parse_spec(artifact_ref: str, raw: dict[str, Any]) -> ArtifactSpec:
    if not isinstance(raw, dict):
        raise ManifestError(
            f"artifact {artifact_ref!r} must be an object, got {type(raw)!r}"
        )
    declared_ref = str(raw.get("artifact_ref", "") or artifact_ref)
    if declared_ref != artifact_ref:
        raise ManifestError(
            f"artifact key {artifact_ref!r} != declared artifact_ref "
            f"{declared_ref!r}"
        )
    files_raw = raw.get("file_list")
    if not isinstance(files_raw, list) or not files_raw:
        raise ManifestError(
            f"artifact {artifact_ref!r} must declare a non-empty 'file_list'"
        )
    file_list = tuple(_parse_file(f) for f in files_raw)
    return ArtifactSpec(
        artifact_ref=artifact_ref,
        file_list=file_list,
        backend_key=str(raw.get("backend_key", "") or ""),
        device=str(raw.get("device", "") or ""),
        precision=str(raw.get("precision", "") or ""),
        hf_repo=str(raw.get("hf_repo", "") or ""),
        revision=str(raw.get("revision", "") or "main"),
        sha256=str(raw.get("sha256", "") or ""),
        runtime_contract=dict(raw.get("runtime_contract") or {}),
    )


def parse_manifest(data: dict[str, Any]) -> ArtifactManifest:
    """Validate and parse an in-memory manifest dict."""
    if not isinstance(data, dict):
        raise ManifestError(f"manifest must be a JSON object, got {type(data)!r}")
    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported schema_version {schema_version!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    artifacts_raw = data.get("artifacts")
    if not isinstance(artifacts_raw, dict) or not artifacts_raw:
        raise ManifestError("manifest must declare a non-empty 'artifacts' map")
    artifacts = {
        ref: _parse_spec(ref, spec) for ref, spec in artifacts_raw.items()
    }
    return ArtifactManifest(
        schema_version=schema_version,
        artifacts=artifacts,
        hf_endpoint=str(data.get("hf_endpoint", "") or ""),
    )


def load_manifest(manifest_path: str | Path) -> ArtifactManifest:
    """Load and validate a manifest from a local JSON file path."""
    path = Path(manifest_path)
    if not path.exists():
        raise ManifestError(f"manifest_path not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {path}: {exc}") from exc
    return parse_manifest(data)


def default_manifest_path() -> Path:
    """Path to the sample manifest bundled with the voxedge package."""
    return Path(__file__).with_name("manifest.json")
