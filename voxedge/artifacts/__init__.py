"""Backend-agnostic runtime artifact manifest + verified download helper.

The schema and verification code are pure Python; ``huggingface_hub`` is an
optional dependency needed only for the default network downloader
(``voxedge[artifacts]``). Endpoint selection may honor ``HF_ENDPOINT``.

Public API::

    from voxedge.artifacts import (
        load_manifest, parse_manifest, default_manifest_path,
        resolve_artifact, ArtifactInstallResult,
        ManifestError, ArtifactError,
    )
"""

from __future__ import annotations

from .download import (
    DEFAULT_HF_ENDPOINT,
    ArtifactError,
    ArtifactInstallResult,
    resolve_artifact,
)
from .manifest import (
    SCHEMA_VERSION,
    ArtifactFile,
    ArtifactManifest,
    ArtifactSpec,
    ManifestError,
    default_manifest_path,
    load_manifest,
    parse_manifest,
)

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_HF_ENDPOINT",
    "ArtifactFile",
    "ArtifactSpec",
    "ArtifactManifest",
    "ArtifactInstallResult",
    "ManifestError",
    "ArtifactError",
    "parse_manifest",
    "load_manifest",
    "default_manifest_path",
    "resolve_artifact",
]
