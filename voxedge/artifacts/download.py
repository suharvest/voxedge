"""Env-free, backend-agnostic artifact download + install helper.

Generalises the RK-specific ``ensure_rk_artifacts`` prototype
(``voxedge/backends/rk/artifacts.py``) into a backend-agnostic helper keyed by
a stable ``artifact_ref``. The public entry point is :func:`resolve_artifact`,
which:

  1. looks up the artifact by ``artifact_ref`` in the manifest,
  2. for each declared file, downloads it from the HF repo (or any injected
     downloader) into ``install_root/<path>``,
  3. verifies the SHA-256 of every file (already-present + correct files are
     skipped),
  4. fails loudly with :class:`ArtifactError` on a missing file or SHA mismatch
     (preflight semantics — e.g. a missing ``style.npy`` voice pack must fail,
     not silently produce silent audio),
  5. returns an :class:`ArtifactInstallResult` recording what was installed vs
     skipped.

**Env-free**: every input is a function argument; nothing reads ``os.environ``.
The install root is supplied by the caller (product / profile) — the manifest
only stores artifact-relative paths.

``huggingface_hub`` is an OPTIONAL dependency. The default downloader uses it,
but tests (and offline callers) inject a local downloader via the
``download_fn`` parameter, so the manifest/SHA logic needs no network.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .manifest import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactSpec,
    default_manifest_path,
    load_manifest,
)

DEFAULT_HF_ENDPOINT = "https://huggingface.co"

# A downloader fetches one file from an HF repo into ``dest``.
# Signature: (repo_id, revision, source_path, dest, endpoint) -> None
DownloadFn = Callable[[str, str, str, Path, str], None]


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be resolved, downloaded, or verified."""


@dataclass
class ArtifactInstallResult:
    """Outcome of :func:`resolve_artifact`."""

    artifact_ref: str
    install_root: Path
    installed: list[str] = field(default_factory=list)  # rel paths downloaded
    skipped: list[str] = field(default_factory=list)  # rel paths already present
    files: list[Path] = field(default_factory=list)  # absolute installed paths

    @property
    def ready(self) -> bool:
        """True when every declared file is present on disk (installed or
        already-present). ``resolve_artifact`` raises before returning if any
        file is missing/mismatched, so a returned result is always ready."""
        return bool(self.files) and not self.missing()

    @property
    def all_skipped(self) -> bool:
        """True when nothing needed downloading (every file already present)."""
        return bool(self.files) and not self.installed

    def missing(self) -> list[str]:  # pragma: no cover - resolve raises first
        return [str(p) for p in self.files if not p.exists()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hf_download(
    repo_id: str, revision: str, source_path: str, dest: Path, endpoint: str
) -> None:
    """Default downloader backed by ``huggingface_hub.hf_hub_download``.

    Imported lazily so the core package stays installable without the optional
    ``huggingface_hub`` dependency.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ArtifactError(
            "huggingface_hub is required to download artifacts; install "
            "voxedge[artifacts] or inject a custom download_fn."
        ) from exc

    import shutil

    src = hf_hub_download(
        repo_id=repo_id,
        filename=source_path,
        revision=revision or "main",
        endpoint=endpoint or DEFAULT_HF_ENDPOINT,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() != dest.resolve():
        shutil.copyfile(src, dest)


def _resolve_one(
    spec: ArtifactSpec,
    item: ArtifactFile,
    install_root: Path,
    repo_id: str,
    revision: str,
    endpoint: str,
    download_fn: DownloadFn,
    result: ArtifactInstallResult,
) -> None:
    dest = install_root / item.rel_path
    result.files.append(dest)

    # Already present and correct → skip.
    if dest.exists() and _sha256(dest) == item.sha256:
        result.skipped.append(item.rel_path)
        return

    if not repo_id:
        raise ArtifactError(
            f"artifact {spec.artifact_ref!r} file {item.rel_path!r} is missing "
            f"and no hf_repo is declared to download it from "
            f"(preflight fail: required file absent)."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_fn(repo_id, revision, item.source_rel, dest, endpoint)
    except ArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise any downloader failure
        raise ArtifactError(
            f"download failed for {spec.artifact_ref!r} file "
            f"{item.source_rel!r}: {exc}"
        ) from exc

    if not dest.exists():
        raise ArtifactError(
            f"download reported success but file missing: {dest} "
            f"(artifact {spec.artifact_ref!r})"
        )

    got = _sha256(dest)
    if got != item.sha256:
        dest.unlink(missing_ok=True)
        raise ArtifactError(
            f"sha256 mismatch for {item.rel_path!r} in artifact "
            f"{spec.artifact_ref!r}: got {got}, expected {item.sha256}"
        )
    result.installed.append(item.rel_path)


def resolve_artifact(
    artifact_ref: str,
    install_root: str | Path,
    *,
    manifest_path: Optional[str | Path] = None,
    manifest: Optional[ArtifactManifest] = None,
    hf_endpoint: str = DEFAULT_HF_ENDPOINT,
    download_fn: Optional[DownloadFn] = None,
) -> ArtifactInstallResult:
    """Resolve, download, SHA-verify and install a named artifact.

    Args:
        artifact_ref: stable artifact name to resolve in the manifest.
        install_root: caller-chosen base dir; files land at
            ``install_root/<file.path>`` (manifest stores relative paths only).
        manifest_path: path to a JSON manifest. Defaults to the bundled sample
            manifest when neither ``manifest_path`` nor ``manifest`` is given.
        manifest: a pre-parsed manifest (takes precedence over ``manifest_path``).
        hf_endpoint: HF endpoint base URL (env-free; passed by the caller).
        download_fn: per-file downloader override. Defaults to a
            ``huggingface_hub``-backed downloader. Tests inject a local one.

    Returns:
        :class:`ArtifactInstallResult` recording installed/skipped files.

    Raises:
        ArtifactError: artifact_ref missing, a required file is absent with no
            repo to fetch it, a download fails, or a SHA-256 mismatch occurs.
    """
    if manifest is None:
        path = manifest_path if manifest_path is not None else default_manifest_path()
        manifest = load_manifest(path)

    spec = manifest.get(artifact_ref)

    root = Path(install_root)
    endpoint = (hf_endpoint or manifest.hf_endpoint or DEFAULT_HF_ENDPOINT).rstrip("/")
    repo_id = spec.hf_repo
    revision = spec.revision or "main"
    dl = download_fn or _hf_download

    result = ArtifactInstallResult(artifact_ref=artifact_ref, install_root=root)
    for item in spec.file_list:
        _resolve_one(spec, item, root, repo_id, revision, endpoint, dl, result)
    return result
