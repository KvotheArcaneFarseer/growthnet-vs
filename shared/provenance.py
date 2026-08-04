"""Shared provenance helpers for GrowthNet entry points."""

from __future__ import annotations

import subprocess
import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(repo_root: str | Path | None = None, unknown: str = "UNKNOWN") -> str:
    """Return the current git commit hash, or ``unknown`` if unavailable."""
    command = ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(
            command,
            cwd=Path(repo_root) if repo_root is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return unknown


def freeze_environment(output_path: str | Path, python_executable: str = "python") -> None:
    """Write ``pip freeze`` output for the requested Python executable."""
    try:
        freeze = subprocess.check_output(
            [python_executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        freeze = "pip freeze unavailable"
    Path(output_path).write_text(freeze, encoding="utf-8")
