from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()


def write_run_provenance(
    run_dir: str | Path,
    dataset_id: str,
    seed: Optional[int] = None,
    env_method: str = "auto",
) -> None:
    """
    Writes provenance files into a code run directory:
    1. git_commit.txt
    2. dataset_id.txt
    3. seed.txt (if provided)
    4. python_version.txt
    5. env_method.txt
    6. one environment freeze file (pip_freeze.txt or conda_env.yml or modules.txt)
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # git commit
    try:
        commit = _run(["git", "rev-parse", "HEAD"])
    except Exception:
        commit = "unknown"
    (run_dir / "git_commit.txt").write_text(commit + "\n")

    # dataset id and seed
    (run_dir / "dataset_id.txt").write_text(str(dataset_id) + "\n")
    if seed is not None:
        (run_dir / "seed.txt").write_text(str(seed) + "\n")

    # python version
    try:
        pyver = _run(["python", "--version"])
    except Exception:
        pyver = "unknown"
    (run_dir / "python_version.txt").write_text(pyver + "\n")

    # pick env method
    chosen = env_method.lower().strip()
    if chosen == "auto":
        if os.environ.get("CONDA_DEFAULT_ENV"):
            chosen = "conda"
        else:
            chosen = "pip"

    (run_dir / "env_method.txt").write_text(chosen + "\n")

    # freeze environment
    if chosen == "conda":
        try:
            yml = _run(["conda", "env", "export", "--no-builds"])
            (run_dir / "conda_env.yml").write_text(yml + "\n")
            return
        except Exception:
            chosen = "pip"
            (run_dir / "env_method.txt").write_text("pip\n")

    if chosen == "pip":
        try:
            freeze = _run(["python", "-m", "pip", "freeze"])
            (run_dir / "pip_freeze.txt").write_text(freeze + "\n")
        except Exception:
            (run_dir / "pip_freeze.txt").write_text("pip freeze failed\n")
        return

    if chosen == "hpc_modules":
        try:
            out = _run(["bash", "-lc", "module list 2>&1"])
            (run_dir / "modules.txt").write_text(out + "\n")
        except Exception:
            (run_dir / "modules.txt").write_text("module list failed or unavailable\n")
        return