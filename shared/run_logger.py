"""Runtime logging for reproducible experiment runs."""
import json
from datetime import datetime
from pathlib import Path

from shared.provenance import freeze_environment as _freeze_environment
from shared.provenance import get_git_commit as _get_git_commit


def get_git_commit():
    """Return current git commit hash, or 'unknown' if not in a repo."""
    return _get_git_commit(unknown="unknown")


def freeze_environment(output_path):
    """Write pip freeze output to a file."""
    _freeze_environment(output_path)


def init_run(base_dir, project, experiment, config):
    """Create a run directory under code_runs/<project>/<experiment>/<runid>/.

    Logs git commit, dataset_id, seed, timestamp, and environment freeze.
    Returns the run directory path as a string.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = Path(base_dir) / "code_runs" / project / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "git_commit": get_git_commit(),
        "dataset_id": config.get("dataset_id", "unspecified"),
        "seed": config.get("seed", None),
        "timestamp": timestamp,
        "project": project,
        "experiment": experiment,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    freeze_environment(run_dir / "environment.txt")

    return str(run_dir)
