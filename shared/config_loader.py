"""YAML config loader with required field validation."""
import yaml
from pathlib import Path

REQUIRED_FIELDS = ["dataset_id", "seed", "output_dir"]


def load_config(config_path):
    """Load a YAML config file and validate required fields.

    Required fields: dataset_id, seed, output_dir.
    Returns a dict.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    missing = [field for field in REQUIRED_FIELDS if field not in config]
    if missing:
        raise ValueError(
            f"Config {config_path} missing required fields: {', '.join(missing)}"
        )
    return config
