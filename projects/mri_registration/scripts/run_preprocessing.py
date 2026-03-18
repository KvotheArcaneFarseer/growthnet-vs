#!/usr/bin/env python
"""Entry point for the MRI registration preprocessing pipeline.

Usage:
    python run_preprocessing.py --config ../configs/default.yaml
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.run_logger import init_run


def main():
    parser = argparse.ArgumentParser(description="Run MRI preprocessing pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = init_run(
        base_dir=config["output_dir"],
        project="mri_registration",
        experiment="preprocessing",
        config=config,
    )
    print(f"Run directory: {run_dir}")

    from src.preprocessing import _apply_config, main as preprocess_main
    _apply_config(config)
    preprocess_main()


if __name__ == "__main__":
    main()
