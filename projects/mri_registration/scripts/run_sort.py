#!/usr/bin/env python
"""Entry point for post-registration QC sorting.

Usage:
    python run_sort.py --config ../configs/default.yaml --csv path/to/reviewed.csv
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.run_logger import init_run


def main():
    parser = argparse.ArgumentParser(description="Sort registrations by QC review")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--csv", required=True, help="Path to reviewed CSV")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = init_run(
        base_dir=config["output_dir"],
        project="mri_registration",
        experiment="qc_sort",
        config=config,
    )
    print(f"Run directory: {run_dir}")

    from src.sort_registrations import main as sort_main
    base_dir = config.get("output_base_dir")
    sort_main(base_dir=base_dir, csv_path=args.csv)


if __name__ == "__main__":
    main()
