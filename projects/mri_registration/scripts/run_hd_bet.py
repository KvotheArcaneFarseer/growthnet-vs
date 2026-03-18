#!/usr/bin/env python
"""Entry point for HD-BET brain extraction.

Usage:
    python run_hd_bet.py --config ../configs/default.yaml
    python run_hd_bet.py --config ../configs/default.yaml --meta_csv path/to/meta.csv --data_dir /data
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config_loader import load_config
from shared.run_logger import init_run


def main():
    parser = argparse.ArgumentParser(description="Run HD-BET brain extraction")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--meta_csv", default=None, help="Path to metadata CSV (overrides config)")
    parser.add_argument("--data_dir", default=None, help="Directory containing raw_studies/, hd_input/, hd_output/")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = init_run(
        base_dir=config["output_dir"],
        project="mri_registration",
        experiment="hd_bet",
        config=config,
    )
    print(f"Run directory: {run_dir}")

    from src.hd_bet import _apply_config, main as hd_bet_main
    _apply_config(config)
    hd_bet_main(meta_csv=args.meta_csv, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
