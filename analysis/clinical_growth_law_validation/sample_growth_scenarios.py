#!/usr/bin/env python3
"""Sample longitudinal growth scenarios without generating MRI/NIfTI outputs."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_synthetic_longitudinal_dataset import (  # noqa: E402
    DEFAULT_VISIT_DAYS,
    _target_volumes_for_patient,
)
from shared.reporting import write_csv_rows, write_text  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    write_csv_rows(path, rows, fieldnames)


def sample_scenarios(
    n_patients: int,
    baseline_volume_mm3: float,
    clinical_growth_law: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(n_patients):
        patient_id = f"SAMPLE_{index:05d}"
        patient = {
            "patient_id": patient_id,
            "background_mri_id": "NO_MRI_SAMPLING_ONLY",
            "T1_volume_mm3": str(float(baseline_volume_mm3)),
            "T2_volume_mm3": "nan",
            "T3_volume_mm3": "nan",
            "T4_volume_mm3": "nan",
            "growth_label": "growing",
        }
        visits = _target_volumes_for_patient(
            patient=patient,
            clinical_growth_law=clinical_growth_law,
            visit_days=DEFAULT_VISIT_DAYS,
            seed=seed,
        )
        scenario = str(visits[0]["growth_law_scenario"])
        annual_fraction = float(visits[0]["growth_law_annual_volume_change_fraction"])
        for visit in visits:
            rows.append(
                {
                    "patient_id": patient_id,
                    "timepoint": visit["timepoint"],
                    "visit_day": visit["visit_day"],
                    "baseline_volume_mm3": float(baseline_volume_mm3),
                    "clinical_growth_law": clinical_growth_law,
                    "growth_law_scenario": scenario,
                    "annual_volume_change_fraction": annual_fraction,
                    "target_volume_mm3": float(visit["target_volume_mm3"]),
                }
            )
    return rows


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> None:
    patient_scenarios = {
        row["patient_id"]: row["growth_law_scenario"]
        for row in rows
        if row["timepoint"] == "T1"
    }
    counts = Counter(patient_scenarios.values())
    t4_values_by_scenario: dict[str, list[float]] = {}
    for row in rows:
        if row["timepoint"] == "T4":
            t4_values_by_scenario.setdefault(str(row["growth_law_scenario"]), []).append(float(row["target_volume_mm3"]))

    lines = [
        "# Growth Scenario Sampling Audit",
        "",
        "## Scope",
        "",
        "This no-MRI audit samples target-volume trajectories only. It does not",
        "generate masks, paste tumors into MRI, or validate clinical realism.",
        "",
        "## Scenario Counts",
        "",
        "| Scenario | Count | T4 median mm3 | T4 min mm3 | T4 max mm3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario, count in sorted(counts.items()):
        values = t4_values_by_scenario.get(scenario, [])
        lines.append(
            f"| {scenario} | {count} | {np.median(values):.2f} | {np.min(values):.2f} | {np.max(values):.2f} |"
            if values
            else f"| {scenario} | {count} | n/a | n/a | n/a |"
        )
    lines.extend(["", f"CSV: `{csv_path}`", ""])
    write_text(path, "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample GrowthNet growth scenarios without generating images.")
    parser.add_argument("--n_patients", type=int, default=200)
    parser.add_argument("--baseline_volume_mm3", type=float, default=100.0)
    parser.add_argument("--clinical_growth_law", default="scenario_mixture_v1")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()
    if args.n_patients < 1:
        raise ValueError("--n_patients must be >= 1")
    if args.baseline_volume_mm3 <= 0.0:
        raise ValueError("--baseline_volume_mm3 must be > 0")

    rows = sample_scenarios(
        n_patients=int(args.n_patients),
        baseline_volume_mm3=float(args.baseline_volume_mm3),
        clinical_growth_law=str(args.clinical_growth_law),
        seed=int(args.seed),
    )
    out_csv = Path(args.out_csv).expanduser().resolve()
    _write_csv(
        out_csv,
        rows,
        [
            "patient_id",
            "timepoint",
            "visit_day",
            "baseline_volume_mm3",
            "clinical_growth_law",
            "growth_law_scenario",
            "annual_volume_change_fraction",
            "target_volume_mm3",
        ],
    )
    out_report = Path(args.out_report).expanduser().resolve()
    write_report(out_report, rows, out_csv)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
