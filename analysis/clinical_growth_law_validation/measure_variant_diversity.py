#!/usr/bin/env python3
"""Measure mask diversity for same-patient/timepoint longitudinal variants."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from shared.reporting import write_csv_rows, write_text


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    write_csv_rows(path, rows, fieldnames)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj) > 0


def measure_variant_diversity(metadata_csv: Path) -> list[dict[str, Any]]:
    rows = _read_csv(metadata_csv)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        patient_id = row.get("patient_id", "").strip()
        timepoint = row.get("timepoint", "").strip()
        variant_id = row.get("variant_id", "").strip()
        mask_path = row.get("mask_path", "").strip()
        if patient_id and timepoint and variant_id and mask_path:
            grouped.setdefault((patient_id, timepoint), []).append(row)

    results: list[dict[str, Any]] = []
    for (patient_id, timepoint), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        for left, right in combinations(sorted(group, key=lambda row: row.get("variant_id", "")), 2):
            left_path = Path(left["mask_path"])
            right_path = Path(right["mask_path"])
            result: dict[str, Any] = {
                "patient_id": patient_id,
                "timepoint": timepoint,
                "variant_id_a": left.get("variant_id", ""),
                "variant_id_b": right.get("variant_id", ""),
                "target_volume_mm3": left.get("target_volume_mm3", ""),
                "mask_path_a": str(left_path),
                "mask_path_b": str(right_path),
                "dice": np.nan,
                "voxel_count_a": np.nan,
                "voxel_count_b": np.nan,
                "absolute_voxel_count_difference": np.nan,
                "status": "FAILED",
                "failure_reason": "",
            }
            try:
                mask_a = _load_mask(left_path)
                mask_b = _load_mask(right_path)
                if mask_a.shape != mask_b.shape:
                    raise ValueError(f"mask shapes differ: {mask_a.shape} vs {mask_b.shape}")
                vox_a = int(mask_a.sum())
                vox_b = int(mask_b.sum())
                result.update(
                    {
                        "dice": dice_score(mask_a, mask_b),
                        "voxel_count_a": vox_a,
                        "voxel_count_b": vox_b,
                        "absolute_voxel_count_difference": abs(vox_a - vox_b),
                        "status": "OK",
                    }
                )
            except Exception as exc:
                result["failure_reason"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
    return results


def write_report(path: Path, rows: list[dict[str, Any]], csv_path: Path) -> None:
    ok_rows = [row for row in rows if row["status"] == "OK"]
    dice_values = [float(row["dice"]) for row in ok_rows]
    median_dice = float(np.median(dice_values)) if dice_values else float("nan")
    lines = [
        "# Longitudinal Variant Diversity Report",
        "",
        "## Scope",
        "",
        "This report measures same-patient/timepoint mask diversity between",
        "independently seeded variants. It is an engineering diversity check, not a",
        "clinical morphology validation.",
        "",
        "## Summary",
        "",
        f"- Variant pairs evaluated: {len(rows)}",
        f"- Successful pair comparisons: {len(ok_rows)}",
        f"- Median Dice among successful comparisons: {median_dice:.4f}" if ok_rows else "- Median Dice among successful comparisons: n/a",
        f"- Pairwise CSV: `{csv_path}`",
        "",
        "Interpretation guide:",
        "",
        "- Dice near 1.0 means variants are nearly identical in mask space.",
        "- Lower Dice means stronger spatial/shape diversity.",
        "- Dice alone does not prove anatomical realism.",
        "",
    ]
    write_text(path, "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure same-timepoint longitudinal variant mask diversity.")
    parser.add_argument("--metadata_csv", required=True, help="Longitudinal metadata.csv with variant_id and mask_path columns.")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_report", required=True)
    args = parser.parse_args()

    rows = measure_variant_diversity(Path(args.metadata_csv).expanduser().resolve())
    out_csv = Path(args.out_csv).expanduser().resolve()
    _write_csv(
        out_csv,
        rows,
        [
            "patient_id",
            "timepoint",
            "variant_id_a",
            "variant_id_b",
            "target_volume_mm3",
            "mask_path_a",
            "mask_path_b",
            "dice",
            "voxel_count_a",
            "voxel_count_b",
            "absolute_voxel_count_difference",
            "status",
            "failure_reason",
        ],
    )
    write_report(Path(args.out_report).expanduser().resolve(), rows, out_csv)
    print(f"Wrote {out_csv}")
    print(f"Wrote {Path(args.out_report).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
