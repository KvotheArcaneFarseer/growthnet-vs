#!/usr/bin/env python3
"""Batch runner for synthetic VS embedding cases."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embed_tumor import main as run_embedding_case  # noqa: E402


def _sanitize_case_id(case_id: str) -> str:
    """Return a filesystem-safe case identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id.strip())
    return cleaned or "case"


def _load_case_rows(input_csv: Path, num_cases: int | None) -> list[dict[str, str]]:
    """Load required batch case metadata from CSV."""
    with input_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "mri_path", "seg_path"}
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is missing a header row: {input_csv}")
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV must contain columns {sorted(required)}; missing {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if num_cases is not None:
        rows = rows[:num_cases]
    if not rows:
        raise ValueError(f"No case rows found in {input_csv}")
    return rows


def _flatten_case_metrics(metrics: dict[str, Any], case_id: str, case_out_dir: Path) -> dict[str, Any]:
    """Flatten one case metrics payload into a CSV-friendly row."""
    return {
        "case_id": case_id,
        "case_out_dir": str(case_out_dir),
        "status": "completed",
        "seed": metrics.get("seed"),
        "orientation_method": metrics.get("orientation_method"),
        "orientation_confidence": metrics.get("orientation_confidence"),
        "orientation_score_margin": metrics.get("orientation_score_margin"),
        "orientation_normalized_gap": metrics.get("orientation_normalized_gap"),
        "orientation_low_confidence": metrics.get("orientation_low_confidence"),
        "centroid_offset_mm": metrics.get("centroid_offset_mm"),
        "retained_fraction": metrics.get("retained_fraction"),
        "placed_to_seg_ratio": metrics.get("placed_to_seg_ratio"),
        "worst_clipping_fraction": metrics.get("worst_clipping_fraction"),
        "strategy_agreement": metrics.get("strategy_agreement"),
        "monotone_growth": metrics.get("monotone_growth"),
        "warning_count": len(metrics.get("warnings", [])),
        "hard_failure_count": len(metrics.get("hard_failures", [])),
        "warnings": json.dumps(metrics.get("warnings", [])),
        "hard_failures": json.dumps(metrics.get("hard_failures", [])),
    }


def _error_case_row(case_id: str, case_out_dir: Path, exc: Exception) -> dict[str, Any]:
    """Create a CSV-friendly row for a failed batch case."""
    return {
        "case_id": case_id,
        "case_out_dir": str(case_out_dir),
        "status": "exception",
        "seed": None,
        "orientation_method": None,
        "orientation_confidence": None,
        "orientation_score_margin": None,
        "orientation_normalized_gap": None,
        "orientation_low_confidence": None,
        "centroid_offset_mm": None,
        "retained_fraction": None,
        "placed_to_seg_ratio": None,
        "worst_clipping_fraction": None,
        "strategy_agreement": None,
        "monotone_growth": None,
        "warning_count": 0,
        "hard_failure_count": 1,
        "warnings": json.dumps([]),
        "hard_failures": json.dumps([str(exc)]),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


def _summarize_numeric(values: list[float]) -> dict[str, float | None]:
    """Return mean/min/max summary for a numeric list."""
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _placed_to_seg_distribution(values: list[float]) -> dict[str, float | None]:
    """Return a compact placed-to-seg ratio distribution summary."""
    if not values:
        return {"mean": None, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def _build_failure_report(completed_cases: list[dict[str, Any]], exception_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Build grouped lists for problem-case review."""
    low_conf = sorted(
        completed_cases,
        key=lambda item: float(item.get("orientation_confidence", float("inf"))),
    )
    worst_clip = sorted(
        completed_cases,
        key=lambda item: float(item.get("worst_clipping_fraction", float("inf"))),
    )
    return {
        "lowest_confidence_cases": [
            {
                "case_id": item["case_id"],
                "orientation_confidence": item.get("orientation_confidence"),
                "orientation_method": item.get("orientation_method"),
                "orientation_score_margin": item.get("orientation_score_margin"),
                "case_out_dir": item.get("case_out_dir"),
            }
            for item in low_conf[:5]
        ],
        "worst_clipping_cases": [
            {
                "case_id": item["case_id"],
                "worst_clipping_fraction": item.get("worst_clipping_fraction"),
                "retained_fraction": item.get("retained_fraction"),
                "case_out_dir": item.get("case_out_dir"),
            }
            for item in worst_clip[:5]
        ],
        "warning_cases": [
            {
                "case_id": item["case_id"],
                "warnings": item.get("warnings", []),
                "case_out_dir": item.get("case_out_dir"),
            }
            for item in completed_cases
            if item.get("warnings")
        ],
        "hard_failure_cases": [
            {
                "case_id": item["case_id"],
                "hard_failures": item.get("hard_failures", []),
                "case_out_dir": item.get("case_out_dir"),
            }
            for item in completed_cases
            if item.get("hard_failures")
        ],
        "strategy_disagreement_cases": [
            {
                "case_id": item["case_id"],
                "strategy_results": item.get("strategy_results", []),
                "case_out_dir": item.get("case_out_dir"),
            }
            for item in completed_cases
            if item.get("strategy_agreement") is False
        ],
        "exception_cases": [
            {
                "case_id": item["case_id"],
                "exception_type": item.get("exception_type"),
                "exception_message": item.get("exception_message"),
                "case_out_dir": item.get("case_out_dir"),
            }
            for item in exception_cases
        ],
    }


def _build_batch_summary(case_results: list[dict[str, Any]], input_csv: Path, out_dir: Path) -> dict[str, Any]:
    """Compute aggregate batch metrics from per-case results."""
    completed = [item for item in case_results if item["status"] == "completed"]
    exceptions = [item for item in case_results if item["status"] == "exception"]
    confidence_values = [float(item["orientation_confidence"]) for item in completed if item.get("orientation_confidence") is not None]
    centroid_values = [float(item["centroid_offset_mm"]) for item in completed if item.get("centroid_offset_mm") is not None]
    retained_values = [float(item["retained_fraction"]) for item in completed if item.get("retained_fraction") is not None]
    ratio_values = [float(item["placed_to_seg_ratio"]) for item in completed if item.get("placed_to_seg_ratio") is not None]
    clipping_cases = [item for item in completed if float(item.get("worst_clipping_fraction", 1.0)) < 0.999999]
    disagreement_cases = [item for item in completed if item.get("strategy_agreement") is False]
    warning_cases = [item for item in completed if item.get("warnings")]
    hard_failure_cases = [item for item in completed if item.get("hard_failures")]
    success_cases = [item for item in completed if not item.get("hard_failures")]

    return {
        "input_csv": str(input_csv),
        "out_dir": str(out_dir),
        "total_cases": len(case_results),
        "completed_cases": len(completed),
        "exception_count": len(exceptions),
        "success_count": len(success_cases),
        "warning_count": len(warning_cases),
        "hard_failure_count": len(hard_failure_cases) + len(exceptions),
        "clipping_case_count": len(clipping_cases),
        "strategy_disagreement_count": len(disagreement_cases),
        "orientation_confidence": _summarize_numeric(confidence_values),
        "centroid_offset_mm": _summarize_numeric(centroid_values),
        "retained_fraction": _summarize_numeric(retained_values),
        "placed_to_seg_ratio": _placed_to_seg_distribution(ratio_values),
        "clipping_frequency": (float(len(clipping_cases)) / float(len(completed))) if completed else None,
        "strategy_disagreement_frequency": (float(len(disagreement_cases)) / float(len(completed))) if completed else None,
    }


def _write_summary_csv(case_results: list[dict[str, Any]], csv_path: Path) -> None:
    """Write one flat summary row per batch case."""
    fieldnames: list[str] = []
    for row in case_results:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_results:
            writer.writerow(row)


def run_batch(input_csv: Path, out_dir: Path, num_cases: int | None = None) -> tuple[Path, Path, Path]:
    """Run the embedding pipeline across a CSV-defined batch and write summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows = _load_case_rows(input_csv, num_cases=num_cases)
    case_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for index, row in enumerate(case_rows, start=1):
        case_id = row["case_id"].strip()
        case_dir_name = _sanitize_case_id(case_id)
        case_out_dir = out_dir / case_dir_name
        print(f"[{index}/{len(case_rows)}] Running case `{case_id}` -> {case_out_dir}")
        try:
            run_embedding_case(
                mri_path=Path(row["mri_path"]).expanduser(),
                seg_path=Path(row["seg_path"]).expanduser(),
                out_dir=case_out_dir,
            )
            metrics_path = case_out_dir / "embedding_metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics["case_id"] = case_id
            metrics["case_out_dir"] = str(case_out_dir)
            metrics["status"] = "completed"
            case_results.append(metrics)
            summary_rows.append(_flatten_case_metrics(metrics, case_id=case_id, case_out_dir=case_out_dir))
        except Exception as exc:
            print(f"  [error] Case `{case_id}` failed: {exc}")
            traceback.print_exc()
            error_row = _error_case_row(case_id=case_id, case_out_dir=case_out_dir, exc=exc)
            case_results.append(
                {
                    "case_id": case_id,
                    "case_out_dir": str(case_out_dir),
                    "status": "exception",
                    "warnings": [],
                    "hard_failures": [str(exc)],
                    "strategy_agreement": None,
                    "orientation_confidence": None,
                    "retained_fraction": None,
                    "centroid_offset_mm": None,
                    "placed_to_seg_ratio": None,
                    "worst_clipping_fraction": None,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            summary_rows.append(error_row)

    batch_summary = _build_batch_summary(case_results, input_csv=input_csv, out_dir=out_dir)
    failure_report = _build_failure_report(
        completed_cases=[item for item in case_results if item["status"] == "completed"],
        exception_cases=[item for item in case_results if item["status"] == "exception"],
    )

    summary_json_path = out_dir / "batch_summary.json"
    summary_csv_path = out_dir / "batch_summary.csv"
    failure_json_path = out_dir / "failure_cases.json"

    summary_json_path.write_text(json.dumps(batch_summary, indent=2))
    _write_summary_csv(summary_rows, summary_csv_path)
    failure_json_path.write_text(json.dumps(failure_report, indent=2))

    print("\nBatch summary:")
    print(f"  Total cases           : {batch_summary['total_cases']}")
    print(f"  Success count         : {batch_summary['success_count']}")
    print(f"  Warning count         : {batch_summary['warning_count']}")
    print(f"  Hard failure count    : {batch_summary['hard_failure_count']}")
    print(f"  Mean orientation conf : {batch_summary['orientation_confidence']['mean']}")
    print(f"  Mean centroid offset  : {batch_summary['centroid_offset_mm']['mean']} mm")
    print(f"  Mean retained fraction: {batch_summary['retained_fraction']['mean']}")
    print(f"  Strategy disagreements: {batch_summary['strategy_disagreement_count']}")
    print(f"  Wrote {summary_json_path}")
    print(f"  Wrote {summary_csv_path}")
    print(f"  Wrote {failure_json_path}")

    return summary_json_path, summary_csv_path, failure_json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic VS embedding over a batch CSV.")
    parser.add_argument("--input_csv", required=True, help="CSV with columns case_id,mri_path,seg_path")
    parser.add_argument("--out_dir", required=True, help="Batch output directory")
    parser.add_argument("--num_cases", type=int, default=None, help="Optional cap on the number of cases to run")
    args = parser.parse_args()

    run_batch(
        input_csv=Path(args.input_csv).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        num_cases=args.num_cases,
    )


if __name__ == "__main__":
    main()
