#!/usr/bin/env python3
"""Batch runner for synthetic VS embedding cases."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REQUIRED_CASE_OUTPUTS = (
    "embedding_metrics.json",
    "embedding_metrics.csv",
    "embedded_tumor_volume.nii.gz",
    "embedded_tumor_mask.nii.gz",
    "embedded_tumor_late_volume.nii.gz",
    "embedded_tumor_late_mask.nii.gz",
    "qc_embedding.png",
    "qc_embedding_late.png",
)


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
        "canal_growth_scale": metrics.get("canal_growth_scale"),
        "bulb_growth_scale": metrics.get("bulb_growth_scale"),
        "t0_volume_fraction_of_seg": metrics.get("t0_volume_fraction_of_seg"),
        "target_volume_mm3": metrics.get("target_volume_mm3", metrics.get("target_tumor_volume_mm3")),
        "volume_target_timepoint": metrics.get("volume_target_timepoint"),
        "volume_target_timepoint_index": metrics.get("volume_target_timepoint_index"),
        "volume_target_timepoint_day": metrics.get("volume_target_timepoint_day"),
        "target_timepoint_actual_volume_mm3": metrics.get("target_timepoint_actual_volume_mm3"),
        "target_timepoint_voxels": metrics.get("target_timepoint_voxels"),
        "actual_volume_mm3": metrics.get("actual_volume_mm3"),
        "ravd": metrics.get("ravd"),
        "volume_converged": metrics.get("volume_converged"),
        "volume_iterations": metrics.get("volume_iterations"),
        "volume_scale_final": metrics.get("volume_scale_final"),
        "seed": metrics.get("seed"),
        "orientation_method": metrics.get("orientation_method"),
        "orientation_confidence": metrics.get("orientation_confidence"),
        "orientation_score_margin": metrics.get("orientation_score_margin"),
        "orientation_normalized_gap": metrics.get("orientation_normalized_gap"),
        "orientation_low_confidence": metrics.get("orientation_low_confidence"),
        "primary_axis_error_deg": metrics.get("primary_axis_error_deg"),
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
        "canal_growth_scale": None,
        "bulb_growth_scale": None,
        "t0_volume_fraction_of_seg": None,
        "target_volume_mm3": None,
        "volume_target_timepoint": None,
        "volume_target_timepoint_index": None,
        "volume_target_timepoint_day": None,
        "target_timepoint_actual_volume_mm3": None,
        "target_timepoint_voxels": None,
        "actual_volume_mm3": None,
        "ravd": None,
        "volume_converged": None,
        "volume_iterations": None,
        "volume_scale_final": None,
        "seed": None,
        "orientation_method": None,
        "orientation_confidence": None,
        "orientation_score_margin": None,
        "orientation_normalized_gap": None,
        "orientation_low_confidence": None,
        "primary_axis_error_deg": None,
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


def _optional_float(row: dict[str, str], key: str) -> float | None:
    """Parse an optional float field from a batch CSV row."""
    raw = row.get(key)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return float(value)


def _optional_int(row: dict[str, str], key: str) -> int | None:
    """Parse an optional integer field from a batch CSV row."""
    raw = row.get(key)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    return int(value)


def _case_parameters(row: dict[str, str]) -> dict[str, Any]:
    """Parse batch CSV columns that control one embedding run."""
    canal_growth_scale = _optional_float(row, "canal_growth_scale")
    bulb_growth_scale = _optional_float(row, "bulb_growth_scale")
    t0_volume_fraction_of_seg = _optional_float(row, "t0_volume_fraction_of_seg")
    target_tumor_volume_mm3 = _optional_float(row, "target_tumor_volume_mm3")
    volume_target_timepoint = row.get("volume_target_timepoint", "").strip() or "first"
    volume_ravd_tolerance = _optional_float(row, "volume_ravd_tolerance")
    volume_max_iterations = _optional_int(row, "volume_max_iterations")
    return {
        "canal_growth_scale": 1.0 if canal_growth_scale is None else canal_growth_scale,
        "bulb_growth_scale": 1.0 if bulb_growth_scale is None else bulb_growth_scale,
        "t0_volume_fraction_of_seg": t0_volume_fraction_of_seg,
        "target_tumor_volume_mm3": target_tumor_volume_mm3,
        "volume_target_timepoint": volume_target_timepoint,
        "volume_ravd_tolerance": 0.05 if volume_ravd_tolerance is None else volume_ravd_tolerance,
        "volume_max_iterations": 10 if volume_max_iterations is None else volume_max_iterations,
    }


def _enrich_case_metrics(metrics: dict[str, Any], case_id: str, case_out_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    """Add batch-level metadata to a case metrics payload."""
    metrics["case_id"] = case_id
    metrics["case_out_dir"] = str(case_out_dir)
    metrics["status"] = "completed"
    metrics["canal_growth_scale"] = params["canal_growth_scale"]
    metrics["bulb_growth_scale"] = params["bulb_growth_scale"]
    metrics["t0_volume_fraction_of_seg"] = params["t0_volume_fraction_of_seg"]
    metrics["target_tumor_volume_mm3"] = params["target_tumor_volume_mm3"]
    metrics["volume_target_timepoint"] = params["volume_target_timepoint"]
    metrics["volume_ravd_tolerance"] = params["volume_ravd_tolerance"]
    metrics["volume_max_iterations"] = params["volume_max_iterations"]
    return metrics


def _load_resume_metrics(case_id: str, case_out_dir: Path, params: dict[str, Any]) -> dict[str, Any] | None:
    """Return existing case metrics when all required outputs are present and usable."""
    missing_outputs = [name for name in REQUIRED_CASE_OUTPUTS if not (case_out_dir / name).exists()]
    if missing_outputs:
        return None
    metrics_path = case_out_dir / "embedding_metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if metrics.get("hard_failures"):
        return None
    return _enrich_case_metrics(metrics, case_id=case_id, case_out_dir=case_out_dir, params=params)


def _write_status_event(status_path: Path, event: dict[str, Any]) -> None:
    """Append one durable case status event as JSON Lines."""
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        **event,
    }
    with status_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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
    cases_with_axis_error = [item for item in completed_cases if item.get("primary_axis_error_deg") is not None]
    worst_axis_error = sorted(
        cases_with_axis_error,
        key=lambda item: float(item.get("primary_axis_error_deg", float("-inf"))),
        reverse=True,
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
        "worst_axis_error_cases": [
            {
                "case_id": item["case_id"],
                "primary_axis_error_deg": item.get("primary_axis_error_deg"),
                "orientation_method": item.get("orientation_method"),
                "case_out_dir": item.get("case_out_dir"),
            }
            for item in worst_axis_error[:5]
            if item.get("primary_axis_error_deg") is not None
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
    axis_error_values = [float(item["primary_axis_error_deg"]) for item in completed if item.get("primary_axis_error_deg") is not None]
    ravd_values = [float(item["ravd"]) for item in completed if item.get("ravd") is not None]
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
        "primary_axis_error_deg": _summarize_numeric(axis_error_values),
        "ravd": _summarize_numeric(ravd_values),
        "volume_converged_count": int(sum(1 for item in completed if item.get("volume_converged") is True)),
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


def run_batch(input_csv: Path, out_dir: Path, num_cases: int | None = None, resume: bool = False) -> tuple[Path, Path, Path]:
    """Run the embedding pipeline across a CSV-defined batch and write summaries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    case_rows = _load_case_rows(input_csv, num_cases=num_cases)
    case_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    status_path = out_dir / "batch_case_status.jsonl"

    for index, row in enumerate(case_rows, start=1):
        case_id = row["case_id"].strip()
        case_dir_name = _sanitize_case_id(case_id)
        case_out_dir = out_dir / case_dir_name
        params = _case_parameters(row)
        if resume:
            existing_metrics = _load_resume_metrics(case_id=case_id, case_out_dir=case_out_dir, params=params)
            if existing_metrics is not None:
                print(f"[{index}/{len(case_rows)}] Skipping completed case `{case_id}` -> {case_out_dir}")
                case_results.append(existing_metrics)
                summary_rows.append(_flatten_case_metrics(existing_metrics, case_id=case_id, case_out_dir=case_out_dir))
                _write_status_event(
                    status_path,
                    {
                        "case_id": case_id,
                        "case_out_dir": str(case_out_dir),
                        "status": "skipped_existing",
                        "index": index,
                        "total_cases": len(case_rows),
                    },
                )
                continue
        print(f"[{index}/{len(case_rows)}] Running case `{case_id}` -> {case_out_dir}")
        _write_status_event(
            status_path,
            {
                "case_id": case_id,
                "case_out_dir": str(case_out_dir),
                "status": "started",
                "index": index,
                "total_cases": len(case_rows),
            },
        )
        try:
            from embed_tumor import main as run_embedding_case

            run_embedding_case(
                mri_path=Path(row["mri_path"]).expanduser(),
                seg_path=Path(row["seg_path"]).expanduser(),
                out_dir=case_out_dir,
                canal_growth_scale=params["canal_growth_scale"],
                bulb_growth_scale=params["bulb_growth_scale"],
                target_tumor_volume_mm3=params["target_tumor_volume_mm3"],
                volume_target_timepoint=params["volume_target_timepoint"],
                volume_ravd_tolerance=params["volume_ravd_tolerance"],
                volume_max_iterations=params["volume_max_iterations"],
                t0_volume_fraction_of_seg=params["t0_volume_fraction_of_seg"],
            )
            metrics_path = case_out_dir / "embedding_metrics.json"
            metrics = json.loads(metrics_path.read_text())
            metrics = _enrich_case_metrics(metrics, case_id=case_id, case_out_dir=case_out_dir, params=params)
            case_results.append(metrics)
            summary_rows.append(_flatten_case_metrics(metrics, case_id=case_id, case_out_dir=case_out_dir))
            _write_status_event(
                status_path,
                {
                    "case_id": case_id,
                    "case_out_dir": str(case_out_dir),
                    "status": "completed",
                    "index": index,
                    "total_cases": len(case_rows),
                    "warning_count": len(metrics.get("warnings", [])),
                    "hard_failure_count": len(metrics.get("hard_failures", [])),
                },
            )
        except Exception as exc:
            print(f"  [error] Case `{case_id}` failed: {exc}")
            traceback.print_exc()
            error_row = _error_case_row(case_id=case_id, case_out_dir=case_out_dir, exc=exc)
            case_results.append(
                {
                    "case_id": case_id,
                    "case_out_dir": str(case_out_dir),
                    "status": "exception",
                    "canal_growth_scale": None,
                    "bulb_growth_scale": None,
                    "t0_volume_fraction_of_seg": None,
                    "target_volume_mm3": None,
                    "volume_target_timepoint": None,
                    "volume_target_timepoint_index": None,
                    "volume_target_timepoint_day": None,
                    "target_timepoint_actual_volume_mm3": None,
                    "target_timepoint_voxels": None,
                    "actual_volume_mm3": None,
                    "ravd": None,
                    "volume_converged": None,
                    "volume_iterations": None,
                    "volume_scale_final": None,
                    "warnings": [],
                    "hard_failures": [str(exc)],
                    "strategy_agreement": None,
                    "orientation_confidence": None,
                    "primary_axis_error_deg": None,
                    "retained_fraction": None,
                    "centroid_offset_mm": None,
                    "placed_to_seg_ratio": None,
                    "worst_clipping_fraction": None,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            summary_rows.append(error_row)
            _write_status_event(
                status_path,
                {
                    "case_id": case_id,
                    "case_out_dir": str(case_out_dir),
                    "status": "exception",
                    "index": index,
                    "total_cases": len(case_rows),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )

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
    print(f"  Mean axis error       : {batch_summary['primary_axis_error_deg']['mean']} deg")
    print(f"  Mean RAVD             : {batch_summary['ravd']['mean']}")
    print(f"  Volume converged      : {batch_summary['volume_converged_count']}")
    print(f"  Mean centroid offset  : {batch_summary['centroid_offset_mm']['mean']} mm")
    print(f"  Mean retained fraction: {batch_summary['retained_fraction']['mean']}")
    print(f"  Strategy disagreements: {batch_summary['strategy_disagreement_count']}")
    print(f"  Wrote {summary_json_path}")
    print(f"  Wrote {summary_csv_path}")
    print(f"  Wrote {failure_json_path}")
    print(f"  Wrote {status_path}")

    return summary_json_path, summary_csv_path, failure_json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic VS embedding over a batch CSV.")
    parser.add_argument("--input_csv", required=True, help="CSV with columns case_id,mri_path,seg_path")
    parser.add_argument("--out_dir", required=True, help="Batch output directory")
    parser.add_argument("--num_cases", type=int, default=None, help="Optional cap on the number of cases to run")
    parser.add_argument("--resume", action="store_true", help="Skip cases with a complete existing output set")
    args = parser.parse_args()

    run_batch(
        input_csv=Path(args.input_csv).expanduser(),
        out_dir=Path(args.out_dir).expanduser(),
        num_cases=args.num_cases,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
