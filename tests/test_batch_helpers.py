from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.run_batch_embedding import (
    _build_batch_summary,
    _build_failure_report,
    _error_case_row,
    _flatten_case_metrics,
    _load_case_rows,
    _optional_float,
    _optional_int,
    _sanitize_case_id,
    _write_summary_csv,
)


pytestmark = pytest.mark.fast


def test_sanitize_case_id_is_stable_and_filesystem_safe():
    assert _sanitize_case_id("  patient 126/visit:5  ") == "patient_126_visit_5"
    assert _sanitize_case_id("126_5_1607") == "126_5_1607"
    assert _sanitize_case_id("   ") == "case"


def test_load_case_rows_validates_required_manifest_columns(tmp_path: Path):
    manifest = tmp_path / "cases.csv"
    manifest.write_text("case_id,mri_path,seg_path,extra\nc1,/tmp/mri.nii.gz,/tmp/seg.nii.gz,x\n")

    rows = _load_case_rows(manifest, num_cases=None)

    assert rows == [
        {
            "case_id": "c1",
            "mri_path": "/tmp/mri.nii.gz",
            "seg_path": "/tmp/seg.nii.gz",
            "extra": "x",
        }
    ]


def test_load_case_rows_rejects_missing_manifest_columns(tmp_path: Path):
    manifest = tmp_path / "bad_cases.csv"
    manifest.write_text("case_id,mri_path\nc1,/tmp/mri.nii.gz\n")

    with pytest.raises(ValueError, match="missing .*seg_path"):
        _load_case_rows(manifest, num_cases=None)


def test_optional_numeric_parsers_treat_blank_as_missing():
    row = {"float_value": " 1.25 ", "int_value": " 7 ", "blank": " "}

    assert _optional_float(row, "float_value") == pytest.approx(1.25)
    assert _optional_int(row, "int_value") == 7
    assert _optional_float(row, "blank") is None
    assert _optional_int(row, "missing") is None


def test_flatten_case_metrics_preserves_report_schema(tmp_path: Path):
    metrics = {
        "target_volume_mm3": 125.0,
        "actual_volume_mm3": 124.0,
        "ravd": 0.008,
        "orientation_method": "late_dice",
        "orientation_confidence": 0.4,
        "orientation_score_margin": 0.2,
        "orientation_normalized_gap": 0.4,
        "warnings": ["axis warning"],
        "hard_failures": [],
    }

    row = _flatten_case_metrics(metrics, case_id="case-1", case_out_dir=tmp_path / "case-1")

    expected_fields = {
        "case_id",
        "case_out_dir",
        "status",
        "target_volume_mm3",
        "actual_volume_mm3",
        "ravd",
        "orientation_method",
        "orientation_confidence",
        "orientation_score_margin",
        "orientation_normalized_gap",
        "warning_count",
        "hard_failure_count",
        "warnings",
        "hard_failures",
    }
    assert expected_fields.issubset(row.keys())
    assert row["status"] == "completed"
    assert row["warning_count"] == 1
    assert json.loads(row["warnings"]) == ["axis warning"]


def test_batch_summary_and_failure_report_are_recoverable(tmp_path: Path):
    completed = {
        "case_id": "completed-low-confidence",
        "case_out_dir": str(tmp_path / "completed"),
        "status": "completed",
        "orientation_confidence": 0.01,
        "orientation_score_margin": 0.01,
        "orientation_method": "late_dice",
        "primary_axis_error_deg": 75.0,
        "ravd": 0.1,
        "centroid_offset_mm": 0.2,
        "retained_fraction": 0.99,
        "placed_to_seg_ratio": 0.8,
        "worst_clipping_fraction": 1.0,
        "strategy_agreement": False,
        "volume_converged": True,
        "warnings": ["low confidence"],
        "hard_failures": [],
        "strategy_results": [{"method": "late_dice"}],
    }
    exception = {
        "case_id": "failed",
        "case_out_dir": str(tmp_path / "failed"),
        "status": "exception",
        "exception_type": "RuntimeError",
        "exception_message": "boom",
        "warnings": [],
        "hard_failures": ["boom"],
    }

    summary = _build_batch_summary([completed, exception], input_csv=tmp_path / "cases.csv", out_dir=tmp_path)
    failures = _build_failure_report([completed], [exception])

    assert summary["total_cases"] == 2
    assert summary["completed_cases"] == 1
    assert summary["exception_count"] == 1
    assert summary["hard_failure_count"] == 1
    assert summary["strategy_disagreement_count"] == 1
    assert failures["worst_axis_error_cases"][0]["case_id"] == "completed-low-confidence"
    assert failures["exception_cases"][0]["exception_type"] == "RuntimeError"


def test_summary_csv_writes_union_schema(tmp_path: Path):
    out_csv = tmp_path / "summary.csv"

    _write_summary_csv(
        [
            {"case_id": "a", "status": "completed"},
            {"case_id": "b", "status": "exception", "exception_type": "ValueError"},
        ],
        out_csv,
    )

    with out_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case_id"] == "a"
    assert rows[1]["exception_type"] == "ValueError"


def test_error_case_row_has_interpretable_failure_fields(tmp_path: Path):
    row = _error_case_row("bad case", tmp_path / "bad_case", ValueError("invalid input"))

    assert row["status"] == "exception"
    assert row["exception_type"] == "ValueError"
    assert "invalid input" in row["exception_message"]
    assert json.loads(row["hard_failures"]) == ["invalid input"]
