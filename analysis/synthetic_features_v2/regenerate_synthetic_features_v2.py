#!/usr/bin/env python3
"""Regenerate authoritative local synthetic feature artifacts.

This script is analysis-only. It validates the local synthetic mask/manifest
pairing, runs the existing feature extractor into a new versioned output
directory, and writes provenance/integrity reports without modifying generator
geometry or legacy pulled feature tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "synthetic_features_v2"
MANIFEST_PATH = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "manifests" / "synthetic_lollipop_manifest.csv"
MASK_ROOT = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "masks"
EXTRACTOR_PATH = REPO_ROOT / "scripts" / "extract_real_tumor_features.py"
INPUT_CSV = OUT_DIR / "synthetic_features_v2_input.csv"
PREFLIGHT_CSV = OUT_DIR / "preflight_validation.csv"
FEATURE_CSV = OUT_DIR / "synthetic_features_v2.csv"
FEATURE_JSON = OUT_DIR / "synthetic_features_v2.json"
SUMMARY_JSON = OUT_DIR / "extraction_summary.json"
PROVENANCE_JSON = OUT_DIR / "provenance.json"
INTEGRITY_REPORT = OUT_DIR / "FEATURE_INTEGRITY_REPORT.md"
SCHEMA_VERSION = "synthetic_features_v2_current_extractor_2026-07-18"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"


def read_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    required = {"case_id", "realized_volume_mm3"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    manifest["case_id"] = manifest["case_id"].astype(str)
    manifest["local_mask_path"] = manifest["case_id"].map(lambda cid: str(MASK_ROOT / f"{cid}_synthetic_lollipop.nii.gz"))
    return manifest


def preflight(manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duplicate_case_ids = int(manifest["case_id"].duplicated().sum())
    mask_paths = sorted(MASK_ROOT.glob("*_synthetic_lollipop.nii.gz"))
    mask_case_ids = {path.name[: -len("_synthetic_lollipop.nii.gz")] for path in mask_paths}
    manifest_case_ids = set(manifest["case_id"])

    for record in manifest.to_dict("records"):
        case_id = str(record["case_id"])
        mask_path = Path(str(record["local_mask_path"]))
        row: dict[str, Any] = {
            "case_id": case_id,
            "mask_path": str(mask_path),
            "manifest_realized_volume_mm3": record.get("realized_volume_mm3", math.nan),
            "exists": mask_path.exists(),
            "readable": False,
            "spacing_valid": False,
            "spacing_x_mm": math.nan,
            "spacing_y_mm": math.nan,
            "spacing_z_mm": math.nan,
            "voxel_volume_mm3": math.nan,
            "mask_voxel_count": math.nan,
            "mask_volume_mm3": math.nan,
            "volume_abs_diff_mm3": math.nan,
            "volume_matches_manifest": False,
            "non_empty": False,
            "connected_component_count": math.nan,
            "largest_component_fraction": math.nan,
            "mask_sha256": "",
            "status": "ERROR",
            "failure_reason": "",
        }
        try:
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            nii = nib.load(str(mask_path))
            spacing = np.asarray(nii.header.get_zooms()[:3], dtype=float)
            row["readable"] = True
            row["spacing_x_mm"], row["spacing_y_mm"], row["spacing_z_mm"] = [float(x) for x in spacing]
            row["spacing_valid"] = bool(np.all(np.isfinite(spacing)) and np.all(spacing > 0.0))
            if not row["spacing_valid"]:
                raise ValueError(f"invalid spacing {spacing.tolist()}")
            voxel_volume = float(np.prod(spacing))
            mask = np.asarray(nii.dataobj) > 0
            mask_voxels = int(mask.sum())
            mask_volume = float(mask_voxels * voxel_volume)
            labeled, cc_count = ndi_label(mask)
            if cc_count:
                counts = np.bincount(labeled.ravel())
                counts[0] = 0
                largest_fraction = float(counts.max() / mask_voxels) if mask_voxels else math.nan
            else:
                largest_fraction = math.nan
            manifest_volume = float(record.get("realized_volume_mm3", math.nan))
            abs_diff = abs(mask_volume - manifest_volume)
            row.update(
                {
                    "voxel_volume_mm3": voxel_volume,
                    "mask_voxel_count": mask_voxels,
                    "mask_volume_mm3": mask_volume,
                    "volume_abs_diff_mm3": abs_diff,
                    "volume_matches_manifest": bool(np.isfinite(abs_diff) and abs_diff <= max(1e-6, voxel_volume / 2.0)),
                    "non_empty": bool(mask_voxels > 0),
                    "connected_component_count": int(cc_count),
                    "largest_component_fraction": largest_fraction,
                    "mask_sha256": sha256_file(mask_path),
                    "status": "OK",
                }
            )
        except Exception as exc:
            row["failure_reason"] = str(exc)
        rows.append(row)

    preflight_df = pd.DataFrame(rows)
    summary = {
        "manifest_rows": int(len(manifest)),
        "manifest_unique_case_ids": int(manifest["case_id"].nunique()),
        "duplicate_case_ids": duplicate_case_ids,
        "mask_files_found": int(len(mask_paths)),
        "manifest_without_mask": int(len(manifest_case_ids - mask_case_ids)),
        "mask_without_manifest": int(len(mask_case_ids - manifest_case_ids)),
        "preflight_ok": int((preflight_df["status"] == "OK").sum()),
        "readable": int(preflight_df["readable"].sum()),
        "spacing_valid": int(preflight_df["spacing_valid"].sum()),
        "non_empty": int(preflight_df["non_empty"].sum()),
        "volume_matches_manifest": int(preflight_df["volume_matches_manifest"].sum()),
        "one_to_one_case_matching": bool(
            duplicate_case_ids == 0
            and len(manifest_case_ids - mask_case_ids) == 0
            and len(mask_case_ids - manifest_case_ids) == 0
        ),
    }
    return preflight_df, summary


def write_input_csv(preflight_df: pd.DataFrame) -> None:
    ok = preflight_df[preflight_df["status"] == "OK"][["case_id", "mask_path"]].rename(columns={"mask_path": "seg_path"})
    ok.to_csv(INPUT_CSV, index=False)


def run_extractor() -> tuple[int, str]:
    command = [
        sys.executable,
        str(EXTRACTOR_PATH),
        "--input_csv",
        str(INPUT_CSV),
        "--seg_col",
        "seg_path",
        "--case_id_col",
        "case_id",
        "--out_csv",
        str(FEATURE_CSV),
        "--out_json",
        str(FEATURE_JSON),
        "--summary_json",
        str(SUMMARY_JSON),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return completed.returncode, completed.stdout


def validate_features(preflight_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = pd.read_csv(FEATURE_CSV)
    merged = features.merge(
        preflight_df[["case_id", "mask_path", "mask_sha256", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "mask_volume_mm3", "connected_component_count"]],
        on="case_id",
        how="outer",
        suffixes=("", "_preflight"),
        indicator=True,
        validate="one_to_one",
    )
    numeric_cols = [col for col in features.columns if col not in {"case_id", "seg_path", "failure_reason"}]
    numeric = features[numeric_cols].apply(pd.to_numeric, errors="coerce")
    inf_counts = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    failed = features["failed"].astype(str).str.lower().eq("true") if "failed" in features.columns else pd.Series(False, index=features.index)
    volume_diff = (pd.to_numeric(merged["volume_mm3"], errors="coerce") - pd.to_numeric(merged["mask_volume_mm3"], errors="coerce")).abs()
    impossible_checks = {
        "nonpositive_volume": int((pd.to_numeric(features["volume_mm3"], errors="coerce") <= 0).sum()),
        "nonpositive_spacing": int(
            (
                (pd.to_numeric(features["voxel_spacing_mm_x"], errors="coerce") <= 0)
                | (pd.to_numeric(features["voxel_spacing_mm_y"], errors="coerce") <= 0)
                | (pd.to_numeric(features["voxel_spacing_mm_z"], errors="coerce") <= 0)
            ).sum()
        ),
        "component_count_lt_one": int((pd.to_numeric(features["connected_component_count"], errors="coerce") < 1).sum()),
        "largest_fraction_out_of_range": int(
            (
                (pd.to_numeric(features["largest_component_fraction"], errors="coerce") <= 0)
                | (pd.to_numeric(features["largest_component_fraction"], errors="coerce") > 1)
            ).sum()
        ),
        "negative_surface_area": int((pd.to_numeric(features["surface_area_mm2"], errors="coerce") < 0).sum()),
        "negative_axis_length": int(
            (
                (pd.to_numeric(features["principal_axis_length_major_mm"], errors="coerce") < 0)
                | (pd.to_numeric(features["principal_axis_length_minor1_mm"], errors="coerce") < 0)
                | (pd.to_numeric(features["principal_axis_length_minor2_mm"], errors="coerce") < 0)
            ).sum()
        ),
    }
    integrity = {
        "feature_rows": int(len(features)),
        "feature_unique_case_ids": int(features["case_id"].nunique()),
        "duplicate_feature_case_ids": int(features["case_id"].duplicated().sum()),
        "failed_extractions": int(failed.sum()),
        "outer_join_left_only": int((merged["_merge"] == "left_only").sum()),
        "outer_join_right_only": int((merged["_merge"] == "right_only").sum()),
        "nan_values_in_numeric_columns": int(numeric.isna().sum().sum()),
        "inf_values_in_numeric_columns": inf_counts,
        "volume_max_abs_diff_mm3": float(volume_diff.max()) if len(volume_diff) else math.nan,
        "volume_mismatches_gt_1e_minus_6": int((volume_diff > 1e-6).sum()),
        "multiple_component_rows": int(features["multiple_components"].astype(str).str.lower().eq("true").sum()),
        "surface_area_failed_rows": int(features["surface_area_failed"].astype(str).str.lower().eq("true").sum()),
        "principal_axis_failed_rows": int(features["principal_axis_failed"].astype(str).str.lower().eq("true").sum()),
        **impossible_checks,
    }
    return features, integrity


def write_provenance(preflight_df: pd.DataFrame, preflight_summary: dict[str, Any], integrity: dict[str, Any], command_output: str) -> None:
    case_records = []
    for row in preflight_df.to_dict("records"):
        case_records.append(
            {
                "case_id": row["case_id"],
                "mask_path": row["mask_path"],
                "mask_sha256": row["mask_sha256"],
                "spacing_mm": [row["spacing_x_mm"], row["spacing_y_mm"], row["spacing_z_mm"]],
                "mask_volume_mm3": row["mask_volume_mm3"],
                "preflight_status": row["status"],
            }
        )
    command = [
        sys.executable,
        str(EXTRACTOR_PATH),
        "--input_csv",
        str(INPUT_CSV),
        "--seg_col",
        "seg_path",
        "--case_id_col",
        "case_id",
        "--out_csv",
        str(FEATURE_CSV),
        "--out_json",
        str(FEATURE_JSON),
        "--summary_json",
        str(SUMMARY_JSON),
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "extraction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": git_commit(),
        "extractor_script_path": str(EXTRACTOR_PATH.relative_to(REPO_ROOT)),
        "extractor_sha256": sha256_file(EXTRACTOR_PATH),
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "mask_root": str(MASK_ROOT.relative_to(REPO_ROOT)),
        "spacing_source": "NIfTI header zooms from each local mask",
        "command": " ".join(command),
        "number_expected_cases": preflight_summary["manifest_rows"],
        "number_processed": integrity["feature_rows"],
        "number_succeeded": integrity["feature_rows"] - integrity["failed_extractions"],
        "number_failed": integrity["failed_extractions"],
        "preflight_summary": preflight_summary,
        "integrity_summary": integrity,
        "extractor_stdout": command_output,
        "per_case": case_records,
    }
    PROVENANCE_JSON.write_text(json.dumps(payload, indent=2))


def write_report(preflight_summary: dict[str, Any], integrity: dict[str, Any]) -> None:
    material_failures = []
    if not preflight_summary["one_to_one_case_matching"]:
        material_failures.append("manifest/mask one-to-one matching failed")
    if preflight_summary["manifest_rows"] != preflight_summary["volume_matches_manifest"]:
        material_failures.append("not all mask volumes match manifest realized volumes")
    if integrity["failed_extractions"]:
        material_failures.append("extractor reported failed rows")
    if integrity["duplicate_feature_case_ids"]:
        material_failures.append("duplicate feature case IDs")
    if integrity["volume_mismatches_gt_1e_minus_6"]:
        material_failures.append("feature volume mismatch against direct mask volume")
    for key in (
        "nonpositive_volume",
        "nonpositive_spacing",
        "component_count_lt_one",
        "largest_fraction_out_of_range",
        "negative_surface_area",
        "negative_axis_length",
    ):
        if integrity[key]:
            material_failures.append(key)

    status = "PASS" if not material_failures else "FAIL"
    lines = [
        "# Synthetic Features V2 Integrity Report",
        "",
        f"Status: {status}",
        "",
        "## Authoritative Inputs",
        "",
        f"- Manifest: `{MANIFEST_PATH.relative_to(REPO_ROOT)}`",
        f"- Mask root: `{MASK_ROOT.relative_to(REPO_ROOT)}`",
        f"- Extractor: `{EXTRACTOR_PATH.relative_to(REPO_ROOT)}`",
        "- Legacy pulled synthetic feature CSV/JSON artifacts were not overwritten.",
        "",
        "## Preflight Summary",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    for key, value in preflight_summary.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Feature Integrity Summary", "", "| Check | Value |", "| --- | ---: |"])
    for key, value in integrity.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Regenerated features are authoritative for the current local synthetic masks if status is PASS.",
            "- This report validates engineering integrity and feature/mask consistency; it does not establish morphology realism.",
            "- Multiple-component rows are reported as integrity metadata. They are not automatically material failures because the current extractor records largest-component features while preserving full-mask counts.",
            "",
            "## Material Failures",
            "",
        ]
    )
    if material_failures:
        lines.extend([f"- {item}" for item in material_failures])
    else:
        lines.append("- None.")
    lines.append("")
    INTEGRITY_REPORT.write_text("\n".join(lines))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    preflight_df, preflight_summary = preflight(manifest)
    preflight_df.to_csv(PREFLIGHT_CSV, index=False)
    write_input_csv(preflight_df)

    if preflight_summary["preflight_ok"] == 0:
        raise RuntimeError("No masks passed preflight; refusing to run extraction.")

    rc, output = run_extractor()
    if rc != 0:
        print(output)
        return rc

    _, integrity = validate_features(preflight_df)
    write_provenance(preflight_df, preflight_summary, integrity, output)
    write_report(preflight_summary, integrity)

    print("Synthetic feature regeneration complete")
    print(f"  Expected cases: {preflight_summary['manifest_rows']}")
    print(f"  Processed: {integrity['feature_rows']}")
    print(f"  Failed: {integrity['failed_extractions']}")
    print(f"  CSV: {FEATURE_CSV}")
    print(f"  Integrity report: {INTEGRITY_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
