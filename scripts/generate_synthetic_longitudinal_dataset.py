#!/usr/bin/env python3
"""Minimal longitudinal synthetic VS dataset wrapper.

This MVP keeps orchestration thin: each requested patient/timepoint is generated
by the existing per-case embedding engine, then copied into a longitudinal
folder layout with metadata and basic mask QC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embed_tumor import main as run_embedding_case  # noqa: E402
from scripts.extract_real_tumor_features import _extract_one  # noqa: E402


TIMELINE_COLUMNS = (
    "patient_id",
    "background_mri_id",
    "T1_volume_mm3",
    "T2_volume_mm3",
    "T3_volume_mm3",
    "T4_volume_mm3",
    "growth_label",
)
BACKGROUND_COLUMNS = ("background_mri_id", "mri_path", "seg_path")
TIMEPOINT_COLUMNS = ("T1_volume_mm3", "T2_volume_mm3", "T3_volume_mm3", "T4_volume_mm3")


def _safe_id(value: str) -> str:
    """Return a filesystem-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned or "item"


def _stable_seed(base_seed: int, patient_id: str, timepoint: str) -> int:
    """Return a deterministic per-patient/timepoint seed."""
    digest = hashlib.sha256(f"{base_seed}:{patient_id}:{timepoint}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _read_csv_rows(path: Path, required: Iterable[str]) -> List[Dict[str, str]]:
    """Read a CSV and validate required columns."""
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is missing a header row: {path}")
        missing = set(required).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _load_backgrounds(path: Path) -> Dict[str, Tuple[Path, Path]]:
    """Load background MRI/segmentation paths keyed by background_mri_id."""
    rows = _read_csv_rows(path, BACKGROUND_COLUMNS)
    backgrounds: Dict[str, Tuple[Path, Path]] = {}
    for row in rows:
        bg_id = row["background_mri_id"].strip()
        if not bg_id:
            raise ValueError(f"Blank background_mri_id in {path}")
        if bg_id in backgrounds:
            raise ValueError(f"Duplicate background_mri_id in {path}: {bg_id}")
        backgrounds[bg_id] = (
            Path(row["mri_path"]).expanduser().resolve(),
            Path(row["seg_path"]).expanduser().resolve(),
        )
    return backgrounds


def _growth_mode(growth_label: str) -> str:
    """Map dataset-level growth labels to existing engine growth modes."""
    label = growth_label.strip().lower()
    if label == "stable":
        return "stable"
    if label == "growing":
        return "steady"
    raise ValueError(f"Unsupported growth_label {growth_label!r}; expected stable or growing")


def _qc_mask(mask_path: Path, target_volume_mm3: float, tolerance: float) -> Dict[str, Any]:
    """Run minimal per-mask QC for the longitudinal MVP."""
    feature = _extract_one(
        seg_path=mask_path,
        case_id=mask_path.name,
        reference_axis_vox=None,
        reference_axis_mm=None,
    )
    synthetic_volume_mm3 = float(feature["volume_mm3"])
    connected_components = int(feature["connected_component_count"])
    relative_volume_error = (synthetic_volume_mm3 - target_volume_mm3) / max(target_volume_mm3, 1e-8)

    failure_reasons: List[str] = []
    if bool(feature["empty_mask"]):
        failure_reasons.append("empty_mask")
    if connected_components != 1:
        failure_reasons.append(f"connected_components={connected_components}")
    if abs(relative_volume_error) > tolerance:
        failure_reasons.append(f"relative_volume_error>{tolerance:g}")

    # TODO: Add longitudinal temporal QC after the MVP proves output plumbing:
    # monotonicity for growing cases, within-patient centroid drift, axis drift,
    # and intensity consistency across same-background timepoints.
    return {
        "synthetic_volume_mm3": synthetic_volume_mm3,
        "relative_volume_error": relative_volume_error,
        "connected_components": connected_components,
        "qc_pass": not failure_reasons,
        "qc_failure_reason": ";".join(failure_reasons),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_longitudinal_dataset(
    timeline_csv: Path,
    background_csv: Path,
    out_dir: Path,
    seed: int = 20260523,
    volume_ravd_tolerance: float = 0.05,
    volume_max_iterations: int = 10,
    gen_size: int = 128,
) -> Tuple[Path, Path]:
    """Generate a simple 4-timepoint longitudinal synthetic dataset."""
    timeline_rows = _read_csv_rows(timeline_csv, TIMELINE_COLUMNS)
    backgrounds = _load_backgrounds(background_csv)

    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []

    for patient_index, patient in enumerate(timeline_rows, start=1):
        patient_id = patient["patient_id"].strip()
        background_mri_id = patient["background_mri_id"].strip()
        growth_label = patient["growth_label"].strip()
        print(f"[{patient_index}/{len(timeline_rows)}] Patient {patient_id} on background {background_mri_id}")

        try:
            mri_path, seg_path = backgrounds[background_mri_id]
            if not mri_path.is_file():
                raise FileNotFoundError(f"MRI path does not exist for {background_mri_id}: {mri_path}")
            if not seg_path.is_file():
                raise FileNotFoundError(f"Segmentation path does not exist for {background_mri_id}: {seg_path}")
            growth = _growth_mode(growth_label)
        except Exception as exc:
            reason = str(exc)
            print(f"  [error] {reason}")
            for tp_idx, volume_col in enumerate(TIMEPOINT_COLUMNS, start=1):
                timepoint = f"T{tp_idx}"
                target = float(patient[volume_col])
                metadata_rows.append(
                    {
                        "patient_id": patient_id,
                        "timepoint": timepoint,
                        "background_mri_id": background_mri_id,
                        "growth_label": growth_label,
                        "target_volume_mm3": target,
                        "image_path": "",
                        "mask_path": "",
                    }
                )
                qc_rows.append(
                    {
                        "patient_id": patient_id,
                        "timepoint": timepoint,
                        "target_volume_mm3": target,
                        "synthetic_volume_mm3": math.nan,
                        "relative_volume_error": math.nan,
                        "connected_components": math.nan,
                        "qc_pass": False,
                        "qc_failure_reason": reason,
                    }
                )
            continue

        for tp_idx, volume_col in enumerate(TIMEPOINT_COLUMNS, start=1):
            timepoint = f"T{tp_idx}"
            target_volume_mm3 = float(patient[volume_col])
            case_token = f"{_safe_id(patient_id)}_{timepoint}"
            image_path = images_dir / f"{case_token}_image.nii.gz"
            mask_path = masks_dir / f"{case_token}_mask.nii.gz"

            metadata_rows.append(
                {
                    "patient_id": patient_id,
                    "timepoint": timepoint,
                    "background_mri_id": background_mri_id,
                    "growth_label": growth_label,
                    "target_volume_mm3": target_volume_mm3,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                }
            )

            try:
                with tempfile.TemporaryDirectory(prefix=f"{case_token}_", dir=str(out_dir)) as tmp_name:
                    tmp_out = Path(tmp_name)
                    run_embedding_case(
                        mri_path=mri_path,
                        seg_path=seg_path,
                        out_dir=tmp_out,
                        gen_size=gen_size,
                        # Use a tiny two-point internal series so the existing
                        # engine optimizes the first-timepoint init_scale path.
                        # The MVP exports only embedded_tumor_* for this visit.
                        dates=[0, 1],
                        growth=growth,
                        seed=_stable_seed(seed, patient_id, timepoint),
                        target_tumor_volume_mm3=target_volume_mm3,
                        volume_target_timepoint="first",
                        volume_ravd_tolerance=volume_ravd_tolerance,
                        volume_max_iterations=volume_max_iterations,
                    )
                    shutil.copy2(tmp_out / "embedded_tumor_volume.nii.gz", image_path)
                    shutil.copy2(tmp_out / "embedded_tumor_mask.nii.gz", mask_path)

                qc = _qc_mask(mask_path=mask_path, target_volume_mm3=target_volume_mm3, tolerance=volume_ravd_tolerance)
                qc_rows.append(
                    {
                        "patient_id": patient_id,
                        "timepoint": timepoint,
                        "target_volume_mm3": target_volume_mm3,
                        **qc,
                    }
                )
                status = "PASS" if qc["qc_pass"] else f"FAIL {qc['qc_failure_reason']}"
                print(
                    f"  {timepoint}: target={target_volume_mm3:.1f} "
                    f"synthetic={qc['synthetic_volume_mm3']:.1f} "
                    f"rel_err={qc['relative_volume_error']:.4f} {status}"
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                print(f"  [error] {timepoint} failed: {reason}")
                traceback.print_exc()
                qc_rows.append(
                    {
                        "patient_id": patient_id,
                        "timepoint": timepoint,
                        "target_volume_mm3": target_volume_mm3,
                        "synthetic_volume_mm3": math.nan,
                        "relative_volume_error": math.nan,
                        "connected_components": math.nan,
                        "qc_pass": False,
                        "qc_failure_reason": reason,
                    }
                )

    metadata_path = out_dir / "metadata.csv"
    qc_summary_path = out_dir / "qc_summary.csv"
    _write_csv(
        metadata_path,
        metadata_rows,
        [
            "patient_id",
            "timepoint",
            "background_mri_id",
            "growth_label",
            "target_volume_mm3",
            "image_path",
            "mask_path",
        ],
    )
    _write_csv(
        qc_summary_path,
        qc_rows,
        [
            "patient_id",
            "timepoint",
            "target_volume_mm3",
            "synthetic_volume_mm3",
            "relative_volume_error",
            "connected_components",
            "qc_pass",
            "qc_failure_reason",
        ],
    )
    print(f"\nWrote {metadata_path}")
    print(f"Wrote {qc_summary_path}")
    return metadata_path, qc_summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a minimal longitudinal synthetic VS dataset.")
    parser.add_argument(
        "--timeline_csv",
        default="data/timelines/test_longitudinal_timeline.csv",
        help="CSV with patient_id/background_mri_id/T1..T4/growth_label columns.",
    )
    parser.add_argument(
        "--background_csv",
        required=True,
        help="CSV with columns background_mri_id,mri_path,seg_path.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/synthetic_longitudinal_mvp",
        help="Output directory for images, masks, metadata.csv, and qc_summary.csv.",
    )
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--volume_ravd_tolerance", type=float, default=0.05)
    parser.add_argument("--volume_max_iterations", type=int, default=10)
    parser.add_argument("--gen_size", type=int, default=128)
    args = parser.parse_args()

    generate_longitudinal_dataset(
        timeline_csv=Path(args.timeline_csv).expanduser().resolve(),
        background_csv=Path(args.background_csv).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        seed=int(args.seed),
        volume_ravd_tolerance=float(args.volume_ravd_tolerance),
        volume_max_iterations=int(args.volume_max_iterations),
        gen_size=int(args.gen_size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
