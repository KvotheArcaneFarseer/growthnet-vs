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
import json
import math
import platform
import re
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embed_tumor import main as run_embedding_case  # noqa: E402
from scripts.extract_real_tumor_features import _extract_one  # noqa: E402
from shared.provenance import get_git_commit, sha256_file  # noqa: E402
from shared.reporting import write_csv_rows  # noqa: E402


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
DEFAULT_VISIT_DAYS = (0.0, 365.25, 730.5, 1095.75)
CLINICAL_GROWTH_LAWS = ("none", "empirical_vs_v1", "empirical_vs_v2", "scenario_mixture_v1")


def _safe_id(value: str) -> str:
    """Return a filesystem-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned or "item"


def _stable_seed(base_seed: int, patient_id: str, timepoint: str) -> int:
    """Return a deterministic per-patient/timepoint seed."""
    digest = hashlib.sha256(f"{base_seed}:{patient_id}:{timepoint}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    return sha256_file(path, chunk_size=chunk_size)


def _git_commit() -> str:
    return get_git_commit(REPO_ROOT, unknown="UNKNOWN")


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


def _parse_visit_days(raw: str) -> Tuple[float, float, float, float]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--visit_days must contain four comma-separated day values")
    days = tuple(float(part) for part in parts)
    if any(day < 0.0 for day in days):
        raise ValueError("--visit_days values must be non-negative")
    if any(next_day <= day for day, next_day in zip(days, days[1:])):
        raise ValueError("--visit_days values must be strictly increasing")
    return days  # type: ignore[return-value]


def _empirical_vs_annual_rate(
    growth_label: str,
    rng: np.random.Generator,
    clinical_growth_law: str = "empirical_vs_v1",
) -> Tuple[float, str]:
    """Sample annual volumetric change fraction for an experimental VS law.

    ``empirical_vs_v1`` preserves the original broad-bin experimental sampler.
    ``empirical_vs_v2`` is the preferred local candidate: growing tumors use a
    log-normal annual volume-change distribution calibrated to published
    untreated VS volumetric growth summaries, then clipped to avoid turning a
    dataset generator into an extreme-case simulator.
    """
    if clinical_growth_law not in CLINICAL_GROWTH_LAWS or clinical_growth_law == "none":
        raise ValueError(f"Unsupported empirical clinical_growth_law {clinical_growth_law!r}")
    label = growth_label.strip().lower()
    if label == "stable":
        return float(rng.uniform(-0.05, 0.05)), "stable"
    if label == "growing":
        if clinical_growth_law == "scenario_mixture_v1":
            scenario = str(
                rng.choice(
                    ["slow_growth", "moderate_growth", "fast_growth", "regression"],
                    p=[0.25, 0.45, 0.20, 0.10],
                )
            )
            if scenario == "slow_growth":
                return float(rng.uniform(0.05, 0.20)), scenario
            if scenario == "moderate_growth":
                median = 0.50
                q25 = 0.24
                q75 = 0.93
                sigma = (math.log(q75) - math.log(q25)) / (2.0 * 0.67448975)
                sampled = float(rng.lognormal(mean=math.log(median), sigma=sigma))
                return float(np.clip(sampled, 0.20, 1.20)), scenario
            if scenario == "fast_growth":
                return float(rng.uniform(1.0, 2.0)), scenario
            return float(rng.uniform(-0.20, -0.02)), scenario
        if clinical_growth_law == "empirical_vs_v2":
            median = 0.50
            q25 = 0.24
            q75 = 0.93
            sigma = (math.log(q75) - math.log(q25)) / (2.0 * 0.67448975)
            sampled = float(rng.lognormal(mean=math.log(median), sigma=sigma))
            return float(np.clip(sampled, 0.20, 1.20)), "moderate_growth"
        # In the cited untreated VS volumetric cohort, about 30% of all tumors
        # were fast-growing and 66% grew overall, so fast growth is roughly 45%
        # conditional on being in a growing category.
        if float(rng.uniform()) < 0.45:
            return float(rng.uniform(1.0, 2.0)), "fast_growth"
        return float(rng.uniform(0.20, 1.0)), "moderate_growth"
    raise ValueError(f"Unsupported growth_label {growth_label!r}; expected stable or growing")


def _target_volumes_for_patient(
    patient: Dict[str, str],
    clinical_growth_law: str,
    visit_days: Tuple[float, float, float, float],
    seed: int,
) -> List[Dict[str, Any]]:
    if clinical_growth_law not in CLINICAL_GROWTH_LAWS:
        raise ValueError(f"Unsupported clinical_growth_law {clinical_growth_law!r}")

    if clinical_growth_law == "none":
        return [
            {
                "timepoint": f"T{tp_idx}",
                "volume_col": volume_col,
                "visit_day": float(visit_days[tp_idx - 1]),
                "target_volume_mm3": float(patient[volume_col]),
                "target_volume_source": "timeline_csv",
                "growth_law_name": "none",
                "growth_law_scenario": "timeline_csv",
                "growth_law_annual_volume_change_fraction": math.nan,
            }
            for tp_idx, volume_col in enumerate(TIMEPOINT_COLUMNS, start=1)
        ]

    baseline = float(patient["T1_volume_mm3"])
    if baseline <= 0.0:
        raise ValueError(f"Baseline T1_volume_mm3 must be > 0 for clinical_growth_law={clinical_growth_law}")
    rng = np.random.default_rng(_stable_seed(seed, patient["patient_id"].strip(), "clinical_growth_law"))
    annual_rate, scenario = _empirical_vs_annual_rate(
        patient["growth_label"],
        rng,
        clinical_growth_law=clinical_growth_law,
    )
    log_rate = math.log1p(annual_rate)
    visits: List[Dict[str, Any]] = []
    for tp_idx, volume_col in enumerate(TIMEPOINT_COLUMNS, start=1):
        years = float(visit_days[tp_idx - 1]) / 365.25
        target = baseline * math.exp(log_rate * years)
        visits.append(
            {
                "timepoint": f"T{tp_idx}",
                "volume_col": volume_col,
                "visit_day": float(visit_days[tp_idx - 1]),
                "target_volume_mm3": float(target),
                "target_volume_source": "clinical_growth_law",
                "growth_law_name": clinical_growth_law,
                "growth_law_scenario": scenario,
                "growth_law_annual_volume_change_fraction": float(annual_rate),
            }
        )
    return visits


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
    write_csv_rows(path, rows, fieldnames)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def _is_non_decreasing(values: List[float], tolerance: float = 1e-8) -> bool:
    return all(next_value + tolerance >= value for value, next_value in zip(values, values[1:]))


def _growth_scenario_allows_decrease(scenarios: str) -> bool:
    scenario_set = {item.strip().lower() for item in str(scenarios).split(";") if item.strip()}
    return bool(scenario_set.intersection({"regression", "stable", "timeline_csv"}))


def _evaluate_longitudinal_qc_gate(summary: Dict[str, Any], volume_ravd_tolerance: float) -> Dict[str, str]:
    """Evaluate engineering readiness gates without making scientific validity claims."""
    failure_reasons: List[str] = []

    if not _truthy(summary.get("background_consistent")):
        failure_reasons.append("background_inconsistent")
    if not _truthy(summary.get("visit_days_strictly_increasing")):
        failure_reasons.append("visit_days_not_strictly_increasing")

    qc_fail_count = int(_finite_float(summary.get("qc_fail_count")) or 0)
    if qc_fail_count > 0:
        failure_reasons.append("qc_fail_count>0")

    max_error = _finite_float(summary.get("max_abs_relative_volume_error"))
    if max_error is None:
        failure_reasons.append("max_abs_relative_volume_error_missing")
    elif max_error > volume_ravd_tolerance:
        failure_reasons.append(f"max_abs_relative_volume_error>{volume_ravd_tolerance:g}")

    scenarios = str(summary.get("growth_law_scenarios", ""))
    target_monotone = _truthy(summary.get("target_volume_monotone_non_decreasing"))
    if target_monotone:
        target_status = "PASS"
    elif _growth_scenario_allows_decrease(scenarios):
        target_status = "ALLOWED_BY_SCENARIO"
    else:
        target_status = "WARNING_NONMONOTONE_TARGET"

    actual_monotone_value = summary.get("actual_volume_monotone_non_decreasing_all_variants")
    if actual_monotone_value == "":
        actual_status = "NOT_EVALUATED"
    elif _truthy(actual_monotone_value):
        actual_status = "PASS"
    else:
        actual_status = "WARNING_NONMONOTONE_ACTUAL"

    return {
        "engineering_qc_gate": "FAIL" if failure_reasons else "PASS",
        "engineering_qc_failure_reasons": ";".join(failure_reasons),
        "target_volume_trend_status": target_status,
        "actual_volume_trend_status": actual_status,
    }


def _summarize_longitudinal_qc(
    metadata_rows: List[Dict[str, Any]],
    qc_rows: List[Dict[str, Any]],
    volume_ravd_tolerance: float = 0.05,
) -> List[Dict[str, Any]]:
    """Build conservative patient-level QC summaries from metadata and mask QC."""
    patient_ids = sorted({str(row.get("patient_id", "")).strip() for row in metadata_rows if row.get("patient_id")})
    qc_by_patient: Dict[str, List[Dict[str, Any]]] = {}
    for row in qc_rows:
        qc_by_patient.setdefault(str(row.get("patient_id", "")).strip(), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for patient_id in patient_ids:
        patient_meta = [row for row in metadata_rows if str(row.get("patient_id", "")).strip() == patient_id]
        patient_qc = qc_by_patient.get(patient_id, [])
        backgrounds = sorted({str(row.get("background_mri_id", "")).strip() for row in patient_meta if row.get("background_mri_id")})
        scenarios = sorted({str(row.get("growth_law_scenario", "")).strip() for row in patient_meta if row.get("growth_law_scenario")})
        variants = sorted({str(row.get("variant_id", "")).strip() for row in patient_meta if row.get("variant_id")})

        timepoint_targets: Dict[str, tuple[float, float]] = {}
        for row in patient_meta:
            timepoint = str(row.get("timepoint", "")).strip()
            visit_day = _finite_float(row.get("visit_day"))
            target = _finite_float(row.get("target_volume_mm3"))
            if timepoint and visit_day is not None and target is not None and timepoint not in timepoint_targets:
                timepoint_targets[timepoint] = (visit_day, target)
        ordered_targets = [item for _, item in sorted(timepoint_targets.items(), key=lambda entry: entry[1][0])]
        visit_days = [item[0] for item in ordered_targets]
        target_volumes = [item[1] for item in ordered_targets]

        variant_actual_monotone: List[bool] = []
        for variant_id in variants:
            rows_for_variant = [
                row
                for row in patient_qc
                if str(row.get("variant_id", "")).strip() == variant_id
            ]
            actuals_by_timepoint: Dict[str, tuple[float, float]] = {}
            for row in rows_for_variant:
                timepoint = str(row.get("timepoint", "")).strip()
                meta_match = next(
                    (
                        meta
                        for meta in patient_meta
                        if str(meta.get("timepoint", "")).strip() == timepoint
                        and str(meta.get("variant_id", "")).strip() == variant_id
                    ),
                    None,
                )
                if meta_match is None:
                    continue
                visit_day = _finite_float(meta_match.get("visit_day"))
                actual = _finite_float(row.get("synthetic_volume_mm3"))
                if visit_day is not None and actual is not None:
                    actuals_by_timepoint[timepoint] = (visit_day, actual)
            ordered_actuals = [item[1] for _, item in sorted(actuals_by_timepoint.items(), key=lambda entry: entry[1][0])]
            if len(ordered_actuals) >= 2:
                variant_actual_monotone.append(_is_non_decreasing(ordered_actuals))

        relative_errors = [
            abs(number)
            for number in (_finite_float(row.get("relative_volume_error")) for row in patient_qc)
            if number is not None
        ]
        qc_pass_count = sum(1 for row in patient_qc if _truthy(row.get("qc_pass")))
        qc_fail_count = len(patient_qc) - qc_pass_count
        summary = {
            "patient_id": patient_id,
            "timepoint_count": len(timepoint_targets),
            "variant_count": len(variants),
            "metadata_row_count": len(patient_meta),
            "qc_row_count": len(patient_qc),
            "background_consistent": len(backgrounds) <= 1,
            "background_mri_ids": ";".join(backgrounds),
            "growth_law_scenarios": ";".join(scenarios),
            "visit_days_strictly_increasing": all(next_day > day for day, next_day in zip(visit_days, visit_days[1:])),
            "target_volume_monotone_non_decreasing": _is_non_decreasing(target_volumes),
            "actual_volume_monotone_non_decreasing_all_variants": all(variant_actual_monotone) if variant_actual_monotone else "",
            "qc_pass_count": qc_pass_count,
            "qc_fail_count": qc_fail_count,
            "max_abs_relative_volume_error": max(relative_errors) if relative_errors else math.nan,
        }
        summary.update(_evaluate_longitudinal_qc_gate(summary, volume_ravd_tolerance))
        summaries.append(summary)
    return summaries


def _build_longitudinal_provenance_payload(
    timeline_csv: Path,
    background_csv: Path,
    out_dir: Path,
    metadata_path: Path,
    qc_summary_path: Path,
    longitudinal_qc_summary_path: Path,
    generation_parameters: Dict[str, Any],
    timeline_rows: List[Dict[str, str]],
    metadata_rows: List[Dict[str, Any]],
    qc_rows: List[Dict[str, Any]],
    longitudinal_qc_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": "synthetic_longitudinal_provenance_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "wrapper_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "wrapper_script_sha256": _sha256_file(Path(__file__).resolve()),
        "embed_tumor_script": "embed_tumor.py",
        "embed_tumor_sha256": _sha256_file(REPO_ROOT / "embed_tumor.py"),
        "timeline_csv": str(timeline_csv),
        "timeline_csv_sha256": _sha256_file(timeline_csv),
        "background_csv": str(background_csv),
        "background_csv_sha256": _sha256_file(background_csv),
        "out_dir": str(out_dir),
        "metadata_csv": str(metadata_path),
        "metadata_csv_sha256": _sha256_file(metadata_path),
        "qc_summary_csv": str(qc_summary_path),
        "qc_summary_csv_sha256": _sha256_file(qc_summary_path),
        "longitudinal_qc_summary_csv": str(longitudinal_qc_summary_path),
        "longitudinal_qc_summary_csv_sha256": _sha256_file(longitudinal_qc_summary_path),
        "generation_parameters": dict(generation_parameters),
        "patient_count": len({row["patient_id"] for row in timeline_rows}),
        "timepoint_count": len(metadata_rows),
        "qc_pass_count": int(sum(1 for row in qc_rows if str(row.get("qc_pass")) == "True" or row.get("qc_pass") is True)),
        "qc_fail_count": int(sum(1 for row in qc_rows if not (str(row.get("qc_pass")) == "True" or row.get("qc_pass") is True))),
        "longitudinal_qc_rows": longitudinal_qc_rows,
        "timeline_rows": timeline_rows,
        "metadata_rows": metadata_rows,
        "qc_rows": qc_rows,
    }


def generate_longitudinal_dataset(
    timeline_csv: Path,
    background_csv: Path,
    out_dir: Path,
    seed: int = 20260523,
    volume_ravd_tolerance: float = 0.05,
    volume_max_iterations: int = 10,
    gen_size: int = 128,
    provenance_json: Path | None = None,
    clinical_growth_law: str = "none",
    visit_days: Tuple[float, float, float, float] = DEFAULT_VISIT_DAYS,
    variants_per_timepoint: int = 1,
) -> Tuple[Path, Path]:
    """Generate a simple 4-timepoint longitudinal synthetic dataset."""
    if variants_per_timepoint < 1:
        raise ValueError("variants_per_timepoint must be >= 1")
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
        visit_specs = None
        print(f"[{patient_index}/{len(timeline_rows)}] Patient {patient_id} on background {background_mri_id}")

        try:
            growth = _growth_mode(growth_label)
            visit_specs = _target_volumes_for_patient(
                patient=patient,
                clinical_growth_law=clinical_growth_law,
                visit_days=visit_days,
                seed=seed,
            )
            mri_path, seg_path = backgrounds[background_mri_id]
            if not mri_path.is_file():
                raise FileNotFoundError(f"MRI path does not exist for {background_mri_id}: {mri_path}")
            if not seg_path.is_file():
                raise FileNotFoundError(f"Segmentation path does not exist for {background_mri_id}: {seg_path}")
        except Exception as exc:
            reason = str(exc)
            print(f"  [error] {reason}")
            if visit_specs is None:
                visit_specs = _target_volumes_for_patient(
                    patient=patient,
                    clinical_growth_law="none",
                    visit_days=visit_days,
                    seed=seed,
                )
            for visit in visit_specs:
                timepoint = str(visit["timepoint"])
                target = float(visit["target_volume_mm3"])
                for variant_index in range(1, variants_per_timepoint + 1):
                    variant_id = f"V{variant_index:02d}"
                    metadata_rows.append(
                        {
                            "patient_id": patient_id,
                            "timepoint": timepoint,
                            "visit_day": visit["visit_day"],
                            "background_mri_id": background_mri_id,
                            "growth_label": growth_label,
                            "embedding_growth_mode": "",
                            "target_volume_mm3": target,
                            "target_volume_source": visit["target_volume_source"],
                            "growth_law_name": visit["growth_law_name"],
                            "growth_law_scenario": visit["growth_law_scenario"],
                            "growth_law_annual_volume_change_fraction": visit["growth_law_annual_volume_change_fraction"],
                            "variant_id": variant_id,
                            "variant_seed": "",
                            "visit_seed": "",
                            "source_mri_path": "",
                            "source_seg_path": "",
                            "volume_ravd_tolerance": volume_ravd_tolerance,
                            "volume_max_iterations": volume_max_iterations,
                            "gen_size": gen_size,
                            "image_path": "",
                            "mask_path": "",
                        }
                    )
                    qc_rows.append(
                        {
                            "patient_id": patient_id,
                            "timepoint": timepoint,
                            "variant_id": variant_id,
                            "target_volume_mm3": target,
                            "synthetic_volume_mm3": math.nan,
                            "relative_volume_error": math.nan,
                            "connected_components": math.nan,
                            "qc_pass": False,
                            "qc_failure_reason": reason,
                        }
                    )
            continue

        for visit in visit_specs:
            timepoint = str(visit["timepoint"])
            target_volume_mm3 = float(visit["target_volume_mm3"])
            for variant_index in range(1, variants_per_timepoint + 1):
                variant_id = f"V{variant_index:02d}"
                variant_suffix = "" if variants_per_timepoint == 1 else f"_{variant_id}"
                case_token = f"{_safe_id(patient_id)}_{timepoint}{variant_suffix}"
                image_path = images_dir / f"{case_token}_image.nii.gz"
                mask_path = masks_dir / f"{case_token}_mask.nii.gz"
                visit_seed = _stable_seed(seed, patient_id, timepoint)
                variant_seed = _stable_seed(seed, patient_id, f"{timepoint}:{variant_id}")

                metadata_rows.append(
                    {
                        "patient_id": patient_id,
                        "timepoint": timepoint,
                        "visit_day": visit["visit_day"],
                        "background_mri_id": background_mri_id,
                        "growth_label": growth_label,
                        "embedding_growth_mode": growth,
                        "target_volume_mm3": target_volume_mm3,
                        "target_volume_source": visit["target_volume_source"],
                        "growth_law_name": visit["growth_law_name"],
                        "growth_law_scenario": visit["growth_law_scenario"],
                        "growth_law_annual_volume_change_fraction": visit["growth_law_annual_volume_change_fraction"],
                        "variant_id": variant_id,
                        "variant_seed": int(variant_seed),
                        "visit_seed": int(visit_seed),
                        "source_mri_path": str(mri_path),
                        "source_seg_path": str(seg_path),
                        "volume_ravd_tolerance": volume_ravd_tolerance,
                        "volume_max_iterations": volume_max_iterations,
                        "gen_size": gen_size,
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
                            seed=variant_seed,
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
                            "variant_id": variant_id,
                            "target_volume_mm3": target_volume_mm3,
                            **qc,
                        }
                    )
                    status = "PASS" if qc["qc_pass"] else f"FAIL {qc['qc_failure_reason']}"
                    print(
                        f"  {timepoint}/{variant_id}: target={target_volume_mm3:.1f} "
                        f"synthetic={qc['synthetic_volume_mm3']:.1f} "
                        f"rel_err={qc['relative_volume_error']:.4f} {status}"
                    )
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    print(f"  [error] {timepoint}/{variant_id} failed: {reason}")
                    traceback.print_exc()
                    qc_rows.append(
                        {
                            "patient_id": patient_id,
                            "timepoint": timepoint,
                            "variant_id": variant_id,
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
    longitudinal_qc_summary_path = out_dir / "longitudinal_qc_summary.csv"
    _write_csv(
        metadata_path,
        metadata_rows,
        [
            "patient_id",
            "timepoint",
            "visit_day",
            "background_mri_id",
            "growth_label",
            "embedding_growth_mode",
            "target_volume_mm3",
            "target_volume_source",
            "growth_law_name",
            "growth_law_scenario",
            "growth_law_annual_volume_change_fraction",
            "variant_id",
            "variant_seed",
            "visit_seed",
            "source_mri_path",
            "source_seg_path",
            "volume_ravd_tolerance",
            "volume_max_iterations",
            "gen_size",
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
            "variant_id",
            "target_volume_mm3",
            "synthetic_volume_mm3",
            "relative_volume_error",
            "connected_components",
            "qc_pass",
            "qc_failure_reason",
        ],
    )
    longitudinal_qc_rows = _summarize_longitudinal_qc(
        metadata_rows,
        qc_rows,
        volume_ravd_tolerance=volume_ravd_tolerance,
    )
    _write_csv(
        longitudinal_qc_summary_path,
        longitudinal_qc_rows,
        [
            "patient_id",
            "timepoint_count",
            "variant_count",
            "metadata_row_count",
            "qc_row_count",
            "background_consistent",
            "background_mri_ids",
            "growth_law_scenarios",
            "visit_days_strictly_increasing",
            "target_volume_monotone_non_decreasing",
            "actual_volume_monotone_non_decreasing_all_variants",
            "qc_pass_count",
            "qc_fail_count",
            "max_abs_relative_volume_error",
            "engineering_qc_gate",
            "engineering_qc_failure_reasons",
            "target_volume_trend_status",
            "actual_volume_trend_status",
        ],
    )
    print(f"\nWrote {metadata_path}")
    print(f"Wrote {qc_summary_path}")
    print(f"Wrote {longitudinal_qc_summary_path}")
    if provenance_json is not None:
        provenance_json.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_longitudinal_provenance_payload(
            timeline_csv=timeline_csv,
            background_csv=background_csv,
            out_dir=out_dir,
            metadata_path=metadata_path,
            qc_summary_path=qc_summary_path,
            longitudinal_qc_summary_path=longitudinal_qc_summary_path,
            generation_parameters={
                "seed": seed,
                "volume_ravd_tolerance": volume_ravd_tolerance,
                "volume_max_iterations": volume_max_iterations,
                "gen_size": gen_size,
                "clinical_growth_law": clinical_growth_law,
                "visit_days": list(visit_days),
                "variants_per_timepoint": variants_per_timepoint,
            },
            timeline_rows=timeline_rows,
            metadata_rows=metadata_rows,
            qc_rows=qc_rows,
            longitudinal_qc_rows=longitudinal_qc_rows,
        )
        provenance_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {provenance_json}")
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
    parser.add_argument("--provenance_json", type=str, default=None, help="Optional dataset provenance sidecar JSON path.")
    parser.add_argument(
        "--clinical_growth_law",
        choices=CLINICAL_GROWTH_LAWS,
        default="none",
        help="Optional experimental target-volume generator. Default 'none' preserves T1..T4 CSV targets.",
    )
    parser.add_argument(
        "--visit_days",
        default="0,365.25,730.5,1095.75",
        help="Four comma-separated visit days used by clinical growth-law mode and metadata.",
    )
    parser.add_argument(
        "--variants_per_timepoint",
        type=int,
        default=1,
        help="Number of independently seeded shape variants to emit for each patient/timepoint.",
    )
    args = parser.parse_args()

    generate_longitudinal_dataset(
        timeline_csv=Path(args.timeline_csv).expanduser().resolve(),
        background_csv=Path(args.background_csv).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        seed=int(args.seed),
        volume_ravd_tolerance=float(args.volume_ravd_tolerance),
        volume_max_iterations=int(args.volume_max_iterations),
        gen_size=int(args.gen_size),
        provenance_json=Path(args.provenance_json).expanduser().resolve() if args.provenance_json else None,
        clinical_growth_law=str(args.clinical_growth_law),
        visit_days=_parse_visit_days(str(args.visit_days)),
        variants_per_timepoint=int(args.variants_per_timepoint),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
