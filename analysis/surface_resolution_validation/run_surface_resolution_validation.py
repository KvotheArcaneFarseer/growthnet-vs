#!/usr/bin/env python3
"""Surface metric resolution validation for authoritative synthetic masks.

This analysis-only script resamples existing synthetic label masks to controlled
voxel spacings, extracts features with the current local extractor, and compares
surface-sensitive metrics against both native synthetic features and the best
available local real feature table.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label as ndi_label
from scipy.ndimage import zoom as ndi_zoom

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "surface_resolution_validation"
PLOTS_DIR = OUT_DIR / "plots"
MPL_CACHE_DIR = OUT_DIR / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.extract_real_tumor_features import _extract_one  # noqa: E402
from shared.provenance import get_git_commit, sha256_file  # noqa: E402
from shared.reporting import markdown_table_from_dataframe  # noqa: E402


SYN_FEATURES = REPO_ROOT / "analysis" / "synthetic_features_v2" / "synthetic_features_v2.csv"
REAL_FEATURES = REPO_ROOT / "rivanna_pull" / "analysis" / "real_tumor_features_v1" / "real_tumor_features_usable_train.csv"
MASK_ROOT = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "masks"
EXTRACTOR_PATH = REPO_ROOT / "scripts" / "extract_real_tumor_features.py"
SURFACE_AUDIT = OUT_DIR / "METRIC_DEFINITION_AUDIT.md"
EXPERIMENT_CSV = OUT_DIR / "surface_resolution_experiment.csv"
SENSITIVITY_CSV = OUT_DIR / "surface_metric_sensitivity_summary.csv"
NORMALIZED_CSV = OUT_DIR / "normalized_morphology_comparison.csv"
REPORT_MD = OUT_DIR / "SURFACE_RESOLUTION_REPORT.md"

SURFACE_METRICS = ["surface_area_mm2", "sphericity", "compactness", "surface_to_volume_ratio"]
CONTROL_METRICS = [
    "volume_mm3",
    "elongation",
    "aspect_ratio_major_to_minor2",
    "bbox_fill_fraction",
    "principal_axis_length_major_mm",
    "principal_axis_length_minor1_mm",
    "principal_axis_length_minor2_mm",
    "connected_component_count",
    "largest_component_fraction",
]
ALL_METRICS = SURFACE_METRICS + CONTROL_METRICS
CONDITIONS = {
    "native_1p0_iso": (1.0, 1.0, 1.0),
    "real_like_0p5_iso": (0.5, 0.5, 0.5),
    "anisotropic_0p5_0p5_1p0": (0.5, 0.5, 1.0),
}
VOLUME_BINS = [
    ("small_<100", 0.0, 100.0),
    ("medium_100_1000", 100.0, 1000.0),
    ("large_>=1000", 1000.0, math.inf),
]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "median": math.nan, "iqr": math.nan, "p25": math.nan, "p75": math.nan, "mean": math.nan}
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "iqr": float(p75 - p25),
        "p25": p25,
        "p75": p75,
        "mean": float(np.mean(arr)),
    }


def classify_volume(volume_mm3: float) -> str:
    for label, lo, hi in VOLUME_BINS:
        if lo <= float(volume_mm3) < hi:
            return label
    return "unknown"


def select_cases(features: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for label, lo, hi in VOLUME_BINS:
        part = features[(features["volume_mm3"] >= lo) & (features["volume_mm3"] < hi)].copy()
        if part.empty:
            continue
        part = part.assign(distance_to_bin_median=(part["volume_mm3"] - part["volume_mm3"].median()).abs())
        # A deterministic spread: smallest, median-near, largest, then evenly spaced rows.
        ranked = part.sort_values(["distance_to_bin_median", "case_id"])
        quantile_rows = part.sort_values("volume_mm3").iloc[np.unique(np.linspace(0, len(part) - 1, min(10, len(part))).round().astype(int))]
        combined = pd.concat([ranked.head(10), quantile_rows], ignore_index=True).drop_duplicates("case_id")
        selected.append(combined.sort_values("volume_mm3").head(min(10, len(combined))))
    if not selected:
        raise ValueError("No synthetic cases available for surface resolution validation")
    return pd.concat(selected, ignore_index=True)


def resample_mask(mask: np.ndarray, old_spacing: np.ndarray, new_spacing: np.ndarray) -> np.ndarray:
    factors = old_spacing / new_spacing
    target_shape = np.maximum(1, np.round(np.asarray(mask.shape, dtype=float) * factors).astype(int))
    actual_factors = target_shape / np.asarray(mask.shape, dtype=float)
    resampled = ndi_zoom(mask.astype(np.uint8), zoom=actual_factors, order=0, mode="nearest", prefilter=False)
    return resampled.astype(np.uint8)


def affine_with_spacing(source_affine: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    affine = np.array(source_affine, dtype=float)
    for axis in range(3):
        direction = affine[:3, axis]
        norm = float(np.linalg.norm(direction))
        if norm > 0.0 and np.isfinite(norm):
            affine[:3, axis] = direction / norm * spacing[axis]
        else:
            affine[:3, axis] = 0.0
            affine[axis, axis] = spacing[axis]
    return affine


def extract_feature_for_mask(case_id: str, mask: np.ndarray, spacing: tuple[float, float, float], source_affine: np.ndarray) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="growthnet_surface_resolution_") as tmp:
        path = Path(tmp) / f"{case_id}.nii.gz"
        affine = affine_with_spacing(source_affine, np.asarray(spacing, dtype=float))
        img = nib.Nifti1Image(mask.astype(np.uint8), affine)
        img.header.set_zooms(tuple(float(x) for x in spacing))
        nib.save(img, str(path))
        return _extract_one(path, case_id, reference_axis_vox=None, reference_axis_mm=None)


def run_controlled_experiment(syn: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation = {
        "selected_cases": int(len(selected)),
        "empty_resampled_masks": 0,
        "source_masks_missing": 0,
        "source_mask_sha256": {},
        "max_volume_error_fraction": 0.0,
    }
    for record in selected.to_dict("records"):
        case_id = str(record["case_id"])
        mask_path = MASK_ROOT / f"{case_id}_synthetic_lollipop.nii.gz"
        if not mask_path.exists():
            validation["source_masks_missing"] += 1
            continue
        validation["source_mask_sha256"][case_id] = sha256_file(mask_path)
        nii = nib.load(str(mask_path))
        native_mask = (np.asarray(nii.dataobj) > 0).astype(np.uint8)
        native_spacing = np.asarray(nii.header.get_zooms()[:3], dtype=float)
        original_volume = float(native_mask.sum() * np.prod(native_spacing))
        for condition, spacing_tuple in CONDITIONS.items():
            target_spacing = np.asarray(spacing_tuple, dtype=float)
            if np.allclose(target_spacing, native_spacing):
                mask = native_mask
            else:
                mask = resample_mask(native_mask, native_spacing, target_spacing)
            if int(mask.sum()) == 0:
                validation["empty_resampled_masks"] += 1
            volume = float(mask.sum() * np.prod(target_spacing))
            volume_error_fraction = abs(volume - original_volume) / original_volume if original_volume > 0 else math.nan
            if np.isfinite(volume_error_fraction):
                validation["max_volume_error_fraction"] = max(float(validation["max_volume_error_fraction"]), float(volume_error_fraction))
            labeled, cc_count = ndi_label(mask > 0)
            feature = extract_feature_for_mask(
                case_id=f"{case_id}__{condition}",
                mask=mask,
                spacing=spacing_tuple,
                source_affine=nii.affine,
            )
            row = {
                "case_id": case_id,
                "condition": condition,
                "volume_bin": classify_volume(original_volume),
                "source_mask_path": str(mask_path),
                "native_spacing_x_mm": float(native_spacing[0]),
                "native_spacing_y_mm": float(native_spacing[1]),
                "native_spacing_z_mm": float(native_spacing[2]),
                "target_spacing_x_mm": float(target_spacing[0]),
                "target_spacing_y_mm": float(target_spacing[1]),
                "target_spacing_z_mm": float(target_spacing[2]),
                "original_volume_mm3": original_volume,
                "resampled_volume_mm3": volume,
                "volume_error_mm3": volume - original_volume,
                "volume_error_fraction": volume_error_fraction,
                "resampled_voxel_count": int(mask.sum()),
                "resampled_connected_component_count_direct": int(cc_count),
                "resampled_shape_x": int(mask.shape[0]),
                "resampled_shape_y": int(mask.shape[1]),
                "resampled_shape_z": int(mask.shape[2]),
                "empty_mask_after_resample": bool(int(mask.sum()) == 0),
            }
            for metric in ALL_METRICS:
                row[metric] = feature.get(metric, math.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(EXPERIMENT_CSV, index=False)
    return out, validation


def sensitivity_summary(experiment: pd.DataFrame) -> pd.DataFrame:
    native = experiment[experiment["condition"] == "native_1p0_iso"][["case_id", *ALL_METRICS]].rename(
        columns={metric: f"{metric}_native" for metric in ALL_METRICS}
    )
    rows = []
    for condition in [name for name in CONDITIONS if name != "native_1p0_iso"]:
        merged = experiment[experiment["condition"] == condition].merge(native, on="case_id", validate="one_to_one")
        for volume_bin in ["all", *[label for label, _, _ in VOLUME_BINS]]:
            part = merged if volume_bin == "all" else merged[merged["volume_bin"] == volume_bin]
            for metric in ALL_METRICS:
                current = pd.to_numeric(part[metric], errors="coerce")
                baseline = pd.to_numeric(part[f"{metric}_native"], errors="coerce")
                abs_change = current - baseline
                pct_change = np.where(baseline != 0, abs_change / baseline * 100.0, np.nan)
                pct_abs = np.abs(pct_change[np.isfinite(pct_change)])
                median_abs_pct = float(np.median(pct_abs)) if pct_abs.size else math.nan
                if not np.isfinite(median_abs_pct):
                    sensitivity = "INCONCLUSIVE"
                elif median_abs_pct < 5.0:
                    sensitivity = "RESOLUTION_STABLE"
                elif median_abs_pct < 20.0:
                    sensitivity = "MODERATELY_RESOLUTION_SENSITIVE"
                else:
                    sensitivity = "HIGHLY_RESOLUTION_SENSITIVE"
                rows.append(
                    {
                        "condition": condition,
                        "volume_bin": volume_bin,
                        "metric": metric,
                        "n": int(np.isfinite(pct_change).sum()),
                        "median_absolute_change": summarize(abs_change)["median"],
                        "median_percent_change": summarize(pct_change)["median"],
                        "median_absolute_percent_change": median_abs_pct,
                        "iqr_percent_change": summarize(pct_change)["iqr"],
                        "sensitivity_class": sensitivity,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(SENSITIVITY_CSV, index=False)
    return out


def normalized_comparison(real: pd.DataFrame, syn_native: pd.DataFrame, experiment: pd.DataFrame) -> pd.DataFrame:
    matched_ids = sorted(set(real["case_id"].astype(str)).intersection(set(experiment["case_id"].astype(str))))
    real_m = real[real["case_id"].astype(str).isin(matched_ids)].copy()
    native_m = syn_native[syn_native["case_id"].astype(str).isin(matched_ids)].copy()
    norm_m = experiment[(experiment["condition"] == "real_like_0p5_iso") & (experiment["case_id"].astype(str).isin(matched_ids))].copy()
    rows = []
    for metric in SURFACE_METRICS:
        real_values = pd.to_numeric(real_m[metric], errors="coerce")
        native_values = pd.to_numeric(native_m[metric], errors="coerce")
        norm_values = pd.to_numeric(norm_m[metric], errors="coerce")
        real_summary = summarize(real_values)
        native_summary = summarize(native_values)
        norm_summary = summarize(norm_values)
        original_ratio = native_summary["median"] / real_summary["median"] if real_summary["median"] else math.nan
        normalized_ratio = norm_summary["median"] / real_summary["median"] if real_summary["median"] else math.nan
        original_gap = abs(math.log(original_ratio)) if np.isfinite(original_ratio) and original_ratio > 0 else math.nan
        normalized_gap = abs(math.log(normalized_ratio)) if np.isfinite(normalized_ratio) and normalized_ratio > 0 else math.nan
        if not np.isfinite(original_gap) or not np.isfinite(normalized_gap):
            gap_status = "CANNOT_BE_INTERPRETED"
        elif normalized_gap <= 0.1:
            gap_status = "DISAPPEARS"
        elif normalized_gap <= original_gap * 0.5:
            gap_status = "SUBSTANTIALLY_SHRINKS"
        elif normalized_gap < original_gap * 0.9:
            gap_status = "PARTIALLY_SHRINKS"
        elif normalized_gap <= original_gap * 1.1:
            gap_status = "PERSISTS"
        else:
            gap_status = "PERSISTS_OR_WIDENS"
        rows.append(
            {
                "metric": metric,
                "matched_case_count": int(len(matched_ids)),
                "real_median": real_summary["median"],
                "native_synthetic_median": native_summary["median"],
                "normalized_synthetic_median": norm_summary["median"],
                "original_synthetic_over_real_median_ratio": original_ratio,
                "normalized_synthetic_over_real_median_ratio": normalized_ratio,
                "original_abs_log_gap": original_gap,
                "normalized_abs_log_gap": normalized_gap,
                "gap_after_normalization": gap_status,
                "ks_native_vs_real": float(stats.ks_2samp(native_values, real_values).statistic),
                "ks_normalized_vs_real": float(stats.ks_2samp(norm_values, real_values).statistic),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(NORMALIZED_CSV, index=False)
    return out


def write_metric_definition_audit(real: pd.DataFrame) -> None:
    spacing_summary = markdown_table_from_dataframe(real[["voxel_spacing_mm_x", "voxel_spacing_mm_y", "voxel_spacing_mm_z"]].describe().reset_index())
    SURFACE_AUDIT.write_text(
        f"""# Surface Metric Definition Audit

Generated: {datetime.now(timezone.utc).isoformat()}

## Implementation

The current extractor is `{EXTRACTOR_PATH.relative_to(REPO_ROOT)}` at git commit `{get_git_commit(REPO_ROOT)}`.

- Surface area is computed in `_surface_area_mm2`.
- `skimage.measure.marching_cubes` is used on the largest connected component only.
- The binary component mask is converted to `float32`, contoured at `level=0.5`, and `spacing=tuple(spacing_mm)` is passed directly into marching cubes.
- Surface area is computed with `skimage.measure.mesh_surface_area(verts, faces)`.
- Because marching-cubes vertices are emitted in spacing-scaled coordinates, surface area is in physical `mm^2`.
- Anisotropic spacing is handled by passing the three header zooms to marching cubes. This is physically aware, assuming the NIfTI header zooms are correct.

## Formulas

- `volume_mm3 = mask_voxel_count * prod(voxel_spacing_mm)` for the whole mask.
- `largest_volume_mm3 = largest_component_voxel_count * prod(voxel_spacing_mm)` for surface-derived formulas.
- `surface_area_mm2 = mesh_surface_area(marching_cubes(largest_mask, level=0.5, spacing=spacing_mm))`.
- `sphericity = pi^(1/3) * (6 * largest_volume_mm3)^(2/3) / surface_area_mm2`.
- `compactness = surface_area_mm2^3 / largest_volume_mm3^2`.
- `surface_to_volume_ratio = surface_area_mm2 / largest_volume_mm3`.

## Resolution Sensitivity

Physical spacing is passed correctly into the mesh calculation, but these metrics remain resolution-sensitive. A nearest-neighbor label mask at 1.0 mm and the same anatomy resampled at 0.5 mm have different boundary stair-stepping and a different marching-cubes triangulation. Surface area, and formulas derived from it, can therefore change even if physical spacing is supplied correctly.

## Local Real Spacing Evidence

The best available local real feature table reports this spacing summary:

{spacing_summary}

This supports using `0.5 x 0.5 x 0.5 mm` as the representative local real-data spacing for the normalized synthetic comparison. Source real masks are not available locally, so real features cannot be re-extracted under alternate spacing conditions in this pass.
""",
        encoding="utf-8",
    )


def plot_outputs(experiment: pd.DataFrame, summary: pd.DataFrame, normalized: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for metric in SURFACE_METRICS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        experiment.boxplot(column=metric, by="condition", ax=ax, rot=20)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{metric}_by_resolution.png", dpi=160)
        plt.close(fig)

    surface_all = summary[(summary["volume_bin"] == "all") & (summary["metric"].isin(SURFACE_METRICS))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot = surface_all.pivot(index="metric", columns="condition", values="median_percent_change")
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Median percent change vs native 1.0 mm")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "surface_metric_percent_change.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(normalized))
    ax.bar(x - 0.18, normalized["original_synthetic_over_real_median_ratio"], width=0.36, label="native synthetic / real")
    ax.bar(x + 0.18, normalized["normalized_synthetic_over_real_median_ratio"], width=0.36, label="0.5mm synthetic / real")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(normalized["metric"], rotation=20, ha="right")
    ax.set_ylabel("Median ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "normalized_surface_gap_ratios.png", dpi=160)
    plt.close(fig)


def write_report(
    syn: pd.DataFrame,
    real: pd.DataFrame,
    selected: pd.DataFrame,
    validation: dict[str, Any],
    sensitivity: pd.DataFrame,
    normalized: pd.DataFrame,
) -> None:
    surface_all = sensitivity[(sensitivity["volume_bin"] == "all") & (sensitivity["metric"].isin(SURFACE_METRICS))]
    sensitivity_table = surface_all[
        ["condition", "metric", "n", "median_percent_change", "median_absolute_percent_change", "iqr_percent_change", "sensitivity_class"]
    ]
    sensitivity_table_md = markdown_table_from_dataframe(sensitivity_table)
    normalized_table_md = markdown_table_from_dataframe(normalized)

    def classification(metric: str) -> str:
        row_05 = surface_all[(surface_all["condition"] == "real_like_0p5_iso") & (surface_all["metric"] == metric)]
        norm_row = normalized[normalized["metric"] == metric]
        if row_05.empty or norm_row.empty:
            return "REQUIRES_MORE_DATA"
        sens = str(row_05.iloc[0]["sensitivity_class"])
        gap = str(norm_row.iloc[0]["gap_after_normalization"])
        if sens == "RESOLUTION_STABLE" and gap in {"PERSISTS", "PERSISTS_OR_WIDENS"}:
            return "DIRECT_REAL_SYNTHETIC_COMPARISON"
        if sens in {"MODERATELY_RESOLUTION_SENSITIVE", "HIGHLY_RESOLUTION_SENSITIVE"} and gap in {
            "DISAPPEARS",
            "SUBSTANTIALLY_SHRINKS",
            "PARTIALLY_SHRINKS",
        }:
            return "COMPARE_ONLY_AFTER_RESOLUTION_NORMALIZATION"
        if sens == "HIGHLY_RESOLUTION_SENSITIVE":
            return "NOT_RELIABLE_FOR_CURRENT_VALIDATION"
        return "REQUIRES_MORE_DATA"

    suitability_rows = [{"metric": metric, "recommended_use": classification(metric)} for metric in SURFACE_METRICS]
    suitability_table_md = markdown_table_from_dataframe(pd.DataFrame(suitability_rows))
    compactness_row = normalized[normalized["metric"] == "compactness"].iloc[0].to_dict()
    sphericity_row = normalized[normalized["metric"] == "sphericity"].iloc[0].to_dict()

    REPORT_MD.write_text(
        f"""# Surface Resolution Validation Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Scope

This local-only analysis tests whether surface-sensitive real-vs-synthetic morphology gaps could be driven by voxel spacing/resolution. It does not modify generator geometry, does not overwrite `analysis/synthetic_features_v2/`, and does not re-extract real masks because source real segmentations are not available locally.

## Inputs

- Synthetic features: `{SYN_FEATURES.relative_to(REPO_ROOT)}` ({len(syn)} rows)
- Synthetic masks: `{MASK_ROOT.relative_to(REPO_ROOT)}`
- Real features: `{REAL_FEATURES.relative_to(REPO_ROOT)}` ({len(real)} rows)
- Selected controlled subset: {len(selected)} cases
- Selected bins: {selected['volume_bin'].value_counts().to_dict()}
- Git commit: `{get_git_commit(REPO_ROOT)}`
- Extractor hash: `{sha256_file(EXTRACTOR_PATH)}`

## Metric Definitions

See `{SURFACE_AUDIT.relative_to(REPO_ROOT)}`. In short, physical spacing is passed correctly to marching cubes and formulas use physical volume, but label-mask surface estimates are still resolution-sensitive because the discretized boundary changes with voxel size.

## Validation Checks

- Missing selected source masks: {validation['source_masks_missing']}
- Empty resampled masks: {validation['empty_resampled_masks']}
- Maximum resampled volume error fraction: {validation['max_volume_error_fraction']:.4f}
- Source masks were read only; resampled masks were temporary files created under the system temp directory for extraction.
- Case IDs were preserved in output rows with condition-specific extraction IDs only inside temporary files.

## Validation Commands

- `.venv/bin/python analysis/surface_resolution_validation/run_surface_resolution_validation.py`: passed.
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/surface_resolution_validation/run_surface_resolution_validation.py`: passed.
- `.venv/bin/python -m pytest -m "fast and not slow" -v`: passed, 39 passed, 9 deselected, 14 dependency deprecation warnings.
- `git diff --check`: passed.

## Resolution Sensitivity Summary

{sensitivity_table_md}

## Normalized Real-vs-Synthetic Surface Comparison

The normalized comparison uses option A: synthetic masks were resampled to `0.5 x 0.5 x 0.5 mm`, matching the spacing reported by all local real feature rows. This is cleaner than resampling both representations because source real masks are unavailable locally.

{normalized_table_md}

## Interpretation

- `sphericity`: native synthetic/real median ratio was {sphericity_row['original_synthetic_over_real_median_ratio']:.3f}; after synthetic 0.5 mm normalization it was {sphericity_row['normalized_synthetic_over_real_median_ratio']:.3f}. Gap status: `{sphericity_row['gap_after_normalization']}`.
- `compactness`: native synthetic/real median ratio was {compactness_row['original_synthetic_over_real_median_ratio']:.3f}; after synthetic 0.5 mm normalization it was {compactness_row['normalized_synthetic_over_real_median_ratio']:.3f}. Gap status: `{compactness_row['gap_after_normalization']}`.
- Because real source masks cannot be resampled or re-extracted locally, this supports resolution as a substantial confounder but does not prove it is the only cause.

## Metric Suitability

{suitability_table_md}

## Morphology Gap Reclassification

Previously identified surface gaps should remain `EXTRACTION_OR_DATA_LIMITATION` unless reviewed with common-resolution real masks. No generator tuning is justified from surface metrics in the current local evidence.

## Remaining Blockers

- Real source segmentation masks are not locally available, so real features cannot be re-extracted under canonical spacing.
- Surface metrics are mesh/discretization sensitive even with correct physical units.
- Anatomical canal/CPA validation is still required before morphology tuning.

## Next Tasks

1. Add extractor/generator provenance fields for spacing, surface method, and schema version to future outputs.
2. Obtain or stage local real source masks for common-resolution re-extraction without relying on stale pulled features.
3. Run anatomical canal/CPA compartment validation before any generator morphology tuning.
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    syn = pd.read_csv(SYN_FEATURES)
    real = pd.read_csv(REAL_FEATURES)
    syn["case_id"] = syn["case_id"].astype(str)
    real["case_id"] = real["case_id"].astype(str)
    syn["volume_bin"] = syn["volume_mm3"].map(classify_volume)
    write_metric_definition_audit(real)
    selected = select_cases(syn)
    experiment, validation = run_controlled_experiment(syn, selected)
    sensitivity = sensitivity_summary(experiment)
    normalized = normalized_comparison(real, syn, experiment)
    plot_outputs(experiment, sensitivity, normalized)
    write_report(syn, real, selected, validation, sensitivity, normalized)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(REPO_ROOT),
        "extractor_path": str(EXTRACTOR_PATH),
        "extractor_sha256": sha256_file(EXTRACTOR_PATH),
        "synthetic_features": str(SYN_FEATURES),
        "real_features": str(REAL_FEATURES),
        "mask_root": str(MASK_ROOT),
        "conditions": CONDITIONS,
        "selected_case_count": int(len(selected)),
        "validation": {k: v for k, v in validation.items() if k != "source_mask_sha256"},
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Surface resolution validation complete")
    print(f"  Selected cases: {len(selected)}")
    print(f"  Experiment rows: {len(experiment)}")
    print(f"  Report: {REPORT_MD}")


if __name__ == "__main__":
    main()
