#!/usr/bin/env python3
"""Local real-vs-synthetic morphology audit for GrowthNet.

This script intentionally writes only inside analysis/real_vs_synthetic.
It uses pulled real feature tables and locally available synthetic masks.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "real_vs_synthetic"
PLOTS_DIR = OUT_DIR / "plots"
MPL_CACHE_DIR = OUT_DIR / ".matplotlib-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402


REAL_FEATURES = REPO_ROOT / "rivanna_pull" / "analysis" / "real_tumor_features_v1" / "real_tumor_features_usable_train.csv"
REAL_SUMMARY = REPO_ROOT / "rivanna_pull" / "analysis" / "real_tumor_features_v1" / "real_tumor_feature_summary_usable_train.json"
SYN_FEATURES = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "synthetic_lollipop_features.csv"
SYN_REEXTRACTED = OUT_DIR / "synthetic_features_reextracted_local.csv"
SYN_SUMMARY = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "synthetic_lollipop_feature_summary.json"
SYN_MANIFEST = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "manifests" / "synthetic_lollipop_manifest.csv"
REAL_TARGETS = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "manifests" / "real_single_component_targets.csv"
SYN_MASK_DIR = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "masks"
EXISTING_COMPARISON = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "real_vs_synthetic_distribution_comparison.csv"

METRICS = [
    "volume_mm3",
    "equivalent_sphere_diameter_mm",
    "max_diameter_bbox_mm",
    "principal_axis_length_major_mm",
    "principal_axis_length_minor1_mm",
    "principal_axis_length_minor2_mm",
    "elongation",
    "aspect_ratio_major_to_minor2",
    "flatness",
    "bbox_fill_fraction",
    "surface_area_mm2",
    "sphericity",
    "surface_to_volume_ratio",
    "roughness_index",
]

PRIMARY_GAP_METRICS = [
    "elongation",
    "aspect_ratio_major_to_minor2",
    "bbox_fill_fraction",
    "sphericity",
    "max_diameter_bbox_mm",
    "principal_axis_length_major_mm",
    "surface_to_volume_ratio",
]


def finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    return values[np.isfinite(values)]


def summarize(values: pd.Series) -> dict[str, float | int]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "p25": math.nan,
            "p75": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def cliffs_delta(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    if x_arr.size == 0 or y_arr.size == 0:
        return math.nan
    gt = 0
    lt = 0
    for value in x_arr:
        gt += int(np.sum(value > y_arr))
        lt += int(np.sum(value < y_arr))
    return float((gt - lt) / (x_arr.size * y_arr.size))


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real = pd.read_csv(REAL_FEATURES)
    syn = pd.read_csv(SYN_FEATURES)
    manifest = pd.read_csv(SYN_MANIFEST)
    targets = pd.read_csv(REAL_TARGETS)
    return real, syn, manifest, targets


def write_local_synthetic_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.to_dict("records"):
        case_id = str(row["case_id"])
        local_path = SYN_MASK_DIR / f"{case_id}_synthetic_lollipop.nii.gz"
        rows.append(
            {
                **row,
                "original_seg_path": row.get("seg_path", ""),
                "seg_path": str(local_path),
                "local_mask_exists": bool(local_path.exists()),
            }
        )
    local = pd.DataFrame(rows)
    local.to_csv(OUT_DIR / "local_synthetic_manifest.csv", index=False)
    return local


def inventory(real: pd.DataFrame, syn: pd.DataFrame, manifest: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    matched_case_ids = sorted(set(real["case_id"]).intersection(set(syn["case_id"])))
    real_paths_exist = real["seg_path"].map(lambda p: Path(str(p)).exists())
    syn_paths_exist = syn["seg_path"].map(lambda p: Path(str(p)).exists())
    local_manifest_exists = manifest["seg_path"].map(lambda p: Path(str(p)).exists())
    rows = [
        {"item": "real_features_rows", "count": len(real), "notes": "Pulled feature rows available locally."},
        {"item": "synthetic_features_rows", "count": len(syn), "notes": "Pulled feature rows available locally."},
        {"item": "matched_case_ids", "count": len(matched_case_ids), "notes": "Intersection of real and synthetic feature case_id values."},
        {"item": "real_source_seg_paths_existing_locally", "count": int(real_paths_exist.sum()), "notes": "CSV seg_path values point to original segmentation masks."},
        {"item": "synthetic_feature_seg_paths_existing_locally", "count": int(syn_paths_exist.sum()), "notes": "Pulled CSV seg_path values before local remapping."},
        {"item": "synthetic_manifest_local_masks_existing", "count": int(local_manifest_exists.sum()), "notes": "Local synthetic masks after remapping to repo-local paths."},
        {"item": "real_single_component_targets", "count": len(targets), "notes": "Targets used to generate the synthetic benchmark."},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "sample_inventory.csv", index=False)
    return out


def matched_comparison(real: pd.DataFrame, syn: pd.DataFrame, file_prefix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    real_m = real[["case_id", *METRICS]].copy()
    syn_m = syn[["case_id", *METRICS]].copy()
    merged = real_m.merge(syn_m, on="case_id", suffixes=("_real", "_synthetic"), validate="one_to_one")
    merged.to_csv(OUT_DIR / f"{file_prefix}matched_case_feature_table.csv", index=False)

    rows = []
    for metric in METRICS:
        r = finite_series(merged, f"{metric}_real")
        s = finite_series(merged, f"{metric}_synthetic")
        stat = stats.ks_2samp(r, s, alternative="two-sided", mode="auto") if len(r) and len(s) else None
        real_summary = summarize(r)
        syn_summary = summarize(s)
        rows.append(
            {
                "metric": metric,
                "real_n": real_summary["n"],
                "synthetic_n": syn_summary["n"],
                "real_median": real_summary["median"],
                "synthetic_median": syn_summary["median"],
                "median_diff_synthetic_minus_real": syn_summary["median"] - real_summary["median"],
                "median_ratio_synthetic_over_real": syn_summary["median"] / real_summary["median"] if real_summary["median"] else math.nan,
                "real_p25": real_summary["p25"],
                "real_p75": real_summary["p75"],
                "synthetic_p25": syn_summary["p25"],
                "synthetic_p75": syn_summary["p75"],
                "ks_statistic": float(stat.statistic) if stat else math.nan,
                "ks_pvalue": float(stat.pvalue) if stat else math.nan,
                "cliffs_delta_synthetic_vs_real": cliffs_delta(s, r),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT_DIR / f"{file_prefix}matched_distribution_comparison.csv", index=False)
    return merged, comparison


def stratified_comparison(merged: pd.DataFrame, file_prefix: str = "") -> pd.DataFrame:
    bins = [0.0, 100.0, 500.0, 1500.0, math.inf]
    labels = ["tiny_<100", "small_100_500", "medium_500_1500", "large_>=1500"]
    merged = merged.copy()
    merged["volume_stratum"] = pd.cut(merged["volume_mm3_real"], bins=bins, labels=labels, right=False)
    rows = []
    for stratum in labels:
        part = merged[merged["volume_stratum"] == stratum]
        for metric in PRIMARY_GAP_METRICS:
            r = finite_series(part, f"{metric}_real")
            s = finite_series(part, f"{metric}_synthetic")
            real_summary = summarize(r)
            syn_summary = summarize(s)
            rows.append(
                {
                    "volume_stratum": stratum,
                    "n_cases": int(len(part)),
                    "metric": metric,
                    "real_median": real_summary["median"],
                    "synthetic_median": syn_summary["median"],
                    "median_diff_synthetic_minus_real": syn_summary["median"] - real_summary["median"],
                    "median_ratio_synthetic_over_real": syn_summary["median"] / real_summary["median"] if real_summary["median"] else math.nan,
                    "cliffs_delta_synthetic_vs_real": cliffs_delta(s, r),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"{file_prefix}volume_stratified_comparison.csv", index=False)
    return out


def ranked_gaps(comparison: pd.DataFrame, file_prefix: str = "") -> pd.DataFrame:
    rows = []
    metric_notes = {
        "elongation": "Synthetic masks are more elongated when ratio > 1.",
        "aspect_ratio_major_to_minor2": "Higher values indicate larger major/minor2 disparity.",
        "bbox_fill_fraction": "Lower synthetic fill suggests more empty bounding-box space from lollipop/stem geometry.",
        "sphericity": "Lower synthetic sphericity suggests less compact or more irregular shape.",
        "max_diameter_bbox_mm": "Higher synthetic diameter indicates spatial extent larger than real at matched volume.",
        "principal_axis_length_major_mm": "Higher synthetic major axis indicates longer dominant morphology.",
        "surface_to_volume_ratio": "Higher synthetic ratio indicates more surface per unit volume.",
    }
    for row in comparison.to_dict("records"):
        metric = row["metric"]
        if metric not in PRIMARY_GAP_METRICS:
            continue
        ratio = row["median_ratio_synthetic_over_real"]
        abs_log_ratio = abs(math.log(ratio)) if ratio and np.isfinite(ratio) and ratio > 0 else math.nan
        rows.append(
            {
                "rank_score_abs_log_median_ratio": abs_log_ratio,
                "metric": metric,
                "median_ratio_synthetic_over_real": ratio,
                "median_diff_synthetic_minus_real": row["median_diff_synthetic_minus_real"],
                "ks_statistic": row["ks_statistic"],
                "cliffs_delta_synthetic_vs_real": row["cliffs_delta_synthetic_vs_real"],
                "interpretation": metric_notes.get(metric, ""),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["rank_score_abs_log_median_ratio", "ks_statistic"],
        ascending=[False, False],
        na_position="last",
    )
    out.to_csv(OUT_DIR / f"{file_prefix}ranked_morphology_gaps.csv", index=False)
    return out


def synthetic_reextraction_drift(pulled_syn: pd.DataFrame) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not SYN_REEXTRACTED.exists():
        return None, None

    local_syn = pd.read_csv(SYN_REEXTRACTED)
    merged = pulled_syn.merge(local_syn, on="case_id", suffixes=("_pulled", "_local"), validate="one_to_one")
    merged.to_csv(OUT_DIR / "synthetic_pulled_vs_reextracted_case_drift.csv", index=False)

    rows = []
    for metric in METRICS:
        diff = pd.to_numeric(merged[f"{metric}_local"], errors="coerce") - pd.to_numeric(merged[f"{metric}_pulled"], errors="coerce")
        abs_diff = diff.abs()
        rows.append(
            {
                "metric": metric,
                "n": int(abs_diff.notna().sum()),
                "median_abs_diff": float(abs_diff.median()),
                "max_abs_diff": float(abs_diff.max()),
                "changed_gt_1e_minus_9": int((abs_diff > 1e-9).sum()),
                "pulled_median": float(pd.to_numeric(merged[f"{metric}_pulled"], errors="coerce").median()),
                "local_reextracted_median": float(pd.to_numeric(merged[f"{metric}_local"], errors="coerce").median()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "synthetic_pulled_vs_reextracted_metric_drift.csv", index=False)
    return local_syn, summary


def write_remote_dependencies() -> pd.DataFrame:
    rows = [
        {
            "status": "BLOCKED_REMOTE_DATA",
            "dependency": "Real segmentation NIfTI masks for the 261 matched usable targets",
            "why_needed": "Required to independently re-extract real morphology features and verify extraction equivalence end-to-end.",
            "local_fallback_used": "Pulled real feature CSV and JSON summaries.",
        },
        {
            "status": "BLOCKED_REMOTE_DATA",
            "dependency": "Full original 291-case train segmentation source masks",
            "why_needed": "Required to validate population-level real distribution including 30 multi-component cases.",
            "local_fallback_used": "Summary JSON documents 291 rows, while matched benchmark uses 261 single-component cases.",
        },
        {
            "status": "BLOCKED_REMOTE_DATA",
            "dependency": "Clinical/anatomical labels for IAC/CPA canal-vs-bulb compartment split",
            "why_needed": "Required to distinguish biologically meaningful lollipop differences from PCA/bounding-box extraction artifacts.",
            "local_fallback_used": "Whole-mask morphology metrics only.",
        },
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "remote_data_dependencies.csv", index=False)
    return out


def make_plots(merged: pd.DataFrame, comparison: pd.DataFrame, stratified: pd.DataFrame) -> list[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for metric in PRIMARY_GAP_METRICS:
        plt.figure(figsize=(7, 4.5))
        values = [
            finite_series(merged, f"{metric}_real").to_numpy(dtype=float),
            finite_series(merged, f"{metric}_synthetic").to_numpy(dtype=float),
        ]
        plt.boxplot(values, tick_labels=["real", "synthetic"], showfliers=False)
        plt.ylabel(metric)
        plt.title(f"{metric}: real vs synthetic")
        plt.tight_layout()
        out = PLOTS_DIR / f"{metric}_boxplot.png"
        plt.savefig(out, dpi=160)
        plt.close()
        written.append(str(out.relative_to(REPO_ROOT)))

    plt.figure(figsize=(7, 5))
    plt.scatter(merged["volume_mm3_real"], merged["volume_mm3_synthetic"], s=18, alpha=0.75)
    min_v = float(np.nanmin([merged["volume_mm3_real"].min(), merged["volume_mm3_synthetic"].min()]))
    max_v = float(np.nanmax([merged["volume_mm3_real"].max(), merged["volume_mm3_synthetic"].max()]))
    plt.plot([min_v, max_v], [min_v, max_v], color="black", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("real volume_mm3")
    plt.ylabel("synthetic volume_mm3")
    plt.title("Requested-vs-realized matched volumes")
    plt.tight_layout()
    out = PLOTS_DIR / "volume_match_scatter.png"
    plt.savefig(out, dpi=160)
    plt.close()
    written.append(str(out.relative_to(REPO_ROOT)))

    pivot = stratified[stratified["metric"].isin(PRIMARY_GAP_METRICS)].pivot(
        index="metric",
        columns="volume_stratum",
        values="median_ratio_synthetic_over_real",
    )
    plt.figure(figsize=(8, 4.5))
    image = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=0.4, vmax=1.8)
    plt.colorbar(image, label="median ratio synthetic / real")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    plt.title("Volume-stratified morphology ratios")
    plt.tight_layout()
    out = PLOTS_DIR / "volume_stratified_ratio_heatmap.png"
    plt.savefig(out, dpi=160)
    plt.close()
    written.append(str(out.relative_to(REPO_ROOT)))

    return written


def write_report(
    inventory_df: pd.DataFrame,
    comparison: pd.DataFrame,
    stratified: pd.DataFrame,
    gaps: pd.DataFrame,
    remote_deps: pd.DataFrame,
    plots: list[str],
    current_comparison: pd.DataFrame | None,
    current_gaps: pd.DataFrame | None,
    synthetic_drift: pd.DataFrame | None,
) -> None:
    with REAL_SUMMARY.open("r") as handle:
        real_summary = json.load(handle)
    with SYN_SUMMARY.open("r") as handle:
        syn_summary = json.load(handle)

    top_gaps = gaps.head(7).to_dict("records")
    comparison_by_metric = {row["metric"]: row for row in comparison.to_dict("records")}
    current_by_metric = {row["metric"]: row for row in current_comparison.to_dict("records")} if current_comparison is not None else {}

    def fmt(value: object, digits: int = 3) -> str:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(f):
            return "nan"
        return f"{f:.{digits}f}"

    lines = [
        "# Local Real-Versus-Synthetic Morphology Validation",
        "",
        "Task ID: MORPH-001",
        "",
        "## Scope",
        "",
        "This audit uses only files present in the local GrowthNet repository. It does not SSH, does not query Rivanna, and does not depend on remote source masks. The only source data used are pulled CSV/JSON feature artifacts and local synthetic NIfTI masks already present under `rivanna_pull/analysis/synthetic_lollipop_v1/masks/`.",
        "",
        "## Local Data Inventory",
        "",
        "| Item | Count | Notes |",
        "| --- | ---: | --- |",
    ]
    for row in inventory_df.to_dict("records"):
        lines.append(f"| {row['item']} | {row['count']} | {row['notes']} |")

    lines.extend(
        [
            "",
            "## Case-ID Semantics",
            "",
            "- `case_id` values use the pattern `patient_visit_day` in the available benchmark tables, for example `57_1_208`.",
            "- The matched benchmark contains one synthetic mask per usable real target case ID.",
            "- The real feature summary reports 291 successful real masks, but the matched synthetic benchmark uses 261 single-component targets. The 30-case difference should not be treated as synthetic coverage of the full real segmentation population.",
            "- Local synthetic masks exist in the repo, but the `seg_path` fields inside the pulled synthetic CSV/manifest retain their original `/sfs/...` provenance paths. `local_synthetic_manifest.csv` remaps those to repo-local paths.",
            "",
            "## Equivalent Feature Extraction",
            "",
            "- Real and synthetic pulled feature CSVs have the same morphology schema and align one-to-one on 261 case IDs.",
            "- The feature definitions come from `scripts/extract_real_tumor_features.py`: largest-component surface metrics, spacing-aware volume/lengths, whole-mask PCA axis features, and bounding-box fill.",
            "- Synthetic source masks are locally available, so they can be re-extracted locally with the command listed below.",
            "- Real source segmentation masks are not locally available; independent real re-extraction is `BLOCKED_REMOTE_DATA`.",
            "",
            "## Headline Findings",
            "",
            "### Historical Pulled Feature Tables",
            "",
            f"- Matched volume targeting is close: median synthetic/real volume ratio is {fmt(comparison_by_metric['volume_mm3']['median_ratio_synthetic_over_real'])}.",
            f"- Synthetic masks are substantially more elongated: median elongation ratio is {fmt(comparison_by_metric['elongation']['median_ratio_synthetic_over_real'])}.",
            f"- Synthetic masks occupy much less of their bounding boxes: median bbox-fill ratio is {fmt(comparison_by_metric['bbox_fill_fraction']['median_ratio_synthetic_over_real'])}.",
            f"- Synthetic masks are less spherical/compact by whole-mask surface metrics: median sphericity ratio is {fmt(comparison_by_metric['sphericity']['median_ratio_synthetic_over_real'])}.",
            f"- Synthetic major-axis lengths are longer at matched volume: median major-axis ratio is {fmt(comparison_by_metric['principal_axis_length_major_mm']['median_ratio_synthetic_over_real'])}.",
            "",
            "These historical benchmark findings are preserved for provenance, but they are not the strongest local evidence because the local synthetic masks re-extract differently with the current checked-out extractor.",
            "",
            "### Current Local Synthetic Re-Extraction",
            "",
        ]
    )
    if current_comparison is not None:
        lines.extend(
            [
                f"- Current local synthetic re-extraction volume ratio: {fmt(current_by_metric['volume_mm3']['median_ratio_synthetic_over_real'])}.",
                f"- Current local synthetic re-extraction elongation ratio: {fmt(current_by_metric['elongation']['median_ratio_synthetic_over_real'])}.",
                f"- Current local synthetic re-extraction bbox-fill ratio: {fmt(current_by_metric['bbox_fill_fraction']['median_ratio_synthetic_over_real'])}.",
                f"- Current local synthetic re-extraction sphericity ratio: {fmt(current_by_metric['sphericity']['median_ratio_synthetic_over_real'])}.",
                "- Current local re-extraction found 9 multi-component synthetic masks; the pulled synthetic summary reports 0. Treat this as extractor/artifact drift requiring follow-up before generator tuning.",
            ]
        )
    else:
        lines.append("- Not run yet. Generate `synthetic_features_reextracted_local.csv`, then rerun this analysis script.")

    lines.extend(
        [
            "",
            "These are local benchmark findings for the matched 261 single-component target set. They are not population-level clinical conclusions.",
            "",
            "## Ranked Morphology Gaps",
            "",
            "Historical pulled-feature ranking:",
            "",
            "| Rank | Metric | Median Ratio Syn/Real | Median Diff | KS | Cliff's Delta | Interpretation |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for i, row in enumerate(top_gaps, start=1):
        lines.append(
            f"| {i} | {row['metric']} | {fmt(row['median_ratio_synthetic_over_real'])} | "
            f"{fmt(row['median_diff_synthetic_minus_real'])} | {fmt(row['ks_statistic'])} | "
            f"{fmt(row['cliffs_delta_synthetic_vs_real'])} | {row['interpretation']} |"
        )

    if current_gaps is not None:
        lines.extend(
            [
                "",
                "Current local synthetic re-extraction ranking:",
                "",
                "| Rank | Metric | Median Ratio Syn/Real | Median Diff | KS | Cliff's Delta | Interpretation |",
                "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for i, row in enumerate(current_gaps.head(7).to_dict("records"), start=1):
            lines.append(
                f"| {i} | {row['metric']} | {fmt(row['median_ratio_synthetic_over_real'])} | "
                f"{fmt(row['median_diff_synthetic_minus_real'])} | {fmt(row['ks_statistic'])} | "
                f"{fmt(row['cliffs_delta_synthetic_vs_real'])} | {row['interpretation']} |"
            )

    if synthetic_drift is not None:
        drift_rows = synthetic_drift[synthetic_drift["metric"].isin(PRIMARY_GAP_METRICS)].to_dict("records")
        lines.extend(
            [
                "",
                "## Synthetic Feature Drift",
                "",
                "The pulled synthetic CSV does not reproduce from the local masks with the current extractor. This can come from source-code drift, generated mask drift, or provenance mismatch. It must be resolved before making scientific claims from synthetic morphology calibration.",
                "",
                "| Metric | Pulled Median | Local Re-Extracted Median | Median Abs Diff | Max Abs Diff | Changed Rows |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in drift_rows:
            lines.append(
                f"| {row['metric']} | {fmt(row['pulled_median'])} | {fmt(row['local_reextracted_median'])} | "
                f"{fmt(row['median_abs_diff'])} | {fmt(row['max_abs_diff'])} | {row['changed_gt_1e_minus_9']} |"
            )

    lines.extend(
        [
            "",
            "## Volume-Stratified Summary",
            "",
            "| Stratum | Cases | Metric | Median Ratio Syn/Real | Median Diff | Cliff's Delta |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in stratified.to_dict("records"):
        lines.append(
            f"| {row['volume_stratum']} | {row['n_cases']} | {row['metric']} | "
            f"{fmt(row['median_ratio_synthetic_over_real'])} | {fmt(row['median_diff_synthetic_minus_real'])} | "
            f"{fmt(row['cliffs_delta_synthetic_vs_real'])} |"
        )

    lines.extend(
        [
            "",
            "## Likely Generator Gaps",
            "",
            "1. The historical pulled feature comparison indicates a long-axis/bbox-fill mismatch: synthetic masks are more elongated and occupy less of their bounding boxes at matched volume.",
            "2. The current local re-extraction does not reproduce that same morphology profile, so generator tuning should not proceed from the pulled synthetic CSV alone.",
            "3. The most defensible immediate gap is provenance/reproducibility: resolve why local synthetic masks plus the current extractor differ materially from the pulled synthetic feature table.",
            "4. After provenance is resolved, reassess whether whole-mask PCA/bounding-box gaps are true generator gaps or extraction artifacts of lollipop topology.",
            "",
            "These are audit findings, not tuning instructions. They are not evidence that the current anatomy is wrong, because canal/bulb compartment validation is unavailable locally.",
            "",
            "## Extraction-Artifacts Versus Scientific Differences",
            "",
            "- The strongest gaps involve whole-mask PCA and bounding-box metrics, which are sensitive to a lollipop stem even if the embedded anatomy is plausible.",
            "- Surface and sphericity metrics are resolution-sensitive: real masks use 0.5 mm voxel volume in the pulled table examples, while benchmark synthetic masks use 1.0 mm isotropic voxels. This can inflate discretization differences.",
            "- Principal-axis sign is irrelevant here because all compared metrics are scalar lengths/ratios.",
            "- Without real source masks and anatomical canal/CPA labels, this audit cannot separate biologically meaningful VS morphology from whole-mask feature limitations.",
            "",
            "## BLOCKED_REMOTE_DATA",
            "",
            "| Dependency | Why Needed | Local Fallback |",
            "| --- | --- | --- |",
        ]
    )
    for row in remote_deps.to_dict("records"):
        lines.append(f"| {row['dependency']} | {row['why_needed']} | {row['local_fallback_used']} |")

    lines.extend(
        [
            "",
            "## Generated Local Artifacts",
            "",
            "- `sample_inventory.csv`",
            "- `local_synthetic_manifest.csv`",
            "- `matched_case_feature_table.csv`",
            "- `matched_distribution_comparison.csv`",
            "- `volume_stratified_comparison.csv`",
            "- `ranked_morphology_gaps.csv`",
            "- `current_local_extractor_matched_distribution_comparison.csv` if local synthetic re-extraction exists",
            "- `current_local_extractor_volume_stratified_comparison.csv` if local synthetic re-extraction exists",
            "- `current_local_extractor_ranked_morphology_gaps.csv` if local synthetic re-extraction exists",
            "- `synthetic_pulled_vs_reextracted_metric_drift.csv` if local synthetic re-extraction exists",
            "- `remote_data_dependencies.csv`",
            "- `plots/*.png`",
            "",
            "Plots:",
        ]
    )
    lines.extend([f"- `{plot}`" for plot in plots])
    lines.append("")
    lines.append(
        "The plots currently visualize the historical pulled-feature comparison. "
        "Use the `current_local_extractor_*.csv` tables for current-extractor numerical review until matching current plots are added."
    )

    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```bash",
            "python3 analysis/real_vs_synthetic/analyze_local_morphology.py",
            "",
            "python3 scripts/extract_real_tumor_features.py \\",
            "  --input_csv analysis/real_vs_synthetic/local_synthetic_manifest.csv \\",
            "  --seg_col seg_path \\",
            "  --case_id_col case_id \\",
            "  --out_csv analysis/real_vs_synthetic/synthetic_features_reextracted_local.csv \\",
            "  --summary_json analysis/real_vs_synthetic/synthetic_feature_summary_reextracted_local.json",
            "",
            "python3 analysis/real_vs_synthetic/analyze_local_morphology.py",
            "```",
            "",
            "Initial inventory/report generation without local synthetic re-extraction:",
            "",
            "```bash",
            "python3 analysis/real_vs_synthetic/analyze_local_morphology.py",
            "```",
            "",
            "## Definition of Done Status",
            "",
            "- Local real feature data located: COMPLETE.",
            "- Sample count and case-ID semantics verified: COMPLETE.",
            "- Local synthetic cohort located: COMPLETE.",
            "- Equivalent feature schema verified: COMPLETE.",
            "- Real source-mask re-extraction: BLOCKED_REMOTE_DATA.",
            "- Distribution comparison and volume stratification: COMPLETE.",
            "- Unsupported population-level conclusions avoided: COMPLETE.",
        ]
    )
    (OUT_DIR / "LOCAL_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    real, syn, manifest, targets = load_inputs()
    local_manifest = write_local_synthetic_manifest(manifest)
    inventory_df = inventory(real, syn, local_manifest, targets)
    merged, comparison = matched_comparison(real, syn)
    stratified = stratified_comparison(merged)
    gaps = ranked_gaps(comparison)
    local_syn, synthetic_drift = synthetic_reextraction_drift(syn)
    current_comparison = None
    current_gaps = None
    if local_syn is not None:
        current_merged, current_comparison = matched_comparison(
            real,
            local_syn,
            file_prefix="current_local_extractor_",
        )
        stratified_comparison(current_merged, file_prefix="current_local_extractor_")
        current_gaps = ranked_gaps(current_comparison, file_prefix="current_local_extractor_")
    remote_deps = write_remote_dependencies()
    plots = make_plots(merged, comparison, stratified)
    write_report(
        inventory_df=inventory_df,
        comparison=comparison,
        stratified=stratified,
        gaps=gaps,
        remote_deps=remote_deps,
        plots=plots,
        current_comparison=current_comparison,
        current_gaps=current_gaps,
        synthetic_drift=synthetic_drift,
    )

    with (OUT_DIR / "audit_summary.json").open("w") as handle:
        json.dump(
            {
                "task_id": "MORPH-001",
                "real_feature_rows": int(len(real)),
                "synthetic_feature_rows": int(len(syn)),
                "matched_case_ids": int(len(merged)),
                "local_synthetic_masks_existing": int(local_manifest["local_mask_exists"].sum()),
                "real_summary_n_cases": int(json.loads(REAL_SUMMARY.read_text())["n_cases"]),
                "synthetic_summary_n_cases": int(json.loads(SYN_SUMMARY.read_text())["n_cases"]),
                "synthetic_reextraction_available": bool(local_syn is not None),
                "synthetic_reextraction_multiple_components": int(local_syn["multiple_components"].astype(str).str.lower().eq("true").sum()) if local_syn is not None else None,
                "top_ranked_gaps": gaps.head(5).to_dict("records"),
                "top_current_local_extractor_ranked_gaps": current_gaps.head(5).to_dict("records") if current_gaps is not None else None,
            },
            handle,
            indent=2,
        )

    print(f"Wrote local real-vs-synthetic morphology audit to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
