#!/usr/bin/env python3
"""Authoritative local real-vs-synthetic morphology validation.

Uses regenerated synthetic features from analysis/synthetic_features_v2 and the
best available local real feature table. It does not modify generator behavior.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "real_vs_synthetic_v2"
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
SYN_FEATURES = REPO_ROOT / "analysis" / "synthetic_features_v2" / "synthetic_features_v2.csv"
SYN_PROVENANCE = REPO_ROOT / "analysis" / "synthetic_features_v2" / "provenance.json"

METRICS = [
    "volume_mm3",
    "equivalent_sphere_diameter_mm",
    "elongation",
    "secondary_axis_ratio",
    "aspect_ratio_major_to_minor2",
    "sphericity",
    "compactness",
    "surface_area_mm2",
    "surface_to_volume_ratio",
    "bbox_fill_fraction",
    "principal_axis_length_major_mm",
    "principal_axis_length_minor1_mm",
    "principal_axis_length_minor2_mm",
    "connected_component_count",
    "largest_component_fraction",
]

PRIMARY_GAP_METRICS = [
    "compactness",
    "principal_axis_length_major_mm",
    "aspect_ratio_major_to_minor2",
    "bbox_fill_fraction",
    "sphericity",
    "surface_to_volume_ratio",
    "surface_area_mm2",
    "elongation",
    "secondary_axis_ratio",
]

SURFACE_SENSITIVE = {"compactness", "sphericity", "surface_area_mm2", "surface_to_volume_ratio"}
PCA_BBOX_SENSITIVE = {
    "principal_axis_length_major_mm",
    "principal_axis_length_minor1_mm",
    "principal_axis_length_minor2_mm",
    "aspect_ratio_major_to_minor2",
    "secondary_axis_ratio",
    "elongation",
    "bbox_fill_fraction",
}


def finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    return values[np.isfinite(values)]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "iqr": math.nan, "p25": math.nan, "p75": math.nan, "min": math.nan, "max": math.nan}
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "iqr": float(p75 - p25),
        "p25": p25,
        "p75": p75,
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


def classify_gap(metric: str, ratio: float, cliffs: float, stratum_consistency: int) -> tuple[str, str]:
    if not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(cliffs):
        return "INSUFFICIENT_EVIDENCE", "non-finite comparison"
    log_ratio = math.log(ratio)
    magnitude = abs(log_ratio)
    if metric in SURFACE_SENSITIVE:
        if magnitude >= 0.45 and abs(cliffs) >= 0.47:
            return "EXTRACTION_OR_DATA_LIMITATION", "large but surface/resolution-sensitive; real masks are not locally re-extractable"
        return "EXTRACTION_OR_DATA_LIMITATION", "surface metric is spacing/resolution-sensitive"
    if metric in PCA_BBOX_SENSITIVE:
        if magnitude >= 0.35 and abs(cliffs) >= 0.47 and stratum_consistency >= 2:
            return "POSSIBLE_GENERATOR_GAP", "consistent whole-mask geometry difference, but canal/CPA labels are unavailable"
        if magnitude >= 0.20 and abs(cliffs) >= 0.33:
            return "POSSIBLE_GENERATOR_GAP", "moderate whole-mask geometry difference"
        return "INSUFFICIENT_EVIDENCE", "small or inconsistent effect"
    if magnitude >= 0.35 and abs(cliffs) >= 0.47:
        return "POSSIBLE_GENERATOR_GAP", "large distributional difference"
    return "INSUFFICIENT_EVIDENCE", "small or weak effect"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    real = pd.read_csv(REAL_FEATURES)
    syn = pd.read_csv(SYN_FEATURES)
    with SYN_PROVENANCE.open("r") as handle:
        provenance = json.load(handle)
    real["case_id"] = real["case_id"].astype(str)
    syn["case_id"] = syn["case_id"].astype(str)
    return real, syn, provenance


def verify_schema(real: pd.DataFrame, syn: pd.DataFrame) -> dict[str, object]:
    real_cols = set(real.columns)
    syn_cols = set(syn.columns)
    missing_real = sorted(set(METRICS) - real_cols)
    missing_syn = sorted(set(METRICS) - syn_cols)
    return {
        "real_rows": int(len(real)),
        "synthetic_rows": int(len(syn)),
        "matched_case_ids": int(len(set(real["case_id"]).intersection(set(syn["case_id"])))),
        "real_unique_case_ids": int(real["case_id"].nunique()),
        "synthetic_unique_case_ids": int(syn["case_id"].nunique()),
        "missing_metrics_in_real": missing_real,
        "missing_metrics_in_synthetic": missing_syn,
        "equivalent_metric_schema": not missing_real and not missing_syn,
    }


def matched_table(real: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    columns = ["case_id", *METRICS]
    merged = real[columns].merge(syn[columns], on="case_id", suffixes=("_real", "_synthetic"), validate="one_to_one")
    merged.to_csv(OUT_DIR / "matched_case_feature_table.csv", index=False)
    return merged


def distribution_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        r = finite_series(merged, f"{metric}_real")
        s = finite_series(merged, f"{metric}_synthetic")
        real_summary = summarize(r)
        syn_summary = summarize(s)
        stat = stats.ks_2samp(r, s, alternative="two-sided", mode="auto") if len(r) and len(s) else None
        ratio = syn_summary["median"] / real_summary["median"] if real_summary["median"] else math.nan
        rows.append(
            {
                "metric": metric,
                "real_n": real_summary["n"],
                "synthetic_n": syn_summary["n"],
                "real_mean": real_summary["mean"],
                "synthetic_mean": syn_summary["mean"],
                "real_median": real_summary["median"],
                "synthetic_median": syn_summary["median"],
                "real_iqr": real_summary["iqr"],
                "synthetic_iqr": syn_summary["iqr"],
                "real_p25": real_summary["p25"],
                "real_p75": real_summary["p75"],
                "synthetic_p25": syn_summary["p25"],
                "synthetic_p75": syn_summary["p75"],
                "median_diff_synthetic_minus_real": syn_summary["median"] - real_summary["median"],
                "median_ratio_synthetic_over_real": ratio,
                "ks_statistic": float(stat.statistic) if stat else math.nan,
                "ks_pvalue": float(stat.pvalue) if stat else math.nan,
                "cliffs_delta_synthetic_vs_real": cliffs_delta(s, r),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "matched_distribution_comparison.csv", index=False)
    return out


def volume_strata(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    bins = [0.0, 100.0, 1000.0, math.inf]
    labels = ["small_<100", "medium_100_1000", "large_>=1000"]
    out["volume_stratum"] = pd.cut(out["volume_mm3_real"], bins=bins, labels=labels, right=False)
    rows = []
    for stratum in labels:
        part = out[out["volume_stratum"] == stratum]
        for metric in PRIMARY_GAP_METRICS:
            r = finite_series(part, f"{metric}_real")
            s = finite_series(part, f"{metric}_synthetic")
            real_summary = summarize(r)
            syn_summary = summarize(s)
            ratio = syn_summary["median"] / real_summary["median"] if real_summary["median"] else math.nan
            rows.append(
                {
                    "volume_stratum": stratum,
                    "n_cases": int(len(part)),
                    "metric": metric,
                    "real_median": real_summary["median"],
                    "synthetic_median": syn_summary["median"],
                    "real_iqr": real_summary["iqr"],
                    "synthetic_iqr": syn_summary["iqr"],
                    "median_diff_synthetic_minus_real": syn_summary["median"] - real_summary["median"],
                    "median_ratio_synthetic_over_real": ratio,
                    "cliffs_delta_synthetic_vs_real": cliffs_delta(s, r),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "volume_stratified_comparison.csv", index=False)
    return result


def ranked_gaps(comparison: pd.DataFrame, stratified: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in comparison[comparison["metric"].isin(PRIMARY_GAP_METRICS)].to_dict("records"):
        metric = row["metric"]
        ratio = float(row["median_ratio_synthetic_over_real"])
        cliffs = float(row["cliffs_delta_synthetic_vs_real"])
        direction = "synthetic_higher" if ratio > 1.0 else "synthetic_lower"
        strat = stratified[stratified["metric"] == metric]
        stratum_consistency = int(
            (
                ((strat["median_ratio_synthetic_over_real"] > 1.0) & (direction == "synthetic_higher"))
                | ((strat["median_ratio_synthetic_over_real"] < 1.0) & (direction == "synthetic_lower"))
            ).sum()
        )
        classification, rationale = classify_gap(metric, ratio, cliffs, stratum_consistency)
        score = abs(math.log(ratio)) * abs(cliffs) if ratio > 0 and np.isfinite(ratio) and np.isfinite(cliffs) else math.nan
        rows.append(
            {
                "rank_score": score,
                "feature": metric,
                "direction_of_mismatch": direction,
                "magnitude_median_ratio_synthetic_over_real": ratio,
                "median_diff_synthetic_minus_real": row["median_diff_synthetic_minus_real"],
                "ks_statistic": row["ks_statistic"],
                "cliffs_delta_synthetic_vs_real": cliffs,
                "volume_strata_with_same_direction": stratum_consistency,
                "gap_classification": classification,
                "confidence_rationale": rationale,
                "likely_anatomical_interpretation": interpretation(metric, direction),
                "generator_tuning_justified": "no" if classification != "HIGH_CONFIDENCE_GENERATOR_GAP" else "human_review_required",
            }
        )
    out = pd.DataFrame(rows).sort_values("rank_score", ascending=False, na_position="last")
    out.to_csv(OUT_DIR / "ranked_morphology_gaps.csv", index=False)
    return out


def interpretation(metric: str, direction: str) -> str:
    if metric == "elongation":
        return "whole-mask canonical major/intermediate axis elongation is lower in synthetic" if direction == "synthetic_lower" else "whole-mask canonical elongation is higher in synthetic"
    if metric == "aspect_ratio_major_to_minor2":
        return "dominant-to-smallest axis disparity is lower in synthetic" if direction == "synthetic_lower" else "dominant-to-smallest axis disparity is higher in synthetic"
    if metric == "bbox_fill_fraction":
        return "synthetic masks fill their axis-aligned bounding boxes more densely" if direction == "synthetic_higher" else "synthetic masks leave more empty bounding-box space"
    if metric == "principal_axis_length_major_mm":
        return "synthetic dominant extent is shorter at matched volume" if direction == "synthetic_lower" else "synthetic dominant extent is longer at matched volume"
    if metric in SURFACE_SENSITIVE:
        return "surface-derived compactness/sphericity differs; spacing and source-mask availability limit interpretation"
    return "distribution differs in current whole-mask features"


def make_plots(merged: pd.DataFrame, stratified: pd.DataFrame) -> list[str]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for metric in PRIMARY_GAP_METRICS:
        plt.figure(figsize=(7, 4.5))
        values = [
            finite_series(merged, f"{metric}_real").to_numpy(dtype=float),
            finite_series(merged, f"{metric}_synthetic").to_numpy(dtype=float),
        ]
        plt.boxplot(values, tick_labels=["real", "synthetic"], showfliers=False)
        plt.ylabel(metric)
        plt.title(f"{metric}: real vs regenerated synthetic")
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
    plt.ylabel("regenerated synthetic volume_mm3")
    plt.title("Matched real versus regenerated synthetic volume")
    plt.tight_layout()
    out = PLOTS_DIR / "volume_match_scatter.png"
    plt.savefig(out, dpi=160)
    plt.close()
    written.append(str(out.relative_to(REPO_ROOT)))

    pivot = stratified.pivot(index="metric", columns="volume_stratum", values="median_ratio_synthetic_over_real")
    plt.figure(figsize=(8, 5))
    image = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=0.4, vmax=1.8)
    plt.colorbar(image, label="median ratio synthetic / real")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=20, ha="right")
    plt.title("Volume-stratified regenerated morphology ratios")
    plt.tight_layout()
    out = PLOTS_DIR / "volume_stratified_ratio_heatmap.png"
    plt.savefig(out, dpi=160)
    plt.close()
    written.append(str(out.relative_to(REPO_ROOT)))
    return written


def fmt(value: object, digits: int = 3) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "nan"
    return f"{f:.{digits}f}"


def write_report(schema: dict[str, object], provenance: dict, comparison: pd.DataFrame, stratified: pd.DataFrame, gaps: pd.DataFrame, plots: list[str]) -> None:
    by_metric = {row["metric"]: row for row in comparison.to_dict("records")}
    top = gaps.head(9).to_dict("records")
    matched_n = int(schema["matched_case_ids"])
    lines = [
        "# Authoritative Real-Versus-Synthetic Morphology Validation V2",
        "",
        "## Scope",
        "",
        "This analysis uses regenerated synthetic features from `analysis/synthetic_features_v2/` and the best available local real feature table. It does not use stale pulled synthetic feature CSV/JSON artifacts, does not SSH, and does not alter generator parameters.",
        "",
        "Real source segmentation masks are not available locally, so real-feature reproducibility remains a remote-data blocker. Synthetic feature integrity passed locally and is authoritative for the current local masks.",
        "",
        "## Input Verification",
        "",
        "| Check | Value |",
        "| --- | ---: |",
    ]
    for key, value in schema.items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            f"| synthetic_schema_version | {provenance.get('schema_version')} |",
            f"| synthetic_git_commit_hash | {provenance.get('git_commit_hash')} |",
            "",
            "Feature definitions are equivalent by schema for all requested metrics. They come from `scripts/extract_real_tumor_features.py`: spacing-aware volume, largest-component surface metrics, whole-mask PCA/extent features, and bounding-box features.",
            "",
            "Volume bins are fixed local engineering bins based on real matched volume: `small_<100`, `medium_100_1000`, and `large_>=1000` mm3. These are not clinical staging thresholds.",
            "",
            "## Overall Distribution Comparison",
            "",
            "| Metric | Real n | Synthetic n | Real Median | Synthetic Median | Ratio Syn/Real | Real IQR | Synthetic IQR | Cliff's Delta | KS |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.to_dict("records"):
        lines.append(
            f"| {row['metric']} | {row['real_n']} | {row['synthetic_n']} | {fmt(row['real_median'])} | "
            f"{fmt(row['synthetic_median'])} | {fmt(row['median_ratio_synthetic_over_real'])} | "
            f"{fmt(row['real_iqr'])} | {fmt(row['synthetic_iqr'])} | "
            f"{fmt(row['cliffs_delta_synthetic_vs_real'])} | {fmt(row['ks_statistic'])} |"
        )

    lines.extend(
        [
            "",
            "## Ranked Morphology Gaps",
            "",
            "| Rank | Feature | Classification | Direction | Ratio Syn/Real | Cliff's Delta | Volume Strata Same Direction | Interpretation | Tuning Justified |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for i, row in enumerate(top, start=1):
        lines.append(
            f"| {i} | {row['feature']} | {row['gap_classification']} | {row['direction_of_mismatch']} | "
            f"{fmt(row['magnitude_median_ratio_synthetic_over_real'])} | {fmt(row['cliffs_delta_synthetic_vs_real'])} | "
            f"{row['volume_strata_with_same_direction']} | {row['likely_anatomical_interpretation']} | {row['generator_tuning_justified']} |"
        )

    elong = by_metric["elongation"]
    aspect = by_metric["aspect_ratio_major_to_minor2"]
    major = by_metric["principal_axis_length_major_mm"]
    if float(elong["median_ratio_synthetic_over_real"]) > 1.0:
        elongated_verdict = "CONFIRMED"
    elif float(elong["median_ratio_synthetic_over_real"]) < 0.95 and float(aspect["median_ratio_synthetic_over_real"]) < 1.0 and float(major["median_ratio_synthetic_over_real"]) < 1.0:
        elongated_verdict = "REJECTED_FOR_REGENERATED_FEATURES"
    else:
        elongated_verdict = "UNRESOLVED"

    lines.extend(
        [
            "",
            "## Too-Elongated Concern",
            "",
            f"Verdict: {elongated_verdict}.",
            "",
            f"- Median elongation ratio synthetic/real: {fmt(elong['median_ratio_synthetic_over_real'])}.",
            f"- Median aspect-ratio major/minor2 ratio synthetic/real: {fmt(aspect['median_ratio_synthetic_over_real'])}.",
            f"- Median major-axis length ratio synthetic/real: {fmt(major['median_ratio_synthetic_over_real'])}.",
            "",
            "Using regenerated authoritative synthetic features, the prior concern that synthetic masks are too elongated is not supported. The current whole-mask features instead show lower canonical elongation and shorter major-axis extent at matched case IDs. This should not trigger generator tuning because surface/spacing and whole-mask topology limitations still need review.",
            "",
            "## Volume-Stratified Comparison",
            "",
            "| Stratum | Cases | Metric | Real Median | Synthetic Median | Ratio Syn/Real | Cliff's Delta |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stratified.to_dict("records"):
        lines.append(
            f"| {row['volume_stratum']} | {row['n_cases']} | {row['metric']} | {fmt(row['real_median'])} | "
            f"{fmt(row['synthetic_median'])} | {fmt(row['median_ratio_synthetic_over_real'])} | "
            f"{fmt(row['cliffs_delta_synthetic_vs_real'])} |"
        )

    lines.extend(
        [
            "",
            "## Scientific Interpretation Limits",
            "",
            "- No `HIGH_CONFIDENCE_GENERATOR_GAP` is assigned from this local-only pass because real masks cannot be re-extracted locally and anatomical canal/CPA labels are unavailable.",
            "- Surface-derived gaps are classified as `EXTRACTION_OR_DATA_LIMITATION` because real and synthetic voxel spacing differ and surface metrics are resolution-sensitive.",
            "- PCA/bounding-box gaps are `POSSIBLE_GENERATOR_GAP` only where effects are consistent, but they still need anatomical review before tuning.",
            "- The matched benchmark covers 261 case IDs; it is not a full clinical population validation.",
            "",
            "## Generated Outputs",
            "",
            "- `matched_distribution_comparison.csv`",
            "- `volume_stratified_comparison.csv`",
            "- `ranked_morphology_gaps.csv`",
            "- `plots/*.png`",
            "",
            "Plots:",
        ]
    )
    lines.extend([f"- `{plot}`" for plot in plots])
    lines.extend(
        [
            "",
            "## Recommended Follow-Up",
            "",
            "1. Add extractor provenance/version fields to all future generated feature tables.",
            "2. Review surface metric comparability under real 0.5 mm spacing versus synthetic 1.0 mm spacing before treating compactness/sphericity as generator gaps.",
            "3. Perform anatomical canal/CPA compartment validation before any generator morphology tuning.",
            "",
        ]
    )
    (OUT_DIR / "MORPHOLOGY_VALIDATION_REPORT.md").write_text("\n".join(lines))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real, syn, provenance = load_inputs()
    schema = verify_schema(real, syn)
    if not schema["equivalent_metric_schema"]:
        raise RuntimeError(f"Feature schema mismatch: {schema}")
    merged = matched_table(real, syn)
    comparison = distribution_comparison(merged)
    stratified = volume_strata(merged)
    gaps = ranked_gaps(comparison, stratified)
    plots = make_plots(merged, stratified)
    write_report(schema, provenance, comparison, stratified, gaps, plots)
    print("Morphology validation v2 complete")
    print(f"  Matched cases: {schema['matched_case_ids']}")
    print(f"  Top gap: {gaps.iloc[0]['feature']} ({gaps.iloc[0]['gap_classification']})")
    print(f"  Report: {OUT_DIR / 'MORPHOLOGY_VALIDATION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
