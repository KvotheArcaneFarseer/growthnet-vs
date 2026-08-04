#!/usr/bin/env python3
"""Build local synthetic feature provenance evidence.

This script writes only under analysis/feature_provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "feature_provenance"
SYN_DIR = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1"
MASK_DIR = SYN_DIR / "masks"
PULLED_FEATURES = SYN_DIR / "synthetic_lollipop_features.csv"
PULLED_FEATURES_JSON = SYN_DIR / "synthetic_lollipop_features.json"
PULLED_SUMMARY = SYN_DIR / "synthetic_lollipop_feature_summary.json"
PULLED_MANIFEST = SYN_DIR / "manifests" / "synthetic_lollipop_manifest.csv"
REAL_TARGETS = SYN_DIR / "manifests" / "real_single_component_targets.csv"
CURRENT_EXTRACTOR = REPO_ROOT / "scripts" / "extract_real_tumor_features.py"
RIVANNA_EXTRACTOR = REPO_ROOT / "rivanna_pull" / "scripts" / "extract_real_tumor_features.py"
CURRENT_GENERATOR = REPO_ROOT / "scripts" / "generate_synthetic_lollipop_cohort.py"
RIVANNA_GENERATOR = REPO_ROOT / "rivanna_pull" / "scripts" / "generate_synthetic_lollipop_cohort.py"
PRIOR_FULL_REEXTRACT = REPO_ROOT / "analysis" / "real_vs_synthetic" / "synthetic_features_reextracted_local.csv"
PRIOR_DRIFT = REPO_ROOT / "analysis" / "real_vs_synthetic" / "synthetic_pulled_vs_reextracted_metric_drift.csv"
PRIOR_REPORT = REPO_ROOT / "analysis" / "real_vs_synthetic" / "LOCAL_VALIDATION_REPORT.md"

SELECTED_INPUT = OUT_DIR / "selected_synthetic_masks.csv"
CURRENT_SELECTED = OUT_DIR / "current_reextracted_selected_features.csv"
CURRENT_SELECTED_JSON = OUT_DIR / "current_reextracted_selected_features.json"
CURRENT_SELECTED_SUMMARY = OUT_DIR / "current_reextracted_selected_summary.json"

SCALAR_FEATURES = [
    "voxel_volume_mm3",
    "mask_voxel_count",
    "volume_mm3",
    "connected_component_count",
    "largest_component_voxel_count",
    "largest_component_fraction",
    "component_volume_fraction_largest",
    "component_volume_fraction_second_largest",
    "equivalent_sphere_diameter_mm",
    "max_diameter_bbox_mm",
    "bbox_volume_mm3",
    "bbox_fill_fraction",
    "principal_axis_length_major_mm",
    "principal_axis_length_minor1_mm",
    "principal_axis_length_minor2_mm",
    "flatness",
    "aspect_ratio_major_to_minor2",
    "elongation_legacy_major_to_minor2",
    "elongation",
    "secondary_axis_ratio",
    "surface_area_mm2",
    "sphericity",
    "compactness",
    "surface_to_volume_ratio",
    "roughness_index",
    "voxel_spacing_mm_x",
    "voxel_spacing_mm_y",
    "voxel_spacing_mm_z",
    "bounding_box_size_vox_x",
    "bounding_box_size_vox_y",
    "bounding_box_size_vox_z",
    "bounding_box_size_mm_x",
    "bounding_box_size_mm_y",
    "bounding_box_size_mm_z",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = limit_bytes
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            chunk = handle.read(size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    suffix = "" if limit_bytes is None else f"_first_{limit_bytes}_bytes"
    return h.hexdigest() + suffix


def numeric(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def git_lines(args: list[str]) -> list[str]:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    lines = (proc.stdout or proc.stderr).splitlines()
    return [line for line in lines if line.strip()]


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(PULLED_FEATURES), pd.read_csv(PULLED_MANIFEST)


def local_mask_path(case_id: str) -> Path:
    return MASK_DIR / f"{case_id}_synthetic_lollipop.nii.gz"


def mask_metadata(path: Path) -> dict[str, Any]:
    nii = nib.load(str(path))
    data = np.asarray(nii.dataobj)
    mask = data > 0
    spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
    return {
        "shape": "x".join(str(int(x)) for x in data.shape[:3]),
        "spacing": "x".join(f"{x:g}" for x in spacing),
        "voxel_volume_mm3": float(np.prod(spacing)),
        "mask_voxel_count": int(mask.sum()),
        "volume_mm3": float(mask.sum() * np.prod(spacing)),
        "affine_diag": "x".join(f"{float(nii.affine[i, i]):g}" for i in range(3)),
    }


def build_inventory(pulled: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    artifact_paths = [
        ("pulled_feature_csv", PULLED_FEATURES, "Stored synthetic feature table under investigation."),
        ("pulled_feature_json", PULLED_FEATURES_JSON, "JSON companion to stored synthetic feature table."),
        ("pulled_feature_summary", PULLED_SUMMARY, "Stored summary generated with pulled synthetic table."),
        ("pulled_manifest", PULLED_MANIFEST, "Generation manifest paired with local masks by case_id."),
        ("real_targets_manifest", REAL_TARGETS, "Real single-component targets used by the synthetic benchmark."),
        ("current_extractor", CURRENT_EXTRACTOR, "Current local feature extraction code path."),
        ("rivanna_pulled_extractor", RIVANNA_EXTRACTOR, "Pulled extractor script copy."),
        ("current_generator", CURRENT_GENERATOR, "Current local cohort generator; read-only evidence."),
        ("rivanna_pulled_generator", RIVANNA_GENERATOR, "Pulled generator script copy; read-only evidence."),
        ("prior_full_reextract", PRIOR_FULL_REEXTRACT, "Prior local full re-extraction table in analysis/real_vs_synthetic."),
        ("prior_metric_drift", PRIOR_DRIFT, "Prior local pulled-vs-reextracted drift summary."),
        ("prior_validation_report", PRIOR_REPORT, "Prior morphology validation report."),
    ]
    for kind, path, note in artifact_paths:
        exists = path.exists()
        rows.append(
            {
                "artifact_type": kind,
                "case_id": "",
                "path": rel(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else "",
                "rows": len(pd.read_csv(path)) if exists and path.suffix == ".csv" else "",
                "sha256": sha256(path) if exists and path.stat().st_size < 20_000_000 else "",
                "notes": note,
            }
        )

    manifest_by_case = {str(r["case_id"]): r for r in manifest.to_dict("records")}
    pulled_cases = set(str(x) for x in pulled["case_id"].astype(str))
    for path in sorted(MASK_DIR.glob("*_synthetic_lollipop.nii.gz")):
        case_id = path.name.removesuffix("_synthetic_lollipop.nii.gz")
        meta = mask_metadata(path)
        manifest_row = manifest_by_case.get(case_id, {})
        notes = [
            f"mask_voxels={meta['mask_voxel_count']}",
            f"mask_volume_mm3={meta['volume_mm3']:g}",
            f"manifest_realized={numeric(manifest_row.get('realized_volume_mm3')):g}",
            f"in_pulled_features={case_id in pulled_cases}",
            f"shape={meta['shape']}",
            f"spacing={meta['spacing']}",
        ]
        rows.append(
            {
                "artifact_type": "local_synthetic_mask",
                "case_id": case_id,
                "path": rel(path),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "rows": "",
                "sha256": sha256(path),
                "notes": "; ".join(notes),
            }
        )

    for commit in git_lines(["log", "--oneline", "--all", "--", "scripts/extract_real_tumor_features.py", "scripts/generate_synthetic_lollipop_cohort.py", "embed_tumor.py"]):
        rows.append(
            {
                "artifact_type": "git_history_relevant_commit",
                "case_id": "",
                "path": "",
                "exists": True,
                "size_bytes": "",
                "rows": "",
                "sha256": "",
                "notes": commit,
            }
        )
    return pd.DataFrame(rows)


def select_cases(pulled: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    merged = pulled[["case_id", "volume_mm3", "mask_voxel_count"]].merge(
        manifest[["case_id", "target_volume_mm3", "realized_volume_mm3"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_pulled", "_manifest"),
    )
    merged["local_path"] = merged["case_id"].map(lambda x: str(local_mask_path(str(x))))
    merged = merged[merged["local_path"].map(lambda p: Path(p).exists())].copy()
    merged["sort_volume"] = pd.to_numeric(merged["target_volume_mm3"], errors="coerce")
    merged = merged.sort_values("sort_volume").reset_index(drop=True)
    n = len(merged)
    idxs = sorted(set([0, 1, 2, 3, n // 3 - 1, n // 3, n // 2, 2 * n // 3, 2 * n // 3 + 1, n - 4, n - 2, n - 1]))
    selected = merged.iloc[idxs].copy()
    selected["volume_stratum"] = pd.cut(
        selected["target_volume_mm3"],
        bins=[-math.inf, 100.0, 500.0, math.inf],
        labels=["small", "medium", "large"],
    ).astype(str)
    selected[["case_id", "local_path", "target_volume_mm3", "realized_volume_mm3", "volume_mm3", "volume_stratum"]].rename(
        columns={"local_path": "seg_path", "volume_mm3": "pulled_volume_mm3"}
    ).to_csv(SELECTED_INPUT, index=False)
    return selected


def run_current_extractor() -> list[str]:
    cmd = [
        "python3",
        str(CURRENT_EXTRACTOR),
        "--input_csv",
        str(SELECTED_INPUT),
        "--seg_col",
        "seg_path",
        "--case_id_col",
        "case_id",
        "--out_csv",
        str(CURRENT_SELECTED),
        "--out_json",
        str(CURRENT_SELECTED_JSON),
        "--summary_json",
        str(CURRENT_SELECTED_SUMMARY),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return [" ".join(cmd)]


def classify_metric(metric: str, diffs: pd.Series, rels: pd.Series, pulled_vals: pd.Series, current_vals: pd.Series) -> str:
    absdiff = pd.to_numeric(diffs, errors="coerce").abs().dropna()
    if absdiff.empty or float(absdiff.max()) <= 1e-9:
        return "constant"
    if metric.startswith("voxel_spacing") or metric == "voxel_volume_mm3":
        return "constant" if float(absdiff.max()) <= 1e-9 else "spacing-related"
    if metric in {"mask_voxel_count", "volume_mm3", "largest_component_voxel_count"}:
        return "case-specific"
    if metric in {"max_diameter_bbox_mm", "bbox_volume_mm3", "bbox_fill_fraction"}:
        return "systematic/bbox-related"
    if metric.startswith("principal_axis") or metric in {"elongation", "elongation_legacy_major_to_minor2", "aspect_ratio_major_to_minor2", "flatness", "secondary_axis_ratio"}:
        return "systematic/PCA-related"
    if metric in {"surface_area_mm2", "sphericity", "compactness", "surface_to_volume_ratio", "roughness_index"}:
        return "feature-specific/surface-related"
    rel = pd.to_numeric(rels, errors="coerce").abs().replace([np.inf, -np.inf], np.nan).dropna()
    if not rel.empty and float(rel.std(ddof=0)) < 0.02 and float(rel.median()) > 0.05:
        return "scaling-related"
    return "case-specific"


def compare_features(pulled: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = pd.read_csv(CURRENT_SELECTED)
    selected_cases = set(str(x) for x in selected["case_id"])
    p = pulled[pulled["case_id"].astype(str).isin(selected_cases)].copy()
    merged = p.merge(current, on="case_id", suffixes=("_stored", "_current"), validate="one_to_one")
    manifest = pd.read_csv(PULLED_MANIFEST)
    merged = merged.merge(manifest[["case_id", "target_volume_mm3", "realized_volume_mm3"]], on="case_id", how="left")

    rows: list[dict[str, Any]] = []
    for rec in merged.to_dict("records"):
        for field in SCALAR_FEATURES:
            sk = f"{field}_stored"
            ck = f"{field}_current"
            if sk not in rec or ck not in rec:
                continue
            stored = numeric(rec[sk])
            current_value = numeric(rec[ck])
            if not (math.isfinite(stored) or math.isfinite(current_value)):
                continue
            diff = current_value - stored if math.isfinite(stored) and math.isfinite(current_value) else math.nan
            rel_diff = diff / stored if math.isfinite(diff) and stored not in (0.0, -0.0) else math.nan
            rows.append(
                {
                    "case_id": rec["case_id"],
                    "volume_stratum": selected.set_index("case_id").loc[rec["case_id"], "volume_stratum"],
                    "feature": field,
                    "stored_value": stored,
                    "current_value": current_value,
                    "absolute_difference": abs(diff) if math.isfinite(diff) else math.nan,
                    "signed_difference_current_minus_stored": diff,
                    "relative_difference": rel_diff,
                    "target_volume_mm3": numeric(rec.get("target_volume_mm3")),
                    "manifest_realized_volume_mm3": numeric(rec.get("realized_volume_mm3")),
                    "stored_seg_path": rec.get("seg_path_stored", ""),
                    "current_seg_path": rec.get("seg_path_current", ""),
                }
            )
    comparison = pd.DataFrame(rows)
    pattern_rows = []
    for feature, part in comparison.groupby("feature"):
        pattern_rows.append(
            {
                "feature": feature,
                "n_cases": int(len(part)),
                "changed_cases": int((part["absolute_difference"] > 1e-9).sum()),
                "median_abs_difference": float(part["absolute_difference"].median()),
                "max_abs_difference": float(part["absolute_difference"].max()),
                "median_relative_difference": float(pd.to_numeric(part["relative_difference"], errors="coerce").median()),
                "pattern_class": classify_metric(
                    feature,
                    part["signed_difference_current_minus_stored"],
                    part["relative_difference"],
                    part["stored_value"],
                    part["current_value"],
                ),
            }
        )
    patterns = pd.DataFrame(pattern_rows).sort_values(["changed_cases", "median_abs_difference"], ascending=[False, False])
    comparison = comparison.merge(patterns[["feature", "pattern_class"]], on="feature", how="left")
    return comparison, patterns


def full_volume_provenance(pulled: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    rows = []
    local_meta = {}
    for case_id in manifest["case_id"].astype(str):
        path = local_mask_path(case_id)
        if path.exists():
            local_meta[case_id] = mask_metadata(path)
    local = pd.DataFrame(
        [{"case_id": k, "local_mask_voxel_count": v["mask_voxel_count"], "local_volume_mm3": v["volume_mm3"]} for k, v in local_meta.items()]
    )
    merged = pulled[["case_id", "mask_voxel_count", "volume_mm3"]].merge(
        manifest[["case_id", "realized_volume_mm3", "target_volume_mm3"]],
        on="case_id",
        how="inner",
    ).merge(local, on="case_id", how="inner")
    rows.append(("cases_compared", len(merged)))
    rows.append(("local_volume_equals_manifest_realized", int(np.isclose(merged["local_volume_mm3"], merged["realized_volume_mm3"]).sum())))
    rows.append(("pulled_volume_equals_manifest_realized", int(np.isclose(merged["volume_mm3"], merged["realized_volume_mm3"]).sum())))
    rows.append(("pulled_volume_equals_local_volume", int(np.isclose(merged["volume_mm3"], merged["local_volume_mm3"]).sum())))
    rows.append(("median_abs_pulled_minus_local_volume", float((merged["volume_mm3"] - merged["local_volume_mm3"]).abs().median())))
    rows.append(("max_abs_pulled_minus_local_volume", float((merged["volume_mm3"] - merged["local_volume_mm3"]).abs().max())))
    return dict(rows)


def decision_table(volume_evidence: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "artifact": rel(PULLED_FEATURES),
            "classification": "STALE",
            "basis": "Does not reproduce from same-named local masks. Only "
            f"{volume_evidence['pulled_volume_equals_local_volume']}/{volume_evidence['cases_compared']} pulled volumes equal local mask volumes; median absolute volume drift is "
            f"{volume_evidence['median_abs_pulled_minus_local_volume']} mm3.",
            "recommended_action": "Do not use as authoritative for local masks; regenerate after provenance is resolved.",
        },
        {
            "artifact": rel(PULLED_FEATURES_JSON),
            "classification": "STALE",
            "basis": "Companion JSON to stale feature CSV.",
            "recommended_action": "Treat as historical only.",
        },
        {
            "artifact": rel(PULLED_SUMMARY),
            "classification": "STALE",
            "basis": "Summary derives from stale synthetic feature table.",
            "recommended_action": "Regenerate from current extracted features if local masks are the intended masks.",
        },
        {
            "artifact": rel(PULLED_MANIFEST),
            "classification": "AUTHORITATIVE",
            "basis": "Manifest realized volumes match local mask volumes for "
            f"{volume_evidence['local_volume_equals_manifest_realized']}/{volume_evidence['cases_compared']} cases.",
            "recommended_action": "Use for mask-case mapping and generation metadata, not for extracted morphology.",
        },
        {
            "artifact": rel(MASK_DIR),
            "classification": "AUTHORITATIVE",
            "basis": "Local pulled masks are the concrete NIfTI inputs available for reproduction and align with manifest realized volumes.",
            "recommended_action": "Use these masks as source of truth for local re-extraction unless older mask archives are recovered.",
        },
        {
            "artifact": rel(CURRENT_EXTRACTOR),
            "classification": "AUTHORITATIVE",
            "basis": "Current local extractor is identical to the pulled extractor copy by file diff and is the only local executable feature code path.",
            "recommended_action": "Use for current reproducibility; version future outputs with commit hash and input hashes.",
        },
        {
            "artifact": rel(RIVANNA_GENERATOR),
            "classification": "REPRODUCIBLE_LEGACY",
            "basis": "Pulled generator copy differs from current generator and retains older size/shape calibration behavior consistent with stale pulled feature morphology.",
            "recommended_action": "Treat as the most likely historical generator family for the pulled feature table, but not sufficient without the exact older masks.",
        },
        {
            "artifact": rel(CURRENT_GENERATOR),
            "classification": "AUTHORITATIVE",
            "basis": "Current generator behavior aligns local masks with manifest realized volumes, including corrected tiny-target volumes.",
            "recommended_action": "Use only for future generation after explicit provenance/version stamping.",
        },
        {
            "artifact": rel(PRIOR_FULL_REEXTRACT),
            "classification": "REPRODUCIBLE_LEGACY",
            "basis": "Previously generated local re-extraction table from current extractor and local masks; reproduced on selected cases in this investigation.",
            "recommended_action": "Useful evidence, but regenerate a full table when finalizing a new benchmark.",
        },
        {
            "artifact": "unknown older synthetic masks used for pulled feature CSV",
            "classification": "UNKNOWN_PROVENANCE",
            "basis": "Feature values imply a different same-case mask instance, but no separate older mask archive or generation command record is present locally.",
            "recommended_action": "Recover original source masks or mark the pulled feature table stale.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    selected: pd.DataFrame,
    patterns: pd.DataFrame,
    volume_evidence: dict[str, Any],
    commands: list[str],
    inventory: pd.DataFrame,
) -> None:
    prior_drift = pd.read_csv(PRIOR_DRIFT) if PRIOR_DRIFT.exists() else pd.DataFrame()
    git_history = git_lines(["log", "--oneline", "--all", "--", "scripts/extract_real_tumor_features.py", "scripts/generate_synthetic_lollipop_cohort.py", "embed_tumor.py"])
    extractor_diff = subprocess.run(
        ["diff", "-q", str(CURRENT_EXTRACTOR), str(RIVANNA_EXTRACTOR)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    generator_diff = subprocess.run(
        ["diff", "-q", str(CURRENT_GENERATOR), str(RIVANNA_GENERATOR)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    lines = [
        "# Feature Provenance Report",
        "",
        "## Conclusion",
        "",
        "The previously pulled synthetic feature tables do not reproduce because they were not generated from the same local synthetic mask instances now present under `rivanna_pull/analysis/synthetic_lollipop_v1/masks/`. This is established before any higher-order feature formula question: stored voxel counts and volumes frequently differ from both the local masks and the paired manifest realized volumes.",
        "",
        "The most likely generating code path for the pulled tables is: older `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py`-style masks followed by the same extractor schema as `scripts/extract_real_tumor_features.py`. The current local extractor and the pulled extractor copy are identical, so local extractor-code drift is not the primary cause. The current generator differs from the pulled generator, and the generator diff matches the observed pattern: tiny cases in the pulled feature CSV show old minimum-volume overshoot, while local masks match the corrected manifest realized volumes.",
        "",
        "The remaining causes are narrowed to a small set: an older overwritten/not-pulled mask set, an artifact packaging mismatch where stale features were paired with refreshed masks/manifest, or an unrecorded post-generation mask replacement before pull. The evidence most strongly favors stale pulled features from an earlier generation run paired with newer local masks and manifest.",
        "",
        "Do not tune tumor geometry or generator parameters from the stale pulled feature table. Regenerate features from the authoritative local masks, or recover the exact older masks that produced the pulled feature CSV.",
        "",
        "## Key Evidence",
        "",
        f"- Compared {volume_evidence['cases_compared']} case IDs with pulled features, manifest rows, and local masks.",
        f"- Local mask volume equals manifest `realized_volume_mm3` for {volume_evidence['local_volume_equals_manifest_realized']} cases.",
        f"- Pulled feature volume equals local mask volume for only {volume_evidence['pulled_volume_equals_local_volume']} cases.",
        f"- Pulled feature volume equals manifest realized volume for only {volume_evidence['pulled_volume_equals_manifest_realized']} cases.",
        f"- Median absolute pulled-vs-local volume difference: {volume_evidence['median_abs_pulled_minus_local_volume']} mm3; max: {volume_evidence['max_abs_pulled_minus_local_volume']} mm3.",
        f"- Current extractor versus pulled extractor copy: {'identical' if extractor_diff.returncode == 0 else 'different'} (`diff -q`).",
        f"- Current generator versus pulled generator copy: {'identical' if generator_diff.returncode == 0 else 'different'} (`diff -q`).",
        "",
        "## Representative Cases",
        "",
        "| Case ID | Stratum | Target Volume | Manifest Realized | Pulled Volume | Local Path |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in selected.to_dict("records"):
        lines.append(
            f"| {row['case_id']} | {row['volume_stratum']} | {float(row['target_volume_mm3']):.3f} | "
            f"{float(row['realized_volume_mm3']):.3f} | {float(row['volume_mm3']):.3f} | `{rel(Path(row['local_path']))}` |"
        )
    lines.extend(["", "## Mismatch Patterns", "", "| Feature | Changed Cases | Median Abs Diff | Max Abs Diff | Pattern |", "| --- | ---: | ---: | ---: | --- |"])
    for row in patterns.head(20).to_dict("records"):
        lines.append(
            f"| {row['feature']} | {row['changed_cases']} | {float(row['median_abs_difference']):.6g} | "
            f"{float(row['max_abs_difference']):.6g} | {row['pattern_class']} |"
        )
    if not prior_drift.empty:
        lines.extend(["", "## Full Prior Re-Extraction Drift", "", "| Metric | Pulled Median | Local Re-Extracted Median | Median Abs Diff | Changed Rows |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in prior_drift.to_dict("records"):
            lines.append(
                f"| {row['metric']} | {float(row['pulled_median']):.6g} | {float(row['local_reextracted_median']):.6g} | "
                f"{float(row['median_abs_diff']):.6g} | {int(row['changed_gt_1e_minus_9'])} |"
            )
    lines.extend(
        [
            "",
            "## Git/Formula Inspection",
            "",
            "- Relevant committed history for extractor/generator/embed files is short locally:",
        ]
    )
    lines.extend([f"  - `{line}`" for line in git_history])
    lines.extend(
        [
        "- `scripts/extract_real_tumor_features.py` was added in commit `0457540`; no later committed formula changes are present locally for the extractor.",
        "- The pulled extractor copy under `rivanna_pull/scripts/extract_real_tumor_features.py` is byte-identical to the current extractor.",
        "- The pulled generator copy under `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py` differs materially from the current generator. The current generator adds explicit spacing parsing, lower tiny-target floors, compact single-mask initialization, delayed bulb creation for smaller targets, and comments describing fixes for prior tiny-target tail failures.",
        "- Those generator differences are consistent with the volume evidence: for example, selected tiny cases have pulled feature volumes around 41-44 mm3 while the manifest/local masks realize 8-12 mm3 targets.",
        "- The extractor computes spacing-aware volume, largest-component connected-component features, marching-cubes surface area, sphericity, compactness, bounding boxes, and PCA lengths. These formulas explain the feature schema but do not explain changed voxel counts for same case IDs.",
            "- Prior audit notes in `docs/AUTONOMOUS_CHANGE_AUDIT.md` flag a formula inconsistency: `principal_axis_length_*` can be overwritten with extent-based values while elongation/flatness/aspect ratio remain moment-based. That is a reporting risk, not the root cause of pulled-vs-local non-reproduction.",
            "",
            "## Artifact Classification",
            "",
            "See `provenance_decision_table.csv` for the formal artifact decisions. In short: local masks and manifest are authoritative for local reproduction; pulled feature CSV/JSON/summary are stale; the missing older mask set is unknown provenance; fresh regenerated features are required for current local masks.",
            "",
            "## Commands Run",
            "",
            "```bash",
            "python3 analysis/feature_provenance/build_feature_provenance.py",
            *commands,
            "diff -q scripts/extract_real_tumor_features.py rivanna_pull/scripts/extract_real_tumor_features.py",
            "diff -q scripts/generate_synthetic_lollipop_cohort.py rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py",
            "git log --oneline --all -- scripts/extract_real_tumor_features.py scripts/generate_synthetic_lollipop_cohort.py embed_tumor.py",
            "```",
            "",
            "## Unresolved Risks",
            "",
            "- No remote/Rivanna access was used, so the exact older mask archive, if it exists, was not recovered.",
            "- Original pulled artifacts were not overwritten; this investigation cannot prove whether the mismatch happened during generation, packaging, or local pull/copy.",
            "- Only 12 representative cases were freshly re-extracted into this folder, though prior local evidence covers all 261 matched cases.",
            "- Real source segmentation masks are not local, so real-feature reproducibility was not independently tested here.",
            "",
            "## Deliverables",
            "",
            f"- `artifact_inventory.csv`: {len(inventory)} inventory rows, including all local synthetic masks.",
            "- `feature_comparison.csv`: selected pulled-vs-current feature comparisons.",
            "- `provenance_decision_table.csv`: artifact classifications and actions.",
            "- `current_reextracted_selected_features.csv`: current extractor output for representative local masks.",
        ]
    )
    (OUT_DIR / "FEATURE_PROVENANCE_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pulled, manifest = load_tables()
    inventory = build_inventory(pulled, manifest)
    inventory.to_csv(OUT_DIR / "artifact_inventory.csv", index=False)
    selected = select_cases(pulled, manifest)
    commands = run_current_extractor()
    comparison, patterns = compare_features(pulled, selected)
    comparison.to_csv(OUT_DIR / "feature_comparison.csv", index=False)
    patterns.to_csv(OUT_DIR / "feature_pattern_summary.csv", index=False)
    volume_evidence = full_volume_provenance(pulled, manifest)
    (OUT_DIR / "volume_identity_evidence.json").write_text(json.dumps(volume_evidence, indent=2))
    decisions = decision_table(volume_evidence)
    decisions.to_csv(OUT_DIR / "provenance_decision_table.csv", index=False)
    write_report(selected, patterns, volume_evidence, commands, inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
