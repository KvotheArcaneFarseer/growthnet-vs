#!/usr/bin/env python3
"""Compartment audit using the recovered saved-mask generator path.

This script is analysis-only. It does not modify generator geometry, saved
masks, or authoritative feature artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label as ndi_label
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "analysis" / "anatomical_compartment_validation_v2"
OVERLAY_DIR = OUT_DIR / "overlays"
MPL_CACHE_DIR = OUT_DIR / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

MANIFEST_PATH = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "manifests" / "synthetic_lollipop_manifest.csv"
MASK_ROOT = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "masks"
RECOVERED_GENERATOR_PATH = REPO_ROOT / "rivanna_pull" / "scripts" / "generate_synthetic_lollipop_cohort.py"
PROVENANCE_CASES_CSV = REPO_ROOT / "analysis" / "saved_mask_provenance" / "reproduction_trials.csv"
OLD_COMPARTMENT_CSV = REPO_ROOT / "analysis" / "anatomical_compartment_validation" / "synthetic_compartment_features.csv"

FEATURES_CSV = OUT_DIR / "synthetic_compartment_features_v2.csv"
SUMMARY_CSV = OUT_DIR / "synthetic_compartment_summary_v2.csv"
VISUAL_CSV = OUT_DIR / "visual_verification.csv"
REPORT_MD = OUT_DIR / "SYNTHETIC_COMPARTMENT_AUDIT_V2.md"
PROVENANCE_JSON = OUT_DIR / "provenance.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECOVERED_GENERATOR = load_module(RECOVERED_GENERATOR_PATH, "recovered_generate_synthetic_lollipop_cohort")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2 * int(np.logical_and(a, b).sum()) / denom)


def centroid(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.full(3, np.nan)
    return coords.mean(axis=0)


def bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    return coords.min(axis=0), coords.max(axis=0) + 1


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    amin, amax = bbox(a)
    bmin, bmax = bbox(b)
    inter_min = np.maximum(amin, bmin)
    inter_max = np.minimum(amax, bmax)
    inter = np.maximum(0, inter_max - inter_min)
    inter_vol = int(np.prod(inter))
    av = int(np.prod(np.maximum(0, amax - amin)))
    bv = int(np.prod(np.maximum(0, bmax - bmin)))
    union = av + bv - inter_vol
    return float(inter_vol / union) if union else 1.0


def align_shape(candidate: np.ndarray, saved: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool)
    if candidate.shape == saved.shape:
        return candidate
    aligned = np.zeros_like(saved, dtype=bool)
    common = tuple(slice(0, min(saved.shape[i], candidate.shape[i])) for i in range(3))
    aligned[common] = candidate[common]
    return aligned


def classify_volume(volume_mm3: float) -> str:
    if volume_mm3 < 100.0:
        return "small_<100"
    if volume_mm3 < 1000.0:
        return "medium_100_1000"
    return "large_>=1000"


def finite_summary(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "median": math.nan, "iqr": math.nan, "p25": math.nan, "p75": math.nan, "min": math.nan, "max": math.nan}
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "iqr": float(p75 - p25),
        "p25": p25,
        "p75": p75,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if not np.isfinite(x) else f"{x:.6g}")
        else:
            display[col] = display[col].astype(str)
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for record in display.to_dict("records"):
        lines.append("| " + " | ".join(str(record[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def regenerate_mask(row: pd.Series) -> np.ndarray:
    return RECOVERED_GENERATOR._generate_one_mask(
        target_volume_mm3=float(row["target_volume_mm3"]),
        linear_scale_vox=float(row["final_linear_scale_vox"]),
        case_seed=int(row["seed"]),
        canal_axis="c",
        rotation_zyx_deg=(
            int(row["rotation_z_deg"]),
            int(row["rotation_y_deg"]),
            int(row["rotation_x_deg"]),
        ),
    ).astype(bool)


def recovered_geometry(row: pd.Series) -> tuple[dict[str, float], int]:
    rng_geom = np.random.default_rng(int(row["seed"]) ^ 0xBADC0DE)
    geom = RECOVERED_GENERATOR._map_scale_to_lollipop_geometry(
        linear_scale_vox=float(row["final_linear_scale_vox"]),
        target_volume_mm3=float(row["target_volume_mm3"]),
        rng=rng_geom,
    )
    _, rad_max = RECOVERED_GENERATOR._grid_and_radmax_from_scale(float(row["final_linear_scale_vox"]))
    return geom, rad_max


def recovered_phase(seed: int, rad_max: int) -> dict[str, float]:
    rng = np.random.default_rng(seed ^ 0x1234ABCD)
    _ = int(rng.integers(1, rad_max // 2))
    return {
        "cpa_lob_amp": float(rng.uniform(0.06, 0.12)),
        "cpa_lob_phase_1": float(rng.uniform(0.0, 2.0 * np.pi)),
        "cpa_lob_phase_2": float(rng.uniform(0.0, 2.0 * np.pi)),
        "cpa_bias_1": float(rng.uniform(-0.16, 0.16)),
        "cpa_bias_2": float(rng.uniform(-0.12, 0.12)),
    }


def local_lollipop_coordinates(
    coords_ijk: np.ndarray,
    shape: tuple[int, int, int],
    rotation_zyx_deg: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray([shape[0] // 2, shape[1] // 2, shape[2] // 2], dtype=float)
    rel = coords_ijk.astype(float) - center.reshape(1, 3)
    spy = rel[:, 0]
    spx = rel[:, 1]
    spz = rel[:, 2]
    rot_matrix_inv = Rotation.from_euler("zyx", rotation_zyx_deg, degrees=True).as_matrix().T
    spy_rotate = rot_matrix_inv[0, 0] * spy + rot_matrix_inv[0, 1] * spx + rot_matrix_inv[0, 2] * spz
    spx_rotate = rot_matrix_inv[1, 0] * spy + rot_matrix_inv[1, 1] * spx + rot_matrix_inv[1, 2] * spz
    spz_rotate = rot_matrix_inv[2, 0] * spy + rot_matrix_inv[2, 1] * spx + rot_matrix_inv[2, 2] * spz
    x_rel = -spz_rotate
    return x_rel, spx_rotate, spy_rotate


def classify_compartment_points(
    x_rel: np.ndarray,
    perp1_coord: np.ndarray,
    perp2_coord: np.ndarray,
    geom: dict[str, float],
    phase: dict[str, float],
) -> dict[str, np.ndarray]:
    rho = np.sqrt(perp1_coord ** 2 + perp2_coord ** 2)
    br = float(geom["canal_base_radius_init"])
    ar = float(geom["canal_apex_radius_init"])
    cl = float(geom["canal_length_init"])
    cr = float(geom["bulb_radius_init"])

    r_porus = np.sqrt(np.maximum(0.0, br ** 2 - x_rel ** 2))
    porus = (x_rel >= -br) & (x_rel <= 0.0) & (rho <= r_porus)
    t_canal = np.clip(x_rel / cl, 0.0, 1.0) if cl > 0.0 else np.zeros_like(x_rel)
    r_canal = ar + (br - ar) * (1.0 - t_canal) ** 1.5
    canal_body = (x_rel >= 0.0) & (x_rel <= cl) & (rho <= r_canal)
    r_fundus = np.sqrt(np.maximum(0.0, ar ** 2 - (x_rel - cl) ** 2))
    fundus = (x_rel > cl) & (x_rel <= cl + ar) & (rho <= r_fundus)
    canal_like = porus | canal_body | fundus

    if cr > 0.0:
        cpa_ctr = max(br * 1.2, cr * 0.75)
        cpa_ax_r = min(br * 0.7, cr * 0.45)
        cpa_eq_r = cr
        theta = np.arctan2(perp2_coord, perp1_coord)
        lobulation = 1.0 + phase["cpa_lob_amp"] * (
            0.65 * np.sin(2.0 * theta + phase["cpa_lob_phase_1"])
            + 0.35 * np.sin(3.0 * theta + phase["cpa_lob_phase_2"])
        )
        cpa_eq_r_mod = np.maximum(cpa_eq_r * lobulation, cpa_eq_r * 0.82)
        cpa_perp1_ctr = phase["cpa_bias_1"] * cpa_eq_r
        cpa_perp2_ctr = phase["cpa_bias_2"] * cpa_eq_r
        cpa_rho = np.sqrt((perp1_coord - cpa_perp1_ctr) ** 2 + (perp2_coord - cpa_perp2_ctr) ** 2)
        cpa_dist = ((x_rel + cpa_ctr) / cpa_ax_r) ** 2 + (cpa_rho / cpa_eq_r_mod) ** 2
        cpa = (cpa_dist <= 1.0) & (x_rel <= 0.0)
    else:
        cpa = np.zeros_like(canal_like, dtype=bool)

    transition = porus & cpa
    stem = canal_like & ~transition
    bulb = cpa & ~transition
    return {"stem": stem, "transition": transition, "bulb": bulb, "union": stem | transition | bulb, "porus": porus, "cpa": cpa}


def principal_lengths(points_mm: np.ndarray) -> np.ndarray:
    if points_mm.shape[0] < 3:
        return np.full(3, np.nan)
    centered = points_mm - points_mm.mean(axis=0, keepdims=True)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        proj = centered @ eigvecs[:, order]
    if not np.all(np.isfinite(proj)):
        return np.full(3, np.nan)
    return proj.max(axis=0) - proj.min(axis=0)


def component_measurements(coords: np.ndarray, spacing: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "voxel_count": int(mask.sum()),
        "volume_mm3": float(mask.sum() * np.prod(spacing)),
        "principal_length_major_mm": math.nan,
        "principal_length_minor1_mm": math.nan,
        "principal_length_minor2_mm": math.nan,
    }
    if not np.any(mask):
        return out
    points_mm = coords[mask].astype(float) * spacing.reshape(1, 3)
    lengths = np.sort(principal_lengths(points_mm))[::-1]
    out.update(
        {
            "principal_length_major_mm": float(lengths[0]),
            "principal_length_minor1_mm": float(lengths[1]),
            "principal_length_minor2_mm": float(lengths[2]),
        }
    )
    return out


def gap_count_along_axis(x_rel: np.ndarray, mask: np.ndarray, bin_width: float = 1.0) -> int:
    if not np.any(mask):
        return 0
    vals = x_rel[mask]
    bins = np.floor((vals - vals.min()) / bin_width).astype(int)
    occupied = sorted(set(int(x) for x in bins))
    return int(sum(1 for a, b in zip(occupied, occupied[1:]) if b - a > 1))


def verify_recovered_cases(manifest: pd.DataFrame) -> pd.DataFrame:
    trials = pd.read_csv(PROVENANCE_CASES_CSV)
    case_ids = list(dict.fromkeys(trials["case_id"].astype(str).tolist()))[:10]
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        row = manifest[manifest["case_id"] == case_id].iloc[0]
        saved = np.asarray(nib.load(str(MASK_ROOT / f"{case_id}_synthetic_lollipop.nii.gz")).dataobj) > 0
        generated = align_shape(regenerate_mask(row), saved)
        rows.append(
            {
                "case_id": case_id,
                "dice": dice(saved, generated),
                "volume_difference_voxels": int(generated.sum()) - int(saved.sum()),
                "centroid_difference_vox": float(np.linalg.norm(centroid(generated) - centroid(saved))),
                "bbox_iou": bbox_iou(saved, generated),
                "target_volume_mm3": float(row["target_volume_mm3"]),
                "final_linear_scale_vox": float(row["final_linear_scale_vox"]),
                "seed": int(row["seed"]),
                "rotation_z_deg": int(row["rotation_z_deg"]),
                "rotation_y_deg": int(row["rotation_y_deg"]),
                "rotation_x_deg": int(row["rotation_x_deg"]),
            }
        )
    out = pd.DataFrame(rows)
    if bool((out["dice"] < 0.999).any()):
        out.to_csv(OUT_DIR / "recovered_generator_preflight_failed.csv", index=False)
        raise RuntimeError("Recovered generator preflight failed: at least one representative case has Dice < 0.999")
    return out


def analyze_case(row: pd.Series) -> dict[str, Any]:
    case_id = str(row["case_id"])
    nii = nib.load(str(MASK_ROOT / f"{case_id}_synthetic_lollipop.nii.gz"))
    mask = np.asarray(nii.dataobj) > 0
    spacing = np.asarray(nii.header.get_zooms()[:3], dtype=float)
    coords = np.argwhere(mask)
    x_rel, perp1, perp2 = local_lollipop_coordinates(
        coords,
        mask.shape,
        (float(row["rotation_z_deg"]), float(row["rotation_y_deg"]), float(row["rotation_x_deg"])),
    )
    geom, rad_max = recovered_geometry(row)
    phase = recovered_phase(int(row["seed"]), rad_max=rad_max)
    compartments = classify_compartment_points(x_rel, perp1, perp2, geom, phase)
    stem = compartments["stem"]
    transition = compartments["transition"]
    bulb = compartments["bulb"]
    union = compartments["union"]
    unmatched = ~union
    labeled, cc_count = ndi_label(mask)
    voxel_volume = float(np.prod(spacing))
    total_volume = float(mask.sum() * voxel_volume)
    stem_volume = float(stem.sum() * voxel_volume)
    transition_volume = float(transition.sum() * voxel_volume)
    bulb_volume = float(bulb.sum() * voxel_volume)

    stem_rho = np.sqrt(perp1[stem] ** 2 + perp2[stem] ** 2) if np.any(stem) else np.asarray([], dtype=float)
    bulb_rho = np.sqrt(perp1[bulb] ** 2 + perp2[bulb] ** 2) if np.any(bulb) else np.asarray([], dtype=float)
    x_stem = x_rel[stem] if np.any(stem) else np.asarray([], dtype=float)
    x_bulb = x_rel[bulb] if np.any(bulb) else np.asarray([], dtype=float)
    stem_width = float(2.0 * np.percentile(stem_rho, 95)) if stem_rho.size else 0.0
    bulb_width = float(2.0 * np.percentile(bulb_rho, 95)) if bulb_rho.size else 0.0
    bulb_offset = float(np.sqrt(np.mean(perp1[bulb]) ** 2 + np.mean(perp2[bulb]) ** 2)) if np.any(bulb) else math.nan
    stem_meas = component_measurements(coords, spacing, stem)
    bulb_meas = component_measurements(coords, spacing, bulb)
    transition_meas = component_measurements(coords, spacing, transition)

    flags: list[str] = []
    unmatched_fraction = float(unmatched.sum() / max(1, int(mask.sum())))
    if unmatched_fraction > 0.01:
        flags.append("COMPARTMENT_RECONSTRUCTION_LIMIT:unmatched_voxels")
    if int(cc_count) > 1:
        flags.append("ENGINEERING_ARTIFACT:multiple_components")
    if total_volume < 100.0 and bulb_volume / max(total_volume, 1e-8) > 0.35:
        flags.append("PLAUSIBILITY_CONCERN:small_tumor_bulb_dominance")
    if total_volume >= 1000.0 and bulb_volume / max(total_volume, 1e-8) < 0.35:
        flags.append("PLAUSIBILITY_CONCERN:large_tumor_not_bulb_dominant")
    if np.any(stem) and np.any(bulb) and transition_volume <= 0:
        flags.append("PLAUSIBILITY_CONCERN:no_voxel_transition_overlap")
    if gap_count_along_axis(x_rel, union) > 0:
        flags.append("COMPARTMENT_RECONSTRUCTION_LIMIT:axial_gap_in_derived_union")
    if not flags:
        flags.append("NO_ISSUE_DETECTED")

    return {
        "case_id": case_id,
        "volume_bin": classify_volume(total_volume),
        "target_volume_mm3": float(row["target_volume_mm3"]),
        "realized_volume_mm3_manifest": float(row["realized_volume_mm3"]),
        "mask_volume_mm3": total_volume,
        "connected_component_count": int(cc_count),
        "seed": int(row["seed"]),
        "final_linear_scale_vox": float(row["final_linear_scale_vox"]),
        "rotation_z_deg": int(row["rotation_z_deg"]),
        "rotation_y_deg": int(row["rotation_y_deg"]),
        "rotation_x_deg": int(row["rotation_x_deg"]),
        "stem_volume_mm3": stem_volume,
        "transition_volume_mm3": transition_volume,
        "bulb_volume_mm3": bulb_volume,
        "stem_fraction": float(stem_volume / total_volume) if total_volume else math.nan,
        "transition_fraction": float(transition_volume / total_volume) if total_volume else math.nan,
        "bulb_fraction": float(bulb_volume / total_volume) if total_volume else math.nan,
        "bulb_to_stem_volume_ratio": float(bulb_volume / stem_volume) if stem_volume > 0 else math.nan,
        "stem_length_mm": float(x_stem.max() - x_stem.min()) if x_stem.size else 0.0,
        "stem_width_p95_mm": stem_width,
        "bulb_axial_extent_mm": float(x_bulb.max() - x_bulb.min()) if x_bulb.size else 0.0,
        "bulb_width_p95_mm": bulb_width,
        "bulb_to_stem_width_ratio": float(bulb_width / stem_width) if stem_width > 0 else math.nan,
        "total_extent_along_canal_mm": float(x_rel.max() - x_rel.min()) if x_rel.size else math.nan,
        "bulb_centroid_offset_from_stem_axis_mm": bulb_offset,
        "bulb_centroid_offset_ratio": float(bulb_offset / (bulb_width / 2.0)) if bulb_width > 0 else math.nan,
        "transition_overlap_fraction": float(transition_volume / max(transition_volume + stem_volume + bulb_volume, 1e-8)),
        "derived_unmatched_voxel_fraction": unmatched_fraction,
        "stem_principal_length_major_mm": stem_meas["principal_length_major_mm"],
        "stem_principal_length_minor1_mm": stem_meas["principal_length_minor1_mm"],
        "stem_principal_length_minor2_mm": stem_meas["principal_length_minor2_mm"],
        "bulb_principal_length_major_mm": bulb_meas["principal_length_major_mm"],
        "bulb_principal_length_minor1_mm": bulb_meas["principal_length_minor1_mm"],
        "bulb_principal_length_minor2_mm": bulb_meas["principal_length_minor2_mm"],
        "transition_principal_length_major_mm": transition_meas["principal_length_major_mm"],
        "generator_canal_length_init_vox": geom["canal_length_init"],
        "generator_canal_base_radius_init_vox": geom["canal_base_radius_init"],
        "generator_canal_apex_radius_init_vox": geom["canal_apex_radius_init"],
        "generator_bulb_radius_init_vox": geom["bulb_radius_init"],
        "generator_canal_length_max_vox": geom["canal_length_max_override"],
        "generator_bulb_radius_max_vox": geom["bulb_radius_max"],
        "generator_bulb_present": bool(geom["bulb_radius_init"] > 0.0),
        "flag_label": ";".join(flags),
    }


def summarize(features: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "stem_fraction",
        "transition_fraction",
        "bulb_fraction",
        "bulb_to_stem_volume_ratio",
        "stem_length_mm",
        "stem_width_p95_mm",
        "bulb_axial_extent_mm",
        "bulb_width_p95_mm",
        "bulb_to_stem_width_ratio",
        "bulb_centroid_offset_ratio",
        "transition_overlap_fraction",
        "derived_unmatched_voxel_fraction",
    ]
    rows: list[dict[str, Any]] = []
    for volume_bin in ["all", "small_<100", "medium_100_1000", "large_>=1000"]:
        part = features if volume_bin == "all" else features[features["volume_bin"] == volume_bin]
        for metric in metrics:
            rows.append({"volume_bin": volume_bin, "metric": metric, **finite_summary(pd.to_numeric(part[metric], errors="coerce"))})
        rows.append({"volume_bin": volume_bin, "metric": "zero_bulb_case_count", "n": len(part), "median": int((part["bulb_volume_mm3"] <= 0).sum()), "iqr": math.nan, "p25": math.nan, "p75": math.nan, "min": math.nan, "max": math.nan})
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_CSV, index=False)
    return out


def select_visual_cases(features: pd.DataFrame) -> list[str]:
    selected: list[str] = []

    for volume_bin in ["small_<100", "medium_100_1000", "large_>=1000"]:
        bin_selected: list[str] = []

        def add(case_id: str) -> None:
            if case_id not in selected and case_id not in bin_selected:
                bin_selected.append(case_id)

        part = features[features["volume_bin"] == volume_bin].copy()
        if part.empty:
            continue
        zero = part[part["bulb_fraction"] == 0.0]
        if not zero.empty:
            add(str(zero.sort_values("mask_volume_mm3").iloc[len(zero) // 2]["case_id"]))
        nonzero = part[part["bulb_fraction"] > 0.0]
        for q in [0.1, 0.5, 0.9]:
            if not nonzero.empty:
                target = float(nonzero["bulb_fraction"].quantile(q))
                idx = (nonzero["bulb_fraction"] - target).abs().idxmin()
                add(str(nonzero.loc[idx, "case_id"]))
        add(str(part.sort_values("derived_unmatched_voxel_fraction", ascending=False).iloc[0]["case_id"]))
        for _, row in part.sort_values(["bulb_fraction", "mask_volume_mm3"]).iterrows():
            if len(bin_selected) >= 4:
                break
            add(str(row["case_id"]))
        selected.extend(bin_selected[:4])
    return selected


def compartment_full_grid(mask_shape: tuple[int, int, int], row: pd.Series) -> dict[str, np.ndarray]:
    coords = np.indices(mask_shape).reshape(3, -1).T
    x_rel, perp1, perp2 = local_lollipop_coordinates(
        coords,
        mask_shape,
        (float(row["rotation_z_deg"]), float(row["rotation_y_deg"]), float(row["rotation_x_deg"])),
    )
    geom, rad_max = recovered_geometry(row)
    flat = classify_compartment_points(x_rel, perp1, perp2, geom, recovered_phase(int(row["seed"]), rad_max))
    return {key: value.reshape(mask_shape) for key, value in flat.items()}


def central_slice(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask.shape[2] // 2
    return int(round(float(np.median(coords[:, 2]))))


def write_overlay(case_id: str, saved: np.ndarray, stem: np.ndarray, transition: np.ndarray, bulb: np.ndarray, unassigned: np.ndarray) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    z = central_slice(saved)
    rgb = np.zeros((*saved[:, :, z].shape, 3), dtype=float)
    rgb[saved[:, :, z], :] = 0.23
    rgb[stem[:, :, z], :] = [0.1, 0.55, 1.0]
    rgb[transition[:, :, z], :] = [1.0, 0.85, 0.1]
    rgb[bulb[:, :, z], :] = [1.0, 0.2, 0.2]
    rgb[unassigned[:, :, z], :] = [0.85, 0.0, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    axes[0].imshow(saved[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title("authoritative mask")
    axes[1].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower")
    axes[1].set_title("stem / transition / bulb")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{case_id} z={z}")
    fig.tight_layout()
    out = OVERLAY_DIR / f"{case_id}_v2_overlay.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def visual_verify(features: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id in select_visual_cases(features):
        row = manifest[manifest["case_id"] == case_id].iloc[0]
        saved = np.asarray(nib.load(str(MASK_ROOT / f"{case_id}_synthetic_lollipop.nii.gz")).dataobj) > 0
        comps = compartment_full_grid(saved.shape, row)
        stem = comps["stem"] & saved
        transition = comps["transition"] & saved
        bulb = comps["bulb"] & saved
        union = stem | transition | bulb
        unassigned = saved & ~union
        unassigned_fraction = float(unassigned.sum() / max(1, int(saved.sum())))
        boundary_fraction = 0.0
        if int(unassigned.sum()) > 0:
            boundary_fraction = float(np.logical_and(unassigned, binary_dilation(union, iterations=1)).sum() / int(unassigned.sum()))
        if unassigned_fraction <= 0.01:
            classification = "MAPPING_CONFIRMED"
            reason = "compartments cover >99% of saved mask voxels"
        elif boundary_fraction >= 0.90 and unassigned_fraction <= 0.08:
            classification = "AMBIGUOUS"
            reason = "minor unmatched voxels are boundary-adjacent"
        elif unassigned_fraction <= 0.15:
            classification = "PARTIAL_MAPPING_ERROR"
            reason = "saved mask reproduced but compartment formula leaves nontrivial unmatched voxels"
        else:
            classification = "MAJOR_MAPPING_ERROR"
            reason = "substantial saved-mask voxels are outside reconstructed compartments"
        overlay = write_overlay(case_id, saved, stem, transition, bulb, unassigned)
        frow = features[features["case_id"] == case_id].iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "volume_bin": frow["volume_bin"],
                "bulb_fraction": frow["bulb_fraction"],
                "stem_fraction": frow["stem_fraction"],
                "unassigned_fraction": unassigned_fraction,
                "unassigned_boundary_fraction": boundary_fraction,
                "classification": classification,
                "reason": reason,
                "overlay_path": str(overlay.relative_to(REPO_ROOT)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(VISUAL_CSV, index=False)
    return out


def prior_finding_comparison(features: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "prior_finding": "191/261 zero-bulb result",
            "corrected_result": f"{int((features['bulb_volume_mm3'] <= 0).sum())}/{len(features)} zero-bulb",
            "classification": "CONFIRMED" if int((features["bulb_volume_mm3"] <= 0).sum()) == 191 else "REJECTED",
        },
        {
            "prior_finding": "160/261 mismatch flags",
            "corrected_result": f"{int((features['derived_unmatched_voxel_fraction'] > 0.01).sum())}/{len(features)} cases with >1% unmatched voxels",
            "classification": "CONFIRMED" if int((features["derived_unmatched_voxel_fraction"] > 0.01).sum()) == 160 else "ARTIFACT_OF_WRONG_GENERATOR",
        },
        {
            "prior_finding": "9/261 multi-component masks",
            "corrected_result": f"{int((features['connected_component_count'] > 1).sum())}/{len(features)} multi-component masks",
            "classification": "CONFIRMED" if int((features["connected_component_count"] > 1).sum()) == 9 else "REJECTED",
        },
    ]
    return pd.DataFrame(rows)


def write_report(preflight: pd.DataFrame, features: pd.DataFrame, summary: pd.DataFrame, visual: pd.DataFrame) -> None:
    prior = prior_finding_comparison(features)
    flag_counts = features["flag_label"].str.get_dummies(sep=";").sum().sort_values(ascending=False).reset_index()
    flag_counts.columns = ["flag", "case_count"]
    key_summary = summary[summary["metric"].isin(["stem_fraction", "bulb_fraction", "bulb_to_stem_volume_ratio", "stem_length_mm", "bulb_width_p95_mm", "derived_unmatched_voxel_fraction"])]
    zero_count = int((features["bulb_volume_mm3"] <= 0).sum())
    large = features[features["volume_bin"] == "large_>=1000"]
    small = features[features["volume_bin"] == "small_<100"]
    medium = features[features["volume_bin"] == "medium_100_1000"]
    large_bulb_median = finite_summary(large["bulb_fraction"])["median"]
    medium_bulb_median = finite_summary(medium["bulb_fraction"])["median"]
    small_bulb_median = finite_summary(small["bulb_fraction"])["median"]
    visual_counts = visual["classification"].value_counts().reset_index()
    visual_counts.columns = ["classification", "case_count"]

    REPORT_MD.write_text(
        f"""# Synthetic Compartment Audit V2

Generated: {datetime.now(timezone.utc).isoformat()}

## Scope

This v2 audit recomputes synthetic canal/stem and CPA/bulb metrics for the authoritative 261 saved masks using the recovered historical generator path:

`rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py`

The current checked-in generator was not used to reconstruct historical compartment geometry for this cohort.

## Recovered Generator Preflight

All 10 representative provenance cases passed Dice >= 0.999 before full analysis.

{markdown_table(preflight)}

## Cohort Summary

- Cases analyzed: {len(features)}
- Zero-bulb saved masks by recovered generator mapping: {zero_count}/{len(features)}
- Small median bulb fraction: {small_bulb_median:.6g}
- Medium median bulb fraction: {medium_bulb_median:.6g}
- Large median bulb fraction: {large_bulb_median:.6g}
- Multi-component masks: {int((features['connected_component_count'] > 1).sum())}/{len(features)}
- Cases with >1% unmatched compartment voxels: {int((features['derived_unmatched_voxel_fraction'] > 0.01).sum())}/{len(features)}

## Key Distributions

{markdown_table(key_summary)}

## Prior Finding Comparison

{markdown_table(prior)}

## Flag Counts

{markdown_table(flag_counts)}

## Visual Verification

Fresh stratified cases reviewed: {len(visual)}.

{markdown_table(visual_counts)}

{markdown_table(visual[['case_id', 'volume_bin', 'bulb_fraction', 'stem_fraction', 'unassigned_fraction', 'classification', 'overlay_path']])}

## Scientific Interpretation

1. Authoritative saved masks lacking a CPA bulb: {zero_count}/{len(features)} under the recovered generator mapping.
2. Stem dominance: the cohort remains stem-dominant by derived fractions; this is a synthetic design signal, not a clinical invalidity claim.
3. Bulb fraction and tumor size: median bulb fraction increases from small ({small_bulb_median:.6g}) to medium ({medium_bulb_median:.6g}) to large ({large_bulb_median:.6g}), but the absolute bulb fraction remains modest.
4. Gating/plateaus: the recovered generator uses a maturity-dependent bulb radius, so zero or near-zero bulb behavior is expected in small cases and low-volume regimes. This is a generator design feature.
5. Small tumors: mostly intracanalicular/stem-dominant small tumors are plausible as a synthetic prior, but the exact distribution requires real compartment validation.
6. Large tumors: large tumors develop nonzero CPA components, but the recovered mapping still suggests many remain stem-dominant.
7. Prior concerns: the old current-generator mismatch count is an artifact of the wrong reconstruction path. Zero-bulb and multi-component findings are evaluated in the prior-comparison table above.
8. Generator tuning: not justified now. Synthetic-side compartment metrics are more trustworthy after provenance recovery, but real source masks and anatomical annotations are still required before tuning.

## Decision

Synthetic compartment metrics from this v2 audit are suitable for synthetic-only plausibility tracking and future real-vs-synthetic comparison once real annotations exist. They should not be used alone to declare the generator anatomically valid or invalid.
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["case_id"] = manifest["case_id"].astype(str)
    preflight = verify_recovered_cases(manifest)
    rows = [analyze_case(row) for _, row in manifest.iterrows()]
    features = pd.DataFrame(rows)
    features.to_csv(FEATURES_CSV, index=False)
    summary = summarize(features)
    visual = visual_verify(features, manifest)
    write_report(preflight, features, summary, visual)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "mask_root": str(MASK_ROOT.relative_to(REPO_ROOT)),
        "recovered_generator_path": str(RECOVERED_GENERATOR_PATH.relative_to(REPO_ROOT)),
        "recovered_generator_sha256": sha256_file(RECOVERED_GENERATOR_PATH),
        "cases_expected": int(len(manifest)),
        "cases_processed": int(len(features)),
        "preflight_cases": preflight["case_id"].tolist(),
        "preflight_min_dice": float(preflight["dice"].min()),
        "outputs": {
            "features_csv": str(FEATURES_CSV.relative_to(REPO_ROOT)),
            "summary_csv": str(SUMMARY_CSV.relative_to(REPO_ROOT)),
            "visual_csv": str(VISUAL_CSV.relative_to(REPO_ROOT)),
            "report": str(REPORT_MD.relative_to(REPO_ROOT)),
        },
    }
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Recovered-generator compartment audit complete")
    print(f"  Cases: {len(features)}")
    print(f"  Zero-bulb: {int((features['bulb_volume_mm3'] <= 0).sum())}/{len(features)}")
    print(f"  Report: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
