#!/usr/bin/env python3
"""Analyze synthetic lollipop canal/CPA compartment relationships.

This script is analysis-only. It does not modify generator geometry or overwrite
authoritative feature artifacts. It reconstructs the generator's local lollipop
coordinate frame from the manifest and measures compartment-level properties
from the current local synthetic masks.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_synthetic_lollipop_cohort import (  # noqa: E402
    _grid_and_radmax_from_scale,
    _map_scale_to_lollipop_geometry,
)
from shared.provenance import get_git_commit, sha256_file  # noqa: E402
from shared.reporting import markdown_table_from_dataframe  # noqa: E402


OUT_DIR = REPO_ROOT / "analysis" / "anatomical_compartment_validation"
PLOTS_DIR = OUT_DIR / "plots"
MPL_CACHE_DIR = OUT_DIR / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))
MANIFEST_PATH = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "manifests" / "synthetic_lollipop_manifest.csv"
MASK_ROOT = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "masks"
SYN_FEATURES = REPO_ROOT / "analysis" / "synthetic_features_v2" / "synthetic_features_v2.csv"
REAL_FEATURES = REPO_ROOT / "rivanna_pull" / "analysis" / "real_tumor_features_v1" / "real_tumor_features_usable_train.csv"
SYNTHETIC_COMPARTMENT_CSV = OUT_DIR / "synthetic_compartment_features.csv"
SUMMARY_CSV = OUT_DIR / "synthetic_compartment_summary.csv"
INVENTORY_MD = OUT_DIR / "LOCAL_ANATOMY_INVENTORY.md"
SPEC_MD = OUT_DIR / "COMPARTMENT_METRIC_SPEC.md"
REAL_SPEC_MD = OUT_DIR / "REAL_MASK_ACQUISITION_SPEC.md"
REPORT_MD = OUT_DIR / "SYNTHETIC_COMPARTMENT_AUDIT.md"


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


def lollipop_geometry_from_manifest_row(row: pd.Series) -> dict[str, float]:
    seed = int(row["seed"])
    rng_geom = np.random.default_rng(seed ^ 0xBADC0DE)
    return _map_scale_to_lollipop_geometry(
        linear_scale_vox=float(row["final_linear_scale_vox"]),
        target_volume_mm3=float(row["target_volume_mm3"]),
        rng=rng_geom,
    )


def lollipop_phase_state(seed: int, rad_min: int, rad_max: int) -> dict[str, float]:
    rng = np.random.default_rng(seed ^ 0x1234ABCD)
    _ = int(rng.integers(rad_min, rad_max // 2))
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
    canal_axis: str = "c",
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
    if canal_axis == "a":
        canal_coord, perp1_coord, perp2_coord = spx_rotate, spy_rotate, spz_rotate
    elif canal_axis == "b":
        canal_coord, perp1_coord, perp2_coord = spy_rotate, spx_rotate, spz_rotate
    else:
        canal_coord, perp1_coord, perp2_coord = spz_rotate, spx_rotate, spy_rotate
    x_rel = -canal_coord
    return x_rel, perp1_coord, perp2_coord


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
    union = stem | transition | bulb
    return {
        "stem": stem,
        "transition": transition,
        "bulb": bulb,
        "porus": porus,
        "canal_body": canal_body,
        "fundus": fundus,
        "canal_like": canal_like,
        "cpa": cpa,
        "union": union,
    }


def principal_direction_and_lengths(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_mm.shape[0] < 3:
        return np.full(3, np.nan), np.full(3, np.nan)
    centered = points_mm - points_mm.mean(axis=0, keepdims=True)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvec = eigvecs[:, order[0]]
    norm = np.linalg.norm(eigvec)
    direction = eigvec / norm if norm else np.full(3, np.nan)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        proj = centered @ eigvecs[:, order]
    if not np.all(np.isfinite(proj)):
        return direction, np.full(3, np.nan)
    lengths = proj.max(axis=0) - proj.min(axis=0)
    return direction, lengths


def component_measurements(coords: np.ndarray, spacing: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "voxel_count": int(mask.sum()),
        "volume_mm3": float(mask.sum() * np.prod(spacing)),
        "centroid_x_mm": math.nan,
        "centroid_y_mm": math.nan,
        "centroid_z_mm": math.nan,
        "principal_length_major_mm": math.nan,
        "principal_length_minor1_mm": math.nan,
        "principal_length_minor2_mm": math.nan,
    }
    if not np.any(mask):
        return out
    points_mm = coords[mask].astype(float) * spacing.reshape(1, 3)
    centroid = points_mm.mean(axis=0)
    _, lengths = principal_direction_and_lengths(points_mm)
    lengths = np.sort(lengths)[::-1] if np.all(np.isfinite(lengths)) else np.full(3, np.nan)
    out.update(
        {
            "centroid_x_mm": float(centroid[0]),
            "centroid_y_mm": float(centroid[1]),
            "centroid_z_mm": float(centroid[2]),
            "principal_length_major_mm": float(lengths[0]),
            "principal_length_minor1_mm": float(lengths[1]),
            "principal_length_minor2_mm": float(lengths[2]),
        }
    )
    return out


def gap_count_along_sorted_axis(x_rel: np.ndarray, mask: np.ndarray, bin_width: float = 1.0) -> int:
    if not np.any(mask):
        return 0
    vals = x_rel[mask]
    bins = np.floor((vals - vals.min()) / bin_width).astype(int)
    occupied = sorted(set(int(x) for x in bins))
    if len(occupied) <= 1:
        return 0
    return int(sum(1 for a, b in zip(occupied, occupied[1:]) if b - a > 1))


def analyze_case(row: pd.Series) -> dict[str, Any]:
    case_id = str(row["case_id"])
    mask_path = MASK_ROOT / f"{case_id}_synthetic_lollipop.nii.gz"
    nii = nib.load(str(mask_path))
    mask = np.asarray(nii.dataobj) > 0
    spacing = np.asarray(nii.header.get_zooms()[:3], dtype=float)
    coords = np.argwhere(mask)
    x_rel, perp1, perp2 = local_lollipop_coordinates(
        coords,
        shape=mask.shape,
        rotation_zyx_deg=(float(row["rotation_z_deg"]), float(row["rotation_y_deg"]), float(row["rotation_x_deg"])),
        canal_axis="c",
    )
    size, rad_max = _grid_and_radmax_from_scale(float(row["final_linear_scale_vox"]))
    geom = lollipop_geometry_from_manifest_row(row)
    phase = lollipop_phase_state(int(row["seed"]), rad_min=1, rad_max=rad_max)
    compartments = classify_compartment_points(x_rel, perp1, perp2, geom, phase)
    voxel_volume = float(np.prod(spacing))
    labeled, cc_count = ndi_label(mask)

    stem = compartments["stem"]
    transition = compartments["transition"]
    bulb = compartments["bulb"]
    unmatched = ~compartments["union"]
    stem_meas = component_measurements(coords, spacing, stem)
    bulb_meas = component_measurements(coords, spacing, bulb)
    transition_meas = component_measurements(coords, spacing, transition)
    total_volume = float(mask.sum() * voxel_volume)
    stem_volume = float(stem.sum() * voxel_volume)
    bulb_volume = float(bulb.sum() * voxel_volume)
    transition_volume = float(transition.sum() * voxel_volume)
    x_stem = x_rel[stem] if np.any(stem) else np.asarray([], dtype=float)
    x_bulb = x_rel[bulb] if np.any(bulb) else np.asarray([], dtype=float)
    stem_length = float(x_stem.max() - x_stem.min()) if x_stem.size else 0.0
    bulb_axial_extent = float(x_bulb.max() - x_bulb.min()) if x_bulb.size else 0.0
    stem_rho = np.sqrt(perp1[stem] ** 2 + perp2[stem] ** 2) if np.any(stem) else np.asarray([], dtype=float)
    bulb_rho = np.sqrt(perp1[bulb] ** 2 + perp2[bulb] ** 2) if np.any(bulb) else np.asarray([], dtype=float)
    stem_width = float(2.0 * np.percentile(stem_rho, 95)) if stem_rho.size else 0.0
    bulb_width = float(2.0 * np.percentile(bulb_rho, 95)) if bulb_rho.size else 0.0
    bulb_centroid_offset = math.nan
    if np.any(bulb):
        bulb_centroid_offset = float(np.sqrt(np.mean(perp1[bulb]) ** 2 + np.mean(perp2[bulb]) ** 2))
    off_axis_ratio = bulb_centroid_offset / (bulb_width / 2.0) if bulb_width > 0 else math.nan
    expected_dim_mismatch = int(size != mask.shape[0] or size != mask.shape[1] or size != mask.shape[2])

    flags: list[str] = []
    if expected_dim_mismatch:
        flags.append("ENGINEERING_ARTIFACT:expected_grid_size_differs_from_mask")
    if float(unmatched.sum()) / max(1, int(mask.sum())) > 0.01:
        flags.append("ENGINEERING_ARTIFACT:derived_compartment_mismatch")
    if int(cc_count) > 1:
        flags.append("ENGINEERING_ARTIFACT:multiple_components")
    if total_volume < 100.0 and bulb_volume / max(total_volume, 1e-8) > 0.35:
        flags.append("PLAUSIBILITY_CONCERN:small_tumor_bulb_dominance")
    if total_volume >= 1000.0 and bulb_volume / max(total_volume, 1e-8) < 0.35:
        flags.append("PLAUSIBILITY_CONCERN:large_tumor_not_bulb_dominant")
    if np.any(stem) and np.any(bulb) and transition_volume <= 0:
        flags.append("PLAUSIBILITY_CONCERN:no_voxel_transition_overlap")
    if gap_count_along_sorted_axis(x_rel, compartments["union"]) > 0:
        flags.append("ENGINEERING_ARTIFACT:axial_gap_in_derived_union")
    if not flags:
        flags.append("NO_ISSUE_DETECTED")

    return {
        "case_id": case_id,
        "mask_path": str(mask_path),
        "volume_bin": classify_volume(total_volume),
        "target_volume_mm3": float(row["target_volume_mm3"]),
        "realized_volume_mm3_manifest": float(row["realized_volume_mm3"]),
        "mask_volume_mm3": total_volume,
        "voxel_spacing_x_mm": float(spacing[0]),
        "voxel_spacing_y_mm": float(spacing[1]),
        "voxel_spacing_z_mm": float(spacing[2]),
        "connected_component_count": int(cc_count),
        "final_linear_scale_vox": float(row["final_linear_scale_vox"]),
        "seed": int(row["seed"]),
        "expected_grid_size": int(size),
        "mask_grid_size_x": int(mask.shape[0]),
        "mask_grid_size_y": int(mask.shape[1]),
        "mask_grid_size_z": int(mask.shape[2]),
        "derived_unmatched_voxel_fraction": float(unmatched.sum() / max(1, int(mask.sum()))),
        "stem_volume_mm3": stem_volume,
        "transition_volume_mm3": transition_volume,
        "bulb_volume_mm3": bulb_volume,
        "stem_fraction": float(stem_volume / total_volume) if total_volume > 0 else math.nan,
        "transition_fraction": float(transition_volume / total_volume) if total_volume > 0 else math.nan,
        "bulb_fraction": float(bulb_volume / total_volume) if total_volume > 0 else math.nan,
        "bulb_to_stem_volume_ratio": float(bulb_volume / stem_volume) if stem_volume > 0 else math.nan,
        "stem_length_mm": stem_length,
        "stem_width_p95_mm": stem_width,
        "bulb_axial_extent_mm": bulb_axial_extent,
        "bulb_width_p95_mm": bulb_width,
        "bulb_to_stem_width_ratio": float(bulb_width / stem_width) if stem_width > 0 else math.nan,
        "total_extent_along_canal_mm": float(x_rel.max() - x_rel.min()) if x_rel.size else math.nan,
        "bulb_centroid_offset_from_stem_axis_mm": bulb_centroid_offset,
        "bulb_centroid_offset_ratio": off_axis_ratio,
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


def write_inventory(manifest: pd.DataFrame, real: pd.DataFrame) -> None:
    mask_count = len(list(MASK_ROOT.glob("*_synthetic_lollipop.nii.gz")))
    real_spacing_summary = real[["voxel_spacing_mm_x", "voxel_spacing_mm_y", "voxel_spacing_mm_z"]].describe().reset_index()
    rows = pd.DataFrame(
        [
            ["Synthetic masks", "AVAILABLE_LOCAL", f"{mask_count} NIfTI masks under `{MASK_ROOT.relative_to(REPO_ROOT)}`"],
            ["Synthetic manifest", "AVAILABLE_LOCAL", f"{len(manifest)} rows with case ID, target/realized volume, calibration scale, rotations, and seed"],
            ["Generator implementation", "AVAILABLE_LOCAL", "`projects/vivit/src/data/synthetic.py` and `scripts/generate_synthetic_lollipop_cohort.py` define lollipop geometry"],
            ["Authoritative synthetic features", "AVAILABLE_LOCAL", "`analysis/synthetic_features_v2/synthetic_features_v2.csv`"],
            ["Pulled real feature table", "AVAILABLE_LOCAL", f"{len(real)} feature rows; spacing columns show local feature provenance but not compartment labels"],
            ["Canal/CPA labels", "REQUIRES_HUMAN_ANNOTATION", "No explicit real canal, porus, fundus, or CPA labels found locally"],
            ["Synthetic compartment labels", "DERIVABLE_LOCAL", "Can be reconstructed from generator convention, manifest scale, seed, and rotations"],
            ["Real compartment labels", "REQUIRES_REAL_MASKS", "Whole-mask feature table cannot separate intracanalicular and CPA components"],
            ["Anatomical landmarks", "REQUIRES_HUMAN_ANNOTATION", "Porus/fundus/canal axis landmarks are not present in local real artifacts"],
            ["Remote clinical source data", "BLOCKED_REMOTE_DATA", "Original real source masks/MRI are not present locally"],
        ],
        columns=["evidence", "classification", "local_status"],
    )
    INVENTORY_MD.write_text(
        f"""# Local Anatomy Inventory

Generated: {datetime.now(timezone.utc).isoformat()}

Scope: local repository only. No Rivanna, SSH, or remote data were used.

## Evidence Inventory

{markdown_table_from_dataframe(rows)}

## Real Feature Spacing Evidence

{markdown_table_from_dataframe(real_spacing_summary)}

## What Can Be Validated Locally

- Synthetic lollipop compartment geometry can be plausibility-checked from current masks and generator metadata.
- Synthetic stem/bulb volume fractions, widths, axial extents, centroid offsets, transition overlap, and artificial plateaus can be measured.
- Whole-mask real-vs-synthetic feature tables can be used only to define volume strata and broad context.

## What Cannot Be Validated Locally

- Real intracanalicular versus CPA compartment volumes.
- Real porus/fundus boundaries.
- Real canal-axis direction unless source masks and/or landmarks are provided.
- Clinical/anatomical validity of generator compartment ratios.
""",
        encoding="utf-8",
    )


def write_metric_spec() -> None:
    SPEC_MD.write_text(
        """# Compartment Metric Specification

## Generator Convention

The lollipop generator defines a local canal coordinate `x_rel = -canal_coord`.

- `x_rel > 0`: intracanalicular canal/fundus direction.
- `x_rel <= 0`: extracanalicular CPA side.
- `porus`: rounded opening spanning `x_rel in [-br, 0]`.
- `canal_body`: tapered stem spanning `x_rel in [0, cl]`.
- `fundus`: rounded cap spanning `x_rel in (cl, cl + ar]`.
- `cpa`: oblate bulb centered into negative `x_rel`.

## Derived Compartments

- `stem = (porus OR canal_body OR fundus) AND NOT transition`.
- `transition = porus AND cpa`.
- `bulb = cpa AND NOT transition`.

This preserves overlap at the porus as a transition region rather than double-counting it.

## Measurements

- Stem volume: stem voxel count times voxel volume.
- Stem length: max-min `x_rel` over stem voxels.
- Stem width: twice the 95th percentile radial distance from the stem axis.
- Stem principal direction/lengths: PCA over stem voxel centers in physical units.
- Bulb volume: bulb voxel count times voxel volume.
- Bulb axial extent: max-min `x_rel` over bulb voxels.
- Bulb width: twice the 95th percentile radial distance from the stem axis.
- Bulb centroid offset from stem axis: radial distance of bulb centroid in local perpendicular coordinates.
- Bulb-to-stem volume ratio: bulb volume divided by stem volume.
- Bulb-to-stem width ratio: bulb width divided by stem width.
- Total canal-axis extent: max-min `x_rel` over all tumor voxels.
- Transition smoothness proxy: non-zero transition volume and no axial gap in the derived union.

## Assumptions

- Synthetic masks are current authoritative masks.
- Manifest `seed`, `final_linear_scale_vox`, and rotations correspond to local masks.
- Voxel spacing is read from each NIfTI header.
- Metrics are synthetic plausibility checks, not clinical truth.
- Real compartment validation requires source masks and anatomical reference points.
""",
        encoding="utf-8",
    )


def write_real_acquisition_spec() -> None:
    REAL_SPEC_MD.write_text(
        """# Real Mask Acquisition Specification

## Minimal Useful Validation Subset

Start with 30 real cases:

- 10 small tumors: <100 mm3.
- 10 medium tumors: 100-1000 mm3.
- 10 large tumors: >=1000 mm3.

Prefer cases that match synthetic case IDs already present in the local real feature table so existing whole-mask features remain comparable.

## Required Files

- Real binary vestibular schwannoma segmentation mask for each case.
- NIfTI header or sidecar metadata preserving voxel spacing and affine.
- Case ID mapping to the current real feature table.

## MRI Requirement

Segmentation alone is sufficient for first-pass compartment geometry if manual landmarks are supplied. MRI is recommended, but not strictly required, to annotate porus/fundus and verify canal/CPA context.

## Required Annotations

At minimum, each case needs one of:

- Porus and fundus landmarks defining the canal axis and canal segment.
- A canal/IAC mask and a CPA region boundary.
- Expert-confirmed canal-axis vector plus a porus boundary point.

Manual annotation is required for the initial validation subset unless a trusted local atlas/landmarking pipeline is added and validated.

## Automatically Derivable After Annotation

- Intracanalicular volume.
- CPA/extracanalicular volume.
- Bulb-to-stem volume ratio.
- Stem width and length.
- Bulb offset from canal axis.
- Transition continuity at porus.
- Volume-stratified compartment trends.

## Questions Answerable Once Available

- Do synthetic small tumors overproduce or underproduce CPA bulb volume?
- Do large synthetic tumors become appropriately bulb-dominant?
- Is the synthetic stem width/length relationship in the observed real range?
- Is off-axis CPA growth comparable to real masks?
- Are transitions continuous at the porus without artificial discontinuities?
""",
        encoding="utf-8",
    )


def summarize_compartments(features: pd.DataFrame) -> pd.DataFrame:
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
        "derived_unmatched_voxel_fraction",
    ]
    rows = []
    for volume_bin in ["all", "small_<100", "medium_100_1000", "large_>=1000"]:
        part = features if volume_bin == "all" else features[features["volume_bin"] == volume_bin]
        for metric in metrics:
            summary = finite_summary(pd.to_numeric(part[metric], errors="coerce"))
            rows.append({"volume_bin": volume_bin, "metric": metric, **summary})
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_CSV, index=False)
    return out


def write_report(features: pd.DataFrame, summary: pd.DataFrame) -> None:
    flag_counts = features["flag_label"].str.get_dummies(sep=";").sum().sort_values(ascending=False).reset_index()
    flag_counts.columns = ["flag", "case_count"]
    key_summary = summary[
        summary["metric"].isin(
            [
                "stem_fraction",
                "bulb_fraction",
                "bulb_to_stem_volume_ratio",
                "stem_length_mm",
                "bulb_width_p95_mm",
                "bulb_centroid_offset_ratio",
            ]
        )
    ]
    no_bulb_count = int((features["generator_bulb_present"] == False).sum())  # noqa: E712
    large_count = int((features["volume_bin"] == "large_>=1000").sum())
    large_bulb_present = int(((features["volume_bin"] == "large_>=1000") & (features["generator_bulb_present"] == True)).sum())  # noqa: E712
    large_bulb_fraction = finite_summary(features.loc[features["volume_bin"] == "large_>=1000", "bulb_fraction"])["median"]
    large_stem_fraction = finite_summary(features.loc[features["volume_bin"] == "large_>=1000", "stem_fraction"])["median"]

    REPORT_MD.write_text(
        f"""# Synthetic Compartment Audit

Generated: {datetime.now(timezone.utc).isoformat()}

## Scope

This is a synthetic-only anatomical compartment plausibility audit. It reconstructs the current lollipop generator's canal/CPA coordinate frame from local manifest metadata and measures current authoritative masks. It does not validate real anatomy and does not justify generator tuning by itself.

## Inputs

- Manifest: `{MANIFEST_PATH.relative_to(REPO_ROOT)}` ({len(features)} analyzed rows)
- Mask root: `{MASK_ROOT.relative_to(REPO_ROOT)}`
- Synthetic features: `{SYN_FEATURES.relative_to(REPO_ROOT)}`
- Git commit: `{get_git_commit(REPO_ROOT)}`

## Reproducibility Commands

- `MPLCONFIGDIR=analysis/anatomical_compartment_validation/.matplotlib-cache XDG_CACHE_HOME=analysis/anatomical_compartment_validation/.matplotlib-cache .venv/bin/python analysis/anatomical_compartment_validation/analyze_synthetic_compartments.py`
- `.venv/bin/python -m pytest tests/test_anatomical_compartment_helpers.py -v`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/anatomical_compartment_validation/analyze_synthetic_compartments.py tests/test_anatomical_compartment_helpers.py`
- `.venv/bin/python -m pytest -m "fast and not slow" -v`
- `git diff --check`

## Outputs

- `{SYNTHETIC_COMPARTMENT_CSV.relative_to(REPO_ROOT)}`
- `{SUMMARY_CSV.relative_to(REPO_ROOT)}`
- `{INVENTORY_MD.relative_to(REPO_ROOT)}`
- `{SPEC_MD.relative_to(REPO_ROOT)}`
- `{REAL_SPEC_MD.relative_to(REPO_ROOT)}`

## Summary

{markdown_table_from_dataframe(key_summary)}

## Flag Counts

{markdown_table_from_dataframe(flag_counts)}

## Interpretation

- Synthetic stem and bulb compartments are derivable locally because the generator encodes a known local canal axis and CPA side.
- {no_bulb_count}/261 standalone cohort masks have no generated CPA bulb by design in the single-mask generator mapping, because `bulb_radius_init` is zero until high maturity.
- {large_bulb_present}/{large_count} large masks have a generated CPA bulb, but the derived median large-case bulb fraction is only {large_bulb_fraction:.3f} while median stem fraction is {large_stem_fraction:.3f}.
- This large-case stem dominance is a synthetic plausibility concern and likely reflects the standalone single-timepoint initialization/calibration regime. It is not a proven biological invalidity claim without real compartment labels.
- The no-bulb small/medium regime is an engineering design choice in the standalone cohort generator, not a proven biological statement.
- Cases with multiple connected components or derived compartment mismatch are flagged as engineering artifacts for review.
- Real canal/CPA plausibility remains blocked by missing real masks and landmarks.

## Generator Tuning Decision

No generator tuning is justified from this pass. The local result identifies synthetic compartment regimes and possible artifacts, but real anatomical validation requires source masks and porus/fundus or canal-axis annotations.
""",
        encoding="utf-8",
    )


def plot_outputs(features: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ["stem_fraction", "bulb_fraction", "bulb_to_stem_volume_ratio", "bulb_to_stem_width_ratio"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        features.boxplot(column=metric, by="volume_bin", ax=ax, rot=20)
        ax.set_title(metric)
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{metric}_by_volume_bin.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(features["mask_volume_mm3"], features["bulb_fraction"], s=18, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Volume mm3")
    ax.set_ylabel("Bulb fraction")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bulb_fraction_vs_volume.png", dpi=160)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    real = pd.read_csv(REAL_FEATURES)
    manifest["case_id"] = manifest["case_id"].astype(str)
    write_inventory(manifest, real)
    write_metric_spec()
    write_real_acquisition_spec()
    rows = [analyze_case(row) for _, row in manifest.iterrows()]
    features = pd.DataFrame(rows)
    features.to_csv(SYNTHETIC_COMPARTMENT_CSV, index=False)
    summary = summarize_compartments(features)
    plot_outputs(features)
    write_report(features, summary)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(REPO_ROOT),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "mask_root": str(MASK_ROOT),
        "case_count": int(len(features)),
        "generator_files": [
            str(REPO_ROOT / "projects" / "vivit" / "src" / "data" / "synthetic.py"),
            str(REPO_ROOT / "scripts" / "generate_synthetic_lollipop_cohort.py"),
        ],
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Anatomical compartment validation complete")
    print(f"  Cases analyzed: {len(features)}")
    print(f"  Output: {SYNTHETIC_COMPARTMENT_CSV}")
    print(f"  Report: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
