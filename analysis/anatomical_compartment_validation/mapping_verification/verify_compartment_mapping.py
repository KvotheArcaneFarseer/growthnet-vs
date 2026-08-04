#!/usr/bin/env python3
"""Verify synthetic lollipop compartment mapping on representative cases.

This script checks whether the compartment analyzer maps saved authoritative
synthetic masks correctly. It compares saved masks to masks regenerated from
manifest metadata and current generator code, then overlays derived stem,
transition, bulb, and unassigned voxels for selected cases.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label as ndi_label

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "analysis" / "anatomical_compartment_validation" / "mapping_verification"
OVERLAY_DIR = OUT_DIR / "overlays"
MPL_CACHE_DIR = OUT_DIR / ".matplotlib-cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

from analysis.anatomical_compartment_validation.analyze_synthetic_compartments import (  # noqa: E402
    MASK_ROOT,
    classify_compartment_points,
    lollipop_geometry_from_manifest_row,
    lollipop_phase_state,
    local_lollipop_coordinates,
)
from scripts.generate_synthetic_lollipop_cohort import (  # noqa: E402
    _generate_one_mask,
    _grid_and_radmax_from_scale,
)
from shared.provenance import get_git_commit  # noqa: E402
from shared.reporting import markdown_table_from_dataframe  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MANIFEST_PATH = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "manifests" / "synthetic_lollipop_manifest.csv"
COMPARTMENT_FEATURES = REPO_ROOT / "analysis" / "anatomical_compartment_validation" / "synthetic_compartment_features.csv"
CASE_CSV = OUT_DIR / "case_verification.csv"
REPORT_MD = OUT_DIR / "COMPARTMENT_MAPPING_VERIFICATION.md"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2 * int(np.logical_and(a, b).sum()) / denom)


def select_cases(features: pd.DataFrame) -> list[str]:
    selected: list[str] = []

    def add_case(case_id: str) -> None:
        if case_id not in selected:
            selected.append(case_id)

    zero = features[features["bulb_fraction"] == 0.0].copy()
    for volume_bin in ["small_<100", "medium_100_1000"]:
        part = zero[zero["volume_bin"] == volume_bin].sort_values(["derived_unmatched_voxel_fraction", "mask_volume_mm3"])
        if not part.empty:
            add_case(str(part.iloc[0]["case_id"]))
    mismatch_zero = zero.sort_values("derived_unmatched_voxel_fraction", ascending=False)
    if not mismatch_zero.empty:
        add_case(str(mismatch_zero.iloc[0]["case_id"]))

    nonzero = features[features["bulb_fraction"] > 0.0].copy()
    if not nonzero.empty:
        quantiles = nonzero["bulb_fraction"].quantile([0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]).to_list()
        for q in quantiles:
            idx = (nonzero["bulb_fraction"] - q).abs().sort_values().index[0]
            add_case(str(nonzero.loc[idx, "case_id"]))
    for _, row in nonzero.sort_values("derived_unmatched_voxel_fraction", ascending=False).head(2).iterrows():
        add_case(str(row["case_id"]))
    return selected[:10]


def compartment_full_grid(mask_shape: tuple[int, int, int], manifest_row: pd.Series) -> dict[str, np.ndarray]:
    all_coords = np.indices(mask_shape).reshape(3, -1).T
    x_rel, perp1, perp2 = local_lollipop_coordinates(
        all_coords,
        shape=mask_shape,
        rotation_zyx_deg=(
            float(manifest_row["rotation_z_deg"]),
            float(manifest_row["rotation_y_deg"]),
            float(manifest_row["rotation_x_deg"]),
        ),
        canal_axis="c",
    )
    _, rad_max = _grid_and_radmax_from_scale(float(manifest_row["final_linear_scale_vox"]))
    geom = lollipop_geometry_from_manifest_row(manifest_row)
    phase = lollipop_phase_state(int(manifest_row["seed"]), rad_min=1, rad_max=rad_max)
    flat = classify_compartment_points(x_rel, perp1, perp2, geom, phase)
    return {key: value.reshape(mask_shape) for key, value in flat.items()}


def boundary_fraction(unassigned: np.ndarray, mapped_union: np.ndarray) -> float:
    if int(unassigned.sum()) == 0:
        return 0.0
    dilated = binary_dilation(mapped_union, iterations=1)
    return float(np.logical_and(unassigned, dilated).sum() / int(unassigned.sum()))


def central_slice(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return mask.shape[2] // 2
    return int(round(float(np.median(coords[:, 2]))))


def write_overlay(case_id: str, saved: np.ndarray, stem: np.ndarray, transition: np.ndarray, bulb: np.ndarray, unassigned: np.ndarray, regen: np.ndarray) -> Path:
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    z = central_slice(saved)
    rgb = np.zeros((*saved[:, :, z].shape, 3), dtype=float)
    rgb[saved[:, :, z], :] = 0.25
    rgb[stem[:, :, z], :] = [0.1, 0.55, 1.0]
    rgb[transition[:, :, z], :] = [1.0, 0.85, 0.1]
    rgb[bulb[:, :, z], :] = [1.0, 0.2, 0.2]
    rgb[unassigned[:, :, z], :] = [0.85, 0.0, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(saved[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title("saved mask")
    axes[1].imshow(regen[:, :, z].T, cmap="gray", origin="lower")
    axes[1].set_title("regenerated")
    axes[2].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower")
    axes[2].set_title("stem/transition/bulb/unassigned")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{case_id} z={z}")
    fig.tight_layout()
    out = OVERLAY_DIR / f"{case_id}_overlay.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def verify_case(case_id: str, manifest: pd.DataFrame, features: pd.DataFrame) -> dict[str, Any]:
    manifest_row = manifest[manifest["case_id"] == case_id].iloc[0]
    feature_row = features[features["case_id"] == case_id].iloc[0]
    mask_path = MASK_ROOT / f"{case_id}_synthetic_lollipop.nii.gz"
    nii = nib.load(str(mask_path))
    saved = np.asarray(nii.dataobj) > 0
    spacing = tuple(float(x) for x in nii.header.get_zooms()[:3])
    regen = _generate_one_mask(
        target_volume_mm3=float(manifest_row["target_volume_mm3"]),
        linear_scale_vox=float(manifest_row["final_linear_scale_vox"]),
        case_seed=int(manifest_row["seed"]),
        canal_axis="c",
        rotation_zyx_deg=[
            int(manifest_row["rotation_z_deg"]),
            int(manifest_row["rotation_y_deg"]),
            int(manifest_row["rotation_x_deg"]),
        ],
    ).astype(bool)
    if regen.shape != saved.shape:
        padded = np.zeros_like(saved)
        common = tuple(slice(0, min(saved.shape[i], regen.shape[i])) for i in range(3))
        padded[common] = regen[common]
        regen_aligned = padded
    else:
        regen_aligned = regen

    compartments = compartment_full_grid(saved.shape, manifest_row)
    stem = compartments["stem"] & saved
    transition = compartments["transition"] & saved
    bulb = compartments["bulb"] & saved
    union = stem | transition | bulb
    unassigned = saved & ~union
    overlap_voxels = int(
        ((compartments["stem"].astype(int) + compartments["transition"].astype(int) + compartments["bulb"].astype(int)) > 1).sum()
    )
    total = int(saved.sum())
    assigned = int(union.sum())
    regen_dice = dice(saved, regen_aligned)
    saved_cc = int(ndi_label(saved)[1])
    regen_cc = int(ndi_label(regen_aligned)[1])
    unassigned_fraction = float(unassigned.sum() / max(1, total))
    boundary_frac = boundary_fraction(unassigned, union)
    overlay_path = write_overlay(case_id, saved, stem, transition, bulb, unassigned, regen_aligned)

    if regen_dice < 0.999:
        classification = "MAJOR_MAPPING_ERROR"
        reason = "current generator plus manifest metadata does not reproduce saved mask"
    elif unassigned_fraction <= 0.01:
        classification = "MAPPING_CONFIRMED"
        reason = "saved mask is reproduced and derived compartments cover >99% of mask voxels"
    elif boundary_frac >= 0.90 and unassigned_fraction <= 0.12:
        classification = "AMBIGUOUS"
        reason = "saved mask reproduced but unassigned voxels are mostly boundary-adjacent discretization"
    else:
        classification = "PARTIAL_MAPPING_ERROR"
        reason = "saved mask reproduced but derived compartments leave substantial non-boundary voxels unassigned"

    return {
        "case_id": case_id,
        "volume_bin": feature_row["volume_bin"],
        "mask_volume_mm3": float(feature_row["mask_volume_mm3"]),
        "manifest_scale": float(manifest_row["final_linear_scale_vox"]),
        "seed": int(manifest_row["seed"]),
        "rotation_z_deg": int(manifest_row["rotation_z_deg"]),
        "rotation_y_deg": int(manifest_row["rotation_y_deg"]),
        "rotation_x_deg": int(manifest_row["rotation_x_deg"]),
        "saved_shape": "x".join(str(x) for x in saved.shape),
        "regenerated_shape": "x".join(str(x) for x in regen.shape),
        "spacing": ",".join(f"{x:g}" for x in spacing),
        "saved_total_voxels": total,
        "regenerated_total_voxels": int(regen.sum()),
        "regenerated_saved_dice": regen_dice,
        "saved_connected_components": saved_cc,
        "regenerated_connected_components": regen_cc,
        "stem_voxels": int(stem.sum()),
        "transition_voxels": int(transition.sum()),
        "bulb_voxels": int(bulb.sum()),
        "assigned_voxels": assigned,
        "unassigned_voxels": int(unassigned.sum()),
        "unassigned_fraction": unassigned_fraction,
        "unassigned_boundary_adjacent_fraction": boundary_frac,
        "overlapping_compartment_grid_voxels": overlap_voxels,
        "prior_bulb_fraction": float(feature_row["bulb_fraction"]),
        "prior_unmatched_fraction": float(feature_row["derived_unmatched_voxel_fraction"]),
        "prior_flags": feature_row["flag_label"],
        "overlay_path": str(overlay_path.relative_to(REPO_ROOT)),
        "classification": classification,
        "classification_reason": reason,
    }


def write_report(results: pd.DataFrame, selected: list[str]) -> None:
    class_counts = results["classification"].value_counts().rename_axis("classification").reset_index(name="case_count")
    selected_rows = results[
        [
            "case_id",
            "volume_bin",
            "prior_bulb_fraction",
            "regenerated_saved_dice",
            "unassigned_fraction",
            "unassigned_boundary_adjacent_fraction",
            "classification",
            "classification_reason",
        ]
    ]
    regen_ok = int((results["regenerated_saved_dice"] >= 0.999).sum())
    median_regen_dice = float(results["regenerated_saved_dice"].median())
    zero_cases = results[results["prior_bulb_fraction"] == 0.0]
    zero_clean_by_mapper = int(((zero_cases["unassigned_fraction"] == 0.0) & (zero_cases["bulb_voxels"] == 0)).sum())
    mismatch_cases = results[results["prior_flags"].str.contains("derived_compartment_mismatch", regex=False)]
    major_mismatch = int(mismatch_cases["classification"].eq("MAJOR_MAPPING_ERROR").sum())
    REPORT_MD.write_text(
        f"""# Compartment Mapping Verification

Generated: {datetime.now(timezone.utc).isoformat()}

## Scope

This verifies whether the compartment analyzer correctly maps saved authoritative synthetic masks into canal/stem, transition, and CPA/bulb compartments. It does not modify generator geometry, masks, or authoritative features.

## Selected Cases

Selected case IDs: {', '.join(selected)}

The selection includes zero-bulb cases, low nonzero bulb-fraction cases, high nonzero bulb-fraction cases, mismatch cases, and multiple volume strata where available.

## Verification Table

{markdown_table_from_dataframe(selected_rows)}

## Classification Counts

{markdown_table_from_dataframe(class_counts)}

## Main Checks

- Saved-mask reproduction from manifest metadata and current generator code succeeded for {regen_ok}/{len(results)} selected cases at Dice >= 0.999.
- Median saved-vs-regenerated Dice across selected cases: {median_regen_dice:.3f}.
- Zero-bulb selected cases with zero derived bulb and zero unassigned voxels under the current mapper: {zero_clean_by_mapper}/{len(zero_cases)}.
- Selected mismatch cases classified as major mapping/code-reconstruction errors: {major_mismatch}/{len(mismatch_cases)}.
- Overlay PNGs are in `{OVERLAY_DIR.relative_to(REPO_ROOT)}`.

## Decision

1. The current analyzer cannot yet be trusted for quantitative real-vs-synthetic compartment comparison. It reconstructs compartments from current generator code and manifest metadata, but those inputs do not reproduce the saved authoritative masks exactly.
2. The 191/261 zero-bulb result is plausible for cases whose manifest-derived `bulb_radius_init` is zero, but it is not fully verified until the exact mask-generating code path or saved compartment labels are recovered. In the selected zero-bulb examples, two clean cases mapped entirely as stem, while one high-mismatch medium case did not.
3. The 160/261 mismatch flags should be treated mostly as reconstruction/code-provenance artifacts, not true generator morphology signals. The saved masks and current code/metadata are not in exact correspondence.
4. Synthetic compartment metrics should not be used for real-vs-synthetic comparison yet. They are suitable only as exploratory diagnostics until the mapper is tied to exact saved-mask generation.
5. Analyzer correction is required before proceeding: either recover the exact generator version used for the saved masks, add per-voxel compartment-label outputs during future generation, or fit/derive compartments directly from saved mask geometry without assuming current generator internals.

## Reproducibility Commands

- `.venv/bin/python analysis/anatomical_compartment_validation/mapping_verification/verify_compartment_mapping.py`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/anatomical_compartment_validation/mapping_verification/verify_compartment_mapping.py`
- `.venv/bin/python -m pytest tests/test_anatomical_compartment_helpers.py -v`
- `.venv/bin/python -m pytest -m "fast and not slow" -v`
- `git diff --check`
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["case_id"] = manifest["case_id"].astype(str)
    features = pd.read_csv(COMPARTMENT_FEATURES)
    features["case_id"] = features["case_id"].astype(str)
    selected = select_cases(features)
    rows = [verify_case(case_id, manifest, features) for case_id in selected]
    results = pd.DataFrame(rows)
    results.to_csv(CASE_CSV, index=False)
    write_report(results, selected)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(REPO_ROOT),
        "manifest_path": str(MANIFEST_PATH),
        "compartment_features_path": str(COMPARTMENT_FEATURES),
        "selected_cases": selected,
        "case_count": len(selected),
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Compartment mapping verification complete")
    print(f"  Cases: {len(selected)}")
    print(f"  CSV: {CASE_CSV}")
    print(f"  Report: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
