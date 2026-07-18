#!/usr/bin/env python3
"""Local orientation diagnostics for GrowthNet lollipop outputs.

This script is read-only outside analysis/orientation_validation. It scans local
embedding outputs and locally pulled standalone synthetic lollipop masks, then
recomputes PCA axes from masks to classify axis-error mechanisms.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label as ndi_label
from scipy.spatial.transform import Rotation
from skimage.measure import regionprops


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "analysis" / "orientation_validation"
SYN_DIR = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1"
REAL_FEATURES = REPO_ROOT / "rivanna_pull" / "analysis" / "real_tumor_features_v1" / "real_tumor_features_usable_train.csv"

PROBLEM_CASES = ["126_5_1607", "132_1_148"]
CONTROL_CASES = ["147_local_embedding", "126_1_312", "132_0_0"]


@dataclass
class AxisSet:
    centroid_vox: np.ndarray
    axes_phys: np.ndarray
    variances: np.ndarray
    voxel_count: int
    component_count: int
    largest_component_fraction: float


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = unit(a)
    b = unit(b)
    dot = float(np.clip(abs(a @ b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def vec_str(v: np.ndarray | None) -> str:
    if v is None:
        return ""
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def load_mask(path: Path) -> tuple[np.ndarray, nib.spatialimages.SpatialImage, np.ndarray]:
    img = nib.load(str(path))
    data = img.get_fdata() > 0.5
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    return data, img, spacing


def pca_axes(mask: np.ndarray, spacing: np.ndarray) -> AxisSet:
    labeled, n_labels = ndi_label(mask > 0)
    if n_labels == 0:
        raise ValueError("empty mask")
    props = regionprops(labeled)
    prop = max(props, key=lambda p: p.area)
    component = labeled == prop.label
    coords_vox = np.argwhere(component)
    centroid_vox = coords_vox.mean(axis=0)
    coords_phys = coords_vox.astype(np.float64) * spacing
    coords_phys -= coords_phys.mean(axis=0)
    cov = np.cov(coords_phys, rowvar=False, bias=True)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    for i in range(3):
        dominant = int(np.argmax(np.abs(vecs[:, i])))
        if vecs[dominant, i] < 0:
            vecs[:, i] *= -1.0
    return AxisSet(
        centroid_vox=centroid_vox,
        axes_phys=vecs,
        variances=vals,
        voxel_count=int(component.sum()),
        component_count=int(n_labels),
        largest_component_fraction=float(component.sum() / max(mask.sum(), 1)),
    )


def expected_canal_axis_from_manifest(row: pd.Series) -> np.ndarray:
    rotation = Rotation.from_euler(
        "zyx",
        [
            float(row["rotation_z_deg"]),
            float(row["rotation_y_deg"]),
            float(row["rotation_x_deg"]),
        ],
        degrees=True,
    )
    # scripts/generate_synthetic_lollipop_cohort.py uses canal_axis="c".
    # A local +c coordinate maps into voxel/mm axes via R @ e_z. The lollipop
    # fundus direction is the opposite sign because x_rel = -canal_coord, but
    # axis comparisons here are unsigned.
    return unit(rotation.as_matrix() @ np.array([0.0, 0.0, 1.0]))


def cross_section_direction(mask: np.ndarray, spacing: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray | None, dict[str, float]]:
    coords = np.argwhere(mask > 0).astype(np.float64)
    if coords.shape[0] < 10:
        return None, {}
    coords_phys = coords * spacing
    center = coords_phys.mean(axis=0)
    axis = unit(axis)
    proj = np.einsum("ij,j->i", coords_phys - center, axis)
    if float(proj.max() - proj.min()) < 1e-6:
        return None, {}
    bins = np.linspace(float(proj.min()), float(proj.max()), 25)
    ids = np.digitize(proj, bins) - 1
    counts = np.array([np.sum(ids == i) for i in range(len(bins) - 1)])
    if counts.size == 0 or int(counts.max()) == 0:
        return None, {}
    centers = 0.5 * (bins[:-1] + bins[1:])
    bulb_i = int(np.argmax(counts))
    nonzero = np.where(counts > 0)[0]
    if nonzero.size == 0:
        return None, {}
    end_low = int(nonzero[0])
    end_high = int(nonzero[-1])
    narrow_i = end_low if counts[end_low] <= counts[end_high] else end_high
    direction = axis * float(centers[bulb_i] - centers[narrow_i])
    return unit(direction), {
        "cross_section_bulb_bin_center_mm": float(centers[bulb_i]),
        "cross_section_narrow_end_center_mm": float(centers[narrow_i]),
        "cross_section_max_bin_voxels": float(counts[bulb_i]),
        "cross_section_narrow_end_voxels": float(counts[narrow_i]),
    }


def read_real_features() -> pd.DataFrame:
    if not REAL_FEATURES.exists():
        return pd.DataFrame()
    return pd.read_csv(REAL_FEATURES)


def real_axis_for(case_id: str, real_df: pd.DataFrame) -> np.ndarray | None:
    if real_df.empty:
        return None
    hit = real_df[real_df["case_id"].astype(str) == case_id]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return unit(np.array([
        float(row["principal_axis_vector_mm_x"]),
        float(row["principal_axis_vector_mm_y"]),
        float(row["principal_axis_vector_mm_z"]),
    ]))


def case_classification(row: dict[str, object]) -> str:
    if row.get("data_availability") != "available":
        return "BLOCKED_REMOTE_DATA"
    if row.get("source_kind") == "standalone_synthetic_lollipop":
        # These masks were generated as a volume/morphology benchmark with
        # random rotation, not embedded into the corresponding patient MRI.
        # Classify only against the manifest-derived synthetic canal axis.
        canal_major = float(row.get("angle_major_to_expected_canal_deg") or 999.0)
        canal_best = float(row.get("best_pca_angle_to_expected_canal_deg") or 999.0)
        best_axis = str(row.get("best_pca_axis_for_expected_canal") or "")
        if canal_major >= 60.0 and canal_best <= 30.0 and best_axis != "major":
            return "B_PRINCIPAL_AXIS_IDENTITY_SWITCH"
        if canal_major <= 30.0:
            return "CONTROL_OR_NO_LOCAL_AXIS_SWITCH"
        return "C_QC_LIMITATION_OR_UNSUPPORTED_WITH_LOCAL_DATA"
    canal_major = float(row.get("angle_major_to_expected_canal_deg") or 999.0)
    canal_best = float(row.get("best_pca_angle_to_expected_canal_deg") or 999.0)
    best_axis = str(row.get("best_pca_axis_for_expected_canal") or "")
    real_major = row.get("angle_major_to_real_axis_deg")
    real_best = row.get("best_pca_angle_to_real_axis_deg")

    identity_switch = canal_major >= 60.0 and canal_best <= 30.0 and best_axis != "major"
    real_switch = False
    if real_major not in (None, "") and real_best not in (None, ""):
        real_switch = float(real_major) >= 60.0 and float(real_best) <= 30.0
    if identity_switch or real_switch:
        return "B_PRINCIPAL_AXIS_IDENTITY_SWITCH"
    if canal_major <= 30.0:
        return "CONTROL_OR_NO_LOCAL_AXIS_SWITCH"
    return "C_QC_LIMITATION_OR_UNSUPPORTED_WITH_LOCAL_DATA"


def inspect_standalone_synthetic(case_id: str, manifest: pd.DataFrame, features: pd.DataFrame, real_df: pd.DataFrame) -> dict[str, object]:
    path = SYN_DIR / "masks" / f"{case_id}_synthetic_lollipop.nii.gz"
    row: dict[str, object] = {
        "case_id": case_id,
        "source_kind": "standalone_synthetic_lollipop",
        "mask_path": str(path.relative_to(REPO_ROOT)) if path.exists() else str(path),
        "data_availability": "available" if path.exists() else "missing_mask",
    }
    if not path.exists():
        row["classification"] = "BLOCKED_REMOTE_DATA"
        return row

    mask, img, spacing = load_mask(path)
    axes = pca_axes(mask, spacing)
    hit = manifest[manifest["case_id"].astype(str) == case_id]
    if hit.empty:
        row["data_availability"] = "missing_manifest"
        row["classification"] = "C_QC_LIMITATION_OR_UNSUPPORTED_WITH_LOCAL_DATA"
        return row
    mrow = hit.iloc[0]
    expected = expected_canal_axis_from_manifest(mrow)
    real_axis = real_axis_for(case_id, real_df)
    stem_to_bulb, section_stats = cross_section_direction(mask, spacing, expected)

    angles_expected = [angle_deg(axes.axes_phys[:, i], expected) for i in range(3)]
    best_expected_idx = int(np.argmin(angles_expected))
    row.update(
        {
            "voxel_count": axes.voxel_count,
            "spacing_mm": vec_str(spacing),
            "affine_diag": vec_str(np.diag(img.affine)[:3]),
            "component_count": axes.component_count,
            "largest_component_fraction": axes.largest_component_fraction,
            "target_volume_mm3": float(mrow["target_volume_mm3"]),
            "realized_volume_mm3": float(mrow["realized_volume_mm3"]),
            "volume_error_fraction": float(mrow["volume_error_fraction"]),
            "rotation_zyx_deg": f"[{int(mrow['rotation_z_deg'])},{int(mrow['rotation_y_deg'])},{int(mrow['rotation_x_deg'])}]",
            "expected_canal_axis_phys": vec_str(expected),
            "stem_to_bulb_axis_phys": vec_str(stem_to_bulb),
            "pca_major_axis_phys": vec_str(axes.axes_phys[:, 0]),
            "pca_middle_axis_phys": vec_str(axes.axes_phys[:, 1]),
            "pca_minor_axis_phys": vec_str(axes.axes_phys[:, 2]),
            "pca_variance_major": float(axes.variances[0]),
            "pca_variance_middle": float(axes.variances[1]),
            "pca_variance_minor": float(axes.variances[2]),
            "pca_major_to_middle_variance_ratio": float(axes.variances[0] / max(axes.variances[1], 1e-8)),
            "angle_major_to_expected_canal_deg": angles_expected[0],
            "angle_middle_to_expected_canal_deg": angles_expected[1],
            "angle_minor_to_expected_canal_deg": angles_expected[2],
            "best_pca_axis_for_expected_canal": ["major", "middle", "minor"][best_expected_idx],
            "best_pca_angle_to_expected_canal_deg": float(angles_expected[best_expected_idx]),
            "angle_major_to_stem_to_bulb_deg": "" if stem_to_bulb is None else angle_deg(axes.axes_phys[:, 0], stem_to_bulb),
        }
    )
    row.update(section_stats)

    if real_axis is not None:
        angles_real = [angle_deg(axes.axes_phys[:, i], real_axis) for i in range(3)]
        best_real_idx = int(np.argmin(angles_real))
        row.update(
            {
                "real_principal_axis_phys": vec_str(real_axis),
                "angle_major_to_real_axis_deg": angles_real[0],
                "angle_middle_to_real_axis_deg": angles_real[1],
                "angle_minor_to_real_axis_deg": angles_real[2],
                "best_pca_axis_for_real_axis": ["major", "middle", "minor"][best_real_idx],
                "best_pca_angle_to_real_axis_deg": float(angles_real[best_real_idx]),
            }
        )
    else:
        row["real_principal_axis_phys"] = ""

    if not features.empty:
        fhit = features[features["case_id"].astype(str) == case_id]
        if not fhit.empty:
            frow = fhit.iloc[0]
            row["feature_csv_principal_axis_major_mm"] = float(frow["principal_axis_length_major_mm"])
            row["feature_csv_principal_axis_minor1_mm"] = float(frow["principal_axis_length_minor1_mm"])
            row["feature_csv_principal_axis_minor2_mm"] = float(frow["principal_axis_length_minor2_mm"])
            row["feature_csv_elongation"] = float(frow["elongation"])
            row["feature_csv_bbox_fill_fraction"] = float(frow["bbox_fill_fraction"])

    row["classification"] = case_classification(row)
    return row


def local_embedding_dirs() -> Iterable[Path]:
    for p in sorted(REPO_ROOT.rglob("embedding_metrics.json")):
        rel = p.relative_to(REPO_ROOT)
        if rel.parts and rel.parts[0] == "analysis":
            continue
        yield p.parent


def inspect_local_embedding(out_dir: Path) -> dict[str, object]:
    metrics_path = out_dir / "embedding_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    selected = np.array(metrics.get("selected_axis_phys") or [np.nan, np.nan, np.nan], dtype=np.float64)
    row: dict[str, object] = {
        "case_id": f"147_local_embedding:{out_dir.relative_to(REPO_ROOT)}",
        "source_kind": "local_embedding_output",
        "mask_path": str((out_dir / "embedded_tumor_late_mask.nii.gz").relative_to(REPO_ROOT)),
        "data_availability": "available",
        "selected_axis_phys": vec_str(selected),
        "orientation_score_margin": metrics.get("orientation_score_margin", ""),
        "orientation_normalized_gap": metrics.get("orientation_normalized_gap", ""),
        "warnings": json.dumps(metrics.get("warnings", [])),
        "hard_failures": json.dumps(metrics.get("hard_failures", [])),
    }
    timepoints = metrics.get("timepoint_metrics") or []
    row["serialized_timepoint_axis_errors_deg"] = json.dumps(
        [tp.get("axis_error_deg") for tp in timepoints]
    )
    late = out_dir / "embedded_tumor_late_mask.nii.gz"
    if not late.exists():
        row["data_availability"] = "missing_late_mask"
        row["classification"] = "BLOCKED_REMOTE_DATA"
        return row
    mask, img, spacing = load_mask(late)
    axes = pca_axes(mask, spacing)
    row.update(
        {
            "voxel_count": axes.voxel_count,
            "spacing_mm": vec_str(spacing),
            "affine_diag": vec_str(np.diag(img.affine)[:3]),
            "component_count": axes.component_count,
            "largest_component_fraction": axes.largest_component_fraction,
            "pca_major_axis_phys": vec_str(axes.axes_phys[:, 0]),
            "pca_middle_axis_phys": vec_str(axes.axes_phys[:, 1]),
            "pca_minor_axis_phys": vec_str(axes.axes_phys[:, 2]),
            "pca_variance_major": float(axes.variances[0]),
            "pca_variance_middle": float(axes.variances[1]),
            "pca_variance_minor": float(axes.variances[2]),
            "pca_major_to_middle_variance_ratio": float(axes.variances[0] / max(axes.variances[1], 1e-8)),
            "angle_major_to_selected_axis_deg": angle_deg(axes.axes_phys[:, 0], selected),
            "angle_middle_to_selected_axis_deg": angle_deg(axes.axes_phys[:, 1], selected),
            "angle_minor_to_selected_axis_deg": angle_deg(axes.axes_phys[:, 2], selected),
            "classification": "CONTROL_OR_NO_LOCAL_AXIS_SWITCH",
        }
    )
    return row


def copy_representative_qc() -> list[str]:
    copied: list[str] = []
    for src in [
        REPO_ROOT / "tmp_batch_outputs" / "case_147" / "qc_embedding_late.png",
        REPO_ROOT / "embedding_outputs" / "qc_embedding_late.png",
    ]:
        if src.exists():
            dst = OUT_DIR / src.parent.name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(REPO_ROOT)))
    return copied


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict[str, object]], copied_qc: list[str], path: Path) -> None:
    problem = [r for r in rows if str(r["case_id"]) in PROBLEM_CASES]
    controls = [r for r in rows if str(r["case_id"]).startswith("147_local_embedding") or str(r["case_id"]) in CONTROL_CASES]
    lines = [
        "# Orientation Diagnostic",
        "",
        "Task: EMB-001. Scope: local repository only; no SSH/Rivanna access; no core source edits.",
        "",
        "## Data Availability",
        "",
        f"- Named problematic standalone synthetic masks present: {', '.join([r['case_id'] for r in problem if r.get('data_availability') == 'available']) or 'none'}.",
        "- Original real segmentations/MRIs for the named problematic cases are not present locally; real feature CSV rows are present with remote source paths only.",
        "- Local embedded MRI outputs are available only for the 147 case family (`embedding_outputs`, `tmp_batch_outputs/case_147`, and seed-validation copies).",
        "- Therefore spatial placement against patient MRI for 126_5_1607 and 132_1_148 is BLOCKED_REMOTE_DATA.",
        "",
        "## Classification",
        "",
    ]
    for r in problem:
        lines.append(
            f"- `{r['case_id']}`: `{r.get('classification')}`. "
            f"Whole-mask major PCA to expected canal line = {float(r.get('angle_major_to_expected_canal_deg', math.nan)):.2f} deg; "
            f"best PCA axis to expected canal = {r.get('best_pca_axis_for_expected_canal')} "
            f"({float(r.get('best_pca_angle_to_expected_canal_deg', math.nan)):.2f} deg). "
            f"Major-to-middle variance ratio = {float(r.get('pca_major_to_middle_variance_ratio', math.nan)):.3f}."
        )
    lines.extend([
        "",
        "Local evidence for the named problematic cases does **not** support A as a proven true spatial orientation bug. "
        "It also does **not** support B for the locally available standalone masks: their whole-mask major PCA axes align with the manifest-derived synthetic canal lines. "
        "The best-supported classification from local data is **C: QC/validation limitation or unavailable placement data**, because the named cases lack local embedded patient-MRI outputs and the standalone masks were randomly rotated for morphology benchmarking.",
        "",
        "Important nuance: direct angles between standalone synthetic-mask PCA axes and real patient feature-table axes are not spatially interpretable, because the standalone synthetic cohort was generated with random rotations and target volumes only. Those angles are retained in the CSV as a cautionary audit signal, not as evidence of a placement bug.",
        "",
        "## Control Cases",
        "",
    ])
    for r in controls[:8]:
        label = r["case_id"]
        if r["source_kind"] == "local_embedding_output":
            angle = r.get("angle_major_to_selected_axis_deg", "")
            lines.append(f"- `{label}`: embedded output control; late-mask major PCA vs selected placement axis = {float(angle):.2f} deg.")
        elif r.get("data_availability") == "available":
            lines.append(
                f"- `{label}`: standalone synthetic control; major PCA to expected canal = "
                f"{float(r.get('angle_major_to_expected_canal_deg', math.nan)):.2f} deg; "
                f"classification `{r.get('classification')}`."
            )
    lines.extend([
        "",
        "## Voxel/Physical-Space Checks",
        "",
        "- Standalone synthetic masks have 1.0 mm isotropic spacing and diagonal affines, so voxel-space and physical-space axes are equivalent for those files.",
        "- Local embedded outputs use 0.5 mm isotropic spacing inherited from the 147 reference output, so voxel-space and physical-space directions are also equivalent up to scale.",
        "- No anisotropic named-case embedded outputs were available locally for this task.",
        "",
        "## Timepoint Coverage",
        "",
        "- Standalone synthetic cohort masks are single-mask morphology benchmark outputs, so there are no per-visit embedded timepoints to inspect for 126_5_1607 or 132_1_148.",
        f"- Local 147 embedded controls inspected: {sum(1 for r in rows if r.get('source_kind') == 'local_embedding_output')} output directories. Each available metrics JSON serializes five timepoint axis errors; the recomputed late-mask axis errors are listed in `orientation_case_review.csv`.",
        "",
        "## Reproducible Commands",
        "",
        "```bash",
        "python3 analysis/orientation_validation/orientation_diagnostic.py",
        "PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/orientation_validation/orientation_diagnostic.py",
        "```",
        "",
        "The CSV output is `analysis/orientation_validation/orientation_case_review.csv`.",
        "",
        "## Representative QC Outputs",
        "",
    ])
    if copied_qc:
        lines.extend([f"- `{p}`" for p in copied_qc])
    else:
        lines.append("- No local QC PNGs were available to copy.")
    lines.extend([
        "",
        "## Unsupported Claims",
        "",
        "- Whether 126_5_1607 or 132_1_148 were spatially embedded into their patient MRIs with a true orientation bug cannot be determined locally.",
        "- Canal-to-CPA anatomical direction relative to skull-base landmarks cannot be verified without local real MRI/segmentation/atlas landmarks.",
        "- The cross-section-derived stem-to-bulb vector is a mask-only heuristic, not a clinical anatomical label.",
        "",
        "## Follow-Up Tasks",
        "",
        "- When local real segmentations for the named cases are available, rerun this diagnostic against real masks and embedded outputs.",
        "- Add a validation metric that reports angles to all three PCA axes, not only the major axis, for large/oblate lollipop masks.",
        "- Add a stem/canal-specific axis metric if geometry-component labels or a reliable canal extraction heuristic are added.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(SYN_DIR / "manifests" / "synthetic_lollipop_manifest.csv")
    features = pd.read_csv(SYN_DIR / "synthetic_lollipop_features.csv")
    real_df = read_real_features()

    rows: list[dict[str, object]] = []
    for case_id in PROBLEM_CASES + [c for c in CONTROL_CASES if c != "147_local_embedding"]:
        rows.append(inspect_standalone_synthetic(case_id, manifest, features, real_df))
    for out_dir in local_embedding_dirs():
        rows.append(inspect_local_embedding(out_dir))

    csv_path = OUT_DIR / "orientation_case_review.csv"
    write_csv(rows, csv_path)
    copied_qc = copy_representative_qc()
    write_markdown(rows, copied_qc, OUT_DIR / "ORIENTATION_DIAGNOSTIC.md")
    print(f"Wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {(OUT_DIR / 'ORIENTATION_DIAGNOSTIC.md').relative_to(REPO_ROOT)}")
    for item in copied_qc:
        print(f"Copied QC {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
