#!/usr/bin/env python3
"""Extract real vestibular schwannoma segmentation features from NIfTI masks.

Usage examples:
  Single segmentation:
    python3 scripts/extract_real_tumor_features.py \
      --seg_path /path/to/case_seg.nii.gz \
      --out_csv /tmp/real_tumor_features.csv

  Glob input:
    python3 scripts/extract_real_tumor_features.py \
      --glob "/data/segs/*_VS*.nii.gz" \
      --out_csv /tmp/real_tumor_features.csv

  CSV batch input:
    python3 scripts/extract_real_tumor_features.py \
      --input_csv /path/to/cases.csv \
      --seg_col seg_path \
      --case_id_col case_id \
      --out_csv /tmp/real_tumor_features.csv \
      --out_json /tmp/real_tumor_features.json

  With dataset summary and axis sign normalization:
    python3 scripts/extract_real_tumor_features.py \
      --glob "/data/segs/*.nii.gz" \
      --reference_axis_vox 0,0,1 \
      --out_csv /tmp/real_tumor_features.csv \
      --summary_json /tmp/real_tumor_feature_summary.json

Notes:
  - Surface area is computed from marching cubes with physical spacing in mm.
  - Principal-axis sign is inherently ambiguous from inertia alone. Use
    --reference_axis_vox or --reference_axis_mm to enforce deterministic sign.
  - TODO: Canal/bulb compartment features require explicit anatomical
    landmarking (IAC/CPA masks) or a reliable inferred porus boundary.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy.ndimage import label as ndi_label
from skimage.measure import marching_cubes, mesh_surface_area


TINY_COMPONENT_FRACTION_THRESHOLD = 0.8
SUMMARY_METRIC_FIELDS = (
    "volume_mm3",
    "surface_area_mm2",
    "sphericity",
    "elongation",
    "flatness",
    "aspect_ratio_major_to_minor2",
    "bbox_fill_fraction",
    "surface_to_volume_ratio",
)


def _safe_case_id_from_path(seg_path: Path) -> str:
    name = seg_path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return seg_path.stem


def _validate_spacing(spacing: Sequence[float], seg_path: Path) -> np.ndarray:
    arr = np.asarray(spacing, dtype=np.float64)
    if arr.shape[0] < 3:
        raise ValueError(f"Invalid spacing for {seg_path}: expected at least 3 values, got {arr}")
    arr = arr[:3]
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"Invalid non-positive voxel spacing for {seg_path}: {arr.tolist()}")
    return arr


def _largest_component(mask: np.ndarray) -> Tuple[np.ndarray, int, int, float, int]:
    labeled, cc_count = ndi_label(mask)
    if cc_count == 0:
        return np.zeros_like(mask, dtype=bool), 0, 0, 0.0, 0

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))
    largest_count = int(counts[largest_label])
    counts_for_second = counts.copy()
    counts_for_second[largest_label] = 0
    second_count = int(np.max(counts_for_second)) if counts_for_second.size > 1 else 0
    total_count = int(mask.sum())
    frac = float(largest_count / total_count) if total_count > 0 else 0.0
    return labeled == largest_label, int(cc_count), largest_count, frac, second_count


def _principal_axes_features(
    coords_vox: np.ndarray,
    spacing_mm: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    if coords_vox.shape[0] < 3:
        return None, None, None, None, True

    coords_mm = coords_vox * spacing_mm.reshape(1, 3)
    centroid_mm = coords_mm.mean(axis=0)
    centered_mm = coords_mm - centroid_mm

    try:
        cov = np.cov(centered_mm, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None, None, None, None, True

    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], a_min=0.0, a_max=None)
    eigvecs = eigvecs[:, order]

    axis_mm = eigvecs[:, 0]
    norm_mm = float(np.linalg.norm(axis_mm))
    if norm_mm == 0.0:
        return None, None, None, None, True
    axis_mm = axis_mm / norm_mm

    # Convert physical direction back to voxel-index direction.
    axis_vox = axis_mm / spacing_mm
    norm_vox = float(np.linalg.norm(axis_vox))
    if norm_vox == 0.0:
        return None, None, None, None, True
    axis_vox = axis_vox / norm_vox

    # "Moments-based" lengths: approximate full axis lengths as 4*sigma.
    # Kept for backward compatibility and continuity with prior output files.
    axis_lengths_mm = 4.0 * np.sqrt(eigvals)

    # "Extent-based" lengths: project points onto principal directions in mm.
    # This is often easier to interpret clinically (span along each axis).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        proj = centered_mm @ eigvecs
    if not np.all(np.isfinite(proj)):
        extent_lengths_mm = None
    else:
        extent_lengths_mm = proj.max(axis=0) - proj.min(axis=0)
    return axis_vox, axis_mm, axis_lengths_mm, extent_lengths_mm, False


def _parse_axis_arg(value: Optional[str], name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    raw = [part.strip() for part in value.split(",")]
    if len(raw) != 3:
        raise ValueError(f"{name} must have 3 comma-separated values (x,y,z).")
    vec = np.asarray([float(part) for part in raw], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError(f"{name} must be finite and non-zero.")
    return vec / norm


def _maybe_align_principal_axis_sign(
    axis_vox: Optional[np.ndarray],
    axis_mm: Optional[np.ndarray],
    reference_axis_vox: Optional[np.ndarray],
    reference_axis_mm: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
    if axis_vox is None or axis_mm is None:
        return axis_vox, axis_mm, False
    if reference_axis_vox is None and reference_axis_mm is None:
        return axis_vox, axis_mm, False

    aligned_vox = np.array(axis_vox, dtype=np.float64)
    aligned_mm = np.array(axis_mm, dtype=np.float64)
    should_flip = False

    if reference_axis_mm is not None:
        should_flip = float(np.dot(aligned_mm, reference_axis_mm)) < 0.0
    else:
        should_flip = float(np.dot(aligned_vox, reference_axis_vox)) < 0.0

    if should_flip:
        aligned_vox = -aligned_vox
        aligned_mm = -aligned_mm
    return aligned_vox, aligned_mm, True


def _surface_area_mm2(component_mask: np.ndarray, spacing_mm: np.ndarray) -> Tuple[Optional[float], bool]:
    if int(component_mask.sum()) < 4:
        return None, True
    try:
        verts, faces, _, _ = marching_cubes(component_mask.astype(np.float32), level=0.5, spacing=tuple(float(x) for x in spacing_mm))
        area = float(mesh_surface_area(verts, faces))
    except Exception:
        return None, True
    if not np.isfinite(area) or area <= 0.0:
        return None, True
    return area, False


def _nan3() -> List[float]:
    return [math.nan, math.nan, math.nan]


def _base_feature_row(case_id: str, seg_path: Path) -> Dict[str, Any]:
    """Return a fully shaped feature row with default NaN/False values.

    Coordinate/unit conventions:
      - *_vox vectors are voxel-index space.
      - *_mm scalar sizes use spacing-scaled physical mm units.
      - centroid_mm is NIfTI affine world-space mm.
    """
    return {
        "case_id": case_id,
        "seg_path": str(seg_path),
        "failed": False,
        "failure_reason": None,
        "voxel_spacing_mm": _nan3(),
        "voxel_volume_mm3": math.nan,
        "mask_voxel_count": math.nan,
        "volume_mm3": math.nan,
        "connected_component_count": math.nan,
        "largest_component_voxel_count": math.nan,
        "largest_component_fraction": math.nan,
        "centroid_vox": _nan3(),
        "centroid_mm": _nan3(),
        "bounding_box_min_vox": _nan3(),
        "bounding_box_max_vox": _nan3(),
        "bounding_box_size_vox": _nan3(),
        "bounding_box_size_mm": _nan3(),
        "principal_axis_vector_vox": _nan3(),
        "principal_axis_vector_mm": _nan3(),
        "principal_axis_lengths_mm": _nan3(),
        "principal_axis_length_major_mm": math.nan,
        "principal_axis_length_minor1_mm": math.nan,
        "principal_axis_length_minor2_mm": math.nan,
        "principal_axis_sign_aligned": False,
        "equivalent_sphere_diameter_mm": math.nan,
        "max_diameter_bbox_mm": math.nan,
        "bbox_volume_mm3": math.nan,
        "bbox_fill_fraction": math.nan,
        "flatness": math.nan,
        "aspect_ratio_major_to_minor2": math.nan,
        "elongation_legacy_major_to_minor2": math.nan,
        "elongation": math.nan,
        "secondary_axis_ratio": math.nan,
        "surface_area_mm2": math.nan,
        "sphericity": math.nan,
        "compactness": math.nan,
        "surface_to_volume_ratio": math.nan,
        "roughness_index": math.nan,
        "component_volume_fraction_largest": math.nan,
        "component_volume_fraction_second_largest": 0.0,
        "empty_mask": False,
        "multiple_components": False,
        "tiny_component_fraction": False,
        "surface_area_failed": False,
        "principal_axis_failed": False,
    }


def _extract_one(
    seg_path: Path,
    case_id: str,
    reference_axis_vox: Optional[np.ndarray],
    reference_axis_mm: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Extract one segmentation's features.

    Unit conventions used in outputs:
      - `*_vox`: voxel-index space.
      - `*_mm` lengths/areas/volumes: physical spacing-aware mm units.
      - `centroid_mm`: world-space coordinates from the NIfTI affine.
      - principal-axis vectors: direction only; sign is ambiguous unless
        reference-axis sign normalization is requested.
    """
    nii = nib.load(str(seg_path))
    data = np.asarray(nii.dataobj)
    spacing_mm = _validate_spacing(nii.header.get_zooms(), seg_path)
    voxel_volume_mm3 = float(np.prod(spacing_mm))
    feature = _base_feature_row(case_id=case_id, seg_path=seg_path)
    feature["voxel_spacing_mm"] = [float(x) for x in spacing_mm.tolist()]
    feature["voxel_volume_mm3"] = voxel_volume_mm3

    mask = data > 0
    mask_voxel_count = int(mask.sum())
    volume_mm3 = float(mask_voxel_count * voxel_volume_mm3)
    feature["mask_voxel_count"] = mask_voxel_count
    feature["volume_mm3"] = volume_mm3

    largest_mask, cc_count, largest_count, largest_frac, second_count = _largest_component(mask)
    feature["connected_component_count"] = cc_count
    feature["largest_component_voxel_count"] = largest_count
    feature["largest_component_fraction"] = largest_frac

    empty_mask = mask_voxel_count == 0
    multiple_components = cc_count > 1
    tiny_component_fraction = (not empty_mask) and (largest_frac < TINY_COMPONENT_FRACTION_THRESHOLD)
    feature["empty_mask"] = empty_mask
    feature["multiple_components"] = multiple_components
    feature["tiny_component_fraction"] = tiny_component_fraction

    principal_axis_sign_aligned = False
    surface_area_failed = False
    principal_axis_failed = False

    if not empty_mask and largest_count > 0:
        coords_vox = np.argwhere(largest_mask).astype(np.float64)
        centroid_vox_arr = coords_vox.mean(axis=0)
        centroid_mm_arr = apply_affine(nii.affine, centroid_vox_arr)

        mins = coords_vox.min(axis=0)
        maxs = coords_vox.max(axis=0)
        size_vox = (maxs - mins + 1.0)
        size_mm = size_vox * spacing_mm

        axis_vox, axis_mm, axis_lengths_mm_arr, extent_lengths_mm_arr, principal_axis_failed = _principal_axes_features(coords_vox, spacing_mm)
        axis_vox, axis_mm, principal_axis_sign_aligned = _maybe_align_principal_axis_sign(
            axis_vox=axis_vox,
            axis_mm=axis_mm,
            reference_axis_vox=reference_axis_vox,
            reference_axis_mm=reference_axis_mm,
        )
        area_mm2, surface_area_failed = _surface_area_mm2(largest_mask, spacing_mm)

        feature["centroid_vox"] = [float(x) for x in centroid_vox_arr.tolist()]
        feature["centroid_mm"] = [float(x) for x in centroid_mm_arr.tolist()]
        feature["bounding_box_min_vox"] = [float(x) for x in mins.tolist()]
        feature["bounding_box_max_vox"] = [float(x) for x in maxs.tolist()]
        feature["bounding_box_size_vox"] = [float(x) for x in size_vox.tolist()]
        feature["bounding_box_size_mm"] = [float(x) for x in size_mm.tolist()]
        max_diameter_bbox_mm = float(np.max(size_mm))
        bbox_volume_mm3 = float(np.prod(size_mm))
        feature["max_diameter_bbox_mm"] = max_diameter_bbox_mm
        feature["bbox_volume_mm3"] = bbox_volume_mm3
        if bbox_volume_mm3 > 0.0:
            feature["bbox_fill_fraction"] = float(volume_mm3 / bbox_volume_mm3)

        component_volume_fraction_largest = float(largest_count / mask_voxel_count) if mask_voxel_count > 0 else math.nan
        component_volume_fraction_second_largest = float(second_count / mask_voxel_count) if mask_voxel_count > 0 else 0.0
        feature["component_volume_fraction_largest"] = component_volume_fraction_largest
        feature["component_volume_fraction_second_largest"] = component_volume_fraction_second_largest

        if volume_mm3 > 0.0:
            feature["equivalent_sphere_diameter_mm"] = float(2.0 * ((3.0 * volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0))

        if axis_vox is not None and axis_mm is not None and axis_lengths_mm_arr is not None:
            feature["principal_axis_vector_vox"] = [float(x) for x in axis_vox.tolist()]
            feature["principal_axis_vector_mm"] = [float(x) for x in axis_mm.tolist()]
            feature["principal_axis_lengths_mm"] = [float(x) for x in axis_lengths_mm_arr.tolist()]
            # Keep prior moments-based output while exposing explicit major/minor metrics.
            moments_sorted = np.sort(axis_lengths_mm_arr)[::-1]
            major_m = float(moments_sorted[0])
            minor1_m = float(moments_sorted[1])
            minor2_m = float(moments_sorted[2])
            feature["principal_axis_length_major_mm"] = major_m
            feature["principal_axis_length_minor1_mm"] = minor1_m
            feature["principal_axis_length_minor2_mm"] = minor2_m

            if minor1_m > 0.0 and minor2_m > 0.0:
                # Canonical elongation definition (major / intermediate minor axis).
                feature["elongation"] = float(major_m / minor1_m)
                # Legacy elongation definition kept explicitly for backwards analyses.
                feature["elongation_legacy_major_to_minor2"] = float(major_m / minor2_m)
                feature["flatness"] = float(minor2_m / minor1_m)
                feature["aspect_ratio_major_to_minor2"] = float(major_m / minor2_m)
                feature["secondary_axis_ratio"] = float(minor1_m / major_m) if major_m > 0.0 else math.nan
            else:
                principal_axis_failed = True

            # If extent-based axis lengths are available and valid, prefer them
            # for the principal_axis_length_* outputs.
            if extent_lengths_mm_arr is not None:
                ext_sorted = np.sort(extent_lengths_mm_arr)[::-1]
                if float(ext_sorted[2]) > 0.0:
                    feature["principal_axis_length_major_mm"] = float(ext_sorted[0])
                    feature["principal_axis_length_minor1_mm"] = float(ext_sorted[1])
                    feature["principal_axis_length_minor2_mm"] = float(ext_sorted[2])

        if area_mm2 is not None:
            surface_area_mm2 = float(area_mm2)
            feature["surface_area_mm2"] = surface_area_mm2
            largest_volume_mm3 = float(largest_count * voxel_volume_mm3)
            if largest_volume_mm3 > 0.0 and surface_area_mm2 > 0.0:
                sphericity = float((math.pi ** (1.0 / 3.0)) * ((6.0 * largest_volume_mm3) ** (2.0 / 3.0)) / surface_area_mm2)
                feature["sphericity"] = sphericity
                # Kept for backward compatibility with previous extractor outputs:
                # compactness = S^3 / V^2 (dimension: 1/length).
                feature["compactness"] = float((surface_area_mm2 ** 3.0) / (largest_volume_mm3 ** 2.0))
                feature["surface_to_volume_ratio"] = float(surface_area_mm2 / largest_volume_mm3)
                feature["roughness_index"] = float(1.0 / sphericity) if sphericity > 0.0 else math.nan

    feature["principal_axis_sign_aligned"] = principal_axis_sign_aligned
    feature["surface_area_failed"] = surface_area_failed
    feature["principal_axis_failed"] = principal_axis_failed
    return feature


def _rows_from_input(args: argparse.Namespace) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []

    if args.seg_path:
        p = Path(args.seg_path).expanduser().resolve()
        rows.append((_safe_case_id_from_path(p), p))

    if args.glob:
        for matched in sorted(glob.glob(args.glob)):
            p = Path(matched).expanduser().resolve()
            rows.append((_safe_case_id_from_path(p), p))

    if args.input_csv:
        with Path(args.input_csv).expanduser().resolve().open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV missing header: {args.input_csv}")
            if args.seg_col not in reader.fieldnames:
                raise ValueError(f"seg_col '{args.seg_col}' missing from CSV columns: {reader.fieldnames}")
            for i, row in enumerate(reader):
                seg_raw = (row.get(args.seg_col) or "").strip()
                if not seg_raw:
                    continue
                seg_path = Path(seg_raw).expanduser().resolve()
                if args.case_id_col and args.case_id_col in row and (row.get(args.case_id_col) or "").strip():
                    case_id = str(row[args.case_id_col]).strip()
                else:
                    case_id = _safe_case_id_from_path(seg_path)
                if not case_id:
                    case_id = f"row_{i}"
                rows.append((case_id, seg_path))

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[Tuple[str, Path]] = []
    for case_id, p in rows:
        key = (case_id, str(p))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((case_id, p))

    if not deduped:
        raise ValueError("No segmentations provided. Use --seg_path, --glob, or --input_csv.")
    return deduped


def _flatten_for_csv(feature: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case_id": feature["case_id"],
        "seg_path": feature["seg_path"],
        "failed": feature["failed"],
        "failure_reason": feature["failure_reason"],
        "voxel_volume_mm3": feature["voxel_volume_mm3"],
        "mask_voxel_count": feature["mask_voxel_count"],
        "volume_mm3": feature["volume_mm3"],
        "connected_component_count": feature["connected_component_count"],
        "largest_component_voxel_count": feature["largest_component_voxel_count"],
        "largest_component_fraction": feature["largest_component_fraction"],
        "component_volume_fraction_largest": feature["component_volume_fraction_largest"],
        "component_volume_fraction_second_largest": feature["component_volume_fraction_second_largest"],
        "equivalent_sphere_diameter_mm": feature["equivalent_sphere_diameter_mm"],
        "max_diameter_bbox_mm": feature["max_diameter_bbox_mm"],
        "bbox_volume_mm3": feature["bbox_volume_mm3"],
        "bbox_fill_fraction": feature["bbox_fill_fraction"],
        "principal_axis_length_major_mm": feature["principal_axis_length_major_mm"],
        "principal_axis_length_minor1_mm": feature["principal_axis_length_minor1_mm"],
        "principal_axis_length_minor2_mm": feature["principal_axis_length_minor2_mm"],
        "principal_axis_sign_aligned": feature["principal_axis_sign_aligned"],
        "flatness": feature["flatness"],
        "aspect_ratio_major_to_minor2": feature["aspect_ratio_major_to_minor2"],
        "elongation_legacy_major_to_minor2": feature["elongation_legacy_major_to_minor2"],
        "elongation": feature["elongation"],
        "secondary_axis_ratio": feature["secondary_axis_ratio"],
        "surface_area_mm2": feature["surface_area_mm2"],
        "sphericity": feature["sphericity"],
        "compactness": feature["compactness"],
        "surface_to_volume_ratio": feature["surface_to_volume_ratio"],
        "roughness_index": feature["roughness_index"],
        "empty_mask": feature["empty_mask"],
        "multiple_components": feature["multiple_components"],
        "tiny_component_fraction": feature["tiny_component_fraction"],
        "surface_area_failed": feature["surface_area_failed"],
        "principal_axis_failed": feature["principal_axis_failed"],
    }

    vectors = {
        "voxel_spacing_mm": feature["voxel_spacing_mm"],
        "centroid_vox": feature["centroid_vox"],
        "centroid_mm": feature["centroid_mm"],
        "bounding_box_min_vox": feature["bounding_box_min_vox"],
        "bounding_box_max_vox": feature["bounding_box_max_vox"],
        "bounding_box_size_vox": feature["bounding_box_size_vox"],
        "bounding_box_size_mm": feature["bounding_box_size_mm"],
        "principal_axis_vector_vox": feature["principal_axis_vector_vox"],
        "principal_axis_vector_mm": feature["principal_axis_vector_mm"],
        "principal_axis_lengths_mm": feature["principal_axis_lengths_mm"],
    }

    for key, values in vectors.items():
        for idx, axis in enumerate(("x", "y", "z")):
            row[f"{key}_{axis}"] = values[idx]

    return row


def _write_csv(features: List[Dict[str, Any]], out_csv: Path) -> None:
    rows = [_flatten_for_csv(item) for item in features]
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract real-tumor segmentation features into CSV/JSON.")
    parser.add_argument("--seg_path", type=str, default=None, help="Single segmentation NIfTI path.")
    parser.add_argument("--glob", type=str, default=None, help="Glob pattern for segmentation files.")
    parser.add_argument("--input_csv", type=str, default=None, help="CSV containing segmentation paths.")
    parser.add_argument("--seg_col", type=str, default="seg_path", help="Segmentation path column in --input_csv.")
    parser.add_argument("--case_id_col", type=str, default="case_id", help="Optional case ID column in --input_csv.")
    parser.add_argument("--reference_axis_vox", type=str, default=None, help="Optional axis for principal-axis sign alignment in voxel space, format: x,y,z")
    parser.add_argument("--reference_axis_mm", type=str, default=None, help="Optional axis for principal-axis sign alignment in physical/mm space, format: x,y,z")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path.")
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--summary_json", type=str, default=None, help="Optional output JSON path for dataset-level summary stats.")
    return parser


def _summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "q25": None, "q75": None}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not any([args.seg_path, args.glob, args.input_csv]):
        parser.error("Provide at least one input source: --seg_path, --glob, or --input_csv")
    if args.reference_axis_vox and args.reference_axis_mm:
        parser.error("Provide at most one of --reference_axis_vox or --reference_axis_mm")

    reference_axis_vox = _parse_axis_arg(args.reference_axis_vox, "--reference_axis_vox")
    reference_axis_mm = _parse_axis_arg(args.reference_axis_mm, "--reference_axis_mm")

    items = _rows_from_input(args)

    features: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    multiple_components_count = 0

    for case_id, seg_path in items:
        try:
            feat = _extract_one(
                seg_path=seg_path,
                case_id=case_id,
                reference_axis_vox=reference_axis_vox,
                reference_axis_mm=reference_axis_mm,
            )
            features.append(feat)
            if bool(feat.get("multiple_components", False)):
                multiple_components_count += 1
        except Exception as exc:
            reason = str(exc)
            failed.append({"case_id": case_id, "seg_path": str(seg_path), "error": reason})
            fail_row = _base_feature_row(case_id=case_id, seg_path=seg_path)
            fail_row["failed"] = True
            fail_row["failure_reason"] = reason
            features.append(fail_row)

    out_csv = Path(args.out_csv).expanduser().resolve()
    _write_csv(features, out_csv)

    successful = [item for item in features if not bool(item.get("failed", False))]
    summary_payload = {
        "n_cases": len(features),
        "n_successful": len(successful),
        "n_failed": len(failed),
        "n_empty_mask": int(sum(bool(item.get("empty_mask", False)) for item in successful)),
        "n_multiple_components": int(sum(bool(item.get("multiple_components", False)) for item in successful)),
        "n_surface_area_failed": int(sum(bool(item.get("surface_area_failed", False)) for item in successful)),
        "n_principal_axis_failed": int(sum(bool(item.get("principal_axis_failed", False)) for item in successful)),
    }
    for field in SUMMARY_METRIC_FIELDS:
        summary_payload[field] = _summary_stats([float(item.get(field, math.nan)) for item in successful])

    if args.out_json:
        out_json = Path(args.out_json).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features": features,
            "failed": failed,
            "summary": dict(summary_payload, tiny_component_fraction_threshold=TINY_COMPONENT_FRACTION_THRESHOLD),
        }
        out_json.write_text(json.dumps(payload, indent=2))

    if args.summary_json:
        summary_json = Path(args.summary_json).expanduser().resolve()
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary_payload, indent=2))

    print("Real-tumor feature extraction complete")
    print(f"  Processed: {len(features)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Multiple components: {multiple_components_count}")
    print(f"  CSV: {out_csv}")
    if args.out_json:
        print(f"  JSON: {Path(args.out_json).expanduser().resolve()}")
    if args.summary_json:
        print(f"  Summary JSON: {Path(args.summary_json).expanduser().resolve()}")

    if failed:
        print("  Failed cases:")
        for item in failed:
            print(f"    - {item['case_id']} :: {item['seg_path']} :: {item['error']}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
