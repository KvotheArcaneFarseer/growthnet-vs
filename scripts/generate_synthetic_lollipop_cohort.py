#!/usr/bin/env python3
"""Generate a synthetic lollipop tumor cohort from target volumes.

Reads a CSV with at least: case_id,target_volume_mm3
Writes one synthetic mask per case plus a manifest CSV.

This script is intentionally self-contained and only depends on
projects.vivit.src.data.synthetic.create_synthetic_time_3d.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np


# Keep repo-root import fix.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.vivit.src.data.synthetic import create_synthetic_time_3d  # noqa: E402


def _stable_seed(base_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{case_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _parse_spacing(raw: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--voxel_spacing_mm must be 'sx,sy,sz'")
    vals = tuple(float(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise ValueError("--voxel_spacing_mm values must be > 0")
    return vals


def _read_targets(csv_path: Path, case_col: str, vol_col: str) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV missing header: {csv_path}")
        if case_col not in reader.fieldnames or vol_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must include columns '{case_col}' and '{vol_col}'. Found: {reader.fieldnames}"
            )
        rows = [dict(r) for r in reader]
    if not rows:
        raise ValueError(f"No rows in targets CSV: {csv_path}")
    return rows


def _target_shape_controls(target_volume_mm3: float) -> Tuple[float, float]:
    """Return (tinyness, maturity) in [0,1] based on target volume.

    tinyness=1 => very small tumor (reduce dramatic bulb/stem exaggeration)
    maturity=1 => larger target (allow fuller lollipop expression)
    """
    # Soft thresholds chosen to keep the single-mask benchmark from becoming a
    # dramatic canal-dominant lollipop too early in the size distribution.
    maturity = float(np.clip((target_volume_mm3 - 120.0) / 1200.0, 0.0, 1.0))
    tinyness = 1.0 - maturity
    return tinyness, maturity


def _map_scale_to_lollipop_geometry(
    linear_scale_vox: float,
    target_volume_mm3: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Map one scalar size parameter to lollipop geometry.

    This mapping is intentionally less axis-stretched than a dramatic stem+bulb prior,
    and it softens the morphology for tiny targets.
    """
    tinyness, maturity = _target_shape_controls(target_volume_mm3)

    # Deterministic per-case variability (small amplitude).
    j1 = float(rng.uniform(-0.035, 0.035))
    j2 = float(rng.uniform(-0.035, 0.035))
    j3 = float(rng.uniform(-0.040, 0.040))
    j4 = float(rng.uniform(-0.040, 0.040))

    # Softer, less elongated base geometry.  Keep the canal cross-section close
    # to round so the principal axes do not collapse into a thin stem.
    base_radius_max = linear_scale_vox * (1.00 + 0.03 * maturity + j1)
    apex_radius_max = base_radius_max * (0.99 + 0.01 * maturity + j2)

    # Keep canal short-to-moderate to reduce major/minor disparity.
    canal_length_max = linear_scale_vox * (0.62 + 0.32 * maturity + j3)

    # Bulb remains present but with controlled size progression.
    bulb_radius_max = linear_scale_vox * (0.88 + 0.22 * maturity + j4)

    # Single-mask generation uses the initial dimensions.  Make that first mask
    # compact: shorter canal and fuller cross-section, with very low absolute
    # floors so tiny targets can calibrate below the old minimum-volume tail.
    canal_length_init = max(0.08, canal_length_max * (0.10 + 0.10 * maturity))
    base_radius_init = max(0.10, base_radius_max * (0.31 + 0.08 * maturity))
    apex_radius_init = max(0.10, apex_radius_max * (0.32 + 0.07 * maturity))

    # Tiny/medium targets: keep bulb absent so the first single mask does not
    # spawn a disconnected CPA island.  Larger targets get a connected, modest
    # CPA component instead of a tiny detached one.
    if maturity < 0.75:
        bulb_radius_init = 0.0
    else:
        bulb_radius_init = max(0.0, bulb_radius_max * (0.18 + 0.02 * (maturity - 0.75) / 0.25))

    # Single-mask benchmark: keep growth multipliers conservative.
    canal_growth_scale = 0.54 + 0.20 * maturity
    bulb_growth_scale = 0.58 + 0.22 * maturity

    # Guardrails.
    base_radius_max = max(0.10, base_radius_max)
    apex_radius_max = max(0.08, min(apex_radius_max, base_radius_max * 1.01))
    canal_length_max = max(0.08, canal_length_max)
    bulb_radius_max = max(0.08, bulb_radius_max)

    return {
        "canal_base_radius_init": float(base_radius_init),
        "canal_apex_radius_init": float(apex_radius_init),
        "canal_length_init": float(canal_length_init),
        "bulb_radius_init": float(bulb_radius_init),
        "canal_base_radius_max": float(base_radius_max),
        "canal_apex_radius_max": float(apex_radius_max),
        "canal_length_max_override": float(canal_length_max),
        "bulb_radius_max": float(bulb_radius_max),
        "canal_growth_scale": float(canal_growth_scale),
        "bulb_growth_scale": float(bulb_growth_scale),
    }


def _grid_and_radmax_from_scale(scale_vox: float) -> Tuple[int, int]:
    # Dynamic cube size to avoid clipping while staying efficient.
    span = int(math.ceil(34.0 + 8.0 * scale_vox))
    size = max(56, min(224, span))
    rad_max = max(6, min(size // 3, int(math.ceil(3.2 * scale_vox + 5.0))))
    return size, rad_max


def _generate_one_mask(
    target_volume_mm3: float,
    linear_scale_vox: float,
    case_seed: int,
    canal_axis: str,
    rotation_zyx_deg: Sequence[int],
) -> np.ndarray:
    # Split RNG streams so geometry jitter is deterministic and independent of binary search order.
    rng_geom = np.random.default_rng(case_seed ^ 0xBADC0DE)
    rng_series = np.random.default_rng(case_seed ^ 0x1234ABCD)

    geom = _map_scale_to_lollipop_geometry(
        linear_scale_vox=linear_scale_vox,
        target_volume_mm3=target_volume_mm3,
        rng=rng_geom,
    )

    size, rad_max = _grid_and_radmax_from_scale(scale_vox=linear_scale_vox)

    _, labels = create_synthetic_time_3d(
        height=size,
        width=size,
        depth=size,
        dates=[0],
        rotation_degrees=list(rotation_zyx_deg),
        rad_max=rad_max,
        rad_min=1,
        noise_max=0.05,
        num_seg_classes=5,
        channel_dim=None,
        growth="steady",
        growth_direction="a",
        off_axis_growth_p=0.0,
        geometry_mode="lollipop",
        canal_axis=canal_axis,
        centered=True,
        random_state=rng_series,
        **geom,
    )

    # Single-mask benchmark: use the only generated label.
    label = labels[0]
    if label.ndim == 4:
        # Defensive: should not happen with channel_dim=None.
        label = np.squeeze(label)
    return (label > 0).astype(np.uint8)


def _volume_mm3(mask: np.ndarray, voxel_volume_mm3: float) -> float:
    return float(int(mask.sum()) * voxel_volume_mm3)


def _calibrate_scale(
    target_volume_mm3: float,
    case_seed: int,
    canal_axis: str,
    rotation_zyx_deg: Sequence[int],
    voxel_volume_mm3: float,
    tolerance_frac: float,
    max_iters: int,
    min_scale_vox: float,
    max_scale_vox: float,
) -> Tuple[np.ndarray, float, float, int]:
    """Binary-search-like calibration over linear_scale_vox with bounded expansion."""

    # Sphere-equivalent heuristic for initial scale.
    eq_radius_mm = ((3.0 * max(target_volume_mm3, 1e-6)) / (4.0 * math.pi)) ** (1.0 / 3.0)
    init_scale = float(np.clip(eq_radius_mm * 0.85, min_scale_vox, max_scale_vox))

    low = float(np.clip(init_scale * 0.35, min_scale_vox, max_scale_vox))
    high = float(np.clip(init_scale * 2.00, min_scale_vox, max_scale_vox))

    def eval_scale(scale: float) -> Tuple[np.ndarray, float, float]:
        m = _generate_one_mask(
            target_volume_mm3=target_volume_mm3,
            linear_scale_vox=scale,
            case_seed=case_seed,
            canal_axis=canal_axis,
            rotation_zyx_deg=rotation_zyx_deg,
        )
        v = _volume_mm3(m, voxel_volume_mm3)
        frac = abs(v - target_volume_mm3) / max(target_volume_mm3, 1e-8)
        return m, v, frac

    low_mask, low_vol, low_err = eval_scale(low)
    high_mask, high_vol, high_err = eval_scale(high)

    best_mask, best_scale, best_vol, best_err = low_mask, low, low_vol, low_err
    if high_err < best_err:
        best_mask, best_scale, best_vol, best_err = high_mask, high, high_vol, high_err

    # Downward expansion for tiny targets (critical for prior tail failures).
    for _ in range(10):
        if low_vol <= target_volume_mm3 or low <= min_scale_vox + 1e-9:
            break
        cand = max(min_scale_vox, low / 2.0)
        if abs(cand - low) < 1e-9:
            break
        low = cand
        low_mask, low_vol, low_err = eval_scale(low)
        if low_err < best_err:
            best_mask, best_scale, best_vol, best_err = low_mask, low, low_vol, low_err

    # Upward expansion for large targets.
    for _ in range(8):
        if high_vol >= target_volume_mm3 or high >= max_scale_vox - 1e-9:
            break
        cand = min(max_scale_vox, high * 1.6)
        if abs(cand - high) < 1e-9:
            break
        high = cand
        high_mask, high_vol, high_err = eval_scale(high)
        if high_err < best_err:
            best_mask, best_scale, best_vol, best_err = high_mask, high, high_vol, high_err

    n_iters = 0
    for i in range(max_iters):
        n_iters = i + 1
        mid = 0.5 * (low + high)
        mid_mask, mid_vol, mid_err = eval_scale(mid)

        if mid_err < best_err:
            best_mask, best_scale, best_vol, best_err = mid_mask, mid, mid_vol, mid_err

        if mid_err <= tolerance_frac:
            return mid_mask, mid, mid_vol, n_iters

        # If not bracketed, still move bounds by directionality.
        if mid_vol < target_volume_mm3:
            low = mid
        else:
            high = mid

        if abs(high - low) < 1e-5:
            break

    return best_mask, best_scale, best_vol, n_iters


def _save_mask(mask: np.ndarray, spacing_mm: Tuple[float, float, float], out_path: Path) -> None:
    affine = np.diag([spacing_mm[0], spacing_mm[1], spacing_mm[2], 1.0]).astype(np.float32)
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nii.header.set_zooms((spacing_mm[0], spacing_mm[1], spacing_mm[2]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(out_path))


def _safe_case_id(case_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in case_id).strip("_") or "case"


def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic lollipop cohort from target volumes CSV")
    p.add_argument("--targets_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--manifest_csv", type=str, default=None, help="Default: <out_dir>/synthetic_manifest.csv")
    p.add_argument("--case_id_col", type=str, default="case_id")
    p.add_argument("--target_volume_col", type=str, default="target_volume_mm3")
    p.add_argument("--seed", type=int, default=20260426)
    p.add_argument("--canal_axis", type=str, default="c", choices=["a", "b", "c"])
    p.add_argument("--voxel_spacing_mm", type=str, default="1.0,1.0,1.0")
    p.add_argument("--volume_tolerance_frac", type=float, default=0.03)
    p.add_argument("--max_calibration_iters", type=int, default=14)
    p.add_argument("--min_scale_vox", type=float, default=0.08)
    p.add_argument("--max_scale_vox", type=float, default=64.0)
    args = p.parse_args()

    targets_csv = Path(args.targets_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = Path(args.manifest_csv).expanduser().resolve() if args.manifest_csv else out_dir / "synthetic_manifest.csv"

    spacing_mm = _parse_spacing(args.voxel_spacing_mm)
    voxel_volume_mm3 = float(spacing_mm[0] * spacing_mm[1] * spacing_mm[2])
    rows = _read_targets(targets_csv, case_col=args.case_id_col, vol_col=args.target_volume_col)

    manifest_rows: List[Dict[str, object]] = []

    for row in rows:
        case_id = str(row[args.case_id_col]).strip()
        target_volume_mm3 = float(row[args.target_volume_col])
        if target_volume_mm3 <= 0:
            raise ValueError(f"Target volume must be > 0 for case {case_id}: {target_volume_mm3}")

        case_seed = _stable_seed(args.seed, case_id)
        rng_case = np.random.default_rng(case_seed)
        rotation_zyx_deg = [
            int(rng_case.integers(0, 180)),
            int(rng_case.integers(0, 180)),
            int(rng_case.integers(0, 180)),
        ]

        mask, final_scale, realized_volume_mm3, n_iters = _calibrate_scale(
            target_volume_mm3=target_volume_mm3,
            case_seed=case_seed,
            canal_axis=args.canal_axis,
            rotation_zyx_deg=rotation_zyx_deg,
            voxel_volume_mm3=voxel_volume_mm3,
            tolerance_frac=float(args.volume_tolerance_frac),
            max_iters=int(args.max_calibration_iters),
            min_scale_vox=float(args.min_scale_vox),
            max_scale_vox=float(args.max_scale_vox),
        )

        case_name = _safe_case_id(case_id)
        seg_path = out_dir / f"{case_name}_synthetic_lollipop_mask.nii.gz"
        _save_mask(mask=mask, spacing_mm=spacing_mm, out_path=seg_path)

        vol_err = float(realized_volume_mm3 - target_volume_mm3)
        vol_err_frac = float(abs(vol_err) / max(target_volume_mm3, 1e-8))

        manifest_rows.append(
            {
                "case_id": case_id,
                "seg_path": str(seg_path),
                "target_volume_mm3": float(target_volume_mm3),
                "realized_volume_mm3": float(realized_volume_mm3),
                "volume_error_mm3": vol_err,
                "volume_error_fraction": vol_err_frac,
                "n_calibration_iters": int(n_iters),
                "final_linear_scale_vox": float(final_scale),
                "rotation_z_deg": int(rotation_zyx_deg[0]),
                "rotation_y_deg": int(rotation_zyx_deg[1]),
                "rotation_x_deg": int(rotation_zyx_deg[2]),
                "seed": int(case_seed),
            }
        )

    fieldnames = [
        "case_id",
        "seg_path",
        "target_volume_mm3",
        "realized_volume_mm3",
        "volume_error_mm3",
        "volume_error_fraction",
        "n_calibration_iters",
        "final_linear_scale_vox",
        "rotation_z_deg",
        "rotation_y_deg",
        "rotation_x_deg",
        "seed",
    ]
    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Synthetic lollipop cohort generation complete")
    print(f"  cases: {len(manifest_rows)}")
    print(f"  out_dir: {out_dir}")
    print(f"  manifest: {manifest_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
