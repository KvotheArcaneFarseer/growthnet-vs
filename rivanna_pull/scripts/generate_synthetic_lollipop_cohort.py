from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def _save_mask_nifti(mask_3d: np.ndarray, out_path: Path, voxel_size_mm: float) -> None:
    mask_3d = np.squeeze(np.asarray(mask_3d))
    if mask_3d.ndim != 3:
        raise ValueError(f"Expected 3D mask after squeeze, got shape {mask_3d.shape}")
    mask_3d = (mask_3d > 0).astype(np.uint8)

    affine = np.diag([voxel_size_mm, voxel_size_mm, voxel_size_mm, 1.0]).astype(float)
    img = nib.Nifti1Image(mask_3d, affine)
    nib.save(img, str(out_path))


def _load_targets(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(csv_path)))
    out = []
    for row in rows:
        out.append({
            "case_id": str(row["case_id"]),
            "target_volume_mm3": float(row["target_volume_mm3"]),
        })
    return out


def _target_shape_controls(target_volume_mm3: float) -> Tuple[float, float]:
    """Return (tinyness, maturity) in [0,1] based on target volume.

    tinyness=1 => very small tumor (reduce dramatic bulb/stem exaggeration)
    maturity=1 => larger target (allow fuller lollipop expression)
    """
    maturity = float(np.clip((target_volume_mm3 - 80.0) / 700.0, 0.0, 1.0))
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
    j1 = float(rng.uniform(-0.05, 0.05))
    j2 = float(rng.uniform(-0.05, 0.05))
    j3 = float(rng.uniform(-0.06, 0.06))
    j4 = float(rng.uniform(-0.06, 0.06))

    # Softer, less elongated base geometry.
    base_radius_max = linear_scale_vox * (0.98 + j1)
    apex_radius_max = base_radius_max * (0.97 + 0.02 * maturity + j2)

    # Keep canal short-to-moderate to reduce major/minor disparity.
    canal_length_max = linear_scale_vox * (0.72 + 0.40 * maturity + j3)

    # Bulb remains present but with controlled size progression.
    bulb_radius_max = linear_scale_vox * (0.92 + 0.28 * maturity + j4)

    # Avoid tiny-volume floor overshoot by allowing small initial values.
    canal_length_init = max(0.20, canal_length_max * (0.12 + 0.14 * maturity))
    base_radius_init = max(0.22, base_radius_max * (0.28 + 0.10 * maturity))
    apex_radius_init = max(0.20, apex_radius_max * (0.30 + 0.08 * maturity))

    # Tiny targets: keep bulb minimal to avoid hard minimum-size snap.
    bulb_radius_init = max(0.0, bulb_radius_max * (0.00 + 0.16 * maturity))

    # Single-mask benchmark: keep growth multipliers conservative.
    canal_growth_scale = 0.60 + 0.22 * maturity
    bulb_growth_scale = 0.62 + 0.24 * maturity

    # Guardrails.
    base_radius_max = max(0.28, base_radius_max)
    apex_radius_max = max(0.24, min(apex_radius_max, base_radius_max * 1.02))
    canal_length_max = max(0.25, canal_length_max)
    bulb_radius_max = max(0.22, bulb_radius_max)

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
    span = int(math.ceil(34.0 + 8.0 * scale_vox))
    size = max(56, min(224, span))
    rad_max = max(6, min(size // 3, int(math.ceil(3.2 * scale_vox + 5.0))))
    return size, rad_max


def _generate_one_mask(
    target_volume_mm3: float,
    linear_scale_vox: float,
    case_seed: int,
    canal_axis: str,
    rotation_zyx_deg: Tuple[int, int, int],
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

    label = labels[0]
    if label.ndim == 4:
        label = np.squeeze(label)
    return (label > 0).astype(np.uint8)


def _volume_mm3(mask: np.ndarray, voxel_volume_mm3: float) -> float:
    return float(int(mask.sum()) * voxel_volume_mm3)


def _calibrate_scale(
    target_volume_mm3: float,
    case_seed: int,
    canal_axis: str,
    rotation_zyx_deg: Tuple[int, int, int],
    voxel_volume_mm3: float,
    tolerance_frac: float,
    max_iters: int,
    min_scale_vox: float,
    max_scale_vox: float,
) -> Tuple[np.ndarray, float, float, int]:
    """Binary-search-like calibration over linear_scale_vox with bounded expansion."""

    # Sphere-equivalent heuristic for initial scale
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

    # Downward expansion for tiny targets
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

    # Upward expansion for large targets
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

        if mid_vol < target_volume_mm3:
            low = mid
        else:
            high = mid

        if abs(high - low) < 1e-5:
            break

    return best_mask, best_scale, best_vol, n_iters


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic lollipop cohort matched to real tumor target volumes."
    )
    parser.add_argument("--targets_csv", type=str, required=True, help="CSV with columns case_id,target_volume_mm3")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save synthetic masks")
    parser.add_argument("--out_manifest_csv", type=str, required=True, help="Output manifest CSV")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed")
    parser.add_argument("--voxel_size_mm", type=float, default=1.0, help="Assumed isotropic voxel size for synthetic masks")
    parser.add_argument("--volume_tol_frac", type=float, default=0.03, help="Relative volume tolerance for iterative calibration")
    parser.add_argument("--max_calibration_iters", type=int, default=14, help="Maximum number of calibration iterations per case")
    parser.add_argument("--min_scale_vox", type=float, default=0.08, help="Minimum allowed linear scale in voxel units")
    parser.add_argument("--max_scale_vox", type=float, default=64.0, help="Maximum allowed linear scale in voxel units")
    parser.add_argument("--canal_axis", type=str, default="c", choices=["a", "b", "c"])
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    targets_csv = Path(args.targets_csv)
    out_dir = Path(args.out_dir)
    out_manifest_csv = Path(args.out_manifest_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    targets = _load_targets(targets_csv)
    print(f"Loaded {len(targets)} target volumes from {targets_csv}")

    voxel_volume_mm3 = float(args.voxel_size_mm) ** 3
    out_rows: List[Dict[str, object]] = []

    for target in targets:
        case_id = target["case_id"]
        target_volume_mm3 = float(target["target_volume_mm3"])
        case_seed = _stable_seed(int(args.seed), case_id)

        rng_case = np.random.default_rng(case_seed)
        rotation_zyx_deg = (
            int(rng_case.integers(0, 180)),
            int(rng_case.integers(0, 180)),
            int(rng_case.integers(0, 180)),
        )

        best_mask, final_scale, realized_volume_mm3, n_iters = _calibrate_scale(
            target_volume_mm3=target_volume_mm3,
            case_seed=case_seed,
            canal_axis=args.canal_axis,
            rotation_zyx_deg=rotation_zyx_deg,
            voxel_volume_mm3=voxel_volume_mm3,
            tolerance_frac=float(args.volume_tol_frac),
            max_iters=int(args.max_calibration_iters),
            min_scale_vox=float(args.min_scale_vox),
            max_scale_vox=float(args.max_scale_vox),
        )

        out_path = out_dir / f"{case_id}_synthetic_lollipop.nii.gz"
        _save_mask_nifti(best_mask, out_path, float(args.voxel_size_mm))

        vol_err = float(realized_volume_mm3 - target_volume_mm3)
        vol_err_frac = float(abs(vol_err) / max(target_volume_mm3, 1e-8))

        out_rows.append({
            "case_id": case_id,
            "seg_path": str(out_path.resolve()),
            "target_volume_mm3": target_volume_mm3,
            "realized_volume_mm3": float(realized_volume_mm3),
            "volume_error_mm3": vol_err,
            "volume_error_fraction": vol_err_frac,
            "n_calibration_iters": int(n_iters),
            "final_linear_scale_vox": float(final_scale),
            "rotation_z_deg": int(rotation_zyx_deg[0]),
            "rotation_y_deg": int(rotation_zyx_deg[1]),
            "rotation_x_deg": int(rotation_zyx_deg[2]),
            "seed": int(case_seed),
        })

    with open(out_manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    abs_errs = [abs(float(r["volume_error_mm3"])) for r in out_rows]
    ratios = [
        float(r["realized_volume_mm3"]) / float(r["target_volume_mm3"])
        for r in out_rows
        if float(r["target_volume_mm3"]) > 0
    ]

    print(f"Wrote synthetic masks to: {out_dir}")
    print(f"Wrote manifest to: {out_manifest_csv}")
    print(f"N cases: {len(out_rows)}")
    print(f"Median abs volume error (mm^3): {float(np.median(abs_errs)):.4f}")
    print(f"Median realized/target ratio: {float(np.median(ratios)):.4f}")


if __name__ == "__main__":
    main()
