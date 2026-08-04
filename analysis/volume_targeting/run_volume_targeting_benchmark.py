#!/usr/bin/env python3
"""Local volume-targeting benchmark for the standalone lollipop generator.

This script intentionally writes only inside analysis/volume_targeting. It imports
the current repository generator code without modifying it.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(THIS_DIR / ".mplconfig"))

from scripts.generate_synthetic_lollipop_cohort import (  # noqa: E402
    _calibrate_scale,
    _generate_one_mask,
    _stable_seed,
    _volume_mm3,
)
from shared.reporting import write_csv_rows  # noqa: E402


DEFAULT_TARGETS = [
    25.0,
    50.0,
    75.0,
    100.0,
    150.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
    16000.0,
]
DEFAULT_SEEDS = [20260426, 20260427, 20260428]
DEFAULT_SPACINGS = [(1.0, 1.0, 1.0), (0.6, 0.6, 1.2), (0.8, 0.8, 0.8)]


def _case_seed(base_seed: int, target_volume_mm3: float, spacing: tuple[float, float, float]) -> int:
    key = f"vol_{target_volume_mm3:g}_spacing_{spacing[0]:g}_{spacing[1]:g}_{spacing[2]:g}"
    return _stable_seed(base_seed, key)


def _rotation_for_seed(case_seed: int) -> list[int]:
    rng_case = np.random.default_rng(case_seed)
    return [
        int(rng_case.integers(0, 180)),
        int(rng_case.integers(0, 180)),
        int(rng_case.integers(0, 180)),
    ]


def _mask_margin_vox(mask: np.ndarray) -> int | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = np.array(mask.shape) - 1 - coords.max(axis=0)
    return int(min(np.min(mins), np.min(maxs)))


def _component_count(mask: np.ndarray) -> int:
    try:
        from scipy import ndimage
    except Exception:
        return -1
    _, n_components = ndimage.label(mask > 0)
    return int(n_components)


def _probe_scale_curve(
    target_volume_mm3: float,
    case_seed: int,
    rotation_zyx_deg: list[int],
    voxel_volume_mm3: float,
    min_scale_vox: float,
    max_scale_vox: float,
) -> tuple[bool, bool, str]:
    eq_radius_mm = ((3.0 * max(target_volume_mm3, 1e-6)) / (4.0 * math.pi)) ** (1.0 / 3.0)
    center = float(np.clip(eq_radius_mm * 0.85, min_scale_vox, max_scale_vox))
    low = max(min_scale_vox, center * 0.35)
    high = min(max_scale_vox, center * 2.0)
    scales = np.geomspace(max(low, 1e-5), max(high, low + 1e-5), num=7)
    volumes = []
    for scale in scales:
        mask = _generate_one_mask(
            target_volume_mm3=target_volume_mm3,
            linear_scale_vox=float(scale),
            case_seed=case_seed,
            canal_axis="c",
            rotation_zyx_deg=rotation_zyx_deg,
        )
        volumes.append(_volume_mm3(mask, voxel_volume_mm3))
    diffs = np.diff(volumes)
    non_monotonic = bool(np.any(diffs < 0))
    plateau = bool(np.any(np.abs(diffs) <= max(1.0, 0.002 * target_volume_mm3)))
    curve = ";".join(f"{s:.5g}:{v:.1f}" for s, v in zip(scales, volumes))
    return non_monotonic, plateau, curve


def _run_case(
    target_volume_mm3: float,
    base_seed: int,
    spacing: tuple[float, float, float],
    tolerance_frac: float,
    max_iters: int,
    min_scale_vox: float,
    max_scale_vox: float,
) -> dict[str, object]:
    case_seed = _case_seed(base_seed, target_volume_mm3, spacing)
    rotation = _rotation_for_seed(case_seed)
    voxel_volume_mm3 = float(np.prod(spacing))
    started = time.perf_counter()
    warning_messages: list[str] = []
    status = "OK"
    mask = None
    final_scale = math.nan
    achieved = math.nan
    iterations = 0
    error_message = ""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            mask, final_scale, achieved, iterations = _calibrate_scale(
                target_volume_mm3=target_volume_mm3,
                case_seed=case_seed,
                canal_axis="c",
                rotation_zyx_deg=rotation,
                voxel_volume_mm3=voxel_volume_mm3,
                tolerance_frac=tolerance_frac,
                max_iters=max_iters,
                min_scale_vox=min_scale_vox,
                max_scale_vox=max_scale_vox,
            )
        except Exception as exc:
            status = "ERROR"
            error_message = f"{type(exc).__name__}: {exc}"
        warning_messages = [str(w.message) for w in caught]

    elapsed = time.perf_counter() - started
    signed_error = achieved - target_volume_mm3 if math.isfinite(achieved) else math.nan
    abs_error = abs(signed_error) if math.isfinite(signed_error) else math.nan
    ravd = abs_error / max(target_volume_mm3, 1e-8) if math.isfinite(abs_error) else math.nan
    converged = bool(math.isfinite(ravd) and ravd <= tolerance_frac)
    direction = "hit"
    if math.isfinite(signed_error):
        if signed_error > 0:
            direction = "overshoot"
        elif signed_error < 0:
            direction = "undershoot"

    margin = _mask_margin_vox(mask) if mask is not None else None
    clipped = bool(margin == 0) if margin is not None else False
    components = _component_count(mask) if mask is not None else -1
    non_monotonic, plateau, curve = (False, False, "")
    if status == "OK":
        non_monotonic, plateau, curve = _probe_scale_curve(
            target_volume_mm3=target_volume_mm3,
            case_seed=case_seed,
            rotation_zyx_deg=rotation,
            voxel_volume_mm3=voxel_volume_mm3,
            min_scale_vox=min_scale_vox,
            max_scale_vox=max_scale_vox,
        )

    return {
        "target_volume_mm3": target_volume_mm3,
        "achieved_volume_mm3": achieved,
        "signed_error_mm3": signed_error,
        "absolute_error_mm3": abs_error,
        "relative_absolute_volume_difference": ravd,
        "converged_within_tolerance": converged,
        "tolerance_fraction": tolerance_frac,
        "n_calibration_iters": iterations,
        "final_linear_scale_vox": final_scale,
        "base_seed": base_seed,
        "case_seed": case_seed,
        "spacing_x_mm": spacing[0],
        "spacing_y_mm": spacing[1],
        "spacing_z_mm": spacing[2],
        "voxel_volume_mm3": voxel_volume_mm3,
        "rotation_z_deg": rotation[0],
        "rotation_y_deg": rotation[1],
        "rotation_x_deg": rotation[2],
        "mask_voxels": int(mask.sum()) if mask is not None else "",
        "mask_shape": "x".join(str(v) for v in mask.shape) if mask is not None else "",
        "component_count": components,
        "bbox_margin_vox": "" if margin is None else margin,
        "clipped_to_generation_grid": clipped,
        "direction": direction,
        "local_scale_curve_non_monotonic": non_monotonic,
        "local_scale_curve_plateau": plateau,
        "local_scale_curve": curve,
        "status": status,
        "error_message": error_message,
        "warnings": " | ".join(sorted(set(warning_messages))),
        "elapsed_seconds": elapsed,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv_rows(path, rows, fieldnames)


def _write_plots(out_dir: Path, rows: list[dict[str, object]]) -> list[Path]:
    plot_paths: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plotting unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return plot_paths

    ok_rows = [r for r in rows if r["status"] == "OK"]
    if not ok_rows:
        return plot_paths

    fig, ax = plt.subplots(figsize=(7, 6))
    for spacing in sorted({(r["spacing_x_mm"], r["spacing_y_mm"], r["spacing_z_mm"]) for r in ok_rows}):
        xs = [float(r["target_volume_mm3"]) for r in ok_rows if (r["spacing_x_mm"], r["spacing_y_mm"], r["spacing_z_mm"]) == spacing]
        ys = [float(r["achieved_volume_mm3"]) for r in ok_rows if (r["spacing_x_mm"], r["spacing_y_mm"], r["spacing_z_mm"]) == spacing]
        ax.scatter(xs, ys, s=24, alpha=0.75, label=f"{spacing[0]:g},{spacing[1]:g},{spacing[2]:g} mm")
    lim_low = min(min(float(r["target_volume_mm3"]), float(r["achieved_volume_mm3"])) for r in ok_rows) * 0.75
    lim_high = max(max(float(r["target_volume_mm3"]), float(r["achieved_volume_mm3"])) for r in ok_rows) * 1.25
    ax.plot([lim_low, lim_high], [lim_low, lim_high], color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Requested volume (mm3)")
    ax.set_ylabel("Achieved volume (mm3)")
    ax.set_title("Requested vs achieved standalone lollipop volumes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "requested_vs_achieved.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    for spacing in sorted({(r["spacing_x_mm"], r["spacing_y_mm"], r["spacing_z_mm"]) for r in ok_rows}):
        grouped: dict[float, list[float]] = {}
        for r in ok_rows:
            if (r["spacing_x_mm"], r["spacing_y_mm"], r["spacing_z_mm"]) == spacing:
                grouped.setdefault(float(r["target_volume_mm3"]), []).append(
                    float(r["relative_absolute_volume_difference"])
                )
        xs = sorted(grouped)
        ys = [float(np.median(grouped[x])) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{spacing[0]:g},{spacing[1]:g},{spacing[2]:g} mm")
    ax.axhline(0.03, color="black", linestyle="--", linewidth=1, label="3% tolerance")
    ax.set_xscale("log")
    ax.set_xlabel("Requested volume (mm3)")
    ax.set_ylabel("Median RAVD")
    ax.set_title("Volume targeting error by requested volume")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "ravd_by_target_volume.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plot_paths.append(path)

    return plot_paths


def _parse_targets(raw: str | None) -> list[float]:
    if not raw:
        return DEFAULT_TARGETS
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def _parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return DEFAULT_SEEDS
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def _parse_spacings(raw: str | None) -> list[tuple[float, float, float]]:
    if not raw:
        return DEFAULT_SPACINGS
    spacings = []
    for item in raw.split(";"):
        parts = [float(v.strip()) for v in item.split(",") if v.strip()]
        if len(parts) != 3:
            raise ValueError(f"Spacing must have three comma-separated values: {item!r}")
        spacings.append((parts[0], parts[1], parts[2]))
    return spacings


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_csv", default=str(THIS_DIR / "volume_targeting_benchmark.csv"))
    parser.add_argument("--targets", default=None, help="Comma-separated target volumes in mm3.")
    parser.add_argument("--seeds", default=None, help="Comma-separated base seeds.")
    parser.add_argument("--spacings", default=None, help="Semicolon-separated sx,sy,sz triplets.")
    parser.add_argument("--tolerance_frac", type=float, default=0.03)
    parser.add_argument("--max_iters", type=int, default=14)
    parser.add_argument("--min_scale_vox", type=float, default=0.08)
    parser.add_argument("--max_scale_vox", type=float, default=64.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    targets = _parse_targets(args.targets)
    seeds = _parse_seeds(args.seeds)
    spacings = _parse_spacings(args.spacings)

    rows = []
    total = len(targets) * len(seeds) * len(spacings)
    print(f"Running {total} local volume-targeting cases")
    for spacing in spacings:
        for target in targets:
            for seed in seeds:
                row = _run_case(
                    target_volume_mm3=target,
                    base_seed=seed,
                    spacing=spacing,
                    tolerance_frac=float(args.tolerance_frac),
                    max_iters=int(args.max_iters),
                    min_scale_vox=float(args.min_scale_vox),
                    max_scale_vox=float(args.max_scale_vox),
                )
                rows.append(row)
                print(
                    f"  target={target:8.1f} spacing={spacing} seed={seed} "
                    f"achieved={row['achieved_volume_mm3']:8.1f} "
                    f"ravd={row['relative_absolute_volume_difference']:.4f} "
                    f"status={row['status']}"
                )

    out_csv = Path(args.out_csv).resolve()
    _write_csv(out_csv, rows)
    plot_paths = _write_plots(out_csv.parent, rows)
    print(f"Wrote {out_csv}")
    for path in plot_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
