"""
Anatomically guided synthetic tumor embedding — lollipop geometry.

Loads a real MRI + its binary segmentation, computes the segmentation centroid
and principal long-axis via skimage regionprops, generates a synthetic lollipop
tumor time series, then embeds the series into the MRI so that:
  - the smallest-timepoint tumor centroid coincides with the segmentation centroid
  - the tumor growth axis (lollipop canal) aligns with the segmentation long axis

Outputs (all in embedding_outputs/):
  embedded_tumor_volume.nii.gz        — MRI with smallest-timepoint tumor inserted
  embedded_tumor_mask.nii.gz          — binary mask of the smallest-timepoint tumor
  embedded_tumor_late_volume.nii.gz   — MRI with last-timepoint (most elongated) tumor
  embedded_tumor_late_mask.nii.gz     — binary mask of the last-timepoint tumor
  embedded_t{i:02d}_volume.nii.gz     — full time series (all timepoints)
  embedded_t{i:02d}_mask.nii.gz       — masks for all timepoints
  qc_embedding.png                    — 3×3 QC grid for smallest timepoint
  qc_embedding_late.png               — 3×3 QC grid for last (elongated) timepoint

Usage
-----
    python embed_tumor.py [--mri PATH] [--seg PATH] [--out_dir DIR]

Defaults point at the reference pair in ~/Downloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import matplotlib
matplotlib.use("Agg")  # headless — no Qt needed
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt, label as ndi_label
from scipy.spatial.transform import Rotation
from skimage.measure import regionprops

# ── repo root on sys.path ─────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from projects.vivit.src.data.synthetic import create_synthetic_time_3d


# ── helpers ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PrincipalAxesResult:
    """Segmentation principal-axis measurements in both voxel and physical spaces."""
    centroid_vox: np.ndarray
    long_axis_vox: np.ndarray
    long_axis_phys: np.ndarray
    eigenvalues_vox: np.ndarray
    eigenvalues_phys: np.ndarray


@dataclass(frozen=True)
class OrientationInputs:
    """Inputs shared by all axis-sign selection strategies."""
    seg_mask: np.ndarray
    unresolved_axis_vox: np.ndarray
    default_axis_syn: np.ndarray
    syn_centroid: np.ndarray
    seg_centroid: np.ndarray
    out_shape: tuple[int, ...]
    target_spacing: np.ndarray
    syn_masks_by_name: dict[str, np.ndarray]


@dataclass(frozen=True)
class OrientationCandidate:
    """One signed-axis candidate scored by an orientation strategy."""
    axis_vox: np.ndarray
    score: float
    label: str
    extras: dict[str, float | str]


@dataclass(frozen=True)
class OrientationResult:
    """Result of a sign-selection strategy for a single case."""
    axis_vox: np.ndarray
    method: str
    confidence: float
    low_confidence: bool
    candidates: list[OrientationCandidate]
    debug: dict[str, float | str]


@dataclass(frozen=True)
class TimepointMetrics:
    """Placement metrics for one synthetic timepoint."""
    timepoint_index: int
    day: int
    source_voxels: int
    placed_voxels: int
    source_volume_mm3: float
    placed_volume_mm3: float
    retained_fraction: float
    centroid_offset_mm: float
    axis_error_deg: float | None
    clipped: bool


@dataclass(frozen=True)
class ValidationFinding:
    """One validation issue classified by severity."""
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationThresholds:
    """Default thresholds for warnings vs hard failures."""
    centroid_offset_warn_mm: float = 1.5
    centroid_offset_fail_mm: float = 3.0
    axis_error_deg_warn: float = 30.0
    axis_error_deg_fail: float = 60.0
    retained_fraction_warn: float = 0.95
    retained_fraction_fail: float = 0.80
    # retained_fraction > this indicates a resampling artifact, not true growth
    retained_fraction_high_warn: float = 1.05
    worst_clipping_warn: float = 0.90
    worst_clipping_fail: float = 0.75
    # orientation_confidence_warn: normalized gap threshold (gap / (|best|+|worst|))
    orientation_confidence_warn: float = 0.08
    # orientation_score_margin_warn: raw Dice score difference threshold
    orientation_score_margin_warn: float = 0.05
    placed_to_seg_ratio_warn_low: float = 0.20
    placed_to_seg_ratio_warn_high: float = 1.80
    placed_to_seg_ratio_fail_low: float = 0.10
    placed_to_seg_ratio_fail_high: float = 2.50
    monotone_drop_warn_fraction: float = 0.02


@dataclass(frozen=True)
class EmbeddingCaseMetrics:
    """Structured metrics and validation output for one embedding case."""
    case_id: str
    seed: int
    source_mri: str
    source_seg: str
    orientation_method: str
    orientation_confidence: float
    orientation_score_margin: float
    orientation_normalized_gap: float
    orientation_low_confidence: bool
    selected_axis_vox: list[float]
    selected_axis_phys: list[float]
    source_voxel_count: int
    placed_voxel_count: int
    max_placed_voxel_count: int
    source_volume_mm3: float
    placed_volume_mm3: float
    max_placed_volume_mm3: float
    retained_fraction: float
    centroid_offset_vox: float
    centroid_offset_mm: float
    primary_axis_error_deg: float | None
    cpa_radius_anatomy_mm: float
    cpa_radius_effective_mm: float
    cpa_radius_override_active: bool
    volume_optimization_variable: str | None
    target_volume_mm3: float | None
    volume_target_timepoint: str | None
    volume_target_timepoint_index: int | None
    volume_target_timepoint_day: int | None
    target_timepoint_actual_volume_mm3: float | None
    target_timepoint_voxels: int | None
    actual_volume_mm3: float | None
    ravd: float | None
    volume_converged: bool | None
    volume_iterations: int | None
    volume_scale_final: float | None
    target_seg_voxel_count: int
    placed_to_seg_ratio: float
    monotone_growth: bool
    worst_clipping_fraction: float
    strategy_agreement: bool
    warnings: list[str]
    hard_failures: list[str]
    findings: list[ValidationFinding]
    strategy_results: list[dict]
    timepoint_metrics: list[TimepointMetrics]


class OrientationStrategy(Protocol):
    """Protocol for modular orientation/sign-selection strategies."""
    name: str

    def select_axis(self, inputs: OrientationInputs) -> OrientationResult:
        ...

def largest_component(mask: np.ndarray) -> tuple[np.ndarray, object]:
    """Return the largest connected component mask and its regionprops entry."""
    labeled, num_labels = ndi_label(mask > 0)
    if num_labels == 0:
        raise ValueError("Mask is empty; no connected component found.")
    props = regionprops(labeled)
    prop = max(props, key=lambda p: p.area)
    component_mask = labeled == prop.label
    return component_mask, prop


def _regionprops_with_spacing(labeled: np.ndarray, spacing: np.ndarray) -> tuple[list[object], bool]:
    """Use spacing-aware regionprops when available, otherwise fall back to voxel units."""
    try:
        return regionprops(labeled, spacing=tuple(float(v) for v in spacing)), True
    except TypeError:
        return regionprops(labeled), False


def _stabilize_axis_sign(axis: np.ndarray) -> np.ndarray:
    """Force a deterministic eigenvector sign before later anatomy-aware selection."""
    dominant_idx = int(np.argmax(np.abs(axis)))
    if axis[dominant_idx] < 0:
        return -axis
    return axis


def _spacing_matrix(spacing: np.ndarray) -> np.ndarray:
    """Diagonal voxel-index -> physical-mm scaling for orthogonal MRI grids."""
    return np.diag(np.asarray(spacing, dtype=np.float64))


def _axis_vox_to_phys(axis_vox: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Convert a voxel-index direction vector into a physical/mm direction."""
    axis_phys = _spacing_matrix(spacing) @ np.asarray(axis_vox, dtype=np.float64)
    norm = np.linalg.norm(axis_phys)
    if norm == 0.0:
        raise ValueError("Cannot convert a zero-length voxel axis to physical space.")
    return axis_phys / norm


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Unsigned angle in degrees between two direction vectors."""
    unit_a = np.asarray(vec_a, dtype=np.float64)
    unit_b = np.asarray(vec_b, dtype=np.float64)
    unit_a /= np.linalg.norm(unit_a)
    unit_b /= np.linalg.norm(unit_b)
    cos_theta = float(np.clip(np.abs(unit_a @ unit_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def stable_seed_from_case(case_id: str) -> int:
    """Stable 32-bit seed derived from a case identifier."""
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice overlap for two binary masks."""
    a = mask_a > 0
    b = mask_b > 0
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    inter = int(np.logical_and(a, b).sum())
    return 2.0 * inter / denom


def _resolve_signed_axis_in_physical_space(axis_vox: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    """Convert a signed voxel-space axis into its signed physical-space counterpart."""
    axis_phys = _axis_vox_to_phys(axis_vox, spacing)
    if np.dot(axis_phys, _axis_vox_to_phys(axis_vox, spacing)) < 0.0:
        return -axis_phys
    return axis_phys


def _place_mask_for_candidate(inputs: OrientationInputs, candidate_axis_vox: np.ndarray, syn_mask: np.ndarray) -> np.ndarray:
    """Place a synthetic mask into MRI space for one signed-axis candidate."""
    candidate_axis_phys = _resolve_signed_axis_in_physical_space(candidate_axis_vox, inputs.target_spacing)
    try:
        rot, _ = Rotation.align_vectors(
            candidate_axis_phys[np.newaxis],
            inputs.default_axis_syn[np.newaxis],
        )
        rot_matrix = rot.as_matrix()
    except Exception:
        rot_matrix = np.eye(3)
    return rotate_and_translate(
        syn_mask.astype(np.float32),
        rot_matrix,
        inputs.syn_centroid,
        inputs.seg_centroid,
        out_shape=inputs.out_shape,
        dst_spacing=inputs.target_spacing,
        order=0,
    ) > 0.5


def _orientation_confidence(candidates: list[OrientationCandidate]) -> tuple[float, bool, dict[str, float | str]]:
    """Derive a simple ambiguity signal from candidate scores."""
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = ordered[0].score
    worst = ordered[-1].score
    margin = float(best - worst)
    denom = abs(best) + abs(worst) + 1e-8
    normalized_gap = float(margin / denom)
    low_confidence = bool(margin < 0.05 or normalized_gap < 0.05)
    return normalized_gap, low_confidence, {
        "best_score": float(best),
        "worst_score": float(worst),
        "score_margin": margin,
        "normalized_gap": normalized_gap,
    }


class LateDiceOrientationStrategy:
    """
    Default sign-selection strategy.

    Verified from projects/vivit/src/data/synthetic.py geometry, not just comments:
      - x_rel = -canal_coord
      - canal/fundus occupy x_rel >= 0
      - the CPA bulb is centred at x_rel = -cpa_radius*0.55

    Therefore, for the default synthetic setup used here (canal_axis="c" with no
    pre-rotation), negative local-z is the IAC/fundus stem side and positive
    local-z is the extracanalicular CPA bulb side.
    """
    name = "late_dice"

    def select_axis(self, inputs: OrientationInputs) -> OrientationResult:
        syn_mask = inputs.syn_masks_by_name["late"]
        candidates: list[OrientationCandidate] = []
        for sign in (1.0, -1.0):
            candidate_axis = inputs.unresolved_axis_vox * sign
            placed_mask = _place_mask_for_candidate(inputs, candidate_axis, syn_mask)
            score = _dice_score(placed_mask, inputs.seg_mask)
            candidates.append(
                OrientationCandidate(
                    axis_vox=candidate_axis,
                    score=float(score),
                    label=f"sign_{'pos' if sign > 0 else 'neg'}",
                    extras={"reference_mask": "late"},
                )
            )
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        confidence, low_confidence, debug = _orientation_confidence(candidates)
        best = candidates[0]
        if len(candidates) == 2 and np.isclose(candidates[0].score, candidates[1].score, atol=1e-6):
            best = OrientationCandidate(
                axis_vox=_stabilize_axis_sign(inputs.unresolved_axis_vox),
                score=best.score,
                label="tie_break_stable_sign",
                extras={"reference_mask": "late"},
            )
            debug["tie_break"] = "stable_sign"
        return OrientationResult(
            axis_vox=best.axis_vox,
            method=self.name,
            confidence=confidence,
            low_confidence=low_confidence,
            candidates=candidates,
            debug=debug,
        )


class MidGrowthDiceOrientationStrategy:
    """Alternative Dice strategy using a mid-growth synthetic mask for comparison."""
    name = "mid_growth_dice"

    def select_axis(self, inputs: OrientationInputs) -> OrientationResult:
        syn_mask = inputs.syn_masks_by_name["mid"]
        candidates: list[OrientationCandidate] = []
        for sign in (1.0, -1.0):
            candidate_axis = inputs.unresolved_axis_vox * sign
            placed_mask = _place_mask_for_candidate(inputs, candidate_axis, syn_mask)
            score = _dice_score(placed_mask, inputs.seg_mask)
            candidates.append(
                OrientationCandidate(
                    axis_vox=candidate_axis,
                    score=float(score),
                    label=f"sign_{'pos' if sign > 0 else 'neg'}",
                    extras={"reference_mask": "mid"},
                )
            )
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        confidence, low_confidence, debug = _orientation_confidence(candidates)
        return OrientationResult(
            axis_vox=candidates[0].axis_vox,
            method=self.name,
            confidence=confidence,
            low_confidence=low_confidence,
            candidates=candidates,
            debug=debug,
        )


def compare_orientation_strategies(
    inputs: OrientationInputs,
    strategies: list[OrientationStrategy],
) -> list[OrientationResult]:
    """Evaluate multiple orientation strategies for the same case."""
    return [strategy.select_axis(inputs) for strategy in strategies]


def _serialize_orientation_result(result: OrientationResult) -> dict:
    """Convert an orientation result into a JSON/CSV-friendly dict."""
    return {
        "method": result.method,
        "selected_axis_vox": [float(v) for v in result.axis_vox],
        "confidence": float(result.confidence),
        "low_confidence": bool(result.low_confidence),
        "debug": {
            key: float(val) if isinstance(val, (np.floating, float, int)) else val
            for key, val in result.debug.items()
        },
        "candidates": [
            {
                "axis_vox": [float(v) for v in candidate.axis_vox],
                "score": float(candidate.score),
                "label": candidate.label,
                "extras": {
                    key: float(val) if isinstance(val, (np.floating, float, int)) else val
                    for key, val in candidate.extras.items()
                },
            }
            for candidate in result.candidates
        ],
    }


def _strategies_agree(results: list[OrientationResult]) -> bool:
    """Whether all evaluated strategies chose the same signed axis."""
    if not results:
        return True
    ref = results[0].axis_vox / np.linalg.norm(results[0].axis_vox)
    return all(
        np.allclose(result.axis_vox / np.linalg.norm(result.axis_vox), ref, atol=1e-6)
        for result in results[1:]
    )


def _is_monotone_non_decreasing(values: list[int], tolerance_fraction: float) -> bool:
    """Allow tiny drops while flagging meaningful non-monotone growth."""
    for prev, curr in zip(values, values[1:]):
        if prev > 0 and curr < prev * (1.0 - tolerance_fraction):
            return False
    return True


def validate_embedding_case(
    case_id: str,
    seed: int,
    mri_path: Path,
    seg_path: Path,
    seg_voxel_count: int,
    seg_volume_mm3: float,
    selected_orientation: OrientationResult,
    comparison_results: list[OrientationResult],
    selected_axis_phys: np.ndarray,
    selected_axis_vox: np.ndarray,
    timepoint_metrics: list[TimepointMetrics],
    primary_timepoint_index: int,
    primary_source_voxels: int,
    primary_placed_voxels: int,
    primary_centroid_offset_vox: float,
    primary_centroid_offset_mm: float,
    cpa_radius_anatomy_mm: float,
    cpa_radius_effective_mm: float,
    cpa_radius_override_active: bool,
    volume_optimization_variable: str | None = None,
    target_volume_mm3: float | None = None,
    volume_target_timepoint: str | None = None,
    volume_target_timepoint_index: int | None = None,
    volume_target_timepoint_day: int | None = None,
    target_timepoint_actual_volume_mm3: float | None = None,
    target_timepoint_voxels: int | None = None,
    actual_volume_mm3: float | None = None,
    ravd: float | None = None,
    volume_converged: bool | None = None,
    volume_iterations: int | None = None,
    volume_scale_final: float | None = None,
    volume_target_below_min: bool = False,
    volume_target_above_max_or_clipped: bool = False,
    thresholds: ValidationThresholds | None = None,
) -> EmbeddingCaseMetrics:
    """Build per-case metrics and classify validation findings."""
    thresholds = thresholds or ValidationThresholds()
    findings: list[ValidationFinding] = []

    primary_tp = next((tp for tp in timepoint_metrics if tp.timepoint_index == primary_timepoint_index), None)
    source_volume_mm3 = float(primary_tp.source_volume_mm3) if primary_tp is not None else float(primary_source_voxels)
    placed_volume_mm3 = float(primary_tp.placed_volume_mm3) if primary_tp is not None else float(primary_placed_voxels)
    primary_axis_error_deg = None if primary_tp is None else primary_tp.axis_error_deg
    retained_fraction = float(placed_volume_mm3 / max(source_volume_mm3, 1e-8))
    max_placed_voxel_count = max((tp.placed_voxels for tp in timepoint_metrics), default=primary_placed_voxels)
    max_placed_volume_mm3 = max((tp.placed_volume_mm3 for tp in timepoint_metrics), default=float(primary_placed_voxels))
    placed_to_seg_ratio = float(max_placed_volume_mm3 / max(seg_volume_mm3, 1e-8))
    worst_clipping_fraction = min((tp.retained_fraction for tp in timepoint_metrics), default=1.0)
    monotone_growth = _is_monotone_non_decreasing(
        [tp.placed_voxels for tp in timepoint_metrics],
        thresholds.monotone_drop_warn_fraction,
    )
    strategy_agreement = _strategies_agree(comparison_results)

    def add_finding(severity: str, code: str, message: str) -> None:
        findings.append(ValidationFinding(severity=severity, code=code, message=message))

    if primary_placed_voxels <= 0:
        add_finding("hard_failure", "empty_output", "Primary placed mask is empty.")
    if primary_centroid_offset_mm >= thresholds.centroid_offset_fail_mm:
        add_finding("hard_failure", "centroid_offset_fail", f"Centroid offset {primary_centroid_offset_mm:.2f} mm exceeds failure threshold.")
    elif primary_centroid_offset_mm >= thresholds.centroid_offset_warn_mm:
        add_finding("warning", "centroid_offset_warn", f"Centroid offset {primary_centroid_offset_mm:.2f} mm exceeds warning threshold.")
    if primary_axis_error_deg is not None and primary_axis_error_deg >= thresholds.axis_error_deg_fail:
        add_finding("hard_failure", "axis_error_fail", f"Primary axis error {primary_axis_error_deg:.2f} deg exceeds failure threshold.")
    elif primary_axis_error_deg is not None and primary_axis_error_deg >= thresholds.axis_error_deg_warn:
        add_finding("warning", "axis_error_warn", f"Primary axis error {primary_axis_error_deg:.2f} deg exceeds warning threshold.")

    if retained_fraction <= thresholds.retained_fraction_fail:
        add_finding("hard_failure", "retained_fraction_fail", f"Retained fraction {retained_fraction:.3f} is too low.")
    elif retained_fraction <= thresholds.retained_fraction_warn:
        add_finding("warning", "retained_fraction_warn", f"Retained fraction {retained_fraction:.3f} is below the warning threshold.")

    if worst_clipping_fraction <= thresholds.worst_clipping_fail:
        add_finding("hard_failure", "worst_clipping_fail", f"Worst retained fraction {worst_clipping_fraction:.3f} indicates severe clipping.")
    elif worst_clipping_fraction <= thresholds.worst_clipping_warn:
        add_finding("warning", "worst_clipping_warn", f"Worst retained fraction {worst_clipping_fraction:.3f} indicates clipping.")

    if not monotone_growth:
        add_finding(
            "warning",
            "non_monotone_growth",
            "Placed mask voxel counts are not monotone non-decreasing across timepoints.",
        )

    if retained_fraction > thresholds.retained_fraction_high_warn:
        add_finding(
            "warning",
            "retained_fraction_high",
            f"Retained fraction {retained_fraction:.4f} exceeds 1.0 — likely a resampling artifact.",
        )

    _score_margin = float(selected_orientation.debug.get("score_margin", 0.0))
    _normalized_gap = float(selected_orientation.confidence)
    if _score_margin < thresholds.orientation_score_margin_warn:
        add_finding(
            "warning",
            "orientation_low_score_margin",
            f"Orientation score margin {_score_margin:.4f} is below threshold "
            f"{thresholds.orientation_score_margin_warn} — candidates have similar Dice scores.",
        )
    if _normalized_gap <= thresholds.orientation_confidence_warn:
        add_finding(
            "warning",
            "orientation_low_normalized_gap",
            f"Orientation normalized gap {_normalized_gap:.4f} is below threshold "
            f"{thresholds.orientation_confidence_warn}.",
        )

    if not strategy_agreement:
        add_finding("warning", "strategy_disagreement", "Orientation strategies disagree on the selected signed axis.")

    if placed_to_seg_ratio <= thresholds.placed_to_seg_ratio_fail_low or placed_to_seg_ratio >= thresholds.placed_to_seg_ratio_fail_high:
        add_finding("hard_failure", "placed_to_seg_ratio_fail", f"Placed/seg volume ratio {placed_to_seg_ratio:.3f} is implausible.")
    elif placed_to_seg_ratio <= thresholds.placed_to_seg_ratio_warn_low or placed_to_seg_ratio >= thresholds.placed_to_seg_ratio_warn_high:
        add_finding("warning", "placed_to_seg_ratio_warn", f"Placed/seg volume ratio {placed_to_seg_ratio:.3f} is outside the warning band.")
    if volume_target_below_min:
        add_finding(
            "warning",
            "volume_target_below_min",
            "Requested target volume is below the minimum producible volume within current scale bounds; returned best available mask.",
        )
    if volume_target_above_max_or_clipped:
        add_finding(
            "warning",
            "volume_target_above_max_or_clipped",
            "Requested target volume could not be reached under current generation/placement bounds; returned best available mask.",
        )

    warnings = [finding.message for finding in findings if finding.severity == "warning"]
    hard_failures = [finding.message for finding in findings if finding.severity == "hard_failure"]

    return EmbeddingCaseMetrics(
        case_id=case_id,
        seed=int(seed),
        source_mri=str(mri_path),
        source_seg=str(seg_path),
        orientation_method=selected_orientation.method,
        orientation_confidence=float(selected_orientation.confidence),
        orientation_score_margin=float(selected_orientation.debug.get("score_margin", 0.0)),
        orientation_normalized_gap=float(selected_orientation.debug.get("normalized_gap", selected_orientation.confidence)),
        orientation_low_confidence=bool(selected_orientation.low_confidence),
        selected_axis_vox=[float(v) for v in selected_axis_vox],
        selected_axis_phys=[float(v) for v in selected_axis_phys],
        source_voxel_count=int(primary_source_voxels),
        placed_voxel_count=int(primary_placed_voxels),
        max_placed_voxel_count=int(max_placed_voxel_count),
        source_volume_mm3=float(source_volume_mm3),
        placed_volume_mm3=float(placed_volume_mm3),
        max_placed_volume_mm3=float(max_placed_volume_mm3),
        retained_fraction=retained_fraction,
        centroid_offset_vox=float(primary_centroid_offset_vox),
        centroid_offset_mm=float(primary_centroid_offset_mm),
        primary_axis_error_deg=None if primary_axis_error_deg is None else float(primary_axis_error_deg),
        cpa_radius_anatomy_mm=float(cpa_radius_anatomy_mm),
        cpa_radius_effective_mm=float(cpa_radius_effective_mm),
        cpa_radius_override_active=bool(cpa_radius_override_active),
        volume_optimization_variable=None if volume_optimization_variable is None else str(volume_optimization_variable),
        target_volume_mm3=None if target_volume_mm3 is None else float(target_volume_mm3),
        volume_target_timepoint=None if volume_target_timepoint is None else str(volume_target_timepoint),
        volume_target_timepoint_index=None if volume_target_timepoint_index is None else int(volume_target_timepoint_index),
        volume_target_timepoint_day=None if volume_target_timepoint_day is None else int(volume_target_timepoint_day),
        target_timepoint_actual_volume_mm3=None if target_timepoint_actual_volume_mm3 is None else float(target_timepoint_actual_volume_mm3),
        target_timepoint_voxels=None if target_timepoint_voxels is None else int(target_timepoint_voxels),
        actual_volume_mm3=None if actual_volume_mm3 is None else float(actual_volume_mm3),
        ravd=None if ravd is None else float(ravd),
        volume_converged=None if volume_converged is None else bool(volume_converged),
        volume_iterations=None if volume_iterations is None else int(volume_iterations),
        volume_scale_final=None if volume_scale_final is None else float(volume_scale_final),
        target_seg_voxel_count=int(seg_voxel_count),
        placed_to_seg_ratio=placed_to_seg_ratio,
        monotone_growth=bool(monotone_growth),
        worst_clipping_fraction=float(worst_clipping_fraction),
        strategy_agreement=bool(strategy_agreement),
        warnings=warnings,
        hard_failures=hard_failures,
        findings=findings,
        strategy_results=[_serialize_orientation_result(result) for result in comparison_results],
        timepoint_metrics=timepoint_metrics,
    )


def write_case_reports(metrics: EmbeddingCaseMetrics, out_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and CSV reports for one case."""
    json_path = out_dir / "embedding_metrics.json"
    csv_path = out_dir / "embedding_metrics.csv"

    json_payload = asdict(metrics)
    json_path.write_text(json.dumps(json_payload, indent=2))

    csv_row = {
        "case_id": metrics.case_id,
        "seed": metrics.seed,
        "source_mri": metrics.source_mri,
        "source_seg": metrics.source_seg,
        "orientation_method": metrics.orientation_method,
        "orientation_confidence": metrics.orientation_confidence,
        "orientation_score_margin": metrics.orientation_score_margin,
        "orientation_normalized_gap": metrics.orientation_normalized_gap,
        "orientation_low_confidence": metrics.orientation_low_confidence,
        "selected_axis_vox": json.dumps(metrics.selected_axis_vox),
        "selected_axis_phys": json.dumps(metrics.selected_axis_phys),
        "source_voxel_count": metrics.source_voxel_count,
        "placed_voxel_count": metrics.placed_voxel_count,
        "max_placed_voxel_count": metrics.max_placed_voxel_count,
        "source_volume_mm3": metrics.source_volume_mm3,
        "placed_volume_mm3": metrics.placed_volume_mm3,
        "max_placed_volume_mm3": metrics.max_placed_volume_mm3,
        "retained_fraction": metrics.retained_fraction,
        "centroid_offset_vox": metrics.centroid_offset_vox,
        "centroid_offset_mm": metrics.centroid_offset_mm,
        "primary_axis_error_deg": metrics.primary_axis_error_deg,
        "cpa_radius_anatomy_mm": metrics.cpa_radius_anatomy_mm,
        "cpa_radius_effective_mm": metrics.cpa_radius_effective_mm,
        "cpa_radius_override_active": metrics.cpa_radius_override_active,
        "volume_optimization_variable": metrics.volume_optimization_variable,
        "target_volume_mm3": metrics.target_volume_mm3,
        "volume_target_timepoint": metrics.volume_target_timepoint,
        "volume_target_timepoint_index": metrics.volume_target_timepoint_index,
        "volume_target_timepoint_day": metrics.volume_target_timepoint_day,
        "target_timepoint_actual_volume_mm3": metrics.target_timepoint_actual_volume_mm3,
        "target_timepoint_voxels": metrics.target_timepoint_voxels,
        "actual_volume_mm3": metrics.actual_volume_mm3,
        "ravd": metrics.ravd,
        "volume_converged": metrics.volume_converged,
        "volume_iterations": metrics.volume_iterations,
        "volume_scale_final": metrics.volume_scale_final,
        "target_seg_voxel_count": metrics.target_seg_voxel_count,
        "placed_to_seg_ratio": metrics.placed_to_seg_ratio,
        "monotone_growth": metrics.monotone_growth,
        "worst_clipping_fraction": metrics.worst_clipping_fraction,
        "strategy_agreement": metrics.strategy_agreement,
        "warnings": json.dumps(metrics.warnings),
        "hard_failures": json.dumps(metrics.hard_failures),
        "findings": json.dumps([asdict(finding) for finding in metrics.findings]),
        "strategy_results": json.dumps(metrics.strategy_results),
    }
    fieldnames = list(csv_row.keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(csv_row)

    return json_path, csv_path


def principal_axes(mask: np.ndarray, spacing: np.ndarray) -> PrincipalAxesResult:
    """
    Return (centroid, long_axis, eigenvalues) for a binary 3-D mask.

    Uses skimage.measure.regionprops inertia_tensor.
    The eigenvector of the *smallest* eigenvalue is the long axis (least spread
    of mass = most elongated direction).

    centroid_vox    : (3,) float array, voxel indices
    long_axis_vox   : (3,) unit float array in voxel-index space
    long_axis_phys  : (3,) unit float array in physical/mm space
    """
    labeled, _ = ndi_label(mask > 0)
    props_vox = regionprops(labeled)
    if not props_vox:
        raise ValueError("regionprops found no labelled regions in the mask.")
    prop_vox = max(props_vox, key=lambda p: p.area)
    props_phys, has_spacing_props = _regionprops_with_spacing(labeled, spacing)
    prop_phys_by_label = {p.label: p for p in props_phys}
    prop_phys = prop_phys_by_label.get(prop_vox.label, prop_vox)

    centroid_vox = np.array(prop_vox.centroid, dtype=np.float64)

    eigvals_vox, eigvecs_vox = np.linalg.eigh(prop_vox.inertia_tensor)
    sort_idx_vox = np.argsort(eigvals_vox)
    long_axis_vox = eigvecs_vox[:, sort_idx_vox[0]]
    long_axis_vox /= np.linalg.norm(long_axis_vox)
    long_axis_vox = _stabilize_axis_sign(long_axis_vox)

    eigvals_phys, eigvecs_phys = np.linalg.eigh(prop_phys.inertia_tensor)
    sort_idx_phys = np.argsort(eigvals_phys)
    long_axis_phys = eigvecs_phys[:, sort_idx_phys[0]]
    long_axis_phys /= np.linalg.norm(long_axis_phys)
    if np.dot(long_axis_phys, _axis_vox_to_phys(long_axis_vox, spacing)) < 0.0:
        long_axis_phys = -long_axis_phys
    long_axis_phys = _stabilize_axis_sign(long_axis_phys)

    return PrincipalAxesResult(
        centroid_vox=centroid_vox,
        long_axis_vox=long_axis_vox,
        long_axis_phys=long_axis_phys,
        eigenvalues_vox=eigvals_vox[sort_idx_vox],
        eigenvalues_phys=eigvals_phys[sort_idx_phys],
    )


def rotate_and_translate(
    vol: np.ndarray,
    rot_mat_phys: np.ndarray,
    src_centroid: np.ndarray,
    dst_centroid: np.ndarray,
    out_shape: tuple[int, ...],
    dst_spacing: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Rigid-body transform: rotate *vol* around *src_centroid*, then translate
    so *src_centroid* lands at *dst_centroid* in the output volume.

    The synthetic source lives in isotropic synthetic space (1 synthetic voxel = 1 mm).
    The target MRI lives in voxel-index space with physical spacing `dst_spacing`.

    scipy.ndimage.affine_transform maps output voxel indices `j_out` to input
    synthetic coordinates `i_in` via:
        i_in = matrix @ j_out + offset

    Using physical-space rotation R and target spacing matrix S:
        x_out_mm = S @ j_out
        i_in = R^T (x_out_mm - S @ dst_centroid) + src_centroid
             = (R^T S) @ j_out + [src_centroid - R^T S @ dst_centroid]
    """
    spacing_mat = _spacing_matrix(dst_spacing)
    R_inv = rot_mat_phys.T
    matrix = R_inv @ spacing_mat
    offset = np.asarray(src_centroid, dtype=np.float64) - matrix @ np.asarray(dst_centroid, dtype=np.float64)
    return affine_transform(
        vol.astype(np.float32),
        matrix,
        offset=offset,
        output_shape=out_shape,
        order=order,
        mode="constant",
        cval=0.0,
    )


def _legacy_rotate_and_translate_for_validation(
    vol: np.ndarray,
    rot_mat: np.ndarray,
    src_centroid: np.ndarray,
    dst_centroid: np.ndarray,
    out_shape: tuple[int, ...],
    order: int = 1,
) -> np.ndarray:
    """Legacy index-space transform retained only for anisotropy validation diagnostics."""
    R_inv = rot_mat.T
    offset = np.asarray(src_centroid, dtype=np.float64) - R_inv @ np.asarray(dst_centroid, dtype=np.float64)
    return affine_transform(
        vol.astype(np.float32),
        R_inv,
        offset=offset,
        output_shape=out_shape,
        order=order,
        mode="constant",
        cval=0.0,
    )


def save_like(reference_nib: nib.spatialimages.SpatialImage, data: np.ndarray, out_path: Path) -> None:
    """Save data with a copied header/affine from a reference NIfTI."""
    header = reference_nib.header.copy()
    header.set_data_dtype(np.asarray(data).dtype)
    out_img = nib.Nifti1Image(np.asarray(data), reference_nib.affine, header)
    out_img.set_qform(reference_nib.get_qform(), code=int(reference_nib.header["qform_code"]))
    out_img.set_sform(reference_nib.get_sform(), code=int(reference_nib.header["sform_code"]))
    nib.save(out_img, str(out_path))


def overlay_rgba(base_slice: np.ndarray, mask_slice: np.ndarray,
                 color_rgb: tuple[float, float, float], alpha: float = 0.55
                 ) -> np.ndarray:
    """
    Return an RGBA image blending a grayscale slice with a coloured mask.

    base_slice   : (H, W) float, already normalised to [0, 1]
    mask_slice   : (H, W) bool / 0-1
    color_rgb    : tuple of 3 floats in [0, 1]
    """
    rgb = np.stack([base_slice] * 3, axis=-1)
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1))], axis=-1)
    where = mask_slice > 0
    for ch, c in enumerate(color_rgb):
        rgba[where, ch] = (1 - alpha) * rgb[where, ch] + alpha * c
    return rgba


def _save_qc_png(
    mri_data: np.ndarray,
    seg_data: np.ndarray,
    emb_vol: np.ndarray,
    emb_mask: np.ndarray,
    seg_centroid: np.ndarray,
    seg_long_axis: np.ndarray,
    mri_p1: float,
    mri_p99: float,
    out_path: Path,
    title_suffix: str = "",
    audit_suffix: str = "",
) -> None:
    """Save a 3×4 QC PNG with a tumor-centered ROI zoom column."""
    H, W, D = mri_data.shape
    zoom_box = 80

    # Choose the slice plane centred on the embedded mask
    if emb_mask.sum() > 0:
        coords = np.argwhere(emb_mask > 0)
        ax_slice  = int(np.round(np.mean(coords[:, 2])))  # depth
        cor_slice = int(np.round(np.mean(coords[:, 1])))  # width
        sag_slice = int(np.round(np.mean(coords[:, 0])))  # height
    else:
        ax_slice, cor_slice, sag_slice = D // 2, W // 2, H // 2

    ax_slice  = int(np.clip(ax_slice,  0, D - 1))
    cor_slice = int(np.clip(cor_slice, 0, W - 1))
    sag_slice = int(np.clip(sag_slice, 0, H - 1))

    def norm_slice(s: np.ndarray) -> np.ndarray:
        return np.clip((s.astype(np.float32) - mri_p1) / (mri_p99 - mri_p1 + 1e-8), 0, 1)

    def crop_bounds(length: int, center: int, box: int) -> tuple[int, int]:
        box = int(min(box, length))
        start = max(0, center - box // 2)
        end = start + box
        if end > length:
            end = length
            start = end - box
        return start, end

    def roi_norm_slice(s: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
        mask_idx = np.argwhere(mask_slice > 0)
        if mask_idx.size == 0:
            return norm_slice(s)
        pad = 8
        row0 = max(0, int(mask_idx[:, 0].min()) - pad)
        row1 = min(s.shape[0], int(mask_idx[:, 0].max()) + pad + 1)
        col0 = max(0, int(mask_idx[:, 1].min()) - pad)
        col1 = min(s.shape[1], int(mask_idx[:, 1].max()) + pad + 1)
        roi = s[row0:row1, col0:col1].astype(np.float32)
        if roi.size == 0:
            return norm_slice(s)
        roi_p1 = float(np.percentile(roi, 1))
        roi_p99 = float(np.percentile(roi, 99))
        return np.clip((s.astype(np.float32) - roi_p1) / (roi_p99 - roi_p1 + 1e-8), 0, 1)

    def crop_centered(arr: np.ndarray, center_rc: tuple[int, int]) -> np.ndarray:
        row0, row1 = crop_bounds(arr.shape[0], int(center_rc[0]), zoom_box)
        col0, col1 = crop_bounds(arr.shape[1], int(center_rc[1]), zoom_box)
        return arr[row0:row1, col0:col1]

    planes = [
        ("Axial",
         mri_data[:, :, ax_slice].T,   seg_data[:, :, ax_slice].T,
         emb_vol[:, :, ax_slice].T,    emb_mask[:, :, ax_slice].T),
        ("Coronal",
         mri_data[:, cor_slice, :].T,  seg_data[:, cor_slice, :].T,
         emb_vol[:, cor_slice, :].T,   emb_mask[:, cor_slice, :].T),
        ("Sagittal",
         mri_data[sag_slice, :, :].T,  seg_data[sag_slice, :, :].T,
         emb_vol[sag_slice, :, :].T,   emb_mask[sag_slice, :, :].T),
    ]

    col_titles = ["Original MRI", "Real Segmentation", "Embedded + Mask", "ROI Zoom (2.5x)"]
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    heading = f"Embedded lollipop QC  —  {title_suffix}\n" if title_suffix else "Embedded lollipop QC\n"
    fig.suptitle(
        heading +
        f"Seg centroid ≈ {np.round(seg_centroid, 1)}   "
        f"Long axis ≈ {np.round(seg_long_axis, 3)}"
        + (f"   {audit_suffix}" if audit_suffix else ""),
        fontsize=10,
    )

    for row, (plane_name, orig, seg_sl, emb, emb_mask_sl) in enumerate(planes):
        orig_n = norm_slice(orig)
        emb_n  = norm_slice(emb)
        zoom_base_n = roi_norm_slice(emb, emb_mask_sl)
        overlay_zoom = overlay_rgba(zoom_base_n, emb_mask_sl, (0.0, 0.85, 1.0), alpha=0.6)
        if np.any(emb_mask_sl):
            zoom_center = tuple(np.round(np.argwhere(emb_mask_sl > 0).mean(axis=0)).astype(int))
        else:
            zoom_center = (emb_mask_sl.shape[0] // 2, emb_mask_sl.shape[1] // 2)
        overlay_zoom = crop_centered(overlay_zoom, zoom_center)

        axes[row, 0].imshow(orig_n, cmap="gray", vmin=0, vmax=1,
                            origin="lower", aspect="auto")
        axes[row, 0].set_ylabel(plane_name, fontsize=9)

        axes[row, 1].imshow(overlay_rgba(orig_n, seg_sl, (1.0, 0.15, 0.15), alpha=0.55),
                            origin="lower", aspect="auto")

        axes[row, 2].imshow(overlay_rgba(emb_n, emb_mask_sl, (0.0, 0.85, 1.0), alpha=0.6),
                            origin="lower", aspect="auto")

        axes[row, 3].imshow(overlay_zoom, origin="lower", aspect="auto")

        for col in range(4):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved QC: {out_path}")


def feather_mask_alpha(mask: np.ndarray, voxel_spacing: np.ndarray, blend_width_mm: float = 1.25) -> np.ndarray:
    """
    Build a narrow interior feathering alpha for boundary blending.

    Interior voxels stay mostly synthetic; only a thin inner shell near the
    boundary is blended with the native MRI.
    """
    if blend_width_mm <= 0.0:
        return (mask > 0).astype(np.float32)
    inside_dist_mm = distance_transform_edt(mask > 0, sampling=tuple(float(v) for v in voxel_spacing))
    t = np.clip(inside_dist_mm / blend_width_mm, 0.0, 1.0).astype(np.float32)
    # Smoothstep for gentle transition without broad blurring.
    alpha = t * t * (3.0 - 2.0 * t)
    alpha[mask <= 0] = 0.0
    return alpha


def run_anisotropy_validation() -> None:
    """Compare the legacy and spacing-aware rigid transforms on an anisotropic target grid."""
    print("\nRunning anisotropy validation…")
    target_spacing = np.array([0.5, 0.5, 3.0], dtype=np.float64)
    target_shape = (96, 96, 48)
    target_centroid = np.array([47.5, 43.0, 20.0], dtype=np.float64)
    target_axis_phys = np.array([0.72, 0.38, 0.58], dtype=np.float64)
    target_axis_phys /= np.linalg.norm(target_axis_phys)
    target_axis_vox = np.linalg.inv(_spacing_matrix(target_spacing)) @ target_axis_phys
    target_axis_vox /= np.linalg.norm(target_axis_vox)

    syn_size = 64
    _, syn_labels = create_synthetic_time_3d(
        height=syn_size,
        width=syn_size,
        depth=syn_size,
        dates=[0, 120, 240],
        rotation_degrees=[0, 0, 0],
        channel_dim=0,
        growth="steady",
        growth_direction="a",
        geometry_mode="lollipop",
        canal_axis="c",
        canal_base_radius_max=3.5,
        canal_apex_radius_max=2.0,
        canal_length_max_override=10.0,
        bulb_radius_max=6.0,
        centered=True,
        random_state=np.random.default_rng(0),
    )
    src_mask = syn_labels[-1][0].astype(np.float32)
    _, src_prop = largest_component(src_mask > 0.5)
    src_centroid = np.array(src_prop.centroid, dtype=np.float64)
    default_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    rot_phys, _ = Rotation.align_vectors(target_axis_phys[np.newaxis], default_axis[np.newaxis])
    rot_phys_matrix = rot_phys.as_matrix()
    rot_legacy, _ = Rotation.align_vectors(target_axis_phys[np.newaxis], default_axis[np.newaxis])
    rot_legacy_matrix = rot_legacy.as_matrix()

    legacy_mask = _legacy_rotate_and_translate_for_validation(
        src_mask,
        rot_legacy_matrix,
        src_centroid,
        target_centroid,
        out_shape=target_shape,
        order=0,
    ) > 0.5
    corrected_mask = rotate_and_translate(
        src_mask,
        rot_phys_matrix,
        src_centroid,
        target_centroid,
        out_shape=target_shape,
        dst_spacing=target_spacing,
        order=0,
    ) > 0.5

    def metrics(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        axes = principal_axes(mask.astype(np.uint8), target_spacing)
        centroid_offset_mm = float(np.linalg.norm((axes.centroid_vox - target_centroid) * target_spacing))
        angle_phys_deg = _angle_deg(axes.long_axis_phys, target_axis_phys)
        return axes.centroid_vox, axes.long_axis_phys, centroid_offset_mm, angle_phys_deg

    legacy_centroid, legacy_axis_phys, legacy_ctr_mm, legacy_ang = metrics(legacy_mask)
    corrected_centroid, corrected_axis_phys, corrected_ctr_mm, corrected_ang = metrics(corrected_mask)

    print(f"  Target spacing          : {np.round(target_spacing, 4)} mm")
    print(f"  Intended axis (phys)    : {np.round(target_axis_phys, 4)}")
    print(f"  Intended axis (vox)     : {np.round(target_axis_vox, 4)}")
    print(f"  Legacy centroid offset  : {legacy_ctr_mm:.3f} mm")
    print(f"  Legacy axis error       : {legacy_ang:.3f} deg  realized={np.round(legacy_axis_phys, 4)}")
    print(f"  Corrected centroid off. : {corrected_ctr_mm:.3f} mm")
    print(f"  Corrected axis error    : {corrected_ang:.3f} deg  realized={np.round(corrected_axis_phys, 4)}")
    print(f"  Legacy retained voxels  : {int(legacy_mask.sum())}")
    print(f"  Corrected retained vox. : {int(corrected_mask.sum())}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(
    mri_path: Path,
    seg_path: Path,
    out_dir: Path,
    gen_size: int = 128,
    dates: list[int] | None = None,
    rad_min: int = 4,
    rad_max: int = 30,
    growth: str = "steady",
    canal_growth_scale: float = 1.0,
    bulb_growth_scale: float = 1.0,
    t0_volume_fraction_of_seg: float | None = None,
    validate_anisotropy: bool = False,
    seed: int | None = None,
    target_tumor_volume_mm3: float | None = None,
    volume_target_timepoint: str = "first",
    volume_ravd_tolerance: float = 0.05,
    volume_max_iterations: int = 10,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    if dates is None:
        dates = [0, 60, 120, 180, 240]
    if t0_volume_fraction_of_seg is not None and t0_volume_fraction_of_seg <= 0.0:
        raise ValueError("t0_volume_fraction_of_seg must be > 0 when provided.")
    if target_tumor_volume_mm3 is not None and target_tumor_volume_mm3 <= 0.0:
        raise ValueError("target_tumor_volume_mm3 must be > 0 when provided.")
    if volume_ravd_tolerance <= 0.0:
        raise ValueError("volume_ravd_tolerance must be > 0.")
    if volume_max_iterations < 1:
        raise ValueError("volume_max_iterations must be >= 1.")
    if target_tumor_volume_mm3 is not None and t0_volume_fraction_of_seg is not None:
        raise ValueError("Provide either target_tumor_volume_mm3 or t0_volume_fraction_of_seg, not both.")

    volume_target_timepoint_raw = str(volume_target_timepoint).strip().lower()
    if volume_target_timepoint_raw == "first":
        volume_target_timepoint_index = 0
    elif volume_target_timepoint_raw == "last":
        volume_target_timepoint_index = len(dates) - 1
    else:
        try:
            volume_target_timepoint_index = int(volume_target_timepoint_raw)
        except ValueError as exc:
            raise ValueError(
                "volume_target_timepoint must be 'first', 'last', or an integer index string."
            ) from exc
        if volume_target_timepoint_index < 0 or volume_target_timepoint_index >= len(dates):
            raise ValueError(
                f"volume_target_timepoint index {volume_target_timepoint_index} is out of bounds for "
                f"{len(dates)} timepoints."
            )
    volume_target_timepoint_day = int(dates[volume_target_timepoint_index])

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading MRI and segmentation…")
    mri_nib = nib.load(str(mri_path))
    seg_nib = nib.load(str(seg_path))
    mri_data = mri_nib.get_fdata().astype(np.float32)
    seg_data = (seg_nib.get_fdata() > 0.5).astype(np.uint8)
    if mri_data.shape != seg_data.shape:
        raise ValueError(f"MRI/segmentation shape mismatch: {mri_data.shape} vs {seg_data.shape}")
    if not np.allclose(mri_nib.affine, seg_nib.affine, atol=1e-4):
        raise ValueError("MRI and segmentation affines differ; inputs must be coregistered on the same grid.")
    H, W, D = mri_data.shape
    voxel_spacing = np.array(mri_nib.header.get_zooms()[:3], dtype=np.float64)
    if voxel_spacing.shape[0] != 3 or not np.all(np.isfinite(voxel_spacing)) or np.any(voxel_spacing <= 0.0):
        raise ValueError(f"Invalid voxel spacing from MRI header: {voxel_spacing!r}")
    voxel_volume_mm3 = float(np.prod(voxel_spacing))
    case_id = f"{mri_path.resolve()}__{seg_path.resolve()}"
    seed = int(seed) if seed is not None else stable_seed_from_case(case_id)
    rng = np.random.default_rng(seed)
    print(f"  MRI  shape : {mri_data.shape}  dtype: {mri_data.dtype}")
    print(f"  Seg  shape : {seg_data.shape}   nonzero: {int(seg_data.sum())} voxels")
    print(f"  Voxel size : {tuple(np.round(voxel_spacing, 4))} mm")
    print(f"  Seed       : {seed}")

    # ── 2. Segmentation geometry ──────────────────────────────────────────────
    print("\nComputing segmentation geometry…")
    seg_component, _ = largest_component(seg_data)
    if int(seg_component.sum()) != int(seg_data.sum()):
        print(f"  [warn] Segmentation has multiple components; using largest component with {int(seg_component.sum())} voxels.")
    seg_axes = principal_axes(seg_component, voxel_spacing)
    seg_centroid = seg_axes.centroid_vox
    seg_long_axis_vox = seg_axes.long_axis_vox
    seg_long_axis_phys = seg_axes.long_axis_phys
    print(f"  Centroid        : {np.round(seg_centroid, 2)}")
    print(f"  Long axis (vox) : {np.round(seg_long_axis_vox, 4)}")
    print(f"  Long axis (mm)  : {np.round(seg_long_axis_phys, 4)}")
    print(f"  Inertia eigvals : vox={np.round(seg_axes.eigenvalues_vox, 1)}  mm={np.round(seg_axes.eigenvalues_phys, 1)}")

    # ── 2b. Compute mm-based lollipop geometry (IAC literature + seg stats) ─────
    # IAC dimensions from literature:
    #   canal length  ≈ 10 mm   (porus acousticus → fundus)
    #   porus opening ≈  7 mm diameter  →  base_radius = 3.5 mm
    #   fundus        ≈ 4–5 mm diameter →  apex_radius = 2.0 mm
    # Synthetic geometry stays in isotropic synthetic space where 1 syn voxel = 1 mm.
    # The target MRI spacing is handled later by the placement transform.
    # The CPA (extrameatal) bulb is sized from the actual segmentation volume,
    # not the IAC literature, so it reflects the patient's individual anatomy.
    iac_canal_mm  = 10.0   # literature: IAC canal length
    iac_base_r_mm =  3.5   # porus opening radius
    iac_apex_r_mm =  2.0   # fundus tip radius
    iac_canal_syn   = iac_canal_mm
    iac_base_r_syn  = iac_base_r_mm
    iac_apex_r_syn  = iac_apex_r_mm

    # CPA bulb radius derived from segmentation volume.
    seg_vol_mm3 = float(seg_component.sum()) * float(np.prod(voxel_spacing))
    V_iac_mm3 = (
        np.pi * iac_canal_mm / 3.0
        * (iac_base_r_mm ** 2 + iac_base_r_mm * iac_apex_r_mm + iac_apex_r_mm ** 2)
    )  # truncated-cone formula
    V_cpa_mm3 = max(0.0, seg_vol_mm3 - V_iac_mm3)
    if V_cpa_mm3 > 0.0:
        cpa_radius_mm = (3.0 * V_cpa_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        cpa_radius_mm = iac_base_r_mm  # fallback: match porus opening
    cpa_radius_anatomy_mm = float(cpa_radius_mm)
    cpa_radius_effective_mm = float(cpa_radius_anatomy_mm)
    cpa_radius_override_active = False
    if target_tumor_volume_mm3 is not None:
        target_cpa_mm3 = max(0.0, float(target_tumor_volume_mm3) - V_iac_mm3)
        if target_cpa_mm3 > 0.0:
            # Existing spherical approximation.
            required_cpa_radius_sphere_mm = (3.0 * target_cpa_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
            # Oblate-aware approximation from generator geometry:
            # cpa_ax_r = min(br*0.7, cr*0.45), cpa_eq_r = cr.
            # V ≈ 4/3*pi*cpa_ax_r*cr^2.
            br_mm = iac_base_r_mm
            cpa_ax_cap_mm = br_mm * 0.7
            cr_transition_mm = cpa_ax_cap_mm / 0.45
            v_transition_mm3 = (4.0 / 3.0) * np.pi * cpa_ax_cap_mm * (cr_transition_mm ** 2)
            if target_cpa_mm3 <= v_transition_mm3:
                required_cpa_radius_oblate_mm = (target_cpa_mm3 / ((4.0 / 3.0) * np.pi * 0.45)) ** (1.0 / 3.0)
            else:
                required_cpa_radius_oblate_mm = (target_cpa_mm3 / ((4.0 / 3.0) * np.pi * cpa_ax_cap_mm)) ** 0.5
            required_cpa_radius_mm = max(required_cpa_radius_sphere_mm, required_cpa_radius_oblate_mm)
            if required_cpa_radius_mm > cpa_radius_anatomy_mm:
                cpa_radius_override_active = True
                cpa_radius_effective_mm = float(required_cpa_radius_mm)
                print(
                    f"  [warn] Target volume {float(target_tumor_volume_mm3):.1f} mm^3 requires larger CPA radius: "
                    f"anatomy={cpa_radius_anatomy_mm:.2f} mm -> effective={cpa_radius_effective_mm:.2f} mm"
                )
    if (
        target_tumor_volume_mm3 is not None
        and volume_target_timepoint_index == 0
        and (cpa_radius_override_active or (float(target_tumor_volume_mm3) / max(seg_vol_mm3, 1e-8)) > 0.5)
    ):
        print(
            "  [warn] first-timepoint volume targeting is in stress-test mode for this target; "
            "it may invalidate longitudinal growth semantics and principal-axis validation."
        )
    cpa_radius_syn = cpa_radius_effective_mm

    print("\nLollipop geometry  (synthetic isotropic space = 1 mm / syn voxel):")
    print(f"  IAC canal length : {iac_canal_mm:.1f} mm  = {iac_canal_syn:.1f} syn vox  [literature]")
    print(f"  Porus radius     : {iac_base_r_mm:.1f} mm  = {iac_base_r_syn:.1f} syn vox  [literature]")
    print(f"  Fundus radius    : {iac_apex_r_mm:.1f} mm  = {iac_apex_r_syn:.1f} syn vox  [literature]")
    print(f"  Seg volume       : {seg_vol_mm3:.0f} mm³  (IAC cone ≈ {V_iac_mm3:.0f} mm³)")
    print(f"  CPA bulb radius (anatomy) : {cpa_radius_anatomy_mm:.1f} mm")
    print(f"  CPA bulb radius (effective): {cpa_radius_effective_mm:.1f} mm  = {cpa_radius_syn:.1f} syn vox")

    # Ensure gen_size is large enough to hold the full lollipop at final dimensions.
    # With centered=True the porus sits at cube centre; max half-extents are:
    #   + canal direction : canal_length + apex_radius
    #   − canal direction : 1.55 * cpa_radius  (CPA center offset + bulb radius)
    #   ± perp directions : max(base_radius, cpa_radius)
    max_half_vox = max(iac_canal_syn + iac_apex_r_syn, 1.55 * cpa_radius_syn, max(iac_base_r_syn, cpa_radius_syn))
    min_gen = int(np.ceil(2 * max_half_vox)) + 8  # +8 vox safety margin
    if gen_size < min_gen:
        gen_size = min_gen
        print(f"  [info] gen_size bumped to {gen_size} vox to fit full lollipop")

    # ── 3. Generate synthetic lollipop time series ────────────────────────────
    # canal_axis="c" → canal along depth dim (spz_rotate) → local axis [0,0,1]
    # centered=True  → porus placed at cube centre, avoiding boundary clipping
    print(f"\nGenerating lollipop tumor series  (size={gen_size}, dates={dates})…")

    def _generate_synthetic_series(
        init_scale: float | None = None,
        bulb_radius_max_mm: float | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        kwargs: dict[str, float | int | list[int] | str | bool | np.random.Generator] = {
            "height": gen_size,
            "width": gen_size,
            "depth": gen_size,
            "dates": dates,
            "rotation_degrees": [0, 0, 0],  # no pre-rotation; we apply our own below
            "rad_max": rad_max,
            "rad_min": rad_min,
            "channel_dim": 0,               # → (1, H, W, D)
            "growth": growth,
            "growth_direction": "a",        # only affects a/b/c radii, irrelevant for lollipop
            "geometry_mode": "lollipop",
            "canal_axis": "c",              # canal grows along depth dim → default axis [0,0,1]
            "canal_base_radius_max": iac_base_r_syn,
            "canal_apex_radius_max": iac_apex_r_syn,
            "canal_length_max_override": iac_canal_syn,
            "bulb_radius_max": cpa_radius_syn if bulb_radius_max_mm is None else float(bulb_radius_max_mm),
            "canal_growth_scale": canal_growth_scale,
            "bulb_growth_scale": bulb_growth_scale,
            "centered": True,               # fix porus at cube centre — prevents boundary clipping
            "random_state": np.random.default_rng(seed),
        }
        if init_scale is not None:
            kwargs.update(
                {
                    "canal_length_init": min(iac_canal_syn, max(1.0, 2.0 * init_scale)),
                    "canal_base_radius_init": min(iac_base_r_syn, max(0.75, iac_base_r_syn * 0.25 * init_scale)),
                    "canal_apex_radius_init": min(iac_apex_r_syn, max(0.5, iac_apex_r_syn * 0.25 * init_scale)),
                    "bulb_radius_init": min(cpa_radius_syn, max(1.0, 2.0 * init_scale)),
                }
            )
        return create_synthetic_time_3d(**kwargs)

    default_axis = np.array([0.0, 0.0, 1.0])
    strategies: list[OrientationStrategy] = [
        LateDiceOrientationStrategy(),
        MidGrowthDiceOrientationStrategy(),
    ]

    volume_debug_enabled = target_tumor_volume_mm3 is not None

    def _prepare_series(
        init_scale: float | None,
        candidate_tag: str = "candidate",
        bulb_radius_max_mm: float | None = None,
    ) -> dict[str, object]:
        syn_images_local, syn_labels_local = _generate_synthetic_series(
            init_scale=init_scale,
            bulb_radius_max_mm=bulb_radius_max_mm,
        )
        voxel_counts_local = [int(np.sum(lab[0] > 0)) for lab in syn_labels_local]
        smallest_idx_local = int(np.argmin(voxel_counts_local))
        last_idx_local = len(dates) - 1
        mid_idx_local = max(0, min(last_idx_local, len(dates) // 2))
        smallest_mask_local = syn_labels_local[smallest_idx_local][0]
        target_mask_local = syn_labels_local[volume_target_timepoint_index][0]
        _, syn_prop_local = largest_component(smallest_mask_local)
        syn_centroid_local = np.array(syn_prop_local.centroid, dtype=np.float64)
        orientation_inputs_local = OrientationInputs(
            seg_mask=seg_component,
            unresolved_axis_vox=seg_axes.long_axis_vox,
            default_axis_syn=default_axis,
            syn_centroid=syn_centroid_local,
            seg_centroid=seg_centroid,
            out_shape=(H, W, D),
            target_spacing=voxel_spacing,
            syn_masks_by_name={
                "late": syn_labels_local[last_idx_local][0],
                "mid": syn_labels_local[mid_idx_local][0],
            },
        )
        comparison_results_local = compare_orientation_strategies(orientation_inputs_local, strategies)
        default_orientation_local = next(
            result for result in comparison_results_local if result.method == LateDiceOrientationStrategy.name
        )
        selected_axis_vox_local = default_orientation_local.axis_vox
        selected_axis_phys_local = _axis_vox_to_phys(selected_axis_vox_local, voxel_spacing)
        try:
            rot_local, _ = Rotation.align_vectors(
                selected_axis_phys_local[np.newaxis],  # target (MRI physical space)
                default_axis[np.newaxis],              # source (synthetic isotropic space)
            )
            rot_matrix_local = rot_local.as_matrix()
        except Exception:
            rot_matrix_local = np.eye(3)
        placed_target_local = rotate_and_translate(
            target_mask_local.astype(np.float32),
            rot_matrix_local,
            syn_centroid_local,
            seg_centroid,
            out_shape=(H, W, D),
            dst_spacing=voxel_spacing,
            order=0,
        )
        placed_target_local = (placed_target_local > 0.5).astype(np.uint8)
        source_target_vox_local = int(np.sum(target_mask_local > 0))
        placed_target_vox_local = int(np.sum(placed_target_local > 0))
        retained_target_fraction_local = float(placed_target_vox_local / max(source_target_vox_local, 1))
        clipped_target_local = bool(source_target_vox_local > 0 and placed_target_vox_local < source_target_vox_local)
        actual_volume_mm3_local = float(placed_target_vox_local) * voxel_volume_mm3
        if volume_debug_enabled:
            scale_str = "default" if init_scale is None else f"{float(init_scale):.6f}"
            print(
                f"  [volume-candidate:{candidate_tag}] "
                f"scale={scale_str} "
                f"target=t{volume_target_timepoint_index} day={volume_target_timepoint_day} "
                f"source_vox={source_target_vox_local} "
                f"placed_vox={placed_target_vox_local} "
                f"actual_mm3={actual_volume_mm3_local:.1f} "
                f"retained={retained_target_fraction_local:.4f} "
                f"clipped={clipped_target_local} "
                f"bulb_radius_max_mm={cpa_radius_syn if bulb_radius_max_mm is None else float(bulb_radius_max_mm):.4f} "
                f"(smallest=t{smallest_idx_local} day={int(dates[smallest_idx_local])})"
            )
        return {
            "init_scale": init_scale,
            "syn_images": syn_images_local,
            "syn_labels": syn_labels_local,
            "voxel_counts": voxel_counts_local,
            "smallest_idx": smallest_idx_local,
            "last_idx": last_idx_local,
            "mid_idx": mid_idx_local,
            "syn_centroid": syn_centroid_local,
            "comparison_results": comparison_results_local,
            "default_orientation": default_orientation_local,
            "selected_axis_vox": selected_axis_vox_local,
            "selected_axis_phys": selected_axis_phys_local,
            "rot_matrix": rot_matrix_local,
            "actual_volume_mm3": actual_volume_mm3_local,
            "source_target_voxels": source_target_vox_local,
            "placed_target_voxels": placed_target_vox_local,
            "retained_target_fraction": retained_target_fraction_local,
            "clipped_target": clipped_target_local,
            "target_timepoint_index": volume_target_timepoint_index,
            "target_timepoint_day": volume_target_timepoint_day,
            "bulb_radius_max_mm": cpa_radius_syn if bulb_radius_max_mm is None else float(bulb_radius_max_mm),
        }

    volume_target_mm3: float | None = None
    volume_actual_mm3: float | None = None
    target_timepoint_actual_volume_mm3: float | None = None
    target_timepoint_voxels: int | None = None
    volume_ravd: float | None = None
    volume_converged: bool | None = None
    volume_iterations: int | None = None
    volume_scale_final: float | None = None
    volume_optimization_variable: str | None = None
    volume_target_below_min = False
    volume_target_above_max_or_clipped = False
    calibration_scale: float | None = None

    if target_tumor_volume_mm3 is not None:
        volume_target_mm3 = float(target_tumor_volume_mm3)
        optimize_last = volume_target_timepoint_index == (len(dates) - 1)
        if optimize_last:
            volume_optimization_variable = "cpa_radius_effective_mm"
            fixed_init_scale: float | None = None
            low_val = max(0.5, cpa_radius_anatomy_mm * 0.35)
            high_val = max(cpa_radius_effective_mm, cpa_radius_anatomy_mm)
            prep_low = _prepare_series(fixed_init_scale, candidate_tag="low", bulb_radius_max_mm=low_val)
            prep_high = _prepare_series(fixed_init_scale, candidate_tag="high", bulb_radius_max_mm=high_val)
            low_vol = float(prep_low["actual_volume_mm3"])
            high_vol = float(prep_high["actual_volume_mm3"])
            for _ in range(8):
                if low_vol <= volume_target_mm3:
                    break
                next_low = low_val / 1.5
                if next_low <= 1e-4:
                    break
                high_val, prep_high, high_vol = low_val, prep_low, low_vol
                low_val = next_low
                prep_low = _prepare_series(fixed_init_scale, candidate_tag="low-expand", bulb_radius_max_mm=low_val)
                low_vol = float(prep_low["actual_volume_mm3"])
            if low_vol > volume_target_mm3:
                volume_target_below_min = True
            high_expand_attempts = 8
            high_plateau_detected = False
            for _ in range(high_expand_attempts):
                if high_vol >= volume_target_mm3:
                    break
                low_val, prep_low, low_vol = high_val, prep_high, high_vol
                high_val *= 1.5
                prev_high_vol = high_vol
                prep_high = _prepare_series(fixed_init_scale, candidate_tag="high-expand", bulb_radius_max_mm=high_val)
                high_vol = float(prep_high["actual_volume_mm3"])
                if high_vol <= prev_high_vol * 1.001:
                    high_plateau_detected = True
            if high_vol < volume_target_mm3:
                volume_target_above_max_or_clipped = True
                clipped_hint = bool(prep_high.get("clipped_target", False))
                plateau_hint = bool(high_plateau_detected)
                reason_bits = []
                if clipped_hint:
                    reason_bits.append("clipping")
                if plateau_hint:
                    reason_bits.append("volume plateau")
                reason = ", ".join(reason_bits) if reason_bits else "bound-limited search"
                print(
                    f"  [warn] Requested target volume {volume_target_mm3:.1f} mm^3 could not be reached: "
                    f"highest tested cpa_radius_effective_mm={high_val:.5f} produced {high_vol:.1f} mm^3 "
                    f"(reason: {reason}). Using best-candidate fallback (volume_converged=false)."
                )
            best_prep = prep_low
            best_ravd = abs(float(best_prep["actual_volume_mm3"]) - volume_target_mm3) / volume_target_mm3
            high_ravd = abs(float(prep_high["actual_volume_mm3"]) - volume_target_mm3) / volume_target_mm3
            if high_ravd < best_ravd:
                best_prep = prep_high
                best_ravd = high_ravd
            volume_iterations = 0
            volume_converged = False
            for _ in range(volume_max_iterations):
                volume_iterations += 1
                mid_val = 0.5 * (low_val + high_val)
                prep_mid = _prepare_series(fixed_init_scale, candidate_tag="mid", bulb_radius_max_mm=mid_val)
                mid_vol = float(prep_mid["actual_volume_mm3"])
                mid_ravd = abs(mid_vol - volume_target_mm3) / volume_target_mm3
                if mid_ravd < best_ravd:
                    best_ravd = mid_ravd
                    best_prep = prep_mid
                if mid_ravd <= volume_ravd_tolerance:
                    volume_converged = True
                    best_prep = prep_mid
                    best_ravd = mid_ravd
                    break
                if mid_vol < volume_target_mm3:
                    low_val = mid_val
                else:
                    high_val = mid_val
            prepared = best_prep
            cpa_radius_effective_mm = float(prepared["bulb_radius_max_mm"])
            cpa_radius_override_active = cpa_radius_effective_mm > cpa_radius_anatomy_mm
            volume_actual_mm3 = float(prepared["actual_volume_mm3"])
            volume_ravd = best_ravd
            volume_scale_final = cpa_radius_effective_mm
        else:
            volume_optimization_variable = "init_scale"
            low_scale = 0.25
            high_scale = 3.0
            prep_low = _prepare_series(low_scale, candidate_tag="low", bulb_radius_max_mm=cpa_radius_effective_mm)
            prep_high = _prepare_series(high_scale, candidate_tag="high", bulb_radius_max_mm=cpa_radius_effective_mm)
            low_vol = float(prep_low["actual_volume_mm3"])
            high_vol = float(prep_high["actual_volume_mm3"])
            if high_vol < low_vol:
                print(
                    f"  [warn] Volume monotonicity sanity check failed: "
                    f"scale {high_scale:.3f} produced {high_vol:.1f} mm^3 < {low_vol:.1f} mm^3 at scale {low_scale:.3f}"
                )
            for _ in range(8):
                if low_vol <= volume_target_mm3:
                    break
                next_low_scale = low_scale / 1.5
                if next_low_scale <= 1e-4:
                    break
                high_scale, prep_high, high_vol = low_scale, prep_low, low_vol
                low_scale = next_low_scale
                prep_low = _prepare_series(low_scale, candidate_tag="low-expand", bulb_radius_max_mm=cpa_radius_effective_mm)
                low_vol = float(prep_low["actual_volume_mm3"])
            if low_vol > volume_target_mm3:
                volume_target_below_min = True
                print(
                    f"  [warn] Requested target volume {volume_target_mm3:.1f} mm^3 is below "
                    f"minimum producible volume {low_vol:.1f} mm^3 at scale {low_scale:.5f}; "
                    "using best-candidate fallback (volume_converged=false)."
                )
            high_expand_attempts = 8
            high_plateau_detected = False
            for _ in range(high_expand_attempts):
                if high_vol >= volume_target_mm3:
                    break
                low_scale, prep_low, low_vol = high_scale, prep_high, high_vol
                high_scale *= 1.5
                prev_high_vol = high_vol
                prep_high = _prepare_series(high_scale, candidate_tag="high-expand", bulb_radius_max_mm=cpa_radius_effective_mm)
                high_vol = float(prep_high["actual_volume_mm3"])
                if high_vol <= prev_high_vol * 1.001:
                    high_plateau_detected = True
            if high_vol < volume_target_mm3:
                volume_target_above_max_or_clipped = True
                clipped_hint = bool(prep_high.get("clipped_target", False))
                plateau_hint = bool(high_plateau_detected)
                reason_bits = []
                if clipped_hint:
                    reason_bits.append("clipping")
                if plateau_hint:
                    reason_bits.append("volume plateau")
                reason = ", ".join(reason_bits) if reason_bits else "bound-limited search"
                print(
                    f"  [warn] Requested target volume {volume_target_mm3:.1f} mm^3 could not be reached: "
                    f"highest tested scale={high_scale:.5f} produced {high_vol:.1f} mm^3 "
                    f"(reason: {reason}). Using best-candidate fallback (volume_converged=false)."
                )
            best_prep = prep_low
            best_ravd = abs(float(best_prep["actual_volume_mm3"]) - volume_target_mm3) / volume_target_mm3
            high_ravd = abs(float(prep_high["actual_volume_mm3"]) - volume_target_mm3) / volume_target_mm3
            if high_ravd < best_ravd:
                best_prep = prep_high
                best_ravd = high_ravd
            volume_iterations = 0
            volume_converged = False
            for _ in range(volume_max_iterations):
                volume_iterations += 1
                mid_scale = 0.5 * (low_scale + high_scale)
                prep_mid = _prepare_series(mid_scale, candidate_tag="mid", bulb_radius_max_mm=cpa_radius_effective_mm)
                mid_vol = float(prep_mid["actual_volume_mm3"])
                mid_ravd = abs(mid_vol - volume_target_mm3) / volume_target_mm3
                if mid_ravd < best_ravd:
                    best_ravd = mid_ravd
                    best_prep = prep_mid
                if mid_ravd <= volume_ravd_tolerance:
                    volume_converged = True
                    best_prep = prep_mid
                    best_ravd = mid_ravd
                    break
                if mid_vol < volume_target_mm3:
                    low_scale = mid_scale
                else:
                    high_scale = mid_scale
            prepared = best_prep
            volume_actual_mm3 = float(prepared["actual_volume_mm3"])
            volume_ravd = best_ravd
            scale_value = prepared["init_scale"]
            volume_scale_final = None if scale_value is None else float(scale_value)
        print(
            f"  [volume] target={volume_target_mm3:.1f} mm^3 "
            f"actual={volume_actual_mm3:.1f} mm^3 ravd={volume_ravd:.4f} "
            f"converged={volume_converged} iters={volume_iterations} "
            f"{volume_optimization_variable}={volume_scale_final}"
        )
        if not volume_converged and (volume_target_below_min or volume_target_above_max_or_clipped):
            print(
                "  [volume] non-converged result is expected under current bounds; "
                "see warnings above for bound/plateau/clipping diagnostics."
            )
    else:
        initial_scale: float | None = None
        if t0_volume_fraction_of_seg is not None:
            baseline = _generate_synthetic_series()
            baseline_t0_vox = float(np.sum(baseline[1][0][0] > 0))  # 1 syn voxel = 1 mm^3
            target_t0_volume_mm3 = float(seg_vol_mm3 * t0_volume_fraction_of_seg)
            if baseline_t0_vox > 0.0 and target_t0_volume_mm3 > 0.0:
                calibration_scale = float((target_t0_volume_mm3 / baseline_t0_vox) ** (1.0 / 3.0))
                calibration_scale = float(np.clip(calibration_scale, 0.5, 3.0))
                print(
                    f"  [calibrate] target t0 volume={target_t0_volume_mm3:.1f} mm^3  "
                    f"observed={baseline_t0_vox:.1f} mm^3  scale={calibration_scale:.3f}"
                )
                initial_scale = calibration_scale
        prepared = _prepare_series(initial_scale)

    syn_images = prepared["syn_images"]
    syn_labels = prepared["syn_labels"]
    voxel_counts = prepared["voxel_counts"]
    smallest_idx = int(prepared["smallest_idx"])
    last_idx = int(prepared["last_idx"])
    syn_centroid = prepared["syn_centroid"]
    comparison_results = prepared["comparison_results"]
    default_orientation = prepared["default_orientation"]
    seg_long_axis_vox = prepared["selected_axis_vox"]
    seg_long_axis_phys = prepared["selected_axis_phys"]
    rot_matrix = prepared["rot_matrix"]
    target_timepoint_voxels = int(prepared["placed_target_voxels"])
    target_timepoint_actual_volume_mm3 = float(prepared["actual_volume_mm3"])

    print(f"  Generated {len(syn_images)} timepoints.")
    for idx, cnt in enumerate(voxel_counts):
        print(f"    t{idx}  day={dates[idx]:4d}  mask_voxels={cnt}")

    print(f"\n  Smallest timepoint : t{smallest_idx}  (day {dates[smallest_idx]},  {voxel_counts[smallest_idx]} voxels)")
    print(f"  Last timepoint     : t{last_idx}  (day {dates[last_idx]},  {voxel_counts[last_idx]} voxels) ← lollipop QC")
    print(
        f"  Volume target t/p  : {volume_target_timepoint_raw} -> "
        f"t{volume_target_timepoint_index} (day {volume_target_timepoint_day}, "
        f"vox={voxel_counts[volume_target_timepoint_index]})"
    )
    print(f"  Synthetic centroid (syn space) : {np.round(syn_centroid, 2)}")

    print("\n  Orientation strategy comparison:")
    for result in comparison_results:
        axis_str = np.round(result.axis_vox, 4)
        candidate_str = ", ".join(
            f"{np.round(candidate.axis_vox, 4)}:{candidate.score:.4f}"
            for candidate in result.candidates
        )
        status = "LOW_CONF" if result.low_confidence else "ok"
        print(
            f"    {result.method:>16}  selected={axis_str}  "
            f"confidence={result.confidence:.4f}  status={status}  candidates=[{candidate_str}]"
        )
    print(
        f"  Default strategy  : {default_orientation.method}  "
        f"margin={default_orientation.debug['score_margin']:.4f}  "
        f"norm_gap={default_orientation.debug['normalized_gap']:.4f}"
    )
    print(f"  Selected axis     : vox={np.round(seg_long_axis_vox, 4)}  mm={np.round(seg_long_axis_phys, 4)}  (positive side = CPA bulb)")
    print(f"\n  Rotation matrix (synthetic mm axis → MRI physical axis):\n{np.round(rot_matrix, 4)}")

    # ── 6. Embed each timepoint ────────────────────────────────────────────────
    print("\nEmbedding all timepoints…")
    all_embedded_vols: list[np.ndarray] = []
    all_embedded_masks: list[np.ndarray] = []
    timepoint_metrics: list[TimepointMetrics] = []

    # MRI intensity range for scaling synthetic signal
    mri_p1, mri_p99 = float(np.percentile(mri_data, 1)), float(np.percentile(mri_data, 99))
    blend_width_mm = 1.25

    for t_idx, (syn_img_ch, syn_lab_ch) in enumerate(zip(syn_images, syn_labels)):
        syn_img_vol = syn_img_ch[0]   # (gen_size, gen_size, gen_size) float32 in [0,1]
        syn_lab_vol = syn_lab_ch[0]   # same shape, binary

        # Use nearest-neighbor for masks to avoid partial-volume label artifacts.
        placed_mask = rotate_and_translate(
            syn_lab_vol.astype(np.float32),
            rot_matrix, syn_centroid, seg_centroid,
            out_shape=(H, W, D), dst_spacing=voxel_spacing, order=0,
        )
        placed_mask = (placed_mask > 0.5).astype(np.uint8)

        # Rotate + translate image
        placed_img = rotate_and_translate(
            syn_img_vol,
            rot_matrix, syn_centroid, seg_centroid,
            out_shape=(H, W, D), dst_spacing=voxel_spacing, order=1,
        )

        # Scale synthetic intensity to MRI range, then embed
        syn_scaled = placed_img * (mri_p99 - mri_p1) + mri_p1
        emb_vol = mri_data.copy()
        mask_bool = placed_mask > 0
        blend_alpha = feather_mask_alpha(placed_mask, voxel_spacing, blend_width_mm=blend_width_mm)
        emb_vol[mask_bool] = (
            (1.0 - blend_alpha[mask_bool]) * mri_data[mask_bool] +
            blend_alpha[mask_bool] * syn_scaled[mask_bool]
        )

        all_embedded_vols.append(emb_vol)
        all_embedded_masks.append(placed_mask)

        src_voxels = int(np.sum(syn_lab_vol > 0))
        cnt = int(placed_mask.sum())
        ctr_str = "—"
        source_volume_mm3 = float(src_voxels)
        placed_volume_mm3 = float(cnt) * float(np.prod(voxel_spacing))
        retained = float(placed_volume_mm3 / max(source_volume_mm3, 1e-8))
        axis_err_deg: float | None = None
        ctr_mm = float("nan")
        if cnt > 0:
            _, props_emb = largest_component(placed_mask)
            ctr = np.round(np.array(props_emb.centroid), 1)
            ctr_str = str(ctr)
            emb_axes = principal_axes(placed_mask, voxel_spacing)
            axis_err_deg = _angle_deg(emb_axes.long_axis_phys, seg_long_axis_phys)
            ctr_mm = np.linalg.norm((emb_axes.centroid_vox - seg_centroid) * voxel_spacing)
            ctr_str = f"{ctr}  axis_err={axis_err_deg:.2f}deg  ctr_off={ctr_mm:.2f}mm"
        else:
            ctr_mm = float("inf")
        clip_note = ""
        if src_voxels > 0 and cnt < src_voxels:
            if retained < 0.98:
                clip_note = f"  [warn clipped to {retained:.1%}]"
        timepoint_metrics.append(
            TimepointMetrics(
                timepoint_index=t_idx,
                day=int(dates[t_idx]),
                source_voxels=src_voxels,
                placed_voxels=cnt,
                source_volume_mm3=source_volume_mm3,
                placed_volume_mm3=placed_volume_mm3,
                retained_fraction=retained,
                centroid_offset_mm=float(ctr_mm),
                axis_error_deg=None if axis_err_deg is None else float(axis_err_deg),
                clipped=bool(src_voxels > 0 and cnt < src_voxels),
            )
        )
        print(f"    t{t_idx}  day={dates[t_idx]:4d}  placed_voxels={cnt}  centroid≈{ctr_str}{clip_note}")

    # Final selected target-timepoint volume metrics (same objective index for all candidates).
    target_timepoint_voxels = int(np.sum(all_embedded_masks[volume_target_timepoint_index] > 0))
    target_timepoint_actual_volume_mm3 = float(target_timepoint_voxels) * voxel_volume_mm3
    if volume_target_mm3 is not None:
        volume_actual_mm3 = target_timepoint_actual_volume_mm3
        volume_ravd = abs(volume_actual_mm3 - volume_target_mm3) / volume_target_mm3

    # Primary output = smallest timepoint
    emb_vol_small  = all_embedded_vols[smallest_idx]
    emb_mask_small = all_embedded_masks[smallest_idx]

    # Diagnostic: centroid offset for smallest timepoint
    if int(emb_mask_small.sum()) > 0:
        _, props_final = largest_component(emb_mask_small)
        emb_ctr = np.array(props_final.centroid)
        offset_vox = float(np.linalg.norm(emb_ctr - seg_centroid))
        offset_mm_small = float(np.linalg.norm((emb_ctr - seg_centroid) * voxel_spacing))
        print(f"\n  [t{smallest_idx}] Embedded mask centroid : {np.round(emb_ctr, 2)}")
        print(f"  [t{smallest_idx}] Target centroid        : {np.round(seg_centroid, 2)}")
        print(f"  [t{smallest_idx}] Centroid offset        : {offset_vox:.2f} voxels")
    else:
        offset_vox = float("inf")
        offset_mm_small = float("inf")

    case_metrics = validate_embedding_case(
        case_id=case_id,
        seed=seed,
        mri_path=mri_path,
        seg_path=seg_path,
        seg_voxel_count=int(seg_component.sum()),
        seg_volume_mm3=seg_vol_mm3,
        selected_orientation=default_orientation,
        comparison_results=comparison_results,
        selected_axis_phys=seg_long_axis_phys,
        selected_axis_vox=seg_long_axis_vox,
        timepoint_metrics=timepoint_metrics,
        primary_timepoint_index=smallest_idx,
        primary_source_voxels=voxel_counts[smallest_idx],
        primary_placed_voxels=int(emb_mask_small.sum()),
        primary_centroid_offset_vox=offset_vox,
        primary_centroid_offset_mm=offset_mm_small,
        cpa_radius_anatomy_mm=cpa_radius_anatomy_mm,
        cpa_radius_effective_mm=cpa_radius_effective_mm,
        cpa_radius_override_active=cpa_radius_override_active,
        volume_optimization_variable=volume_optimization_variable,
        target_volume_mm3=volume_target_mm3,
        volume_target_timepoint=volume_target_timepoint_raw,
        volume_target_timepoint_index=volume_target_timepoint_index,
        volume_target_timepoint_day=volume_target_timepoint_day,
        target_timepoint_actual_volume_mm3=target_timepoint_actual_volume_mm3,
        target_timepoint_voxels=target_timepoint_voxels,
        actual_volume_mm3=volume_actual_mm3,
        ravd=volume_ravd,
        volume_converged=volume_converged,
        volume_iterations=volume_iterations,
        volume_scale_final=volume_scale_final,
        volume_target_below_min=volume_target_below_min,
        volume_target_above_max_or_clipped=volume_target_above_max_or_clipped,
    )
    metrics_json_path, metrics_csv_path = write_case_reports(case_metrics, out_dir)
    audit_suffix = (
        f"orient={case_metrics.orientation_method} "
        f"conf={case_metrics.orientation_confidence:.3f} "
        f"{'LOW_CONF' if case_metrics.orientation_low_confidence else 'OK'}"
    )

    # ── 7. Save NIfTI outputs ──────────────────────────────────────────────────
    print("\nSaving NIfTI outputs…")

    # Smallest timepoint (primary, backward-compatible filenames)
    vol_path  = out_dir / "embedded_tumor_volume.nii.gz"
    mask_path = out_dir / "embedded_tumor_mask.nii.gz"
    save_like(mri_nib, emb_vol_small.astype(np.float32), vol_path)
    save_like(mri_nib, emb_mask_small.astype(np.uint8), mask_path)
    print(f"  {vol_path}")
    print(f"  {mask_path}")

    # Last (most elongated) timepoint — shows clear lollipop morphology
    emb_vol_late  = all_embedded_vols[last_idx]
    emb_mask_late = all_embedded_masks[last_idx]
    late_vol_path  = out_dir / "embedded_tumor_late_volume.nii.gz"
    late_mask_path = out_dir / "embedded_tumor_late_mask.nii.gz"
    save_like(mri_nib, emb_vol_late.astype(np.float32), late_vol_path)
    save_like(mri_nib, emb_mask_late.astype(np.uint8), late_mask_path)
    print(f"  {late_vol_path}")
    print(f"  {late_mask_path}")

    # Full time series
    for t_idx, (evol, emask) in enumerate(zip(all_embedded_vols, all_embedded_masks)):
        tp_vol  = out_dir / f"embedded_t{t_idx:02d}_volume.nii.gz"
        tp_mask = out_dir / f"embedded_t{t_idx:02d}_mask.nii.gz"
        save_like(mri_nib, evol.astype(np.float32), tp_vol)
        save_like(mri_nib, emask.astype(np.uint8), tp_mask)
    print(f"  Full series: embedded_t00..t{last_idx:02d}_{{volume,mask}}.nii.gz")

    # ── 8. QC PNGs ────────────────────────────────────────────────────────────
    print("\nGenerating QC PNGs…")

    _save_qc_png(
        mri_data, seg_data, emb_vol_small, emb_mask_small,
        seg_centroid, seg_long_axis_vox, mri_p1, mri_p99,
        out_dir / "qc_embedding.png",
        title_suffix=f"t{smallest_idx} (smallest, day {dates[smallest_idx]})",
        audit_suffix=audit_suffix,
    )

    _save_qc_png(
        mri_data, seg_data, emb_vol_late, emb_mask_late,
        seg_centroid, seg_long_axis_vox, mri_p1, mri_p99,
        out_dir / "qc_embedding_late.png",
        title_suffix=f"t{last_idx} (last/elongated, day {dates[last_idx]})",
        audit_suffix=audit_suffix,
    )

    # ── 9. Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EMBEDDING COMPLETE  (lollipop geometry, mm-based IAC dimensions)")
    print(f"  Geometry          : lollipop, canal_axis=c, centered=True")
    print(f"  Voxel spacing     : {np.round(voxel_spacing, 4)} mm")
    print(f"  IAC canal length  : {iac_canal_mm:.1f} mm  = {iac_canal_syn:.1f} syn vox  [literature]")
    print(f"  Porus radius      : {iac_base_r_mm:.1f} mm  = {iac_base_r_syn:.1f} syn vox  [literature]")
    print(f"  Fundus radius     : {iac_apex_r_mm:.1f} mm  = {iac_apex_r_syn:.1f} syn vox  [literature]")
    print(
        f"  CPA bulb radius   : anatomy={cpa_radius_anatomy_mm:.1f} mm, "
        f"effective={cpa_radius_effective_mm:.1f} mm"
        f"{' [override active]' if cpa_radius_override_active else ''}"
    )
    print(f"  Seg volume        : {seg_vol_mm3:.0f} mm³")
    print(f"  Seed              : {seed}")
    print(f"  Dates             : {dates}")
    print(f"  Growth            : {growth}")
    print(f"  Canal growth x    : {canal_growth_scale:.2f}")
    print(f"  Bulb growth x     : {bulb_growth_scale:.2f}")
    if target_tumor_volume_mm3 is not None:
        print(f"  Target vol mm3    : {target_tumor_volume_mm3:.1f}")
        print(f"  Vol opt variable  : {volume_optimization_variable}")
        print(
            f"  Target vol t/p    : {volume_target_timepoint_raw} -> "
            f"t{volume_target_timepoint_index} day={volume_target_timepoint_day}"
        )
        print(f"  Target t/p voxels : {target_timepoint_voxels}")
        print(f"  Actual vol mm3    : {volume_actual_mm3:.1f}" if volume_actual_mm3 is not None else "  Actual vol mm3    : n/a")
        print(f"  RAVD              : {volume_ravd:.4f}" if volume_ravd is not None else "  RAVD              : n/a")
        print(f"  Vol converged     : {volume_converged}")
        print(f"  Vol iterations    : {volume_iterations}")
        print(f"  Vol scale final   : {volume_scale_final:.4f}" if volume_scale_final is not None else "  Vol scale final   : n/a")
    if t0_volume_fraction_of_seg is not None:
        print(f"  T0 vol frac/seg   : {t0_volume_fraction_of_seg:.3f}")
        print(f"  T0 calib scale    : {calibration_scale:.3f}" if calibration_scale is not None else "  T0 calib scale    : n/a")
    print(f"  Smallest t/p used : t{smallest_idx}  (day {dates[smallest_idx]},  "
          f"{voxel_counts[smallest_idx]} voxels)")
    print(f"  Last t/p (QC)     : t{last_idx}  (day {dates[last_idx]},  "
          f"{voxel_counts[last_idx]} voxels)")
    print(f"  Seg centroid      : {np.round(seg_centroid, 2)}")
    print(f"  Seg long axis     : vox={np.round(seg_long_axis_vox, 4)}  mm={np.round(seg_long_axis_phys, 4)}")
    print(f"  Synthetic centroid: {np.round(syn_centroid, 2)}")
    print(f"  Validation        : warnings={len(case_metrics.warnings)}  hard_failures={len(case_metrics.hard_failures)}")
    if case_metrics.warnings:
        print(f"  Warning codes     : {[finding.code for finding in case_metrics.findings if finding.severity == 'warning']}")
    if case_metrics.hard_failures:
        print(f"  Failure codes     : {[finding.code for finding in case_metrics.findings if finding.severity == 'hard_failure']}")
    print(f"\nOutputs:")
    print(f"  {vol_path}")
    print(f"  {mask_path}")
    print(f"  {late_vol_path}")
    print(f"  {late_mask_path}")
    print(f"  {out_dir}/embedded_t00..t{last_idx:02d}_{{volume,mask}}.nii.gz")
    print(f"  {out_dir}/qc_embedding.png")
    print(f"  {out_dir}/qc_embedding_late.png")
    print(f"  {metrics_json_path}")
    print(f"  {metrics_csv_path}")
    if validate_anisotropy:
        run_anisotropy_validation()
    print("=" * 65)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    _downloads = Path.home() / "Downloads"
    _out = Path(__file__).resolve().parent / "embedding_outputs"

    parser = argparse.ArgumentParser(
        description="Anatomically guided lollipop tumor embedding"
    )
    parser.add_argument(
        "--mri",
        default=str(_downloads / "147_0_0_t2_thin_image_coregistered.nii.gz"),
        help="Path to the real MRI NIfTI file",
    )
    parser.add_argument(
        "--seg",
        default=str(_downloads / "147_0_0_t2_thin_R_VS__uvauser2__coregistered.nii.gz"),
        help="Path to the binary segmentation NIfTI file",
    )
    parser.add_argument("--out_dir", default=str(_out), help="Output directory")
    parser.add_argument("--gen_size", type=int, default=128,
                        help="Synthetic tumor generation volume size (isotropic)")
    parser.add_argument("--rad_min",  type=int, default=4,  help="Min initial tumor radius")
    parser.add_argument("--rad_max",  type=int, default=30, help="Max initial tumor radius")
    parser.add_argument("--growth",   default="steady",
                        choices=["steady", "decreasing", "fat-tailed", "stable"])
    parser.add_argument("--canal_growth_scale", type=float, default=1.0,
                        help="Multiplier for lollipop canal/stem growth increments")
    parser.add_argument("--bulb_growth_scale", type=float, default=1.0,
                        help="Multiplier for lollipop CPA bulb growth increments")
    parser.add_argument("--target_tumor_volume_mm3", type=float, default=None,
                        help="Optional target tumor volume (mm^3) for binary-search calibration")
    parser.add_argument("--volume_target_timepoint", type=str, default="first",
                        help="Timepoint objective for volume targeting: first, last, or integer index string")
    parser.add_argument("--volume_ravd_tolerance", type=float, default=0.05,
                        help="RAVD tolerance for volume calibration loop")
    parser.add_argument("--volume_max_iterations", type=int, default=10,
                        help="Maximum binary-search iterations for volume calibration")
    parser.add_argument("--t0_volume_fraction_of_seg", type=float, default=None,
                        help="Optional target fraction of real seg volume for synthetic t0 calibration")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional deterministic seed for synthetic series generation")
    parser.add_argument(
        "--validate_anisotropy",
        action="store_true",
        help="Run a synthetic anisotropic placement validation comparing legacy vs corrected transforms",
    )
    args = parser.parse_args()

    main(
        mri_path=Path(args.mri),
        seg_path=Path(args.seg),
        out_dir=Path(args.out_dir),
        gen_size=args.gen_size,
        rad_min=args.rad_min,
        rad_max=args.rad_max,
        growth=args.growth,
        canal_growth_scale=args.canal_growth_scale,
        bulb_growth_scale=args.bulb_growth_scale,
        target_tumor_volume_mm3=args.target_tumor_volume_mm3,
        volume_target_timepoint=args.volume_target_timepoint,
        volume_ravd_tolerance=args.volume_ravd_tolerance,
        volume_max_iterations=args.volume_max_iterations,
        t0_volume_fraction_of_seg=args.t0_volume_fraction_of_seg,
        validate_anisotropy=args.validate_anisotropy,
        seed=args.seed,
    )
