from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from embed_tumor import (
    OrientationCandidate,
    OrientationResult,
    TimepointMetrics,
    ValidationThresholds,
    _angle_deg,
    _axis_vox_to_phys,
    _dice_score,
    _is_monotone_non_decreasing,
    _orientation_confidence,
    principal_axes,
    rotate_and_translate,
    stable_seed_from_case,
    validate_embedding_case,
    write_case_reports,
)


pytestmark = pytest.mark.fast


def _orientation_result(
    confidence: float = 0.5,
    score_margin: float = 0.2,
    axis: np.ndarray | None = None,
    method: str = "test_strategy",
) -> OrientationResult:
    axis = np.asarray([1.0, 0.0, 0.0] if axis is None else axis, dtype=np.float64)
    return OrientationResult(
        axis_vox=axis,
        method=method,
        confidence=confidence,
        low_confidence=False,
        candidates=[
            OrientationCandidate(axis_vox=axis, score=1.0, label="positive", extras={}),
            OrientationCandidate(axis_vox=-axis, score=1.0 - score_margin, label="negative", extras={}),
        ],
        debug={
            "best_score": 1.0,
            "worst_score": 1.0 - score_margin,
            "score_margin": score_margin,
            "normalized_gap": confidence,
        },
    )


def test_stable_seed_from_case_is_deterministic_and_32_bit():
    seed_a = stable_seed_from_case("126_5_1607")
    seed_b = stable_seed_from_case("126_5_1607")
    seed_c = stable_seed_from_case("132_1_148")

    assert seed_a == seed_b
    assert seed_a != seed_c
    assert 0 <= seed_a < 2**32


def test_axis_vox_to_phys_uses_spacing_and_rejects_zero_axis():
    axis_phys = _axis_vox_to_phys(np.array([1.0, 1.0, 0.0]), np.array([1.0, 3.0, 1.0]))

    expected = np.array([1.0, 3.0, 0.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(axis_phys, expected)

    with pytest.raises(ValueError, match="zero-length"):
        _axis_vox_to_phys(np.zeros(3), np.ones(3))


def test_angle_deg_is_unsigned_for_axis_directions():
    assert _angle_deg(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert _angle_deg(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])) == pytest.approx(90.0)


def test_principal_axes_reports_voxel_and_physical_axes_separately():
    mask = np.zeros((11, 11, 11), dtype=np.uint8)
    # Voxel extent is longest along x, but anisotropic y spacing makes physical
    # extent longest along y. This locks the dual-space convention.
    mask[3:8, 4:7, 5:8] = 1
    spacing = np.array([1.0, 3.0, 1.0], dtype=np.float64)

    axes = principal_axes(mask, spacing)

    assert np.argmax(np.abs(axes.long_axis_vox)) == 0
    assert np.argmax(np.abs(axes.long_axis_phys)) == 1
    assert np.allclose(axes.centroid_vox, np.array([5.0, 5.0, 6.0]))


def test_rotate_and_translate_honors_anisotropic_target_spacing():
    vol = np.zeros((9, 9, 9), dtype=np.float32)
    vol[4, 4, 4] = 1.0

    placed = rotate_and_translate(
        vol=vol,
        rot_mat_phys=np.eye(3),
        src_centroid=np.array([4.0, 4.0, 4.0]),
        dst_centroid=np.array([6.0, 3.0, 2.0]),
        out_shape=(10, 10, 10),
        dst_spacing=np.array([1.0, 2.0, 1.0]),
        order=0,
    )

    coords = np.argwhere(placed > 0.5)
    assert coords.tolist() == [[6, 3, 2]]


def test_dice_score_handles_empty_and_partial_masks():
    empty = np.zeros((3, 3, 3), dtype=np.uint8)
    mask_a = empty.copy()
    mask_b = empty.copy()
    mask_a[1, 1, 1] = 1
    mask_b[1, 1, 1] = 1
    mask_b[1, 1, 2] = 1

    assert _dice_score(empty, empty) == pytest.approx(1.0)
    assert _dice_score(mask_a, mask_b) == pytest.approx(2.0 / 3.0)


def test_orientation_confidence_flags_ambiguous_scores():
    clear_candidates = [
        OrientationCandidate(np.array([1.0, 0.0, 0.0]), 0.9, "positive", {}),
        OrientationCandidate(np.array([-1.0, 0.0, 0.0]), 0.1, "negative", {}),
    ]
    ambiguous_candidates = [
        OrientationCandidate(np.array([1.0, 0.0, 0.0]), 0.51, "positive", {}),
        OrientationCandidate(np.array([-1.0, 0.0, 0.0]), 0.50, "negative", {}),
    ]

    clear_confidence, clear_low, clear_debug = _orientation_confidence(clear_candidates)
    ambiguous_confidence, ambiguous_low, ambiguous_debug = _orientation_confidence(ambiguous_candidates)

    assert clear_low is False
    assert clear_confidence > ambiguous_confidence
    assert ambiguous_low is True
    assert ambiguous_debug["score_margin"] == pytest.approx(0.01)
    assert clear_debug["best_score"] == pytest.approx(0.9)


def test_monotone_non_decreasing_allows_small_drops_only():
    assert _is_monotone_non_decreasing([100, 99, 120], tolerance_fraction=0.02) is True
    assert _is_monotone_non_decreasing([100, 90, 120], tolerance_fraction=0.02) is False


def test_validate_embedding_case_emits_threshold_findings():
    metrics = validate_embedding_case(
        case_id="threshold-case",
        seed=1,
        mri_path=Path("/local/mri.nii.gz"),
        seg_path=Path("/local/seg.nii.gz"),
        seg_voxel_count=100,
        seg_volume_mm3=100.0,
        selected_orientation=_orientation_result(confidence=0.01, score_margin=0.01),
        comparison_results=[_orientation_result(confidence=0.01, score_margin=0.01)],
        selected_axis_phys=np.array([1.0, 0.0, 0.0]),
        selected_axis_vox=np.array([1.0, 0.0, 0.0]),
        timepoint_metrics=[
            TimepointMetrics(
                timepoint_index=0,
                day=0,
                source_voxels=100,
                placed_voxels=70,
                source_volume_mm3=100.0,
                placed_volume_mm3=70.0,
                retained_fraction=0.70,
                centroid_offset_mm=4.0,
                axis_error_deg=70.0,
                clipped=True,
            ),
            TimepointMetrics(
                timepoint_index=1,
                day=180,
                source_voxels=120,
                placed_voxels=60,
                source_volume_mm3=120.0,
                placed_volume_mm3=60.0,
                retained_fraction=0.50,
                centroid_offset_mm=4.0,
                axis_error_deg=70.0,
                clipped=True,
            ),
        ],
        primary_timepoint_index=0,
        primary_source_voxels=100,
        primary_placed_voxels=70,
        primary_centroid_offset_vox=4.0,
        primary_centroid_offset_mm=4.0,
        cpa_radius_anatomy_mm=3.0,
        cpa_radius_effective_mm=3.0,
        cpa_radius_override_active=False,
        thresholds=ValidationThresholds(),
    )

    codes = {finding.code for finding in metrics.findings}
    assert {"centroid_offset_fail", "axis_error_fail", "retained_fraction_fail"}.issubset(codes)
    assert "orientation_low_score_margin" in codes
    assert "orientation_low_normalized_gap" in codes
    assert metrics.hard_failures
    assert metrics.warnings


def test_write_case_reports_preserves_json_and_csv_schema(tmp_path: Path):
    metrics = validate_embedding_case(
        case_id="schema-case",
        seed=7,
        mri_path=Path("/local/mri.nii.gz"),
        seg_path=Path("/local/seg.nii.gz"),
        seg_voxel_count=10,
        seg_volume_mm3=10.0,
        selected_orientation=_orientation_result(confidence=0.5, score_margin=0.2),
        comparison_results=[_orientation_result(confidence=0.5, score_margin=0.2)],
        selected_axis_phys=np.array([1.0, 0.0, 0.0]),
        selected_axis_vox=np.array([1.0, 0.0, 0.0]),
        timepoint_metrics=[
            TimepointMetrics(0, 0, 10, 10, 10.0, 10.0, 1.0, 0.0, 0.0, False),
        ],
        primary_timepoint_index=0,
        primary_source_voxels=10,
        primary_placed_voxels=10,
        primary_centroid_offset_vox=0.0,
        primary_centroid_offset_mm=0.0,
        cpa_radius_anatomy_mm=2.0,
        cpa_radius_effective_mm=2.0,
        cpa_radius_override_active=False,
    )

    json_path, csv_path = write_case_reports(metrics, tmp_path)

    payload = json.loads(json_path.read_text())
    with csv_path.open(newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert payload["case_id"] == "schema-case"
    assert payload["findings"] == [asdict(finding) for finding in metrics.findings]
    assert csv_row["case_id"] == "schema-case"
    assert "strategy_results" in csv_row
