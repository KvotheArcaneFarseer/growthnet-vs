from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from projects.vivit.src.data.synthetic import create_synthetic_time_3d
from scripts.generate_synthetic_lollipop_cohort import (
    _calibrate_scale,
    _parse_spacing,
    _read_targets,
    _safe_case_id,
    _stable_seed,
    _target_shape_controls,
    _volume_mm3,
)


pytestmark = pytest.mark.fast


def test_synthetic_cohort_seed_and_case_id_helpers_are_stable():
    assert _stable_seed(123, "case-a") == _stable_seed(123, "case-a")
    assert _stable_seed(123, "case-a") != _stable_seed(124, "case-a")
    assert _safe_case_id(" patient 1/visit:2 ") == "patient_1_visit_2"
    assert _safe_case_id(" ") == "case"


def test_parse_spacing_rejects_invalid_values():
    assert _parse_spacing("1.0, 0.5, 2") == (1.0, 0.5, 2.0)

    with pytest.raises(ValueError, match="must be 'sx,sy,sz'"):
        _parse_spacing("1,2")
    with pytest.raises(ValueError, match="must be > 0"):
        _parse_spacing("1,0,1")


def test_read_targets_validates_manifest_columns_and_rows(tmp_path: Path):
    targets = tmp_path / "targets.csv"
    targets.write_text("case_id,target_volume_mm3\ncase-a,125\n")

    assert _read_targets(targets, case_col="case_id", vol_col="target_volume_mm3") == [
        {"case_id": "case-a", "target_volume_mm3": "125"}
    ]

    empty = tmp_path / "empty.csv"
    empty.write_text("case_id,target_volume_mm3\n")
    with pytest.raises(ValueError, match="No rows"):
        _read_targets(empty, case_col="case_id", vol_col="target_volume_mm3")


def test_target_shape_controls_are_bounded_and_monotone():
    tiny_tinyness, tiny_maturity = _target_shape_controls(10.0)
    large_tinyness, large_maturity = _target_shape_controls(5000.0)

    assert tiny_tinyness == pytest.approx(1.0)
    assert tiny_maturity == pytest.approx(0.0)
    assert large_tinyness == pytest.approx(0.0)
    assert large_maturity == pytest.approx(1.0)


def test_volume_mm3_uses_voxel_count_and_spacing_product():
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:2] = 1

    assert _volume_mm3(mask, voxel_volume_mm3=2.5) == pytest.approx(10.0)


def test_create_synthetic_time_3d_is_deterministic_for_fixed_generator_seed():
    kwargs = dict(
        height=32,
        width=32,
        depth=32,
        dates=[0, 30],
        rotation_degrees=[10, 20, 30],
        rad_max=6,
        rad_min=2,
        noise_max=0.02,
        num_seg_classes=5,
        channel_dim=None,
        growth="steady",
        growth_direction="a",
        off_axis_growth_p=0.0,
        geometry_mode="lollipop",
        canal_axis="c",
        centered=True,
    )

    images_a, labels_a = create_synthetic_time_3d(random_state=np.random.default_rng(123), **kwargs)
    images_b, labels_b = create_synthetic_time_3d(random_state=np.random.default_rng(123), **kwargs)
    images_c, labels_c = create_synthetic_time_3d(random_state=np.random.default_rng(124), **kwargs)

    assert all(np.array_equal(a, b) for a, b in zip(labels_a, labels_b))
    assert all(np.allclose(a, b) for a, b in zip(images_a, images_b))
    assert any(not np.array_equal(a, c) for a, c in zip(labels_a, labels_c))


@pytest.mark.integration
def test_calibrate_scale_reaches_small_local_target_volume():
    target_volume_mm3 = 32.0

    mask, scale, realized_volume, n_iters = _calibrate_scale(
        target_volume_mm3=target_volume_mm3,
        case_seed=12345,
        canal_axis="c",
        rotation_zyx_deg=[0, 0, 0],
        voxel_volume_mm3=1.0,
        tolerance_frac=0.35,
        max_iters=8,
        min_scale_vox=0.25,
        max_scale_vox=12.0,
    )

    assert mask.dtype == np.uint8
    assert int(mask.sum()) > 0
    assert scale > 0.0
    assert n_iters <= 8
    assert abs(realized_volume - target_volume_mm3) / target_volume_mm3 <= 0.35
