from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.extract_real_tumor_features import (
    _extract_one,
    _maybe_align_principal_axis_sign,
    _parse_axis_arg,
    _rows_from_input,
    _safe_case_id_from_path,
    _validate_spacing,
)


pytestmark = pytest.mark.fast


def test_safe_case_id_from_path_handles_nii_gz_suffix():
    assert _safe_case_id_from_path(Path("126_5_1607_seg.nii.gz")) == "126_5_1607_seg"
    assert _safe_case_id_from_path(Path("mask.nii")) == "mask"


def test_validate_spacing_accepts_first_three_positive_values():
    spacing = _validate_spacing((0.5, 0.5, 1.0, 2.0), Path("seg.nii.gz"))

    assert np.allclose(spacing, np.array([0.5, 0.5, 1.0]))

    with pytest.raises(ValueError, match="non-positive"):
        _validate_spacing((1.0, 0.0, 1.0), Path("seg.nii.gz"))


def test_parse_axis_arg_normalizes_and_rejects_bad_values():
    assert np.allclose(_parse_axis_arg("0,0,2", "axis"), np.array([0.0, 0.0, 1.0]))
    assert _parse_axis_arg(None, "axis") is None

    with pytest.raises(ValueError, match="3 comma-separated"):
        _parse_axis_arg("0,1", "axis")
    with pytest.raises(ValueError, match="finite and non-zero"):
        _parse_axis_arg("0,0,0", "axis")


def test_maybe_align_principal_axis_sign_prefers_reference_mm():
    axis_vox = np.array([1.0, 0.0, 0.0])
    axis_mm = np.array([1.0, 0.0, 0.0])

    aligned_vox, aligned_mm, aligned = _maybe_align_principal_axis_sign(
        axis_vox=axis_vox,
        axis_mm=axis_mm,
        reference_axis_vox=np.array([1.0, 0.0, 0.0]),
        reference_axis_mm=np.array([-1.0, 0.0, 0.0]),
    )

    assert aligned is True
    assert np.allclose(aligned_vox, -axis_vox)
    assert np.allclose(aligned_mm, -axis_mm)


def test_rows_from_input_deduplicates_csv_rows(tmp_path: Path):
    seg_path = tmp_path / "case_seg.nii.gz"
    seg_path.write_text("placeholder")
    input_csv = tmp_path / "cases.csv"
    input_csv.write_text(
        "case_id,seg_path\n"
        f"case-a,{seg_path}\n"
        f"case-a,{seg_path}\n"
    )
    args = Namespace(seg_path=None, glob=None, input_csv=str(input_csv), seg_col="seg_path", case_id_col="case_id")

    rows = _rows_from_input(args)

    assert rows == [("case-a", seg_path.resolve())]


@pytest.mark.integration
def test_extract_one_reports_empty_mask_without_failing(tmp_path: Path):
    seg_path = tmp_path / "empty_seg.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.uint8), np.diag([0.5, 0.5, 1.0, 1.0])), str(seg_path))

    feature = _extract_one(seg_path=seg_path, case_id="empty", reference_axis_vox=None, reference_axis_mm=None)

    assert feature["case_id"] == "empty"
    assert feature["empty_mask"] is True
    assert feature["connected_component_count"] == 0
    assert feature["volume_mm3"] == pytest.approx(0.0)
