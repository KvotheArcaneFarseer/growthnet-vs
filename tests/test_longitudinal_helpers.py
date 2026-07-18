from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.generate_synthetic_longitudinal_dataset import (
    BACKGROUND_COLUMNS,
    TIMELINE_COLUMNS,
    TIMEPOINT_COLUMNS,
    _growth_mode,
    _load_backgrounds,
    _qc_mask,
    _read_csv_rows,
    _safe_id,
    _stable_seed,
)


pytestmark = pytest.mark.fast


def test_longitudinal_constants_match_four_visit_schema():
    assert TIMEPOINT_COLUMNS == ("T1_volume_mm3", "T2_volume_mm3", "T3_volume_mm3", "T4_volume_mm3")
    for required in ("patient_id", "background_mri_id", "growth_label"):
        assert required in TIMELINE_COLUMNS
    assert BACKGROUND_COLUMNS == ("background_mri_id", "mri_path", "seg_path")


def test_safe_id_and_stable_seed_are_deterministic():
    assert _safe_id(" patient 1 / visit ") == "patient_1_visit"
    assert _safe_id(" ") == "item"
    assert _stable_seed(10, "patient-a", "T1") == _stable_seed(10, "patient-a", "T1")
    assert _stable_seed(10, "patient-a", "T1") != _stable_seed(10, "patient-a", "T2")


def test_read_csv_rows_validates_headers_and_nonempty_input(tmp_path: Path):
    timeline = tmp_path / "timeline.csv"
    timeline.write_text(
        "patient_id,background_mri_id,T1_volume_mm3,T2_volume_mm3,T3_volume_mm3,T4_volume_mm3,growth_label\n"
        "p1,bg1,10,20,30,40,growing\n"
    )

    rows = _read_csv_rows(timeline, TIMELINE_COLUMNS)

    assert rows[0]["patient_id"] == "p1"

    missing = tmp_path / "missing.csv"
    missing.write_text("patient_id\np1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _read_csv_rows(missing, TIMELINE_COLUMNS)

    empty = tmp_path / "empty.csv"
    empty.write_text(",".join(TIMELINE_COLUMNS) + "\n")
    with pytest.raises(ValueError, match="No rows found"):
        _read_csv_rows(empty, TIMELINE_COLUMNS)


def test_load_backgrounds_rejects_duplicate_ids(tmp_path: Path):
    csv_path = tmp_path / "backgrounds.csv"
    csv_path.write_text(
        "background_mri_id,mri_path,seg_path\n"
        "bg1,/tmp/mri-a.nii.gz,/tmp/seg-a.nii.gz\n"
        "bg1,/tmp/mri-b.nii.gz,/tmp/seg-b.nii.gz\n"
    )

    with pytest.raises(ValueError, match="Duplicate background_mri_id"):
        _load_backgrounds(csv_path)


def test_growth_mode_maps_only_supported_labels():
    assert _growth_mode(" stable ") == "stable"
    assert _growth_mode("Growing") == "steady"
    with pytest.raises(ValueError, match="Unsupported growth_label"):
        _growth_mode("rapid")


@pytest.mark.integration
def test_qc_mask_reports_volume_and_component_failures(tmp_path: Path):
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask[6, 6, 6] = 1
    seg_path = tmp_path / "two_component_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(seg_path))

    qc = _qc_mask(mask_path=seg_path, target_volume_mm3=8.0, tolerance=0.05)

    assert qc["synthetic_volume_mm3"] == pytest.approx(9.0)
    assert qc["connected_components"] == 2
    assert qc["qc_pass"] is False
    assert "connected_components=2" in qc["qc_failure_reason"]
    assert "relative_volume_error>0.05" in qc["qc_failure_reason"]
