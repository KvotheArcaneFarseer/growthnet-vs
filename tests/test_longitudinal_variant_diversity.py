from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


pytestmark = pytest.mark.fast


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "clinical_growth_law_validation"
    / "measure_variant_diversity.py"
)
SPEC = importlib.util.spec_from_file_location("measure_variant_diversity", SCRIPT_PATH)
variant_diversity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(variant_diversity)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patient_id",
                "timepoint",
                "variant_id",
                "target_volume_mm3",
                "mask_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_mask(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.uint8), np.eye(4)), str(path))


def test_variant_diversity_measures_pairwise_dice(tmp_path: Path):
    mask_a = np.zeros((6, 6, 6), dtype=np.uint8)
    mask_b = np.zeros((6, 6, 6), dtype=np.uint8)
    mask_a[1:3, 1:3, 1:3] = 1
    mask_b[2:4, 1:3, 1:3] = 1
    path_a = tmp_path / "p1_t1_v1.nii.gz"
    path_b = tmp_path / "p1_t1_v2.nii.gz"
    _write_mask(path_a, mask_a)
    _write_mask(path_b, mask_b)
    metadata = tmp_path / "metadata.csv"
    _write_csv(
        metadata,
        [
            {
                "patient_id": "P001",
                "timepoint": "T1",
                "variant_id": "V01",
                "target_volume_mm3": "8",
                "mask_path": str(path_a),
            },
            {
                "patient_id": "P001",
                "timepoint": "T1",
                "variant_id": "V02",
                "target_volume_mm3": "8",
                "mask_path": str(path_b),
            },
        ],
    )

    rows = variant_diversity.measure_variant_diversity(metadata)

    assert len(rows) == 1
    assert rows[0]["patient_id"] == "P001"
    assert rows[0]["timepoint"] == "T1"
    assert rows[0]["variant_id_a"] == "V01"
    assert rows[0]["variant_id_b"] == "V02"
    assert rows[0]["status"] == "OK"
    assert rows[0]["voxel_count_a"] == 8
    assert rows[0]["voxel_count_b"] == 8
    assert rows[0]["dice"] == pytest.approx(0.5)
