from pathlib import Path

import pytest

from projects.vivit.src.data.synthetic_longitudinal_loader import (
    build_synthetic_longitudinal_sequence_index,
)


pytestmark = pytest.mark.fast


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "patient_id,timepoint,visit_day,variant_id,image_path,mask_path\n"
        + "\n".join(
            ",".join(
                [
                    row["patient_id"],
                    row["timepoint"],
                    row["visit_day"],
                    row["variant_id"],
                    row["image_path"],
                    row["mask_path"],
                ]
            )
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_sequence_index_groups_variants_as_separate_trajectories(tmp_path: Path):
    image_a = tmp_path / "images" / "p1_v1_t1.nii.gz"
    image_b = tmp_path / "images" / "p1_v1_t2.nii.gz"
    image_c = tmp_path / "images" / "p1_v2_t1.nii.gz"
    mask_a = tmp_path / "masks" / "p1_v1_t1.nii.gz"
    mask_b = tmp_path / "masks" / "p1_v1_t2.nii.gz"
    mask_c = tmp_path / "masks" / "p1_v2_t1.nii.gz"
    for path in (image_a, image_b, image_c, mask_a, mask_b, mask_c):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "patient_id": "P001",
                "timepoint": "T2",
                "visit_day": "365.25",
                "variant_id": "V01",
                "image_path": "images/p1_v1_t2.nii.gz",
                "mask_path": "masks/p1_v1_t2.nii.gz",
            },
            {
                "patient_id": "P001",
                "timepoint": "T1",
                "visit_day": "0",
                "variant_id": "V01",
                "image_path": "images/p1_v1_t1.nii.gz",
                "mask_path": "masks/p1_v1_t1.nii.gz",
            },
            {
                "patient_id": "P001",
                "timepoint": "T1",
                "visit_day": "0",
                "variant_id": "V02",
                "image_path": "images/p1_v2_t1.nii.gz",
                "mask_path": "masks/p1_v2_t1.nii.gz",
            },
        ],
    )

    sequences = build_synthetic_longitudinal_sequence_index(metadata)

    assert [sequence["patient_id"] for sequence in sequences] == ["P001__V01", "P001__V02"]
    assert sequences[0]["source_patient_id"] == "P001"
    assert sequences[0]["variant_id"] == "V01"
    assert sequences[0]["timepoints"] == ["T1", "T2"]
    assert sequences[0]["dates"] == [0.0, 365.25]
    assert sequences[0]["scan_ids"] == ["P001_V01_T1", "P001_V01_T2"]
    assert sequences[1]["timepoints"] == ["T1"]


def test_sequence_index_rejects_duplicate_variant_timepoints(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "patient_id": "P001",
                "timepoint": "T1",
                "visit_day": "0",
                "variant_id": "V01",
                "image_path": "a.nii.gz",
                "mask_path": "a_mask.nii.gz",
            },
            {
                "patient_id": "P001",
                "timepoint": "T1",
                "visit_day": "0",
                "variant_id": "V01",
                "image_path": "b.nii.gz",
                "mask_path": "b_mask.nii.gz",
            },
        ],
    )

    with pytest.raises(ValueError, match="Duplicate timepoint"):
        build_synthetic_longitudinal_sequence_index(metadata, require_paths=False)


def test_sequence_index_validates_required_columns(tmp_path: Path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("patient_id,timepoint\nP001,T1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        build_synthetic_longitudinal_sequence_index(metadata, require_paths=False)
