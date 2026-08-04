"""Adapters for GrowthNet synthetic longitudinal metadata outputs.

The generator writes a flat artifact layout with ``metadata.csv`` plus image and
mask paths. This module converts that metadata into the sequence-record shape
used by the ViViT temporal data path without requiring generated files to be
copied into the legacy split-folder layout.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


REQUIRED_METADATA_COLUMNS = (
    "patient_id",
    "timepoint",
    "visit_day",
    "variant_id",
    "image_path",
    "mask_path",
)


def _read_metadata_rows(metadata_csv: Path) -> List[Dict[str, str]]:
    with metadata_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV is missing a header row: {metadata_csv}")
        missing = sorted(set(REQUIRED_METADATA_COLUMNS).difference(reader.fieldnames))
        if missing:
            raise ValueError(f"{metadata_csv} is missing required columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No metadata rows found in {metadata_csv}")
    return rows


def _resolve_metadata_path(metadata_csv: Path, value: str) -> Path:
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = metadata_csv.parent / path
    return path


def build_synthetic_longitudinal_sequence_index(
    metadata_csv: str | Path,
    require_paths: bool = True,
) -> List[Dict[str, Any]]:
    """Build ViViT-compatible sequence records from generator metadata.

    Multi-variant outputs are grouped by ``(patient_id, variant_id)`` so each
    variant remains a separate temporal trajectory.
    """
    metadata_path = Path(metadata_csv)
    rows = _read_metadata_rows(metadata_path)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for row in rows:
        patient_id = str(row.get("patient_id", "")).strip()
        variant_id = str(row.get("variant_id", "")).strip() or "V01"
        timepoint = str(row.get("timepoint", "")).strip()
        if not patient_id:
            raise ValueError("metadata row has blank patient_id")
        if not timepoint:
            raise ValueError(f"metadata row for patient {patient_id!r} has blank timepoint")
        try:
            visit_day = float(row.get("visit_day", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metadata row for {patient_id}/{variant_id}/{timepoint} has invalid visit_day") from exc

        image_path = _resolve_metadata_path(metadata_path, row.get("image_path", ""))
        mask_path = _resolve_metadata_path(metadata_path, row.get("mask_path", ""))
        if require_paths:
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing synthetic image path: {image_path}")
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing synthetic mask path: {mask_path}")

        grouped.setdefault((patient_id, variant_id), []).append(
            {
                "timepoint": timepoint,
                "visit_day": visit_day,
                "scan_id": f"{patient_id}_{variant_id}_{timepoint}",
                "image_path": str(image_path),
                "label_path": str(mask_path),
            }
        )

    sequences: List[Dict[str, Any]] = []
    for (patient_id, variant_id), scans in sorted(grouped.items()):
        seen_timepoints = set()
        for scan in scans:
            timepoint = scan["timepoint"]
            if timepoint in seen_timepoints:
                raise ValueError(f"Duplicate timepoint {timepoint!r} for {patient_id}/{variant_id}")
            seen_timepoints.add(timepoint)

        scans = sorted(scans, key=lambda item: (item["visit_day"], item["timepoint"]))
        visit_days = [scan["visit_day"] for scan in scans]
        first_day = min(visit_days)
        sequences.append(
            {
                "patient_id": f"{patient_id}__{variant_id}",
                "source_patient_id": patient_id,
                "variant_id": variant_id,
                "scan_ids": [scan["scan_id"] for scan in scans],
                "timepoints": [scan["timepoint"] for scan in scans],
                "dates": [day - first_day for day in visit_days],
                "visit_days": visit_days,
                "image_paths": [scan["image_path"] for scan in scans],
                "label_paths": [scan["label_path"] for scan in scans],
            }
        )
    return sequences


def load_synthetic_longitudinal_sequences(
    metadata_csv: str | Path,
    require_labels: bool = True,
) -> List[Dict[str, Any]]:
    """Load synthetic longitudinal sequences as ViViT temporal records."""
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

    loader = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
        ]
    )
    records: List[Dict[str, Any]] = []
    for indexed in build_synthetic_longitudinal_sequence_index(metadata_csv, require_paths=True):
        images: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for image_path, label_path in zip(indexed["image_paths"], indexed["label_paths"]):
            data_dict = {"image": image_path, "label": label_path}
            out = loader(data_dict)
            images.append(np.asarray(out["image"]))
            if "label" in out:
                labels.append(np.asarray(out["label"]))
            elif require_labels:
                raise FileNotFoundError(f"Missing synthetic label for image {image_path}")
            else:
                labels.append(np.zeros_like(images[-1], dtype=images[-1].dtype))

        record = dict(indexed)
        record["images"] = images
        record["labels"] = labels
        records.append(record)
    return records
