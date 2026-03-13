"""
Temporal Medical Image Data Loader

This module provides utilities for loading and organizing temporal medical imaging data
(e.g., longitudinal brain scans) from a structured directory hierarchy. It handles:

- Parsing of patient scan identifiers with temporal information
- Resolution of image and label files from nested directory structures
- Loading multi-timepoint sequences using MONAI transforms
- Construction of train/validation/test splits from JSON specifications

The primary entry point is `load_temporal_splits_from_json()`, which expects a directory
structure like:
    <root>/
        train/
            <scan_id>/
                <scan_id>_T1_pre.nii.gz
                <scan_id>_T1_seg.nii.gz
        val/
            <scan_id>/...
        test/
            <scan_id>/...
        train_val_test_split.json

The JSON file specifies which patients belong to each split, and the loader assembles
temporal sequences for each patient by sorting scans by acquisition date.

Primarily authored by GPT-5.
"""

from __future__ import annotations
import os
import json
import re
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd
)

# -----------------------------
# Helpers & parsing
# -----------------------------

def _as_path(p: Union[str, os.PathLike]) -> Path:
    """
    Convert a string or PathLike object to a Path object.
    
    Args:
        p : Union[str, os.PathLike] : Path-like object to convert
    
    Returns:
        path : pathlib.Path : Converted Path object
    """
    return Path(str(p))

_SCAN_ID_RE = re.compile(r"^(?P<patient>\d+)_(?P<index>\d+)_(?P<day>\d+)$")

def parse_scan_id(scan_id: str) -> Tuple[str, int, int]:
    """
    Parse a scan identifier string into patient ID, scan index, and day components.
    
    Expected format: "patient_index_day" (e.g., "120_0_0")
    
    Args:
        scan_id : str : Scan identifier string to parse
    
    Returns:
        patient_id : str : Patient identifier string
        index : int : Scan index for this patient
        day : int : Day/timepoint indicator
    
    Raises:
        ValueError : If scan_id does not match the expected pattern
    """
    m = _SCAN_ID_RE.match(scan_id)
    if not m:
        raise ValueError(f"scan_id '{scan_id}' does not match 'patient_index_day'.")
    return m.group("patient"), int(m.group("index")), int(m.group("day"))

# Track roots to help import utils.py even if CWD is elsewhere
_UTILS_SEARCH_PATHS: List[Path] = []

def _register_utils_search_root(root: Path) -> None:
    """
    Register a directory path for utility module searching.
    
    Adds a root directory to the global search paths list if not already present,
    enabling module imports from non-standard locations.
    
    Args:
        root : pathlib.Path : Directory path to register for module searching
    
    Returns:
        None
    """
    root = root.resolve()
    if root not in _UTILS_SEARCH_PATHS:
        _UTILS_SEARCH_PATHS.append(root)

# ---------------------------------------
# File resolution under <root>/<split>/
# ---------------------------------------

def _resolve_scan_dir(root: Path, split: str, scan_id: str) -> Path:
    """
    Resolve the directory path for a specific scan within a data split.
    
    Constructs and validates the path: <root>/<split>/<scan_id>/
    
    Args:
        root : pathlib.Path : Root directory containing all splits
        split : str : Split name (e.g., "train", "val", "test")
        scan_id : str : Scan identifier (e.g., "120_0_0")
    
    Returns:
        scan_dir : pathlib.Path : Resolved directory path for the scan
    
    Raises:
        FileNotFoundError : If the scan directory does not exist
    """
    d = root / split / scan_id
    if not d.is_dir():
        raise FileNotFoundError(f"Missing scan directory: {d}")
    return d

def _resolve_image_file(dir_path: Path, scan_id: str) -> Path:
    """
    Locate the image NIfTI file for a scan within its directory.
    
    Searches for image files using multiple naming patterns in order of preference:
    1. {scan_id}_T1_pre.nii.gz
    2. {scan_id}_t1p_pre.nii.gz
    3. {scan_id}_pre_tpl.nii.gz
    4. Pattern-based fallbacks with wildcards
    
    Args:
        dir_path : pathlib.Path : Directory containing the scan files
        scan_id : str : Scan identifier for filename matching
    
    Returns:
        image_path : pathlib.Path : Path to the located image file
    
    Raises:
        FileNotFoundError : If no matching image file is found
    """
    exact = dir_path / f"{scan_id}_T1_pre.nii.gz"
    if exact.is_file():
        return exact
    exact = dir_path / f"{scan_id}_t1p_pre.nii.gz"
    if exact.is_file():
        return exact
    exact = dir_path / f"{scan_id}_pre_tpl.nii.gz"
    if exact.is_file():
        return exact
    # Conservative fallbacks
    for pat in (f"{scan_id}_T1p_pre*.nii.gz", f"{scan_id}_T1*.nii.gz", f"{scan_id}_*.nii.gz"):
        matches = sorted(dir_path.glob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No image NIfTI found in {dir_path}. Expected '{scan_id}_T1p_pre_tpl.nii.gz' or close variant."
    )

def _resolve_label_file(dir_path: Path, scan_id: str, require_labels: bool) -> Optional[Path]:
    """
    Locate the label/segmentation NIfTI file for a scan within its directory.
    
    Searches for label files using multiple naming patterns in order of preference:
    1. {scan_id}_T1_seg.nii.gz
    2. {scan_id}_T1p_seg_tpl.nii.gz
    3. {scan_id}_t1p_seg_tpl.nii.gz
    4. {scan_id}_seg_tpl.nii.gz
    5. Pattern-based fallbacks with wildcards (seg, mask, label)
    
    Args:
        dir_path : pathlib.Path : Directory containing the scan files
        scan_id : str : Scan identifier for filename matching
        require_labels : bool : If True, raise error when label file not found; 
                                if False, return None for missing labels
    
    Returns:
        label_path : Optional[pathlib.Path] : Path to the label file, or None if not found 
                                               and require_labels is False
    
    Raises:
        FileNotFoundError : If no matching label file is found and require_labels is True
    """
    exact = dir_path / f"{scan_id}_T1_seg.nii.gz"
    if exact.is_file():
        return exact
    exact = dir_path / f"{scan_id}_T1p_seg_tpl.nii.gz"
    if exact.is_file():
        return exact
    exact = dir_path / f"{scan_id}_t1p_seg_tpl.nii.gz"
    if exact.is_file():
        return exact
    exact = dir_path / f"{scan_id}_seg_tpl.nii.gz"
    if exact.is_file():
        return exact
    # Conservative fallbacks
    for pat in (
        f"{scan_id}_T1p_seg*.nii.gz",
        f"{scan_id}_seg*.nii.gz",
        "*seg*.nii.gz",
        "*mask*.nii.gz",
        "*label*.nii.gz",
    ):
        matches = sorted(dir_path.glob(pat))
        if matches:
            return matches[0]
    if require_labels:
        raise FileNotFoundError(f"No label NIfTI found in {dir_path}. Expected '{scan_id}_T1p_seg_tpl.nii.gz'.")
    return None

# -------------------------------
# JSON parsing & data construction
# -------------------------------

def _normalize_dates(days: List[float]) -> List[float]:
    """
    Normalize a list of day values to start from zero.
    
    Subtracts the minimum value from all elements so the earliest timepoint becomes day 0.
    
    Args:
        days : List[float] : List of day/time values to normalize
    
    Returns:
        normalized_days : List[float] : List of normalized day values starting from 0
    """
    if not days:
        return []
    m = min(days)
    return [d - m for d in days]

def _load_one_sequence(
    root: Path,
    split: str,
    patient_entry: Dict[str, Any],
    require_labels: bool = True,
) -> Dict[str, Any]:
    """
    Load a complete temporal sequence for one patient from the file system.
    
    Processes all scans for a patient, resolving file paths under <root>/<split>/<scan_id>/,
    loading images and labels using MONAI transforms, and organizing them temporally.
    
    Args:
        root : pathlib.Path : Root directory containing all data splits
        split : str : Split name (e.g., "train", "val", "test")
        patient_entry : Dict[str, Any] : Dictionary containing patient metadata with keys:
                                         - "patient_id" (optional): patient identifier
                                         - "scans": list of scan dictionaries with:
                                           - "scan_id": scan identifier string
                                           - "days_since_first" (optional): temporal offset
        require_labels : bool : If True, raise error for missing labels; if False, use zeros
    
    Returns:
        sequence_data : Dict[str, Any] : Dictionary containing:
            - "images": List[np.ndarray] : List of image arrays, each with shape (C, H, W, D)
            - "labels": List[np.ndarray] : List of label arrays, each with shape (C, H, W, D)
            - "dates": List[float] : Normalized day values (starting from 0)
            - "patient_id": str : Patient identifier
            - "scan_ids": List[str] : List of scan identifiers in temporal order
    
    Raises:
        ValueError : If scan_id is missing or malformed
        FileNotFoundError : If required files cannot be located
    """
    pid = str(patient_entry.get("patient_id", "")).strip()
    scans = list(patient_entry.get("scans", []))

    # Collect (scan_id, day) using third component of scan_id; fallback to 'days_since_first'
    parsed: List[Tuple[str, float]] = []
    for s in scans:
        scan_id = str(s.get("scan_id", "") or s.get("id", "")).strip()
        if not scan_id:
            raise ValueError("Each scan must include a 'scan_id' (e.g., '120_0_0').")
        try:
            p_patient, _, day = parse_scan_id(scan_id)
            if not pid:
                pid = p_patient
            day_f = float(day)
        except Exception:
            day_f = float(s.get("days_since_first", 0.0))
        parsed.append((scan_id, day_f))

    # Sort by day ascending and compute 0-based offsets
    parsed.sort(key=lambda x: x[1])
    rel_days = _normalize_dates([d for _, d in parsed])

    # MONAI loader: Load + Ensure channel-first
    loader = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
    ])

    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    scan_ids: List[str] = []

    for (scan_id, _day) in parsed:
        scan_dir = _resolve_scan_dir(root, split, scan_id)
        img_fp = _resolve_image_file(scan_dir, scan_id)
        lab_fp = _resolve_label_file(scan_dir, scan_id, require_labels=require_labels)

        data_dict = {"image": str(img_fp)}
        if lab_fp is not None:
            data_dict["label"] = str(lab_fp)

        out = loader(data_dict)
        img_np = np.asarray(out["image"])  # C,H,W,D
        if "label" in out:
            lab_np = np.asarray(out["label"])
        else:
            lab_np = np.zeros_like(img_np, dtype=img_np.dtype)

        images.append(img_np)
        labels.append(lab_np)
        scan_ids.append(scan_id)

    return {
        "images": images,
        "labels": labels,
        "dates": rel_days,
        "patient_id": pid if pid else "unknown_patient",
        "scan_ids": scan_ids,
    }

def _load_split(
    root: Path,
    split_name: str,
    split_entries: List[Dict[str, Any]],
    shuffle_patients: bool,
    require_labels: bool,
) -> List[Dict[str, Any]]:
    """
    Load all patient sequences for a single data split.
    
    Processes all patients in the split by loading their temporal sequences and
    optionally shuffling the patient order.
    
    Args:
        root : pathlib.Path : Root directory containing all data splits
        split_name : str : Name of the split (e.g., "train", "val", "test")
        split_entries : List[Dict[str, Any]] : List of patient entry dictionaries from JSON
        shuffle_patients : bool : If True, randomize the order of patients in the returned list
        require_labels : bool : If True, raise error for missing labels; if False, use zeros
    
    Returns:
        split_data : List[Dict[str, Any]] : List of patient sequence dictionaries, each containing:
            - "images": List[np.ndarray] with shapes (C, H, W, D)
            - "labels": List[np.ndarray] with shapes (C, H, W, D)
            - "dates": List[float]
            - "patient_id": str
            - "scan_ids": List[str]
    """
    data = [_load_one_sequence(root, split_name, p, require_labels=require_labels) for p in split_entries]
    if shuffle_patients:
        random.shuffle(data)
    return data

# -----------------------------------------
# NEW signature: pass only the root directory
# -----------------------------------------

def load_temporal_splits_from_json(
    root: Union[str, os.PathLike],
    json_filename: str = "train_val_test_split.json",
    shuffle_patients: bool = True,
    require_labels: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load train/validation/test splits of temporal medical imaging data from JSON specification.
    
    Reads a JSON file containing train/val/test split definitions and loads all patient sequences
    from the corresponding directory structure. Expected directory layout:
        <root>/
            train/<scan_id>/...
            val/<scan_id>/...
            test/<scan_id>/...
            <json_filename>
    
    The JSON file should contain:
        {
            "train": [{"patient_id": "...", "scans": [...]}, ...],
            "val": [...],
            "test": [...]
        }
    
    Args:
        root : Union[str, os.PathLike] : Root directory containing splits and JSON file
                                         (e.g., '/standard/gam_ai_group/T1_split')
        json_filename : str : Name of the JSON file containing split definitions
                              (default: "train_val_test_split.json")
        shuffle_patients : bool : If True, randomize patient order within each split
                                  (default: True)
        require_labels : bool : If True, raise error for missing label files; 
                                if False, use zero-filled arrays matching image shape
                                (default: True)
    
    Returns:
        splits : Dict[str, List[Dict[str, Any]]] : Dictionary with keys "train", "val", "test", 
                                                   each containing a list of patient sequence 
                                                   dictionaries with:
            - "images": List[np.ndarray] : Image arrays with shape (C, H, W, D)
            - "labels": List[np.ndarray] : Label arrays with shape (C, H, W, D)
            - "dates": List[float] : Normalized day values
            - "patient_id": str : Patient identifier
            - "scan_ids": List[str] : Scan identifiers in temporal order
    
    Raises:
        FileNotFoundError : If JSON file or required data files are not found
        KeyError : If JSON file is missing required split keys ("train", "val", "test")
    """
    root = _as_path(root).resolve()
    _register_utils_search_root(root)  # so utils.py at <root>/utils.py can be found

    json_path = root / json_filename
    if not json_path.is_file():
        raise FileNotFoundError(f"Could not find JSON at '{json_path}'")

    with open(json_path, "r") as f:
        spec = json.load(f)

    for key in ("train", "val", "test"):
        if key not in spec:
            raise KeyError(f"JSON missing split '{key}'")

    train = _load_split(root, "train", spec["train"], shuffle_patients, require_labels)
    val   = _load_split(root, "val",   spec["val"],   shuffle_patients, require_labels)
    test  = _load_split(root, "test",  spec["test"],  shuffle_patients, require_labels)
    return {"train": train, "val": val, "test": test}