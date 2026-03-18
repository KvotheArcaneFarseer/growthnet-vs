#!/usr/bin/env python3
"""
Sort coregistered studies into acceptable/ and manual_correction/ folders
based on Registration Classification propagation rules:
  - patient_anchor: propagate to ALL rows for that patient
  - timepoint_anchor: propagate to ALL rows at that timepoint
  - no anchor: flag only the individual row
"""

import argparse
import csv
import shutil
from pathlib import Path
from collections import defaultdict

# Defaults — overridden by CLI arguments
BASE_DIR = Path(__file__).parent / "df_coregistered_02_23_2026_V6.7"
REVIEWED_CSV = BASE_DIR / "reviewed.df_coregistered_02_23_2026_V6.7.csv"

def load_rows():
    with open(REVIEWED_CSV, "r") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames

def propagate_classifications(rows):
    """Return a set of row indices that need manual correction."""
    manual_indices = set()
    patient_propagate = set()
    timepoint_propagate = set()

    # First pass: collect propagation targets from classified=1 rows
    for i, r in enumerate(rows):
        if r.get("Registration Classification ", "").strip() != "1":
            continue
        anchor = r.get("anchor", "").strip()
        patient = r["Reworked Patient ID"].strip()
        tp = r["Patient_MRI_Days Tracker"].strip()

        if anchor == "patient_anchor":
            patient_propagate.add(patient)
        elif anchor == "timepoint_anchor":
            timepoint_propagate.add(tp)
        else:
            # No anchor: flag only this row
            manual_indices.add(i)

    # Second pass: propagate
    for i, r in enumerate(rows):
        patient = r["Reworked Patient ID"].strip()
        tp = r["Patient_MRI_Days Tracker"].strip()
        if patient in patient_propagate or tp in timepoint_propagate:
            manual_indices.add(i)

    return manual_indices, patient_propagate, timepoint_propagate

def move_files(rows, manual_indices):
    """Move files into acceptable/ or manual_correction/ keeping directory structure."""
    acceptable_dir = BASE_DIR / "acceptable"
    manual_dir = BASE_DIR / "manual_correction"

    moved = {"acceptable": 0, "manual_correction": 0}
    missing = {"image": 0, "seg": 0, "overlay": 0}

    for i, r in enumerate(rows):
        dest_base = manual_dir if i in manual_indices else acceptable_dir

        # Move coregistered image
        img_path = r.get("coregistered_image", "").strip()
        if img_path:
            src = BASE_DIR.parent / img_path
            dst = dest_base / Path(img_path).relative_to(BASE_DIR.name)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            else:
                missing["image"] += 1

        # Move coregistered segmentation
        seg_path = r.get("coregistered_segmentation", "").strip()
        if seg_path:
            src = BASE_DIR.parent / seg_path
            dst = dest_base / Path(seg_path).relative_to(BASE_DIR.name)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            else:
                missing["seg"] += 1

        # Move overlay plot
        overlay_path = r.get("coregistered_overlay_plot", "").strip()
        if overlay_path:
            src = BASE_DIR.parent / overlay_path
            dst = dest_base / Path(overlay_path).relative_to(BASE_DIR.name)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            else:
                missing["overlay"] += 1

        category = "manual_correction" if i in manual_indices else "acceptable"
        moved[category] += 1

    return moved, missing

def print_statistics(rows, manual_indices, patient_propagate, timepoint_propagate):
    total = len(rows)
    n_manual = len(manual_indices)
    n_acceptable = total - n_manual

    print("=" * 60)
    print("REGISTRATION CLASSIFICATION - PROPAGATION REPORT")
    print("=" * 60)

    # Source breakdown
    originally_flagged = [i for i, r in enumerate(rows)
                          if r.get("Registration Classification ", "").strip() == "1"]
    print(f"\nOriginal classified=1 rows:  {len(originally_flagged)}")
    print(f"  patient_anchor:            {sum(1 for i in originally_flagged if rows[i].get('anchor','').strip() == 'patient_anchor')}")
    print(f"  timepoint_anchor:          {sum(1 for i in originally_flagged if rows[i].get('anchor','').strip() == 'timepoint_anchor')}")
    print(f"  no anchor (individual):    {sum(1 for i in originally_flagged if rows[i].get('anchor','').strip() == '')}")

    print(f"\nPropagation targets:")
    print(f"  Patients propagated:       {len(patient_propagate)} -> {sorted(patient_propagate, key=lambda x: int(x))}")
    print(f"  Timepoints propagated:     {len(timepoint_propagate)}")

    print(f"\n{'=' * 60}")
    print(f"AFTER PROPAGATION")
    print(f"{'=' * 60}")
    print(f"  manual_correction:  {n_manual:5d}  ({100*n_manual/total:.1f}%)")
    print(f"  acceptable:         {n_acceptable:5d}  ({100*n_acceptable/total:.1f}%)")
    print(f"  total:              {total:5d}")

    # Per study type
    study_stats = defaultdict(lambda: {"acceptable": 0, "manual_correction": 0})
    for i, r in enumerate(rows):
        study = r.get("Study Type", "").strip()
        cat = "manual_correction" if i in manual_indices else "acceptable"
        study_stats[study][cat] += 1

    print(f"\nPer study type:")
    print(f"  {'Study Type':<20} {'Acceptable':>10} {'Manual':>10} {'Total':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for study in sorted(study_stats):
        s = study_stats[study]
        t = s["acceptable"] + s["manual_correction"]
        print(f"  {study:<20} {s['acceptable']:>10} {s['manual_correction']:>10} {t:>10}")

    # Per patient count for manual_correction
    patient_manual = defaultdict(int)
    for i in manual_indices:
        patient_manual[rows[i]["Reworked Patient ID"].strip()] += 1
    print(f"\nPatients with manual_correction rows: {len(patient_manual)}")
    print(f"  (rows per patient: min={min(patient_manual.values())}, max={max(patient_manual.values())}, "
          f"mean={sum(patient_manual.values())/len(patient_manual):.1f})")


def main(base_dir=None, csv_path=None):
    global BASE_DIR, REVIEWED_CSV
    if base_dir is not None:
        BASE_DIR = Path(base_dir)
    if csv_path is not None:
        REVIEWED_CSV = Path(csv_path)

    rows, fieldnames = load_rows()
    manual_indices, patient_propagate, timepoint_propagate = propagate_classifications(rows)

    print_statistics(rows, manual_indices, patient_propagate, timepoint_propagate)

    print(f"\n{'=' * 60}")
    print("MOVING FILES...")
    print(f"{'=' * 60}")
    moved, missing = move_files(rows, manual_indices)
    print(f"  Rows sorted to acceptable:         {moved['acceptable']}")
    print(f"  Rows sorted to manual_correction:   {moved['manual_correction']}")
    if any(missing.values()):
        print(f"  Missing files (not found on disk):")
        for k, v in missing.items():
            if v:
                print(f"    {k}: {v}")

    # Clean up empty directories left behind in raw_studies/
    raw_studies = BASE_DIR / "raw_studies"
    if raw_studies.exists():
        for d in sorted(raw_studies.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if raw_studies.exists() and not any(raw_studies.iterdir()):
            raw_studies.rmdir()

    # Clean up empty overlay_plots/
    overlay_dir = BASE_DIR / "overlay_plots"
    if overlay_dir.exists() and not any(overlay_dir.iterdir()):
        overlay_dir.rmdir()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort coregistered studies by QC classification")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory containing coregistered output "
                             f"(default: {BASE_DIR})")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to reviewed CSV with Registration Classification column "
                             f"(default: {REVIEWED_CSV})")
    args = parser.parse_args()
    main(base_dir=args.base_dir, csv_path=args.csv)
