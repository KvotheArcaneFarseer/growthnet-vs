# Longitudinal Architecture Bridge

Last updated: 2026-08-04

Scope: local GrowthNet repository only.

## Purpose

The synthetic longitudinal generator writes a flat local artifact layout:

- `images/*.nii.gz`
- `masks/*.nii.gz`
- `metadata.csv`
- `qc_summary.csv`
- `longitudinal_qc_summary.csv`
- optional provenance JSON

The existing ViViT temporal loader expects a legacy split-folder layout with
`train/`, `val/`, `test/`, and `train_val_test_split.json`.

Rather than rewriting generated outputs, GrowthNet now has a small adapter:

`projects/vivit/src/data/synthetic_longitudinal_loader.py`

## Current Contract

The adapter reads generator `metadata.csv` and builds ViViT-style sequence
records with:

- `images`
- `labels`
- `dates`
- `patient_id`
- `scan_ids`

For indexing, it also preserves:

- `source_patient_id`
- `variant_id`
- `timepoints`
- `visit_days`
- `image_paths`
- `label_paths`

Multi-variant outputs are grouped as separate trajectories by:

`(patient_id, variant_id)`

This prevents different same-timepoint tumor-shape variants from being mixed
inside one temporal patient sequence.

## QC Gate Contract

`longitudinal_qc_summary.csv` now includes machine-readable engineering gate
fields:

- `engineering_qc_gate`
- `engineering_qc_failure_reasons`
- `target_volume_trend_status`
- `actual_volume_trend_status`

The engineering gate is intentionally separate from scientific validity. It
checks local dataset integrity and smoke-readiness, not whether the generated
growth pattern is clinically validated.

Hard engineering failures include:

- inconsistent background MRI within a patient,
- non-increasing visit days,
- any failed per-mask QC row,
- maximum absolute relative volume error above the configured tolerance,
- missing volume-error values.

Target and achieved volume trends are reported as statuses. Declared regression,
stable, or CSV-authored trajectories may legitimately be non-monotone.

## Remaining Architecture Work

1. Add a tiny dataloader smoke using generated NIfTI fixtures.
2. Add optional metadata-based train/val/test splitting.
3. Add an exporter only if legacy experiments require the split-folder layout.
4. Add schema documentation for `metadata.csv`, `qc_summary.csv`, and
   `longitudinal_qc_summary.csv`.
5. Keep generated artifacts immutable; add adapters around them instead.
