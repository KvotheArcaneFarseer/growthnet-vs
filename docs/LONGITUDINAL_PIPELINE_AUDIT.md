# Longitudinal Pipeline Audit

Task: LONG-001  
Scope: local laptop repository only  
Date: 2026-08-01

## Summary

The local longitudinal dataset wrapper is implemented as `scripts/generate_synthetic_longitudinal_dataset.py`. It is an MVP orchestration layer over the existing single-case embedding engine, not an independent clinical growth simulator.

The wrapper currently reads a patient timeline CSV and a background MRI manifest, generates one embedded synthetic tumor visit per requested timepoint, copies the exported `embedded_tumor_volume.nii.gz` and `embedded_tumor_mask.nii.gz` into a longitudinal folder layout, then writes `metadata.csv` and `qc_summary.csv`.

This audit did not modify the embedding engine, tumor geometry, or growth model.

## Current Execution Path

1. `main()` parses:
   - `--timeline_csv`
   - `--background_csv`
   - `--out_dir`
   - `--seed`
   - `--volume_ravd_tolerance`
   - `--volume_max_iterations`
   - `--gen_size`
   - `--provenance_json`
   - `--clinical_growth_law`
   - `--visit_days`
2. `generate_longitudinal_dataset()` reads timeline rows requiring:
   - `patient_id`
   - `background_mri_id`
   - `T1_volume_mm3`
   - `T2_volume_mm3`
   - `T3_volume_mm3`
   - `T4_volume_mm3`
   - `growth_label`
3. `_load_backgrounds()` reads background rows requiring:
   - `background_mri_id`
   - `mri_path`
   - `seg_path`
4. For each patient:
   - `background_mri_id` is resolved to one MRI and one segmentation.
   - `growth_label` is mapped by `_growth_mode()`.
   - unsupported labels fail all visits for that patient.
5. For each visit `T1` through `T4`:
   - the requested volume is read from the corresponding `T*_volume_mm3` column.
   - a deterministic seed is derived by `_stable_seed(base_seed, patient_id, timepoint)`.
   - `embed_tumor.main()` is invoked with `dates=[0, 1]`, `volume_target_timepoint="first"`, and `target_tumor_volume_mm3=<visit volume>`.
   - only the first embedded timepoint output is exported for that visit.
6. `_qc_mask()` extracts mask morphology from the copied mask and records:
   - achieved synthetic volume
   - relative volume error
   - connected component count
   - pass/fail and failure reason

## Completed Behavior

- CSV schema validation exists for timeline and background manifests.
- Background IDs are checked for blank and duplicate entries.
- Patient identity is preserved in `metadata.csv` and `qc_summary.csv`.
- The same `background_mri_id` is used across every visit for a patient.
- Visit ordering is fixed as `T1`, `T2`, `T3`, `T4` by `TIMEPOINT_COLUMNS`.
- Deterministic per-visit seeds are derived from base seed, patient ID, and timepoint.
- `stable` maps to embedding growth mode `stable`.
- `growing` maps to embedding growth mode `steady`.
- Missing background IDs, missing files, unsupported labels, and per-visit failures are captured in output CSV rows instead of aborting the entire dataset.
- Volume target and achieved volume are recorded in separate metadata/QC outputs.

## Partial Or Missing Behavior

- Only four fixed visit columns are supported. Sparse or irregular time-volume constraints are not supported.
- The wrapper does not accept explicit visit dates, intervals, scan dates, or elapsed days.
- The wrapper does not preserve one continuous synthetic tumor trajectory across visits. Each visit is generated as an independent single-visit target using a deterministic seed.
- Background image consistency is intended by using one `background_mri_id`, but the code does not currently record source `mri_path` and `seg_path` in `metadata.csv`.
- Requested versus achieved volume is evaluated per visit, but longitudinal trend checks are not implemented.
- The wrapper does not compute monotonicity, centroid drift, axis drift, or intensity consistency across visits.
- Metadata now includes visit seed, source background paths, embedding growth mode, volume tolerance, max iterations, and generation size.
- Optional dataset-level provenance JSON is available via `--provenance_json`.
- Optional experimental target-volume generation is available via `--clinical_growth_law empirical_vs_v1`; default `none` preserves explicit `T1..T4` CSV volumes.
- Metadata does not yet include embedding metrics JSON paths.
- There is no ViViT-compatible split/layout export from this wrapper.
- There is no locally validated end-to-end run unless a local background manifest with readable NIfTI MRI and segmentation files is supplied.

## Scientific Boundaries

The wrapper must not be interpreted as a clinical growth law. A `growing` row only selects the existing embedding engine's `steady` mode for the short internal generation call. The requested `T1` through `T4` volumes are user-supplied constraints; the wrapper does not infer, fit, extrapolate, or validate biological tumor growth trajectories.

Trajectory prediction should remain separate from geometry generation. A future clinical or statistical growth model should produce explicit time-volume constraints, and this wrapper should consume those constraints without embedding assumptions about clinical progression.

## Local Validation Added

The new tests in `tests/test_longitudinal_dataset_audit.py` cover:

- deterministic seed generation
- filesystem-safe ID handling
- supported and unsupported growth labels
- patient/timepoint failure rows when background data is missing
- metadata and QC output behavior with the heavy embedding call monkeypatched
- deterministic per-visit seed routing into the embedding call
- optional longitudinal provenance payload construction
- additive metadata fields for visit seed, source paths, embedding growth mode, and generation parameters

These tests do not require Rivanna, private Downloads files, SLURM, or real MRI data.

## Local Example Timeline

`data/timelines/local_longitudinal_example.csv` provides a tiny non-clinical example with one stable and one growing synthetic patient. It intentionally contains timeline constraints only. A separate local background manifest is still required for an actual run because background MRI/segmentation files are local-user-specific.

## Optional Local Integration Command

If local NIfTI background files are available, create a background manifest with:

```csv
background_mri_id,mri_path,seg_path
BG_LOCAL_001,/absolute/path/to/background_mri.nii.gz,/absolute/path/to/background_seg.nii.gz
```

Then run:

```bash
python3 scripts/generate_synthetic_longitudinal_dataset.py \
  --timeline_csv data/timelines/local_longitudinal_example.csv \
  --background_csv /absolute/path/to/local_backgrounds.csv \
  --out_dir /tmp/growthnet_longitudinal_local_smoke \
  --seed 20260523 \
  --gen_size 64 \
  --volume_max_iterations 3
```

This command is intentionally not run by the test suite because local MRI/segmentation files are not part of the repository.

## Follow-Up Tasks

| Task ID | Status | Objective | Files Likely Involved | Validation |
|---|---|---|---|---|
| LONG-002 | NOT_STARTED | Add support for sparse/irregular visits with explicit `timepoint`, `day`, and `target_volume_mm3` rows. | `scripts/generate_synthetic_longitudinal_dataset.py`, tests | unit tests for sorting and sparse constraints |
| LONG-003 | COMPLETE | Add seed and generation parameters to `metadata.csv`. | wrapper, tests | metadata schema test |
| LONG-004 | COMPLETE | Record source background MRI/seg paths or immutable IDs in metadata for auditability. | wrapper, tests | metadata schema test |
| LONG-005 | NOT_STARTED | Add longitudinal QC for monotonicity, centroid drift, axis drift, and intensity consistency. | wrapper or QC helper, tests | synthetic pass/fail fixtures |
| LONG-006 | NOT_STARTED | Add an optional ViViT-compatible export layout and split manifest. | wrapper, `projects/vivit/src/data/temporal_loader.py` adapter docs | loader smoke test |
| LONG-007 | BLOCKED_REMOTE_DATA | Validate end-to-end behavior over real curated longitudinal backgrounds. | local or future remote manifests | cohort-level QC report |
| LONG-008 | HUMAN_REVIEW_REQUIRED | Decide acceptable interpretation of `stable` and `growing` labels in synthetic-only datasets. | docs and config | clinical/scientific review |
| LONG-009 | COMPLETE | Add optional dataset-level provenance JSON. | wrapper, tests | focused longitudinal tests |
| LONG-010 | COMPLETE | Encode default-off empirical vestibular schwannoma volumetric growth-law candidate. | wrapper, tests, docs | focused tests and no-MRI smoke |
| LONG-011 | NOT_STARTED | Validate or recalibrate growth-law parameters against real longitudinal masks. | analysis docs and future real data | real-data validation report |

## Definition Of Done For LONG-001

- Current wrapper state is documented from code inspection.
- Deterministic local behavior has tests.
- Missing behavior is converted into explicit follow-up tasks.
- Unsupported clinical assumptions are not added.
- Real-data validation remains marked as unavailable locally unless explicit local background files are supplied.
