# Orientation Diagnostic

Task: EMB-001. Scope: local repository only; no SSH/Rivanna access; no core source edits.

## Data Availability

- Named problematic standalone synthetic masks present: 126_5_1607, 132_1_148.
- Original real segmentations/MRIs for the named problematic cases are not present locally; real feature CSV rows are present with remote source paths only.
- Local embedded MRI outputs are available only for the 147 case family (`embedding_outputs`, `tmp_batch_outputs/case_147`, and seed-validation copies).
- Therefore spatial placement against patient MRI for 126_5_1607 and 132_1_148 is BLOCKED_REMOTE_DATA.

## Classification

- `126_5_1607`: `CONTROL_OR_NO_LOCAL_AXIS_SWITCH`. Whole-mask major PCA to expected canal line = 0.43 deg; best PCA axis to expected canal = major (0.43 deg). Major-to-middle variance ratio = 2.304.
- `132_1_148`: `CONTROL_OR_NO_LOCAL_AXIS_SWITCH`. Whole-mask major PCA to expected canal line = 0.60 deg; best PCA axis to expected canal = major (0.60 deg). Major-to-middle variance ratio = 2.241.

Local evidence for the named problematic cases does **not** support A as a proven true spatial orientation bug. It also does **not** support B for the locally available standalone masks: their whole-mask major PCA axes align with the manifest-derived synthetic canal lines. The best-supported classification from local data is **C: QC/validation limitation or unavailable placement data**, because the named cases lack local embedded patient-MRI outputs and the standalone masks were randomly rotated for morphology benchmarking.

Important nuance: direct angles between standalone synthetic-mask PCA axes and real patient feature-table axes are not spatially interpretable, because the standalone synthetic cohort was generated with random rotations and target volumes only. Those angles are retained in the CSV as a cautionary audit signal, not as evidence of a placement bug.

## Control Cases

- `126_1_312`: standalone synthetic control; major PCA to expected canal = 2.77 deg; classification `CONTROL_OR_NO_LOCAL_AXIS_SWITCH`.
- `132_0_0`: standalone synthetic control; major PCA to expected canal = 0.43 deg; classification `CONTROL_OR_NO_LOCAL_AXIS_SWITCH`.
- `147_local_embedding:embedding_outputs`: embedded output control; late-mask major PCA vs selected placement axis = 6.19 deg.
- `147_local_embedding:tmp_batch_outputs/case_147`: embedded output control; late-mask major PCA vs selected placement axis = 2.08 deg.
- `147_local_embedding:tmp_seed_validation/auto_seed_a`: embedded output control; late-mask major PCA vs selected placement axis = 2.08 deg.
- `147_local_embedding:tmp_seed_validation/auto_seed_b`: embedded output control; late-mask major PCA vs selected placement axis = 2.08 deg.
- `147_local_embedding:tmp_seed_validation/diff_seed`: embedded output control; late-mask major PCA vs selected placement axis = 3.39 deg.
- `147_local_embedding:tmp_seed_validation/same_seed_a`: embedded output control; late-mask major PCA vs selected placement axis = 1.87 deg.

## Voxel/Physical-Space Checks

- Standalone synthetic masks have 1.0 mm isotropic spacing and diagonal affines, so voxel-space and physical-space axes are equivalent for those files.
- Local embedded outputs use 0.5 mm isotropic spacing inherited from the 147 reference output, so voxel-space and physical-space directions are also equivalent up to scale.
- No anisotropic named-case embedded outputs were available locally for this task.

## Timepoint Coverage

- Standalone synthetic cohort masks are single-mask morphology benchmark outputs, so there are no per-visit embedded timepoints to inspect for 126_5_1607 or 132_1_148.
- Local 147 embedded controls inspected: 7 output directories. Each available metrics JSON serializes five timepoint axis errors; the recomputed late-mask axis errors are listed in `orientation_case_review.csv`.

## Reproducible Commands

```bash
python3 analysis/orientation_validation/orientation_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/orientation_validation/orientation_diagnostic.py
```

The CSV output is `analysis/orientation_validation/orientation_case_review.csv`.

## Representative QC Outputs

- `analysis/orientation_validation/case_147/qc_embedding_late.png`
- `analysis/orientation_validation/embedding_outputs/qc_embedding_late.png`

## Unsupported Claims

- Whether 126_5_1607 or 132_1_148 were spatially embedded into their patient MRIs with a true orientation bug cannot be determined locally.
- Canal-to-CPA anatomical direction relative to skull-base landmarks cannot be verified without local real MRI/segmentation/atlas landmarks.
- The cross-section-derived stem-to-bulb vector is a mask-only heuristic, not a clinical anatomical label.

## Follow-Up Tasks

- When local real segmentations for the named cases are available, rerun this diagnostic against real masks and embedded outputs.
- Add a validation metric that reports angles to all three PCA axes, not only the major axis, for large/oblate lollipop masks.
- Add a stem/canal-specific axis metric if geometry-component labels or a reliable canal extraction heuristic are added.
