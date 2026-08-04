# Growth Law Output Validation

## Scope

This is a local engineering smoke test for `clinical_growth_law=empirical_vs_v2`.
It uses `embedding_outputs/embedded_tumor_volume.nii.gz` and
`embedding_outputs/embedded_tumor_mask.nii.gz` as a reusable local MRI/mask
fixture, then runs the real embedding path for four visits. This checks
whether law-derived target volumes are propagated into the pasted mask and
QC overlay. It is not a clinical validation dataset.

## Result

- Visits attempted: 4
- Visits passing volume tolerance: 4
- Visits with embedding QC hard failures: 4
- Max RAVD among volume-passing visits: 0.0819
- Results CSV: `analysis/clinical_growth_law_validation/growth_law_overlay_smoke.csv`
- QC overlays: `analysis/clinical_growth_law_validation/overlays/`

## Per-Visit Summary

| Timepoint | Visit day | Target mm3 | Actual mm3 | RAVD | Converged | Volume | Embedding QC |
|---|---:|---:|---:|---:|---|---|---|
| T1 | 0.00 | 100.00 | 96.00 | 0.0400 | True | PASS | FAIL |
| T2 | 365.25 | 173.18 | 165.50 | 0.0444 | True | PASS | FAIL |
| T3 | 730.50 | 299.92 | 317.12 | 0.0573 | True | PASS | FAIL |
| T4 | 1095.75 | 519.42 | 476.88 | 0.0819 | True | PASS | FAIL |

## Interpretation

The longitudinal law produces target volumes in mm3. The embedding engine then
optimizes the generated tumor size for each visit and reports the actual
pasted mask volume. Any nonzero RAVD here is an embedding/voxelization
realization error, not a growth-law math error.

The local source MRI/mask fixture contains a very small prior embedded mask.
Its `placed_to_seg_ratio` warnings/failures therefore limit anatomical QC
interpretation. They do not invalidate the target-to-output volume check.

Scientific validation remains separate: this smoke does not prove that the
shape trajectory, anatomy, or patient-level growth process is clinically
realistic.
