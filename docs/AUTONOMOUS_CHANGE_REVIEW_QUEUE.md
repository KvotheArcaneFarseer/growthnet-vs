# Autonomous Change Review Queue

Last updated: 2026-07-18

## Resolved In This Patch

| Item | File | Original Verdict | Resolution | Status |
|---|---|---|---|---|
| Batch target volume summary fallback | `scripts/run_batch_embedding.py` | MODIFY | Added fallback from `target_tumor_volume_mm3` to `target_volume_mm3`; focused test added. | COMPLETE |
| Batch missing axis-error sort crash | `scripts/run_batch_embedding.py` | MODIFY | Filtered missing `primary_axis_error_deg` before sorting; focused test added. | COMPLETE |

## Still Open

| Item | File | Verdict | Required Action | Human Review |
|---|---|---|---|---|
| Embedding target-volume geometry and schema | `embed_tumor.py` | HUMAN_REVIEW_REQUIRED | Decide whether non-anatomical calibration/CPA override semantics are acceptable and how to document/gate them. | yes |
| ViViT lollipop growth scale API | `projects/vivit/src/data/synthetic.py` | KEEP_WITH_FOLLOWUP | Add default-equivalence and invalid-value tests before commit; consider keyword-only placement for compatibility. | yes before scientific use |
| Feature extractor axis metric inconsistency | `scripts/extract_real_tumor_features.py` | MODIFY | Resolve moment-vs-extent metric naming/definitions and affine-world direction semantics. | yes |
| Embedding helper tests depend on uncommitted API | `tests/test_embedding_helpers.py` | MODIFY | Either commit matching `embed_tumor.py` changes after review or revise tests to committed API. | no |
| Longitudinal MVP semantics | `scripts/generate_synthetic_longitudinal_dataset.py` | KEEP_WITH_FOLLOWUP | Add invalid-volume and duplicate-patient handling; do not claim continuous trajectories. | yes for label semantics |
| Lollipop cohort manifest provenance | `scripts/generate_synthetic_lollipop_cohort.py` | KEEP_WITH_FOLLOWUP | Add spacing/canal-axis/tolerance/convergence/collision provenance before authoritative use. | yes |

## Blocked Validation

`python3 -m pytest ...` remains blocked because `pytest` is not installed in the active Python 3.9.6 environment. No dependency installation was performed.
