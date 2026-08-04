# Autonomous Change Audit

Last updated: 2026-07-18

Audit boundary:
- Latest commit audited: `0457540 Add GrowthNet local roadmap, validation audits, and reliability updates`.
- Current uncommitted source changes exist in `embed_tumor.py` and `projects/vivit/src/data/synthetic.py`; those were separated from committed latest-session changes.
- `docs/AUTONOMOUS_CHANGE_AUDIT.md` did not exist when execution began, so this document consolidates the completed audit-agent findings and records the approved low-risk fixes applied afterward.

## Integration Verdict

| File | Verdict | Risk | Tests | Human Review |
|---|---|---|---|---|
| `embed_tumor.py` | HUMAN_REVIEW_REQUIRED | High scientific | selected `.venv` pytest suite passed; tests cover engineering helpers, not scientific validity | yes |
| `projects/vivit/src/data/synthetic.py` | KEEP_WITH_FOLLOWUP | Medium scientific/API | no focused scale-parameter test run | yes before scientific use |
| `scripts/run_batch_embedding.py` | KEEP | Low engineering after patch | selected `.venv` pytest suite passed | no |
| `scripts/generate_synthetic_longitudinal_dataset.py` | KEEP_WITH_FOLLOWUP | Medium scientific semantics | selected `.venv` pytest suite passed for helper/MVP tests | yes |
| `scripts/extract_real_tumor_features.py` | MODIFY | Medium scientific/metric schema | selected `.venv` pytest suite passed for helper tests | yes before metric-definition change |
| `scripts/generate_synthetic_lollipop_cohort.py` | KEEP_WITH_FOLLOWUP | Medium provenance/aniso | selected `.venv` pytest suite passed for helper/small calibration tests | yes before authoritative use |
| `tests/test_batch_helpers.py` | KEEP | Low | selected `.venv` pytest suite passed | no |
| `tests/test_embedding_helpers.py` | MODIFY | Medium integration | selected `.venv` pytest suite passed against current working tree | no, but committed implementation/test boundary still needs cleanup |
| `tests/test_feature_extractor_helpers.py` | KEEP_WITH_FOLLOWUP | Low | selected `.venv` pytest suite passed | no |
| `tests/test_longitudinal_dataset_audit.py` | KEEP_WITH_FOLLOWUP | Low | selected `.venv` pytest suite passed | no |
| `tests/test_longitudinal_helpers.py` | KEEP_WITH_FOLLOWUP | Low | selected `.venv` pytest suite passed | no |
| `tests/test_synthetic_generation.py` | KEEP_WITH_FOLLOWUP | Low | selected `.venv` pytest suite passed | no |

## Applied Low-Risk MODIFY Items

Only low-risk, engineering-only, backward-compatible batch-runner items were changed:

1. `scripts/run_batch_embedding.py`: `_flatten_case_metrics` now populates `target_volume_mm3` from either `target_volume_mm3` or `target_tumor_volume_mm3`.
   - Behavior before: batch CSV rows could drop the requested target volume when embedding metrics used `target_tumor_volume_mm3`.
   - Behavior after: existing `target_volume_mm3` remains preferred; embedding-style `target_tumor_volume_mm3` is used as fallback.
   - Verdict after patch: KEEP.

2. `scripts/run_batch_embedding.py`: `_build_failure_report` now filters out cases with missing `primary_axis_error_deg` before sorting.
   - Behavior before: mixed old/new metrics with `primary_axis_error_deg=None` could raise `TypeError` via `float(None)`.
   - Behavior after: cases without axis error are omitted from `worst_axis_error_cases`.
   - Verdict after patch: KEEP.

Focused tests added first in `tests/test_batch_helpers.py`:
- `test_flatten_case_metrics_accepts_embedding_target_volume_key`
- `test_failure_report_ignores_missing_axis_error_values`

## Source File Audit

### `embed_tumor.py`

Purpose: main synthetic tumor embedding pipeline with lollipop geometry, placement, orientation validation, QC images, NIfTI outputs, and metrics reports.

Latest committed changes: none in `0457540`.

Current uncommitted changes include target-volume calibration, new validation fields, new CLI options, ROI QC layout, and expanded metrics schema. These are scientific and schema-affecting changes. They should not be committed or modified without human review because target-volume calibration can override anatomy-derived CPA radius and alter tumor geometry semantics.

Recommendation: HUMAN_REVIEW_REQUIRED.

### `projects/vivit/src/data/synthetic.py`

Purpose: synthetic 3D time-series generator used by ViViT experiments and lollipop mask generation.

Latest committed changes: none in `0457540`.

Current uncommitted changes add `canal_growth_scale` and `bulb_growth_scale` to lollipop growth routing. Defaults preserve existing behavior, but the public function signature changes and non-default values alter geometry. `NaN` values are not rejected.

Recommendation: KEEP_WITH_FOLLOWUP as uncommitted exploratory tuning; MODIFY before committing if NaN validation or positional compatibility is required.

### `scripts/run_batch_embedding.py`

Purpose: CSV batch runner for local embedding cases and aggregate summaries.

Latest committed changes: thread-env defaults, lazy embedding import, optional `--resume`, required-output checks, JSONL status logging, optional volume/growth parameters, expanded summaries, and worst-axis-error reporting.

Behavior/API/schema:
- CLI added `--resume`.
- `run_batch` added `resume=False`.
- Batch CSV summary and JSON schemas gained additional volume and orientation fields.
- New `batch_case_status.jsonl` sidecar is written.
- Default recompute behavior is preserved unless `--resume` is provided.

Risks after patch:
- `--resume` still does not validate input paths, code version, parameter provenance, or seed before skipping.
- JSONL timestamps are intentionally nondeterministic.

Recommendation: KEEP. Follow-up: add provenance validation before relying on resume for long-running batches.

### `scripts/generate_synthetic_longitudinal_dataset.py`

Purpose: MVP wrapper that maps a timeline CSV and background CSV to four synthetic visits using `embed_tumor.main`.

Latest committed changes: net-new CLI/API with fixed `T1`-`T4` schema, deterministic per-visit seeds, metadata output, QC summary, and per-timepoint failure rows.

Risks:
- Visits are independently generated target-volume cases, not a continuous longitudinal trajectory.
- `stable`/`growing` labels are metadata-plus-engine-mode only and are not validated against volume trends.
- Invalid volume strings and duplicate patient IDs need better engineering handling.

Recommendation: KEEP_WITH_FOLLOWUP for MVP plumbing; human review required before scientific longitudinal claims.

### `scripts/extract_real_tumor_features.py`

Purpose: new feature extractor for NIfTI tumor masks.

Latest committed changes: net-new CLI/helper module for spacing-aware volume, largest-component geometry, PCA, surface area, sphericity, compactness, component flags, CSV/JSON output, and summary stats.

Risks:
- `principal_axis_length_major/minor*` may be overwritten with extent-based lengths while `elongation`, `flatness`, and `aspect_ratio_major_to_minor2` remain moment-based.
- `principal_axis_vector_mm` is spacing-scaled voxel direction, not full affine-world direction for rotated/sheared affines.
- JSON output can contain non-standard `NaN`.

Recommendation: MODIFY, but not executed here because these are metric-definition/scientific-output changes requiring human review.

### `scripts/generate_synthetic_lollipop_cohort.py`

Purpose: new target-volume standalone synthetic lollipop mask generator.

Latest committed changes: net-new CLI for target CSV input, deterministic seeds, target-volume calibration, NIfTI output, and manifest output.

Risks:
- Manifest lacks spacing, canal axis, tolerance, convergence flag, mask shape, and clipping margin.
- Anisotropic spacing can match volume while distorting physical morphology because scale is initialized in voxel units from a mm heuristic.
- Sanitized case ID collisions can overwrite masks.

Recommendation: KEEP_WITH_FOLLOWUP for exploratory local generation; MODIFY before authoritative anisotropic/provenance use.

## Test Execution

Environment:
- System `python3 --version`: Python 3.9.6.
- System `python3 -m pip show pytest`: pytest not installed.
- Repo-local `.venv/bin/python --version`: Python 3.9.6.
- Repo-local `.venv/bin/python -m pytest --version`: pytest 8.4.2.

Commands:
- `python3 -m pytest tests/test_batch_helpers.py -v`: blocked, no pytest.
- `python3 -m pytest -m "fast and not slow" -v`: blocked, no pytest.
- `.venv/bin/python -m pytest -m "fast and not slow" -v`: passed, 39 passed, 9 deselected, 14 warnings in 2.30s.
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile scripts/run_batch_embedding.py tests/test_batch_helpers.py`: passed.
- `git diff --check`: passed.
- Direct helper regression smoke for target-volume fallback and missing-axis-error filtering: passed.

Warning classification:
- The 14 pytest warnings are Matplotlib/PyParsing deprecation warnings from dependencies.
- They are not GrowthNet test failures.
- Passing this suite supports local engineering confidence only; it does not establish scientific validity.

GitNexus:
- Required GitNexus tools were not exposed in this session, so `gitnexus_detect_changes()` could not be run before commit.
