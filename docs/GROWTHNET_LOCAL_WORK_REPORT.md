# GrowthNet Local Work Report

Last updated: 2026-07-18

## Session Objective

Create a local-only master action plan, launch safe parallel specialist agents, execute local work where possible, and consolidate engineering and scientific findings without using Rivanna or remote datasets.

## Initial Repository Findings

- Branch: `main`.
- Recent commit: `242631b Split orientation confidence and score-margin warnings`.
- Worktree is dirty:
  - Modified: `.gitignore`, `embed_tumor.py`, `projects/vivit/src/data/synthetic.py`, `scripts/run_batch_embedding.py`.
  - Untracked: local docs, analysis pulls, lab exports, generated manifests, local timeline data, handoff/status reports.
- Core generated local output folders exist:
  - `embedding_outputs/`
  - `tmp_batch_outputs/`
  - `tmp_seed_validation/`
  - `animation_outputs/`
  - `lab_meeting_exports/`
- Local real/synthetic analysis artifacts exist under `rivanna_pull/analysis/**`; these are local files and may be analyzed, but not treated as newly accessible remote data.

## Local Validation Already Run

```bash
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile embed_tumor.py scripts/run_batch_embedding.py scripts/extract_real_tumor_features.py scripts/generate_synthetic_lollipop_cohort.py scripts/generate_synthetic_longitudinal_dataset.py projects/vivit/src/data/synthetic.py shared/config_loader.py shared/run_logger.py projects/_shared/src/run_recording.py
```

Result: passed.

```bash
python3 -m pytest projects/mri_registration/tests/test_smoke.py -q
```

Result: failed because `pytest` is not installed in the active Python environment.

## Local Constraints

- No SSH.
- No SLURM.
- No Rivanna assumptions.
- No remote datasets.
- No destructive cleanup.
- No core scientific behavior changes without evidence and review.

## Agent Launch Log

- Agent 1 Orientation and Embedding Validation launched as `019f7394-28f1-70c2-bf00-9f7b4aa1a741`.
- Agent 2 Batch Pipeline Local Reliability launched as `019f7394-5221-7b01-89fe-68d1ce3a2d51`.
- Agent 3 Longitudinal Dataset Audit launched as `019f7394-77ce-7db2-9cad-4d67cd7eeb72`.
- Agent 4 Volume Targeting Validation launched as `019f7394-a4d6-7811-967d-89fba88611e7`.
- Agent 5 Morphology and Real-vs-Synthetic Audit launched as `019f7394-c7cb-7111-8227-25f2bf7453ba`.
- Agent 6 Testing and Reproducibility launched as `019f7394-eb01-7e40-b956-cd44ff60c414`.
- Agent 3 completed LONG-001; its thread was closed after result capture.
- Agent 7 Documentation and Repository Audit launched as `019f7397-e294-7581-a140-d63d4353950f` after Agent 3 completed.
- Agent 8 Integration and Review remains queued until specialist outputs exist.

## Integrated Specialist Result: LONG-001

Status: ACCEPT_WITH_FOLLOWUP.

Files added:
- `docs/LONGITUDINAL_PIPELINE_AUDIT.md`
- `data/timelines/local_longitudinal_example.csv`
- `tests/test_longitudinal_dataset_audit.py`

Validation reported:
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile scripts/generate_synthetic_longitudinal_dataset.py tests/test_longitudinal_dataset_audit.py`
- `MPLCONFIGDIR=/tmp/growthnet_mpl PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m unittest tests.test_longitudinal_dataset_audit -v`

Result: 3 unittest tests passed. `pytest` remains unavailable locally.

Follow-up:
- Human scientific review is required before interpreting `stable` and `growing` labels as clinically meaningful growth categories.

## Integrated Specialist Result: EMB-001

Status: ACCEPT_WITH_FOLLOWUP.

Files added:
- `analysis/orientation_validation/orientation_diagnostic.py`
- `analysis/orientation_validation/orientation_case_review.csv`
- `analysis/orientation_validation/ORIENTATION_DIAGNOSTIC.md`
- representative copied QC PNGs under `analysis/orientation_validation/`

Validation reported:
- `python3 analysis/orientation_validation/orientation_diagnostic.py`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/orientation_validation/orientation_diagnostic.py`

Scientific finding:
- Local standalone masks for `126_5_1607` and `132_1_148` do not show a synthetic-space spatial orientation bug.
- Their major PCA axes align with manifest-derived synthetic canal axes at approximately 0.43 and 0.60 degrees.
- Embedded patient-space orientation for those named cases cannot be evaluated locally because original MRI/segmentation and embedded outputs are not present. This is `BLOCKED_REMOTE_DATA`.

Follow-up:
- Human/scientific review should decide whether future validation should report whole-mask PCA, canal/stem-specific axis, canal-to-CPA axis, or all of them.

## Integrated Specialist Result: BATCH-001

Status: ACCEPT_WITH_FOLLOWUP.

Files changed:
- `scripts/run_batch_embedding.py`
- `docs/BATCH_LOCAL_RELIABILITY.md`
- `analysis/batch_reliability/LOCAL_BATCH_SMOKE.md`
- `analysis/batch_reliability/local_batch_smoke_cases.csv`
- `analysis/batch_reliability/local_batch_resume_cases.csv`

Engineering finding:
- The local batch runner now has conservative BLAS thread defaults, lazy `embed_tumor` import, opt-in `--resume`, complete-output validation before skipping, and append-only `batch_case_status.jsonl` events.
- Default behavior remains backward compatible: without `--resume`, cases recompute.

Validation rerun by lead:
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile scripts/run_batch_embedding.py`
- `python3 scripts/run_batch_embedding.py --help`
- `git diff --check`

Result: passed.

Limitation:
- The local smoke used generated outputs as stand-ins and therefore validated runner mechanics, not scientific embedding quality.

## Integrated Specialist Result: VOL-001

Status: ACCEPT_WITH_FOLLOWUP.

Files added:
- `analysis/volume_targeting/run_volume_targeting_benchmark.py`
- `analysis/volume_targeting/volume_targeting_benchmark.csv`
- `analysis/volume_targeting/VOLUME_TARGETING_REPORT.md`
- `analysis/volume_targeting/requested_vs_achieved.png`
- `analysis/volume_targeting/ravd_by_target_volume.png`

Scientific finding:
- Standalone local mask generation completed 36/36 benchmark cases.
- 30/36 met 3% RAVD.
- All tested targets at or above 100 mm3 met 3% RAVD.
- Tiny targets below 100 mm3 are dominated by voxel quantization and need absolute-error-aware thresholds.

Validation rerun by lead:
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/volume_targeting/run_volume_targeting_benchmark.py`

Limitation:
- This validates standalone masks only, not embedded tumor placement inside patient MRI frames.

## Integrated Specialist Result: MORPH-001

Status: HUMAN_REVIEW_REQUIRED.

Files added:
- `analysis/real_vs_synthetic/analyze_local_morphology.py`
- `analysis/real_vs_synthetic/LOCAL_VALIDATION_REPORT.md`
- derived CSV/JSON benchmark outputs under `analysis/real_vs_synthetic/`
- plots under `analysis/real_vs_synthetic/plots/`

Scientific finding:
- Local real feature artifacts contain 291 pulled real feature rows.
- The matched local benchmark has 261 real/synthetic case IDs.
- Real source segmentation masks are not present locally; independent real re-extraction is `BLOCKED_REMOTE_DATA`.
- The pulled synthetic feature CSV does not reproduce from local synthetic masks with the current extractor. Example drift: pulled synthetic median elongation 2.466 versus local re-extracted median elongation 1.216; pulled bbox fill 0.170 versus local re-extracted 0.481; local re-extraction found 9 multi-component masks where pulled summary reported 0.

Validation rerun by lead:
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/real_vs_synthetic/analyze_local_morphology.py`

Decision:
- Do not proceed with generator morphology tuning until provenance/extractor drift is resolved.

## Integrated Specialist Result: TEST-001

Status: READY_FOR_VALIDATION.

Files added:
- `pytest.ini`
- `tests/conftest.py`
- `tests/test_batch_helpers.py`
- `tests/test_embedding_helpers.py`
- `tests/test_feature_extractor_helpers.py`
- `tests/test_longitudinal_helpers.py`
- `tests/test_synthetic_generation.py`
- `docs/LOCAL_TESTING.md`

Engineering finding:
- A focused local pytest suite now exists for helper-level batch, embedding, feature extraction, synthetic generation, and longitudinal behavior.

Validation rerun by lead:
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m compileall tests`

Result: passed.

Blocked validation:
- `python3 -m pytest -m fast` failed because `pytest` is not installed in the active Python environment.

## Integrated Specialist Result: DOC-001

Status: ACCEPT_WITH_FOLLOWUP.

Files changed:
- `README.md`
- `docs/LOCAL_REPRODUCIBILITY.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/CURRENT_PROJECT_STATUS.md`

Engineering finding:
- Documentation now distinguishes implemented, partial, experimental, local-only validated, remote-blocked, and unvalidated areas.

Human review:
- Scientific wording should be reviewed before publication or external presentation use.

## Integration Review Result: REVIEW-001

Status: COMPLETE.

Lead validation rerun:
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile embed_tumor.py scripts/run_batch_embedding.py scripts/extract_real_tumor_features.py scripts/generate_synthetic_lollipop_cohort.py scripts/generate_synthetic_longitudinal_dataset.py projects/vivit/src/data/synthetic.py shared/config_loader.py shared/run_logger.py projects/_shared/src/run_recording.py analysis/orientation_validation/orientation_diagnostic.py analysis/volume_targeting/run_volume_targeting_benchmark.py analysis/real_vs_synthetic/analyze_local_morphology.py`
- `MPLCONFIGDIR=/tmp/growthnet_mpl PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m unittest tests.test_longitudinal_dataset_audit -v`
- `python3 scripts/run_batch_embedding.py --help`
- `git diff --check`
- `git diff --stat`

Result:
- Py-compile passed.
- Longitudinal unittest passed: 3 tests.
- Batch help passed.
- Diff whitespace check passed.
- Pytest remains blocked by missing dependency.

## Consolidated Findings

Engineering:
- The local repository is now better organized for autonomous continuation: plan, status, local report, reproducibility docs, limitations docs, batch reliability docs, testing docs, and focused analysis outputs exist.
- Batch execution is more recoverable locally through opt-in resume and per-case JSONL status.
- Helper-level tests exist but need a local environment with `pytest` before they can be treated as executable CI coverage.
- GitNexus MCP tools required by repository policy were not exposed in this session, so symbol-level impact/detect checks could not be run.

Scientific:
- Local standalone orientation evidence does not support a synthetic-space spatial orientation bug for the named problematic masks; embedded patient-space diagnosis remains blocked by missing local MRI/seg/output data.
- Standalone mask volume targeting is quantitatively strong at or above 100 mm3 in the small local grid, weaker below 100 mm3 due to voxel quantization.
- Real-vs-synthetic morphology analysis is locally supportable only from pulled feature artifacts and local synthetic masks. Provenance/extractor drift is a hard blocker before generator tuning.
- Longitudinal generation is an MVP orchestration wrapper, not a clinically validated growth model.

## Current Task Classification

- `PLAN-001`: COMPLETE.
- `EMB-001`: ACCEPT_WITH_FOLLOWUP.
- `BATCH-001`: ACCEPT_WITH_FOLLOWUP.
- `LONG-001`: ACCEPT_WITH_FOLLOWUP.
- `VOL-001`: ACCEPT_WITH_FOLLOWUP.
- `MORPH-001`: HUMAN_REVIEW_REQUIRED.
- `TEST-001`: READY_FOR_VALIDATION, blocked by missing `pytest`.
- `DOC-001`: ACCEPT_WITH_FOLLOWUP.
- `REVIEW-001`: COMPLETE.

## Remote Data Requirements

Known remote-dependent work is tracked in `docs/GROWTHNET_ACTION_PLAN.md` as `BLOCKED_REMOTE_DATA`; it must not block local reliability, tests, documentation, or local artifact analysis.

Current remote-data blockers:
- Original MRI/segmentation and embedded outputs for the named orientation cases, including `126_5_1607` and `132_1_148`, if patient-space diagnosis is required.
- Real source segmentation masks for the 261 matched morphology cases.
- Full 291-case source real segmentation cohort.
- Anatomical labels for IAC/CPA canal-vs-bulb validation.
- Any training-ready cohort generation or HPC-scale model training artifacts.

## Human Decisions Required

- Decide the authoritative orientation validation metrics: whole-mask PCA, stem/canal axis, canal-to-CPA axis, or a multi-metric report.
- Decide how to handle pulled synthetic feature provenance drift before morphology tuning.
- Decide whether proposed volume-targeting thresholds are engineering smoke thresholds only or scientific acceptance criteria.
- Review longitudinal `stable` and `growing` label semantics before using them in training or publication framing.
- Decide whether to install/pin `pytest` and other local dependencies in a repository-managed environment spec.

## Recommended Next Commands

```bash
python3 -m pytest -m "fast and not slow"
```

Requires installing `pytest` first in the active local environment.

```bash
python3 analysis/real_vs_synthetic/analyze_local_morphology.py
```

Use this to regenerate the local morphology audit after any provenance fix.

```bash
python3 analysis/volume_targeting/run_volume_targeting_benchmark.py \
  --targets 25,50,75,100,250,500,1000,4000,16000 \
  --seeds 20260426,20260427 \
  --spacings '1,1,1;0.6,0.6,1.2' \
  --out_csv analysis/volume_targeting/volume_targeting_benchmark.csv
```

Use this to reproduce the standalone volume-targeting benchmark.

## Completion Estimate

- Engineering completeness: approximately 65%.
- Scientific validation completeness: approximately 35%.
- Training-dataset readiness: approximately 25%.

These estimates separate code operability from scientific validation. The main local blockers are dependency pinning, provenance drift, and missing real source data for independent validation.
