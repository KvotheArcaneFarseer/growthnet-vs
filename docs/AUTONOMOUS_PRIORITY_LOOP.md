# Autonomous Priority Loop

Last updated: 2026-08-04

Scope: local GrowthNet repository only. No Rivanna access, no destructive
cleanup, no generator-geometry tuning, and no scientific claims beyond local
evidence.

## Loop 1 Task List

### Task 1 - Scenario Evidence Ledger

Agent: Literature/Scientific Framing Agent

Objective: keep growth modeling framed as scenario families rather than one
universal clinical law.

Subtasks:

- Separate CSV-authored target volumes from generated scenario targets.
- Preserve `empirical_vs_v1`, `empirical_vs_v2`, and `scenario_mixture_v1` as
  explicitly experimental controls.
- Keep literature claims in docs conservative.

Status: COMPLETE for local framing; ACCEPT_WITH_FOLLOWUP scientifically.

### Task 2 - Patient-Level Longitudinal QC

Agent: Longitudinal QC Agent

Objective: add a patient-level summary across per-visit/per-variant outputs.

Subtasks:

- Summarize patient timepoint count and variant count.
- Check visit-day ordering.
- Check background MRI consistency.
- Check target-volume monotonicity.
- Count QC pass/fail rows.
- Track maximum absolute relative volume error.

Status: COMPLETE.

Outputs:

- `longitudinal_qc_summary.csv` from
  `scripts/generate_synthetic_longitudinal_dataset.py`
- Provenance hash and row payload for the new summary.
- Tests in `tests/test_longitudinal_dataset_audit.py`.

### Task 3 - Same-Timepoint Variant Diversity Tooling

Agent: Morphology QA Agent

Objective: measure whether multiple variants for the same patient/timepoint are
actually distinct.

Subtasks:

- Group metadata rows by patient/timepoint.
- Pair all variants within each group.
- Load mask NIfTIs.
- Compute pairwise Dice and voxel-count differences.
- Write CSV and Markdown report.

Status: COMPLETE as an engineering tool; ACCEPT_WITH_FOLLOWUP scientifically.

Outputs:

- `analysis/clinical_growth_law_validation/measure_variant_diversity.py`
- `tests/test_longitudinal_variant_diversity.py`

### Task 4 - Output-Volume Smoke Review

Agent: Embedding Validation Agent

Objective: verify target-volume propagation into pasted mask output.

Subtasks:

- Reuse local fixture only for engineering smoke.
- Record target, actual, RAVD, convergence, warnings, and QC status.
- Avoid calling fixture output clinically valid.

Status: COMPLETE for local engineering smoke.

### Task 5 - Documentation / Provenance Sync

Agent: Documentation Agent

Objective: keep docs synchronized with the new scenario and variant controls.

Subtasks:

- Record `scenario_mixture_v1` as scenario control, not clinical law.
- Record `--variants_per_timepoint`.
- Record new QC and diversity artifacts.

Status: COMPLETE.

### Task 6 - Real-Data Blocker Specification

Agent: Real-Data Readiness Agent

Objective: identify which next validations require real local data.

Subtasks:

- Require real longitudinal masks with visit dates for scenario calibration.
- Require real MRI/seg pairs for anatomical embedding QC.
- Require annotations or landmarks for canal/CPA compartment validation.

Status: ACCEPT_WITH_FOLLOWUP; blocked work remains `BLOCKED_REMOTE_DATA`.

## Loop 2 Task List

### Task 1 - CLI Documentation Refresh

Agent: Documentation Agent

Objective: update local reproducibility docs with `--variants_per_timepoint`,
`scenario_mixture_v1`, and `longitudinal_qc_summary.csv`.

Subtasks:

- Add command examples.
- Explain output schema additions.
- Warn that scenario modes are simulation controls.

Status: COMPLETE.

### Task 2 - Longitudinal QC Schema Test Expansion

Agent: Testing Agent

Objective: assert that actual generator runs write the patient-level QC summary.

Subtasks:

- Extend mocked integration test to check file existence.
- Assert key columns and provenance hash fields.
- Keep test fixture cheap and local.

Status: COMPLETE.

Outputs:

- `analysis/clinical_growth_law_validation/sample_growth_scenarios.py`
- `analysis/clinical_growth_law_validation/scenario_sampling_audit.csv`
- `analysis/clinical_growth_law_validation/SCENARIO_SAMPLING_AUDIT.md`

Finding: in a 200-patient no-MRI sample with 100 mm3 baseline,
`scenario_mixture_v1` produced 44 fast, 92 moderate, 45 slow, and 19 regression
patients. The moderate-growth clamp can still produce T4 = 1064.8 mm3 from a
100 mm3 baseline over three years, so annual-rate bounds need human scientific
review before training use.

### Task 3 - Variant Diversity Smoke Command

Agent: Morphology QA Agent

Objective: run diversity tooling on a generated multi-variant local fixture when
available.

Subtasks:

- Generate or reuse a small local multi-variant output.
- Run `measure_variant_diversity.py`.
- Inspect Dice distribution.

Status: COMPLETE as local engineering smoke.

Outputs:

- `analysis/clinical_growth_law_validation/variant_smoke_timeline.csv`
- `analysis/clinical_growth_law_validation/variant_smoke_backgrounds.csv`
- `analysis/clinical_growth_law_validation/variant_smoke_output/`

Finding: 7/8 variants passed the loose 15% smoke RAVD threshold. The one
failure had RAVD 0.1523 against a 64 mm3 target, consistent with small-target
voxel quantization. Pairwise same-timepoint Dice ranged from 0.8027 to 0.9399
with median 0.8822.

### Task 4 - Scenario Mixture Distribution Audit

Agent: Clinical Modeling Agent

Objective: sample many deterministic synthetic patients and report scenario
frequencies and target-volume ranges without generating NIfTIs.

Subtasks:

- Build a no-MRI sampling script or notebook.
- Record scenario frequencies.
- Record volume trajectories by scenario.
- Detect unreasonable outliers.

Status: HUMAN_REVIEW_REQUIRED.

### Task 5 - Growth Scenario Human Review Packet

Agent: Scientific Review Agent

Objective: prepare the exact items requiring instructor/scientific review.

Subtasks:

- Scenario labels.
- Scenario probabilities.
- Annual-rate bounds.
- Whether regression should be allowed under `growth_label=growing`.

Status: NEXT.

### Task 6 - Training Dataset Readiness Gate

Agent: Dataset Integration Agent

Objective: define when longitudinal outputs become training-ready.

Subtasks:

- Required metadata columns.
- Required QC pass criteria.
- Required variant-diversity thresholds.
- Required validation data.

Status: COMPLETE as documentation; implementation validation remains open.

Output: `docs/TRAINING_DATASET_READINESS_GATES.md`.

### Task 7 - Worktree Hygiene Plan

Agent: Release Engineering Agent

Objective: separate older autonomous changes from current loop changes before
committing.

Subtasks:

- Inventory dirty files.
- Classify source, tests, docs, analysis, generated artifacts.
- Recommend commit grouping.

Status: NEXT; do not execute destructive cleanup automatically.

## Loop 1 Validation

- Focused longitudinal/variant tests:
  `.venv/bin/python -m pytest tests/test_longitudinal_helpers.py tests/test_longitudinal_dataset_audit.py tests/test_longitudinal_variant_diversity.py -v`
- Result: 19 passed, 14 dependency deprecation warnings.

## Loop 3 Task List

Planning date: 2026-08-04

Online validation basis:

- Recent VS growth-prediction work uses time-conditioned shape modeling rather
  than a single scalar law, supporting continued focus on longitudinal shape
  trajectories and loader validation.
- Synthetic longitudinal health-data reviews emphasize provenance, train/test
  split rules, leakage control, and validation against real holdouts.
- Medical-image segmentation metric references support Dice as a useful overlap
  metric, but not a complete morphology or boundary-quality metric.

### Task 1 - Volume-Stratified Variant Smoke

Agent: Variant QC Agent

Objective: test whether same-timepoint variant diversity and RAVD behave
differently for small, medium, and large requested volumes.

Subtasks:

- Generate or reuse local-only small/medium/large multi-variant fixture outputs.
- Run `measure_variant_diversity.py` for each volume stratum.
- Summarize Dice, RAVD, voxel-count variance, and mask non-emptiness by volume.

Parallelization: Can run in parallel with documentation, provenance audit, and
loader inspection.

Validation: local CSV/report outputs exist; no original artifacts overwritten.

### Task 2 - Longitudinal QC Gate Hardening

Agent: Longitudinal QC Agent

Objective: turn current summary fields into explicit engineering pass/fail gates.

Subtasks:

- Define engineering-smoke thresholds separately from scientific thresholds.
- Add gate fields for visit ordering, background consistency, target monotonicity,
  achieved-volume monotonicity, and max RAVD.
- Add focused tests with passing and failing synthetic metadata/QC rows.

Parallelization: Can run in parallel with Task 1 because it can use mocked rows.

Validation: focused tests pass and failure reasons are interpretable.

Status: COMPLETE for engineering gate fields and focused tests.

Outputs:

- `engineering_qc_gate`
- `engineering_qc_failure_reasons`
- `target_volume_trend_status`
- `actual_volume_trend_status`
- Focused tests in `tests/test_longitudinal_dataset_audit.py`.

### Task 3 - Scenario Mixture Review Packet

Agent: Scientific Framing Agent

Objective: prepare the minimal human-review packet for `scenario_mixture_v1`.

Subtasks:

- List current scenario probabilities and annual-rate bounds.
- Flag the observed 100 mm3 to 1064.8 mm3 three-year moderate-growth upper case.
- Separate simulation-control language from clinical-validation language.

Parallelization: Can run in parallel with engineering tasks.

Validation: docs clearly mark decisions as `HUMAN_REVIEW_REQUIRED`.

### Task 4 - Downstream Dataset Loader Smoke

Agent: Dataset Integration Agent

Objective: determine whether current longitudinal outputs can be read by the
local ViViT/downstream dataset path.

Subtasks:

- Inspect expected local data-loader schema and patient/timepoint grouping.
- Attempt a tiny loader-only smoke using existing variant smoke output.
- Verify paths, labels, visit ordering, variant grouping, and tensor shapes.

Parallelization: Can begin after Task 1 output is available; loader inspection
can start immediately.

Validation: loader smoke passes or produces a precise local blocker.

Status: PARTIAL_COMPLETE.

Findings:

- Existing ViViT loader expects a split-folder layout, while the longitudinal
  generator emits flat `images/`, `masks/`, and CSV metadata artifacts.
- `projects/vivit/src/data/synthetic_longitudinal_loader.py` now provides a
  metadata-based adapter that groups by `(patient_id, variant_id)` and creates
  ViViT-style sequence records.
- Full NIfTI/tensor loading smoke remains pending.

### Task 5 - Provenance And Artifact Audit

Agent: Provenance Agent

Objective: confirm newly generated longitudinal artifacts are clearly marked as
experimental and reproducible.

Subtasks:

- Verify law mode, scenario, target source, visit day, variant seed, and source
  paths in metadata/provenance.
- Confirm stale pulled artifacts were not overwritten.
- Update artifact-status docs if any new output should be labeled experimental.

Parallelization: Can run after sample output exists; docs work can proceed in
parallel.

Validation: provenance fields are complete and experimental outputs are labeled.

### Task 6 - Training Readiness Decision Gate

Agent: Integration Review Agent

Objective: combine Loop 3 evidence into a readiness classification.

Subtasks:

- Combine volume-stratified QC, diversity, loader, and provenance findings.
- Classify as `NOT_READY`, `ENGINEERING_SMOKE_READY`, or
  `HUMAN_REVIEW_REQUIRED`.
- Produce the next three actions for scientific validation blockers.

Parallelization: Runs after Tasks 1-5 report.

Validation: final status is evidence-backed and does not claim clinical validity.

### Task 7 - Worktree Hygiene Plan

Agent: Release Engineering Agent

Objective: prepare a non-destructive commit/cleanup plan for the dirty local tree.

Subtasks:

- Classify dirty files as source, tests, docs, analysis outputs, pulled artifacts,
  or generated smoke outputs.
- Recommend commit grouping without reverting user/previous-agent work.
- Identify large/generated outputs that should remain uncommitted unless
  intentionally tracked.

Parallelization: Can run in parallel with all tasks.

Validation: produces a reviewable grouping plan only; no cleanup performed.

Status: COMPLETE as inspection-only worktree hygiene guidance.

Finding: stage future commits by purpose; do not mix source behavior changes,
analysis artifact dumps, pulled artifacts, and documentation cleanup.
