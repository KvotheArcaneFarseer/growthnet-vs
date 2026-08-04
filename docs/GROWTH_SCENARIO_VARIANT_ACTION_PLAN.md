# Growth Scenario and Shape-Variant Action Plan

## Executive Summary

The project should not encode a single vestibular schwannoma growth law as if it
were clinically definitive. Local implementation should instead support
explicitly named growth scenarios, preserve per-scenario provenance, and emit
multiple independent shape variants for the same patient/timepoint/target
volume. This matches the current evidence better: the literature reports stable,
slow, moderate, fast, and occasionally regressive behavior, with volume-doubling
and percent-volume-change models used inconsistently across cohorts.

Current implementation status:

- `none`: preserves timeline CSV volumes.
- `empirical_vs_v1`: broad experimental stable/growing/fast bins.
- `empirical_vs_v2`: conservative log-normal volumetric growth candidate.
- `scenario_mixture_v1`: explicit scenario sampler across slow, moderate, fast,
  and regression phenotypes for `growth_label=growing`.
- `--variants_per_timepoint`: emits multiple seed-varied tumor shapes for the
  same patient/timepoint/target volume.

## Literature-Matched Growth Framing

Top relevant evidence families identified online:

1. Multi-institutional volumetric natural history: large wait-and-scan cohort,
   volumetric growth threshold >=20%, and median annual growth summaries.
2. Untreated VS volumetric growth-rate cohorts: mean annual growth around one
   third per year, but with substantial stable and fast-growing subsets.
3. Prospective volume-doubling-time studies: VDT is often presented as a more
   realistic growth descriptor than linear diameter alone.
4. Linear growth reviews: useful for clinical follow-up intuition, less direct
   for this pipeline because the generator optimizes mask volume.
5. Spontaneous stabilization/regression studies: justify keeping non-monotonic
   or regressive scenarios as explicit experimental phenotypes.
6. Advanced longitudinal MRI modeling papers: support the idea that shape
   evolution can be multi-modal and time-conditioned rather than a scalar law.

## Agentized Execution Plan

### Task 1 - Literature Evidence Table

Owner: Literature Agent

Objective: maintain a cited table of vestibular schwannoma growth models.

Subtasks:

- Record cohort size, modality, growth definition, follow-up interval, and
  reported growth-rate statistics.
- Distinguish volumetric percent/year, volume doubling time, and linear mm/year.
- Mark each source as `DIRECT_VOLUME_TARGET`, `CONTEXT_ONLY`, or
  `ADVANCED_MODEL_REFERENCE`.

Validation: citations and claims must link to source URLs or PDFs.

### Task 2 - Scenario Taxonomy

Owner: Clinical Modeling Agent

Objective: define named target-volume scenarios without overclaiming biology.

Subtasks:

- Keep `timeline_csv` as the authoritative mode for externally supplied targets.
- Define stable, slow growth, moderate growth, fast growth, and regression
  scenarios.
- Preserve sampled annual volume-change fraction and scenario name in metadata.
- Do not tune generator shape parameters inside this task.

Validation: helper tests must assert deterministic sampling and bounded outputs.

### Task 3 - Shape Variants Per Timepoint

Owner: Dataset Orchestration Agent

Objective: generate multiple tumor shapes for the same patient/timepoint.

Subtasks:

- Add `variants_per_timepoint` control.
- Keep default output naming unchanged for one variant.
- Add `variant_id` and `variant_seed` metadata.
- Ensure each variant shares the same target volume for a timepoint but receives
  a distinct seed.

Validation: mocked integration test confirms row count, seed uniqueness, and
  output file naming.

### Task 4 - Output Volume Realization

Owner: Embedding Validation Agent

Objective: verify law-derived target volumes become pasted mask volumes.

Subtasks:

- Run local smoke tests on an available MRI/mask fixture.
- Record target volume, actual embedded mask volume, RAVD, convergence, and QC
  overlay path.
- Separate volume-target success from anatomical embedding QC success.

Validation: output CSV and report under `analysis/clinical_growth_law_validation/`.

### Task 5 - Shape Diversity Metrics

Owner: Morphology QA Agent

Objective: quantify whether same-timepoint variants are meaningfully different.

Subtasks:

- Compare Dice overlap between variants at the same target volume.
- Compare elongation, principal axes, surface metrics, and compartment metrics
  where valid.
- Flag low-diversity variants as engineering limitations rather than clinical
  failures.

Validation: add a small non-clinical benchmark once real local source MRIs are
available.

### Task 6 - Longitudinal Consistency QC

Owner: Longitudinal QC Agent

Objective: validate patient-level consistency across visits and variants.

Subtasks:

- Confirm visit ordering and visit-day provenance.
- Track background MRI consistency across visits.
- Track scenario consistency within a synthetic patient.
- Allow non-monotonic volume only for explicit regression or mixed scenarios.

Validation: schema checks and small mocked generation tests.

### Task 7 - Real-Data Calibration Readiness

Owner: Scientific Validation Agent

Objective: prepare, but do not fake, future calibration against real masks.

Subtasks:

- Specify minimum real longitudinal segmentation subset.
- Specify needed visit dates and volume annotations.
- Define metrics for target-volume, shape, and compartment validation.
- Mark unavailable evidence as `BLOCKED_REMOTE_DATA`.

Validation: acquisition spec reviewed by human scientific lead.

### Task 8 - Documentation and Provenance

Owner: Documentation Agent

Objective: make every generated target explainable.

Subtasks:

- Document all scenario modes and their limitations.
- Record source mode, scenario, annual fraction, target volume, actual volume,
  visit seed, and variant seed.
- Warn that scenarios are simulation controls, not clinical predictions.

Validation: docs match CLI help and tests.

## Dependency Graph

- Task 1 feeds Task 2 and Task 8.
- Task 2 feeds Task 3 and Task 6.
- Task 3 feeds Task 4 and Task 5.
- Task 4 feeds Task 8.
- Task 5 and Task 6 can run in parallel after Task 3.
- Task 7 is partly blocked until real longitudinal masks are available.

## Current Local Validation

Implemented local checks:

- `tests/test_longitudinal_helpers.py`
- `tests/test_longitudinal_dataset_audit.py`
- `analysis/clinical_growth_law_validation/run_growth_law_overlay_smoke.py`

The overlay smoke currently shows target-volume propagation succeeds locally,
but the reused local fixture fails anatomical embedding QC because its source
segmentation is much smaller than the pasted target tumors. This is an
engineering fixture limitation, not a clinical finding.
