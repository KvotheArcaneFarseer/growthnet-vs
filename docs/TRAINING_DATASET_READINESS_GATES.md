# Training Dataset Readiness Gates

Last updated: 2026-08-01

Scope: local GrowthNet repository only. These gates define what must be true
before longitudinal synthetic outputs should be treated as training-ready. They
do not certify clinical validity.

## Current Verdict

Status: NOT_TRAINING_READY

The longitudinal generator now has useful engineering controls:

- scenario-based target-volume generation,
- multiple shape variants per timepoint,
- per-visit metadata,
- per-mask QC,
- patient-level longitudinal QC,
- optional provenance JSON,
- variant-diversity tooling.

However, training readiness remains blocked by scientific and dataset-validation
requirements. In particular, real longitudinal vestibular schwannoma masks with
visit dates are not locally available for calibration or validation.

## Required Gates

### Gate 1 - Provenance Completeness

Agent: Provenance Agent

Requirement:

- Every generated dataset must include `metadata.csv`, `qc_summary.csv`,
  `longitudinal_qc_summary.csv`, and provenance JSON.
- Provenance must hash timeline CSV, background CSV, metadata, per-mask QC, and
  longitudinal QC.
- Metadata must include `growth_law_name`, `growth_law_scenario`,
  `target_volume_source`, `variant_id`, `variant_seed`, and source MRI/seg paths.

Current status: PARTIAL_PASS locally.

### Gate 2 - Volume Realization

Agent: Embedding Validation Agent

Requirement:

- Per-mask `relative_volume_error` must pass an engineering smoke threshold.
- Small tumors below 100 mm3 require separate tolerance because voxel
  quantization dominates.
- Scientific acceptance thresholds must remain separate from smoke thresholds.

Current status: PARTIAL_PASS locally.

Evidence:

- `analysis/clinical_growth_law_validation/variant_smoke_output/qc_summary.csv`
  had 7/8 variants pass at 15% tolerance.
- The single failure had RAVD 0.1523 against a 64 mm3 target, consistent with
  small-target quantization sensitivity.

### Gate 3 - Longitudinal Consistency

Agent: Longitudinal QC Agent

Requirement:

- Visit days must be strictly increasing.
- Background MRI ID must be consistent within patient unless explicitly varied.
- Target-volume trend must match the declared scenario.
- Achieved-volume trend should be checked per variant.
- Regression scenarios must be explicitly labeled if volume decreases.

Current status: PARTIAL_PASS locally.

Evidence:

- `longitudinal_qc_summary.csv` is now emitted.
- `longitudinal_qc_summary.csv` now includes machine-readable engineering gate
  fields: `engineering_qc_gate`, `engineering_qc_failure_reasons`,
  `target_volume_trend_status`, and `actual_volume_trend_status`.
- Current smoke reports visit ordering and background consistency pass.
- Achieved-volume monotonicity failed in the stable tiny-target smoke because
  small-target voxelization produced minor visit-to-visit variation.

### Gate 4 - Variant Diversity

Agent: Morphology QA Agent

Requirement:

- Multi-variant outputs must be measured for same-patient/timepoint diversity.
- Pairwise Dice near 1.0 should be flagged as low diversity.
- Low Dice alone should not be interpreted as anatomical realism.
- Diversity thresholds must be volume-stratified.

Current status: TOOL_READY, THRESHOLDS_UNSET.

Evidence:

- `measure_variant_diversity.py` ran on the local smoke.
- Pairwise Dice ranged from 0.8027 to 0.9399; median Dice was 0.8822.

### Gate 5 - Morphology Validity

Agent: Scientific Morphology Agent

Requirement:

- Use regenerated authoritative synthetic features only.
- Do not use stale pulled synthetic feature tables for tuning.
- Surface metrics must be resolution-normalized before real-vs-synthetic
  interpretation.
- Compartment metrics require recovered-generator semantics for saved masks.

Current status: ACCEPT_WITH_FOLLOWUP.

### Gate 6 - Real Longitudinal Calibration

Agent: Scientific Validation Agent

Requirement:

- Real longitudinal VS masks must be available locally.
- Visit dates must be known.
- Tumor volumes must be re-extracted with the same feature code path.
- Scenario probabilities and annual-rate caps must be reviewed against real
  data before training use.

Current status: BLOCKED_REMOTE_DATA / HUMAN_REVIEW_REQUIRED.

### Gate 7 - Training Loader Integration

Agent: Dataset Integration Agent

Requirement:

- Generated image/mask layout must be consumed by the ViViT or downstream
  temporal model dataset wrapper.
- Patient/timepoint/variant grouping must be unambiguous.
- A tiny local training or dataloader smoke must pass.

Current status: NOT_STARTED.

Update:

- `projects/vivit/src/data/synthetic_longitudinal_loader.py` now provides a
  metadata-based adapter that groups generated outputs by
  `(patient_id, variant_id)` and exposes ViViT-style sequence records.
- Focused index tests pass locally.
- A full NIfTI-loading dataloader smoke remains pending.

## Current Local Smoke Evidence

Multi-variant smoke command:

```bash
.venv/bin/python scripts/generate_synthetic_longitudinal_dataset.py \
  --timeline_csv analysis/clinical_growth_law_validation/variant_smoke_timeline.csv \
  --background_csv analysis/clinical_growth_law_validation/variant_smoke_backgrounds.csv \
  --out_dir analysis/clinical_growth_law_validation/variant_smoke_output \
  --seed 20260523 \
  --gen_size 64 \
  --volume_max_iterations 3 \
  --volume_ravd_tolerance 0.15 \
  --variants_per_timepoint 2 \
  --provenance_json analysis/clinical_growth_law_validation/variant_smoke_output/provenance.json
```

Variant-diversity command:

```bash
.venv/bin/python analysis/clinical_growth_law_validation/measure_variant_diversity.py \
  --metadata_csv analysis/clinical_growth_law_validation/variant_smoke_output/metadata.csv \
  --out_csv analysis/clinical_growth_law_validation/variant_smoke_output/variant_diversity.csv \
  --out_report analysis/clinical_growth_law_validation/variant_smoke_output/VARIANT_DIVERSITY.md
```

## Next Actions

1. Review `scenario_mixture_v1` probabilities and annual-rate caps with the
   instructor or scientific lead.
2. Add volume-stratified variant-diversity thresholds after generating a
   non-clinical fixture at small, medium, and large target volumes.
3. Validate generated longitudinal outputs through the downstream model dataset
   loader before any training run.
