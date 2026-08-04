# Current Project Status

Last updated: 2026-08-04

Scope: local repository state only. No Rivanna, SSH, SLURM, remote datasets, or private clinical data were assumed for this status.

## Maturity

GrowthNet is a research-prototype-plus repository. The local embedding and synthetic generation pipeline is operational, and several validation audits now exist. The project is not yet a fully validated clinical synthetic dataset factory, not yet a reproducible training dataset release, and not yet publication-ready.

## Implemented

- `embed_tumor.py`: single-case synthetic vestibular schwannoma embedding with lollipop geometry, spacing-aware placement, orientation selection, QC images, NIfTI outputs, and metrics reports.
- `projects/vivit/src/data/synthetic.py`: synthetic time-series generator with lollipop and legacy geometry support.
- `scripts/run_batch_embedding.py`: CSV batch runner with local resource thread caps, per-case summaries, failure reports, and resume support.
- `scripts/generate_synthetic_lollipop_cohort.py`: standalone target-volume synthetic mask generation with optional provenance JSON and optional stem/transition/CPA-bulb compartment-label sidecars for future cohorts.
- `scripts/extract_real_tumor_features.py`: spacing-aware mask feature extraction and summary generation.
- `scripts/generate_synthetic_longitudinal_dataset.py`: MVP four-visit longitudinal wrapper over the embedding engine with additive per-visit audit metadata and optional dataset provenance JSON.
- `projects/vivit/src/data/synthetic_longitudinal_loader.py`: metadata-based adapter that groups generated longitudinal outputs into ViViT-style temporal sequence records without rewriting generated artifacts.
- `shared/provenance.py` and `shared/reporting.py`: shared provenance and reporting helpers now used by active synthetic cohort and longitudinal generation scripts.
- `.github/workflows/fast-tests.yml`: laptop-safe fast-test CI workflow.
- `analysis/orientation_validation/`: local orientation diagnostic artifacts.
- `analysis/real_vs_synthetic/`: historical local matched morphology benchmark, plots, and report; retained for provenance.
- `analysis/synthetic_features_v2/`: authoritative regenerated synthetic feature dataset for current local masks, with preflight validation, per-case hashes, extraction summary, provenance, and integrity report.
- `analysis/real_vs_synthetic_v2/`: real-versus-regenerated-synthetic morphology validation using only authoritative synthetic features and the best available local real feature table.
- `analysis/surface_resolution_validation/`: controlled local resolution experiment for surface-sensitive morphology metrics, including metric-definition audit, resampling benchmark, normalized comparison, plots, and report.
- `analysis/anatomical_compartment_validation/`: original synthetic canal/stem and CPA/bulb compartment plausibility audit. Its quantitative cohort counts are now exploratory/superseded because it used the current checked-in generator path for reconstruction.
- `analysis/anatomical_compartment_validation_v2/`: recovered-generator synthetic compartment audit using the exact local pulled generator path that reproduces the saved masks.
- `analysis/saved_mask_provenance/`: local saved-mask generation provenance recovery, including evidence inventory, generator history, representative reproduction trials, and final report.
- `analysis/volume_targeting/`: local volume-targeting benchmark helper and smoke artifacts.
- `docs/CURRENT_AUTHORITATIVE_ARTIFACTS.md`: authoritative local artifact index distinguishing current, historical, stale, and superseded outputs.
- `docs/LOCAL_QC_DASHBOARD.md`: compact engineering/scientific QC dashboard and blocker summary.
- `docs/LONGITUDINAL_SYNTHETIC_DATASET_ACTION_PLAN.md`: longitudinal readiness roadmap with agent-owned tasks and remaining scientific blockers.
- `docs/CLINICAL_GROWTH_LAW_ACTION_PLAN.md`: clinical growth-law encoding and validation roadmap with literature basis and review gates.
- `tests/`: focused local pytest tests for embedding helpers, batch helpers, synthetic generation, feature extraction, and longitudinal helpers.

## Partially Complete

- Local batch pipeline reliability: resume/status behavior exists, but broader multi-case local validation depends on available local NIfTI inputs.
- Longitudinal dataset generation: metadata, QC outputs, per-visit seed/source/generation metadata, optional provenance JSON, explicit patient-level engineering QC gates, a metadata-based ViViT sequence adapter, and a default-off experimental empirical VS volumetric growth-law mode exist. Sparse arbitrary-row timelines, continuous shape trajectory semantics, full NIfTI dataloader smoke, and real-data clinical validation are still missing.
- Volume targeting: implementation exists, and smoke benchmark artifacts exist, but recommended production thresholds still require broader review.
- Morphology validation: v2 comparison now uses regenerated authoritative synthetic features, surface metric resolution sensitivity has been quantified locally, and recovered-generator synthetic compartment statistics exist. Real source-mask re-extraction and real anatomical compartment validation remain unavailable.
- Synthetic feature provenance: pulled synthetic feature CSV/JSON are classified as stale relative to local masks; regenerated v2 synthetic features are now the authoritative local synthetic feature table.
- ViViT/TemporalUNETR integration: model and legacy split-folder loader code exists. Generated longitudinal metadata can now be converted into ViViT-style sequence records by adapter, but full NIfTI loading and local training have not yet been proven.
- Architecture remediation: dependency manifest, fast CI workflow, and first shared helper layer exist. Larger package splitting, config migration, artifact hygiene, and documentation consolidation remain open.
- Documentation: this local status layer is current, while several project sub-READMEs remain historical templates.

## Experimental

- Orientation diagnosis for large-tumor axis errors.
- Lollipop morphology calibration and benchmarking.
- Volume-targeting parameter benchmarking.
- Napari visualization and animation scripts.
- Lab meeting deck/3D export scripts.
- Graphify/Obsidian architecture exports.
- MRI registration notebooks and HPC run scripts.

## Locally Validated Evidence

- Core script syntax/import smoke was reported passing for embedding, batch, feature extraction, synthetic cohort generation, longitudinal generation, and synthetic data helper modules.
- Longitudinal helper behavior has deterministic local unit tests from `tests/test_longitudinal_dataset_audit.py`; a unittest run passed for those audit tests.
- Orientation audit found that local standalone masks for `126_5_1607` and `132_1_148` align their whole-mask major PCA axis with the manifest-derived canal line. The missing patient-MRI embedding data prevents spatial bug confirmation for those named cases.
- Morphology audit compared 261 matched local synthetic benchmark rows against pulled real feature rows. The pulled synthetic feature table is now known to be stale relative to local masks and manifest, so those historical morphology gaps must not drive tuning until regenerated current features are used.
- Feature provenance audit found local masks match manifest realized volumes for 261/261 cases, while pulled synthetic feature volumes match local mask volumes for only 9/261 cases. The current extractor is byte-identical to the pulled extractor copy, narrowing the likely cause to stale features paired with newer masks/manifest.
- Synthetic feature regeneration v2 processed 261/261 authoritative local masks with 0 extraction failures. Integrity checks passed: unique case IDs, readable masks, valid spacing, non-empty masks, no NaN/Inf values, and exact feature-volume agreement with direct mask-derived volume.
- Morphology validation v2 compared 261 matched case IDs using regenerated synthetic features. Volume matching is strong (median synthetic/real volume ratio 0.988). The prior "synthetic too elongated" concern is rejected for regenerated features overall: median elongation ratio 0.883, aspect-ratio major/minor2 ratio 0.724, and major-axis length ratio 0.689.
- Highest-confidence v2 morphology differences are not tuning-ready scientific findings: compactness/sphericity differences are classified as extraction/data limitations; aspect-ratio and bbox-fill differences are possible whole-mask generator gaps requiring human scientific review.
- Surface metric resolution validation used 30 stratified synthetic masks across native 1.0 mm isotropic, 0.5 mm isotropic, and 0.5 x 0.5 x 1.0 mm anisotropic conditions. The extractor passes physical spacing correctly into marching cubes, but surface metrics remain resolution-sensitive. Resampling synthetic masks from 1.0 mm to 0.5 mm shifted median surface area by +18.0%, sphericity by -15.2%, compactness by +64.1%, and surface-to-volume ratio by +18.0%. In the selected matched comparison, sphericity and surface-to-volume gaps disappeared after synthetic 0.5 mm normalization, and compactness substantially shrank. These findings support resolution as a substantial confounder, not as proof of final morphology realism.
- Saved-mask provenance recovery tested 10 representative cases and found that `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py`, with the pulled manifest seed, final scale, target volume, and rotations, reproduces all selected saved masks exactly. The current checked-in generator reproduces the same masks with median Dice 0.887, so it is not the correct reconstruction path for this saved cohort.
- Recovered-generator compartment validation reran the synthetic canal/stem and CPA/bulb audit for all 261 authoritative saved masks using `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py`. Preflight reproduced 10/10 representative cases exactly. Corrected results: 135/261 zero-bulb masks, 0/261 cases with >1% unmatched compartment voxels, 9/261 multi-component masks, and 12/12 fresh visual overlays classified `MAPPING_CONFIRMED`. The prior 191/261 zero-bulb count is rejected, the prior 160/261 mismatch flags are an artifact of the wrong generator path, and the prior 9/261 multi-component finding is confirmed.
- Standalone volume-targeting benchmark was rerun on 36 local cases. All cases completed; 30/36 were within 3% RAVD. Performance was strong at and above 100 mm3 and degraded below 100 mm3 due to voxel quantization.
- Future synthetic cohort generation now has tested optional provenance and compartment-label exports. A two-case local smoke run wrote masks, compartment NIfTIs, manifest label columns, and `synthetic_lollipop_provenance_v1` JSON successfully.
- Longitudinal wrapper provenance and additive metadata were validated locally. Focused tests passed, and a no-MRI smoke run wrote failure rows plus `synthetic_longitudinal_provenance_v1` JSON with patient/timepoint/QC pass-fail counts.
- Experimental `empirical_vs_v1` clinical growth-law encoding was added as an explicit optional mode. It derives `T1..T4` target volumes from baseline `T1_volume_mm3`, visit days, and a deterministic annual volumetric rate sampled from published stable/growing categories. This is encoded, not clinically validated.
- Selected fast/non-slow local pytest suite passed in `.venv` after shared-helper migration: 62 passed, 13 deselected, 14 Matplotlib/PyParsing deprecation warnings.
- Batch runner local smoke artifacts exist under `analysis/batch_reliability/`.

## Remotely Validated Or Pulled Provenance

The repository includes `rivanna_pull/` artifacts, including real feature summaries and synthetic benchmark feature tables. These are local copies of prior outputs, not fresh local re-extractions from original real source masks. Treat them as provenance/evidence artifacts, not as proof that this laptop contains the full source dataset.

## Unvalidated

- Full real multi-patient embedding robustness.
- Original-source real feature extraction equivalence.
- Clinical validity of canal/CPA morphology and orientation metrics.
- Real compartment-level validation of intracanalicular/canal-stem and CPA/bulb relationships.
- Final scientific volume-targeting acceptance thresholds.
- Final scientific interpretation of surface-derived real-vs-synthetic metrics without common-resolution real-mask re-extraction.
- Longitudinal clinical realism.
- Clinical validity of `empirical_vs_v1` growth-law parameters.
- Training-ready dataset generation at scale.
- Full local ViViT NIfTI-loading smoke, training, or inference on generated longitudinal outputs.
- Publication-ready reproducibility for all figures/tables.

## Current Working Tree Notes

The working tree is intentionally active and contains source edits, tests, documentation, analysis scripts, generated plots, and pulled artifacts. Do not assume `git status` is clean. Avoid destructive cleanup. Review file ownership in `docs/GROWTHNET_ACTION_PLAN.md` before editing.

Known modified or newly added areas during the local orchestration include:

- `.gitignore`
- `embed_tumor.py`
- `projects/vivit/src/data/synthetic.py`
- `scripts/run_batch_embedding.py`
- `scripts/generate_synthetic_longitudinal_dataset.py`
- `scripts/generate_synthetic_lollipop_cohort.py`
- `scripts/extract_real_tumor_features.py`
- `tests/`
- `analysis/`
- `docs/`
- `data/timelines/`

## Recommended Local Priorities

1. Choose the real compartment annotation standard for validation: porus/fundus landmarks, canal-axis vector plus porus point, IAC mask, or explicit CPA boundary.
2. Obtain or stage a 30-case real-mask subset with 10 small, 10 medium, and 10 large tumors for common-resolution compartment validation.
3. Compare recovered-generator synthetic compartment distributions against annotated real masks once available.
4. Add longitudinal growth-law QC and validate `empirical_vs_v1` against real longitudinal masks when available.
5. Resolve feature metric-definition issues before using PCA/elongation/sphericity outputs as scientific acceptance metrics.
6. Expand embedded-pipeline volume-targeting validation using local fixtures; keep standalone smoke thresholds separate from scientific thresholds.
7. Keep documentation and status reports evidence-bound; do not promote local engineering test success to clinical/scientific validity.
8. Mark any work needing unavailable clinical files as `BLOCKED_REMOTE_DATA`.
## 2026-08-01 - Longitudinal Growth Scenario Status

The longitudinal wrapper now supports explicit growth-scenario target generation
instead of a single asserted clinical law. `none` remains the default and
preserves CSV-specified target volumes. `empirical_vs_v2` and
`scenario_mixture_v1` are local experimental controls for generating plausible
volume trajectories; they are not clinical predictors.

The wrapper also supports `--variants_per_timepoint`, allowing multiple
independently seeded tumor shapes for the same patient/timepoint/target volume.
This addresses the dataset requirement for multiple possible tumor shapes at the
same elapsed time period while keeping target-volume provenance auditable.

The wrapper now writes `longitudinal_qc_summary.csv`, a patient-level summary
covering timepoint count, variant count, background consistency, visit ordering,
target-volume monotonicity, per-variant achieved-volume monotonicity where
available, QC pass/fail counts, maximum absolute relative volume error,
machine-readable engineering QC gate status, failure reasons, and trend-status
fields.

A metadata-based adapter now exists at
`projects/vivit/src/data/synthetic_longitudinal_loader.py`. It groups
multi-variant longitudinal outputs by `(patient_id, variant_id)` so same-period
shape variants become separate temporal trajectories. Adapter index tests pass;
full NIfTI-loading and training smoke remain pending.

Variant-diversity tooling now exists at
`analysis/clinical_growth_law_validation/measure_variant_diversity.py` for
pairwise Dice/voxel-count comparisons among same-patient/timepoint variants.
A no-MRI scenario sampler exists at
`analysis/clinical_growth_law_validation/sample_growth_scenarios.py`.
Training-readiness gates are documented in
`docs/TRAINING_DATASET_READINESS_GATES.md`.

Local output validation shows law-derived targets can propagate into pasted MRI
mask volumes within smoke-test tolerance on the available local fixture. The
same fixture is not anatomically suitable for clinical embedding QC because the
source segmentation is tiny relative to the generated targets.

The 200-patient no-MRI scenario sampling audit found `scenario_mixture_v1`
produced 44 fast-growth, 92 moderate-growth, 45 slow-growth, and 19 regression
patients from a 100 mm3 baseline. The moderate-growth upper clamp can still
produce a 1064.8 mm3 T4 target over three years, so scenario probabilities and
annual-rate caps require human scientific review before training use.

A small local multi-variant smoke using the existing embedded fixture generated
one stable synthetic patient, four visits, and two variants per visit. Seven of
eight variants passed the loose 15% volume smoke threshold; the single miss was
RAVD 0.1523 for a 64 mm3 target. Pairwise same-timepoint variant Dice ranged
from 0.8027 to 0.9399 with median 0.8822. This confirms the variant plumbing and
diversity analyzer work locally, but it remains a fixture-only engineering test.
