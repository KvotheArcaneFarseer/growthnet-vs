# GrowthNet Agent Status

Last updated: 2026-07-18

Scope: local-only orchestration. Agents must not SSH, use Rivanna, delete generated data, or edit outside this repository.

## Launch Status

| Agent | Task IDs | Scope | File Ownership | Status | Branch/Isolation | Last Update |
|---|---|---|---|---|---|---|
| Agent 1 Orientation and Embedding Validation | EMB-001 | diagnose local axis-error mechanism | `analysis/orientation_validation/**` | COMPLETE | sub-agent `019f7394-28f1-70c2-bf00-9f7b4aa1a741` | local diagnostic passed |
| Agent 2 Batch Pipeline Local Reliability | BATCH-001 | local batch resumability/resource audit | `analysis/batch_reliability/**`, optional `scripts/run_batch_embedding.py` | COMPLETE | sub-agent `019f7394-5221-7b01-89fe-68d1ce3a2d51` | resume/status implementation validated locally |
| Agent 3 Longitudinal Dataset Audit | LONG-001 | audit wrapper and metadata/repro tests | `docs/LONGITUDINAL_PIPELINE_AUDIT.md`, `data/timelines/local_longitudinal_example.csv`, `tests/test_longitudinal_*.py` | COMPLETE | sub-agent `019f7394-77ce-7db2-9cad-4d67cd7eeb72` closed | unittest passed; `.venv` pytest suite later passed |
| Agent 4 Volume Targeting Validation | VOL-001 | quantify target-volume accuracy | `analysis/volume_targeting/**` | COMPLETE | sub-agent `019f7394-a4d6-7811-967d-89fba88611e7` | 36-case local standalone benchmark complete |
| Agent 5 Morphology and Real-vs-Synthetic Audit | MORPH-001 | local morphology comparison | `analysis/real_vs_synthetic/**` | COMPLETE | sub-agent `019f7394-c7cb-7111-8227-25f2bf7453ba` | local comparison complete; provenance drift found |
| Agent 6 Testing and Reproducibility | TEST-001 | fast deterministic tests | `tests/**`, `pytest.ini`, optional test docs | COMPLETE | sub-agent `019f7394-eb01-7e40-b956-cd44ff60c414` | `.venv` pytest suite passed |
| Agent 7 Documentation and Repository Audit | DOC-001 | local project docs | `README.md`, `docs/LOCAL_REPRODUCIBILITY.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/CURRENT_PROJECT_STATUS.md` | COMPLETE | sub-agent `019f7397-e294-7581-a140-d63d4353950f` | local docs updated |
| Agent 8 Integration and Review | REVIEW-001 | review queue | `docs/GROWTHNET_AGENT_STATUS.md` | COMPLETE | lead integration review | specialist outputs reviewed and classified |
| Feature Provenance Investigation | MORPH-PROV-001 | resolve synthetic feature provenance drift | `analysis/feature_provenance/**` | COMPLETE | sub-agent `019f73c8-8115-7653-a1f6-e458575b8aff` | stale pulled synthetic features identified |
| Volume Targeting Revalidation | VOL-REVAL-001 | rerun standalone benchmark and threshold split | `analysis/volume_targeting/**` | COMPLETE | sub-agent `019f73c8-a7ef-78e0-88c4-92352c222ddf` | 36-case benchmark complete |
| Synthetic Feature Regeneration | SYNFEAT-001 | regenerate authoritative features from local masks | `analysis/synthetic_features_v2/**` | COMPLETE | local orchestrator | 261/261 features regenerated with provenance and integrity PASS |
| Morphology Validation V2 | MORPH-003 | compare real features to regenerated synthetic features | `analysis/real_vs_synthetic_v2/**` | ACCEPT_WITH_FOLLOWUP | local orchestrator | v2 morphology analysis complete; no generator tuning justified yet |
| Surface Metric Resolution Validation | SURFACE-001 | test spacing/resolution sensitivity of surface metrics | `analysis/surface_resolution_validation/**` | COMPLETE | local orchestrator | 30-case stratified controlled resampling experiment complete |
| Anatomical Compartment Validation | COMPART-001 | derive synthetic canal/stem and CPA/bulb metrics | `analysis/anatomical_compartment_validation/**`, `tests/test_anatomical_compartment_helpers.py` | ACCEPT_WITH_FOLLOWUP | local orchestrator | 261-case synthetic-only compartment audit complete; real validation blocked |
| Saved Mask Provenance Recovery | PROV-MASK-001 | recover code path that generated authoritative saved synthetic masks | `analysis/saved_mask_provenance/**` | COMPLETE | local orchestrator | pulled cohort script reproduced 10/10 representative saved masks exactly |
| Recovered-Generator Compartment Validation | COMPART-002 | recompute compartment metrics using recovered saved-mask generator path | `analysis/anatomical_compartment_validation_v2/**` | COMPLETE | local orchestrator | 261-case v2 audit complete; 12/12 visual mappings confirmed |
| Future Provenance and QC | PROV-FUTURE-001, QC-001 | add future sidecar provenance/labels and artifact/QC index | `scripts/generate_synthetic_lollipop_cohort.py`, `tests/test_synthetic_generation.py`, `docs/CURRENT_AUTHORITATIVE_ARTIFACTS.md`, `docs/LOCAL_QC_DASHBOARD.md` | COMPLETE | local orchestrator | optional provenance/compartment labels validated by tests and smoke run |
| Longitudinal Readiness | LONG-PLAN-001, LONG-PROV-001, LONG-META-001 | plan longitudinal generator readiness and add provenance/metadata traceability | `scripts/generate_synthetic_longitudinal_dataset.py`, `tests/test_longitudinal_*.py`, `docs/LONGITUDINAL_SYNTHETIC_DATASET_ACTION_PLAN.md` | COMPLETE | local orchestrator | optional longitudinal provenance and additive metadata validated locally |
| Clinical Growth Law Encoding | LONG-GROWTH-001 | encode default-off empirical VS volumetric growth-law candidate | `scripts/generate_synthetic_longitudinal_dataset.py`, `tests/test_longitudinal_*.py`, `docs/CLINICAL_GROWTH_LAW_ACTION_PLAN.md` | COMPLETE | local orchestrator | optional `empirical_vs_v1` target-volume mode added; validation remains blocked by real data |

## Review Queue

| Task ID | Review Status | Reviewer Notes | Human Review |
|---|---|---|---|
| EMB-001 | ACCEPT_WITH_FOLLOWUP | Local standalone masks do not support spatial bug or PCA identity switch; patient-space named cases are BLOCKED_REMOTE_DATA. | yes |
| BATCH-001 | ACCEPT_WITH_FOLLOWUP | Resume/status behavior is backward compatible by default and validated locally; scientific smoke lacks clean original MRI/seg fixture. | no |
| LONG-001 | ACCEPT_WITH_FOLLOWUP | Scoped audit and tests. Follow-up needs human review of `stable`/`growing` label semantics. | yes |
| VOL-001 | ACCEPT_WITH_FOLLOWUP | Standalone mask volume targeting revalidated: 36/36 OK, 30/36 within 3% RAVD. Engineering smoke thresholds are supported; final scientific thresholds remain insufficiently evidenced. | yes |
| MORPH-001 | ACCEPT_WITH_FOLLOWUP | Provenance narrowed: pulled synthetic feature tables are stale relative to local masks/manifest. Use local masks plus current extractor as authoritative local path; do not tune morphology from stale pulled features. | yes |
| SYNFEAT-001 | COMPLETE | Authoritative synthetic features regenerated into `analysis/synthetic_features_v2/`: 261 expected, 261 processed, 0 failed; mask/manifest one-to-one and volume integrity passed. | no |
| MORPH-003 | ACCEPT_WITH_FOLLOWUP | Real-vs-synthetic v2 uses regenerated synthetic features and the best available local real feature table. The prior "too elongated" concern is rejected for regenerated features, but real source masks and anatomical canal/CPA labels remain unavailable, so no generator tuning is justified. | yes |
| SURFACE-001 | ACCEPT_WITH_FOLLOWUP | Surface metric resolution validation found surface area, sphericity, compactness, and surface-to-volume ratio are resolution-sensitive even though physical spacing is passed correctly to marching cubes. Synthetic 0.5 mm normalization made sphericity and surface-to-volume gaps disappear and compactness substantially shrink on the selected matched subset. Keep these metrics as resolution-confounded until real source masks can be re-extracted at common spacing. | yes |
| COMPART-001 | ACCEPT_WITH_FOLLOWUP | Synthetic canal/stem and CPA/bulb metrics are derivable locally from generator metadata and current masks. The standalone 261-case cohort shows bulb absence in 191/261 cases and low large-case bulb fraction (median 0.033), flagged as plausibility concerns/design artifacts. Real anatomical validation remains blocked by real masks and porus/fundus or canal-axis annotations. No generator tuning is justified yet. | yes |
| PROV-MASK-001 | ACCEPT_WITH_FOLLOWUP | The saved-mask generation path was recovered for 10/10 representative cases: `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py` plus the manifest exactly reproduces saved masks, while the current local generator gives median Dice 0.887. Prior current-generator compartment mapping conclusions must be rerun with the recovered path before quantitative use. | yes |
| COMPART-002 | ACCEPT_WITH_FOLLOWUP | V2 compartment audit used the recovered pulled generator path. Preflight reproduced 10/10 provenance cases exactly. Corrected zero-bulb count is 135/261, prior 160/261 mismatch flags are an artifact of the wrong generator path, and prior 9/261 multi-component finding is confirmed. Cohort remains stem-dominant; no generator tuning is justified without real compartment validation. | yes |
| PROV-FUTURE-001 | COMPLETE | Future standalone synthetic generation now has default-off provenance JSON and compartment-label sidecars. Existing default outputs remain backward compatible. Focused synthetic tests and a two-case smoke run passed. | no |
| QC-001 | COMPLETE | `docs/CURRENT_AUTHORITATIVE_ARTIFACTS.md` and `docs/LOCAL_QC_DASHBOARD.md` now centralize trusted artifacts, superseded outputs, blockers, and recommended checks. | no |
| LONG-PROV-001 | COMPLETE | Longitudinal wrapper now supports optional `synthetic_longitudinal_provenance_v1` JSON with input/output hashes, code hashes, generation parameters, and QC counts. | no |
| LONG-META-001 | COMPLETE | Longitudinal `metadata.csv` now includes additive traceability fields: visit seed, source MRI/seg paths, embedding growth mode, tolerance, max iterations, and gen size. | no |
| LONG-GROWTH-001 | HUMAN_REVIEW_REQUIRED | `empirical_vs_v1` is encoded and locally tested as a default-off experimental candidate based on published volumetric growth categories. It is not clinically validated and must not be used for training claims without real longitudinal validation and human scientific review. | yes |
| TEST-001 | COMPLETE | `.venv/bin/python -m pytest -m "fast and not slow" -v` passed after growth-law tests were added: 48 passed, 9 deselected, 14 warnings. Warnings are Matplotlib/PyParsing dependency deprecations, not GrowthNet test failures. This establishes local engineering test health only, not scientific validity. | no |
| DOC-001 | ACCEPT_WITH_FOLLOWUP | Local docs now distinguish implemented, partial, experimental, unvalidated, and remote-blocked work; scientific wording should be reviewed before publication use. | yes |

## Coordination Rules In Force

- No concurrent core edits to `embed_tumor.py` or `projects/vivit/src/data/synthetic.py`.
- No generator morphology tuning until `MORPH-001` evidence exists and is reviewed.
- No QC redesign until `EMB-001` diagnosis is supported.
- No full training-ready cohort generation in this session.
- No destructive cleanup.
- No remote data assumptions. Mark remote-only work `BLOCKED_REMOTE_DATA`.

## Current Validation Notes

- Test validation: COMPLETE for selected local fast/non-slow suite in `.venv` on Python 3.9.6 / pytest 8.4.2; current result is 48 passed, 9 deselected, 14 dependency deprecation warnings.
- Feature provenance: COMPLETE for local synthetic artifacts. Pulled synthetic feature CSV/JSON are STALE; local masks and manifest are AUTHORITATIVE for local reproduction.
- Feature regeneration: COMPLETE. `analysis/synthetic_features_v2/synthetic_features_v2.csv` is authoritative for current local synthetic masks; legacy pulled synthetic feature artifacts remain stale and untouched.
- Morphology validation v2: ACCEPT_WITH_FOLLOWUP. Regenerated features reject the prior overall "too elongated" concern, while surface-derived differences and whole-mask PCA/bbox differences require human scientific review before tuning.
- Surface metric resolution validation: COMPLETE locally, ACCEPT_WITH_FOLLOWUP scientifically. Current extractor passes physical spacing into marching cubes correctly, but label-mask surface metrics remain resolution-sensitive. In a 30-case stratified synthetic resampling experiment, 1.0 mm to 0.5 mm normalization shifted median surface area by +18.0%, sphericity by -15.2%, compactness by +64.1%, and surface-to-volume ratio by +18.0%. Surface-derived real-vs-synthetic gaps should be compared only after resolution normalization and real source-mask re-extraction when available.
- Anatomical compartment validation: ACCEPT_WITH_FOLLOWUP. The original compartment audit is superseded for quantitative cohort counts by `analysis/anatomical_compartment_validation_v2/`, which uses the recovered saved-mask generator path. V2 found 135/261 zero-bulb cases, 0/261 cases with >1% unmatched compartment voxels, and 9/261 multi-component masks. This remains a synthetic plausibility audit, not real anatomical validation.
- Saved-mask provenance recovery: COMPLETE for the 10-case representative recovery search. The pulled cohort generator script exactly reproduces all selected saved masks with manifest seed, final scale, target volume, and rotations. The current checked-in generator does not reconstruct those masks, so current-generator compartment mapping outputs must not be treated as authoritative for the saved cohort.
- Future provenance and QC: COMPLETE for standalone synthetic generation. New flags `--provenance_json` and `--write_compartment_labels` add auditable future sidecars without changing default behavior. Artifact/QC dashboard docs were added.
- Longitudinal readiness: COMPLETE for local provenance and additive metadata. Sparse/irregular visits, temporal QC, model export, and real longitudinal validation remain open.
- Clinical growth-law encoding: COMPLETE as a default-off engineering feature, HUMAN_REVIEW_REQUIRED scientifically. Real longitudinal masks are required to validate or recalibrate parameters.
- Volume targeting: ACCEPT_WITH_FOLLOWUP. Standalone generator smoke thresholds are supportable; scientific acceptance thresholds require broader embedded and cohort validation.

## Required Agent Report Format

Each agent final report must include:

- assigned task IDs
- work performed
- files changed
- commands run
- tests passed
- tests failed
- generated outputs
- unresolved risks
- new subtasks discovered
- whether human review is needed
## 2026-08-01 - Growth Scenario / Variant Agents

| Agent | Scope | Status | Evidence | Follow-up |
|---|---|---|---|---|
| Literature Agent | Online VS growth-model scan and scenario framing | ACCEPT_WITH_FOLLOWUP | `docs/GROWTH_SCENARIO_VARIANT_ACTION_PLAN.md` | Convert source table into a stricter citation matrix before publication claims. |
| Clinical Modeling Agent | Scenario-based target-volume controls | COMPLETE | `scripts/generate_synthetic_longitudinal_dataset.py`, `tests/test_longitudinal_helpers.py` | Calibrate scenario weights only after real longitudinal masks are available. |
| Dataset Orchestration Agent | Multiple same-timepoint shape variants | COMPLETE | `--variants_per_timepoint`, `variant_id`, `variant_seed`, mocked integration test | Add shape-diversity metrics once non-fixture local MRI/seg pairs exist. |
| Embedding Validation Agent | Target-to-pasted-mask local smoke | ACCEPT_WITH_FOLLOWUP | `analysis/clinical_growth_law_validation/GROWTH_LAW_OUTPUT_VALIDATION.md` | Repeat on a clinically meaningful local source MRI/seg subset. |
| Longitudinal QC Agent | Patient-level QC summary | COMPLETE | `longitudinal_qc_summary.csv`, provenance hash, focused tests | Add scenario-specific pass/fail gates after human review of growth semantics. |
| Morphology QA Agent | Same-timepoint variant diversity analyzer | COMPLETE | `analysis/clinical_growth_law_validation/measure_variant_diversity.py`, `tests/test_longitudinal_variant_diversity.py` | Run on real multi-variant outputs once a non-fixture local MRI/seg pair is staged. |
| Clinical Modeling Agent | No-MRI scenario sampling audit | HUMAN_REVIEW_REQUIRED | `analysis/clinical_growth_law_validation/SCENARIO_SAMPLING_AUDIT.md` | Review scenario probabilities and annual-rate caps before training use. |

## 2026-08-04 Architecture Pass

| Agent | Scope | Status | Evidence | Follow-up |
|---|---|---|---|---|
| Longitudinal QC Agent | Machine-readable engineering QC gates | COMPLETE | `engineering_qc_gate`, failure reasons, trend statuses, focused tests | Keep scientific acceptance thresholds separate from engineering smoke gates. |
| Dataset Integration Agent | Metadata-to-ViViT sequence adapter | PARTIAL_COMPLETE | `projects/vivit/src/data/synthetic_longitudinal_loader.py`, `tests/test_synthetic_longitudinal_loader.py` | Run full NIfTI/tensor loading smoke before training use. |
| Release Engineering Agent | Dirty worktree commit grouping guidance | COMPLETE | inspection-only subagent report | Stage commits by purpose; do not mix source, docs, analysis artifacts, and pulled outputs. |
