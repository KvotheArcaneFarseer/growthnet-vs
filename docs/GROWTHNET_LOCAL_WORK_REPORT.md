# GrowthNet Local Work Report

Last updated: 2026-07-18

## Session Objective

Create a local-only master action plan, launch safe parallel specialist agents, execute local work where possible, and consolidate engineering and scientific findings without using Rivanna or remote datasets.

## Initial Repository Findings

- Branch: `main`.
- Recent commit at the time of this update: `b7db1cc Fix batch audit regressions`.
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

Result: 3 unittest tests passed. Later validation found pytest available in the repo-local `.venv`.

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

Revalidation on 2026-07-18:
- `analysis/volume_targeting/run_volume_targeting_benchmark.py` CLI was inspected before execution.
- Command run: `MPLCONFIGDIR=/tmp/growthnet_volume_targeting_mplconfig python3 analysis/volume_targeting/run_volume_targeting_benchmark.py --targets 25,50,75,100,250,500,1000,4000,16000 --seeds 20260426,20260427 --spacings '1,1,1;0.6,0.6,1.2' --out_csv analysis/volume_targeting/volume_targeting_benchmark.csv`.
- Result: 36/36 status OK; 30/36 converged within 3% RAVD.
- Stratified result: `<100 mm3` had 6/12 within 3%, median RAVD 3.33%, max 8.00%; `100-1000 mm3` had 16/16 within 3%, max RAVD 2.71%; `>1000 mm3` had 8/8 within 3%, max RAVD 2.49%.
- Spacing result: anisotropic `0.6 x 0.6 x 1.2` did not degrade this benchmark and improved tiny-volume quantization in some cases.
- Threshold interpretation: engineering smoke thresholds are now explicit; evidence remains insufficient for final scientific acceptance thresholds.

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

## Integrated Specialist Result: MORPH-PROV-001

Status: COMPLETE for local provenance narrowing; ACCEPT_WITH_FOLLOWUP for morphology validation.

Files added:
- `analysis/feature_provenance/FEATURE_PROVENANCE_REPORT.md`
- `analysis/feature_provenance/feature_comparison.csv`
- `analysis/feature_provenance/artifact_inventory.csv`
- `analysis/feature_provenance/provenance_decision_table.csv`
- supporting selected-case and evidence files under `analysis/feature_provenance/`

Provenance finding:
- The pulled synthetic feature table is stale relative to the local synthetic masks.
- Local masks match the pulled manifest `realized_volume_mm3` for 261/261 cases.
- Pulled feature volumes match local mask volumes for only 9/261 cases.
- The current extractor is byte-identical to the pulled extractor copy, so extractor-code drift is not the primary cause.
- The most likely cause is stale features generated from an older generator/mask run paired with newer masks/manifest.

Artifact decisions:
- AUTHORITATIVE for local reproduction: `rivanna_pull/analysis/synthetic_lollipop_v1/masks`, `rivanna_pull/analysis/synthetic_lollipop_v1/manifests/synthetic_lollipop_manifest.csv`, and `scripts/extract_real_tumor_features.py`.
- STALE: pulled synthetic feature CSV/JSON/summary under `rivanna_pull/analysis/synthetic_lollipop_v1/`.
- REPRODUCIBLE_LEGACY: `analysis/real_vs_synthetic/synthetic_features_reextracted_local.csv` and the pulled legacy generator copy as historical context.
- UNKNOWN_PROVENANCE: older masks implied by the stale pulled feature CSV but not present locally.

Decision:
- Current synthetic feature tables from the pulled artifact set should not be trusted as authoritative for local masks.
- Morphology tuning remains blocked until regenerated current features are treated as the benchmark source or the exact older masks are recovered.

## Integrated Specialist Result: TEST-001

Status: COMPLETE.

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

Initial validation rerun by lead:
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m compileall tests`

Result: passed.

Full selected local validation:
- Python: `Python 3.9.6` from `.venv/bin/python`.
- Pytest: `pytest 8.4.2`.
- Command: `.venv/bin/python -m pytest -m "fast and not slow" -v`.
- Result: 39 passed, 9 deselected, 14 warnings in 2.30s.
- Warning classification: Matplotlib/PyParsing dependency deprecation warnings only; no GrowthNet test failures.

Interpretation:
- This completes local engineering validation for the selected fast/non-slow test suite.
- It does not establish scientific validity, morphology realism, clinical longitudinal validity, or training-dataset readiness.

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
- System `python3` lacks pytest, but the repo-local `.venv` now runs the selected pytest suite successfully.

## Consolidated Findings

Engineering:
- The local repository is now better organized for autonomous continuation: plan, status, local report, reproducibility docs, limitations docs, batch reliability docs, testing docs, and focused analysis outputs exist.
- Batch execution is more recoverable locally through opt-in resume and per-case JSONL status.
- The selected local pytest suite now passes in the repo-local `.venv`.
- Feature provenance now has an explicit local source of truth: local masks plus current extractor, with future outputs needing commit/input-hash versioning.
- GitNexus MCP tools required by repository policy were not exposed in this session, so symbol-level impact/detect checks could not be run.

Scientific:
- Local standalone orientation evidence does not support a synthetic-space spatial orientation bug for the named problematic masks; embedded patient-space diagnosis remains blocked by missing local MRI/seg/output data.
- Standalone mask volume targeting is quantitatively strong at or above 100 mm3 in the small local grid, weaker below 100 mm3 due to voxel quantization.
- Real-vs-synthetic morphology analysis is locally supportable only from pulled feature artifacts and local synthetic masks. Provenance/extractor drift is a hard blocker before generator tuning.
- Synthetic feature provenance drift is now narrowed: the stale pulled synthetic feature tables likely came from older mask instances, not from current extractor drift.
- Longitudinal generation is an MVP orchestration wrapper, not a clinically validated growth model.

## Current Task Classification

- `PLAN-001`: COMPLETE.
- `EMB-001`: ACCEPT_WITH_FOLLOWUP.
- `BATCH-001`: ACCEPT_WITH_FOLLOWUP.
- `LONG-001`: ACCEPT_WITH_FOLLOWUP.
- `VOL-001`: ACCEPT_WITH_FOLLOWUP.
- `MORPH-001`: ACCEPT_WITH_FOLLOWUP; historical morphology audit is superseded for synthetic features by v2 regenerated authoritative outputs.
- `SYNFEAT-001`: COMPLETE; authoritative synthetic features regenerated from current local masks into `analysis/synthetic_features_v2/` with provenance and integrity PASS.
- `MORPH-003`: ACCEPT_WITH_FOLLOWUP; v2 morphology analysis uses regenerated synthetic features and rejects the prior overall "too elongated" concern, but does not justify generator tuning.
- `SURFACE-001`: ACCEPT_WITH_FOLLOWUP; local controlled resampling shows surface-derived morphology gaps are strongly resolution-confounded.
- `COMPART-001`: ACCEPT_WITH_FOLLOWUP; synthetic canal/stem and CPA/bulb metrics were derived locally, but real anatomical validation remains blocked by missing real masks and landmarks.
- `TEST-001`: COMPLETE.
- `DOC-001`: ACCEPT_WITH_FOLLOWUP.
- `REVIEW-001`: COMPLETE.

## Integrated Specialist Result: SYNFEAT-001

Status: COMPLETE.

Files created:
- `analysis/synthetic_features_v2/synthetic_features_v2.csv`
- `analysis/synthetic_features_v2/synthetic_features_v2.json`
- `analysis/synthetic_features_v2/extraction_summary.json`
- `analysis/synthetic_features_v2/provenance.json`
- `analysis/synthetic_features_v2/preflight_validation.csv`
- `analysis/synthetic_features_v2/FEATURE_INTEGRITY_REPORT.md`
- `analysis/synthetic_features_v2/regenerate_synthetic_features_v2.py`

Validation result:
- Expected masks: 261.
- Manifest rows: 261.
- Mask files found: 261.
- Unique case IDs: 261.
- Readable masks: 261.
- Valid spacing: 261.
- Non-empty masks: 261.
- Manifest realized volume matches direct mask volume: 261/261.
- Extracted feature rows: 261.
- Extraction failures: 0.
- NaN/Inf numeric values: 0.
- Feature volume versus direct mask-derived volume max absolute difference: 0.0 mm3.

Interpretation:
- `analysis/synthetic_features_v2/synthetic_features_v2.csv` is now the authoritative local synthetic feature table for the current local masks.
- Legacy pulled synthetic feature CSV/JSON artifacts remain stale and were not overwritten.
- This validates feature integrity and provenance, not morphology realism.

## Integrated Specialist Result: MORPH-003

Status: ACCEPT_WITH_FOLLOWUP.

Files created:
- `analysis/real_vs_synthetic_v2/MORPHOLOGY_VALIDATION_REPORT.md`
- `analysis/real_vs_synthetic_v2/matched_distribution_comparison.csv`
- `analysis/real_vs_synthetic_v2/volume_stratified_comparison.csv`
- `analysis/real_vs_synthetic_v2/ranked_morphology_gaps.csv`
- `analysis/real_vs_synthetic_v2/plots/*.png`
- `analysis/real_vs_synthetic_v2/analyze_morphology_v2.py`

Validation result:
- Real feature rows available locally: 291.
- Regenerated synthetic feature rows: 261.
- Matched case IDs analyzed: 261.
- Requested morphology metrics are present in both tables.
- Median synthetic/real volume ratio: 0.988.
- Median synthetic/real elongation ratio: 0.883.
- Median synthetic/real aspect-ratio major/minor2 ratio: 0.724.
- Median synthetic/real major-axis length ratio: 0.689.

Scientific finding:
- The prior concern that regenerated synthetic tumors are overall "too elongated" is rejected for the current authoritative features.
- The top ranked difference is compactness, but it is classified as `EXTRACTION_OR_DATA_LIMITATION` because surface metrics are spacing/resolution-sensitive and real masks cannot be re-extracted locally.
- Aspect ratio and bbox-fill differences are `POSSIBLE_GENERATOR_GAP`, not high-confidence generator gaps.
- No generator parameter change is justified in this pass.

## Integrated Specialist Result: SURFACE-001

Status: ACCEPT_WITH_FOLLOWUP.

Files created:
- `analysis/surface_resolution_validation/run_surface_resolution_validation.py`
- `analysis/surface_resolution_validation/METRIC_DEFINITION_AUDIT.md`
- `analysis/surface_resolution_validation/surface_resolution_experiment.csv`
- `analysis/surface_resolution_validation/surface_metric_sensitivity_summary.csv`
- `analysis/surface_resolution_validation/normalized_morphology_comparison.csv`
- `analysis/surface_resolution_validation/SURFACE_RESOLUTION_REPORT.md`
- `analysis/surface_resolution_validation/provenance.json`
- `analysis/surface_resolution_validation/plots/*.png`

Validation result:
- Controlled subset: 30 synthetic cases, stratified as 10 small, 10 medium, and 10 large.
- Conditions tested: native 1.0 mm isotropic, 0.5 mm isotropic, and 0.5 x 0.5 x 1.0 mm anisotropic.
- Experiment rows extracted: 90.
- Missing selected source masks: 0.
- Empty resampled masks: 0.
- Maximum resampled physical volume error fraction: 0.0000.
- Source masks were read only; generated resampled masks were temporary extraction inputs.

Validation commands:
- `.venv/bin/python analysis/surface_resolution_validation/run_surface_resolution_validation.py`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/surface_resolution_validation/run_surface_resolution_validation.py`
- `.venv/bin/python -m pytest -m "fast and not slow" -v`
- `git diff --check`

Validation command results:
- Surface resolution script passed.
- Py-compile passed with `PYTHONPYCACHEPREFIX` pointed at `/tmp`; plain system `python3 -m py_compile` attempted to write bytecode under `~/Library/Caches` and was blocked by the local sandbox.
- Fast pytest suite passed at that point: 39 passed, 9 deselected, 14 Matplotlib/PyParsing dependency deprecation warnings.
- Diff whitespace check passed.

Metric-definition result:
- The extractor uses `skimage.measure.marching_cubes(..., spacing=spacing_mm)` on the largest connected component and computes mesh area in physical mm2.
- Sphericity, compactness, and surface-to-volume ratio use largest-component physical volume.
- Anisotropic spacing is handled physically by the implementation, assuming NIfTI header zooms are correct.
- Surface estimates remain resolution-sensitive because the binary label boundary and resulting mesh change with voxel size.

Scientific finding:
- 1.0 mm to 0.5 mm synthetic resampling changed median surface area by +18.0%, sphericity by -15.2%, compactness by +64.1%, and surface-to-volume ratio by +18.0%.
- Native selected-subset synthetic/real median ratios were 1.267 for sphericity and 0.492 for compactness.
- After synthetic 0.5 mm normalization, selected-subset synthetic/real median ratios changed to 1.071 for sphericity and 0.813 for compactness.
- Sphericity and surface-to-volume gaps disappeared under the local normalization rule; compactness substantially shrank.
- These results support resolution/spacing as a substantial confounder, but do not prove it is the only cause because real source masks are unavailable locally.
- No generator tuning is justified from surface metrics in the current evidence.

## Integrated Specialist Result: COMPART-001

Status: ACCEPT_WITH_FOLLOWUP.

Files created:
- `analysis/anatomical_compartment_validation/analyze_synthetic_compartments.py`
- `analysis/anatomical_compartment_validation/LOCAL_ANATOMY_INVENTORY.md`
- `analysis/anatomical_compartment_validation/COMPARTMENT_METRIC_SPEC.md`
- `analysis/anatomical_compartment_validation/REAL_MASK_ACQUISITION_SPEC.md`
- `analysis/anatomical_compartment_validation/SYNTHETIC_COMPARTMENT_AUDIT.md`
- `analysis/anatomical_compartment_validation/synthetic_compartment_features.csv`
- `analysis/anatomical_compartment_validation/synthetic_compartment_summary.csv`
- `analysis/anatomical_compartment_validation/provenance.json`
- `analysis/anatomical_compartment_validation/plots/*.png`
- `tests/test_anatomical_compartment_helpers.py`

Validation result:
- Authoritative synthetic masks analyzed: 261/261.
- Volume strata: 79 small, 111 medium, 71 large.
- Helper tests for local coordinate convention and compartment classification passed.
- Synthetic compartment labels were derived from generator metadata and current masks; no generator code or authoritative feature artifacts were modified.

Validation commands:
- `MPLCONFIGDIR=analysis/anatomical_compartment_validation/.matplotlib-cache XDG_CACHE_HOME=analysis/anatomical_compartment_validation/.matplotlib-cache .venv/bin/python analysis/anatomical_compartment_validation/analyze_synthetic_compartments.py`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/anatomical_compartment_validation/analyze_synthetic_compartments.py tests/test_anatomical_compartment_helpers.py`
- `.venv/bin/python -m pytest tests/test_anatomical_compartment_helpers.py -v`
- `.venv/bin/python -m pytest -m "fast and not slow" -v`
- `git diff --check`

Validation command results:
- Compartment analyzer passed.
- Py-compile passed.
- Focused compartment helper tests passed: 2 passed, 14 dependency deprecation warnings.
- Fast pytest suite passed after adding the compartment tests: 41 passed, 9 deselected, 14 Matplotlib/PyParsing dependency deprecation warnings.
- Diff whitespace check passed.

Synthetic-only finding:
- Locally measurable stem metrics include stem volume, volume fraction, length, width, principal lengths, canal-axis extent, and transition continuity proxies.
- Locally measurable bulb metrics include bulb volume, volume fraction, width, axial extent, centroid offset from stem axis, and bulb-to-stem volume/width ratios.
- 191/261 masks have no generated CPA bulb in the standalone single-mask cohort mapping.
- All small cases and all medium cases have median bulb fraction 0.0.
- Large cases have a generated bulb in 70/71 cases, but median large-case bulb fraction is 0.033 and median stem fraction is 0.899.
- This is a synthetic plausibility concern and likely reflects the standalone single-timepoint initialization/calibration regime. It is not a proven biological invalidity claim without real compartment labels.
- Derived compartment mismatch was flagged in 160/261 cases, and 9/261 masks have multiple connected components. These are engineering review targets, not automatic scientific failures.
- No generator tuning is justified from this pass.

## Integrated Specialist Result: PROV-MASK-001

Status: ACCEPT_WITH_FOLLOWUP.

Files created:
- `analysis/saved_mask_provenance/recover_saved_mask_provenance.py`
- `analysis/saved_mask_provenance/MASK_GENERATION_EVIDENCE_INVENTORY.md`
- `analysis/saved_mask_provenance/GENERATOR_HISTORY.md`
- `analysis/saved_mask_provenance/reproduction_trials.csv`
- `analysis/saved_mask_provenance/SAVED_MASK_PROVENANCE_REPORT.md`
- `analysis/saved_mask_provenance/provenance.json`

Validation result:
- Representative cases tested: 10, using the same cases from compartment mapping verification.
- Current checked-in generator plus manifest reproduced saved masks poorly: median Dice 0.887, range 0.818 to 0.918.
- Pulled cohort script copy `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py` plus manifest reproduced 10/10 selected saved masks exactly: Dice 1.000, zero volume error, zero centroid error, bbox IoU 1.000, and principal-axis error 0 degrees.

Validation commands:
- `MPLCONFIGDIR=/tmp/growthnet_mpl XDG_CACHE_HOME=/tmp/growthnet_cache .venv/bin/python analysis/saved_mask_provenance/recover_saved_mask_provenance.py`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/saved_mask_provenance/recover_saved_mask_provenance.py`
- `.venv/bin/python -m pytest -m "fast and not slow" -v`
- `git diff --check`

Scientific finding:
- The saved-mask generation path is recovered for the representative subset via the pulled cohort script copy, not the current checked-in generator script.
- The 191/261 zero-bulb and 160/261 mismatch findings from the current-generator compartment analyzer must be downgraded until the compartment audit is rerun with the recovered generation path.
- Synthetic-side compartment reconstruction is now locally unblocked for a corrected rerun.
- Real-vs-synthetic anatomical compartment validation remains blocked by missing real source masks and canal/CPA landmarks or labels.

## Integrated Specialist Result: COMPART-002

Status: ACCEPT_WITH_FOLLOWUP.

Files created:
- `analysis/anatomical_compartment_validation_v2/analyze_recovered_generator_compartments.py`
- `analysis/anatomical_compartment_validation_v2/SYNTHETIC_COMPARTMENT_AUDIT_V2.md`
- `analysis/anatomical_compartment_validation_v2/synthetic_compartment_features_v2.csv`
- `analysis/anatomical_compartment_validation_v2/synthetic_compartment_summary_v2.csv`
- `analysis/anatomical_compartment_validation_v2/visual_verification.csv`
- `analysis/anatomical_compartment_validation_v2/provenance.json`
- `analysis/anatomical_compartment_validation_v2/overlays/*.png`

Validation result:
- Recovered-generator preflight: 10/10 representative provenance cases reproduced with Dice 1.000, zero volume difference, zero centroid difference, and bbox IoU 1.000.
- Authoritative saved masks analyzed: 261/261.
- Corrected zero-bulb count: 135/261.
- Zero-bulb by stratum: 79/79 small, 56/111 medium, 0/71 large.
- Median bulb fraction by stratum: small 0.000, medium 0.000, large 0.038.
- Median stem fraction by stratum: small 1.000, medium 1.000, large 0.961.
- Cases with >1% unmatched compartment voxels: 0/261.
- Multi-component masks: 9/261.
- Fresh visual verification: 12 reviewed, 12 `MAPPING_CONFIRMED`.

Validation commands:
- `MPLCONFIGDIR=/tmp/growthnet_mpl XDG_CACHE_HOME=/tmp/growthnet_cache .venv/bin/python analysis/anatomical_compartment_validation_v2/analyze_recovered_generator_compartments.py`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/anatomical_compartment_validation_v2/analyze_recovered_generator_compartments.py`
- `.venv/bin/python -m pytest -m "fast and not slow" -v`
- `git diff --check`

Scientific finding:
- The prior 191/261 zero-bulb result is rejected and replaced by 135/261 using the recovered generator path.
- The prior 160/261 mismatch flags are an artifact of using the wrong generator path.
- The prior 9/261 multi-component finding is confirmed.
- The cohort remains stem-dominant, especially for small and medium masks. Large masks develop nonzero CPA/bulb components, but median bulb fraction remains modest at 0.038.
- These are synthetic design signals and plausibility observations. They do not establish clinical invalidity without real compartment annotations.
- No generator tuning is justified before real-compartment validation.

## Integrated Specialist Result: PROV-FUTURE-001 and QC-001

Status: COMPLETE.

Files changed:
- `scripts/generate_synthetic_lollipop_cohort.py`
- `tests/test_synthetic_generation.py`

Files created:
- `docs/CURRENT_AUTHORITATIVE_ARTIFACTS.md`
- `docs/LOCAL_QC_DASHBOARD.md`

Implementation result:
- Added optional `--provenance_json` to standalone synthetic cohort generation.
- Added optional `--write_compartment_labels` to emit per-case uint8 NIfTI sidecars with label schema `0=background;1=stem_canal;2=transition;3=cpa_bulb`.
- Default behavior remains backward compatible: no provenance file or label sidecars are written unless requested.
- Manifest gains `compartment_label_path` and `compartment_label_schema` only when label writing is enabled.
- Provenance schema is `synthetic_lollipop_provenance_v1` and includes generator script hash, synthetic module hash, targets hash, manifest hash, run parameters, dependency versions, spacing, and per-case rows.

Validation result:
- Focused synthetic tests passed: 9 passed, 14 dependency warnings.
- Two-case smoke run wrote masks, compartment labels, manifest label columns, and provenance JSON successfully.
- Smoke label values were valid: small case labels `[0, 1]`; larger case labels `[0, 1, 2, 3]`.

Validation commands:
- `.venv/bin/python -m pytest tests/test_synthetic_generation.py -v`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile scripts/generate_synthetic_lollipop_cohort.py`
- `.venv/bin/python scripts/generate_synthetic_lollipop_cohort.py --targets_csv /private/tmp/growthnet_provenance_smoke_1785558735/targets.csv --out_dir /private/tmp/growthnet_provenance_smoke_1785558735/out --manifest_csv /private/tmp/growthnet_provenance_smoke_1785558735/manifest.csv --provenance_json /private/tmp/growthnet_provenance_smoke_1785558735/provenance.json --write_compartment_labels --max_calibration_iters 6`

Scientific finding:
- This change improves future auditability only. It does not alter generator morphology, tune anatomy, or retroactively add true saved compartment labels to historical masks.
- Real-vs-synthetic compartment validation still requires real masks and anatomical annotations.

## Integrated Specialist Result: LONG-PLAN-001, LONG-PROV-001, and LONG-META-001

Status: COMPLETE.

Files changed:
- `scripts/generate_synthetic_longitudinal_dataset.py`
- `tests/test_longitudinal_helpers.py`
- `tests/test_longitudinal_dataset_audit.py`
- `docs/LONGITUDINAL_PIPELINE_AUDIT.md`

Files created:
- `docs/LONGITUDINAL_SYNTHETIC_DATASET_ACTION_PLAN.md`

Implementation result:
- Created a longitudinal synthetic dataset readiness action plan with agent-owned tasks and subtasks.
- Added optional `--provenance_json` to the longitudinal wrapper.
- Added `synthetic_longitudinal_provenance_v1` payload generation with wrapper hash, `embed_tumor.py` hash, timeline/background hashes, output CSV hashes, generation parameters, timeline rows, metadata rows, QC rows, and QC pass/fail counts.
- Added additive `metadata.csv` traceability columns:
  - `embedding_growth_mode`
  - `visit_seed`
  - `source_mri_path`
  - `source_seg_path`
  - `volume_ravd_tolerance`
  - `volume_max_iterations`
  - `gen_size`

Validation result:
- Focused longitudinal tests passed: 10 passed, 14 dependency warnings.
- No-MRI smoke run passed by recording four failure rows and writing provenance JSON without launching heavy embedding.
- Smoke provenance recorded `synthetic_longitudinal_provenance_v1`, 1 patient, 4 timepoints, 0 QC passes, and 4 QC failures.

Validation commands:
- `.venv/bin/python -m pytest tests/test_longitudinal_helpers.py tests/test_longitudinal_dataset_audit.py -v`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile scripts/generate_synthetic_longitudinal_dataset.py`
- `.venv/bin/python scripts/generate_synthetic_longitudinal_dataset.py --timeline_csv /private/tmp/growthnet_longitudinal_provenance_smoke_1785579519/timeline.csv --background_csv /private/tmp/growthnet_longitudinal_provenance_smoke_1785579519/backgrounds.csv --out_dir /private/tmp/growthnet_longitudinal_provenance_smoke_1785579519/out --provenance_json /private/tmp/growthnet_longitudinal_provenance_smoke_1785579519/out/provenance.json`

Scientific finding:
- This improves traceability and failure auditability only.
- The wrapper still does not define a clinical growth law.
- Sparse visits, elapsed days, continuous trajectory semantics, longitudinal drift QC, ViViT export, and real longitudinal validation remain open.

## Integrated Specialist Result: LONG-GROWTH-001

Status: HUMAN_REVIEW_REQUIRED scientifically; COMPLETE as default-off engineering encoding.

Files changed:
- `scripts/generate_synthetic_longitudinal_dataset.py`
- `tests/test_longitudinal_helpers.py`
- `tests/test_longitudinal_dataset_audit.py`
- `docs/LONGITUDINAL_PIPELINE_AUDIT.md`
- `docs/LONGITUDINAL_SYNTHETIC_DATASET_ACTION_PLAN.md`

Files created:
- `docs/CLINICAL_GROWTH_LAW_ACTION_PLAN.md`

Implementation result:
- Added `--clinical_growth_law` with choices `none` and `empirical_vs_v1`.
- Added `--visit_days` parser for four strictly increasing non-negative visit days.
- Added deterministic per-patient empirical annual volumetric-rate sampling.
- Added derived target-volume generation from baseline `T1_volume_mm3`.
- Added metadata/provenance fields for target source, growth-law name, visit day, and annual volume-change fraction.
- Default mode remains `none`, preserving explicit timeline CSV `T1..T4` volumes.

Evidence basis:
- Untreated VS volumetric observation literature uses a >=20% annual volume-change threshold for growth, with a fast-growing subgroup above 100% annual growth.
- `empirical_vs_v1` encodes those thresholds as a transparent candidate prior, not as validated clinical truth.

Validation result:
- Focused longitudinal tests passed after adding growth-law tests: 14 passed, 14 dependency warnings.
- No-MRI smoke run with `--clinical_growth_law empirical_vs_v1` wrote metadata/provenance and recorded law-derived target volumes despite background failure.

Scientific finding:
- The law is suitable only for local synthetic experimentation and pipeline development.
- Clinical validation, calibration, and training-use approval remain blocked by real longitudinal masks and human scientific review.

## Remote Data Requirements

Known remote-dependent work is tracked in `docs/GROWTHNET_ACTION_PLAN.md` as `BLOCKED_REMOTE_DATA`; it must not block local reliability, tests, documentation, or local artifact analysis.

Current remote-data blockers:
- Original MRI/segmentation and embedded outputs for the named orientation cases, including `126_5_1607` and `132_1_148`, if patient-space diagnosis is required.
- Real source segmentation masks for the 261 matched morphology cases.
- Full 291-case source real segmentation cohort.
- Common-resolution re-extraction of real source masks for surface-derived morphology metrics.
- Anatomical labels for IAC/CPA canal-vs-bulb validation.
- Porus/fundus landmarks, canal-axis annotations, IAC mask, or CPA boundary annotations for an initial 30-case real compartment validation subset.
- Any training-ready cohort generation or HPC-scale model training artifacts.

Local-only blockers that do not require remote access:
- Add extractor/generator provenance fields: code commit, input hash, spacing, manifest hash, and extractor schema version.
- Add explicit per-voxel compartment-label outputs for future synthetic cohort generation so compartment analysis no longer depends on reconstruction.
- Resolve feature metric-definition issues before using PCA/elongation/sphericity outputs as scientific acceptance metrics; surface metrics should be compared only after resolution normalization.

## Human Decisions Required

- Decide the authoritative orientation validation metrics: whole-mask PCA, stem/canal axis, canal-to-CPA axis, or a multi-metric report.
- Decide how to handle pulled synthetic feature provenance drift before morphology tuning.
- Decide whether proposed volume-targeting thresholds are engineering smoke thresholds only or scientific acceptance criteria.
- Review longitudinal `stable` and `growing` label semantics before using them in training or publication framing.
- Decide whether to install/pin `pytest` and other local dependencies in a repository-managed environment spec.
- Decide whether the current local masks plus current extractor should become the authoritative morphology benchmark source.
- Decide the preferred real compartment annotation standard: porus/fundus landmarks, canal-axis vector plus porus point, IAC mask, or explicit CPA boundary.

## Recommended Next Commands

```bash
.venv/bin/python -m pytest -m "fast and not slow" -v
```

Expected current selected-suite result: 48 passed, 9 deselected, 14 dependency deprecation warnings.

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

- Engineering completeness: approximately 82%.
- Scientific validation completeness: approximately 55%.
- Training-dataset readiness: approximately 33%.

These estimates separate code operability from scientific validation. Engineering completeness increased because recovered-generator compartment analysis now runs reproducibly for all 261 saved masks with preflight and visual checks. Scientific validation increased modestly because synthetic-side compartment statistics are now provenance-correct, but real source-mask/landmark evidence is still missing. Training readiness is unchanged because dataset semantics and scientific acceptance still block readiness.
## 2026-08-01 - Growth Scenario and Shape-Variant Update

- Added `scenario_mixture_v1` as an explicit longitudinal target-volume mode
  rather than treating one clinical equation as definitive.
- Added `--variants_per_timepoint` to the longitudinal dataset wrapper so the
  same patient/timepoint/target volume can produce multiple independently
  seeded shape variants.
- Added metadata fields for `growth_law_scenario`, `variant_id`, and
  `variant_seed`; default one-variant naming remains backward compatible.
- Added local overlay smoke artifacts under
  `analysis/clinical_growth_law_validation/`. Volume realization passed for the
  local fixture with max RAVD 0.0819, but embedding QC failed because the reused
  source segmentation is too small for anatomical interpretation.
- Scientific status: scenario controls are implemented for local engineering
  validation; real longitudinal clinical validation remains blocked until real
  longitudinal segmentations and visit dates are available.

## 2026-08-04 - Longitudinal Architecture Bridge Update

- Added explicit patient-level engineering QC gate fields to
  `longitudinal_qc_summary.csv`: `engineering_qc_gate`,
  `engineering_qc_failure_reasons`, `target_volume_trend_status`, and
  `actual_volume_trend_status`.
- Kept engineering readiness separate from scientific validity. RAVD failures,
  failed per-mask QC, background inconsistency, and invalid visit ordering are
  engineering gate failures; non-monotone target or achieved volumes are
  reported as trend statuses because regression/stable trajectories may be
  intentional.
- Added `projects/vivit/src/data/synthetic_longitudinal_loader.py`, a
  metadata-based adapter that converts generator `metadata.csv` rows into
  ViViT-style temporal sequence records.
- The adapter groups multi-variant outputs by `(patient_id, variant_id)` so
  same-timepoint shape variants do not get mixed inside one patient trajectory.
- Added `docs/LONGITUDINAL_ARCHITECTURE_BRIDGE.md` to document the generator to
  training-loader boundary.

Validation:

- `.venv/bin/python -m pytest tests/test_synthetic_longitudinal_loader.py tests/test_longitudinal_dataset_audit.py -v`
  passed: 10 passed, 14 dependency deprecation warnings.
- `.venv/bin/python -m pytest -m "fast and not slow" -v` passed: 54 passed,
  13 deselected, 14 dependency deprecation warnings.
- `python3 -m py_compile scripts/generate_synthetic_longitudinal_dataset.py projects/vivit/src/data/synthetic_longitudinal_loader.py tests/test_longitudinal_dataset_audit.py tests/test_synthetic_longitudinal_loader.py`
  passed.
- `git diff --check` passed.

Remaining blocker:

- Full NIfTI/tensor loading smoke through the synthetic longitudinal adapter is
  still pending before claiming local ViViT training readiness.

## 2026-08-04 - Architecture Remediation Foundation Update

- Added a root `pyproject.toml` with core local dependencies and optional extras
  for development, registration, training, and presentation/export workflows.
- Added `.github/workflows/fast-tests.yml` for laptop-safe fast CI.
- Added `docs/ARCHITECTURE_DEPENDENCY_INVENTORY.md`.
- Added `shared/provenance.py` and `shared/reporting.py` to begin replacing
  duplicated git-commit, file-hash, CSV, JSON, and report-writing boilerplate.
- Kept backward compatibility by preserving private helper names in active
  scripts while delegating implementation to shared helpers.
- Migrated `shared/run_logger.py`,
  `scripts/generate_synthetic_lollipop_cohort.py`, and
  `scripts/generate_synthetic_longitudinal_dataset.py` onto the shared helper
  layer.
- Installed `PyYAML 6.0.3` into `.venv` because the new manifest exposed it as a
  missing required dependency for existing shared config-loader tests.

Validation:

- `.venv/bin/python -m pip install -e ".[dev]" --dry-run` passed before
  installing `PyYAML`.
- `.venv/bin/python -m pytest tests/test_shared_helpers.py projects/mri_registration/tests/test_smoke.py -v`
  passed: 9 passed, 3 skipped.
- `.venv/bin/python -m pytest -m "fast and not slow" -v` passed: 60 passed,
  13 deselected, 14 dependency deprecation warnings.
- `python3 -m py_compile shared/provenance.py shared/reporting.py shared/run_logger.py scripts/generate_synthetic_lollipop_cohort.py scripts/generate_synthetic_longitudinal_dataset.py`
  passed.
- `git diff --check` passed.

Remaining:

- Analysis-script helper migration should happen separately because those files
  are tied to generated scientific evidence.
- `embed_tumor.py` should not be split until behavior-lock tests and review
  gates are stronger.

### Follow-on Low-Risk Analysis Helper Migration

- Migrated `analysis/clinical_growth_law_validation/measure_variant_diversity.py`
  to use shared CSV/text writing helpers.
- Migrated `analysis/clinical_growth_law_validation/sample_growth_scenarios.py`
  to use shared CSV/text writing helpers.
- Migrated `analysis/volume_targeting/run_volume_targeting_benchmark.py` to use
  shared CSV writing helpers.
- Added `tests/test_growth_scenario_sampling.py`.
- Added `tests/test_volume_targeting_helpers.py`.

Validation:

- `.venv/bin/python -m pytest tests/test_longitudinal_variant_diversity.py tests/test_growth_scenario_sampling.py tests/test_volume_targeting_helpers.py tests/test_shared_helpers.py -v`
  passed: 9 passed, 14 dependency deprecation warnings.
- `.venv/bin/python -m pytest -m "fast and not slow" -v` passed after the
  follow-on migration: 62 passed, 13 deselected, 14 dependency deprecation
  warnings.
