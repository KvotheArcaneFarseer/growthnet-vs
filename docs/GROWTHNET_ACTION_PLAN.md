# GrowthNet Local Action Plan

Last updated: 2026-07-18

Scope: local laptop repository only. No SSH, no Rivanna assumptions, no remote datasets, no SLURM, no destructive cleanup.

## Current Local State

### Complete
- Single-case synthetic vestibular schwannoma embedding pipeline in `embed_tumor.py`.
- Lollipop and legacy ellipsoid synthetic generator in `projects/vivit/src/data/synthetic.py`.
- CSV-driven batch wrapper in `scripts/run_batch_embedding.py`.
- Real tumor feature extractor in `scripts/extract_real_tumor_features.py`.
- Standalone synthetic lollipop mask cohort generator in `scripts/generate_synthetic_lollipop_cohort.py`.
- Local/generated QC and metrics artifacts for case 147 and seed validation folders.
- Lab meeting export scripts and deck artifacts.
- MRI registration codebase and smoke tests exist, but local test execution currently lacks `pytest`.

### Partially Complete
- Longitudinal dataset wrapper exists in `scripts/generate_synthetic_longitudinal_dataset.py`, but local end-to-end behavior and metadata guarantees need audit.
- Volume targeting exists in `embed_tumor.py`, but local quantitative operating range needs benchmarking.
- Batch runner has metrics aggregation, but resume/resource controls need local reliability audit.
- ViViT/TemporalUNETR training code exists, but no local training dataset integration has been validated.
- Documentation is extensive but stale in places; root and several project READMEs remain placeholders.

### Experimental
- `--validate_anisotropy` diagnostic path.
- `--t0_volume_fraction_of_seg` calibration path.
- Napari visualization and animation scripts.
- Graphify exports and Obsidian vaults.
- Synthetic-vs-real feature distribution comparisons pulled from previous work.

### Missing Or Unvalidated
- Fast local test suite for core embedding behavior.
- Requirements or lockfile for local reproducibility.
- Local orientation diagnostic workflow for large-tumor axis-error reports.
- Local volume targeting benchmark across target ranges and seeds.
- Local longitudinal wrapper integration test.
- Clear local-vs-remote validation status in docs.
- Any scientific claim that generalizes beyond locally available data.

### Technical Debt
- `embed_tumor.py` is monolithic.
- Core scripts use direct imports and path insertion instead of a package.
- Generated outputs, scratch outputs, and provenance artifacts coexist with source files.
- CLI defaults reference a private Downloads case.
- Existing `.gitignore` has duplicate sections and intentional exceptions for the sample timeline.
- GitNexus impact tools are required by project instructions for symbol edits, but callable GitNexus MCP tools are not exposed in this session.

### Scientific Risks
- Approximately 90-degree axis errors in large tumors may be true orientation bugs, principal-axis identity switches, QC limitations, or mixed causes.
- Whole-mask PCA may not represent stem/canal direction when CPA bulb dominates.
- Synthetic morphology differs from real distributions in locally available comparisons, especially elongation and bbox fill.
- Local data does not support population-level conclusions.

## File Ownership And Conflict Plan

| Agent | Primary Ownership | May Read | Must Not Edit |
|---|---|---|---|
| Agent 1 Orientation | `analysis/orientation_validation/**` | outputs, metrics, `embed_tumor.py`, `synthetic.py` | core source |
| Agent 2 Batch Reliability | `analysis/batch_reliability/**`, `docs/BATCH_LOCAL_RELIABILITY.md`; optional `scripts/run_batch_embedding.py` only if needed | batch outputs, `embed_tumor.py` | generator geometry, orientation docs |
| Agent 3 Longitudinal | `docs/LONGITUDINAL_PIPELINE_AUDIT.md`, `data/timelines/local_longitudinal_example.csv`, `tests/test_longitudinal_*.py` | longitudinal script | `embed_tumor.py`, `synthetic.py` |
| Agent 4 Volume Targeting | `analysis/volume_targeting/**` | `embed_tumor.py`, generated temporary outputs | core source |
| Agent 5 Morphology Audit | `analysis/real_vs_synthetic/**` | feature CSV/JSON, feature extractor, cohort generator | generator source |
| Agent 6 Testing | `tests/**`, `pytest.ini` or test docs | all source | source modules unless a testability bug is trivial and isolated |
| Agent 7 Documentation | `README.md`, `docs/LOCAL_REPRODUCIBILITY.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/CURRENT_PROJECT_STATUS.md` | all docs/source | scientific algorithms |
| Agent 8 Review | `docs/GROWTHNET_AGENT_STATUS.md` | all outputs/diffs | major code fixes during review |

Overlapping core files are sequential only. No concurrent edits to `embed_tumor.py` or `projects/vivit/src/data/synthetic.py` are assigned initially.

## Dependency Graph

Critical local path:

`PLAN-001 -> TEST-001 -> EMB-001 -> BATCH-001 -> VOL-001 -> LONG-001 -> DOC-001 -> REVIEW-001`

Parallel work groups:
- Group A investigation: `EMB-001`, `VOL-001`, `MORPH-001`.
- Group B reliability/testing: `BATCH-001`, `LONG-001`, `TEST-001`.
- Group C documentation/status: `DOC-001`, status maintenance.
- Review queue: `REVIEW-001` starts after specialist outputs exist.

Blocked remote-data work:
- `SCI-REMOTE-001`: multi-patient embedding validation on unavailable MRI+seg pairs.
- `SCI-REMOTE-002`: population-level clinical validity claims requiring full curated cohort review.
- `TRAIN-REMOTE-001`: full training-ready cohort generation and HPC-scale training.

Human scientific judgment required:
- `SCI-HUMAN-001`: whether PCA axis errors should be judged against whole-mask axis, stem axis, canal-to-CPA axis, or multiple metrics.
- `SCI-HUMAN-002`: whether morphology should favor distribution matching or visually obvious lollipop anatomy when those conflict.

## Recommended Agent Assignments

| Agent | Specialization | Assigned Tasks | Rationale |
|---|---|---|---|
| Agent 1 | Orientation and embedding validation | `EMB-001` | Requires reading NIfTI affines, PCA axes, mask geometry, and embedding metrics without changing core geometry. |
| Agent 2 | Batch pipeline reliability | `BATCH-001` | Owns resumability, resource controls, status logging, and local smoke behavior in the CSV batch runner. |
| Agent 3 | Longitudinal dataset audit | `LONG-001` | Owns patient/visit metadata, deterministic seed routing, and wrapper limitations while keeping clinical growth assumptions out of scope. |
| Agent 4 | Volume targeting validation | `VOL-001` | Owns requested-versus-achieved quantitative benchmarking and threshold recommendations. |
| Agent 5 | Morphology validation | `MORPH-001`, future `MORPH-002` | Owns real-vs-synthetic feature comparisons, provenance checks, and remote-data dependency classification. |
| Agent 6 | Testing and reproducibility | `TEST-001`, future `TEST-002` | Owns deterministic local tests, markers, fixtures, and runnable test commands. |
| Agent 7 | Documentation | `DOC-001` | Owns README/status/reproducibility/limitations docs and keeps claims evidence-bound. |
| Agent 8 | Integration review | `REVIEW-001` | Owns cross-agent scope checks, validation reruns, and accept/reject/human-review classification. |

## Execution Order

Critical path now completed locally:

`PLAN-001 -> EMB-001/BATCH-001/LONG-001/VOL-001/MORPH-001/TEST-001/DOC-001 -> REVIEW-001`

Next critical path:

`TEST-002 -> full local pytest run -> MORPH-002 -> human scientific review -> embedded-pipeline validation tasks`

Parallel next work:
- `TEST-002` can run independently once local dependency/environment policy is chosen.
- `VOL-002` embedded volume benchmark can run once a clean local MRI/seg fixture is identified or created.
- `LONG-002` through `LONG-006` can run in parallel with morphology provenance work if they do not edit core geometry.
- Documentation follow-up can continue in parallel, but publication-facing scientific claims should wait for human review.

Blocked work:
- Patient-space orientation diagnosis for named cases remains `BLOCKED_REMOTE_DATA`.
- Independent real morphology re-extraction remains `BLOCKED_REMOTE_DATA`.
- Full training dataset generation remains blocked by missing local cohort inputs and scientific review.

## Master Task Table

| Task ID | Milestone | Category | Title | Owner | Dependencies | Files Likely Involved | Status | Expected Outputs | Success Criteria | Validation Command | Risks | Human Review |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-001 | Planning | DOCUMENTATION | Create local action plan and agent status docs | Lead | none | `docs/GROWTHNET_ACTION_PLAN.md`, `docs/GROWTHNET_AGENT_STATUS.md`, `docs/GROWTHNET_LOCAL_WORK_REPORT.md` | COMPLETE | coordination docs | docs created with ownership, dependencies, statuses | `sed -n '1,80p' docs/GROWTHNET_ACTION_PLAN.md` | stale repo assumptions | no |
| EMB-001 | Orientation Validation | EMBEDDING | Diagnose large-tumor axis-error mechanism locally | Agent 1 | PLAN-001 | `analysis/orientation_validation/**` | COMPLETE | CSV, diagnostic markdown, commands | local evidence classified; unsupported claims marked | `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/orientation_validation/orientation_diagnostic.py` | patient-space named cases are `BLOCKED_REMOTE_DATA`; PCA ambiguity remains | yes |
| BATCH-001 | Batch Reliability | TECHNICAL_DEBT | Audit and improve local batch resumability/resource controls | Agent 2 | PLAN-001 | `scripts/run_batch_embedding.py`, `analysis/batch_reliability/**`, docs | COMPLETE | smoke report, status schema, resume-capable runner | all-skipped resume avoids recomputation; status JSONL exists; default recompute behavior preserved | `python3 scripts/run_batch_embedding.py --help` | no clean original MRI/seg local smoke fixture | no |
| LONG-001 | Longitudinal Audit | LONGITUDINAL | Audit longitudinal wrapper and add local metadata/repro tests | Agent 3 | PLAN-001 | `scripts/generate_synthetic_longitudinal_dataset.py`, `tests/**`, docs | COMPLETE | audit doc, example timeline, tests | current behavior documented; deterministic metadata tested | `MPLCONFIGDIR=/tmp/growthnet_mpl PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m unittest tests.test_longitudinal_dataset_audit -v` | clinical label semantics unvalidated | yes |
| VOL-001 | Volume Targeting | SCIENTIFIC_VALIDATION | Quantify local target-volume accuracy | Agent 4 | PLAN-001 | `analysis/volume_targeting/**` | COMPLETE | benchmark CSV, report, plots | target vs achieved reported by range with limitations | `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/volume_targeting/run_volume_targeting_benchmark.py` | thresholds are not scientific acceptance criteria yet | yes |
| MORPH-001 | Morphology Audit | SCIENTIFIC_VALIDATION | Local real-vs-synthetic comparison audit | Agent 5 | PLAN-001 | `analysis/real_vs_synthetic/**` | COMPLETE | report, benchmark CSVs, plots | local comparisons complete and remote gaps marked | `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/real_vs_synthetic/analyze_local_morphology.py` | pulled synthetic feature table does not reproduce from local masks | yes |
| TEST-001 | Testing | TESTING | Add fast local deterministic tests | Agent 6 | PLAN-001 | `tests/**`, `pytest.ini`, docs | READY_FOR_VALIDATION | tests and test docs | tests compile and are local-only; full pytest awaits dependency install | `python3 -m pytest -m fast` | pytest missing locally | no |
| DOC-001 | Documentation | DOCUMENTATION | Update local project status and reproducibility docs | Agent 7 | PLAN-001 | `README.md`, `docs/*.md` | COMPLETE | current status, limits, command reference | docs separate implemented/partial/experimental/local/remote/unvalidated | core script `--help` and py_compile checks | stale scientific claims need review before publication framing | yes |
| REVIEW-001 | Integration | TECHNICAL_DEBT | Review specialist outputs and classify | Agent 8/Lead | specialist outputs | `docs/GROWTHNET_AGENT_STATUS.md` | COMPLETE | ACCEPT/REJECT/HUMAN_REVIEW_REQUIRED per task | diffs inspected and local validation rerun | `git diff --stat`; `git diff --check`; py_compile/unittest checks | GitNexus tools unavailable | no |
| MORPH-002 | Morphology Provenance | SCIENTIFIC_VALIDATION | Resolve pulled synthetic CSV versus local re-extraction drift | Future validation agent | MORPH-001 | `analysis/real_vs_synthetic/**`, `scripts/extract_real_tumor_features.py`, provenance manifests | HUMAN_REVIEW_REQUIRED | drift diagnosis and decision on authoritative feature source | source-code drift, mask drift, or provenance mismatch classified | rerun extractor and compare drift CSV | tuning from wrong target distribution | yes |
| TEST-002 | Test Environment | TESTING | Add or pin local pytest dependency and run suite | Future environment/testing agent | TEST-001 | dependency docs or project config | NOT_STARTED | runnable fast pytest suite | `python3 -m pytest -m "fast and not slow"` passes locally | environment changes outside repo may require approval | no |

## Task Decomposition

### EMB-001
- Objective: determine whether 90-degree axis errors are spatial bugs, PCA identity switches, QC limitations, or mixed.
- Expected files: `analysis/orientation_validation/orientation_case_review.csv`, `analysis/orientation_validation/ORIENTATION_DIAGNOSTIC.md`.
- Likely code paths: `principal_axes`, `TimepointMetrics.axis_error_deg`, generated `embedding_metrics.json`, mask NIfTIs.
- Deliverables: reviewed case table, evidence-backed classification, local inspection commands.
- Tests: metric extraction against local NIfTI outputs; no geometry edits.
- Definition of done: at least two problematic and two controls if available; otherwise explicit local availability finding.
- Conflicts: must not edit `embed_tumor.py` or `synthetic.py`.

### BATCH-001
- Objective: make local batch operation recoverable and auditable without Rivanna.
- Expected files: `analysis/batch_reliability/LOCAL_BATCH_SMOKE.md`, optional `scripts/run_batch_embedding.py`.
- Likely code paths: `_load_case_rows`, `run_batch`, `_flatten_case_metrics`, `_build_batch_summary`.
- Deliverables: smoke report, status semantics, exact local commands, optional resume patch.
- Tests: one-case batch to temp output; rerun verifies skip/resume if implemented.
- Definition of done: small local run passes or failure is diagnosed; existing valid outputs not recomputed unnecessarily.
- Conflicts: if editing `scripts/run_batch_embedding.py`, no other agent may edit it.

### LONG-001
- Objective: audit local longitudinal generation without inventing clinical growth laws.
- Expected files: `docs/LONGITUDINAL_PIPELINE_AUDIT.md`, `data/timelines/local_longitudinal_example.csv`, tests.
- Likely code paths: `_safe_id`, `_stable_seed`, `_growth_mode`, `_qc_mask`, `generate_longitudinal_dataset`.
- Deliverables: audit, reproducibility tests, explicit missing behavior tasks.
- Tests: metadata-only and helper-level tests; integration if local inputs exist.
- Definition of done: patient/background/timepoint metadata behavior is documented and deterministic helpers are tested.
- Conflicts: do not edit core embedding/generator files.

### VOL-001
- Objective: replace vague volume-targeting confidence with local quantitative evidence.
- Expected files: `analysis/volume_targeting/volume_targeting_benchmark.csv`, `analysis/volume_targeting/VOLUME_TARGETING_REPORT.md`, plots.
- Likely code paths: existing local metrics, `generate_synthetic_lollipop_cohort.py`, optional small helper in analysis dir.
- Deliverables: target/achieved/RAVD/convergence table; range-specific conclusions.
- Tests: benchmark command reruns and writes deterministic CSV.
- Definition of done: behavior by volume range is documented and limitations are explicit.
- Conflicts: no source edits unless coordinated.

### MORPH-001
- Objective: determine all local supportable real-vs-synthetic morphology conclusions.
- Expected files: `analysis/real_vs_synthetic/LOCAL_VALIDATION_REPORT.md`, copied/derived CSVs, plots.
- Likely code paths: `scripts/extract_real_tumor_features.py`, `rivanna_pull/analysis/**`.
- Deliverables: sample count verification, stratified gaps, remote dependency list.
- Tests: reproducible comparison script/command if new derived artifacts are created.
- Definition of done: no unsupported population claims; remote-only gaps marked.
- Conflicts: no generator tuning.

### TEST-001
- Objective: improve local test confidence.
- Expected files: `tests/**`, `pytest.ini`, `docs/LOCAL_TESTING.md` if useful.
- Likely code paths: helper functions in scripts and embedding validation.
- Deliverables: deterministic tests separated from slow tests.
- Tests: `python3 -m pytest tests -q`.
- Definition of done: fast tests do not depend on Rivanna or private Downloads files.
- Conflicts: avoid editing implementation unless agreed.

### DOC-001
- Objective: make local project state understandable and auditable.
- Expected files: `README.md`, `docs/LOCAL_REPRODUCIBILITY.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/CURRENT_PROJECT_STATUS.md`.
- Likely code paths: none; documentation only.
- Deliverables: status, reproducibility, limitations, local command reference.
- Tests: commands are local-safe and marked if dependency-bound.
- Definition of done: docs separate implemented/partial/experimental/local/remote/unvalidated.
- Conflicts: must not alter algorithms.

## Local Validation Baseline

Already run in this orchestration:

```bash
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile embed_tumor.py scripts/run_batch_embedding.py scripts/extract_real_tumor_features.py scripts/generate_synthetic_lollipop_cohort.py scripts/generate_synthetic_longitudinal_dataset.py projects/vivit/src/data/synthetic.py shared/config_loader.py shared/run_logger.py projects/_shared/src/run_recording.py
```

Result: passed.

Attempted:

```bash
python3 -m pytest projects/mri_registration/tests/test_smoke.py -q
```

Result: failed because `pytest` is not installed in the active Python environment.

## Progress Tracking Checklist

| Task ID | Not Started | In Progress | Validation | Complete | Measurable Completion Criteria |
|---|---:|---:|---:|---:|---|
| PLAN-001 |  |  |  | x | Plan, status, and local work report exist with ownership, dependency graph, and task statuses. |
| EMB-001 |  |  |  | x | Orientation case CSV and diagnostic report created; named cases classified with local evidence and remote blockers marked. |
| BATCH-001 |  |  |  | x | `--resume`, status JSONL, lazy import, and thread defaults validated by py_compile/help/smoke evidence. |
| LONG-001 |  |  |  | x | Audit doc, example timeline, and deterministic unittest coverage exist; 3 audit tests pass. |
| VOL-001 |  |  |  | x | 36-case benchmark CSV/report/plots exist; performance summarized by target range. |
| MORPH-001 |  |  |  | x | Local feature inventory, matched comparisons, drift report, plots, and remote dependency list exist. |
| TEST-001 |  |  | x |  | Test files compile and are local-only; blocked from complete until pytest runs normally. |
| DOC-001 |  |  |  | x | README and local reproducibility/status/limitations docs updated and scoped to local evidence. |
| REVIEW-001 |  |  |  | x | Specialist outputs reviewed; validation rerun; final status recorded. |
| MORPH-002 |  |  |  |  | Complete when provenance drift is classified and an authoritative synthetic feature source is selected. |
| TEST-002 | x |  |  |  | Complete when dependency/environment spec supports `python3 -m pytest -m "fast and not slow"` locally. |
