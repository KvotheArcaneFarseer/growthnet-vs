# GrowthNet Architecture Remediation Plan

Last updated: 2026-08-04

Scope: local GrowthNet repository only. This plan resolves the structural issues
identified in the architecture audit without changing tumor geometry,
morphology tuning, clinical growth semantics, or authoritative generated
artifacts.

Note: `AGENTS.md` requests GitNexus MCP impact tools before edits. Those tools
are not available in the current toolset, so implementation agents must keep
changes narrow, inspect direct callers with local search, and run focused tests
before touching high-blast-radius code such as `embed_tumor.py`.

## Executive Priority

The highest-value structural goal is to make the active GrowthNet pipeline obey
the repository standard already described in `docs/REPO_STRUCTURE.md`:

- reusable logic belongs under `projects/<project>/src/`,
- scripts are thin entry points,
- configs define run behavior,
- run outputs include git/environment provenance,
- generated artifacts are separated from source,
- tests and CI protect the fast local path.

## Milestone A - Dependency And Environment Baseline

Owner: Environment / Reproducibility Agent

Objective: make the repo installable and reproducible on a new laptop.

Status: PARTIAL_COMPLETE

Risk: LOW

Files likely involved:

- `pyproject.toml`
- `requirements.txt` or `requirements-dev.txt`
- `projects/*/configs/env/`
- `docs/LOCAL_REPRODUCIBILITY.md`
- `docs/REPO_STRUCTURE.md`

Tasks:

1. Create a dependency inventory.
   - Inspect imports across `embed_tumor.py`, `scripts/`, `analysis/`,
     `projects/vivit/`, and `projects/mri_registration/`.
   - Separate runtime, analysis, training, test, and optional visualization
     dependencies.
   - Identify locally installed packages that are not actually imported.
   - Output: dependency inventory table.
   - Validation: `python3 -m pip show <core packages>` and import smoke.

2. Add a root dependency manifest.
   - Prefer `pyproject.toml` for project metadata and test tooling.
   - Add pinned or bounded core dependencies for local pipeline execution.
   - Keep heavy/optional training dependencies clearly marked.
   - Output: installable dependency manifest.
   - Validation: fresh-env install command documented; current `.venv` tests
     still pass.

3. Add environment freeze guidance.
   - Add a repo-level freeze command.
   - Add project-level freeze location under `projects/<project>/configs/env/`.
   - Document when to refresh freezes.
   - Output: updated reproducibility docs.
   - Validation: generated freeze is excluded or committed according to policy.

Definition of done:

- A new contributor can identify and install the local development environment.
- Fast tests run from a documented environment.
- No generated clinical data or artifacts are bundled into the dependency
  change.

Progress on 2026-08-04:

- Added `docs/ARCHITECTURE_DEPENDENCY_INVENTORY.md`.
- Added root `pyproject.toml` with core dependencies and optional extras.
- Documented editable install and environment-freeze commands in
  `docs/LOCAL_REPRODUCIBILITY.md`.
- Validated package metadata with
  `.venv/bin/python -m pip install -e ".[dev]" --dry-run`.

Remaining:

- Validate in a genuinely fresh environment.
- Decide whether to commit a reviewed environment freeze under
  `projects/growthnet/configs/env/`.

## Milestone B - Fast CI Safety Net

Owner: CI / Testing Agent

Objective: make regressions visible before they land.

Status: PARTIAL_COMPLETE

Risk: LOW

Files likely involved:

- `.github/workflows/fast-tests.yml`
- `pytest.ini`
- `docs/LOCAL_REPRODUCIBILITY.md`

Tasks:

1. Add minimal GitHub Actions workflow.
   - Use Python version matching the local validated path where practical.
   - Install dependencies from the new manifest.
   - Run `python -m pytest -m "fast and not slow" -v`.
   - Output: fast CI workflow.
   - Validation: workflow syntax lint or local equivalent where possible.

2. Keep CI laptop-safe.
   - Do not download datasets.
   - Do not run expensive cohort generation.
   - Do not require Rivanna, SLURM, or private paths.
   - Output: CI guardrails in comments/docs.
   - Validation: fast suite remains data-independent.

3. Add optional syntax check stage.
   - Compile core scripts and new package modules.
   - Keep failure messages interpretable.
   - Output: `py_compile` or equivalent CI step.
   - Validation: local `python -m py_compile ...` passes.

Definition of done:

- A push/PR runs the existing fast test suite automatically.
- CI does not depend on local-only data paths or generated artifacts.

Progress on 2026-08-04:

- Added `.github/workflows/fast-tests.yml`.
- Workflow installs `.[dev]`, compiles core scripts, and runs
  `python -m pytest -m "fast and not slow" -v`.

Remaining:

- Confirm workflow runtime on GitHub after push/PR.
- Tune dependency install time if PyTorch/MONAI setup proves too slow.

## Milestone C - Shared Utility Layer

Owner: Shared Infrastructure Agent

Objective: remove duplicated boilerplate without changing scientific behavior.

Status: PARTIAL_COMPLETE

Risk: MEDIUM

Files likely involved:

- `shared/provenance.py`
- `shared/reporting.py`
- `shared/cli.py`
- `shared/path_utils.py`
- `embed_tumor.py`
- `scripts/*.py`
- `analysis/**/*.py`
- tests for shared helpers

Tasks:

1. Inventory duplicated functions.
   - Confirm duplicates such as git commit detection, SHA-256 hashing, report
     writing, markdown tables, and CSV writing.
   - Group duplicates by behavior, not just function name.
   - Identify call sites that must remain untouched until later.
   - Output: utility duplication map.
   - Validation: no edits in this task.

2. Create shared provenance helpers.
   - Implement `get_git_commit()`.
   - Implement `sha256_file()`.
   - Implement environment-freeze helper or wrapper over existing
     `shared/run_logger.py`.
   - Add focused tests.
   - Output: `shared/provenance.py`.
   - Validation: tests cover missing git, missing file, and known hash.

3. Create shared reporting helpers.
   - Implement safe text write, JSON write, markdown table helper, and CSV union
     schema helper if needed.
   - Add focused tests.
   - Output: `shared/reporting.py`.
   - Validation: tests cover deterministic output ordering.

4. Migrate low-risk scripts first.
   - Start with analysis scripts that already have tests or simple outputs.
   - Avoid `embed_tumor.py` until shared helpers are stable.
   - Output: small mechanical diffs.
   - Validation: focused tests and `git diff --check`.

5. Migrate high-use entry points second.
   - Repoint batch, synthetic cohort, feature extraction, and longitudinal
     scripts to shared helpers.
   - Preserve public CLI and output schemas.
   - Output: reduced duplicate infrastructure.
   - Validation: fast suite plus focused script tests.

Definition of done:

- Common provenance/reporting behavior is implemented once.
- Public outputs remain backward compatible.
- No scientific calculations are touched.

Progress on 2026-08-04:

- Added `shared/provenance.py` with `sha256_file`, `get_git_commit`, and
  `freeze_environment`.
- Added `shared/reporting.py` with small text, JSON, CSV, fieldname, and
  Markdown-table helpers.
- Added `tests/test_shared_helpers.py`.
- Routed `shared/run_logger.py` through the new provenance helpers while
  preserving its public API.
- Migrated the active synthetic cohort and longitudinal generation scripts to
  delegate private provenance/CSV helpers to shared modules while keeping their
  existing helper names and output schemas.
- Migrated low-risk analysis utilities to shared reporting helpers:
  `analysis/clinical_growth_law_validation/measure_variant_diversity.py`,
  `analysis/clinical_growth_law_validation/sample_growth_scenarios.py`, and
  `analysis/volume_targeting/run_volume_targeting_benchmark.py`.
- Added focused tests for scenario sampling and volume-targeting CSV helper
  behavior.

Validation:

- Shared/helper focused tests passed.
- Active synthetic/longitudinal focused tests passed.
- Low-risk analysis helper focused tests passed.
- Existing MRI registration shared-utility smoke passed after installing
  `PyYAML` from the new manifest.
- Fast suite passed: 62 passed, 13 deselected, 14 dependency deprecation
  warnings.

Remaining:

- Continue migrating remaining analysis scripts in separate evidence-aware
  batches.
- Migrate `embed_tumor.py` only after characterization tests are expanded.

## Milestone D - Tumor Embedding Package Boundary

Owner: Embedding Architecture Agent

Objective: split `embed_tumor.py` into testable package modules while preserving
CLI behavior.

Status: NOT_STARTED

Risk: HIGH

Files likely involved:

- `embed_tumor.py`
- `projects/tumor_embedding/src/tumor_embedding/geometry.py`
- `projects/tumor_embedding/src/tumor_embedding/orientation.py`
- `projects/tumor_embedding/src/tumor_embedding/validation.py`
- `projects/tumor_embedding/src/tumor_embedding/qc.py`
- `projects/tumor_embedding/src/tumor_embedding/reports.py`
- `projects/tumor_embedding/src/tumor_embedding/cli.py`
- `projects/tumor_embedding/tests/`
- existing `tests/test_embedding_helpers.py`

Tasks:

1. Freeze current behavior with characterization tests.
   - Identify public helper functions already tested.
   - Add missing tests around CLI argument parsing, report schemas, and
     validation threshold outputs.
   - Use tiny synthetic masks only.
   - Output: behavior-lock tests.
   - Validation: focused embedding tests pass before refactor.

2. Create package skeleton.
   - Follow `docs/REPO_STRUCTURE.md`.
   - Add `projects/tumor_embedding/src/tumor_embedding/__init__.py`.
   - Keep `embed_tumor.py` as compatibility shim initially.
   - Output: package shell with no behavior move yet.
   - Validation: imports succeed.

3. Move dataclasses and pure geometry.
   - Move `PrincipalAxesResult` and pure physical/voxel geometry helpers.
   - Preserve names through re-exports where needed.
   - Output: `geometry.py`.
   - Validation: existing embedding helper tests pass.

4. Move orientation logic.
   - Separate axis sign selection and orientation confidence.
   - Preserve current orientation metrics exactly.
   - Output: `orientation.py`.
   - Validation: orientation-focused tests pass.

5. Move validation/report/QC layers.
   - Move validation thresholds and findings.
   - Move QC PNG writing.
   - Move JSON/CSV/Markdown reports after shared reporting helpers exist.
   - Output: `validation.py`, `qc.py`, `reports.py`.
   - Validation: schema tests pass.

6. Convert root script to thin CLI.
   - Keep old `python embed_tumor.py ...` entry point working.
   - Add package CLI entry point if dependency manifest supports it.
   - Output: compatibility wrapper.
   - Validation: smoke command and tests pass.

Definition of done:

- `embed_tumor.py` becomes a thin wrapper.
- Core logic lives in package modules.
- Existing tests and CLI behavior remain compatible.
- No tumor geometry semantics change.

## Milestone E - Hardcoded Path And Config Migration

Owner: Config Architecture Agent

Objective: remove personal-path defaults and move active pipelines toward
config-driven execution.

Status: NOT_STARTED

Risk: MEDIUM

Files likely involved:

- `embed_tumor.py`
- `scripts/run_batch_embedding.py`
- `scripts/generate_synthetic_longitudinal_dataset.py`
- `scripts/generate_synthetic_lollipop_cohort.py`
- `projects/tumor_embedding/configs/`
- `shared/config_loader.py`
- tests

Tasks:

1. Identify hardcoded personal/local paths.
   - Search for `~/`, `/Users/`, `/standard/`, Rivanna paths, and Downloads
     defaults.
   - Classify as doc example, test fixture, local artifact, or production
     default.
   - Output: hardcoded path inventory.
   - Validation: no edits in this task.

2. Replace production defaults with required inputs or config paths.
   - For `embed_tumor.py`, require `--mri` and `--seg` unless a config supplies
     them.
   - Preserve examples in docs, not defaults.
   - Output: safer CLI defaults.
   - Validation: CLI parser tests and error-message tests.

3. Add YAML config examples.
   - Add local smoke config.
   - Add batch embedding config.
   - Add synthetic longitudinal config.
   - Include dataset id, seed, output dir template, and input manifest paths.
   - Output: `projects/tumor_embedding/configs/*.yaml`.
   - Validation: config loader tests.

4. Wire entry points to config loader.
   - Start with optional `--config` support.
   - Keep existing CLI flags as overrides for backward compatibility.
   - Output: config-driven run path.
   - Validation: existing CLI tests plus config smoke tests.

Definition of done:

- No production script default points to a personal Downloads path.
- At least one embedding and one longitudinal run can be launched from YAML.
- Existing CLI workflows remain usable.

## Milestone F - Generated Artifact And Git Hygiene

Owner: Repository Hygiene Agent

Objective: separate source from generated data and machine-local caches.

Status: NOT_STARTED

Risk: MEDIUM_TO_HIGH

Files likely involved:

- `.gitignore`
- `analysis/**`
- `rivanna_pull/**`
- `lab_meeting_exports/**`
- docs describing artifact policy

Tasks:

1. Inventory tracked generated artifacts.
   - List tracked PNGs, NIfTIs, cache files, plots, large CSVs, and JSON outputs.
   - Separate reviewable evidence from reproducible intermediates.
   - Output: artifact tracking inventory.
   - Validation: `git ls-files analysis`.

2. Update `.gitignore`.
   - Ignore Matplotlib caches, local cache folders, generated overlays, large
     NIfTI outputs, and local smoke output directories.
   - Keep source analysis scripts and small reviewed summary reports trackable.
   - Output: ignore rules.
   - Validation: `git check-ignore -v` on known cache paths.

3. Prepare non-destructive untracking plan.
   - Propose `git rm --cached` commands only after review.
   - Do not delete working files.
   - Do not remove authoritative evidence without human approval.
   - Output: review queue.
   - Validation: dry-run command list.

4. Document artifact classes.
   - `SOURCE`
   - `REVIEWABLE_EVIDENCE`
   - `GENERATED_REPRODUCIBLE`
   - `PULLED_PROVENANCE`
   - `SENSITIVE_OR_REMOTE_DATA`
   - Output: artifact policy doc.
   - Validation: docs identify examples from current repo.

Definition of done:

- Future generated caches do not appear in `git status`.
- Existing tracked generated files have a reviewed untracking plan.
- No data is deleted.

## Milestone G - Rivanna Pull Contract

Owner: Provenance / Remote Boundary Agent

Objective: make `rivanna_pull/` read-only provenance, not a competing source
tree.

Status: NOT_STARTED

Risk: MEDIUM

Files likely involved:

- `rivanna_pull/**`
- `scripts/generate_synthetic_lollipop_cohort.py`
- `scripts/extract_real_tumor_features.py`
- `docs/CURRENT_AUTHORITATIVE_ARTIFACTS.md`
- new `docs/RIVANNA_PULL_POLICY.md`

Tasks:

1. Compare duplicate scripts.
   - Diff `scripts/` against `rivanna_pull/scripts/`.
   - Classify differences as historical provenance, source-of-truth changes, or
     unknown drift.
   - Output: duplicate script comparison table.
   - Validation: no edits.

2. Define `rivanna_pull/` policy.
   - Decide and document that pulled files are read-only provenance snapshots
     unless explicitly promoted.
   - Require all active edits to happen outside `rivanna_pull/`.
   - Output: policy doc.
   - Validation: docs link from current status.

3. Add visual/source warnings.
   - Add README inside `rivanna_pull/` if appropriate.
   - Mark duplicate scripts as provenance copies via docs, not code edits.
   - Output: human-readable boundary.
   - Validation: no source behavior change.

4. Add promotion procedure.
   - If a pulled script contains needed behavior, copy the behavior into active
     source through a reviewed patch.
   - Record original hash and source path.
   - Output: promotion checklist.
   - Validation: provenance preserved.

Definition of done:

- Developers know which script copy is authoritative.
- `rivanna_pull/` is not silently edited as active source.

## Milestone H - Documentation Consolidation

Owner: Documentation Architecture Agent

Objective: reduce status-doc sprawl and make canonical docs obvious.

Status: NOT_STARTED

Risk: LOW_TO_MEDIUM

Files likely involved:

- `docs/CURRENT_PROJECT_STATUS.md`
- `docs/GROWTHNET_ACTION_PLAN.md`
- `docs/GROWTHNET_LOCAL_WORK_REPORT.md`
- `docs/AUTONOMOUS_PRIORITY_LOOP.md`
- `docs/archive/`
- root-level session files

Tasks:

1. Classify docs by role.
   - Current status.
   - Action plan.
   - Architecture policy.
   - Scientific validation evidence.
   - Historical session logs.
   - Output: doc role map.
   - Validation: no edits.

2. Choose canonical living docs.
   - Keep `CURRENT_PROJECT_STATUS.md` as current state.
   - Keep `GROWTHNET_ACTION_PLAN.md` as task roadmap.
   - Keep this file as architecture remediation roadmap.
   - Move dated snapshots only after review.
   - Output: canonical-doc index.
   - Validation: docs link to each other consistently.

3. Create archive strategy.
   - Add `docs/archive/README.md`.
   - Define when a doc is archived versus updated.
   - Do not delete historical reports.
   - Output: archive policy.
   - Validation: old links remain understandable.

4. Normalize status language.
   - Use statuses already used in the project:
     `COMPLETE`, `ACCEPT_WITH_FOLLOWUP`, `BLOCKED_REMOTE_DATA`,
     `HUMAN_REVIEW_REQUIRED`, `NOT_STARTED`.
   - Remove unsupported clinical claims.
   - Output: cleaner docs.
   - Validation: status terms are grep-consistent.

Definition of done:

- A new contributor can tell which docs to read first.
- Historical autonomous reports remain available but are not mistaken for the
  latest state.

## Milestone I - Analysis Code Boundary

Owner: Analysis Tooling Agent

Objective: separate reusable analysis code from generated analysis outputs.

Status: NOT_STARTED

Risk: MEDIUM

Files likely involved:

- `analysis/**`
- `projects/growthnet_analysis/src/`
- `projects/growthnet_analysis/scripts/`
- tests

Tasks:

1. Inventory analysis scripts versus outputs.
   - Identify Python files that are reusable tools.
   - Identify CSV/PNG/JSON/Markdown outputs.
   - Identify one-off notebooks or experimental scratch files.
   - Output: analysis inventory.
   - Validation: no edits.

2. Promote reusable analysis scripts.
   - Move stable scripts into a project package or `scripts/` with clear CLI.
   - Leave outputs in `analysis/` or an ignored output directory.
   - Preserve relative path behavior where possible.
   - Output: reusable analysis module layout.
   - Validation: existing analysis commands still run or have documented new
     commands.

3. Add tests for analysis helpers.
   - Test CSV schema parsing.
   - Test metric calculations on tiny arrays.
   - Test report generation without real data.
   - Output: focused tests.
   - Validation: fast suite remains laptop-safe.

Definition of done:

- Analysis code is importable and testable.
- Generated outputs are clearly separated from source.

## Milestone J - Linting And Type Discipline

Owner: Static Analysis Agent

Objective: add automated style/type checks gradually without blocking current
research work.

Status: NOT_STARTED

Risk: LOW_TO_MEDIUM

Files likely involved:

- `pyproject.toml`
- `.github/workflows/fast-tests.yml`
- source modules

Tasks:

1. Add ruff configuration.
   - Start with conservative rules.
   - Exclude generated/pulled artifact trees.
   - Avoid mass formatting in the first pass.
   - Output: ruff config.
   - Validation: ruff runs on selected source paths.

2. Add import/order cleanup only where touched.
   - Do not autoformat the whole repo immediately.
   - Fix violations in new/actively edited modules first.
   - Output: small diffs.
   - Validation: ruff passes on selected paths.

3. Add mypy later as advisory.
   - Start with new shared modules and package boundaries.
   - Keep scientific scripts out of strict mode initially.
   - Output: gradual typing config.
   - Validation: advisory type check passes on selected modules.

Definition of done:

- New architecture modules have automated style checks.
- CI can enforce a narrow stable subset without drowning the repo in legacy
  failures.

## Milestone K - Longitudinal Dataset Pipeline Completion

Owner: Dataset Integration Agent

Objective: turn current longitudinal outputs into a locally smoke-tested
training dataset interface.

Status: PARTIAL_COMPLETE

Risk: MEDIUM

Files likely involved:

- `projects/vivit/src/data/synthetic_longitudinal_loader.py`
- `scripts/generate_synthetic_longitudinal_dataset.py`
- tests
- `docs/TRAINING_DATASET_READINESS_GATES.md`

Tasks:

1. Run full adapter NIfTI smoke.
   - Use a tiny local generated fixture.
   - Load images and masks through the metadata adapter.
   - Verify shapes, channel dimensions, dates, labels, and scan IDs.
   - Output: test or smoke report.
   - Validation: focused loader test passes.

2. Add metadata-based split generation.
   - Split by source patient, not by visit or variant.
   - Prevent leakage across train/val/test.
   - Preserve deterministic seed.
   - Output: split JSON or CSV.
   - Validation: tests verify no patient leakage.

3. Add training-loader smoke.
   - Build a tiny dataloader batch.
   - Do not train a model at scale.
   - Verify tensors can pass through the expected preprocessing stack.
   - Output: integration test or documented smoke command.
   - Validation: laptop-safe smoke passes.

Definition of done:

- Generated longitudinal outputs can be consumed by the local model data path.
- Training-dataset readiness remains scientifically blocked until real
  longitudinal validation exists.

## Milestone L - Governance And Safe Execution

Owner: Technical Lead / Integration Review Agent

Objective: keep remediation work auditable while the worktree is dirty.

Status: NOT_STARTED

Risk: MEDIUM

Files likely involved:

- `docs/GROWTHNET_AGENT_STATUS.md`
- `docs/GROWTHNET_ACTION_PLAN.md`
- `docs/AUTONOMOUS_PRIORITY_LOOP.md`

Tasks:

1. Define commit grouping.
   - Dependencies and CI.
   - Shared utilities.
   - Embedding package split.
   - Config migration.
   - Docs consolidation.
   - Artifact hygiene.
   - Output: commit plan.
   - Validation: staged diffs are coherent.

2. Define file ownership by phase.
   - Prevent concurrent edits to `embed_tumor.py`.
   - Prevent concurrent edits to `.gitignore` during artifact cleanup.
   - Prevent generated artifact untracking without review.
   - Output: ownership table.
   - Validation: no overlapping implementation agents.

3. Add review checkpoints.
   - After shared utilities.
   - Before `embed_tumor.py` split.
   - Before `git rm --cached`.
   - Before CLI default changes.
   - Output: review queue.
   - Validation: high-risk changes are not silently merged.

Definition of done:

- Each remediation milestone can be executed by autonomous agents without
  stepping on each other.
- High-risk changes have explicit review gates.

## Dependency Graph

```text
A Dependency Baseline
  -> B Fast CI
  -> J Linting

C Shared Utility Layer
  -> D Tumor Embedding Package Boundary
  -> E Config Migration
  -> I Analysis Code Boundary

F Artifact Hygiene
  -> H Documentation Consolidation
  -> L Governance

G Rivanna Pull Contract
  -> H Documentation Consolidation
  -> L Governance

K Longitudinal Dataset Pipeline Completion
  depends on current synthetic_longitudinal_loader adapter
  can run in parallel with A/B/C if file ownership is isolated

D Tumor Embedding Package Boundary
  should not start until C shared helpers and behavior-lock tests are stable
```

## Parallel Work Groups

Group 1, safe immediately:

- Milestone A dependency inventory
- Milestone B CI draft
- Milestone F generated artifact inventory
- Milestone G duplicate script comparison
- Milestone H doc role map
- Milestone K adapter NIfTI smoke

Group 2, after inventories:

- Milestone C shared provenance/reporting helpers
- Milestone B CI activation
- Milestone F `.gitignore` update
- Milestone H canonical-doc index

Group 3, after shared helpers:

- Milestone D `embed_tumor.py` package split
- Milestone E config migration
- Milestone I analysis code promotion
- Milestone J static analysis enforcement

## Human Review Gates

Human review required before:

- removing or untracking any currently tracked analysis evidence,
- changing `embed_tumor.py` public CLI defaults,
- moving large parts of `embed_tumor.py`,
- declaring any generated dataset training-ready,
- changing `rivanna_pull/` from read-only provenance to active source,
- setting scientific acceptance thresholds.

## Recommended First Six Implementation Tasks

1. Dependency inventory and `pyproject.toml` draft.
2. Minimal fast-test GitHub Actions workflow.
3. Shared provenance/reporting helper extraction.
4. Full NIfTI smoke for `synthetic_longitudinal_loader.py`.
5. Artifact tracking inventory plus `.gitignore` proposal.
6. Duplicate `scripts/` vs `rivanna_pull/scripts/` policy document.

These six tasks are mostly independent, low to medium risk, and give the repo a
cleaner spine before any high-risk `embed_tumor.py` surgery.
