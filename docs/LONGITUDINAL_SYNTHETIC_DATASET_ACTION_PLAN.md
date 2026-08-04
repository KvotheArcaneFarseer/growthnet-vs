# Longitudinal Synthetic Dataset Generator Action Plan

Last updated: 2026-08-01

Scope: local GrowthNet repository only. This plan improves the longitudinal dataset generator pipeline without changing tumor geometry, tuning morphology, or inventing clinical growth laws.

## Current State

The longitudinal wrapper in `scripts/generate_synthetic_longitudinal_dataset.py` is an MVP orchestration layer over `embed_tumor.py`. It can generate four visits per patient from explicit requested volumes, preserve one background MRI ID per patient, write image/mask outputs, and emit per-visit metadata/QC.

Current output quality is engineering-usable but not training-dataset complete. The strongest local improvements so far are deterministic seed routing, per-visit QC, additive provenance metadata, and optional dataset provenance JSON.

## Missing Factors For Dataset-Generator Readiness

- Sparse and irregular visit schedules are not supported.
- Time intervals and elapsed days are not represented.
- Each visit is generated independently rather than as one continuous tumor trajectory.
- Longitudinal QC does not yet evaluate monotonicity, centroid drift, axis drift, intensity stability, or background consistency beyond ID reuse.
- ViViT-ready export manifests and split files are missing.
- Real longitudinal source masks are unavailable locally.
- `stable` and `growing` label semantics require human scientific review.
- Training-readiness criteria are not yet separated cleanly from engineering smoke criteria.

## Agent-Based Execution Plan

| Task ID | Agent | Objective | Subtasks | Expected Outputs | Validation | Status |
|---|---|---|---|---|---|---|
| LONG-PLAN-001 | Agent L0: Orchestrator | Maintain the longitudinal execution roadmap. | Track dependencies; update status docs; keep scientific claims conservative. | This document and status-ledger updates. | Markdown review plus fast test suite. | COMPLETE |
| LONG-PROV-001 | Agent L1: Provenance | Make longitudinal outputs auditable. | Add optional provenance JSON; hash timeline/background/metadata/QC files; record script hashes and generation parameters; count QC pass/fail rows. | `synthetic_longitudinal_provenance_v1` sidecar when requested. | Focused longitudinal tests and mocked sidecar write. | COMPLETE |
| LONG-META-001 | Agent L2: Metadata | Strengthen per-visit traceability. | Add visit seed; source MRI path; source segmentation path; embedding growth mode; volume tolerance; max iterations; gen size. | Additive metadata columns in `metadata.csv`. | Mocked metadata schema assertions. | COMPLETE |
| LONG-SPARSE-001 | Agent L3: Timeline Semantics | Support sparse and irregular visits without clinical assumptions. | Define row-wise timeline schema; support `patient_id,timepoint,day,target_volume_mm3`; preserve fixed four-visit schema as legacy; sort visits deterministically. | New parser and docs for sparse timelines. | Unit tests for ordering, missing days, duplicate visits. | NOT_STARTED |
| LONG-QC-001 | Agent L4: Longitudinal QC | Add temporal quality checks. | Compute per-patient monotonicity; achieved-vs-requested trend; centroid drift; axis drift; same-background consistency; failure reasons. | `longitudinal_qc_summary.csv` or added QC sections. | Synthetic pass/fail fixtures, no real MRI dependency. | NOT_STARTED |
| LONG-LAYOUT-001 | Agent L5: Training Layout | Produce model-consumable manifests. | Define ViViT/TemporalUNETR manifest columns; add optional export layout; preserve raw output layout. | `vivit_manifest.csv`, split manifest, layout docs. | Loader smoke test or schema test. | NOT_STARTED |
| LONG-SCHEMA-001 | Agent L6: Schema Validation | Prevent silent bad inputs/outputs. | Validate positive volumes; duplicate patient/timepoint pairs; path readability; allowed labels; output row counts. | Clear input validation errors and schema docs. | Unit tests for malformed timelines/backgrounds. | PARTIAL |
| LONG-REAL-001 | Agent L7: Real Data Readiness | Prepare real validation, locally staged only. | Specify required real longitudinal masks/backgrounds; define minimum cohort; mark unavailable data blocked. | Acquisition spec and blocked-data list. | Documentation review. | BLOCKED_REMOTE_DATA |
| LONG-GROWTH-001 | Agent L8: Growth Law Encoding | Encode a default-off empirical VS growth-law candidate. | Add visit-day parser; sample deterministic annual volumetric rate; derive target volumes from T1 baseline; preserve default CSV target behavior. | `--clinical_growth_law empirical_vs_v1` and law metadata. | Focused tests and no-MRI smoke. | COMPLETE |
| LONG-SCI-001 | Agent L9: Scientific Review | Guard clinical interpretation. | Review `stable`/`growing` semantics; decide whether independent-visit generation is acceptable for any training use; define acceptance thresholds. | Human decision record. | Human review required. | HUMAN_REVIEW_REQUIRED |

## Parallelization

Can run now in parallel:

- `LONG-SPARSE-001`
- `LONG-QC-001`
- `LONG-LAYOUT-001`
- `LONG-SCHEMA-001`

Blocked or gated:

- `LONG-REAL-001` is blocked until real longitudinal masks/backgrounds are locally available.
- `LONG-SCI-001` requires human scientific review.
- `LONG-GROWTH-001` is encoded but not clinically validated.
- Any generator tuning remains blocked until real validation supports it.

## Execution Order

1. Finish provenance and metadata auditability. Status: complete.
2. Add sparse/irregular timeline parser while preserving current four-visit CSV compatibility.
3. Encode a default-off empirical clinical growth-law candidate. Status: complete.
4. Add longitudinal QC summaries using local synthetic fixtures.
5. Add model-ready export manifests.
6. Validate with local smoke fixtures and fast tests.
7. Stage annotated real longitudinal data when available.
8. Decide scientific label semantics and training acceptance criteria.

## Non-Negotiables

- Do not infer clinical growth laws from `stable` or `growing` labels.
- Do not change tumor geometry in this workstream.
- Do not use remote data or Rivanna.
- Do not mark training-dataset readiness complete from engineering tests alone.
