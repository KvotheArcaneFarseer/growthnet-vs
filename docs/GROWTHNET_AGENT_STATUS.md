# GrowthNet Agent Status

Last updated: 2026-07-18

Scope: local-only orchestration. Agents must not SSH, use Rivanna, delete generated data, or edit outside this repository.

## Launch Status

| Agent | Task IDs | Scope | File Ownership | Status | Branch/Isolation | Last Update |
|---|---|---|---|---|---|---|
| Agent 1 Orientation and Embedding Validation | EMB-001 | diagnose local axis-error mechanism | `analysis/orientation_validation/**` | COMPLETE | sub-agent `019f7394-28f1-70c2-bf00-9f7b4aa1a741` | local diagnostic passed |
| Agent 2 Batch Pipeline Local Reliability | BATCH-001 | local batch resumability/resource audit | `analysis/batch_reliability/**`, optional `scripts/run_batch_embedding.py` | COMPLETE | sub-agent `019f7394-5221-7b01-89fe-68d1ce3a2d51` | resume/status implementation validated locally |
| Agent 3 Longitudinal Dataset Audit | LONG-001 | audit wrapper and metadata/repro tests | `docs/LONGITUDINAL_PIPELINE_AUDIT.md`, `data/timelines/local_longitudinal_example.csv`, `tests/test_longitudinal_*.py` | COMPLETE | sub-agent `019f7394-77ce-7db2-9cad-4d67cd7eeb72` closed | unittest passed; pytest unavailable |
| Agent 4 Volume Targeting Validation | VOL-001 | quantify target-volume accuracy | `analysis/volume_targeting/**` | COMPLETE | sub-agent `019f7394-a4d6-7811-967d-89fba88611e7` | 36-case local standalone benchmark complete |
| Agent 5 Morphology and Real-vs-Synthetic Audit | MORPH-001 | local morphology comparison | `analysis/real_vs_synthetic/**` | COMPLETE | sub-agent `019f7394-c7cb-7111-8227-25f2bf7453ba` | local comparison complete; provenance drift found |
| Agent 6 Testing and Reproducibility | TEST-001 | fast deterministic tests | `tests/**`, `pytest.ini`, optional test docs | COMPLETE | sub-agent `019f7394-eb01-7e40-b956-cd44ff60c414` | tests compile; pytest unavailable |
| Agent 7 Documentation and Repository Audit | DOC-001 | local project docs | `README.md`, `docs/LOCAL_REPRODUCIBILITY.md`, `docs/KNOWN_LIMITATIONS.md`, `docs/CURRENT_PROJECT_STATUS.md` | COMPLETE | sub-agent `019f7397-e294-7581-a140-d63d4353950f` | local docs updated |
| Agent 8 Integration and Review | REVIEW-001 | review queue | `docs/GROWTHNET_AGENT_STATUS.md` | COMPLETE | lead integration review | specialist outputs reviewed and classified |

## Review Queue

| Task ID | Review Status | Reviewer Notes | Human Review |
|---|---|---|---|
| EMB-001 | ACCEPT_WITH_FOLLOWUP | Local standalone masks do not support spatial bug or PCA identity switch; patient-space named cases are BLOCKED_REMOTE_DATA. | yes |
| BATCH-001 | ACCEPT_WITH_FOLLOWUP | Resume/status behavior is backward compatible by default and validated locally; scientific smoke lacks clean original MRI/seg fixture. | no |
| LONG-001 | ACCEPT_WITH_FOLLOWUP | Scoped audit and tests. Follow-up needs human review of `stable`/`growing` label semantics. | yes |
| VOL-001 | ACCEPT_WITH_FOLLOWUP | Standalone mask volume targeting is quantified; thresholds are engineering recommendations, not publication criteria. | yes |
| MORPH-001 | HUMAN_REVIEW_REQUIRED | Local comparisons are complete, but pulled synthetic features do not reproduce from local masks with current extractor. Do not tune morphology until provenance drift is resolved. | yes |
| TEST-001 | ACCEPT_WITH_FOLLOWUP | Test files compile and helper smoke passed, but normal pytest execution is blocked by missing pytest dependency. | no |
| DOC-001 | ACCEPT_WITH_FOLLOWUP | Local docs now distinguish implemented, partial, experimental, unvalidated, and remote-blocked work; scientific wording should be reviewed before publication use. | yes |

## Coordination Rules In Force

- No concurrent core edits to `embed_tumor.py` or `projects/vivit/src/data/synthetic.py`.
- No generator morphology tuning until `MORPH-001` evidence exists and is reviewed.
- No QC redesign until `EMB-001` diagnosis is supported.
- No full training-ready cohort generation in this session.
- No destructive cleanup.
- No remote data assumptions. Mark remote-only work `BLOCKED_REMOTE_DATA`.

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
