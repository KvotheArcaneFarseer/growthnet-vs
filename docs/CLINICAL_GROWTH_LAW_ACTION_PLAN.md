# Clinical Growth Law Action Plan

Last updated: 2026-08-01

Scope: local GrowthNet repository only. This plan encodes and validates a vestibular schwannoma volumetric growth-law candidate without claiming clinical validity before real longitudinal validation.

## Literature Basis

Initial candidate: `empirical_vs_v1`.

The model is based on untreated vestibular schwannoma observation literature reporting that volumetric growth is often categorized using annual percentage volume change, with common thresholds around stable versus growing and a high-growth subgroup.

Evidence used for initial encoding:

- Lees et al., "Volumetric growth rates of untreated vestibular schwannomas", PubMed: https://pubmed.ncbi.nlm.nih.gov/31374553/
  - Reported mean volumetric growth rate around 33.5% per year.
  - Reported approximate groups: 66% growing, 33% stable, 1% shrinking; 30% fast-growing.
  - Defines growing as >20% volume increase per year and fast-growing as >100% per year.
- "Long-term natural history and patterns of sporadic vestibular schwannoma", PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9340632/
  - Uses volumetric growth threshold >=20%.
  - Reports initial volumetric growth rate distributions and long-term growth patterns.
- "Growth rate of vestibular schwannoma", PubMed: https://pubmed.ncbi.nlm.nih.gov/27450283/
  - Summarizes linear growth rates around 0.99-1.11 mm/year, with higher rates after demonstrated first-follow-up growth.

## Encoded Candidate

`empirical_vs_v1` is an experimental volumetric law:

- Uses baseline `T1_volume_mm3`.
- Uses explicit visit days from `--visit_days`.
- Uses exponential volume evolution: `V(t) = V0 * exp(log(1 + annual_rate) * years)`.
- For `stable`, samples annual volume-change fraction uniformly from `[-0.05, 0.05]`.
- For `growing`, samples a conditional mixture:
  - 45% fast-growing: `[1.0, 2.0]` annual fraction.
  - 55% moderate-growing: `[0.20, 1.0]` annual fraction.
- Deterministic per patient through `seed`, `patient_id`, and `"clinical_growth_law"`.

This is not validated clinical truth. It is a transparent, testable prior.

## Agent-Based Plan

| Task ID | Agent | Objective | Subtasks | Expected Outputs | Validation | Status |
|---|---|---|---|---|---|---|
| GROWTH-LIT-001 | Agent G1: Literature | Maintain evidence basis. | Track primary sources; extract thresholds; separate volumetric from linear findings; document uncertainty. | Literature basis table and citations. | Source review. | COMPLETE |
| GROWTH-ENC-001 | Agent G2: Encoding | Encode a default-off empirical law. | Add parser for visit days; deterministic annual-rate sampler; target-volume generator; metadata fields. | `--clinical_growth_law empirical_vs_v1`. | Unit tests and no-MRI smoke. | COMPLETE |
| GROWTH-PROV-001 | Agent G3: Provenance | Make generated trajectories auditable. | Record law name; visit days; annual rate; target source; generated targets; provenance JSON. | Metadata/provenance fields. | Mocked integration tests. | COMPLETE |
| GROWTH-QC-001 | Agent G4: Synthetic QC | Validate generated target trajectories locally. | Check monotonicity for growing; stability bounds for stable; finite positive targets; pass/fail reasons. | Longitudinal growth-law QC table. | Synthetic fixture tests. | NOT_STARTED |
| GROWTH-CAL-001 | Agent G5: Calibration | Fit/tune parameters only from real longitudinal masks. | Estimate empirical annual-rate distribution; compare candidate distribution; revise only with evidence. | Calibration report and parameter table. | Requires real longitudinal volumes. | BLOCKED_REMOTE_DATA |
| GROWTH-VAL-001 | Agent G6: Validation | Validate clinical plausibility. | Compare synthetic and real trajectories by volume bin, baseline volume, growth label, and follow-up interval. | Clinical growth-law validation report. | Real annotated longitudinal cohort. | BLOCKED_REMOTE_DATA |
| GROWTH-SCI-001 | Agent G7: Scientific Review | Decide whether the law is acceptable for training use. | Review assumptions; approve/reject thresholds; decide whether labels are clinically meaningful. | Human decision record. | Human review. | HUMAN_REVIEW_REQUIRED |
| GROWTH-DOC-001 | Agent G8: Documentation | Keep docs honest. | Mark `empirical_vs_v1` as experimental; document flags; warn against clinical claims. | Status docs and command reference. | Markdown review. | COMPLETE |

## Validation Gate

The encoded law can be used for local synthetic experimentation only if:

- `--clinical_growth_law` is explicitly set.
- Generated targets are positive and finite.
- Metadata records `target_volume_source=clinical_growth_law`.
- Provenance records law name and visit days.

It cannot be used as a clinical validity claim until:

- real longitudinal masks are available,
- real volumes are re-extracted locally,
- growth-rate distributions are compared,
- human scientific review accepts the resulting assumptions.

## Current Command

```bash
.venv/bin/python scripts/generate_synthetic_longitudinal_dataset.py \
  --timeline_csv timeline.csv \
  --background_csv backgrounds.csv \
  --out_dir outputs/synthetic_longitudinal_empirical \
  --provenance_json outputs/synthetic_longitudinal_empirical/provenance.json \
  --clinical_growth_law empirical_vs_v1 \
  --visit_days 0,365.25,730.5,1095.75
```

## Non-Negotiables

- Default mode remains `none`, preserving explicit `T1..T4` timeline volumes.
- Do not call `empirical_vs_v1` clinically validated.
- Do not tune morphology from growth-law behavior.
- Do not fit parameters without real longitudinal data.
