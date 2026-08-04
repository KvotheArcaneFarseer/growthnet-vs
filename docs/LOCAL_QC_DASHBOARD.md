# Local QC Dashboard

Last updated: 2026-08-01

This dashboard is a compact status view for the local GrowthNet repository. It separates engineering readiness from scientific validation.

## Engineering QC

| Area | Current Status | Evidence |
|---|---|---|
| Fast local tests | PASS | `.venv/bin/python -m pytest -m "fast and not slow" -v`: 48 passed, 9 deselected, 14 dependency warnings. |
| Synthetic cohort generation | IMPROVED | Optional provenance JSON and optional compartment-label sidecars added for future runs. Defaults remain backward compatible. |
| Longitudinal wrapper | IMPROVED | Optional provenance JSON and additive per-visit traceability metadata added; no-MRI failure smoke writes auditable failure rows. |
| Clinical growth-law mode | EXPERIMENTAL | Default-off `empirical_vs_v1` target-volume mode added; locally tested for determinism and provenance, not clinically validated. |
| Saved-mask provenance | COMPLETE | Recovered pulled generator path reproduced 10/10 representative saved masks exactly. |
| Authoritative synthetic features | COMPLETE | 261/261 features regenerated in `analysis/synthetic_features_v2/`. |
| Compartment mapping | CURRENT_SYNTHETIC_ONLY | V2 recovered-generator audit confirmed 12/12 visual mappings. |

## Scientific QC

| Area | Current Status | Interpretation |
|---|---|---|
| Whole-mask morphology | ACCEPT_WITH_FOLLOWUP | Regenerated features reject earlier broad "too elongated" concern, but not all morphology questions are settled. |
| Surface metrics | RESOLUTION_CONFOUNDED | Sphericity, compactness, surface area, and surface-to-volume ratio require common-resolution comparison. |
| Synthetic compartments | SYNTHETIC_PLAUSIBILITY_ONLY | V2 shows 135/261 zero-bulb and strong stem dominance. This is a synthetic design signal, not clinical invalidity. |
| Real compartment validation | BLOCKED_REMOTE_DATA | Requires real source masks plus landmarks or compartment annotations. |
| Longitudinal realism | HUMAN_REVIEW_REQUIRED | Current wrapper consumes requested volumes; it does not define a clinical growth law or continuous tumor trajectory. |
| Clinical growth-law validation | BLOCKED_REMOTE_DATA | `empirical_vs_v1` requires real longitudinal masks for validation or recalibration before training claims. |
| Generator tuning | NOT_JUSTIFIED_YET | Wait for real compartment validation before changing morphology. |

## Current Blockers

| Blocker | Type | Next Action |
|---|---|---|
| Real source masks unavailable locally | BLOCKED_REMOTE_DATA | Stage annotated 30-case subset when available. |
| No real canal/CPA annotations | HUMAN_REVIEW_REQUIRED | Choose annotation standard: porus/fundus, canal axis, IAC mask, or CPA boundary. |
| No explicit historical compartment labels | TECHNICAL_DEBT | Future generator now supports optional labels; old saved masks remain reconstruction-based. |
| Surface metrics resolution sensitivity | SCIENTIFIC_VALIDATION | Re-extract real masks at common resolution when source masks are available. |

## Recommended Next Checks

1. Run fast suite after every implementation slice:

```bash
.venv/bin/python -m pytest -m "fast and not slow" -v
```

2. Smoke future synthetic provenance output:

```bash
.venv/bin/python scripts/generate_synthetic_lollipop_cohort.py \
  --targets_csv /path/to/targets.csv \
  --out_dir /tmp/growthnet_synthetic_smoke \
  --provenance_json /tmp/growthnet_synthetic_smoke/provenance.json \
  --write_compartment_labels
```

3. Keep morphology tuning blocked until real compartment validation exists.
