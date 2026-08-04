# Current Authoritative Artifacts

Last updated: 2026-08-01

Scope: local GrowthNet repository only.

## Synthetic Cohort

| Artifact | Status | Notes |
|---|---|---|
| `rivanna_pull/analysis/synthetic_lollipop_v1/masks/` | AUTHORITATIVE_LOCAL_INPUT | The 261 saved masks are the authoritative local synthetic mask cohort. Do not overwrite. |
| `rivanna_pull/analysis/synthetic_lollipop_v1/manifests/synthetic_lollipop_manifest.csv` | AUTHORITATIVE_LOCAL_INPUT | Manifest for the saved masks. It is sufficient for reproduction only with the recovered pulled generator script. |
| `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py` | AUTHORITATIVE_HISTORICAL_GENERATOR | Reproduced 10/10 representative saved masks exactly. |
| `scripts/generate_synthetic_lollipop_cohort.py` | CURRENT_FUTURE_GENERATOR | Current local generator for future runs. It now supports optional provenance JSON and optional compartment-label sidecars. It is not the historical generator for the saved 261-mask cohort. |
| `analysis/synthetic_features_v2/synthetic_features_v2.csv` | AUTHORITATIVE_LOCAL_FEATURES | Regenerated from the authoritative local masks with current extractor; 261/261 succeeded. |

## Superseded Or Legacy

| Artifact | Status | Notes |
|---|---|---|
| Pulled synthetic feature CSV/JSON under `rivanna_pull/analysis/synthetic_lollipop_v1/` | STALE_LEGACY | Does not reproduce from current authoritative masks; do not use for morphology tuning. |
| `analysis/anatomical_compartment_validation/` | EXPLORATORY_SUPERSEDED | Original compartment audit used the wrong generator reconstruction path for saved-mask interpretation. Keep for provenance, not quantitative cohort conclusions. |

## Current Validation Outputs

| Artifact | Status | Main Finding |
|---|---|---|
| `analysis/saved_mask_provenance/` | COMPLETE | Recovered the representative saved-mask generation path. |
| `analysis/anatomical_compartment_validation_v2/` | CURRENT_SYNTHETIC_COMPARTMENT_AUDIT | 135/261 zero-bulb, 0/261 >1% unmatched, 9/261 multi-component, 12/12 visual mappings confirmed. |
| `analysis/real_vs_synthetic_v2/` | ACCEPT_WITH_FOLLOWUP | Regenerated features reject the earlier overall "too elongated" concern; surface and compartment findings still require care. |
| `analysis/surface_resolution_validation/` | ACCEPT_WITH_FOLLOWUP | Surface metrics are resolution-sensitive and should be compared only after normalization/re-extraction. |
| `analysis/volume_targeting/` | ACCEPT_WITH_FOLLOWUP | Standalone smoke thresholds are useful; scientific thresholds need broader validation. |

## Future Output Requirements

Every new synthetic cohort should save:

- provenance JSON
- generator script hash
- synthetic module hash
- manifest hash
- run parameters
- dependency versions
- spacing
- seed and rotation per case
- optional per-voxel compartment labels

The current future generator supports:

```bash
.venv/bin/python scripts/generate_synthetic_lollipop_cohort.py \
  --targets_csv targets.csv \
  --out_dir output_masks \
  --manifest_csv output_manifest.csv \
  --provenance_json output_provenance.json \
  --write_compartment_labels
```
