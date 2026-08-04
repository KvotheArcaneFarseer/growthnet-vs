# Local Anatomy Inventory

Generated: 2026-07-18T12:46:11.550036+00:00

Scope: local repository only. No Rivanna, SSH, or remote data were used.

## Evidence Inventory

| evidence | classification | local_status |
| --- | --- | --- |
| Synthetic masks | AVAILABLE_LOCAL | 261 NIfTI masks under `rivanna_pull/analysis/synthetic_lollipop_v1/masks` |
| Synthetic manifest | AVAILABLE_LOCAL | 261 rows with case ID, target/realized volume, calibration scale, rotations, and seed |
| Generator implementation | AVAILABLE_LOCAL | `projects/vivit/src/data/synthetic.py` and `scripts/generate_synthetic_lollipop_cohort.py` define lollipop geometry |
| Authoritative synthetic features | AVAILABLE_LOCAL | `analysis/synthetic_features_v2/synthetic_features_v2.csv` |
| Pulled real feature table | AVAILABLE_LOCAL | 291 feature rows; spacing columns show local feature provenance but not compartment labels |
| Canal/CPA labels | REQUIRES_HUMAN_ANNOTATION | No explicit real canal, porus, fundus, or CPA labels found locally |
| Synthetic compartment labels | DERIVABLE_LOCAL | Can be reconstructed from generator convention, manifest scale, seed, and rotations |
| Real compartment labels | REQUIRES_REAL_MASKS | Whole-mask feature table cannot separate intracanalicular and CPA components |
| Anatomical landmarks | REQUIRES_HUMAN_ANNOTATION | Porus/fundus/canal axis landmarks are not present in local real artifacts |
| Remote clinical source data | BLOCKED_REMOTE_DATA | Original real source masks/MRI are not present locally |

## Real Feature Spacing Evidence

| index | voxel_spacing_mm_x | voxel_spacing_mm_y | voxel_spacing_mm_z |
| --- | --- | --- | --- |
| count | 291 | 291 | 291 |
| mean | 0.5 | 0.5 | 0.5 |
| std | 0 | 0 | 0 |
| min | 0.5 | 0.5 | 0.5 |
| 25% | 0.5 | 0.5 | 0.5 |
| 50% | 0.5 | 0.5 | 0.5 |
| 75% | 0.5 | 0.5 | 0.5 |
| max | 0.5 | 0.5 | 0.5 |

## What Can Be Validated Locally

- Synthetic lollipop compartment geometry can be plausibility-checked from current masks and generator metadata.
- Synthetic stem/bulb volume fractions, widths, axial extents, centroid offsets, transition overlap, and artificial plateaus can be measured.
- Whole-mask real-vs-synthetic feature tables can be used only to define volume strata and broad context.

## What Cannot Be Validated Locally

- Real intracanalicular versus CPA compartment volumes.
- Real porus/fundus boundaries.
- Real canal-axis direction unless source masks and/or landmarks are provided.
- Clinical/anatomical validity of generator compartment ratios.
