# Synthetic Compartment Audit

Generated: 2026-07-18T12:46:16.689860+00:00

## Scope

This is a synthetic-only anatomical compartment plausibility audit. It reconstructs the current lollipop generator's canal/CPA coordinate frame from local manifest metadata and measures current authoritative masks. It does not validate real anatomy and does not justify generator tuning by itself.

## Inputs

- Manifest: `rivanna_pull/analysis/synthetic_lollipop_v1/manifests/synthetic_lollipop_manifest.csv` (261 analyzed rows)
- Mask root: `rivanna_pull/analysis/synthetic_lollipop_v1/masks`
- Synthetic features: `analysis/synthetic_features_v2/synthetic_features_v2.csv`
- Git commit: `b7db1cc55f708ed71f4a9b11da1aef5cc27e3f5e`

## Reproducibility Commands

- `MPLCONFIGDIR=analysis/anatomical_compartment_validation/.matplotlib-cache XDG_CACHE_HOME=analysis/anatomical_compartment_validation/.matplotlib-cache .venv/bin/python analysis/anatomical_compartment_validation/analyze_synthetic_compartments.py`
- `.venv/bin/python -m pytest tests/test_anatomical_compartment_helpers.py -v`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/anatomical_compartment_validation/analyze_synthetic_compartments.py tests/test_anatomical_compartment_helpers.py`
- `.venv/bin/python -m pytest -m "fast and not slow" -v`
- `git diff --check`

## Outputs

- `analysis/anatomical_compartment_validation/synthetic_compartment_features.csv`
- `analysis/anatomical_compartment_validation/synthetic_compartment_summary.csv`
- `analysis/anatomical_compartment_validation/LOCAL_ANATOMY_INVENTORY.md`
- `analysis/anatomical_compartment_validation/COMPARTMENT_METRIC_SPEC.md`
- `analysis/anatomical_compartment_validation/REAL_MASK_ACQUISITION_SPEC.md`

## Summary

| volume_bin | metric | n | median | iqr | p25 | p75 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | stem_fraction | 261 | 0.94964 | 0.103321 | 0.896679 | 1 | 0.692105 | 1 |
| all | bulb_fraction | 261 | 0 | 0.0279482 | 0 | 0.0279482 | 0 | 0.0385488 |
| all | bulb_to_stem_volume_ratio | 261 | 0 | 0.0307311 | 0 | 0.0307311 | 0 | 0.045657 |
| all | stem_length_mm | 261 | 7.79787 | 7.88467 | 5.20939 | 13.0941 | 2.13889 | 29.2773 |
| all | bulb_width_p95_mm | 261 | 0 | 5.11178 | 0 | 5.11178 | 0 | 11.4013 |
| all | bulb_centroid_offset_ratio | 70 | 0.123404 | 0.0502061 | 0.0907712 | 0.140977 | 0.0132052 | 0.210655 |
| small_<100 | stem_fraction | 79 | 1 | 0 | 1 | 1 | 1 | 1 |
| small_<100 | bulb_fraction | 79 | 0 | 0 | 0 | 0 | 0 | 0 |
| small_<100 | bulb_to_stem_volume_ratio | 79 | 0 | 0 | 0 | 0 | 0 | 0 |
| small_<100 | stem_length_mm | 79 | 4.11154 | 1.61773 | 3.40868 | 5.02642 | 2.13889 | 5.70391 |
| small_<100 | bulb_width_p95_mm | 79 | 0 | 0 | 0 | 0 | 0 | 0 |
| small_<100 | bulb_centroid_offset_ratio | 0 |  |  |  |  |  |  |
| medium_100_1000 | stem_fraction | 111 | 0.940994 | 0.115393 | 0.862567 | 0.97796 | 0.692105 | 1 |
| medium_100_1000 | bulb_fraction | 111 | 0 | 0 | 0 | 0 | 0 | 0 |
| medium_100_1000 | bulb_to_stem_volume_ratio | 111 | 0 | 0 | 0 | 0 | 0 | 0 |
| medium_100_1000 | stem_length_mm | 111 | 8.05852 | 2.45388 | 6.92898 | 9.38286 | 5.35578 | 12.0466 |
| medium_100_1000 | bulb_width_p95_mm | 111 | 0 | 0 | 0 | 0 | 0 | 0 |
| medium_100_1000 | bulb_centroid_offset_ratio | 0 |  |  |  |  |  |  |
| large_>=1000 | stem_fraction | 71 | 0.899367 | 0.0180011 | 0.889692 | 0.907693 | 0.805871 | 0.917454 |
| large_>=1000 | bulb_fraction | 71 | 0.0329245 | 0.00522067 | 0.0297601 | 0.0349808 | 0 | 0.0385488 |
| large_>=1000 | bulb_to_stem_volume_ratio | 71 | 0.0363766 | 0.0057977 | 0.0333836 | 0.0391813 | 0 | 0.045657 |
| large_>=1000 | stem_length_mm | 71 | 16.4163 | 4.39331 | 14.931 | 19.3243 | 12.4947 | 29.2773 |
| large_>=1000 | bulb_width_p95_mm | 71 | 6.33392 | 1.97979 | 5.57928 | 7.55908 | 0 | 11.4013 |
| large_>=1000 | bulb_centroid_offset_ratio | 70 | 0.123404 | 0.0502061 | 0.0907712 | 0.140977 | 0.0132052 | 0.210655 |

## Flag Counts

| flag | case_count |
| --- | --- |
| ENGINEERING_ARTIFACT:derived_compartment_mismatch | 160 |
| NO_ISSUE_DETECTED | 101 |
| PLAUSIBILITY_CONCERN:large_tumor_not_bulb_dominant | 71 |
| PLAUSIBILITY_CONCERN:no_voxel_transition_overlap | 27 |
| ENGINEERING_ARTIFACT:multiple_components | 9 |

## Interpretation

- Synthetic stem and bulb compartments are derivable locally because the generator encodes a known local canal axis and CPA side.
- 191/261 standalone cohort masks have no generated CPA bulb by design in the single-mask generator mapping, because `bulb_radius_init` is zero until high maturity.
- 70/71 large masks have a generated CPA bulb, but the derived median large-case bulb fraction is only 0.033 while median stem fraction is 0.899.
- This large-case stem dominance is a synthetic plausibility concern and likely reflects the standalone single-timepoint initialization/calibration regime. It is not a proven biological invalidity claim without real compartment labels.
- The no-bulb small/medium regime is an engineering design choice in the standalone cohort generator, not a proven biological statement.
- Cases with multiple connected components or derived compartment mismatch are flagged as engineering artifacts for review.
- Real canal/CPA plausibility remains blocked by missing real masks and landmarks.

## Generator Tuning Decision

No generator tuning is justified from this pass. The local result identifies synthetic compartment regimes and possible artifacts, but real anatomical validation requires source masks and porus/fundus or canal-axis annotations.
