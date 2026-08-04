# Synthetic Compartment Audit V2

Generated: 2026-07-18T13:48:46.202484+00:00

## Scope

This v2 audit recomputes synthetic canal/stem and CPA/bulb metrics for the authoritative 261 saved masks using the recovered historical generator path:

`rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py`

The current checked-in generator was not used to reconstruct historical compartment geometry for this cohort.

## Recovered Generator Preflight

All 10 representative provenance cases passed Dice >= 0.999 before full analysis.

| case_id | dice | volume_difference_voxels | centroid_difference_vox | bbox_iou | target_volume_mm3 | final_linear_scale_vox | seed | rotation_z_deg | rotation_y_deg | rotation_x_deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 439_0_0 | 1 | 0 | 0 | 1 | 7.75 | 3.94278 | 3290377460 | 126 | 178 | 168 |
| 664_0_0 | 1 | 0 | 0 | 1 | 103.5 | 9.35438 | 1147440901 | 66 | 172 | 29 |
| 611_2_466 | 1 | 0 | 0 | 1 | 770.875 | 12.8879 | 2245982545 | 71 | 154 | 135 |
| 527_3_625 | 1 | 0 | 0 | 1 | 2835.25 | 20.2262 | 2658358932 | 9 | 55 | 45 |
| 126_5_1607 | 1 | 0 | 0 | 1 | 3320.38 | 20.9693 | 2386771635 | 108 | 122 | 1 |
| 527_2_591 | 1 | 0 | 0 | 1 | 2425.75 | 18.8859 | 1277144720 | 113 | 141 | 64 |
| 577_1_239 | 1 | 0 | 0 | 1 | 1546.25 | 16.7966 | 512279341 | 88 | 102 | 105 |
| 579_1_198 | 1 | 0 | 0 | 1 | 1613.25 | 17.0358 | 1201837816 | 64 | 45 | 116 |
| 560_1_146 | 1 | 0 | 0 | 1 | 3096.88 | 21.1724 | 3546535270 | 159 | 95 | 160 |
| 690_0_0 | 1 | 0 | 0 | 1 | 1834.88 | 18.0701 | 1098450466 | 85 | 20 | 175 |

## Cohort Summary

- Cases analyzed: 261
- Zero-bulb saved masks by recovered generator mapping: 135/261
- Small median bulb fraction: 0
- Medium median bulb fraction: 0
- Large median bulb fraction: 0.0384172
- Multi-component masks: 9/261
- Cases with >1% unmatched compartment voxels: 0/261

## Key Distributions

| volume_bin | metric | n | median | iqr | p25 | p75 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | stem_fraction | 261 | 1 | 0.033105 | 0.966895 | 1 | 0.950213 | 1 |
| all | bulb_fraction | 261 | 0 | 0.0329755 | 0 | 0.0329755 | 0 | 0.0491803 |
| all | bulb_to_stem_volume_ratio | 261 | 0 | 0.0340999 | 0 | 0.0340999 | 0 | 0.0517572 |
| all | stem_length_mm | 261 | 8.25509 | 9.31909 | 5.20939 | 14.5285 | 2.13889 | 30.6688 |
| all | bulb_width_p95_mm | 261 | 0 | 5.20495 | 0 | 5.20495 | 0 | 11.2841 |
| all | derived_unmatched_voxel_fraction | 261 | 0 | 0 | 0 | 0 | 0 | 0 |
| small_<100 | stem_fraction | 79 | 1 | 0 | 1 | 1 | 1 | 1 |
| small_<100 | bulb_fraction | 79 | 0 | 0 | 0 | 0 | 0 | 0 |
| small_<100 | bulb_to_stem_volume_ratio | 79 | 0 | 0 | 0 | 0 | 0 | 0 |
| small_<100 | stem_length_mm | 79 | 4.11154 | 1.61773 | 3.40868 | 5.02642 | 2.13889 | 5.70391 |
| small_<100 | bulb_width_p95_mm | 79 | 0 | 0 | 0 | 0 | 0 | 0 |
| small_<100 | derived_unmatched_voxel_fraction | 79 | 0 | 0 | 0 | 0 | 0 | 0 |
| medium_100_1000 | stem_fraction | 111 | 1 | 0.00841377 | 0.991586 | 1 | 0.95045 | 1 |
| medium_100_1000 | bulb_fraction | 111 | 0 | 0.00841377 | 0 | 0.00841377 | 0 | 0.0484234 |
| medium_100_1000 | bulb_to_stem_volume_ratio | 111 | 0 | 0.00848518 | 0 | 0.00848518 | 0 | 0.0509479 |
| medium_100_1000 | stem_length_mm | 111 | 8.62346 | 3.39605 | 7.06266 | 10.4587 | 5.35578 | 14.0545 |
| medium_100_1000 | bulb_width_p95_mm | 111 | 0 | 1.96652 | 0 | 1.96652 | 0 | 5.29558 |
| medium_100_1000 | derived_unmatched_voxel_fraction | 111 | 0 | 0 | 0 | 0 | 0 | 0 |
| large_>=1000 | stem_fraction | 71 | 0.960954 | 0.00847813 | 0.956961 | 0.965439 | 0.950213 | 0.972178 |
| large_>=1000 | bulb_fraction | 71 | 0.0384172 | 0.0082458 | 0.0342764 | 0.0425222 | 0.0278219 | 0.0491803 |
| large_>=1000 | bulb_to_stem_volume_ratio | 71 | 0.0399521 | 0.00894614 | 0.0355077 | 0.0444539 | 0.0286182 | 0.0517572 |
| large_>=1000 | stem_length_mm | 71 | 17.0859 | 4.48337 | 15.5976 | 20.081 | 14.112 | 30.6688 |
| large_>=1000 | bulb_width_p95_mm | 71 | 6.34788 | 1.82873 | 5.62544 | 7.45417 | 4.83724 | 11.2841 |
| large_>=1000 | derived_unmatched_voxel_fraction | 71 | 0 | 0 | 0 | 0 | 0 | 0 |

## Prior Finding Comparison

| prior_finding | corrected_result | classification |
| --- | --- | --- |
| 191/261 zero-bulb result | 135/261 zero-bulb | REJECTED |
| 160/261 mismatch flags | 0/261 cases with >1% unmatched voxels | ARTIFACT_OF_WRONG_GENERATOR |
| 9/261 multi-component masks | 9/261 multi-component masks | CONFIRMED |

## Flag Counts

| flag | case_count |
| --- | --- |
| NO_ISSUE_DETECTED | 137 |
| PLAUSIBILITY_CONCERN:no_voxel_transition_overlap | 91 |
| PLAUSIBILITY_CONCERN:large_tumor_not_bulb_dominant | 71 |
| ENGINEERING_ARTIFACT:multiple_components | 9 |

## Visual Verification

Fresh stratified cases reviewed: 12.

| classification | case_count |
| --- | --- |
| MAPPING_CONFIRMED | 12 |

| case_id | volume_bin | bulb_fraction | stem_fraction | unassigned_fraction | classification | overlay_path |
| --- | --- | --- | --- | --- | --- | --- |
| 95_1_182 | small_<100 | 0 | 1 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/95_1_182_v2_overlay.png |
| 57_1_208 | small_<100 | 0 | 1 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/57_1_208_v2_overlay.png |
| 439_0_0 | small_<100 | 0 | 1 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/439_0_0_v2_overlay.png |
| 235_0_0 | small_<100 | 0 | 1 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/235_0_0_v2_overlay.png |
| 664_1_175 | medium_100_1000 | 0 | 1 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/664_1_175_v2_overlay.png |
| 478_1_378 | medium_100_1000 | 0.00309598 | 0.996904 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/478_1_378_v2_overlay.png |
| 537_1_110 | medium_100_1000 | 0.00852878 | 0.991471 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/537_1_110_v2_overlay.png |
| 126_4_1231 | medium_100_1000 | 0.0327485 | 0.967251 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/126_4_1231_v2_overlay.png |
| 132_2_232 | large_>=1000 | 0.0318883 | 0.967885 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/132_2_232_v2_overlay.png |
| 647_2_176 | large_>=1000 | 0.0384172 | 0.961583 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/647_2_176_v2_overlay.png |
| 606_0_0 | large_>=1000 | 0.043364 | 0.956636 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/606_0_0_v2_overlay.png |
| 74_2_1367 | large_>=1000 | 0.03125 | 0.96875 | 0 | MAPPING_CONFIRMED | analysis/anatomical_compartment_validation_v2/overlays/74_2_1367_v2_overlay.png |

## Scientific Interpretation

1. Authoritative saved masks lacking a CPA bulb: 135/261 under the recovered generator mapping.
2. Stem dominance: the cohort remains stem-dominant by derived fractions; this is a synthetic design signal, not a clinical invalidity claim.
3. Bulb fraction and tumor size: median bulb fraction increases from small (0) to medium (0) to large (0.0384172), but the absolute bulb fraction remains modest.
4. Gating/plateaus: the recovered generator uses a maturity-dependent bulb radius, so zero or near-zero bulb behavior is expected in small cases and low-volume regimes. This is a generator design feature.
5. Small tumors: mostly intracanalicular/stem-dominant small tumors are plausible as a synthetic prior, but the exact distribution requires real compartment validation.
6. Large tumors: large tumors develop nonzero CPA components, but the recovered mapping still suggests many remain stem-dominant.
7. Prior concerns: the old current-generator mismatch count is an artifact of the wrong reconstruction path. Zero-bulb and multi-component findings are evaluated in the prior-comparison table above.
8. Generator tuning: not justified now. Synthetic-side compartment metrics are more trustworthy after provenance recovery, but real source masks and anatomical annotations are still required before tuning.

## Decision

Synthetic compartment metrics from this v2 audit are suitable for synthetic-only plausibility tracking and future real-vs-synthetic comparison once real annotations exist. They should not be used alone to declare the generator anatomically valid or invalid.
