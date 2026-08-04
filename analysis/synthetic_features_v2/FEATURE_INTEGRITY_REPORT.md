# Synthetic Features V2 Integrity Report

Status: PASS

## Authoritative Inputs

- Manifest: `rivanna_pull/analysis/synthetic_lollipop_v1/manifests/synthetic_lollipop_manifest.csv`
- Mask root: `rivanna_pull/analysis/synthetic_lollipop_v1/masks`
- Extractor: `scripts/extract_real_tumor_features.py`
- Legacy pulled synthetic feature CSV/JSON artifacts were not overwritten.

## Preflight Summary

| Check | Value |
| --- | ---: |
| manifest_rows | 261 |
| manifest_unique_case_ids | 261 |
| duplicate_case_ids | 0 |
| mask_files_found | 261 |
| manifest_without_mask | 0 |
| mask_without_manifest | 0 |
| preflight_ok | 261 |
| readable | 261 |
| spacing_valid | 261 |
| non_empty | 261 |
| volume_matches_manifest | 261 |
| one_to_one_case_matching | True |

## Feature Integrity Summary

| Check | Value |
| --- | ---: |
| feature_rows | 261 |
| feature_unique_case_ids | 261 |
| duplicate_feature_case_ids | 0 |
| failed_extractions | 0 |
| outer_join_left_only | 0 |
| outer_join_right_only | 0 |
| nan_values_in_numeric_columns | 0 |
| inf_values_in_numeric_columns | 0 |
| volume_max_abs_diff_mm3 | 0.0 |
| volume_mismatches_gt_1e_minus_6 | 0 |
| multiple_component_rows | 9 |
| surface_area_failed_rows | 0 |
| principal_axis_failed_rows | 0 |
| nonpositive_volume | 0 |
| nonpositive_spacing | 0 |
| component_count_lt_one | 0 |
| largest_fraction_out_of_range | 0 |
| negative_surface_area | 0 |
| negative_axis_length | 0 |

## Interpretation

- Regenerated features are authoritative for the current local synthetic masks if status is PASS.
- This report validates engineering integrity and feature/mask consistency; it does not establish morphology realism.
- Multiple-component rows are reported as integrity metadata. They are not automatically material failures because the current extractor records largest-component features while preserving full-mask counts.

## Material Failures

- None.
