# Surface Resolution Validation Report

Generated: 2026-07-18T12:32:51.222403+00:00

## Scope

This local-only analysis tests whether surface-sensitive real-vs-synthetic morphology gaps could be driven by voxel spacing/resolution. It does not modify generator geometry, does not overwrite `analysis/synthetic_features_v2/`, and does not re-extract real masks because source real segmentations are not available locally.

## Inputs

- Synthetic features: `analysis/synthetic_features_v2/synthetic_features_v2.csv` (261 rows)
- Synthetic masks: `rivanna_pull/analysis/synthetic_lollipop_v1/masks`
- Real features: `rivanna_pull/analysis/real_tumor_features_v1/real_tumor_features_usable_train.csv` (291 rows)
- Selected controlled subset: 30 cases
- Selected bins: {'small_<100': 10, 'medium_100_1000': 10, 'large_>=1000': 10}
- Git commit: `b7db1cc55f708ed71f4a9b11da1aef5cc27e3f5e`
- Extractor hash: `681fc273340f1a5f9e696df341a270feea8ab6ab115402bbb625717144e72e93`

## Metric Definitions

See `analysis/surface_resolution_validation/METRIC_DEFINITION_AUDIT.md`. In short, physical spacing is passed correctly to marching cubes and formulas use physical volume, but label-mask surface estimates are still resolution-sensitive because the discretized boundary changes with voxel size.

## Validation Checks

- Missing selected source masks: 0
- Empty resampled masks: 0
- Maximum resampled volume error fraction: 0.0000
- Source masks were read only; resampled masks were temporary files created under the system temp directory for extraction.
- Case IDs were preserved in output rows with condition-specific extraction IDs only inside temporary files.

## Validation Commands

- `.venv/bin/python analysis/surface_resolution_validation/run_surface_resolution_validation.py`: passed.
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/surface_resolution_validation/run_surface_resolution_validation.py`: passed.
- `.venv/bin/python -m pytest -m "fast and not slow" -v`: passed, 39 passed, 9 deselected, 14 dependency deprecation warnings.
- `git diff --check`: passed.

## Resolution Sensitivity Summary

| condition | metric | n | median_percent_change | median_absolute_percent_change | iqr_percent_change | sensitivity_class |
| --- | --- | --- | --- | --- | --- | --- |
| real_like_0p5_iso | surface_area_mm2 | 30 | 17.9606 | 17.9606 | 3.95135 | MODERATELY_RESOLUTION_SENSITIVE |
| real_like_0p5_iso | sphericity | 30 | -15.2259 | 15.2259 | 2.78207 | MODERATELY_RESOLUTION_SENSITIVE |
| real_like_0p5_iso | compactness | 30 | 64.1391 | 64.1391 | 16.8424 | HIGHLY_RESOLUTION_SENSITIVE |
| real_like_0p5_iso | surface_to_volume_ratio | 30 | 17.9606 | 17.9606 | 3.95135 | MODERATELY_RESOLUTION_SENSITIVE |
| anisotropic_0p5_0p5_1p0 | surface_area_mm2 | 30 | 13.9258 | 13.9258 | 3.03913 | MODERATELY_RESOLUTION_SENSITIVE |
| anisotropic_0p5_0p5_1p0 | sphericity | 30 | -12.2235 | 12.2235 | 2.30811 | MODERATELY_RESOLUTION_SENSITIVE |
| anisotropic_0p5_0p5_1p0 | compactness | 30 | 47.8655 | 47.8655 | 12.0078 | HIGHLY_RESOLUTION_SENSITIVE |
| anisotropic_0p5_0p5_1p0 | surface_to_volume_ratio | 30 | 13.9258 | 13.9258 | 3.03913 | MODERATELY_RESOLUTION_SENSITIVE |

## Normalized Real-vs-Synthetic Surface Comparison

The normalized comparison uses option A: synthetic masks were resampled to `0.5 x 0.5 x 0.5 mm`, matching the spacing reported by all local real feature rows. This is cleaner than resampling both representations because source real masks are unavailable locally.

| metric | matched_case_count | real_median | native_synthetic_median | normalized_synthetic_median | original_synthetic_over_real_median_ratio | normalized_synthetic_over_real_median_ratio | original_abs_log_gap | normalized_abs_log_gap | gap_after_normalization | ks_native_vs_real | ks_normalized_vs_real |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surface_area_mm2 | 30 | 282.048 | 217.726 | 255.852 | 0.771948 | 0.90712 | 0.258838 | 0.0974801 | DISAPPEARS | 0.2 | 0.166667 |
| sphericity | 30 | 0.728036 | 0.922109 | 0.779928 | 1.26657 | 1.07128 | 0.236313 | 0.0688514 | DISAPPEARS | 0.9 | 0.366667 |
| compactness | 30 | 293.098 | 144.247 | 238.39 | 0.492145 | 0.813347 | 0.708983 | 0.206598 | SUBSTANTIALLY_SHRINKS | 0.9 | 0.366667 |
| surface_to_volume_ratio | 30 | 1.04635 | 0.806336 | 0.947516 | 0.770614 | 0.90554 | 0.260567 | 0.0992236 | DISAPPEARS | 0.233333 | 0.2 |

## Interpretation

- `sphericity`: native synthetic/real median ratio was 1.267; after synthetic 0.5 mm normalization it was 1.071. Gap status: `DISAPPEARS`.
- `compactness`: native synthetic/real median ratio was 0.492; after synthetic 0.5 mm normalization it was 0.813. Gap status: `SUBSTANTIALLY_SHRINKS`.
- Because real source masks cannot be resampled or re-extracted locally, this supports resolution as a substantial confounder but does not prove it is the only cause.

## Metric Suitability

| metric | recommended_use |
| --- | --- |
| surface_area_mm2 | COMPARE_ONLY_AFTER_RESOLUTION_NORMALIZATION |
| sphericity | COMPARE_ONLY_AFTER_RESOLUTION_NORMALIZATION |
| compactness | COMPARE_ONLY_AFTER_RESOLUTION_NORMALIZATION |
| surface_to_volume_ratio | COMPARE_ONLY_AFTER_RESOLUTION_NORMALIZATION |

## Morphology Gap Reclassification

Previously identified surface gaps should remain `EXTRACTION_OR_DATA_LIMITATION` unless reviewed with common-resolution real masks. No generator tuning is justified from surface metrics in the current local evidence.

## Remaining Blockers

- Real source segmentation masks are not locally available, so real features cannot be re-extracted under canonical spacing.
- Surface metrics are mesh/discretization sensitive even with correct physical units.
- Anatomical canal/CPA validation is still required before morphology tuning.

## Next Tasks

1. Add extractor/generator provenance fields for spacing, surface method, and schema version to future outputs.
2. Obtain or stage local real source masks for common-resolution re-extraction without relying on stale pulled features.
3. Run anatomical canal/CPA compartment validation before any generator morphology tuning.
