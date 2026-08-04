# Authoritative Real-Versus-Synthetic Morphology Validation V2

## Scope

This analysis uses regenerated synthetic features from `analysis/synthetic_features_v2/` and the best available local real feature table. It does not use stale pulled synthetic feature CSV/JSON artifacts, does not SSH, and does not alter generator parameters.

Real source segmentation masks are not available locally, so real-feature reproducibility remains a remote-data blocker. Synthetic feature integrity passed locally and is authoritative for the current local masks.

## Input Verification

| Check | Value |
| --- | ---: |
| real_rows | 291 |
| synthetic_rows | 261 |
| matched_case_ids | 261 |
| real_unique_case_ids | 291 |
| synthetic_unique_case_ids | 261 |
| missing_metrics_in_real | [] |
| missing_metrics_in_synthetic | [] |
| equivalent_metric_schema | True |
| synthetic_schema_version | synthetic_features_v2_current_extractor_2026-07-18 |
| synthetic_git_commit_hash | b7db1cc55f708ed71f4a9b11da1aef5cc27e3f5e |

Feature definitions are equivalent by schema for all requested metrics. They come from `scripts/extract_real_tumor_features.py`: spacing-aware volume, largest-component surface metrics, whole-mask PCA/extent features, and bounding-box features.

Volume bins are fixed local engineering bins based on real matched volume: `small_<100`, `medium_100_1000`, and `large_>=1000` mm3. These are not clinical staging thresholds.

## Overall Distribution Comparison

| Metric | Real n | Synthetic n | Real Median | Synthetic Median | Ratio Syn/Real | Real IQR | Synthetic IQR | Cliff's Delta | KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| volume_mm3 | 261 | 261 | 270.375 | 267.000 | 0.988 | 1041.750 | 1014.000 | 0.001 | 0.027 |
| equivalent_sphere_diameter_mm | 261 | 261 | 8.023 | 7.989 | 0.996 | 7.573 | 7.494 | 0.001 | 0.027 |
| elongation | 261 | 261 | 1.377 | 1.216 | 0.883 | 0.523 | 0.370 | -0.328 | 0.322 |
| secondary_axis_ratio | 261 | 261 | 0.726 | 0.822 | 1.133 | 0.251 | 0.215 | 0.328 | 0.322 |
| aspect_ratio_major_to_minor2 | 261 | 261 | 1.730 | 1.252 | 0.724 | 0.762 | 0.334 | -0.726 | 0.598 |
| sphericity | 261 | 261 | 0.727 | 0.921 | 1.267 | 0.101 | 0.098 | 0.940 | 0.889 |
| compactness | 261 | 261 | 294.002 | 144.715 | 0.492 | 122.787 | 50.573 | -0.940 | 0.889 |
| surface_area_mm2 | 261 | 261 | 277.435 | 218.955 | 0.789 | 608.072 | 508.707 | -0.110 | 0.107 |
| surface_to_volume_ratio | 261 | 261 | 1.046 | 0.808 | 0.773 | 0.772 | 0.660 | -0.248 | 0.207 |
| bbox_fill_fraction | 261 | 261 | 0.375 | 0.481 | 1.285 | 0.100 | 0.114 | 0.698 | 0.567 |
| principal_axis_length_major_mm | 261 | 261 | 12.424 | 8.561 | 0.689 | 11.759 | 11.835 | -0.231 | 0.207 |
| principal_axis_length_minor1_mm | 261 | 261 | 7.795 | 7.071 | 0.907 | 8.847 | 6.240 | -0.184 | 0.180 |
| principal_axis_length_minor2_mm | 261 | 261 | 6.294 | 6.569 | 1.044 | 7.207 | 6.475 | -0.030 | 0.092 |
| connected_component_count | 261 | 261 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.034 | 0.034 |
| largest_component_fraction | 261 | 261 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | -0.034 | 0.034 |

## Ranked Morphology Gaps

| Rank | Feature | Classification | Direction | Ratio Syn/Real | Cliff's Delta | Volume Strata Same Direction | Interpretation | Tuning Justified |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | compactness | EXTRACTION_OR_DATA_LIMITATION | synthetic_lower | 0.492 | -0.940 | 3 | surface-derived compactness/sphericity differs; spacing and source-mask availability limit interpretation | no |
| 2 | aspect_ratio_major_to_minor2 | POSSIBLE_GENERATOR_GAP | synthetic_lower | 0.724 | -0.726 | 2 | dominant-to-smallest axis disparity is lower in synthetic | no |
| 3 | sphericity | EXTRACTION_OR_DATA_LIMITATION | synthetic_higher | 1.267 | 0.940 | 3 | surface-derived compactness/sphericity differs; spacing and source-mask availability limit interpretation | no |
| 4 | bbox_fill_fraction | POSSIBLE_GENERATOR_GAP | synthetic_higher | 1.285 | 0.698 | 3 | synthetic masks fill their axis-aligned bounding boxes more densely | no |
| 5 | principal_axis_length_major_mm | INSUFFICIENT_EVIDENCE | synthetic_lower | 0.689 | -0.231 | 3 | synthetic dominant extent is shorter at matched volume | no |
| 6 | surface_to_volume_ratio | EXTRACTION_OR_DATA_LIMITATION | synthetic_lower | 0.773 | -0.248 | 3 | surface-derived compactness/sphericity differs; spacing and source-mask availability limit interpretation | no |
| 7 | secondary_axis_ratio | INSUFFICIENT_EVIDENCE | synthetic_higher | 1.133 | 0.328 | 2 | distribution differs in current whole-mask features | no |
| 8 | elongation | INSUFFICIENT_EVIDENCE | synthetic_lower | 0.883 | -0.328 | 2 | whole-mask canonical major/intermediate axis elongation is lower in synthetic | no |
| 9 | surface_area_mm2 | EXTRACTION_OR_DATA_LIMITATION | synthetic_lower | 0.789 | -0.110 | 3 | surface-derived compactness/sphericity differs; spacing and source-mask availability limit interpretation | no |

## Too-Elongated Concern

Verdict: REJECTED_FOR_REGENERATED_FEATURES.

- Median elongation ratio synthetic/real: 0.883.
- Median aspect-ratio major/minor2 ratio synthetic/real: 0.724.
- Median major-axis length ratio synthetic/real: 0.689.

Using regenerated authoritative synthetic features, the prior concern that synthetic masks are too elongated is not supported. The current whole-mask features instead show lower canonical elongation and shorter major-axis extent at matched case IDs. This should not trigger generator tuning because surface/spacing and whole-mask topology limitations still need review.

## Volume-Stratified Comparison

| Stratum | Cases | Metric | Real Median | Synthetic Median | Ratio Syn/Real | Cliff's Delta |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| small_<100 | 80 | compactness | 238.475 | 124.526 | 0.522 | -0.945 |
| small_<100 | 80 | principal_axis_length_major_mm | 5.991 | 4.126 | 0.689 | -0.677 |
| small_<100 | 80 | aspect_ratio_major_to_minor2 | 1.689 | 1.172 | 0.694 | -0.906 |
| small_<100 | 80 | bbox_fill_fraction | 0.404 | 0.519 | 1.284 | 0.590 |
| small_<100 | 80 | sphericity | 0.780 | 0.968 | 1.242 | 0.945 |
| small_<100 | 80 | surface_to_volume_ratio | 1.717 | 1.389 | 0.809 | -0.577 |
| small_<100 | 80 | surface_area_mm2 | 81.366 | 66.315 | 0.815 | -0.274 |
| small_<100 | 80 | elongation | 1.360 | 1.124 | 0.827 | -0.747 |
| small_<100 | 80 | secondary_axis_ratio | 0.736 | 0.890 | 1.209 | 0.747 |
| medium_100_1000 | 110 | compactness | 308.108 | 145.487 | 0.472 | -0.999 |
| medium_100_1000 | 110 | principal_axis_length_major_mm | 13.205 | 8.888 | 0.673 | -0.554 |
| medium_100_1000 | 110 | aspect_ratio_major_to_minor2 | 2.102 | 1.238 | 0.589 | -0.943 |
| medium_100_1000 | 110 | bbox_fill_fraction | 0.383 | 0.501 | 1.309 | 0.768 |
| medium_100_1000 | 110 | sphericity | 0.716 | 0.919 | 1.284 | 0.999 |
| medium_100_1000 | 110 | surface_to_volume_ratio | 1.027 | 0.788 | 0.768 | -0.644 |
| medium_100_1000 | 110 | surface_area_mm2 | 284.713 | 231.881 | 0.814 | -0.291 |
| medium_100_1000 | 110 | elongation | 1.665 | 1.217 | 0.731 | -0.740 |
| medium_100_1000 | 110 | secondary_axis_ratio | 0.601 | 0.821 | 1.367 | 0.740 |
| large_>=1000 | 71 | compactness | 332.146 | 187.506 | 0.565 | -0.960 |
| large_>=1000 | 71 | principal_axis_length_major_mm | 21.573 | 20.042 | 0.929 | -0.151 |
| large_>=1000 | 71 | aspect_ratio_major_to_minor2 | 1.500 | 1.536 | 1.024 | 0.165 |
| large_>=1000 | 71 | bbox_fill_fraction | 0.336 | 0.444 | 1.320 | 0.837 |
| large_>=1000 | 71 | sphericity | 0.698 | 0.845 | 1.210 | 0.960 |
| large_>=1000 | 71 | surface_to_volume_ratio | 0.558 | 0.474 | 0.848 | -0.526 |
| large_>=1000 | 71 | surface_area_mm2 | 1028.810 | 842.148 | 0.819 | -0.289 |
| large_>=1000 | 71 | elongation | 1.237 | 1.530 | 1.237 | 0.860 |
| large_>=1000 | 71 | secondary_axis_ratio | 0.808 | 0.654 | 0.809 | -0.860 |

## Scientific Interpretation Limits

- No `HIGH_CONFIDENCE_GENERATOR_GAP` is assigned from this local-only pass because real masks cannot be re-extracted locally and anatomical canal/CPA labels are unavailable.
- Surface-derived gaps are classified as `EXTRACTION_OR_DATA_LIMITATION` because real and synthetic voxel spacing differ and surface metrics are resolution-sensitive.
- PCA/bounding-box gaps are `POSSIBLE_GENERATOR_GAP` only where effects are consistent, but they still need anatomical review before tuning.
- The matched benchmark covers 261 case IDs; it is not a full clinical population validation.

## Generated Outputs

- `matched_distribution_comparison.csv`
- `volume_stratified_comparison.csv`
- `ranked_morphology_gaps.csv`
- `plots/*.png`

Plots:
- `analysis/real_vs_synthetic_v2/plots/compactness_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/principal_axis_length_major_mm_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/aspect_ratio_major_to_minor2_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/bbox_fill_fraction_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/sphericity_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/surface_to_volume_ratio_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/surface_area_mm2_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/elongation_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/secondary_axis_ratio_boxplot.png`
- `analysis/real_vs_synthetic_v2/plots/volume_match_scatter.png`
- `analysis/real_vs_synthetic_v2/plots/volume_stratified_ratio_heatmap.png`

## Recommended Follow-Up

1. Add extractor provenance/version fields to all future generated feature tables.
2. Review surface metric comparability under real 0.5 mm spacing versus synthetic 1.0 mm spacing before treating compactness/sphericity as generator gaps.
3. Perform anatomical canal/CPA compartment validation before any generator morphology tuning.
