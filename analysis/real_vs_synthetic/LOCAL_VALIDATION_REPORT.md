# Local Real-Versus-Synthetic Morphology Validation

Task ID: MORPH-001

## Scope

This audit uses only files present in the local GrowthNet repository. It does not SSH, does not query Rivanna, and does not depend on remote source masks. The only source data used are pulled CSV/JSON feature artifacts and local synthetic NIfTI masks already present under `rivanna_pull/analysis/synthetic_lollipop_v1/masks/`.

## Local Data Inventory

| Item | Count | Notes |
| --- | ---: | --- |
| real_features_rows | 291 | Pulled feature rows available locally. |
| synthetic_features_rows | 261 | Pulled feature rows available locally. |
| matched_case_ids | 261 | Intersection of real and synthetic feature case_id values. |
| real_source_seg_paths_existing_locally | 0 | CSV seg_path values point to original segmentation masks. |
| synthetic_feature_seg_paths_existing_locally | 0 | Pulled CSV seg_path values before local remapping. |
| synthetic_manifest_local_masks_existing | 261 | Local synthetic masks after remapping to repo-local paths. |
| real_single_component_targets | 261 | Targets used to generate the synthetic benchmark. |

## Case-ID Semantics

- `case_id` values use the pattern `patient_visit_day` in the available benchmark tables, for example `57_1_208`.
- The matched benchmark contains one synthetic mask per usable real target case ID.
- The real feature summary reports 291 successful real masks, but the matched synthetic benchmark uses 261 single-component targets. The 30-case difference should not be treated as synthetic coverage of the full real segmentation population.
- Local synthetic masks exist in the repo, but the `seg_path` fields inside the pulled synthetic CSV/manifest retain their original `/sfs/...` provenance paths. `local_synthetic_manifest.csv` remaps those to repo-local paths.

## Equivalent Feature Extraction

- Real and synthetic pulled feature CSVs have the same morphology schema and align one-to-one on 261 case IDs.
- The feature definitions come from `scripts/extract_real_tumor_features.py`: largest-component surface metrics, spacing-aware volume/lengths, whole-mask PCA axis features, and bounding-box fill.
- Synthetic source masks are locally available, so they can be re-extracted locally with the command listed below.
- Real source segmentation masks are not locally available; independent real re-extraction is `BLOCKED_REMOTE_DATA`.

## Headline Findings

### Historical Pulled Feature Tables

- Matched volume targeting is close: median synthetic/real volume ratio is 0.969.
- Synthetic masks are substantially more elongated: median elongation ratio is 1.790.
- Synthetic masks occupy much less of their bounding boxes: median bbox-fill ratio is 0.455.
- Synthetic masks are less spherical/compact by whole-mask surface metrics: median sphericity ratio is 0.844.
- Synthetic major-axis lengths are longer at matched volume: median major-axis ratio is 1.319.

These historical benchmark findings are preserved for provenance, but they are not the strongest local evidence because the local synthetic masks re-extract differently with the current checked-out extractor.

### Current Local Synthetic Re-Extraction

- Current local synthetic re-extraction volume ratio: 0.988.
- Current local synthetic re-extraction elongation ratio: 0.883.
- Current local synthetic re-extraction bbox-fill ratio: 1.285.
- Current local synthetic re-extraction sphericity ratio: 1.267.
- Current local re-extraction found 9 multi-component synthetic masks; the pulled synthetic summary reports 0. Treat this as extractor/artifact drift requiring follow-up before generator tuning.

These are local benchmark findings for the matched 261 single-component target set. They are not population-level clinical conclusions.

## Ranked Morphology Gaps

Historical pulled-feature ranking:

| Rank | Metric | Median Ratio Syn/Real | Median Diff | KS | Cliff's Delta | Interpretation |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | bbox_fill_fraction | 0.455 | -0.204 | 0.851 | -0.884 | Lower synthetic fill suggests more empty bounding-box space from lollipop/stem geometry. |
| 2 | elongation | 1.790 | 1.088 | 0.923 | 0.920 | Synthetic masks are more elongated when ratio > 1. |
| 3 | aspect_ratio_major_to_minor2 | 1.577 | 0.999 | 0.820 | 0.765 | Higher values indicate larger major/minor2 disparity. |
| 4 | principal_axis_length_major_mm | 1.319 | 3.965 | 0.249 | 0.284 | Higher synthetic major axis indicates longer dominant morphology. |
| 5 | max_diameter_bbox_mm | 1.280 | 3.500 | 0.245 | 0.287 | Higher synthetic diameter indicates spatial extent larger than real at matched volume. |
| 6 | sphericity | 0.844 | -0.114 | 0.605 | -0.566 | Lower synthetic sphericity suggests less compact or more irregular shape. |
| 7 | surface_to_volume_ratio | 1.178 | 0.186 | 0.146 | 0.125 | Higher synthetic ratio indicates more surface per unit volume. |

Current local synthetic re-extraction ranking:

| Rank | Metric | Median Ratio Syn/Real | Median Diff | KS | Cliff's Delta | Interpretation |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | principal_axis_length_major_mm | 0.689 | -3.863 | 0.207 | -0.231 | Higher synthetic major axis indicates longer dominant morphology. |
| 2 | max_diameter_bbox_mm | 0.720 | -3.500 | 0.192 | -0.215 | Higher synthetic diameter indicates spatial extent larger than real at matched volume. |
| 3 | aspect_ratio_major_to_minor2 | 0.724 | -0.478 | 0.598 | -0.726 | Higher values indicate larger major/minor2 disparity. |
| 4 | surface_to_volume_ratio | 0.773 | -0.237 | 0.207 | -0.248 | Higher synthetic ratio indicates more surface per unit volume. |
| 5 | bbox_fill_fraction | 1.285 | 0.107 | 0.567 | 0.698 | Lower synthetic fill suggests more empty bounding-box space from lollipop/stem geometry. |
| 6 | sphericity | 1.267 | 0.194 | 0.889 | 0.940 | Lower synthetic sphericity suggests less compact or more irregular shape. |
| 7 | elongation | 0.883 | -0.161 | 0.322 | -0.328 | Synthetic masks are more elongated when ratio > 1. |

## Synthetic Feature Drift

The pulled synthetic CSV does not reproduce from the local masks with the current extractor. This can come from source-code drift, generated mask drift, or provenance mismatch. It must be resolved before making scientific claims from synthetic morphology calibration.

| Metric | Pulled Median | Local Re-Extracted Median | Median Abs Diff | Max Abs Diff | Changed Rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| max_diameter_bbox_mm | 16.000 | 9.000 | 7.000 | 24.000 | 261 |
| principal_axis_length_major_mm | 16.388 | 8.561 | 7.586 | 21.005 | 261 |
| elongation | 2.466 | 1.216 | 1.205 | 1.848 | 261 |
| aspect_ratio_major_to_minor2 | 2.729 | 1.252 | 1.424 | 2.050 | 261 |
| bbox_fill_fraction | 0.170 | 0.481 | 0.311 | 0.643 | 261 |
| sphericity | 0.614 | 0.921 | 0.258 | 0.357 | 261 |
| surface_to_volume_ratio | 1.232 | 0.808 | 0.318 | 0.627 | 261 |

## Volume-Stratified Summary

| Stratum | Cases | Metric | Median Ratio Syn/Real | Median Diff | Cliff's Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| tiny_<100 | 80 | elongation | 1.869 | 1.182 | 0.951 |
| tiny_<100 | 80 | aspect_ratio_major_to_minor2 | 1.668 | 1.128 | 0.776 |
| tiny_<100 | 80 | bbox_fill_fraction | 0.627 | -0.151 | -0.683 |
| tiny_<100 | 80 | sphericity | 0.999 | -0.001 | -0.135 |
| tiny_<100 | 80 | max_diameter_bbox_mm | 1.167 | 1.000 | 0.603 |
| tiny_<100 | 80 | principal_axis_length_major_mm | 1.231 | 1.385 | 0.611 |
| tiny_<100 | 80 | surface_to_volume_ratio | 0.966 | -0.059 | -0.079 |
| small_100_500 | 86 | elongation | 1.500 | 0.818 | 0.819 |
| small_100_500 | 86 | aspect_ratio_major_to_minor2 | 1.319 | 0.658 | 0.593 |
| small_100_500 | 86 | bbox_fill_fraction | 0.395 | -0.241 | -1.000 |
| small_100_500 | 86 | sphericity | 0.848 | -0.110 | -0.926 |
| small_100_500 | 86 | max_diameter_bbox_mm | 1.455 | 5.000 | 0.652 |
| small_100_500 | 86 | principal_axis_length_major_mm | 1.350 | 4.061 | 0.647 |
| small_100_500 | 86 | surface_to_volume_ratio | 1.170 | 0.188 | 0.521 |
| medium_500_1500 | 48 | elongation | 1.686 | 1.002 | 0.958 |
| medium_500_1500 | 48 | aspect_ratio_major_to_minor2 | 1.437 | 0.828 | 0.783 |
| medium_500_1500 | 48 | bbox_fill_fraction | 0.437 | -0.191 | -1.000 |
| medium_500_1500 | 48 | sphericity | 0.875 | -0.085 | -0.834 |
| medium_500_1500 | 48 | max_diameter_bbox_mm | 1.307 | 5.750 | 0.877 |
| medium_500_1500 | 48 | principal_axis_length_major_mm | 1.356 | 6.741 | 0.882 |
| medium_500_1500 | 48 | surface_to_volume_ratio | 1.135 | 0.095 | 0.477 |
| large_>=1500 | 47 | elongation | 2.036 | 1.248 | 1.000 |
| large_>=1500 | 47 | aspect_ratio_major_to_minor2 | 1.902 | 1.292 | 1.000 |
| large_>=1500 | 47 | bbox_fill_fraction | 0.412 | -0.199 | -1.000 |
| large_>=1500 | 47 | sphericity | 0.861 | -0.096 | -1.000 |
| large_>=1500 | 47 | max_diameter_bbox_mm | 1.488 | 10.500 | 0.867 |
| large_>=1500 | 47 | principal_axis_length_major_mm | 1.512 | 11.804 | 0.887 |
| large_>=1500 | 47 | surface_to_volume_ratio | 1.209 | 0.103 | 0.594 |

## Likely Generator Gaps

1. The historical pulled feature comparison indicates a long-axis/bbox-fill mismatch: synthetic masks are more elongated and occupy less of their bounding boxes at matched volume.
2. The current local re-extraction does not reproduce that same morphology profile, so generator tuning should not proceed from the pulled synthetic CSV alone.
3. The most defensible immediate gap is provenance/reproducibility: resolve why local synthetic masks plus the current extractor differ materially from the pulled synthetic feature table.
4. After provenance is resolved, reassess whether whole-mask PCA/bounding-box gaps are true generator gaps or extraction artifacts of lollipop topology.

These are audit findings, not tuning instructions. They are not evidence that the current anatomy is wrong, because canal/bulb compartment validation is unavailable locally.

## Extraction-Artifacts Versus Scientific Differences

- The strongest gaps involve whole-mask PCA and bounding-box metrics, which are sensitive to a lollipop stem even if the embedded anatomy is plausible.
- Surface and sphericity metrics are resolution-sensitive: real masks use 0.5 mm voxel volume in the pulled table examples, while benchmark synthetic masks use 1.0 mm isotropic voxels. This can inflate discretization differences.
- Principal-axis sign is irrelevant here because all compared metrics are scalar lengths/ratios.
- Without real source masks and anatomical canal/CPA labels, this audit cannot separate biologically meaningful VS morphology from whole-mask feature limitations.

## BLOCKED_REMOTE_DATA

| Dependency | Why Needed | Local Fallback |
| --- | --- | --- |
| Real segmentation NIfTI masks for the 261 matched usable targets | Required to independently re-extract real morphology features and verify extraction equivalence end-to-end. | Pulled real feature CSV and JSON summaries. |
| Full original 291-case train segmentation source masks | Required to validate population-level real distribution including 30 multi-component cases. | Summary JSON documents 291 rows, while matched benchmark uses 261 single-component cases. |
| Clinical/anatomical labels for IAC/CPA canal-vs-bulb compartment split | Required to distinguish biologically meaningful lollipop differences from PCA/bounding-box extraction artifacts. | Whole-mask morphology metrics only. |

## Generated Local Artifacts

- `sample_inventory.csv`
- `local_synthetic_manifest.csv`
- `matched_case_feature_table.csv`
- `matched_distribution_comparison.csv`
- `volume_stratified_comparison.csv`
- `ranked_morphology_gaps.csv`
- `current_local_extractor_matched_distribution_comparison.csv` if local synthetic re-extraction exists
- `current_local_extractor_volume_stratified_comparison.csv` if local synthetic re-extraction exists
- `current_local_extractor_ranked_morphology_gaps.csv` if local synthetic re-extraction exists
- `synthetic_pulled_vs_reextracted_metric_drift.csv` if local synthetic re-extraction exists
- `remote_data_dependencies.csv`
- `plots/*.png`

Plots:
- `analysis/real_vs_synthetic/plots/elongation_boxplot.png`
- `analysis/real_vs_synthetic/plots/aspect_ratio_major_to_minor2_boxplot.png`
- `analysis/real_vs_synthetic/plots/bbox_fill_fraction_boxplot.png`
- `analysis/real_vs_synthetic/plots/sphericity_boxplot.png`
- `analysis/real_vs_synthetic/plots/max_diameter_bbox_mm_boxplot.png`
- `analysis/real_vs_synthetic/plots/principal_axis_length_major_mm_boxplot.png`
- `analysis/real_vs_synthetic/plots/surface_to_volume_ratio_boxplot.png`
- `analysis/real_vs_synthetic/plots/volume_match_scatter.png`
- `analysis/real_vs_synthetic/plots/volume_stratified_ratio_heatmap.png`

The plots currently visualize the historical pulled-feature comparison. Use the `current_local_extractor_*.csv` tables for current-extractor numerical review until matching current plots are added.

## Reproduction Commands

```bash
python3 analysis/real_vs_synthetic/analyze_local_morphology.py

python3 scripts/extract_real_tumor_features.py \
  --input_csv analysis/real_vs_synthetic/local_synthetic_manifest.csv \
  --seg_col seg_path \
  --case_id_col case_id \
  --out_csv analysis/real_vs_synthetic/synthetic_features_reextracted_local.csv \
  --summary_json analysis/real_vs_synthetic/synthetic_feature_summary_reextracted_local.json

python3 analysis/real_vs_synthetic/analyze_local_morphology.py
```

Initial inventory/report generation without local synthetic re-extraction:

```bash
python3 analysis/real_vs_synthetic/analyze_local_morphology.py
```

## Definition of Done Status

- Local real feature data located: COMPLETE.
- Sample count and case-ID semantics verified: COMPLETE.
- Local synthetic cohort located: COMPLETE.
- Equivalent feature schema verified: COMPLETE.
- Real source-mask re-extraction: BLOCKED_REMOTE_DATA.
- Distribution comparison and volume stratification: COMPLETE.
- Unsupported population-level conclusions avoided: COMPLETE.
