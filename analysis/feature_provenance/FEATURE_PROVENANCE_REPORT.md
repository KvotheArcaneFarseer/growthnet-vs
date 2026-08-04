# Feature Provenance Report

## Conclusion

The previously pulled synthetic feature tables do not reproduce because they were not generated from the same local synthetic mask instances now present under `rivanna_pull/analysis/synthetic_lollipop_v1/masks/`. This is established before any higher-order feature formula question: stored voxel counts and volumes frequently differ from both the local masks and the paired manifest realized volumes.

The most likely generating code path for the pulled tables is: older `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py`-style masks followed by the same extractor schema as `scripts/extract_real_tumor_features.py`. The current local extractor and the pulled extractor copy are identical, so local extractor-code drift is not the primary cause. The current generator differs from the pulled generator, and the generator diff matches the observed pattern: tiny cases in the pulled feature CSV show old minimum-volume overshoot, while local masks match the corrected manifest realized volumes.

The remaining causes are narrowed to a small set: an older overwritten/not-pulled mask set, an artifact packaging mismatch where stale features were paired with refreshed masks/manifest, or an unrecorded post-generation mask replacement before pull. The evidence most strongly favors stale pulled features from an earlier generation run paired with newer local masks and manifest.

Do not tune tumor geometry or generator parameters from the stale pulled feature table. Regenerate features from the authoritative local masks, or recover the exact older masks that produced the pulled feature CSV.

## Key Evidence

- Compared 261 case IDs with pulled features, manifest rows, and local masks.
- Local mask volume equals manifest `realized_volume_mm3` for 261 cases.
- Pulled feature volume equals local mask volume for only 9 cases.
- Pulled feature volume equals manifest realized volume for only 9 cases.
- Median absolute pulled-vs-local volume difference: 8.0 mm3; max: 211.0 mm3.
- Current extractor versus pulled extractor copy: identical (`diff -q`).
- Current generator versus pulled generator copy: different (`diff -q`).

## Representative Cases

| Case ID | Stratum | Target Volume | Manifest Realized | Pulled Volume | Local Path |
| --- | --- | ---: | ---: | ---: | --- |
| 439_0_0 | small | 7.750 | 8.000 | 42.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/439_0_0_synthetic_lollipop.nii.gz` |
| 492_1_391 | small | 8.500 | 9.000 | 44.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/492_1_391_synthetic_lollipop.nii.gz` |
| 235_0_0 | small | 8.875 | 9.000 | 41.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/235_0_0_synthetic_lollipop.nii.gz` |
| 301_0_0 | small | 12.250 | 12.000 | 43.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/301_0_0_synthetic_lollipop.nii.gz` |
| 434_0_0 | medium | 107.750 | 106.000 | 105.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/434_0_0_synthetic_lollipop.nii.gz` |
| 593_3_2370 | medium | 111.500 | 109.000 | 116.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/593_3_2370_synthetic_lollipop.nii.gz` |
| 434_1_160 | medium | 270.375 | 278.000 | 258.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/434_1_160_synthetic_lollipop.nii.gz` |
| 595_3_1268 | large | 636.125 | 645.000 | 605.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/595_3_1268_synthetic_lollipop.nii.gz` |
| 304_1_206 | large | 643.875 | 648.000 | 612.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/304_1_206_synthetic_lollipop.nii.gz` |
| 132_0_0 | large | 7679.750 | 7674.000 | 7663.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/132_0_0_synthetic_lollipop.nii.gz` |
| 540_3_1126 | large | 8889.625 | 8893.000 | 8901.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/540_3_1126_synthetic_lollipop.nii.gz` |
| 132_1_148 | large | 10363.500 | 10057.000 | 10268.000 | `rivanna_pull/analysis/synthetic_lollipop_v1/masks/132_1_148_synthetic_lollipop.nii.gz` |

## Mismatch Patterns

| Feature | Changed Cases | Median Abs Diff | Max Abs Diff | Pattern |
| --- | ---: | ---: | ---: | --- |
| bbox_volume_mm3 | 12 | 741.5 | 45618 | systematic/bbox-related |
| compactness | 12 | 319.594 | 361.114 | feature-specific/surface-related |
| surface_area_mm2 | 12 | 80.9421 | 1189.15 | feature-specific/surface-related |
| largest_component_voxel_count | 12 | 31.5 | 211 | case-specific |
| mask_voxel_count | 12 | 31.5 | 211 | case-specific |
| volume_mm3 | 12 | 31.5 | 211 | case-specific |
| max_diameter_bbox_mm | 12 | 7.5 | 24 | systematic/bbox-related |
| principal_axis_length_major_mm | 12 | 7.30488 | 21.0046 | systematic/PCA-related |
| bounding_box_size_mm_x | 12 | 5.5 | 30 | case-specific |
| bounding_box_size_vox_x | 12 | 5.5 | 30 | case-specific |
| bounding_box_size_mm_y | 12 | 2.5 | 24 | case-specific |
| bounding_box_size_vox_y | 12 | 2.5 | 24 | case-specific |
| bounding_box_size_mm_z | 12 | 2 | 6 | case-specific |
| bounding_box_size_vox_z | 12 | 2 | 6 | case-specific |
| principal_axis_length_minor1_mm | 12 | 1.96572 | 10.211 | systematic/PCA-related |
| principal_axis_length_minor2_mm | 12 | 1.88006 | 6.36575 | systematic/PCA-related |
| aspect_ratio_major_to_minor2 | 12 | 1.27228 | 1.50619 | systematic/PCA-related |
| elongation_legacy_major_to_minor2 | 12 | 1.27228 | 1.50619 | systematic/PCA-related |
| elongation | 12 | 1.18357 | 1.40484 | systematic/PCA-related |
| roughness_index | 12 | 0.495156 | 0.542738 | feature-specific/surface-related |

## Full Prior Re-Extraction Drift

| Metric | Pulled Median | Local Re-Extracted Median | Median Abs Diff | Changed Rows |
| --- | ---: | ---: | ---: | ---: |
| volume_mm3 | 262 | 267 | 8 | 252 |
| equivalent_sphere_diameter_mm | 7.93903 | 7.98922 | 0.0744128 | 252 |
| max_diameter_bbox_mm | 16 | 9 | 7 | 261 |
| principal_axis_length_major_mm | 16.3883 | 8.561 | 7.58644 | 261 |
| principal_axis_length_minor1_mm | 9.23131 | 7.07107 | 2.20872 | 261 |
| principal_axis_length_minor2_mm | 8.28487 | 6.56911 | 1.57664 | 261 |
| elongation | 2.46585 | 1.21599 | 1.20484 | 261 |
| aspect_ratio_major_to_minor2 | 2.72922 | 1.25196 | 1.42401 | 261 |
| flatness | 0.901577 | 0.98749 | 0.0800211 | 261 |
| bbox_fill_fraction | 0.17033 | 0.481481 | 0.310766 | 261 |
| surface_area_mm2 | 325.477 | 218.955 | 112.074 | 261 |
| sphericity | 0.613635 | 0.921114 | 0.257869 | 261 |
| surface_to_volume_ratio | 1.23173 | 0.808311 | 0.317623 | 261 |
| roughness_index | 1.62963 | 1.08564 | 0.491827 | 261 |

## Git/Formula Inspection

- Relevant committed history for extractor/generator/embed files is short locally:
  - `0457540 Add GrowthNet local roadmap, validation audits, and reliability updates`
  - `242631b Split orientation confidence and score-margin warnings`
  - `ccf9ffe Add AI-assisted development workflow (Claude + Codex)`
- `scripts/extract_real_tumor_features.py` was added in commit `0457540`; no later committed formula changes are present locally for the extractor.
- The pulled extractor copy under `rivanna_pull/scripts/extract_real_tumor_features.py` is byte-identical to the current extractor.
- The pulled generator copy under `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py` differs materially from the current generator. The current generator adds explicit spacing parsing, lower tiny-target floors, compact single-mask initialization, delayed bulb creation for smaller targets, and comments describing fixes for prior tiny-target tail failures.
- Those generator differences are consistent with the volume evidence: for example, selected tiny cases have pulled feature volumes around 41-44 mm3 while the manifest/local masks realize 8-12 mm3 targets.
- The extractor computes spacing-aware volume, largest-component connected-component features, marching-cubes surface area, sphericity, compactness, bounding boxes, and PCA lengths. These formulas explain the feature schema but do not explain changed voxel counts for same case IDs.
- Prior audit notes in `docs/AUTONOMOUS_CHANGE_AUDIT.md` flag a formula inconsistency: `principal_axis_length_*` can be overwritten with extent-based values while elongation/flatness/aspect ratio remain moment-based. That is a reporting risk, not the root cause of pulled-vs-local non-reproduction.

## Artifact Classification

See `provenance_decision_table.csv` for the formal artifact decisions. In short: local masks and manifest are authoritative for local reproduction; pulled feature CSV/JSON/summary are stale; the missing older mask set is unknown provenance; fresh regenerated features are required for current local masks.

## Commands Run

```bash
python3 analysis/feature_provenance/build_feature_provenance.py
python3 /Users/kvothearcane/Personal/Coding Projects/GrowthNet/scripts/extract_real_tumor_features.py --input_csv /Users/kvothearcane/Personal/Coding Projects/GrowthNet/analysis/feature_provenance/selected_synthetic_masks.csv --seg_col seg_path --case_id_col case_id --out_csv /Users/kvothearcane/Personal/Coding Projects/GrowthNet/analysis/feature_provenance/current_reextracted_selected_features.csv --out_json /Users/kvothearcane/Personal/Coding Projects/GrowthNet/analysis/feature_provenance/current_reextracted_selected_features.json --summary_json /Users/kvothearcane/Personal/Coding Projects/GrowthNet/analysis/feature_provenance/current_reextracted_selected_summary.json
diff -q scripts/extract_real_tumor_features.py rivanna_pull/scripts/extract_real_tumor_features.py
diff -q scripts/generate_synthetic_lollipop_cohort.py rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py
git log --oneline --all -- scripts/extract_real_tumor_features.py scripts/generate_synthetic_lollipop_cohort.py embed_tumor.py
```

## Unresolved Risks

- No remote/Rivanna access was used, so the exact older mask archive, if it exists, was not recovered.
- Original pulled artifacts were not overwritten; this investigation cannot prove whether the mismatch happened during generation, packaging, or local pull/copy.
- Only 12 representative cases were freshly re-extracted into this folder, though prior local evidence covers all 261 matched cases.
- Real source segmentation masks are not local, so real-feature reproducibility was not independently tested here.

## Deliverables

- `artifact_inventory.csv`: 276 inventory rows, including all local synthetic masks.
- `feature_comparison.csv`: selected pulled-vs-current feature comparisons.
- `provenance_decision_table.csv`: artifact classifications and actions.
- `current_reextracted_selected_features.csv`: current extractor output for representative local masks.
