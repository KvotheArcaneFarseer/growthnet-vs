# Real Tumor Feature Extractor

## Purpose

`scripts/extract_real_tumor_features.py` extracts a standardized morphology feature table from real vestibular schwannoma segmentation NIfTI masks. The output is intended for downstream analysis and future calibration of synthetic tumor generation.

The script is standalone and does not modify embedding or synthetic generation code.

## Usage

### Single file

```bash
python3 scripts/extract_real_tumor_features.py \
  --seg_path /path/to/case_seg.nii.gz \
  --out_csv /tmp/real_tumor_features.csv
```

### Glob input

```bash
python3 scripts/extract_real_tumor_features.py \
  --glob "/data/segs/*_VS*.nii.gz" \
  --out_csv /tmp/real_tumor_features.csv \
  --summary_json /tmp/real_tumor_feature_summary.json
```

### Input CSV

```bash
python3 scripts/extract_real_tumor_features.py \
  --input_csv /path/to/cases.csv \
  --seg_col seg_path \
  --case_id_col case_id \
  --out_csv /tmp/real_tumor_features.csv \
  --out_json /tmp/real_tumor_features.json \
  --summary_json /tmp/real_tumor_feature_summary.json
```

### Optional principal-axis sign alignment

```bash
python3 scripts/extract_real_tumor_features.py \
  --glob "/data/segs/*.nii.gz" \
  --reference_axis_vox 0,0,1 \
  --out_csv /tmp/real_tumor_features.csv
```

You may use `--reference_axis_vox` or `--reference_axis_mm` (not both).

## Output files

- `--out_csv`: flat per-case table (one row per input case).
- `--out_json` (optional): per-case feature payload plus failed-case list and summary.
- `--summary_json` (optional): dataset-level summary statistics and QC counts.

Batch failure handling is robust: unreadable or invalid segmentations produce a row with:
- `failed=True`
- `failure_reason=<error message>`
- unavailable metrics left as `NaN`/`None`

## Important columns

- `volume_mm3`: full-mask volume in mm^3.
- `surface_area_mm2`: largest-component surface area from marching cubes with spacing.
- `sphericity`: standard 3D sphericity metric.
- `compactness`: `surface_area^3 / volume^2` (kept for compatibility; not dimensionless).
- `centroid_vox_*`: centroid in voxel index space.
- `centroid_mm_*`: centroid in NIfTI affine world coordinates (mm).
- `principal_axis_vector_vox_*` / `principal_axis_vector_mm_*`: principal direction vectors.
- `principal_axis_sign_aligned`: indicates whether sign normalization was requested/applied.
- `principal_axis_length_major_mm`, `minor1`, `minor2`: major/intermediate/minor axis lengths.
- `elongation`: `major / minor1`.
- `flatness`: `minor2 / minor1`.
- `bbox_fill_fraction`: `volume_mm3 / bbox_volume_mm3`.

## Known limitations

- No explicit canal/bulb anatomical split is computed yet.
  - That requires anatomical landmarking, IAC/CPA region masks, or a reliable inferred canal axis and porus boundary.
- Principal-axis sign is mathematically ambiguous unless `--reference_axis_vox` or `--reference_axis_mm` is provided.
- `centroid_mm` depends on the image affine (world coordinate frame is dataset-dependent).
- Surface metrics depend on segmentation resolution and marching-cubes discretization.

## Lightweight validation command

```bash
PYTHONPYCACHEPREFIX=/tmp python3 -m py_compile scripts/extract_real_tumor_features.py

python3 scripts/extract_real_tumor_features.py \
  --seg_path /Users/kvothearcane/Downloads/147_0_0_t2_thin_R_VS__uvauser2__coregistered.nii.gz \
  --reference_axis_vox 0,0,1 \
  --out_csv /tmp/real_tumor_features_147_v3.csv \
  --out_json /tmp/real_tumor_features_147_v3.json \
  --summary_json /tmp/real_tumor_feature_summary_147_v3.json
```
