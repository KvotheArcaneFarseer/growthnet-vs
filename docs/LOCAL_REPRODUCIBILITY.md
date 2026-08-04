# Local Reproducibility

Last updated: 2026-08-04

Scope: local laptop repository only. These commands do not require SSH, Rivanna, SLURM, or remote datasets when run as written, except where explicitly marked as requiring user-supplied local NIfTI files.

## Environment Notes

The repository now includes a root `pyproject.toml` for the local GrowthNet
development environment. The default dependencies cover the fast local
embedding, synthetic generation, feature extraction, and ViViT data-adapter
tests. Heavier stacks are separated into optional extras:

- `.[dev]` for pytest.
- `.[registration]` for MRI registration dependencies such as ANTsPyNet and
  TensorFlow.
- `.[training]` for training helpers such as Accelerate.
- `.[presentations]` for deck/export helpers.

The previously validated local environment is `.venv`, which has pytest
available. The system `python3` on this laptop may not.

Create or refresh a local editable environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Capture an environment freeze for a reviewed run:

```bash
mkdir -p projects/growthnet/configs/env
.venv/bin/python -m pip freeze > projects/growthnet/configs/env/growthnet_env_local_$(date +%Y_%m_%d)_pip_freeze.txt
```

Commit freeze files only when they represent a reviewed project environment,
not every scratch environment change.

## CI

The repository includes a minimal GitHub Actions workflow:

```text
.github/workflows/fast-tests.yml
```

It installs `.[dev]`, compiles core scripts, and runs:

```bash
python -m pytest -m "fast and not slow" -v
```

The workflow is laptop-safe by design: it does not access Rivanna, download
datasets, run SLURM, or generate expensive cohorts.

## Fast Local Checks

Syntax/import smoke for the current local embedding and generation scripts:

```bash
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile \
  embed_tumor.py \
  scripts/run_batch_embedding.py \
  scripts/extract_real_tumor_features.py \
  scripts/generate_synthetic_lollipop_cohort.py \
  scripts/generate_synthetic_longitudinal_dataset.py \
  projects/vivit/src/data/synthetic.py
```

Run the local pytest suite when `pytest` is available:

```bash
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache MPLCONFIGDIR=/tmp/growthnet_mpl python3 -m pytest tests -q
```

Run only fast tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache MPLCONFIGDIR=/tmp/growthnet_mpl .venv/bin/python -m pytest -m "fast and not slow" -v
```

Run integration-marked tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache MPLCONFIGDIR=/tmp/growthnet_mpl python3 -m pytest tests -m integration -q
```

## Single-Case Embedding

Requires local MRI and segmentation NIfTI files:

```bash
python3 embed_tumor.py \
  --mri /absolute/path/to/background_mri.nii.gz \
  --seg /absolute/path/to/background_seg.nii.gz \
  --out_dir /tmp/growthnet_embedding_smoke \
  --seed 20260523 \
  --gen_size 64
```

Expected outputs include:

- `embedded_tumor_volume.nii.gz`
- `embedded_tumor_mask.nii.gz`
- `embedded_tumor_late_volume.nii.gz`
- `embedded_tumor_late_mask.nii.gz`
- `embedding_metrics.json`
- `embedding_metrics.csv`
- `qc_embedding.png`
- `qc_embedding_late.png`

Do not rely on the script defaults for reproducibility; pass explicit `--mri`, `--seg`, and `--out_dir` paths.

## Batch Embedding

Create a local CSV:

```csv
case_id,mri_path,seg_path,target_tumor_volume_mm3,volume_target_timepoint
local_case_001,/absolute/path/to/mri.nii.gz,/absolute/path/to/seg.nii.gz,250,first
```

Run a small local batch:

```bash
python3 scripts/run_batch_embedding.py \
  --input_csv /absolute/path/to/local_batch_cases.csv \
  --out_dir /tmp/growthnet_local_batch \
  --num_cases 1 \
  --resume
```

Expected batch outputs include `batch_summary.json`, `batch_summary.csv`, `failure_cases.json`, and `batch_case_status.jsonl`. With `--resume`, cases with the required output set and no hard failures are skipped.

## Synthetic Lollipop Cohort

Create a target-volume CSV:

```csv
case_id,target_volume_mm3
synthetic_001,100
synthetic_002,500
```

Run:

```bash
python3 scripts/generate_synthetic_lollipop_cohort.py \
  --targets_csv /absolute/path/to/targets.csv \
  --out_dir /tmp/growthnet_synthetic_lollipop \
  --seed 20260523 \
  --max_calibration_iters 8
```

The script writes one mask per case and a manifest CSV. It generates standalone synthetic masks, not embedded patient-specific MRI volumes.

## Feature Extraction

Single segmentation:

```bash
python3 scripts/extract_real_tumor_features.py \
  --seg_path /absolute/path/to/seg.nii.gz \
  --out_csv /tmp/growthnet_features.csv \
  --summary_json /tmp/growthnet_feature_summary.json
```

CSV input:

```bash
python3 scripts/extract_real_tumor_features.py \
  --input_csv /absolute/path/to/cases.csv \
  --seg_col seg_path \
  --case_id_col case_id \
  --out_csv /tmp/growthnet_features.csv \
  --summary_json /tmp/growthnet_feature_summary.json
```

## Longitudinal MVP

By default, the wrapper consumes explicit target volumes from the timeline CSV.
It can also generate experimental scenario-based target volumes, but those modes
are simulation controls and are not clinically validated growth predictors.

Example timeline:

```bash
cat data/timelines/local_longitudinal_example.csv
```

Create a local background manifest:

```csv
background_mri_id,mri_path,seg_path
BG_LOCAL_001,/absolute/path/to/background_mri.nii.gz,/absolute/path/to/background_seg.nii.gz
```

Run:

```bash
python3 scripts/generate_synthetic_longitudinal_dataset.py \
  --timeline_csv data/timelines/local_longitudinal_example.csv \
  --background_csv /absolute/path/to/local_backgrounds.csv \
  --out_dir /tmp/growthnet_longitudinal_local_smoke \
  --seed 20260523 \
  --gen_size 64 \
  --volume_max_iterations 3
```

Optional scenario/variant smoke:

```bash
python3 scripts/generate_synthetic_longitudinal_dataset.py \
  --timeline_csv data/timelines/local_longitudinal_example.csv \
  --background_csv /absolute/path/to/local_backgrounds.csv \
  --out_dir /tmp/growthnet_longitudinal_variants \
  --seed 20260523 \
  --gen_size 64 \
  --volume_max_iterations 3 \
  --clinical_growth_law scenario_mixture_v1 \
  --variants_per_timepoint 2
```

Expected outputs are `metadata.csv`, `qc_summary.csv`,
`longitudinal_qc_summary.csv`, `images/`, and `masks/`. Metadata records
`growth_law_scenario`, `variant_id`, and `variant_seed` for auditability.

Measure same-timepoint variant diversity after a multi-variant run:

```bash
python3 analysis/clinical_growth_law_validation/measure_variant_diversity.py \
  --metadata_csv /tmp/growthnet_longitudinal_variants/metadata.csv \
  --out_csv /tmp/growthnet_longitudinal_variants/variant_diversity.csv \
  --out_report /tmp/growthnet_longitudinal_variants/VARIANT_DIVERSITY.md
```

## Local Analysis Reproduction

Orientation diagnostic:

```bash
python3 analysis/orientation_validation/orientation_diagnostic.py
```

Morphology audit:

```bash
python3 analysis/real_vs_synthetic/analyze_local_morphology.py
```

Synthetic feature re-extraction from local synthetic masks:

```bash
python3 scripts/extract_real_tumor_features.py \
  --input_csv analysis/real_vs_synthetic/local_synthetic_manifest.csv \
  --seg_col seg_path \
  --case_id_col case_id \
  --out_csv analysis/real_vs_synthetic/synthetic_features_reextracted_local.csv \
  --summary_json analysis/real_vs_synthetic/synthetic_feature_summary_reextracted_local.json
```

Volume-targeting smoke benchmark:

```bash
python3 analysis/volume_targeting/run_volume_targeting_benchmark.py \
  --out_csv /tmp/growthnet_volume_targeting/volume_targeting_benchmark.csv \
  --targets 50,100,500 \
  --seeds 20260523 \
  --max_iters 8
```

Plots are written next to `--out_csv`. Check the helper's `--help` output before larger runs; exhaustive target grids can become expensive.

## Outputs And Data Policy

Prefer `/tmp/...` or another explicitly chosen local scratch directory for new generated outputs during development. Do not commit large MRI volumes, private clinical data, checkpoints, or secrets. Existing local analysis/provenance artifacts should not be deleted without review.
