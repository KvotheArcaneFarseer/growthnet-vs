# Vestibular Schwannoma MRI Preprocessing & Coregistration Pipeline
**Author:** Matthew Nguyen | **Version:** Pipeline v6.5 / Preprocessing v6.8 | **Year:** 2026

## Quick Reference

| Item | Details |
|------|---------|
| **What it does** | Three-stage pipeline: data prep, hierarchical MRI coregistration to MNI space, QC sorting |
| **Inputs** | `meta_df_entropy.csv` (from Stage 1), MNI template folder |
| **Smoke test** | `python -m pytest tests/test_smoke.py -v` |
| **Rivanna run** | `sbatch scripts/preprocess_batches.slurm` |
| **Output location** | `code_runs/mri_registration/<experiment>/<runid>/` |
| **Reproducibility** | Each run logs git commit, dataset_id, seed, and environment freeze via `shared.run_logger` |

---

## Overview

Three-stage pipeline that processes longitudinal vestibular schwannoma (VS) MRI data into coregistered images in MNI template space for downstream analysis. The pipeline handles multi-batch data from the UVA Carina PACS export, performs brain extraction, applies a seven-step image preprocessing chain, and hierarchically coregisters all patient timepoints to a common MNI template. A post-registration QC sorting step separates acceptable results from those requiring manual correction.

| Stage | Entry Point | Source | Role |
|---|---|---|---|
| 1 | `VS data timeline.preprocessing_final.ipynb` | — | Data inventory, path resolution, feature computation |
| 1b | `scripts/run_hd_bet.py --config configs/default.yaml` | `src/hd_bet.py` | HD-BET brain extraction |
| 2 | `scripts/run_preprocessing.py --config configs/default.yaml` | `src/preprocessing.py` | Image preprocessing, hierarchical coregistration, QC output |
| 3 | `scripts/run_sort.py --config configs/default.yaml --csv reviewed.csv` | `src/sort_registrations.py` | Post-registration QC sorting into acceptable / manual correction |

**Stage 1 must complete before Stage 2.** Stage 1 produces `meta_df_entropy.csv`, which is Stage 2's required input. Stage 3 runs after Stage 2 output has been manually reviewed.

---

## TODO

- [ ] **Consolidate input to a `manifest.csv`** -- Replace the multi-batch `original_meta.csv` loading logic with a single pre-built `manifest.csv` that lists all series paths, accession numbers, and clinical metadata. Use `manifest.csv` as the sole input to the Stage 1 notebook, eliminating the batch-folder loop and fuzzy path recovery cells.
- [ ] **Split Stage 1 notebook into standalone scripts:**
  - [ ] `generate_manifest.py` -- CSV generation only (batch loading, path validation, fuzzy recovery, clinical timeline merge, NIfTI discovery, column cleanup). Outputs `manifest.csv`.
  - [x] `run_hd_bet.py` -- HD-BET brain extraction via `scripts/run_hd_bet.py --config configs/default.yaml`. Calls `src/hd_bet.py`.
  - [ ] `compute_statistics.py` -- Descriptive statistics and feature enrichment only (entropy, voxel volume, file size, segmentation volume, Dice QC). Reads `manifest.csv` + `hd_output/` and writes `meta_df_entropy.csv`.

---

## Quick Start

```bash
# 1. Create and activate environment
conda env create -f preprocessing_environment.yml
conda activate antspy_registration_env

# 2. Place MNI templates (see Template Setup below)
ls Template/
# mni_t1.nii.gz  mni_t1_bet.nii.gz  mni_t2.nii.gz  mni_t2_bet.nii.gz

# 3. Run Stage 1 (cells 1-22 in notebook)
jupyter notebook "VS data timeline.preprocessing_final.ipynb"

# 4. Run HD-BET brain extraction
python scripts/run_hd_bet.py --config configs/default.yaml

# 5. Run Stage 2 preprocessing & coregistration
python scripts/run_preprocessing.py --config configs/default.yaml

# 5a. (Optional) Submit to SLURM cluster instead
sbatch scripts/preprocess_batches.slurm

# 6. Review QC overlays, annotate reviewed CSV, then sort
python scripts/run_sort.py --config configs/default.yaml --csv path/to/reviewed.csv
```

---

## Prerequisites

### Raw Data Requirements

This pipeline expects data organized from the UVA Carina PACS export:

```
/standard/gam_ai_group/ExtractedData-08-13-2025/
├── VS_batch1/Original/original_meta.csv
├── VS_batch2/Original/original_meta.csv
...
└── VS_batch5/Original/original_meta.csv
```

Each `original_meta.csv` must contain at minimum:
- `Downloaded Original Series Path` -- relative path to the series folder
- `Accession Number` -- links to the clinical timeline spreadsheet

Each series folder must contain `.nii.gz` files following these naming conventions:
- **Image file:** filename contains the word `image` (e.g., `series_image.nii.gz`)
- **Segmentation file:** filename contains `uvauser` (e.g., `series_uvauser.nii.gz`); if multiple segmentations exist, the one containing `copy` is preferred

You also need the de-identified Carina clinical database spreadsheet:
```
VS AI Database, Draft 12.01 Jack Veda reconciled.xlsx
```
This file must contain `Carina Accession Number` and `Patient_MRI_Days Tracker` columns.

> **Adapting to other datasets:** The segmentation filename convention (`uvauser`) and the batch folder structure are UVA-specific. To use this pipeline with different data, update `find_nifti_files()` in Cell 10 of the notebook to match your naming convention, and update `root_dir` and the batch loading loop in Cell 1.

### Template Setup

The pipeline registers all images to MNI space. Four template files are required in the `Template/` folder. These can be obtained from the [MNI ICBM 2009c Nonlinear Symmetric](http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009) atlas:

| Filename | Description |
|---|---|
| `mni_t1.nii.gz` | MNI T1-weighted template |
| `mni_t1_bet.nii.gz` | Brain-extracted MNI T1 |
| `mni_t2.nii.gz` | MNI T2-weighted template |
| `mni_t2_bet.nii.gz` | Brain-extracted MNI T2 |

The pipeline matches filenames by looking for `t1` or `t2` and `bet` in the filename -- the exact names above are not required, but the naming pattern must be consistent (e.g., any file with both `t2` and `bet` in the name is treated as the T2 brain-extracted template).

---

## Environment Setup

### Option 1: Conda environment file (recommended)

```bash
conda env create -f preprocessing_environment.yml
conda activate antspy_registration_env
```

The environment file (`preprocessing_environment.yml`) pins all dependencies including antspyx 0.6.1, antspynet 0.3.1, tensorflow 2.17.0, and Python 3.11.

### Option 2: Manual install

```bash
conda create -n nguyen_ants_env python=3.9
conda activate nguyen_ants_env

# ANTs and brain extraction
pip install antspy antspynet

# Image I/O
pip install nibabel SimpleITK tifffile

# Data & plotting
pip install pandas numpy scipy matplotlib tqdm Pillow openpyxl

# TensorFlow (required by antspynet -- CPU-only for Stage 2)
pip install tensorflow
```

### HD-BET (Stage 1 only)

```bash
# Requires a CUDA-compatible GPU
pip install hd-bet

# Verify installation
hd-bet --help
```

Update `hd_bet_path` in Cell 18 of the notebook to point to the installed binary:
```python
hd_bet_path = '/path/to/your/conda/envs/nguyen_ants_env/bin/hd-bet'
```

---

## Repository Structure

```
mri_registration/
├── configs/
│   └── default.yaml                              # Pipeline configuration (paths, params, toggles)
├── scripts/
│   ├── run_preprocessing.py                      # Entry point: Stage 2 preprocessing
│   ├── run_hd_bet.py                             # Entry point: HD-BET brain extraction
│   ├── run_sort.py                               # Entry point: Stage 3 QC sorting
│   ├── preprocess_batches.slurm                  # SLURM job script for preprocessing
│   └── hd_bet.slurm                              # SLURM job script for HD-BET
├── src/
│   ├── preprocessing.py                          # Stage 2 core: preprocessing & coregistration (v6.8)
│   ├── hd_bet.py                                 # HD-BET brain extraction logic
│   └── sort_registrations.py                     # Stage 3 core: QC sorting
├── tests/
│   └── test_smoke.py                             # Smoke tests (imports, config, _apply_config)
├── docs/                                         # Additional documentation
├── VS data timeline.preprocessing_final.ipynb    # Stage 1: data preparation notebook
├── preprocessing_environment.yml                 # Conda environment with pinned deps
├── README.md                                     # This file
├── Template/                                     # MNI T1, T2, and their _bet.nii.gz variants
├── raw_studies/                                  # Stage 1 output: organized NIfTI by modality/patient
│   ├── t2_thin/<PatientTimepoint>/
│   └── t1+_thin/<PatientTimepoint>/
├── hd_output/                                    # HD-BET brain masks (<basename>_bet.nii.gz)
├── meta_df_entropy.csv                           # Stage 1 final output -> Stage 2 input
└── code_runs/mri_registration/                   # Run folders with metadata (via shared.run_logger)
    └── <experiment>/<runid>/                      # Config snapshot, git commit, env freeze per run
```

---

## Stage 1: Data Preparation Notebook

**Script:** `VS data timeline.preprocessing_final.ipynb` (33 cells)

### How to Run

```bash
jupyter notebook "VS data timeline.preprocessing_final.ipynb"
```

Run cells sequentially. Update these paths before running:
- **Cell 1:** `root_dir` -- path to extracted batch data
- **Cell 6:** Excel filename -- de-identified Carina database
- **Cell 18:** `hd_bet_path` -- path to HD-BET binary
- **Cell 22:** `COREGISTERED_FOLDER` -- Stage 2 output folder name (if looking up existing coregistered files)

### Cell-by-Cell Walkthrough

**Cells 1-2 -- Batch loading & path normalization**
Reads `original_meta.csv` from 5 data batches, concatenates them, and normalizes path strings (backslashes to forward slashes, strips leading whitespace per segment). Builds absolute paths by prepending `root_dir/VS_batch{id}/`.

**Cells 3-5 -- Path validation & fuzzy recovery**
Checks whether each path exists on disk. For missing paths, walks the directory tree using `difflib.get_close_matches` to recover paths broken by case mismatches or whitespace. Unrecoverable paths are saved to `missing_paths_<date>.csv`.

**Cell 6 -- Clinical timeline merge**
Loads the Carina Excel file and inner-joins on accession number to attach `Patient_MRI_Days Tracker` (format: `PatientID_ScanOrder_DaysBetweenScans`) to each scan row.

**Cells 8-9 -- Study type filtering**
Renames columns to standard names (`Series Labels` -> `Study Type`, etc.) and keeps only: `t2_thin`, `t1+_thin`, `t1+_thick_cor`, `t1+_thick_ax`, `t2_thick`.

**Cell 10 -- NIfTI discovery & organized copy**
For each row, finds the image file (filename contains `"image"`) and segmentation (filename contains `"uvauser"`). Copies both to `./raw_studies/<SeriesLabel>/<PatientTimepoint>/` with a descriptive filename prefix. Copies are made rather than path references because source data is read-only NAS storage organized by DICOM accession, not by patient/timepoint.

**Cells 12-14 -- Column cleanup & study label filtering**
Drops PHI columns (Patient Name, Birth Date, Study Date, Accession Number), renames columns to standard names, and filters out non-VS study labels.

**Cells 16-21 -- HD-BET brain extraction**
Stages unprocessed images into `hd_input/`, runs HD-BET in batch mode, falls back to per-file if batch fails, then cleans up `hd_input/`. Skips images that already have outputs in `hd_output/`. Outputs (`_bet.nii.gz`) are used by Stage 2 to guide registration and optionally remove background.

**Cell 22 -- Feature enrichment -> `meta_df_entropy.csv`**
Computes per-row: Shannon entropy, voxel volume, file size (MB), ANTsPy spacing, segmentation volume, and paths to any existing coregistered outputs. Saves to `meta_df_entropy.csv`. If the file already exists, it is loaded directly.

**Cells 25-30 -- Post-coregistration Dice QC** *(run after Stage 2)*
Computes pairwise Dice coefficients between all coregistered segmentations within each patient timepoint to validate registration quality. Saves `pairwise_dice_df.csv` and `avg_dice_per_study_df.csv`.

### Stage 1 Outputs

| File | Description |
|---|---|
| `meta_df_entropy.csv` | **Primary output** -- enriched metadata, Stage 2 input |
| `raw_studies/` | Organized NIfTI files by modality and patient/timepoint |
| `hd_output/` | Brain extraction masks |
| `missing_paths_<date>.csv` | Paths that could not be resolved |
| `pairwise_dice_df.csv` | Pairwise Dice QC (post Stage 2) |

---

## Stage 2: Preprocessing & Coregistration Pipeline

**Source:** `src/preprocessing.py` (v6.8) | **Entry point:** `scripts/run_preprocessing.py`

### How to Run

```bash
# Via entry point (recommended — loads config, creates run folder, then calls src/)
python scripts/run_preprocessing.py --config configs/default.yaml

# HPC cluster (SLURM)
sbatch scripts/preprocess_batches.slurm

# Direct execution (legacy — uses inline defaults, no run logging)
python src/preprocessing.py --config configs/default.yaml
```

The SLURM script (`scripts/preprocess_batches.slurm`) targets the `standard` partition with 64 CPUs, 96 GB RAM, and a 32-hour time limit. It loads the `miniforge` module, activates `antspy_registration_env`, and runs `scripts/run_preprocessing.py`. Edit the `--account`, `--mail-user`, and resource parameters at the top of the file before submitting.

**Before running:** Edit `configs/default.yaml` to set paths for your environment:
1. `input_csv` -- path to `meta_df_entropy.csv`
2. `template_dir` -- path to MNI template folder
3. `output_base_dir` -- root for coregistered output
4. Set `max_patients: 5` and `test_mode: true` for a dry-run validation first

### Preprocessing Steps

Applied to each raw image in `get_ants_image()`:

1. **Reorient to LPI** -- Standardizes orientation across all scanners to Left-Posterior-Inferior
2. **Resample to 0.5mm isotropic** -- B-spline interpolation for images, nearest-neighbor for segmentations
3. **Rician denoising** -- Correct noise model for MRI magnitude images (not Gaussian)
4. **N4 bias field correction** (`ants.abp_n4`) -- Removes B1 field inhomogeneity
5. **Percentile intensity standardization** -- Maps 1st-99th percentile to [0, 1]; clips outliers (implants, k-space spikes) without distorting tissue contrast
6. **Brain-masked histogram matching** *(optional, off by default)* -- Matches intensity distribution to MNI template within brain mask only, preventing skull/background from contaminating the match. Falls back to full-image matching if brain extraction fails. Modality (T1 or T2) is auto-detected from the `Study Type` field.
7. **Head masking** *(optional, off by default)* -- Zeros out voxels outside the dilated brain mask (25mm dilation by default). Uses HD-BET masks when available, falls back to Otsu intensity thresholding.

**Design decisions:**
- Percentile standardization was chosen over truncation + normalization because it preserves tissue contrast while handling outlier voxels (metallic implants, k-space artifacts).
- Brain-masked histogram matching prevents the image washout problem seen with full-image histogram matching, where background/skull intensities contaminate the match.
- Head masking uses a defense-in-depth pattern: the `ENABLE_HEAD_MASKING` flag is checked independently in `get_ants_image()`, `get_head_mask()`, and `apply_head_mask()` to prevent accidental application through any future code path.
- GPU is disabled for Stage 2 (`CUDA_VISIBLE_DEVICES=-1`); all processing is CPU-based. TensorFlow logging is suppressed.

### Registration Hierarchy

The three-level hierarchy minimizes error accumulation by separating patient-level, inter-timepoint, and intra-timepoint alignment:

```
Level 1: Patient Anchor ──────────────────────> MNI Template
                          tx_pt_anchor_to_template

Level 2: Timepoint Anchor ──> Patient Anchor ──> MNI Template
                tx_tp_anchor_to_pt_anchor  (composed with Level 1)

Level 3: Study ──> Timepoint Anchor ──> Patient Anchor ──> MNI Template
    tx_study_to_tp_anchor  (composed with Levels 1 & 2)
```

The full transform chain is applied in a single `apply_transforms()` call so each image is only interpolated once, minimizing blurring.

**Why hierarchical:** Direct registration of every image to the MNI template would accumulate alignment errors, especially for low-quality or cross-modality scans. By chaining through anchors, each registration step aligns images that are already close in physical space, producing better results.

**Anchor selection:** At each hierarchy level, the best anchor image is selected by sorting candidates on a configurable metric (default: `file_size_mb` descending). Larger files correlate with higher resolution and greater anatomical coverage -- the properties most important for an anchor. Alternatives: `entropy` (ascending) for simplest image, `voxel_volume` (descending) for largest spatial coverage.

**Retry logic:** If a registration or image load fails at any level, the next best candidate (by anchor metric) is tried up to 5 times. If all preferred-modality candidates fail, the search expands to all available modalities. Failures are logged to the output CSV and processing continues.

### Registration Methods

| Transform | DOF | Use | Rationale |
|---|---|---|---|
| `DenseRigid` | 6 (rotation + translation) | Initialization and intra-timepoint alignment | Dense sampling provides robust alignment for closely-spaced images |
| `Similarity` | 7 (rigid + isotropic scale) | Main inter-patient and inter-timepoint alignment | Isotropic scaling accounts for head size differences between patients |
| `Affine` | 12 | Available but not default | Risks anatomically implausible deformation for brain images |

**Two-step registration (Levels 1-2):**
1. DenseRigid initialization -- fast, coarse alignment to get close to the solution
2. Similarity refinement -- fine-tuned alignment with scaling allowed

**Single-step registration (Level 3):** DenseRigid only, since intra-timepoint studies already share physical space and only need rigid alignment.

**Affine metric selection:**
- Same modality (T2->T2, T1->T1): `GC` (Gradient Correlation) -- exploits matching gradient structure
- Cross-modality (T1->T2): `mattes` (Mutual Information) -- captures statistical intensity relationships

### Registration Parameters

```python
REGISTRATION_PARAMS = {
    'DenseRigid': {
        'aff_iterations': (100000, 5000, 2500, 1000),
        'aff_shrink_factors': (2, 2, 1, 1),
        'aff_smoothing_sigmas': (1.5, 1, 0.5, 0),
        'aff_sampling': 128,
        'aff_random_sampling_rate': 0.8,
        'grad_step': 0.1,
    },
    # Similarity and Affine use the same parameter structure
}
```

- **Iterations:** Multi-resolution pyramid with heavy iterations at coarse levels (100k) tapering to fine levels (1k)
- **Shrink factors:** 2x downsampled for coarse levels, full resolution for fine levels
- **Smoothing sigmas:** Progressive reduction from 1.5 to 0 (no smoothing at finest level)
- **Sampling:** 128 voxels sampled per iteration with 80% random sampling rate
- **Gradient step:** 0.1 for stable convergence

### Center Alignment

```python
CENTER_ALIGNMENT_CONFIG = {
    'pt_anchor_to_template':  True,   # Images may be far from template space
    'tp_anchor_to_pt_anchor': False,  # Same patient, usually close
    'study_to_tp_anchor':     False,  # Same timepoint, share physical space
}
```

When enabled, performs rigid registration on brain extraction masks before the main registration. This prevents the optimizer from getting stuck when images start far apart in physical space (common for patient-to-template alignment). Disabled for intra-patient and intra-timepoint levels where images are already in similar physical space.

Center alignment is configured per-level via `CENTER_ALIGNMENT_CONFIG` in `VS_preprocessing.final.py`.

### Image Validation & Error Handling

- **`fix_zero_spacing()`:** Detects and repairs invalid NIfTI headers with zero, NaN, Inf, or negative spacing values. Uses median of valid dimensions as replacement; defaults to 1.0mm if all dimensions are invalid.
- **`validate_image_for_registration()`:** Pre-registration check for None images, invalid spacing, NaN/Inf voxels. Replaces NaN with 0, clips Inf values.
- **`validate_warped_image()`:** Post-registration check that the output has valid spacing and non-zero dimensions.
- **Error logging:** Errors are written directly into the output CSV rows (`error_details`, `error_timestamp`) rather than a separate file, so metadata and processing status stay co-located for downstream filtering.

### Configuration Reference

#### Core Settings

| Variable | Default | Description |
|---|---|---|
| `SEED` | `42` | Global random seed (numpy, random, TensorFlow) |
| `THREADS` | `8` | ITK/ANTs threads per worker |
| `PARALLEL_WORKERS` | `8` | Patient worker processes (total cores = WORKERS x THREADS) |
| `TEST_MODE` | `False` | Dry run -- skips all I/O and registration |
| `PREPROCESSING_VERSION` | `"6.7"` | Version tag written per row; change when preprocessing changes |
| `MAX_PATIENTS` | `None` | Integer to limit patients processed (for testing) |
| `INCREMENTAL_MODE` | `"force_reprocess"` | Controls skipping behavior for existing outputs |
| `REUSE_TRANSFORMATIONS` | `True` | Reuse existing transform files when re-preprocessing |
| `TARGET_SPACING` | `(0.5, 0.5, 0.5)` | Target voxel spacing in mm |
| `ALLOWED_STUDY_TYPES` | `['t2_thin', 't1+_thin', 't1+_thick_ax', 't1+_thick_cor']` | Study types to process |
| `TEMPLATE_Z_CROP` | `(0, 140)` | Z-axis crop range for MNI template; `None` to disable |
| `PREFER_COREGISTERED` | `False` | Use previously coregistered images as anchors when resuming |
| `EXPERIMENT_SUFFIX` | `_<date>_V<version>` | Auto-generated suffix for output directories |

#### Preprocessing Toggles

| Variable | Default | Description |
|---|---|---|
| `ENABLE_HISTOGRAM_MATCHING` | `False` | Brain-masked histogram matching to template |
| `HISTOGRAM_MATCHING_BINS` | `1024` | Number of histogram bins for matching |
| `HISTOGRAM_MATCHING_POINTS` | `256` | Number of quantile points for matching |
| `ENABLE_HEAD_MASKING` | `False` | Zero out background outside head region |
| `HEAD_MASK_DILATION_MM` | `25.0` | Dilation beyond brain mask to include skull |
| `HEAD_MASK_MIN_COVERAGE` | `5.0` | Minimum mask coverage percentage to consider valid |

#### Incremental Processing Modes

| Mode | Behavior |
|---|---|
| `skip_all` | Skip patient if any study is already coregistered |
| `process_new` | Only process studies without a `coregistered_image` value |
| `smart` | Process new studies + re-process those with an outdated `preprocessing_version` |
| `force_reprocess` | Re-process everything regardless of existing outputs |

`REUSE_TRANSFORMATIONS = True` (default): under `smart` or `force_reprocess`, existing transform files are reused rather than recomputed when only the preprocessing step changed.

### Stage 2 Outputs

**Output CSV** -- All original columns plus:

| New Column | Description |
|---|---|
| `coregistered_image` | Path to coregistered image in template space |
| `coregistered_segmentation` | Path to coregistered segmentation |
| `coregistered_overlay_plot` | Path to QC overlay PNG (axial/sagittal/coronal) |
| `preprocessing_version` | Version string at time of processing |
| `mi_after_registration` | Final mutual information vs. template |
| `anchor` | `'patient_anchor'`, `'timepoint_anchor'`, or `None` |
| `error_details` | Full traceback if processing failed |
| `error_timestamp` | ISO timestamp of failure |

Coregistered files mirror the source directory structure with `_coregistered` appended to filenames.

---

## Stage 3: Post-Registration QC Sorting

**Source:** `src/sort_registrations.py` | **Entry point:** `scripts/run_sort.py`

After Stage 2 completes, QC overlay plots are manually reviewed and registration quality is classified in the output CSV. This script reads the reviewed CSV and sorts coregistered files into `acceptable/` and `manual_correction/` directories.

### Propagation Rules

The key design decision is hierarchical propagation of QC failures based on anchor role:

1. **Patient anchor flagged** (`Registration Classification = 1`, `anchor = patient_anchor`): Every study for that patient is moved to `manual_correction/`, because the patient-level transform that all other timepoints depend on is unreliable.

2. **Timepoint anchor flagged** (`anchor = timepoint_anchor`): Every study at that timepoint is moved to `manual_correction/`, because the timepoint-level transform that other studies at that timepoint depend on is unreliable.

3. **No anchor flagged** (individual study): Only that specific row's files are moved to `manual_correction/`.

This mirrors the registration hierarchy: if an anchor registration fails, all downstream registrations that depend on it are also suspect.

### What Gets Moved

For each row, three files are moved (if they exist):
- Coregistered image (`coregistered_image` column)
- Coregistered segmentation (`coregistered_segmentation` column)
- QC overlay plot (`coregistered_overlay_plot` column)

Directory structure within `acceptable/` and `manual_correction/` mirrors the original layout. Empty directories are cleaned up after sorting.

### Usage

```bash
# Via entry point (recommended)
python scripts/run_sort.py --config configs/default.yaml --csv path/to/reviewed.csv

# Direct execution (legacy)
python src/sort_registrations.py --base-dir /path/to/output --csv path/to/reviewed.csv
```

The script prints a detailed propagation report showing:
- Original classified rows and their anchor types
- Number of patients and timepoints propagated
- Per-study-type breakdown of acceptable vs. manual correction
- Statistics on patients requiring manual correction

---

## Configuration

All pipeline stages are configured through `configs/default.yaml`. The entry points in `scripts/` load the config, create a reproducibility run folder via `shared.run_logger.init_run()`, then call the corresponding `src/` module's `_apply_config(config)` function to override module-level globals before running `main()`.

```
configs/default.yaml  →  scripts/run_*.py  →  src/*.py._apply_config(config)  →  src/*.py.main()
                          └─ init_run() creates code_runs/<project>/<experiment>/<runid>/
```

Each run folder captures: config snapshot, git commit hash, conda environment freeze, and timestamps.

---

## End-to-End Execution

```bash
# 1. Run Stage 1 notebook (cells 1-22)
jupyter notebook "VS data timeline.preprocessing_final.ipynb"

# 2. Validate Stage 1 output
head -5 meta_df_entropy.csv
ls hd_output/ | wc -l   # Should match number of images

# 3. Run HD-BET brain extraction (if not done in notebook)
python scripts/run_hd_bet.py --config configs/default.yaml

# 4. Test Stage 2 on a subset (set max_patients: 5, test_mode: true in config)
python scripts/run_preprocessing.py --config configs/default.yaml

# 5. Full run (set max_patients: null, test_mode: false in config)
python scripts/run_preprocessing.py --config configs/default.yaml

# 6. Check errors
python -c "
import pandas as pd, glob
f = glob.glob('df_coregistered_*.csv')[0]
df = pd.read_csv(f)
errs = df[df['error_details'].notna()]
print(f'{len(errs)}/{len(df)} rows errored')
print(errs[['Reworked Patient ID','Study Type','error_details']].head())
"

# 7. Run Dice QC (notebook cells 25-30)

# 8. Review QC overlays, annotate CSV, then sort
python scripts/run_sort.py --config configs/default.yaml --csv path/to/reviewed.csv
```

---

## Key Design Decisions

### Why hierarchical registration instead of direct-to-template
Direct registration of every image to MNI template would accumulate errors, especially for low-quality or cross-modality scans. The three-level hierarchy ensures each registration step aligns images that are already close in physical space, improving robustness.

### Why single-interpolation transform composition
The full transform chain (Study -> TP Anchor -> Patient Anchor -> Template) is composed and applied in one `apply_transforms()` call. This avoids repeated interpolation that would progressively blur the image. The trade-off is that intermediate coregistered images at each hierarchy level are not saved -- only the final template-space outputs are written.

### Why Similarity over Affine for inter-patient registration
Similarity transforms (7 DOF: rigid + isotropic scaling) account for head size variation between patients without risking the anatomically implausible shearing that full Affine transforms (12 DOF) can introduce. DenseRigid (6 DOF) is sufficient for intra-timepoint alignment where head size is constant.

### Why file size as the default anchor selection metric
Larger MRI files generally correspond to higher resolution and greater anatomical coverage -- both critical properties for an anchor image that all other images at that level will be registered to.

### Why percentile standardization before histogram matching
Percentile clipping (1st-99th -> [0,1]) handles outlier voxels (metallic implants, k-space artifacts) before histogram matching runs, preventing these outliers from skewing the intensity distribution match.

### Why brain-masked histogram matching
Full-image histogram matching includes background air and skull in the intensity distribution, which can wash out brain tissue contrast. Matching only within the brain mask preserves clinically relevant contrast.

### Why propagation-based QC sorting
If a patient anchor registration is bad, every downstream transform that chains through it is unreliable. Propagating the failure flag respects the dependency structure of the registration hierarchy rather than treating each study independently.

---

## Known Limitations

- **File format:** Only `.nii.gz` images are supported. DICOM files must be converted upstream (e.g., via `dcm2niix`).
- **Segmentation naming convention:** The segmentation discovery logic specifically looks for `uvauser` in the filename. Other datasets require updating `find_nifti_files()` in Cell 10.
- **Storage paths:** `root_dir` in Cell 1 and the Carina Excel path in Cell 6 are hard-coded to UVA infrastructure paths.
- **GPU for HD-BET:** Brain extraction in Stage 1 requires a CUDA GPU. CPU fallback exists but is substantially slower. Stage 2 does not require a GPU.
- **Memory scaling:** Each parallel worker loads all images for one patient into memory simultaneously. Very large patient series may require reducing `PARALLEL_WORKERS` (rule of thumb: `RAM_GB / 16`).
- **Single-interpolation design:** Intermediate coregistered images at each hierarchy level are not saved to disk -- only final template-space outputs are written.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `"No valid points in common"` | Images don't overlap in physical space | Ensure `CENTER_ALIGNMENT_CONFIG['pt_anchor_to_template'] = True` |
| `"Invalid spacing"` / zero spacing | Scanner wrote invalid NIfTI header | Handled automatically by `fix_zero_spacing()`; check logs for `[CRITICAL]` |
| HD-BET mask not found | Mask file missing or wrong filename | Re-run HD-BET cells; pipeline falls back to `antspynet.brain_extraction()` automatically |
| Memory errors in parallel mode | Each worker loads full patient image stacks | Reduce `PARALLEL_WORKERS` |
| High error count in output CSV | Various | Inspect `error_details` column; common: `"Registration failed"` (review QC overlays), `"file not found"` (verify `raw_studies/` paths) |
| NaN/Inf in output images | Corrupt input or failed interpolation | `validate_image_for_registration()` auto-replaces NaN with 0 and clips Inf |
