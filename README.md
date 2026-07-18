
# GrowthNet-VS: Synthetic Vestibular Schwannoma Generation Pipeline

A production-grade pipeline for generating and embedding **synthetic vestibular schwannoma (VS) tumors** into MRI scans using anatomically realistic geometry and deterministic modeling.

---

## Overview

This project builds a **synthetic tumor generation system** designed to:

- Model realistic **vestibular schwannoma growth** (IAC → CPA “lollipop” geometry)
- Embed tumors into real MRI volumes
- Generate reproducible datasets for research and machine learning
- Run scalable batch processing on HPC systems (e.g., Rivanna)

---

## Key Features

### Anatomically Realistic Geometry
- Implements **lollipop structure**:
  - Narrow intracanalicular (IAC) stem
  - Expanding CPA bulb
- Convex taper + controlled growth dynamics
- Smooth low-frequency perturbations (no voxel noise)

---

### Deterministic Generation
- Seeded using:
  - Explicit seed OR
  - Hash of MRI + segmentation path
- Guarantees:
  - Reproducibility across runs
  - No frame-to-frame randomness

---

### Synthetic Tumor Embedding
- Inserts tumors into MRI volumes using:
  - Distance-transform-based blending
  - Spatial alignment via segmentation centroid + principal axis
- Produces:
  - Volume + mask pairs
  - Multi-timepoint tumor growth (t0 → t4)

---

### Batch Processing Pipeline
- CSV-driven batch execution
- Generates:
  - Per-case outputs
  - QC visualizations
  - Aggregated metrics

Example metrics:
- centroid offset
- retained fraction
- orientation confidence
- clipping frequency

---

### HPC-Ready
- Designed for scalable execution on clusters (e.g., Rivanna)
- Supports:
  - multi-case processing
  - reproducible dataset generation

---

## Repo-structure
GrowthNet/
│
├── embed_tumor.py # Main tumor embedding pipeline
├── view_napari.py # Visualization tool (Napari)
├── make_lollipop_animation.py # Tumor growth animation generator
├── make_lollipop_napari.py # Interactive visualization
│
├── scripts/
│ ├── run_batch_embedding.py # Batch processing entrypoint
│ ├── export_graphify_obsidian.py # Graphify export utilities
│ └── export_graphify_architecture_obsidian.py
│
├── projects/vivit/src/
│ ├── data/
│ │ └── synthetic.py # Core tumor generation logic
│ │
│ ├── networks/
│ │ ├── t_unetr.py # Model architecture (UNETR variant)
│ │ └── vitautoenc.py # Vision transformer autoencoder
│ │
│ └── train/
│ ├── grid_search.py
│ ├── pretrain_ops.py
│ ├── train_ops.py
│ └── utils.py
│
├── docs/
│ └── ai_workflow/
│ ├── AI_AGENTS.md # Agent system design (Claude + Codex)
│ └── CLAUDE_WORKFLOW.md # Development workflow documentation
│
├── README.md
└── .gitignore
=======
# GrowthNet

GrowthNet is the local research codebase for synthetic vestibular schwannoma growth data generation, embedding, validation, and downstream temporal model experiments.

This repository currently contains a working local synthetic tumor embedding prototype, local batch and longitudinal wrappers, morphology and orientation audit artifacts, and early ViViT/TemporalUNETR model code. It is not yet a production clinical dataset pipeline or a publication-ready training dataset.

## Current Local Status

As of 2026-07-18:

- Implemented locally: single-case synthetic lollipop embedding in `embed_tumor.py`, standalone synthetic mask cohort generation, real/synthetic feature extraction, CSV batch execution, local longitudinal MVP orchestration, QC/metrics output, and focused local tests.
- Partially complete: local batch reliability, longitudinal metadata/QC coverage, volume-targeting operating-range evidence, morphology validation, and documentation.
- Experimental: lollipop morphology benchmarking, orientation diagnostics, volume-targeting benchmarks, Napari/animation helpers, lab meeting export scripts, and Graphify exports.
- Unvalidated locally: population-level clinical claims, full curated longitudinal cohort generation, ViViT training on generated longitudinal data, and any result requiring unavailable original clinical MRI/segmentation files.
- Remote-data-dependent: multi-patient clinical validation, independent re-extraction of pulled real feature tables, clinical/anatomical canal-vs-CPA labeling, and HPC-scale training.

For the detailed local status, see `docs/CURRENT_PROJECT_STATUS.md`.

## Main Local Entry Points

| Purpose | Entry point | Notes |
| --- | --- | --- |
| Single-case embedding | `embed_tumor.py` | Requires local MRI and segmentation NIfTI files. Defaults still reference a private Downloads example; pass explicit paths. |
| Batch embedding | `scripts/run_batch_embedding.py` | CSV input with `case_id,mri_path,seg_path`; supports `--resume`. |
| Synthetic lollipop cohort | `scripts/generate_synthetic_lollipop_cohort.py` | Generates standalone masks from target volumes. |
| Real/synthetic feature extraction | `scripts/extract_real_tumor_features.py` | Extracts spacing-aware morphology metrics from NIfTI masks. |
| Longitudinal MVP | `scripts/generate_synthetic_longitudinal_dataset.py` | Four fixed visits `T1`-`T4`; consumes explicit target volumes and local background manifest. |
| Morphology audit | `analysis/real_vs_synthetic/analyze_local_morphology.py` | Uses local pulled feature artifacts and local synthetic masks. |
| Orientation audit | `analysis/orientation_validation/orientation_diagnostic.py` | Local diagnostic; cannot validate missing real cases. |
| Volume benchmark | `analysis/volume_targeting/run_volume_targeting_benchmark.py` | Local benchmark helper; available outputs include smoke CSV and plots. |

## Local Reproducibility

Start with:

```bash
python3 -m py_compile \
  embed_tumor.py \
  scripts/run_batch_embedding.py \
  scripts/extract_real_tumor_features.py \
  scripts/generate_synthetic_lollipop_cohort.py \
  scripts/generate_synthetic_longitudinal_dataset.py \
  projects/vivit/src/data/synthetic.py
```

If `pytest` is installed:

```bash
python3 -m pytest tests -q
```

The active environment used during the local audit did not have `pytest` installed, so pytest results must be verified in an environment with the test dependency available.

See `docs/LOCAL_REPRODUCIBILITY.md` for local-safe command examples.

## Project Map

- `embed_tumor.py`: monolithic but operational embedding pipeline.
- `scripts/`: local command-line wrappers and analysis utilities.
- `projects/vivit/`: synthetic data helpers, TemporalUNETR/ViViT model code, and early training experiments.
- `projects/mri_registration/`: separate MRI preprocessing/coregistration project with its own dependencies and HPC-oriented docs.
- `analysis/`: local audit outputs, plots, and helper scripts generated by current validation work.
- `rivanna_pull/`: local copies of prior feature artifacts and scripts; original remote source masks are not present.
- `docs/`: project status, action plans, audit notes, limitations, and reproducibility guidance.
- `tests/`: focused local pytest tests for helpers, schemas, deterministic generation, and selected integration paths.

## Important Boundaries

- Do not delete generated data or provenance artifacts without review.
- Do not modify core lollipop geometry, spacing-aware transform logic, or orientation semantics without evidence and scientific review.
- Do not treat pulled CSV/JSON summaries as independent local clinical validation of source masks.
- Do not assume Rivanna, SLURM, remote datasets, or private Downloads files are available.

Known limitations are tracked in `docs/KNOWN_LIMITATIONS.md`.
(Add GrowthNet local roadmap, validation audits, and reliability updates)
