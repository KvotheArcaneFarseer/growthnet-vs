# Known Limitations

Last updated: 2026-07-18

This document separates engineering limitations from scientific limitations. A command that runs successfully is not, by itself, scientific validation.

## Scientific Limitations

- Local data is insufficient for population-level clinical claims. The repository has pulled feature artifacts and local synthetic masks, but not the original full real clinical segmentation corpus.
- The named large-tumor orientation cases `126_5_1607` and `132_1_148` have local standalone synthetic masks and feature rows, but their original local real MRI/segmentation and embedded patient-MRI outputs are not available. Spatial bug claims for those cases are `BLOCKED_REMOTE_DATA`.
- Whole-mask PCA can be a limited orientation target for lollipop geometry. A large CPA bulb or morphology change can alter the principal-axis interpretation even when the canal/stem direction is plausible.
- Canal-vs-CPA compartment validation is not available locally. No atlas landmark, IAC/CPA label, or clinically reviewed porus boundary is present.
- The local morphology audit shows synthetic masks are more elongated and less bounding-box-filled than matched real feature rows. This is evidence of a morphology gap in the available benchmark, not proof that the anatomy is wrong.
- Surface and sphericity comparisons may be affected by voxel resolution and extraction differences. Real feature rows and synthetic benchmark masks are not fully re-extracted from original source masks locally.
- The longitudinal wrapper is not a clinical growth model. It consumes explicit target volumes and maps `stable`/`growing` labels to existing generator modes for orchestration only.
- ViViT/TemporalUNETR training code exists, but a local training-ready longitudinal dataset and successful local training run have not been demonstrated in this audit.

## Engineering Limitations

- There is no root dependency lockfile or root environment file for the embedding/generation tooling.
- `pytest` was unavailable in the active audit environment, so full test execution needs a suitable local environment.
- `embed_tumor.py` is a large monolithic module with many responsibilities: geometry generation, orientation selection, embedding, validation, QC figures, and reporting.
- Several scripts use repo-root path insertion rather than an installed package.
- Some CLI defaults still reference private local paths. Reproducible runs should always pass explicit input and output paths.
- Generated outputs, local analysis artifacts, pulled provenance, and source files coexist in the working tree. This is useful for auditability but raises repository hygiene and large-file risk.
- MRI registration is a separate heavy pipeline with ANTs/ANTsPyNet/TensorFlow dependencies and HPC-oriented documentation. It is not validated as part of the local embedding audit.
- Some project subdirectory READMEs remain placeholders or Rivanna-first templates.

## Current Local Validation Gaps

- Multi-patient embedding validation using original real MRI/segmentation pairs.
- Independent re-extraction of real morphology features from original source masks.
- Anisotropic-spacing stress tests across a representative clinical cohort.
- Local end-to-end longitudinal generation using real local background NIfTI manifests.
- ViViT-compatible export validation and training smoke on generated longitudinal data.
- Human scientific review of acceptable orientation metrics and `stable`/`growing` label semantics.
- Final publication provenance linking every claimed figure/table to commit, config, seed, command, and source data.

## Remote-Data Follow-Up

The following should be marked `BLOCKED_REMOTE_DATA` until the required files are intentionally made available locally or run in a separate remote workflow:

- Original real MRI/segmentation files for unavailable orientation cases.
- Full curated real segmentation corpus for independent feature extraction.
- Longitudinal real background manifests for clinical-scale synthetic series generation.
- Clinical/anatomical labels or expert-reviewed landmarks for canal/CPA compartment validation.
- HPC-scale generated cohorts, checkpoints, and training logs.

Remote-data follow-up must not block local engineering work such as tests, command documentation, local audits, or wrapper reliability.

## Areas That Should Not Change Without Review

- The synthetic-space convention that one synthetic voxel represents one millimeter unless explicitly redesigned.
- Spacing-aware transform logic in the embedding path.
- Principal-axis voxel/physical-space conventions.
- Lollipop canal/bulb topology and sign conventions.
- Metric schema fields consumed by batch summaries and downstream reports.
- Existing generated data and provenance artifacts.
- Dataset label semantics for longitudinal `stable` and `growing` rows.
