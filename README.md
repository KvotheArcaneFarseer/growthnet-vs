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

## Repository Structure
