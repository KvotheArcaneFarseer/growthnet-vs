# GrowthNet Dependency Inventory

Last updated: 2026-08-04

Scope: local repository import scan across `embed_tumor.py`, `scripts/`,
`analysis/`, `projects/`, `tests/`, and `shared/`.

## Purpose

This inventory supports Milestone A in
`docs/ARCHITECTURE_REMEDIATION_PLAN.md`: create a reproducible local dependency
baseline without bundling generated data or remote/HPC assumptions.

## Core Runtime Dependencies

These are used by the local embedding, synthetic generation, feature extraction,
and validation paths:

| Package | Evidence |
|---|---|
| `numpy` | imported by 36 files |
| `scipy` | imported by 15 files |
| `nibabel` | imported by 15 files |
| `matplotlib` | imported by 11 files |
| `pandas` | imported by 13 files |
| `scikit-image` | imported as `skimage` by 3 files |
| `monai` | imported by 17 files |
| `torch` | imported by 12 files |
| `tqdm` | imported by 2 files |
| `Pillow` | imported as `PIL` by 1 file |
| `PyYAML` | imported as `yaml` by 1 file |

## Development Dependency

| Package | Evidence |
|---|---|
| `pytest` | imported by 9 test files; `.venv` has pytest 8.4.2 |

## Optional Heavy Dependencies

These should not be required for the laptop-safe fast test path:

| Extra | Packages | Evidence |
|---|---|---|
| `registration` | `antspyx`, `antspynet`, `tensorflow` | imports `ants`, `antspynet`, and `tensorflow` in MRI registration code |
| `training` | `accelerate` | imported by 3 ViViT/training files |
| `presentations` | `lxml`, `python-pptx` | imported by export/deck tooling |

## Manifest Decision

The root `pyproject.toml` uses:

- default dependencies for the fast local scientific Python stack,
- `.[dev]` for pytest,
- optional extras for registration, training, and presentation/export helpers.

Validation:

```bash
.venv/bin/python -m pip install -e ".[dev]" --dry-run
```

Result on 2026-08-04:

- editable install metadata prepared successfully,
- most dependencies were already satisfied in `.venv`,
- `PyYAML` was missing and was installed into `.venv` as `PyYAML 6.0.3`,
- no source or generated artifacts were modified.
