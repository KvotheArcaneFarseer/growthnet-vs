# Current Project Status

Last updated: 2026-07-18

Scope: local repository state only. No Rivanna, SSH, SLURM, remote datasets, or private clinical data were assumed for this status.

## Maturity

GrowthNet is a research-prototype-plus repository. The local embedding and synthetic generation pipeline is operational, and several validation audits now exist. The project is not yet a fully validated clinical synthetic dataset factory, not yet a reproducible training dataset release, and not yet publication-ready.

## Implemented

- `embed_tumor.py`: single-case synthetic vestibular schwannoma embedding with lollipop geometry, spacing-aware placement, orientation selection, QC images, NIfTI outputs, and metrics reports.
- `projects/vivit/src/data/synthetic.py`: synthetic time-series generator with lollipop and legacy geometry support.
- `scripts/run_batch_embedding.py`: CSV batch runner with local resource thread caps, per-case summaries, failure reports, and resume support.
- `scripts/generate_synthetic_lollipop_cohort.py`: standalone target-volume synthetic mask generation.
- `scripts/extract_real_tumor_features.py`: spacing-aware mask feature extraction and summary generation.
- `scripts/generate_synthetic_longitudinal_dataset.py`: MVP four-visit longitudinal wrapper over the embedding engine.
- `analysis/orientation_validation/`: local orientation diagnostic artifacts.
- `analysis/real_vs_synthetic/`: local matched morphology benchmark, plots, and report.
- `analysis/volume_targeting/`: local volume-targeting benchmark helper and smoke artifacts.
- `tests/`: focused local pytest tests for embedding helpers, batch helpers, synthetic generation, feature extraction, and longitudinal helpers.

## Partially Complete

- Local batch pipeline reliability: resume/status behavior exists, but broader multi-case local validation depends on available local NIfTI inputs.
- Longitudinal dataset generation: metadata and QC outputs exist, but sparse visits, elapsed days, source background provenance, longitudinal drift QC, and ViViT export are missing.
- Volume targeting: implementation exists, and smoke benchmark artifacts exist, but recommended production thresholds still require broader review.
- Morphology validation: local comparison is complete for available pulled artifacts, but source real-mask re-extraction is unavailable.
- ViViT/TemporalUNETR integration: model and data loader code exists, but generated longitudinal data has not been proven through a local training run.
- Documentation: this local status layer is current, while several project sub-READMEs remain historical templates.

## Experimental

- Orientation diagnosis for large-tumor axis errors.
- Lollipop morphology calibration and benchmarking.
- Volume-targeting parameter benchmarking.
- Napari visualization and animation scripts.
- Lab meeting deck/3D export scripts.
- Graphify/Obsidian architecture exports.
- MRI registration notebooks and HPC run scripts.

## Locally Validated Evidence

- Core script syntax/import smoke was reported passing for embedding, batch, feature extraction, synthetic cohort generation, longitudinal generation, and synthetic data helper modules.
- Longitudinal helper behavior has deterministic local unit tests from `tests/test_longitudinal_dataset_audit.py`; a unittest run passed for those audit tests. Pytest was not installed in the active environment.
- Orientation audit found that local standalone masks for `126_5_1607` and `132_1_148` align their whole-mask major PCA axis with the manifest-derived canal line. The missing patient-MRI embedding data prevents spatial bug confirmation for those named cases.
- Morphology audit compared 261 matched local synthetic benchmark rows against pulled real feature rows. Median synthetic/real volume ratio was close, while elongation, bounding-box fill, compactness, and major-axis length showed substantial gaps.
- Batch runner local smoke artifacts exist under `analysis/batch_reliability/`.

## Remotely Validated Or Pulled Provenance

The repository includes `rivanna_pull/` artifacts, including real feature summaries and synthetic benchmark feature tables. These are local copies of prior outputs, not fresh local re-extractions from original real source masks. Treat them as provenance/evidence artifacts, not as proof that this laptop contains the full source dataset.

## Unvalidated

- Full real multi-patient embedding robustness.
- Original-source real feature extraction equivalence.
- Clinical validity of canal/CPA morphology and orientation metrics.
- Longitudinal clinical realism.
- Training-ready dataset generation at scale.
- Local ViViT training or inference on generated longitudinal outputs.
- Publication-ready reproducibility for all figures/tables.

## Current Working Tree Notes

The working tree is intentionally active and contains source edits, tests, documentation, analysis scripts, generated plots, and pulled artifacts. Do not assume `git status` is clean. Avoid destructive cleanup. Review file ownership in `docs/GROWTHNET_ACTION_PLAN.md` before editing.

Known modified or newly added areas during the local orchestration include:

- `.gitignore`
- `embed_tumor.py`
- `projects/vivit/src/data/synthetic.py`
- `scripts/run_batch_embedding.py`
- `scripts/generate_synthetic_longitudinal_dataset.py`
- `scripts/generate_synthetic_lollipop_cohort.py`
- `scripts/extract_real_tumor_features.py`
- `tests/`
- `analysis/`
- `docs/`
- `data/timelines/`

## Recommended Local Priorities

1. Finish review of specialist outputs in `docs/GROWTHNET_AGENT_STATUS.md`.
2. Run the focused pytest suite in an environment with `pytest` and required scientific Python packages.
3. Keep documenting local analysis results without promoting them to clinical claims.
4. Add missing reproducibility/environment specifications.
5. Expand local integration tests using synthetic or explicitly supplied local NIfTI fixtures.
6. Defer generator morphology tuning until orientation and real-vs-synthetic findings are reviewed.
7. Mark any work needing unavailable clinical files as `BLOCKED_REMOTE_DATA`.
