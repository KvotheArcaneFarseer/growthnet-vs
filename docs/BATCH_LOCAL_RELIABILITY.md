# Batch Local Reliability

Last updated: 2026-07-18

Scope: local laptop execution only. This runner must not assume Rivanna, SLURM, SSH, or remote datasets.

## Runner

Batch entry point:

```bash
python3 scripts/run_batch_embedding.py \
  --input_csv <cases.csv> \
  --out_dir <batch_output_dir> \
  --num_cases 1
```

The heavy embedding stack is imported lazily only when a case actually needs to run. `--help` and all-skipped `--resume` runs therefore avoid Matplotlib/font-cache initialization.

Required CSV columns:

- `case_id`
- `mri_path`
- `seg_path`

Optional local tuning columns currently accepted by `scripts/run_batch_embedding.py`:

- `canal_growth_scale`
- `bulb_growth_scale`
- `t0_volume_fraction_of_seg`
- `target_tumor_volume_mm3`
- `volume_target_timepoint`
- `volume_ravd_tolerance`
- `volume_max_iterations`

## Resource Controls

The runner now sets conservative BLAS/thread defaults before importing NumPy:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

User-provided environment values are preserved. For explicit local runs, prefer:

```bash
MPLCONFIGDIR=/tmp/growthnet_mpl \
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 scripts/run_batch_embedding.py \
  --input_csv analysis/batch_reliability/local_batch_smoke_cases.csv \
  --out_dir analysis/batch_reliability/local_smoke_outputs \
  --num_cases 1
```

Observed local warning: Fontconfig may still print unwritable cache messages for Homebrew/user font cache directories. The warning did not stop the smoke run. `MPLCONFIGDIR=/tmp/growthnet_mpl` avoids Matplotlib's Python cache warning but does not fully control Fontconfig.

## Resume Behavior

Use `--resume` to avoid recomputing completed valid cases:

```bash
python3 scripts/run_batch_embedding.py \
  --input_csv <cases.csv> \
  --out_dir <existing_batch_output_dir> \
  --resume
```

A case is considered resumable only when all required case outputs exist and `embedding_metrics.json` parses with no `hard_failures`:

- `embedding_metrics.json`
- `embedding_metrics.csv`
- `embedded_tumor_volume.nii.gz`
- `embedded_tumor_mask.nii.gz`
- `embedded_tumor_late_volume.nii.gz`
- `embedded_tumor_late_mask.nii.gz`
- `qc_embedding.png`
- `qc_embedding_late.png`

Default behavior is backward compatible: without `--resume`, the runner recomputes cases.

## Per-Case Status Schema

Each batch now appends JSON Lines to:

```text
<out_dir>/batch_case_status.jsonl
```

Status event fields:

- `timestamp_utc`: ISO-8601 UTC timestamp.
- `case_id`: original CSV case ID.
- `case_out_dir`: sanitized output directory.
- `status`: one of `started`, `completed`, `exception`, `skipped_existing`.
- `index`: 1-based case index in the selected batch.
- `total_cases`: number of cases after `--num_cases` truncation.
- `warning_count`: present for completed runs.
- `hard_failure_count`: present for completed runs.
- `exception_type`: present for exceptions.
- `exception_message`: present for exceptions.

The status log is append-only. A resumed or repeated run may contain multiple events for the same case.

## Local Reliability Findings

- The local one-case smoke ran the full batch path and wrote batch summary, failure report, case metrics, NIfTI outputs, QC PNGs, and status JSONL.
- The local smoke input used already generated repository outputs as stand-ins, not a real original MRI/segmentation pair. It is valid for runner reliability but not for scientific validation.
- The smoke completed operationally but produced one hard scientific validation failure: `placed_to_seg_ratio_fail`.
- The resume probe skipped a known completed local case without opening dummy input paths, confirming recoverability for complete valid outputs.
- No Rivanna-specific behavior was added.

## Limitations

- This does not prove population-scale batch reliability.
- This does not validate tumor morphology or orientation scientifically.
- Failed cases with `hard_failures` are intentionally not skipped by `--resume`.
- The runner still executes cases serially; no local parallel scheduler was added.
- GitNexus MCP impact tools required by project policy were not exposed in this session, so source edits were validated by local inspection, compile checks, and smoke runs instead.
