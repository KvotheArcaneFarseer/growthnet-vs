# Local Batch Smoke Report

Task ID: BATCH-001

Date: 2026-07-18

Scope: local GrowthNet repository only. No SSH, Rivanna, SLURM, or remote datasets were used.

## Files Used

Smoke manifest:

```text
analysis/batch_reliability/local_batch_smoke_cases.csv
```

Smoke inputs:

```text
embedding_outputs/embedded_tumor_volume.nii.gz
embedding_outputs/embedded_tumor_mask.nii.gz
```

These are generated repository outputs reused as local stand-ins. They are not original clinical input data.

Smoke outputs:

```text
analysis/batch_reliability/local_smoke_outputs/
```

Resume probe manifest:

```text
analysis/batch_reliability/local_batch_resume_cases.csv
```

Resume probe outputs:

```text
analysis/batch_reliability/resume_probe_outputs/
```

Note: an initial setup copy placed one extra snapshot of the case-147 outputs at the root of `resume_probe_outputs/` before the expected `resume_probe_outputs/case_147/` directory was populated. It is an accidental setup artifact and was not deleted during this session.

## Commands Run

Syntax validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache \
python3 -m py_compile scripts/run_batch_embedding.py
```

Full local smoke:

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

Resume probe:

```bash
MPLCONFIGDIR=/tmp/growthnet_mpl \
PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python3 scripts/run_batch_embedding.py \
  --input_csv analysis/batch_reliability/local_batch_resume_cases.csv \
  --out_dir analysis/batch_reliability/resume_probe_outputs \
  --num_cases 1 \
  --resume
```

## Smoke Result

Operational result: completed.

Batch summary:

```text
total_cases: 1
completed_cases: 1
success_count: 0
warning_count: 1
hard_failure_count: 1
orientation_confidence.mean: 0.0007184940105658807
primary_axis_error_deg.mean: 20.352687664773338
ravd.mean: 0.12109375
volume_converged_count: 0
```

Case status events:

```json
{"case_id": "local_smoke_from_existing_embedding", "case_out_dir": "analysis/batch_reliability/local_smoke_outputs/local_smoke_from_existing_embedding", "index": 1, "status": "started", "total_cases": 1}
{"case_id": "local_smoke_from_existing_embedding", "case_out_dir": "analysis/batch_reliability/local_smoke_outputs/local_smoke_from_existing_embedding", "hard_failure_count": 1, "index": 1, "status": "completed", "total_cases": 1, "warning_count": 2}
```

Scientific validation result: not successful. The stand-in case produced `placed_to_seg_ratio_fail`. This is expected because the smoke used an already embedded local mask as the segmentation target. The run is useful for batch reliability only.

## Resume Result

Operational result: completed.

The runner printed:

```text
[1/1] Skipping completed case `case_147` -> analysis/batch_reliability/resume_probe_outputs/case_147
```

Resume summary:

```text
total_cases: 1
completed_cases: 1
success_count: 1
warning_count: 0
hard_failure_count: 0
orientation_confidence.mean: 0.20971068958584738
```

Resume status event:

```json
{"case_id": "case_147", "case_out_dir": "analysis/batch_reliability/resume_probe_outputs/case_147", "index": 1, "status": "skipped_existing", "total_cases": 1}
```

The resume command was run twice during final validation, so the JSONL file contains two `skipped_existing` events for `case_147`.

The resume manifest deliberately used unused dummy input paths. Because the completed output set was valid, the runner did not open those paths.

## Reliability Conclusions

- Local batch execution is auditable through `batch_summary.json`, `batch_summary.csv`, `failure_cases.json`, and `batch_case_status.jsonl`.
- Partial failures are recoverable at the batch level: exceptions are captured into summary rows and status events.
- Scientifically failed cases are not treated as valid resume targets.
- Completed valid cases can be skipped with `--resume`, avoiding recomputation.
- Conservative local BLAS thread defaults are applied unless the user provides explicit environment values.

## Unresolved Risks

- The smoke did not use a clean original MRI/segmentation fixture.
- Fontconfig cache warnings remain noisy on this laptop.
- No full cohort was run, by instruction.
- No remote-data reliability claim is supported by this report.
