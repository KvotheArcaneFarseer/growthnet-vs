# Volume Targeting Validation Report

Task: local standalone volume-targeting benchmark
Scope: standalone lollipop mask generator only
Run date: 2026-07-18
Repository state: local worktree; no source behavior modified

## Benchmark Setup

I inspected `analysis/volume_targeting/run_volume_targeting_benchmark.py` before
running it. The script imports generator helpers from
`scripts/generate_synthetic_lollipop_cohort.py`, runs local calibration cases,
writes a CSV via `--out_csv`, and writes two plots beside that CSV:

- `requested_vs_achieved.png`
- `ravd_by_target_volume.png`

Confirmed CLI arguments:

- `--out_csv`
- `--targets`
- `--seeds`
- `--spacings`
- `--tolerance_frac`
- `--max_iters`
- `--min_scale_vox`
- `--max_scale_vox`

Command run:

```bash
MPLCONFIGDIR=/tmp/growthnet_volume_targeting_mplconfig \
python3 analysis/volume_targeting/run_volume_targeting_benchmark.py \
  --targets 25,50,75,100,250,500,1000,4000,16000 \
  --seeds 20260426,20260427 \
  --spacings '1,1,1;0.6,0.6,1.2' \
  --out_csv analysis/volume_targeting/volume_targeting_benchmark.csv
```

The run completed successfully and reported that it wrote:

- `analysis/volume_targeting/volume_targeting_benchmark.csv`
- `analysis/volume_targeting/requested_vs_achieved.png`
- `analysis/volume_targeting/ravd_by_target_volume.png`

Matplotlib/fontconfig emitted cache-directory warnings during plotting, but the
benchmark process exited with code 0 and both plot writes were reported
successful. No benchmark case emitted generator warnings in the CSV.

## Overall Result

The benchmark generated 36 local calibration cases:

- targets: 25, 50, 75, 100, 250, 500, 1000, 4000, 16000 mm3
- seeds: 20260426 and 20260427
- spacings: 1.0 x 1.0 x 1.0 mm and 0.6 x 0.6 x 1.2 mm
- tolerance: 3% relative absolute volume difference (RAVD)
- max calibration iterations: 14

Summary:

- 36/36 cases completed with `status=OK`.
- 30/36 cases converged within 3% RAVD.
- Median RAVD: 1.64%.
- Mean RAVD: 2.14%.
- Maximum RAVD: 8.00%.
- Median absolute error: 3.00 mm3.
- Maximum absolute error: 316.00 mm3.
- No masks touched the generation-grid boundary.
- All masks were single connected components.
- No local scale-curve non-monotonicity was observed.
- Plateau behavior was flagged in 30/36 cases.

## Results By Target

| Target mm3 | Cases | Converged <=3% | Median RAVD | Max RAVD | Median abs error mm3 | Max abs error mm3 | Max iters | Plateau flags |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 4 | 1/4 | 5.34% | 8.00% | 1.34 | 2.00 | 14 | 4 |
| 50 | 4 | 3/4 | 2.18% | 6.00% | 1.09 | 3.00 | 14 | 4 |
| 75 | 4 | 2/4 | 3.99% | 6.67% | 3.00 | 5.00 | 14 | 4 |
| 100 | 4 | 4/4 | 0.33% | 1.95% | 0.33 | 1.95 | 6 | 4 |
| 250 | 4 | 4/4 | 1.12% | 2.71% | 2.81 | 6.78 | 6 | 4 |
| 500 | 4 | 4/4 | 1.64% | 2.60% | 8.19 | 13.00 | 6 | 4 |
| 1000 | 4 | 4/4 | 0.82% | 1.37% | 8.24 | 13.74 | 6 | 3 |
| 4000 | 4 | 4/4 | 1.92% | 2.49% | 76.82 | 99.68 | 6 | 1 |
| 16000 | 4 | 4/4 | 1.30% | 1.98% | 207.90 | 316.00 | 5 | 2 |

## Stratified Interpretation

### <100 mm3

Cases: 12
Converged within 3% RAVD: 6/12
Median RAVD: 3.33%
Mean RAVD: 4.02%
Maximum RAVD: 8.00%
Median absolute error: 1.57 mm3
Maximum absolute error: 5.00 mm3
Median iterations: 11.5
Maximum iterations: 14

Interpretation: this band is limited by voxel quantization and discrete mask
plateaus. The failed cases are small in absolute terms, but RAVD is unstable at
these targets. All <100 mm3 cases had plateau flags, no clipping, no component
failures, no warnings, and no local non-monotonicity. The calibration frequently
used the full 14 iterations.

### 100-1000 mm3

Cases: 16
Converged within 3% RAVD: 16/16
Median RAVD: 1.00%
Mean RAVD: 1.07%
Maximum RAVD: 2.71%
Median absolute error: 2.48 mm3
Maximum absolute error: 13.74 mm3
Median iterations: 6
Maximum iterations: 6

Interpretation: this is the strongest operating band in the representative
benchmark. Every case passed the 3% RAVD criterion, with no clipping, no
component failures, no warnings, and no local non-monotonicity. Plateau flags
still appeared in 15/16 diagnostic local scale probes, but they did not prevent
convergence.

### >1000 mm3

Cases: 8
Converged within 3% RAVD: 8/8
Median RAVD: 1.45%
Mean RAVD: 1.44%
Maximum RAVD: 2.49%
Median absolute error: 96.66 mm3
Maximum absolute error: 316.00 mm3
Median iterations: 5
Maximum iterations: 6

Interpretation: all large-volume cases passed 3% RAVD. Absolute errors are
larger because the requested volumes are larger, but relative targeting remained
inside the benchmark tolerance. No clipping, component failures, warnings, or
local non-monotonicity were observed. Plateau flags were uncommon here, 3/8
cases, and were not associated with failure.

## Spacing Notes

| Spacing mm | Cases | Converged <=3% | Median RAVD | Max RAVD |
|---|---:|---:|---:|---:|
| 1.0 x 1.0 x 1.0 | 18 | 13/18 | 1.55% | 8.00% |
| 0.6 x 0.6 x 1.2 | 18 | 17/18 | 1.76% | 6.69% |

The anisotropic spacing did not degrade this benchmark. Its smaller voxel volume
of 0.432 mm3 improved tiny-volume quantization relative to 1.0 mm3 voxels in
some cases.

## Clipping, Plateau, And Directionality

- Clipped to generation grid: 0/36.
- Component count not equal to 1: 0/36.
- Local scale-curve non-monotonicity: 0/36.
- Local scale-curve plateau flag: 30/36.
- Overshoot: 18/36.
- Undershoot: 16/36.
- Exact hit: 2/36.

The plateau flag is a diagnostic from a coarse local scale probe, not direct
evidence that final calibration failed. In this run it mainly reflects discrete
voxelization at small scales. It is most relevant below 100 mm3, where it
coincides with higher iteration counts and the observed 3% RAVD failures.

## Threshold Recommendations

### ENGINEERING_SMOKE_THRESHOLD

Purpose: catch obvious local regressions in the standalone generator quickly.
This is an engineering gate, not a scientific validity claim.

Recommended smoke criteria for this benchmark shape:

- For target volumes >=100 mm3: pass if RAVD <=3%.
- For target volumes <100 mm3: pass if RAVD <=8% and absolute error <=5 mm3.
- For any target: fail if `status` is not `OK`.
- For any target: fail if the mask has zero voxels, touches the generation grid,
  or has component count other than 1.
- For any target: fail/review if local scale-curve non-monotonicity appears.
- Plateau flags alone should warn, not fail, unless paired with excessive error
  or max-iteration non-convergence.

Rationale: this benchmark shows clean behavior at and above 100 mm3. Below
100 mm3, 3% RAVD is too strict for smoke testing because 1-5 mm3 absolute misses
can exceed 3% solely from voxel quantization.

### SCIENTIFIC_ACCEPTANCE_THRESHOLD

Purpose: support a scientific claim about clinically meaningful volume
targeting. This benchmark is insufficient for that threshold.

Evidence is insufficient to set a final scientific acceptance threshold because
this run covers only the standalone mask generator, two seeds, two spacings, and
36 cases. It does not exercise embedded placement, patient-frame clipping,
image interpolation, full MRI spacing distributions, or a full cohort.

Provisional scientific review criteria, pending broader validation:

- Treat >=100 mm3 standalone generator cases with RAVD <=3% as technically
  acceptable for further embedded-pipeline testing.
- Treat <100 mm3 cases as requiring absolute-error-aware review, not pure RAVD.
- Do not adopt the engineering smoke allowance for <100 mm3 as a scientific
  acceptance rule without additional seed and spacing sweeps.

## Deliverables

Regenerated or updated successfully:

- `analysis/volume_targeting/volume_targeting_benchmark.csv`
- `analysis/volume_targeting/VOLUME_TARGETING_REPORT.md`
- `analysis/volume_targeting/requested_vs_achieved.png`
- `analysis/volume_targeting/ravd_by_target_volume.png`

No source files were edited, no morphology tuning was performed, no SSH/Rivanna
access was used, and no full cohort was run.
