# Volume Targeting Validation Report

Task: VOL-001  
Scope: local standalone lollipop mask generator only  
Repository state: current dirty worktree as of 2026-07-18  

## Summary

This benchmark quantified the current local behavior of the standalone synthetic
lollipop generator path used by `scripts/generate_synthetic_lollipop_cohort.py`.
No core generator, embedding, script, test, or documentation files were modified.

The benchmark generated 36 local calibration cases:

- target volumes: 25, 50, 75, 100, 250, 500, 1000, 4000, and 16000 mm3
- seeds: 20260426 and 20260427
- spacings: 1.0 x 1.0 x 1.0 mm and 0.6 x 0.6 x 1.2 mm
- tolerance: 3% relative absolute volume difference (RAVD)
- max calibration iterations: 14

Overall result:

- 36/36 cases completed without exceptions.
- 30/36 cases met the 3% RAVD tolerance.
- Median RAVD was 1.64%.
- Mean RAVD was 2.14%.
- Maximum RAVD was 8.00%.
- No generated mask touched the generation-grid boundary.
- All generated masks were single connected components.
- Local scale-curve probes were monotonic for all cases.

## Deliverables

- `analysis/volume_targeting/run_volume_targeting_benchmark.py`
- `analysis/volume_targeting/volume_targeting_benchmark.csv`
- `analysis/volume_targeting/requested_vs_achieved.png`
- `analysis/volume_targeting/ravd_by_target_volume.png`
- `analysis/volume_targeting/_smoke_volume_targeting_benchmark.csv`

The `_smoke_...csv` file is a preliminary 3-case run kept for auditability.

## Command

```bash
python3 analysis/volume_targeting/run_volume_targeting_benchmark.py \
  --targets 25,50,75,100,250,500,1000,4000,16000 \
  --seeds 20260426,20260427 \
  --spacings '1,1,1;0.6,0.6,1.2' \
  --out_csv analysis/volume_targeting/volume_targeting_benchmark.csv
```

Smoke command:

```bash
python3 analysis/volume_targeting/run_volume_targeting_benchmark.py \
  --targets 25,250,2500 \
  --seeds 20260426 \
  --spacings '1,1,1' \
  --out_csv analysis/volume_targeting/_smoke_volume_targeting_benchmark.csv
```

## Results By Target Volume

| Target mm3 | Cases | Median RAVD | Max RAVD | Median abs error mm3 | Cases <=3% |
|---:|---:|---:|---:|---:|---:|
| 25 | 4 | 5.34% | 8.00% | 1.34 | 1/4 |
| 50 | 4 | 2.18% | 6.00% | 1.09 | 3/4 |
| 75 | 4 | 3.99% | 6.67% | 3.00 | 2/4 |
| 100 | 4 | 0.33% | 1.95% | 0.33 | 4/4 |
| 250 | 4 | 1.12% | 2.71% | 2.81 | 4/4 |
| 500 | 4 | 1.64% | 2.60% | 8.19 | 4/4 |
| 1000 | 4 | 0.82% | 1.37% | 8.24 | 4/4 |
| 4000 | 4 | 1.92% | 2.49% | 76.82 | 4/4 |
| 16000 | 4 | 1.30% | 1.98% | 207.90 | 4/4 |

## Results By Spacing

| Spacing mm | Cases | Median RAVD | Max RAVD | Cases <=3% |
|---|---:|---:|---:|---:|
| 1.0 x 1.0 x 1.0 | 18 | 1.55% | 8.00% | 13/18 |
| 0.6 x 0.6 x 1.2 | 18 | 1.76% | 6.69% | 17/18 |

The anisotropic test spacing did not degrade median accuracy in this small grid.
It actually improved the tiny-volume pass rate because the voxel volume is
0.432 mm3 instead of 1.0 mm3, reducing quantization error.

## Supported Local Operating Range

Based on this benchmark, the standalone local generator is well supported for
targets at or above 100 mm3 under the tested spacing regimes:

- 100 to 16000 mm3: 28/28 cases met the 3% RAVD tolerance.
- 25 to 75 mm3: 6/12 cases met the 3% RAVD tolerance.

The small-volume failures are not large absolute misses. They are dominated by
voxel quantization:

- 25 mm3 failures missed by 1 to 2 mm3 for 1 mm isotropic masks, which is still
  4% to 8% RAVD.
- 50 and 75 mm3 failures similarly missed by only a few mm3.

Recommended local interpretation:

- >=100 mm3: use 3% RAVD as the normal acceptance threshold for standalone mask
  generation.
- 50 to <100 mm3: use 3% RAVD as a warning threshold and allow case review when
  absolute error is <=3 mm3.
- <50 mm3: use an absolute-error-aware threshold; 3% RAVD alone is too strict
  for 1 mm3 voxel quantization.

## Overshoot, Undershoot, Plateaus, And Monotonicity

Across the 36 final benchmark cases:

- Overshoot: 18 cases.
- Undershoot: 16 cases.
- Exact hit: 2 cases.
- Local scale-curve non-monotonicity: 0 cases.
- Local scale-curve plateau flag: 30 cases.

The plateau flag is expected and should be interpreted carefully. It comes from
a coarse local scale probe around the initial search window and mostly reflects
discrete voxelization at low scale values, where several neighboring scale
settings can produce the same tiny mask volume. It was not associated with
failed generation-grid clipping or disconnected masks in this run.

## Historical Local Evidence

The repository also contains a pulled historical synthetic manifest at
`rivanna_pull/analysis/synthetic_lollipop_v1/manifests/synthetic_lollipop_manifest.csv`.
That file has 261 rows with target and realized volume fields, but its mask paths
refer to remote storage and were not used as executable local input.

As prior local evidence only:

- target range: 7.75 to 10363.5 mm3
- median target: 270.375 mm3
- median volume error fraction: 1.49%
- mean volume error fraction: 1.90%
- maximum volume error fraction: 14.50%
- 237/261 rows were <=3% error

This historical manifest is consistent with the new local benchmark: most cases
are accurate, but very small volumes are the weak tail.

## Limitations

This is not a full scientific validation of clinical tumor volume targeting.

Important limitations:

- This benchmark covers the standalone synthetic mask generator only.
- It does not exercise `embed_tumor.py` placement, orientation selection,
  interpolation, clipping inside a patient MRI frame, or embedded target-volume
  reporting.
- It uses two base seeds, not a large population sweep.
- It uses two spacing regimes, not the full distribution of real MRI spacings.
- It does not validate anatomical realism, only requested-versus-achieved mask
  volume behavior.
- The local scale-curve probe is diagnostic, not a replacement for recording
  every internal binary-search candidate.

## Recommended Validation Thresholds

For local standalone synthetic lollipop mask generation:

1. Targets >=100 mm3:
   - pass if RAVD <=3%
   - warn if 3% < RAVD <=5%
   - fail/review if RAVD >5%

2. Targets 50 to <100 mm3:
   - pass if RAVD <=3%
   - warn if RAVD <=7% and absolute error <=5 mm3
   - fail/review otherwise

3. Targets <50 mm3:
   - pass if absolute error <=1 mm3 or RAVD <=3%
   - warn if absolute error <=3 mm3
   - fail/review otherwise

4. Any target:
   - fail/review if mask has zero voxels
   - fail/review if component count is not 1
   - fail/review if the mask touches the generation-grid boundary
   - fail/review if local scale-curve non-monotonicity is observed near the
     calibration range

## Follow-Up Tasks

- VOL-002: Add an embedded-pipeline volume-targeting benchmark using local
  synthetic MRI/segmentation fixtures, once that file scope is assigned.
- VOL-003: Add candidate-level calibration trace logging in the generator or
  benchmark path, if a future task grants ownership of the generator script.
- VOL-004: Expand the target grid below 100 mm3 with smaller voxel volumes to
  separate true calibration limitations from voxel quantization.
- VOL-005: Run a larger local seed sweep for 100 to 16000 mm3 before using the
  thresholds as release criteria.

## Human Review

No immediate human scientific decision is required for this local engineering
benchmark. Human review is recommended before adopting the proposed thresholds
as scientific acceptance criteria.
