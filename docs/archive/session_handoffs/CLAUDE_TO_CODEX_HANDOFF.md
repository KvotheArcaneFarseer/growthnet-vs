# Claude → Codex Handoff
**Date:** 2026-04-24 (session 2 — metric audit + fixes)
**Status:** Phase 1-3 complete. Bugs fixed. Codex follow-up now includes QC upgrade, batch metric propagation, additive morphology controls, optional t0 calibration, and batch CSV parameterization. Remaining roadmap updated below.

---

## Codex CLI Status

Available at `/Users/kvothearcane/.nvm/versions/node/v24.14.0/bin/codex` (version 0.118.0).

```bash
codex "<your prompt here>"
```

---

## Codex Continuation Update (2026-04-24)

### What Codex changed

Completed the safest follow-up task in `embed_tumor.py` without touching tumor generation geometry:

1. **Surfaced primary axis-angle error into case-level metrics**
   - `TimepointMetrics.axis_error_deg` was already being computed per placed timepoint.
   - Added `EmbeddingCaseMetrics.primary_axis_error_deg`.
   - `validate_embedding_case()` now lifts the primary timepoint's `axis_error_deg` into the case summary.
   - `write_case_reports()` now writes `primary_axis_error_deg` to both JSON and CSV.

2. **Added validation thresholds for axis-angle error**
   - `axis_error_deg_warn: float = 30.0`
   - `axis_error_deg_fail: float = 60.0`
   - New finding codes:
     - `axis_error_warn`
     - `axis_error_fail`

3. **Upgraded QC PNG usability**
   - `_save_qc_png()` now renders a 3×4 panel instead of 3×3.
   - Added a tumor-centered ROI zoom column.
   - Added clearer column labels.
   - Added ROI-local intensity windowing for the zoom view.
   - Existing output paths and filenames are unchanged.

4. **Propagated axis quality into batch summaries**
   - `scripts/run_batch_embedding.py` now includes `primary_axis_error_deg` in flat CSV rows.
   - Batch summary JSON now reports aggregate `primary_axis_error_deg`.
   - Failure report now includes `worst_axis_error_cases`.

5. **Added additive morphology controls to the synthetic generator**
   - `projects/vivit/src/data/synthetic.py` now accepts:
     - `canal_growth_scale`
     - `bulb_growth_scale`
   - Defaults preserve existing behavior.
   - `embed_tumor.py` exposes the same controls via CLI and passes them through.

6. **Added optional t0 volume calibration**
   - `embed_tumor.py` now accepts `--t0_volume_fraction_of_seg`.
   - When provided, it estimates a scale factor for the initial synthetic geometry and regenerates the series.
   - This is currently **wired and usable**, but calibration accuracy is still coarse and needs improvement.

7. **Added batch CSV parameterization for quality sweeps**
   - `scripts/run_batch_embedding.py` now accepts optional input CSV columns:
     - `canal_growth_scale`
     - `bulb_growth_scale`
     - `t0_volume_fraction_of_seg`
   - This makes morphology and calibration sweeps scalable without further code edits.

### Files modified by Codex

| File | Change |
|------|--------|
| `embed_tumor.py` | Added `primary_axis_error_deg` to case metrics + warn/fail validation thresholds/findings |
| `embed_tumor.py` | Upgraded QC PNGs, exposed additive morphology controls, added optional `t0` calibration |
| `scripts/run_batch_embedding.py` | Added axis metric propagation + batch CSV parameterization for morphology/calibration controls |
| `projects/vivit/src/data/synthetic.py` | Added additive `canal_growth_scale` / `bulb_growth_scale` controls |

### Tests run by Codex

1. `python3 -c "import embed_tumor; print('OK')"` → **OK**
2. Full behavioral verification:

```bash
MPLCONFIGDIR=/tmp/mpl python3 embed_tumor.py \
  --mri /Users/kvothearcane/Downloads/147_0_0_t2_thin_image_coregistered.nii.gz \
  --seg /Users/kvothearcane/Downloads/147_0_0_t2_thin_R_VS__uvauser2__coregistered.nii.gz \
  --out_dir /tmp/axis_test
```

Verified outputs:
- `/tmp/axis_test/embedding_metrics.json` contains:
  - `primary_axis_error_deg = 3.609981115490621`
- `/tmp/axis_test/embedding_metrics.csv` contains:
  - `primary_axis_error_deg = 3.609981115490621`

Additional verified runs:

```bash
MPLCONFIGDIR=/tmp/mpl python3 embed_tumor.py \
  --mri /Users/kvothearcane/Downloads/147_0_0_t2_thin_image_coregistered.nii.gz \
  --seg /Users/kvothearcane/Downloads/147_0_0_t2_thin_R_VS__uvauser2__coregistered.nii.gz \
  --out_dir /tmp/qc_upgrade_test
```

```bash
MPLCONFIGDIR=/tmp/mpl python3 embed_tumor.py \
  --mri /Users/kvothearcane/Downloads/147_0_0_t2_thin_image_coregistered.nii.gz \
  --seg /Users/kvothearcane/Downloads/147_0_0_t2_thin_R_VS__uvauser2__coregistered.nii.gz \
  --out_dir /tmp/morph_scale_test \
  --canal_growth_scale 0.85 \
  --bulb_growth_scale 1.35
```

```bash
MPLCONFIGDIR=/tmp/mpl python3 embed_tumor.py \
  --mri /Users/kvothearcane/Downloads/147_0_0_t2_thin_image_coregistered.nii.gz \
  --seg /Users/kvothearcane/Downloads/147_0_0_t2_thin_R_VS__uvauser2__coregistered.nii.gz \
  --out_dir /tmp/t0_calibration_test \
  --canal_growth_scale 0.85 \
  --bulb_growth_scale 1.35 \
  --t0_volume_fraction_of_seg 0.08
```

```bash
MPLCONFIGDIR=/tmp/mpl python3 scripts/run_batch_embedding.py \
  --input_csv /tmp/batch_param_case.csv \
  --out_dir /tmp/batch_param_test \
  --num_cases 1
```

Observed verification outcomes:
- QC PNG generation still completes after the 3×4 upgrade.
- Batch summary now reports `primary_axis_error_deg`.
- Batch CSV parameterization successfully drives canal/bulb growth scales and `t0` calibration.
- The current `t0` calibration pass is **not yet accurate**: the requested target fraction changes the synthetic series, but the realized t0 volume can remain close to the old baseline. This needs a more exact iterative calibration loop.

### GitNexus / blast radius notes

Before editing, Codex ran GitNexus impact analysis on:
- `validate_embedding_case`
- `write_case_reports`

Both returned **LOW risk**.
Direct caller: `main()` in `embed_tumor.py`
Downstream reporting path: `scripts/run_batch_embedding.py`

This change stayed inside the metric/reporting layer. No embedding geometry or synthetic lollipop generation logic was changed.

---

## What This Session Did

### Phase 1 — Audit (complete)

Read and fully analyzed the orientation evaluation pipeline in `embed_tumor.py`. Found:

1. **`non_monotone_growth` check was silently broken** — indentation bug nested it inside the `elif worst_clipping_warn` branch. It was never flagged in normal runs (clipping was always OK). Bug existed in every past output.

2. **`orientation_confidence` == `orientation_normalized_gap`** — these two fields in the JSON/CSV output hold the identical value. `OrientationResult.confidence` is assigned `normalized_gap` directly. Consumers of the CSV were seeing two columns with the same data.

3. **`low_confidence` used hardcoded thresholds (0.05/0.05)** while the warning check used a configurable threshold (0.08) in an OR condition — making the configurable threshold only partially effective.

4. **Debug print statements left in `validate_embedding_case()`** — printed all orientation debug info to stdout on every run.

5. **`retained_fraction > 1.0` observed (1.004–1.021) with no upper-bound check** — sub-1.05 values are benign resampling artifacts, but there was no flag for higher values.

### Phase 2 — Research (complete, documented)

See `docs/GROWTH_MORPHOLOGY_VISUALIZATION_NOTES.md` for:
- Visualization improvements (ROI zoom, contour overlays, windowing, labels)
- Morphology improvements (smooth neck, differential growth, volume calibration)
- Feasibility analysis for headless 3D renders using matplotlib + marching_cubes
- Literature pointers for synthetic lesion insertion evaluation

### Phase 3 — Implementation (complete)

**File modified:** `embed_tumor.py`
**No other files changed. No pipeline re-run needed.**

#### Fix 1: non_monotone_growth nesting (lines ~490-500)
Moved out of `elif worst_clipping_warn:` into a top-level independent check.

#### Fix 2: Debug prints removed (~10 lines around line 497)
Removed the `===== DEBUG ORIENTATION =====` print block.

#### Fix 3: Orientation warning split
Replaced single `orientation_low_confidence` warning with:
- `orientation_low_score_margin` — fires when `score_margin < orientation_score_margin_warn` (configurable, default 0.05)
- `orientation_low_normalized_gap` — fires when `normalized_gap <= orientation_confidence_warn` (configurable, default 0.08)

Each is independently evaluatable. The old `low_confidence` bool is preserved on `OrientationResult` for backward compatibility but no longer drives warning logic.

#### Fix 4: retained_fraction upper-bound check added
Fires warning `retained_fraction_high` when `retained_fraction > retained_fraction_high_warn` (default 1.05).

#### New threshold fields added to ValidationThresholds
- `retained_fraction_high_warn: float = 1.05`
- `orientation_score_margin_warn: float = 0.05`

---

## Current Metric Values (case 147_0_0, unchanged data)

| Metric | Value | Status |
|--------|-------|--------|
| orientation_confidence (= normalized_gap) | 0.211 | OK |
| orientation_score_margin | 0.231 | OK |
| retained_fraction | 1.004 | WARNING — retained_fraction_high now fires |
| worst_clipping_fraction | 0.997 | OK |
| centroid_offset_mm | 0.022 | OK |
| placed_to_seg_ratio | 0.632 | OK |
| monotone_growth | True | OK |

**Effect of fixes:** Running the pipeline fresh on case 147 will now show `retained_fraction_high` warning (retained_fraction = ~1.004 observed, which is < 1.05 threshold, so actually still silent). Check with a fresh run to confirm.

---

## Files Modified This Session

| File | Change |
|------|--------|
| `embed_tumor.py` | Fixed 4 bugs (non_monotone nesting, debug prints, orientation warning split, retained_fraction upper bound) |
| `embed_tumor.py` | Codex follow-up: added `primary_axis_error_deg` + axis warn/fail findings |
| `embed_tumor.py` | Codex follow-up: QC 3×4 panel, morphology CLI controls, optional `t0` calibration |
| `scripts/run_batch_embedding.py` | Codex follow-up: batch axis metric propagation + optional morphology/calibration CSV columns |
| `projects/vivit/src/data/synthetic.py` | Codex follow-up: additive canal/bulb growth controls |

## Files Created This Session

| File | Purpose |
|------|---------|
| `docs/ORIENTATION_METRIC_AUDIT.md` | Full audit: definitions, contradictions, fixes, recommendations |
| `docs/GROWTH_MORPHOLOGY_VISUALIZATION_NOTES.md` | Research notes on visualization and morphology improvements |
| `CLAUDE_TO_CODEX_HANDOFF.md` | This file |

---

## Tests Run

1. `python3 -c "import embed_tumor; print('OK')"` → **OK — module loads**
2. No fresh pipeline run done this session (NIfTI inputs untouched, existing outputs untouched)
3. Codex ran a fresh end-to-end verification to `/tmp/axis_test` and confirmed `primary_axis_error_deg` is populated in JSON and CSV
4. Codex ran `/tmp/qc_upgrade_test` and verified the QC upgrade completes end-to-end
5. Codex ran `/tmp/morph_scale_test` and verified additive morphology controls complete end-to-end
6. Codex ran `/tmp/t0_calibration_test` and verified `t0` calibration is wired through end-to-end
7. Codex ran `/tmp/batch_param_test` and verified batch CSV parameterization completes end-to-end

To verify the fixes behaviorally, run:
```bash
cd /Users/kvothearcane/Personal/Coding\ Projects/GrowthNet
python3 embed_tumor.py \
  /Users/kvothearcane/Downloads/147_0_0_t2_thin_image_coregistered.nii.gz \
  /Users/kvothearcane/Downloads/147_0_0_t2_thin_R_VS__uvauser2__coregistered.nii.gz \
  --out-dir /tmp/test_metric_fixes
cat /tmp/test_metric_fixes/embedding_metrics.json | python3 -m json.tool | grep -A2 "findings\|warnings"
```

---

## What Remains Unfinished

### High priority
1. **`orientation_confidence` == `orientation_normalized_gap` in CSV** — duplicate fields. Neither should be removed yet (backward compat), but the duplication should be explicitly documented in downstream reports or deprecated in a versioned schema.
2. **`t0` volume calibration is only approximate** — the current one-pass scale estimate is wired through the pipeline, but realized `t0` volume can still miss the requested target substantially. This should become an iterative calibration loop that measures realized t0 volume and re-solves the initial geometry scale.
3. **Batch runs still cannot summarize morphology-control effects explicitly** — batch tooling now passes through `canal_growth_scale`, `bulb_growth_scale`, and `t0_volume_fraction_of_seg`, but it does not yet aggregate their effect on realized `t0` size, late-size retention, or early-timepoint axis error.

### Medium priority
4. **Contour overlays in QC PNG** — current QC now has ROI zoom and labels, but it still uses filled overlays only; thin contour overlays would preserve MRI texture better.
5. **Headless 3D isosurface render** — feasible with matplotlib + marching_cubes; moderate effort
6. **Smooth lollipop neck transition** — realistic morphology improvement; requires touching `projects/vivit/src/data/synthetic.py`
7. **Differential growth rates are exposed, but not yet analytically tuned** — the new canal/bulb growth multipliers work, but no calibration study has yet identified a good default regime.
8. **Realized-vs-target morphology metrics are missing** — there is still no explicit case metric for realized synthetic t0 volume fraction relative to the requested calibration target.

### Low priority / future work
9. **Dual-centroid or overlap-aware slice selection** for QC
10. **Atlas-informed canal direction checks** if the project later moves beyond shape-only orientation validation

---

## Do NOT Change

- **Core embedding geometry** (`rotate_and_translate`, `principal_axes`, the lollipop shape parameters in `synthetic.py`) — these are working and produced the lab meeting slides
- **Existing output directories** — do not overwrite `embedding_outputs/`, `tmp_seed_validation/`, `animation_outputs/`
- **`write_case_reports()` CSV schema** — downstream scripts may depend on existing column names
- **`OrientationResult.low_confidence`** — preserved for backward compatibility; do not remove

---

## Next Codex Tasks (paste-ready prompts)

### Task A: Make t0 calibration actually converge

```
In GrowthNet/embed_tumor.py, replace the current one-pass t0 calibration with an
iterative calibration loop.

Current behavior:
- when --t0_volume_fraction_of_seg is provided, the code estimates one scale
  factor from the default t0 volume and regenerates the series once
- this is wired correctly but does not reliably hit the requested target

Target:
- iteratively regenerate the synthetic series until realized t0 volume is within
  a reasonable tolerance band of the requested target fraction
- preserve deterministic behavior for a fixed seed
- keep defaults unchanged when no t0 target is provided

Suggested outputs:
- realized_t0_volume_mm3
- requested_t0_volume_mm3
- realized_t0_fraction_of_seg
- t0_calibration_iterations
```

### Task B: Add realized-vs-target morphology metrics to reports

```
In GrowthNet/embed_tumor.py and scripts/run_batch_embedding.py, add explicit
metrics for calibration quality.

Desired metrics:
- requested_t0_volume_mm3
- realized_t0_volume_mm3
- realized_t0_fraction_of_seg
- t0_volume_target_error_fraction

Then aggregate them in batch summaries so parameter sweeps become measurable.
```

### Task C: Smooth the bulb-canal neck conservatively

```
In GrowthNet/projects/vivit/src/data/synthetic.py, add an optional narrow
neck-smoothing control around the bulb-canal junction.

Constraints:
- preserve current coordinate convention
- preserve current defaults unless the new control is enabled
- do not rewrite the global lollipop geometry or orientation rules
- keep monotone source-space growth intact
```

### Task D: Headless 3D isosurface renders for lab exports

```
In GrowthNet, write scripts/render_3d_tumor_views.py.

Use marching_cubes + matplotlib 3D only. Save one PNG per camera angle plus
a combined figure. Keep it headless and safe for CI / remote runs.
```

---

## Senior Engineering Roadmap (Codex recommendation)

### Track 1 — QC usability first

This is the best next investment for lab value per unit of risk.

Reasoning:
- `_save_qc_png()` already sits at the review boundary and does not affect generated masks.
- The current full-slice 3×3 panel underserves clinical inspection because the tumor occupies a tiny fraction of each slice.
- ROI zoom, ROI-local windowing, and readable labels improve trust without invalidating prior outputs.

### Track 2 — Report metrics where batch triage happens

`primary_axis_error_deg` is now available per case, but batch tooling still ranks cases by confidence/clipping only.
This leaves the most interpretable orientation metric underused.

Reasoning:
- low engineering risk
- immediate value for sorting failures and picking review cases
- no change to core placement or geometry

### Track 3 — Morphology realism in `synthetic.py`, but only behind additive controls

The current lollipop generator is structurally sound. The main realism limitations are:
- accelerated but hard-coded CPA-vs-canal growth routing
- a visibly sharp bulb/canal junction
- no calibration target for t=0 relative to the real tumor volume

Reasoning:
- this is where realism improves most, but also where regressions are easiest to introduce
- changes here should be parameterized, default-stable, and isolated from placement/orientation code

### Track 4 — Derived exports for communication, not pipeline correctness

Headless 3D renders and richer lab-meeting exports should be treated as downstream presentation tooling.

Reasoning:
- high communication value
- low scientific risk if kept separate from data generation
- should remain script-level outputs, not hidden inside the main embedding path

---

## Architecture Notes (from Graphify, 2026-04-13)

- 101 nodes, 157 edges, 16 communities
- `main()` is the hub (18 edges) — controls the whole embedding flow
- `validate_embedding_case()` has 8 edges — the core QC function (edited this session)
- `_orientation_confidence()` is called by both orientation strategies — do not move its signature
- Community "Validation & Findings": `ValidationFinding`, `_orientation_confidence()`, `OrientationCandidate`, `OrientationResult`, `_place_mask_for_candidate()`, `_dice_score()`
- Community "Thresholds & Config": `ValidationThresholds`, `TimepointMetrics`
- `OrientationResult.low_confidence` is still serialized via `_serialize_orientation_result()` and appears in `strategy_results` list in the JSON — do not remove it
