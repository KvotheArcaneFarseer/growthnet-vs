# Orientation Metric Audit
**Prepared:** 2026-04-24  
**File audited:** `embed_tumor.py`  
**Status:** Bugs fixed (see section 6). Ambiguities documented. Recommendations listed.

---

## 1. Current Metric Definitions

### Orientation scoring chain

`_orientation_confidence(candidates)` → computes:

| Variable | Formula | Meaning |
|----------|---------|---------|
| `best` | `max(candidate.score)` | Highest Dice score among signed-axis candidates |
| `worst` | `min(candidate.score)` | Lowest Dice score |
| `margin` | `best - worst` | Raw score gap (absolute Dice units) |
| `normalized_gap` | `margin / (|best| + |worst| + ε)` | Scale-free gap (0–1) |

`normalized_gap` is returned as the first tuple element.

### OrientationResult fields

| Field | Actual value | Comment |
|-------|-------------|---------|
| `confidence` | `normalized_gap` | Misleading name — it IS the normalized gap, not a separate metric |
| `low_confidence` | `margin < 0.05 OR normalized_gap < 0.05` | Hardcoded thresholds |
| `debug["score_margin"]` | `margin` | Raw Dice gap |
| `debug["normalized_gap"]` | `normalized_gap` | Redundant — same as `confidence` |

### EmbeddingCaseMetrics orientation fields

| Field | Source | Actual content |
|-------|--------|---------------|
| `orientation_confidence` | `result.confidence` | normalized_gap — **duplicate of orientation_normalized_gap** |
| `orientation_score_margin` | `debug["score_margin"]` | Raw Dice margin |
| `orientation_normalized_gap` | `debug["normalized_gap"]` | normalized_gap — **duplicate of orientation_confidence** |
| `orientation_low_confidence` | `result.low_confidence` | Precomputed bool with hardcoded 0.05 thresholds |

### Placement metrics (ValidationThresholds, EmbeddingCaseMetrics)

| Metric | Field | Threshold type |
|--------|-------|---------------|
| Centroid offset | `centroid_offset_mm` | warn ≥ 1.5 mm, fail ≥ 3.0 mm |
| Retained fraction | `retained_fraction` | warn ≤ 0.95, fail ≤ 0.80 |
| Worst clipping | `worst_clipping_fraction` | warn ≤ 0.90, fail ≤ 0.75 |
| Placed/seg ratio | `placed_to_seg_ratio` | warn outside [0.20, 1.80], fail outside [0.10, 2.50] |
| Monotone growth | `monotone_growth` | allowed drop fraction: 0.02 |

---

## 2. Current Warning/Failure Conditions (pre-fix)

| Finding code | Severity | Trigger |
|-------------|---------|---------|
| `empty_output` | hard_failure | placed voxels = 0 |
| `centroid_offset_fail` | hard_failure | offset ≥ 3.0 mm |
| `centroid_offset_warn` | warning | offset ≥ 1.5 mm |
| `retained_fraction_fail` | hard_failure | fraction ≤ 0.80 |
| `retained_fraction_warn` | warning | fraction ≤ 0.95 |
| `worst_clipping_fail` | hard_failure | worst tp fraction ≤ 0.75 |
| `worst_clipping_warn` | warning | worst tp fraction ≤ 0.90 |
| `non_monotone_growth` | warning | **BUG: only fires when clipping is in warning zone** |
| `orientation_low_confidence` | warning | `low_confidence=True` OR `confidence ≤ 0.08` |
| `strategy_disagreement` | warning | strategies pick different signed axis |
| `placed_to_seg_ratio_fail` | hard_failure | ratio outside [0.10, 2.50] |
| `placed_to_seg_ratio_warn` | warning | ratio outside [0.20, 1.80] |

---

## 3. Where Interpretation Was Ambiguous

### 3a. `orientation_confidence` == `orientation_normalized_gap` (redundant fields)

Both fields in the CSV/JSON output hold the identical value. A reader cannot know which to use or whether they encode different things. The name "confidence" suggests a probability, but the value is a normalized distance metric (0 = ambiguous, 1 = completely unambiguous).

### 3b. `low_confidence` uses hardcoded thresholds, warning uses configurable threshold

Pre-fix behavior:
- `low_confidence=True` was set when `margin < 0.05 OR normalized_gap < 0.05` (hardcoded in `_orientation_confidence`)
- The warning check was `if low_confidence OR normalized_gap <= orientation_confidence_warn` (configurable threshold 0.08)
- Changing `orientation_confidence_warn` only affected half the check — cases with `low_confidence=True` (from hardcoded 0.05) still fired regardless

This made the configurable threshold partially ineffective. A user changing `orientation_confidence_warn` to 0.03 would still get warnings for any case where `normalized_gap < 0.05`.

### 3c. `non_monotone_growth` was never independently checked (BUG)

Due to indentation error, the `non_monotone_growth` finding was only evaluated when clipping was in the warning band — not when clipping was fine and not when clipping was severe. Growth monotonicity was effectively invisible in normal runs.

### 3d. `retained_fraction > 1.0` is possible and was silently accepted

Due to resampling into MRI space, the placed tumor can occupy fractionally more mm³ than the source mask (e.g., 1.004 observed). No upper-bound check existed. Values up to ~1.05 are benign resampling artifacts; larger values would indicate a computation error.

### 3e. `placed_to_seg_ratio` uses max-across-timepoints volume

The metric compares `max_placed_volume_mm3` (the largest synthetic tumor across all timepoints) against the real segmentation `seg_volume_mm3`. This is the right comparison for "is the full-grown tumor a plausible size relative to the real tumor?", but it is not labeled as such. A reader comparing it to `placed_volume_mm3` (primary tp) will be confused by the discrepancy.

### 3f. `source_volume_mm3` fallback is raw voxel count

If `TimepointMetrics` is unavailable, `source_volume_mm3` falls back to `float(primary_source_voxels)` — treating voxel count as mm³. This is wrong for any voxel size ≠ 1.0 mm³/voxel. In practice `TimepointMetrics` is always populated, but the fallback is silently incorrect.

---

## 4. Absolute vs Relative Metrics

| Metric | Type | Notes |
|--------|------|-------|
| `orientation_score_margin` | Absolute (Dice units) | Depends on how similar the two candidates' Dice scores are; can be 0–1 |
| `orientation_normalized_gap` | Relative (scale-free) | Normalized by sum of absolute scores |
| `centroid_offset_mm` | Absolute (mm) | Physical space, comparable across cases |
| `centroid_offset_vox` | Absolute (voxels) | NOT comparable across cases with different voxel sizes |
| `retained_fraction` | Relative (ratio) | Should be near 1.0 |
| `placed_to_seg_ratio` | Relative (ratio) | Uses max-tp placed vs real seg |
| `worst_clipping_fraction` | Relative (ratio) | Minimum retained fraction across timepoints |

---

## 5. Fields That Should Be Renamed or Split

| Current name | Problem | Recommendation |
|-------------|---------|---------------|
| `orientation_confidence` | IS normalized_gap, not a calibrated confidence | Rename to `orientation_normalized_gap_v2` or document clearly; keep for backward compat |
| `orientation_normalized_gap` | Duplicate of `orientation_confidence` | Mark as deprecated; it's the same value |
| `orientation_low_confidence` | Bool computed with hardcoded threshold, mixed semantics | Replace with separate `orientation_low_score_margin` (bool) and `orientation_low_normalized_gap` (bool) |
| `placed_to_seg_ratio` | Computed from max tp, not labeled as such | Rename to `max_placed_to_seg_ratio` in future schema version |

---

## 6. Bugs Fixed (2026-04-24)

### Fix 1: `non_monotone_growth` indentation bug
**Before:** Nested inside `elif worst_clipping_fraction <= thresholds.worst_clipping_warn:` — only evaluated when clipping was in warning band.  
**After:** Independent top-level check — fires whenever growth is non-monotone, regardless of clipping state.

### Fix 2: Debug print statements removed
**Before:** `validate_embedding_case()` printed all orientation debug values to stdout on every run.  
**After:** Removed.

### Fix 3: Orientation warning split into two independent findings
**Before:** Single `orientation_low_confidence` warning fired when `low_confidence=True OR normalized_gap <= 0.08`. The `low_confidence` bool used hardcoded 0.05 threshold, making the configurable threshold only partially effective.  
**After:** Two separate findings:
- `orientation_low_score_margin` — fires when raw Dice margin < `orientation_score_margin_warn` (default 0.05, configurable)
- `orientation_low_normalized_gap` — fires when normalized gap ≤ `orientation_confidence_warn` (default 0.08, configurable)

Each finding has an independent threshold, independent code, and independent message. The `low_confidence` bool on `OrientationResult` is preserved for backward compatibility but no longer drives the warning logic.

### Fix 4: `retained_fraction_high` upper-bound check added
**Before:** No check for `retained_fraction > 1.0`. Values like 1.02 (resampling artifact) passed silently.  
**After:** Warning `retained_fraction_high` fires when `retained_fraction > retained_fraction_high_warn` (default 1.05). Sub-artifact values (≤ 1.05) remain silent. Larger values will surface as warnings.

### New threshold fields in `ValidationThresholds`
- `retained_fraction_high_warn: float = 1.05`
- `orientation_score_margin_warn: float = 0.05`

---

## 7. Recommended Further Changes (Not Yet Implemented)

### 7a. Rename `orientation_confidence` → `orientation_normalized_gap` in schema
Would require migrating existing JSON/CSV consumers. Safe to do once all downstream readers are updated.

### 7b. Per-timepoint axis_error_deg
`TimepointMetrics.axis_error_deg` is a field that exists but is always `None` in current output (no physical-space axis comparison is performed per timepoint). Either populate it or remove it to reduce confusion.

### 7c. Overlap with real segmentation
Currently, the orientation strategy uses Dice to choose the axis sign, but no per-case Dice overlap metric is reported in `EmbeddingCaseMetrics`. Adding `placed_vs_real_dice_t0` and `placed_vs_real_dice_late` would make the orientation strategy's score visible in the output.

### 7d. Separate `axis_error_deg` as a standalone metric
The physical-space angle between the selected axis and the real tumor's long axis would be more interpretable than Dice score for orientation quality. Implementation: run `principal_axes()` on the real seg, then `_angle_deg(selected_axis_phys, real_long_axis_phys)`.

### 7e. Fix `source_volume_mm3` fallback
The fallback `float(primary_source_voxels)` is only correct when source voxel volume = 1.0 mm³. Should be `float(primary_source_voxels) * voxel_volume_mm3`.
