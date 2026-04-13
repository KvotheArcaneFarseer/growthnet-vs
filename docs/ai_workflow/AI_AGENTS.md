# AGENTS.md — GrowthNet Synthetic VS Embedding Pipeline

This file governs agent and human behaviour when modifying the synthetic vestibular
schwannoma embedding pipeline. It is written for this repo, not for a generic project.
Read it before touching `embed_tumor.py`, `projects/vivit/src/data/synthetic.py`, or
any file that generates, transforms, or validates synthetic tumor geometry.

# AI Agent System Design (GrowthNet)

This project was developed using a structured AI-assisted workflow:

- **Claude Code** — architecture design, reasoning, and system validation  
- **Codex** — implementation, debugging, and code refinement  

## Purpose

This document defines the constraints and guardrails used during development to ensure:

- Deterministic tumor generation
- Anatomically realistic lollipop geometry (IAC → CPA)
- Reproducible batch processing
- Reliable validation behavior

## Key Principles

- Geometry must follow canal → bulb growth
- No uncontrolled randomness (seeded generation)
- Avoid high-frequency noise; enforce anatomical smoothness
- Separate design (Claude) from execution (Codex)

## Notes

This system reflects an iterative workflow where:
- Claude defines structure and constraints
- Codex implements and refines code

---

## 1. Project mission

Generate anatomically plausible synthetic vestibular schwannoma (VS) time-series and
embed them into real MRI volumes for supervised training data augmentation. Synthetic
tumors must respect the gross anatomy of the internal auditory canal (IAC) and
cerebellopontine angle (CPA), and their geometry must remain physically meaningful
through any pipeline modification.

---

## 2. Hard invariants

These rules must not be broken. If you believe one must change, stop and get explicit
approval before touching the code.

### 2.1 Synthetic space is 1 mm isotropic

The synthetic tumor is generated in a coordinate frame where **1 synthetic voxel = 1 mm**.
This is not an approximation. Literature IAC dimensions (10 mm canal, 3.5 mm porus
radius, 2.0 mm fundus radius) and computed CPA radii are passed directly as synthetic
voxel counts precisely because this identity holds.

Variables that live in this space use the `_syn` suffix (e.g. `canal_length_syn`,
`cpa_radius_syn`). Do not change the naming convention without updating every call site
and every diagnostic print that references synthetic space.

### 2.2 `mean_spacing_mm` must not be reintroduced

`mean_spacing_mm = float(np.mean(voxel_spacing))` was removed because collapsing
anisotropic MRI spacing to a scalar distorts IAC geometry. **Do not add it back in any
form**, including averaging only two axes, using `np.median`, or deriving any single
scalar from `header.get_zooms()` and using it to convert anatomy dimensions.

Anisotropic target-MRI spacing is handled exclusively by the `rotate_and_translate`
function via its `dst_spacing` parameter.

### 2.3 Physical space and voxel space must not be silently mixed

Any variable that represents a length or radius must be labelled:

| Suffix / label | Meaning                                           |
| -------------- | ------------------------------------------------- |
| `_syn`         | Synthetic isotropic space (= mm by invariant 2.1) |
| `_vox`         | Target-MRI voxel indices (anisotropic)            |
| `_mm`          | Physical millimetres in world space               |

Never assign a `_syn` value to a `_vox` variable or pass a `_vox` value where a `_mm`
value is expected without an explicit conversion step that names the transformation.

A coordinate-space mix identical to the one that was removed in Apr 2026 (obs 71) is a
regression, not a refactor.

### 2.4 CPA component must remain extracanalicular

The CPA oblate ovoid mask is restricted to `x_rel <= 0.0`. Negative `x_rel` is the
extrameatal (medial/CPA) side. Positive `x_rel` is intracanalicular (fundus direction).

Do not relax or remove the `x_rel <= 0.0` gate. Any change to the oblate ovoid centre
or equatorial radius that can push CPA voxels into positive `x_rel` territory is a hard
violation.

### 2.5 Lollipop topology must be preserved

The four-component union in `create_synthetic_time_3d` must always include all four
components in their correct spatial relationship:

1. **Porus hemisphere** — rounded opening at `x_rel ∈ [-br, 0]`
2. **Tapered canal** — body from `x_rel = 0` to `x_rel = canal_length`, narrowing toward
   fundus
3. **Fundus hemisphere** — lateral cap at `x_rel ∈ (cl, cl + ar]`
4. **CPA oblate ovoid** — dominant medial bulb centred at `x_rel ≈ -cpa_radius * 0.55`

The CPA bulb must be visually dominant at late timepoints; the canal must be clearly
narrower than the bulb. Do not merge components, swap their order, or allow the canal
and bulb to become the same radius.

### 2.6 Boundary perturbation must be low-frequency only

Only angular modes **2** and **3** are permitted for CPA bulb surface lobulation.
Adding mode 1 (global shift), mode 4, or any higher mode is prohibited until there is
an explicit decision to do so. The purpose is anatomical plausibility, not random noise.

No voxel-level jagged or noisy edge corruption is permitted anywhere on the mask
boundary.

### 2.7 Perturbation phase is fixed per series, not per timepoint

Random phase offsets for surface perturbation are sampled **once** at the beginning of a
synthetic series and held constant across all timepoints. Sampling independently per
timepoint produces discontinuous surface morphology that is not biologically plausible
and breaks temporal consistency for downstream models.

### 2.8 End-to-end execution must be preserved

The following invocation must complete without error on a machine with the project
dependencies installed:

```
MPLCONFIGDIR=/tmp/mpl python3 embed_tumor.py
```

Any change that causes this invocation to crash, hang, or produce no output is a
regression. Do not check in code that you have not run through this command.

### 2.9 Required outputs must be generated

Every pipeline run must produce:

- NIfTI volume output(s) for the embedded time series
- `qc_embedding.png` — early-timepoint QC overlay
- `qc_embedding_late.png` — late-timepoint QC overlay showing lollipop morphology
- Machine-readable validation metrics (centroid offset, retained fraction, volume ratio,
  axis angle error)

Do not remove, rename, or gate these outputs behind a flag without explicit approval.

- `embedding_metrics.json`
- `embedding_metrics.csv`

---

## 3. Coordinate-system rules

These rules follow from the hard invariants but are stated separately because they are
the most common source of subtle bugs in this codebase.

- **Synthetic generation happens in `_syn` space.** `create_synthetic_time_3d` works
  entirely in this space. Its geometry parameters (radii, lengths) are in syn voxels =
  mm.

- **Rotation and translation to target MRI happens in `rotate_and_translate`.** This
  function receives `dst_spacing` and applies the anisotropic transform. It is the only
  place where the synthetic-to-MRI coordinate transformation occurs.

- **Diagnostic prints must identify their space.** When printing a length, radius, or
  centroid, always include the unit label (`syn vox`, `target vox`, `mm`). This was
  explicitly corrected in Apr 2026 (obs 71) and must not regress.

- **The internal canal axis convention:**
  - Positive `x_rel` (= −`canal_coord`) → intracanalicular, toward fundus (lateral)
  - Negative `x_rel` → extracanalicular, toward CPA (medial)
  - Default synthetic axis `[0, 0, 1]` with `canal_axis="c"` points toward fundus

  Do not invert this convention without updating both `synthetic.py` and `embed_tumor.py`
  together, and updating this document.

---

## 4. Default geometry guardrails

These are the current working defaults. They encode clinical plausibility and have been
validated. Change them only with a clear reason and a before/after comparison of
generated geometry.

| Parameter                    | Current formula / value                   | Notes                                   |
| ---------------------------- | ----------------------------------------- | --------------------------------------- |
| CPA growth weight ramp       | `cpa_weight = 0.5 + 3.0 * time_frac`      | Ensures bulb dominates at maturity      |
| CPA init lower bound         | `max(base_radius * 0.4, 2.0)`             | Prevents zero-volume CPA at t=0         |
| Canal taper profile          | `ar + (br - ar) * (1.0 - t_canal) ** 1.5` | Convex taper, narrower toward fundus    |
| CPA oblate ovoid centre      | `max(br * 1.2, cr * 0.75)`                | Ensures offset from porus               |
| CPA axial radius             | `min(br * 0.7, cr * 0.45)`                | Oblate shape, not spherical             |
| Bulb perturbation amplitude  | ~0.06 – 0.12 × bulb equatorial radius     | Low-frequency; higher ends lobulated    |
| Perturbed radius lower clamp | `0.82 * cpa_eq_r`                         | Prevents over-deflation of bulb surface |
| Off-axis CPA bias            | ~±0.12 – ±0.16 × bulb equatorial radius   | Slight asymmetry for realism            |

---

## 5. Validation and reporting defaults

These thresholds are configurable but should not be loosened without justification.
Tightening them is always acceptable.

| Metric                               | Warning threshold     | Hard failure threshold |
| ------------------------------------ | --------------------- | ---------------------- |
| Centroid offset                      | > 5 mm                | > 10 mm                |
| Retained fraction                    | < 0.98                | < 0.90                 |
| Volume ratio (placed / segmentation) | outside [0.8, 1.5]    | outside [0.5, 2.0]     |
| Axis angle error                     | > 20°                 | > 45°                  |
| Orientation confidence               | < 0.55 (warning only) | —                      |

**Monotone volume growth** is the default expectation. If a run produces non-monotone
growth across timepoints, it should be flagged in the output. Only an explicitly named
experimental mode should be allowed to suppress this check.

Hard failures must cause the run to abort or produce a clearly marked invalid output.
Warnings must appear in stdout or the metrics file. Silent suppression of any threshold
breach is not permitted.

Hard failures must, by default, abort the run.
If the pipeline supports explicit override/force mode in the future, invalid outputs must be clearly marked in both stdout and machine-readable metrics.

---

## 6. Scope-control rules

These rules prevent well-intentioned changes from expanding scope beyond what was asked.

- **Do not modify coordinate-system logic and geometry defaults in the same commit.**
  These are independent concerns. Mixed changes make failures hard to diagnose.

- **Do not add new geometry parameters without adding corresponding validation.** If you
  introduce a new radius or length parameter, add a range check and include it in the
  metrics output.

- **Do not add new output artefacts without gating them.** Any new file written by the
  pipeline (additional QC images, intermediate volumes, extra metrics) must be behind a
  flag that defaults to off. The required outputs in §2.9 are already mandatory; do not
  add to the always-on list without explicit approval.

- **Do not touch `vitautoenc.py`, `t_unetr.py`, or the training stack** while working
  on geometry or embedding code. These are separate concerns.

- **When working on geometry, do not modify embedding, intensity compositing, validation thresholds, or training code.**
- **When working on intensity/blending, do not modify geometry formulas, lollipop topology, coordinate transforms, or orientation strategy defaults.**
- **When working on validation/reporting, do not modify generation, geometry, placement, or intensity logic unless the task explicitly requires it.**
- **When working on orientation logic, do not modify geometry growth math or transform math in the same change.**
- **Prefer single-subsystem changes.** If a task appears to require cross-subsystem edits, explain why before proceeding.

---

## 7. Experimental feature safety rules

Rules for work that is intentionally exploratory.

- **Label experimental code.** Any function, parameter, or code path that implements
  a feature not yet validated should have a comment marking it as experimental and the
  date it was added.

- **Experimental features must not affect default behaviour.** Gate them behind a
  parameter that defaults to the current validated behaviour. A caller with no keyword
  arguments must get exactly the current pipeline.

- **Do not enable non-monotone growth by default.** If you build a mode that allows
  tumors to shrink between timepoints, it must be explicitly opted in.

- **Axis sign heuristics are fragile.** The current extent-based `_orient_axis_consistently`
  has a known failure mode (obs 50, 51). Any new orientation strategy must be tested on
  at least two known patient volumes before replacing the current approach. Do not silently
  swap heuristics mid-run.

---

## 8. What future agents must not do

Concrete prohibitions based on past bugs and refactors in this repo:

1. **Do not reintroduce `mean_spacing_mm`** or any scalar derived by averaging
   anisotropic voxel spacing and using it to convert anatomy dimensions.

2. **Do not move geometry parameter conversion into `create_synthetic_time_3d`.** That
   function works in synthetic isotropic space. The caller (`embed_tumor.py`) is
   responsible for deriving syn-space parameters from MRI metadata.

3. **Do not let the CPA oblate ovoid centre drift into positive `x_rel`.** Check the
   centre offset formula whenever you change `cpa_radius` scaling.

4. **Do not collapse the four lollipop components into fewer components.** Three-
   component or two-component approximations that merge the porus with the canal, or the
   canal with the CPA bulb, destroy the topological identity of the shape.

5. **Do not add mode-4+ perturbation or per-voxel random noise to the mask boundary.**
   The boundary must remain smooth at the voxel scale.

6. **Do not sample new perturbation phases per timepoint.** All timepoints in a series
   share the same phase array.

7. **Do not rename `_syn` variables back to `_vox`.** The Apr 2026 rename (obs 71) was
   deliberate. Reverting it conflates two different coordinate spaces.

8. **Do not remove or bypass the `x_rel <= 0.0` CPA mask gate** without a reviewed
   geometry change that re-establishes extracanalicular separation by a different means.

9. **Do not break the `MPLCONFIGDIR=/tmp/mpl python3 embed_tumor.py` invocation.**
   This is the canonical smoke test. Run it before marking any geometry or embedding
   change as complete.

10. **Do not add implicit coordinate-space conversions.** Any transformation between
    synthetic, target-MRI voxel, and world-mm spaces must be explicit, named, and
    documented at the call site.

---

## 9. Advisory guidance

These are qualitative rules. They do not have numeric thresholds attached because the
project has not yet established them through systematic study.

- **Keep canal taper convex.** Linear taper is anatomically plausible but convex taper
  (current default) better matches the funnel shape seen in segmentations. If you change
  the taper profile, visually verify the porus-to-fundus cross-section progression.

- **Prefer volume-derived CPA sizing over pure literature values.** The CPA bulb radius
  is currently derived from the residual volume after subtracting the IAC truncated-cone
  volume. This ties the bulb to the actual patient segmentation. Do not replace this
  with a fixed literature radius unless the patient-specific derivation is demonstrably
  broken.

- **Sign disambiguation should eventually be anatomy-guided, not extent-guided.** The
  known weakness of `_orient_axis_consistently` (obs 50, 51) is that it uses mask
  extent, not anatomical landmarks. Future work should replace or augment this with a
  porus-detection or IAC-entry heuristic. Extent-based stabilization is acceptable
  in the interim because it is better than nothing, but it is not a permanent solution.

- **Quantitative batch metrics matter more than visual QC.** The current QC strategy
  (3×3 PNG grids) is useful for spot-checking but does not catch silent failures across
  heterogeneous datasets. Expanding machine-readable metrics output is higher value than
  improving the PNG layout.

- **Reproducibility infrastructure is missing and should be added.** The main function
  does not expose a `random_state` or `seed` parameter. This makes batch validation
  non-deterministic. Adding seed control is low-risk and high-value; it should be
  prioritised over geometric refinements.

- **Intensity blending uses hard replacement.** The current approach (`emb_vol[mask] =
syn_scaled[mask]`) produces sharp intensity boundaries. Boundary smoothing is a future
  enhancement, not a current requirement, but avoid making the replacement logic more
  entrenched in ways that would make boundary mixing harder to add later.

## 10. Violation protocol

If a requested change would violate a hard invariant in #2:

- do not implement the violating change silently
- explain which invariant would be broken
- propose the smallest alternative that satisfies the task without breaking the invariant
- if no safe alternative exists, stop and request explicit approval

If a validation threshold in #5 is breached:

- do not silently retune parameters to pass
- surface the breach in stdout and machine-readable metrics
- classify it as warning or hard failure using §5 defaults

## 11. Reproducibility rules

- All stochastic geometry perturbations must be sampled once per synthetic series and reused across timepoints.
- New stochastic features must expose or accept deterministic seed / random-state control before becoming default workflow.
- Do not introduce timepoint-to-timepoint random flicker in shape, texture, or placement.
- If a change adds randomness, document:
  - what is sampled
  - when it is sampled
  - whether it is per-case, per-series, per-timepoint, or per-voxel

## 12. QC visual invariants

QC outputs must continue to show:

- a visible narrow intracanalicular stem
- a clear neck at the porus
- a dominant extracanalicular CPA bulb at late timepoints
- no jagged high-frequency edge artifacts
- no CPA leakage into the intracanalicular side

Changes that preserve file generation but destroy these visual properties are regressions.
