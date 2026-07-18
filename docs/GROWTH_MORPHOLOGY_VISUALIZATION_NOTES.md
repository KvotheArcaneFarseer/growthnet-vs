# Growth Morphology and Visualization Notes
**Prepared:** 2026-04-24  
**Status:** Research + observations. No pipeline code changed here.

---

## Part 1 — Visualization

### Current state

`_save_qc_png()` produces a 3×3 grid per timepoint:
- Rows: axial / coronal / sagittal slice
- Columns: Original MRI | Real segmentation overlay (red) | Embedded synthetic tumor (cyan)
- Slice selection: centroid of the **embedded** mask (not the real seg centroid)

This is good for verifying placement, but it has several weaknesses.

### Identified issues

**1. Windowing is case-global, not ROI-local**  
`mri_p1` and `mri_p99` are computed over the whole MRI volume. In thin-slice T2 vestibular imaging, the tumor region has a different intensity range than the whole brain. A tighter ROI window (e.g., percentile range within a bounding box around the tumor) would give better contrast in the embedded region.

**2. No ROI zoom panel**  
The current 3×3 grid shows full-slice views. At typical MRI slice sizes (300-500 px), the tumor region is 10-30 px — too small to evaluate embedding quality visually. A 4th column or a separate 3×1 panel showing 2× zoomed crops around the tumor centroid would be much more useful for QC.

**3. Slice selection tied to synthetic centroid, not anatomical center**  
When the synthetic tumor is placed off-axis, the centroid-selected slice may not show the real tumor at all. The QC panel should show the slice that maximizes BOTH real and synthetic visibility — the intersection of both centroids projected to each plane.

**4. Color choice: cyan/red on grey**  
Cyan (synthetic) and red (real) are distinguishable but cyan has low contrast against the light grey T2 signal. Blue-green or hot yellow would separate better. For grayscale printing, consider using transparency-weighted outlines rather than solid fills.

**5. No captions on subpanels**  
Each 3×3 image has no per-cell labels (no "Axial" / "Real seg" text). These are obvious to the developer but not to a clinician reviewer.

### Recommended QC panel improvements (priority order)

1. **Add ROI zoom column** — 2× to 3× crop around the embedded centroid, one per row. Minimal code change to `_save_qc_png`.
2. **Tighter window** — use percentile range within a bounding box around the placed mask, not global MRI range.
3. **Label rows and columns** — `matplotlib` text annotations or `imshow` title strings.
4. **Dual-centroid slice selection** — pick slice that intersects both real seg centroid and embedded centroid.
5. **Contour outlines** — instead of filled overlay, show a 1-px boundary contour of each mask using `skimage.segmentation.find_boundaries`. This preserves underlying MRI texture.

---

## Part 2 — Growth Morphology

### Current lollipop geometry (from `projects/vivit/src/data/synthetic.py`)

The synthetic VS tumor uses a "lollipop" model:
- **Stem** (intracanalicular component): thin cylinder along the IAC canal axis
- **Bulb** (extracanalicular / CPA component): sphere centred outside the canal
- Growth across timepoints: all dimensions scale uniformly by a growth factor

Verified geometry notes (from `LateDiceOrientationStrategy` docstring):
- `x_rel = -canal_coord` — canal occupies `x_rel >= 0`
- CPA bulb centred at `x_rel = -cpa_radius * 0.55`
- Negative local-z = IAC/fundus side; positive local-z = CPA bulb side

### Known morphological limitations

**1. Uniform scaling does not replicate VS growth behavior**  
Real VS tumors grow anisotropically: the CPA bulb enlarges more than the canalicular stem. A more realistic model would apply differential growth rates: `bulb_radius(t) = r0 * growth_factor^t` while `stem_length(t)` grows more slowly.

**2. Bulb-stem junction is geometrically sharp**  
The current model places a sphere and a cylinder with no blending at the neck. Real tumors have a smooth constriction at the fundus. Using a `skimage.morphology` dilation + erosion pass or an SDF-based blend would smooth this.

**3. No volume constraint against real tumor**  
The synthetic tumor's size at t=0 is fixed in mm (not calibrated to the real tumor's volume). For training data realism, the initial volume should be sampled from a distribution matching real VS size statistics.

**4. Orientation is globally constrained but not anatomy-confirmed**  
The Dice-based orientation strategy selects the sign that maximizes overlap with the real tumor mask. But it does not verify that the stem points into the IAC canal specifically — only that the whole lollipop overlaps with the real mask. For a more anatomically grounded placement, the stem axis should be confirmed against a registered skull-base atlas or the IAC direction in the segmentation.

### Recommended morphology improvements (priority order)

1. **Smooth neck transition** — blend bulb and stem with an SDF-weighted sigmoid rather than hard union. Implementable with `scipy.ndimage.distance_transform_edt` on the union mask.

2. **Differential growth rates** — expose `bulb_growth_factor` and `stem_growth_factor` separately. Default both to 1.0 for backward compat.

3. **Initial volume calibration** — add a `target_t0_volume_mm3` parameter. Scale the lollipop geometry so that placed volume at t=0 matches the target (or a fraction of the real seg volume).

4. **Intensity heterogeneity already implemented** — boundary feathering (`feather_mask_alpha`) and compartment-aware blending are in place. No changes needed here.

5. **Monotone growth enforcement** — current growth is monotone by design (scaling factor increases). But resampling and clipping can cause non-monotone placed voxel counts. The fix is to detect this post-placement and nudge the growth factor slightly (1-2% minimum increase per step) before saving.

---

## Part 3 — Feasibility Notes

### 3D visualization without pyvista

**Currently available:** nibabel, numpy, scikit-image, matplotlib, imageio  
**Not installed:** pyvista, nilearn, VTK

Headless 3D isosurface renders are feasible using:
```python
from skimage.measure import marching_cubes
import mpl_toolkits.mplot3d as mpl3d

verts, faces, normals, values = marching_cubes(embedded_mask, level=0.5, spacing=voxel_spacing)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, alpha=0.7, color='cyan')
ax.view_init(elev=30, azim=45)
plt.savefig('3d_view.png', dpi=150, bbox_inches='tight')
```

This gives clean renders at 2+ camera angles without any interactive display. Render quality is adequate for QC — not for publication.

### What napari adds vs headless matplotlib

| Feature | matplotlib headless | napari interactive |
|---------|--------------------|--------------------|
| 3D surface render | Yes (marching cubes + mpl3d) | Yes (+ transparency, lighting) |
| 2D slice viewer | Yes | Yes (+ synchronized crosshairs) |
| Segmentation overlay | Yes | Yes (+ toggle layers) |
| Headless automation | Yes | Requires display or virtual framebuffer |
| Scriptable camera angles | Yes | Partial (requires viewer object) |

For automated QC screenshots, matplotlib headless is preferred. Napari adds value for interactive inspection.

---

## Part 4 — Literature Notes

*These are directional pointers, not validated for this specific pipeline.*

**Synthetic lesion insertion evaluation:**
- Typical metrics: Dice (overlapping region quality), Hausdorff distance (boundary accuracy), centroid error (placement accuracy), visual realism scores (subjective expert rating)
- Reference task: MICCAI lesion detection challenges (brain metastasis, MS lesions)
- Key insight: Dice alone is insufficient for small lesions — a 20-voxel tumor can have Dice=0 due to 1-voxel offset

**Anatomy-aware orientation QC:**
- Canonical approach: register to MNI or atlas → verify axis alignment in standardized space
- For VS specifically: IAC direction is identifiable from bone CT; T2 thin-slice MRI does not directly show bone, so IAC direction is inferred from tumor shape
- The current Dice-maximization strategy is a proxy for this; not a substitute for atlas-based confirmation

**Principal axis angular error:**
- `_angle_deg(selected_axis_phys, real_long_axis_phys)` — computed but not surfaced in metrics
- This is the single most interpretable orientation quality metric for elongated tumors like VS
- Threshold for "correct orientation": < 15° (heuristic — no validated clinical threshold exists)

**Retained fraction interpretation:**
- A value of 0.95 or higher is generally acceptable for rigid-body embedding (some voxels are inevitably clipped at MRI boundaries or by skull/air masks)
- Values < 0.80 typically indicate the tumor was placed near the edge of the FOV or into a region outside the brain — a genuine failure

---

## Summary Table — Improvements by Effort

| Improvement | Effort | Impact | Status |
|-------------|--------|--------|--------|
| Fix non_monotone_growth bug | Trivial | High (hidden metric) | Done |
| Remove debug prints | Trivial | Medium (clean output) | Done |
| Split orientation warning into gap/margin | Small | High (interpretability) | Done |
| Add retained_fraction upper-bound check | Small | Medium | Done |
| ROI zoom column in QC PNG | Moderate | High (visual QC quality) | Not done |
| Per-panel labels in QC PNG | Small | Medium | Not done |
| Tighter windowing in QC PNG | Moderate | Medium | Not done |
| 3D isosurface renders (headless matplotlib) | Moderate | Medium | Not done |
| Smooth neck transition in lollipop | Moderate | Medium (realism) | Not done |
| Differential growth rates | Moderate | Medium (realism) | Not done |
| Principal axis angular error in metrics | Moderate | High (interpretability) | Not done |
| Initial volume calibration | Moderate | High (training realism) | Not done |
