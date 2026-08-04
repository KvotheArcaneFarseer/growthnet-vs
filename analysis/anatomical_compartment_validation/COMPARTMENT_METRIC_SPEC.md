# Compartment Metric Specification

## Generator Convention

The lollipop generator defines a local canal coordinate `x_rel = -canal_coord`.

- `x_rel > 0`: intracanalicular canal/fundus direction.
- `x_rel <= 0`: extracanalicular CPA side.
- `porus`: rounded opening spanning `x_rel in [-br, 0]`.
- `canal_body`: tapered stem spanning `x_rel in [0, cl]`.
- `fundus`: rounded cap spanning `x_rel in (cl, cl + ar]`.
- `cpa`: oblate bulb centered into negative `x_rel`.

## Derived Compartments

- `stem = (porus OR canal_body OR fundus) AND NOT transition`.
- `transition = porus AND cpa`.
- `bulb = cpa AND NOT transition`.

This preserves overlap at the porus as a transition region rather than double-counting it.

## Measurements

- Stem volume: stem voxel count times voxel volume.
- Stem length: max-min `x_rel` over stem voxels.
- Stem width: twice the 95th percentile radial distance from the stem axis.
- Stem principal direction/lengths: PCA over stem voxel centers in physical units.
- Bulb volume: bulb voxel count times voxel volume.
- Bulb axial extent: max-min `x_rel` over bulb voxels.
- Bulb width: twice the 95th percentile radial distance from the stem axis.
- Bulb centroid offset from stem axis: radial distance of bulb centroid in local perpendicular coordinates.
- Bulb-to-stem volume ratio: bulb volume divided by stem volume.
- Bulb-to-stem width ratio: bulb width divided by stem width.
- Total canal-axis extent: max-min `x_rel` over all tumor voxels.
- Transition smoothness proxy: non-zero transition volume and no axial gap in the derived union.

## Assumptions

- Synthetic masks are current authoritative masks.
- Manifest `seed`, `final_linear_scale_vox`, and rotations correspond to local masks.
- Voxel spacing is read from each NIfTI header.
- Metrics are synthetic plausibility checks, not clinical truth.
- Real compartment validation requires source masks and anatomical reference points.
