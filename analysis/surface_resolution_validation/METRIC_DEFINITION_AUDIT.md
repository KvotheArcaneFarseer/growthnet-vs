# Surface Metric Definition Audit

Generated: 2026-07-18T12:32:28.647634+00:00

## Implementation

The current extractor is `scripts/extract_real_tumor_features.py` at git commit `b7db1cc55f708ed71f4a9b11da1aef5cc27e3f5e`.

- Surface area is computed in `_surface_area_mm2`.
- `skimage.measure.marching_cubes` is used on the largest connected component only.
- The binary component mask is converted to `float32`, contoured at `level=0.5`, and `spacing=tuple(spacing_mm)` is passed directly into marching cubes.
- Surface area is computed with `skimage.measure.mesh_surface_area(verts, faces)`.
- Because marching-cubes vertices are emitted in spacing-scaled coordinates, surface area is in physical `mm^2`.
- Anisotropic spacing is handled by passing the three header zooms to marching cubes. This is physically aware, assuming the NIfTI header zooms are correct.

## Formulas

- `volume_mm3 = mask_voxel_count * prod(voxel_spacing_mm)` for the whole mask.
- `largest_volume_mm3 = largest_component_voxel_count * prod(voxel_spacing_mm)` for surface-derived formulas.
- `surface_area_mm2 = mesh_surface_area(marching_cubes(largest_mask, level=0.5, spacing=spacing_mm))`.
- `sphericity = pi^(1/3) * (6 * largest_volume_mm3)^(2/3) / surface_area_mm2`.
- `compactness = surface_area_mm2^3 / largest_volume_mm3^2`.
- `surface_to_volume_ratio = surface_area_mm2 / largest_volume_mm3`.

## Resolution Sensitivity

Physical spacing is passed correctly into the mesh calculation, but these metrics remain resolution-sensitive. A nearest-neighbor label mask at 1.0 mm and the same anatomy resampled at 0.5 mm have different boundary stair-stepping and a different marching-cubes triangulation. Surface area, and formulas derived from it, can therefore change even if physical spacing is supplied correctly.

## Local Real Spacing Evidence

The best available local real feature table reports this spacing summary:

| index | voxel_spacing_mm_x | voxel_spacing_mm_y | voxel_spacing_mm_z |
| --- | --- | --- | --- |
| count | 291 | 291 | 291 |
| mean | 0.5 | 0.5 | 0.5 |
| std | 0 | 0 | 0 |
| min | 0.5 | 0.5 | 0.5 |
| 25% | 0.5 | 0.5 | 0.5 |
| 50% | 0.5 | 0.5 | 0.5 |
| 75% | 0.5 | 0.5 | 0.5 |
| max | 0.5 | 0.5 | 0.5 |

This supports using `0.5 x 0.5 x 0.5 mm` as the representative local real-data spacing for the normalized synthetic comparison. Source real masks are not available locally, so real features cannot be re-extracted under alternate spacing conditions in this pass.
