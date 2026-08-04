# Compartment Mapping Verification

Generated: 2026-07-18T12:57:02.485551+00:00

## Scope

This verifies whether the compartment analyzer correctly maps saved authoritative synthetic masks into canal/stem, transition, and CPA/bulb compartments. It does not modify generator geometry, masks, or authoritative features.

## Selected Cases

Selected case IDs: 439_0_0, 664_0_0, 611_2_466, 527_3_625, 126_5_1607, 527_2_591, 577_1_239, 579_1_198, 560_1_146, 690_0_0

The selection includes zero-bulb cases, low nonzero bulb-fraction cases, high nonzero bulb-fraction cases, mismatch cases, and multiple volume strata where available.

## Verification Table

| case_id | volume_bin | prior_bulb_fraction | regenerated_saved_dice | unassigned_fraction | unassigned_boundary_adjacent_fraction | classification | classification_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 439_0_0 | small_<100 | 0 | 0.888889 | 0 | 0 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 664_0_0 | medium_100_1000 | 0 | 0.918182 | 0 | 0 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 611_2_466 | medium_100_1000 | 0 | 0.81804 | 0.307895 | 0.576923 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 527_3_625 | large_>=1000 | 0.0281492 | 0.895457 | 0.0672062 | 0.759162 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 126_5_1607 | large_>=1000 | 0.0294464 | 0.894759 | 0.0624264 | 0.806604 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 527_2_591 | large_>=1000 | 0.0309322 | 0.895607 | 0.0584746 | 0.847826 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 577_1_239 | large_>=1000 | 0.0329245 | 0.880606 | 0.0619755 | 0.947917 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 579_1_198 | large_>=1000 | 0.0340265 | 0.885438 | 0.0478891 | 0.947368 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 560_1_146 | large_>=1000 | 0.035557 | 0.885056 | 0.063562 | 0.816832 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |
| 690_0_0 | large_>=1000 | 0.0374862 | 0.885229 | 0.0518192 | 0.925532 | MAJOR_MAPPING_ERROR | current generator plus manifest metadata does not reproduce saved mask |

## Classification Counts

| classification | case_count |
| --- | --- |
| MAJOR_MAPPING_ERROR | 10 |

## Main Checks

- Saved-mask reproduction from manifest metadata and current generator code succeeded for 0/10 selected cases at Dice >= 0.999.
- Median saved-vs-regenerated Dice across selected cases: 0.887.
- Zero-bulb selected cases with zero derived bulb and zero unassigned voxels under the current mapper: 2/3.
- Selected mismatch cases classified as major mapping/code-reconstruction errors: 8/8.
- Overlay PNGs are in `analysis/anatomical_compartment_validation/mapping_verification/overlays`.

## Decision

1. The current analyzer cannot yet be trusted for quantitative real-vs-synthetic compartment comparison. It reconstructs compartments from current generator code and manifest metadata, but those inputs do not reproduce the saved authoritative masks exactly.
2. The 191/261 zero-bulb result is plausible for cases whose manifest-derived `bulb_radius_init` is zero, but it is not fully verified until the exact mask-generating code path or saved compartment labels are recovered. In the selected zero-bulb examples, two clean cases mapped entirely as stem, while one high-mismatch medium case did not.
3. The 160/261 mismatch flags should be treated mostly as reconstruction/code-provenance artifacts, not true generator morphology signals. The saved masks and current code/metadata are not in exact correspondence.
4. Synthetic compartment metrics should not be used for real-vs-synthetic comparison yet. They are suitable only as exploratory diagnostics until the mapper is tied to exact saved-mask generation.
5. Analyzer correction is required before proceeding: either recover the exact generator version used for the saved masks, add per-voxel compartment-label outputs during future generation, or fit/derive compartments directly from saved mask geometry without assuming current generator internals.

## Reproducibility Commands

- `.venv/bin/python analysis/anatomical_compartment_validation/mapping_verification/verify_compartment_mapping.py`
- `PYTHONPYCACHEPREFIX=/tmp/growthnet_pycache python3 -m py_compile analysis/anatomical_compartment_validation/mapping_verification/verify_compartment_mapping.py`
- `.venv/bin/python -m pytest tests/test_anatomical_compartment_helpers.py -v`
- `.venv/bin/python -m pytest -m "fast and not slow" -v`
- `git diff --check`
