# Real Mask Acquisition Specification

## Minimal Useful Validation Subset

Start with 30 real cases:

- 10 small tumors: <100 mm3.
- 10 medium tumors: 100-1000 mm3.
- 10 large tumors: >=1000 mm3.

Prefer cases that match synthetic case IDs already present in the local real feature table so existing whole-mask features remain comparable.

## Required Files

- Real binary vestibular schwannoma segmentation mask for each case.
- NIfTI header or sidecar metadata preserving voxel spacing and affine.
- Case ID mapping to the current real feature table.

## MRI Requirement

Segmentation alone is sufficient for first-pass compartment geometry if manual landmarks are supplied. MRI is recommended, but not strictly required, to annotate porus/fundus and verify canal/CPA context.

## Required Annotations

At minimum, each case needs one of:

- Porus and fundus landmarks defining the canal axis and canal segment.
- A canal/IAC mask and a CPA region boundary.
- Expert-confirmed canal-axis vector plus a porus boundary point.

Manual annotation is required for the initial validation subset unless a trusted local atlas/landmarking pipeline is added and validated.

## Automatically Derivable After Annotation

- Intracanalicular volume.
- CPA/extracanalicular volume.
- Bulb-to-stem volume ratio.
- Stem width and length.
- Bulb offset from canal axis.
- Transition continuity at porus.
- Volume-stratified compartment trends.

## Questions Answerable Once Available

- Do synthetic small tumors overproduce or underproduce CPA bulb volume?
- Do large synthetic tumors become appropriately bulb-dominant?
- Is the synthetic stem width/length relationship in the observed real range?
- Is off-axis CPA growth comparable to real masks?
- Are transitions continuous at the porus without artificial discontinuities?
