# Longitudinal Variant Diversity Report

## Scope

This report measures same-patient/timepoint mask diversity between
independently seeded variants. It is an engineering diversity check, not a
clinical morphology validation.

## Summary

- Variant pairs evaluated: 4
- Successful pair comparisons: 4
- Median Dice among successful comparisons: 0.8822
- Pairwise CSV: `/Users/kvothearcane/Personal/Coding Projects/GrowthNet/analysis/clinical_growth_law_validation/variant_smoke_output/variant_diversity.csv`

Interpretation guide:

- Dice near 1.0 means variants are nearly identical in mask space.
- Lower Dice means stronger spatial/shape diversity.
- Dice alone does not prove anatomical realism.
