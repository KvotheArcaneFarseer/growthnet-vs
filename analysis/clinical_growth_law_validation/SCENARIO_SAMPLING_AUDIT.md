# Growth Scenario Sampling Audit

## Scope

This no-MRI audit samples target-volume trajectories only. It does not
generate masks, paste tumors into MRI, or validate clinical realism.

## Scenario Counts

| Scenario | Count | T4 median mm3 | T4 min mm3 | T4 max mm3 |
|---|---:|---:|---:|---:|
| fast_growth | 44 | 1324.31 | 811.00 | 2686.99 |
| moderate_growth | 92 | 289.79 | 172.80 | 1064.80 |
| regression | 19 | 71.13 | 54.37 | 92.16 |
| slow_growth | 45 | 141.60 | 115.89 | 168.80 |

CSV: `/Users/kvothearcane/Personal/Coding Projects/GrowthNet/analysis/clinical_growth_law_validation/scenario_sampling_audit.csv`
