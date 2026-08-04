# Generator History

Generated: 2026-07-18T13:06:10.525307+00:00

## Relevant Git Log

```text
0457540 Add GrowthNet local roadmap, validation audits, and reliability updates
242631b Split orientation confidence and score-margin warnings
ccf9ffe (personal/main) Add AI-assisted development workflow (Claude + Codex)
383961f Codex lollipop geometry prototype
b7a5ca7 initial commit for src
```

## Diff Summary From First Committed Cohort Script To Current Worktree

```text
No committed diff for tracked files; worktree may contain uncommitted changes.
```

## Candidate Geometry-Affecting Changes

- `383961f` introduced lollipop support in `projects/vivit/src/data/synthetic.py`, but that early prototype used an origin bulb, intracanalicular cylinder, and CPA ellipsoid driven by generic ellipsoid radii. It did not include the later explicit `canal_base_radius_init`, `canal_length_init`, or `bulb_radius_init` API used by the pulled cohort script.
- `0457540` added `scripts/generate_synthetic_lollipop_cohort.py` to git with the current-style conservative geometry mapping.
- The pulled script under `rivanna_pull/scripts/` differs from the current script and is a better candidate for mask generation. It uses maturity `(target_volume_mm3 - 80) / 700`, wider jitter ranges, nonzero bulb radius proportional to maturity, and CLI defaults `--seed 123`, `--voxel_size_mm 1.0`, `--volume_tol_frac 0.03`.
- The current script uses maturity `(target_volume_mm3 - 120) / 1200`, smaller jitter, zero bulb radius until maturity >= 0.75, and default seed `20260426`.
- Current worktree `synthetic.py` contains uncommitted changes relative to committed history, including canal/bulb growth scale arguments. These do not explain single-timepoint t0 masks directly, but they demonstrate that source provenance is not pinned by the mask files.

## Trial Summary

| generator_variant | count | median | min | max |
| --- | --- | --- | --- | --- |
| current_worktree_generator | 10 | 0.887163 | 0.81804 | 0.918182 |
| pulled_rivanna_generator_script_with_current_synthetic_py | 10 | 1 | 1 | 1 |

## Pulled Script Difference Note

See local diff: current script differs from pulled script in maturity ramp, jitter ranges, bulb activation, guardrails, CLI defaults, and output naming.
