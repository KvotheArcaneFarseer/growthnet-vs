# Local Testing

GrowthNet local tests are intended to run without Rivanna, SLURM, private
Downloads paths, or remote datasets.

## Test Groups

- `fast`: deterministic helper tests using tiny arrays and temporary CSV/NIfTI
  fixtures.
- `integration`: local tests that touch optional imaging dependencies or small
  generated NIfTI files.
- `slow`: reserved for expensive local checks; exclude these from routine smoke
  runs.

## Commands

Use `python3` in this repository. The current laptop environment does not expose
a `python` executable.

```bash
python3 -m pytest -m "fast and not slow"
python3 -m pytest -m "integration and not slow"
python3 -m pytest
```

For import/syntax validation when `pytest` is not installed:

```bash
python3 -m compileall tests
```

## Local Scope

These tests intentionally cover local-only behavior:

- batch case manifest parsing and flat summary schema
- failure report aggregation
- filesystem-safe case identifiers
- deterministic seed helpers
- physical-space axis conversion and anisotropic placement helpers
- embedding validation thresholds and report schema
- synthetic generator determinism and small target-volume calibration
- longitudinal metadata helpers and per-mask QC plumbing
- real-feature extractor case-ID, spacing, axis, and empty-mask behavior

They do not certify scientific validity, clinical realism, full cohort behavior,
or remote-data availability. Those remain separate validation tasks.

## Environment Notes

Some imports transitively initialize matplotlib/fontconfig. The root
`tests/conftest.py` sets cache-related environment variables so pytest runs keep
their cache writes inside the repository's `.pytest_cache` tree.
