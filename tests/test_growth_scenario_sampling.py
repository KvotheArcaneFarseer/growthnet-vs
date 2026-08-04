from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.fast


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "clinical_growth_law_validation"
    / "sample_growth_scenarios.py"
)
SPEC = importlib.util.spec_from_file_location("sample_growth_scenarios", SCRIPT_PATH)
sample_growth_scenarios = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sample_growth_scenarios)


def test_sample_scenarios_is_deterministic_and_four_visit(tmp_path: Path):
    rows_a = sample_growth_scenarios.sample_scenarios(
        n_patients=3,
        baseline_volume_mm3=100.0,
        clinical_growth_law="scenario_mixture_v1",
        seed=20260523,
    )
    rows_b = sample_growth_scenarios.sample_scenarios(
        n_patients=3,
        baseline_volume_mm3=100.0,
        clinical_growth_law="scenario_mixture_v1",
        seed=20260523,
    )

    assert rows_a == rows_b
    assert len(rows_a) == 12
    assert {row["timepoint"] for row in rows_a} == {"T1", "T2", "T3", "T4"}
    assert all(row["clinical_growth_law"] == "scenario_mixture_v1" for row in rows_a)

    out_csv = tmp_path / "nested" / "scenarios.csv"
    out_report = tmp_path / "nested" / "report.md"
    sample_growth_scenarios._write_csv(
        out_csv,
        rows_a,
        [
            "patient_id",
            "timepoint",
            "visit_day",
            "baseline_volume_mm3",
            "clinical_growth_law",
            "growth_law_scenario",
            "annual_volume_change_fraction",
            "target_volume_mm3",
        ],
    )
    sample_growth_scenarios.write_report(out_report, rows_a, out_csv)

    assert out_csv.read_text(encoding="utf-8").startswith("patient_id,timepoint,visit_day")
    assert "# Growth Scenario Sampling Audit" in out_report.read_text(encoding="utf-8")
