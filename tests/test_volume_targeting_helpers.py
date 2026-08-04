from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.fast


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "volume_targeting"
    / "run_volume_targeting_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("run_volume_targeting_benchmark", SCRIPT_PATH)
volume_targeting = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(volume_targeting)


def test_volume_targeting_write_csv_preserves_row_schema(tmp_path: Path):
    out_csv = tmp_path / "nested" / "benchmark.csv"

    volume_targeting._write_csv(
        out_csv,
        [
            {
                "target_volume_mm3": 100.0,
                "achieved_volume_mm3": 101.0,
                "status": "OK",
            }
        ],
    )

    assert out_csv.read_text(encoding="utf-8") == (
        "target_volume_mm3,achieved_volume_mm3,status\n"
        "100.0,101.0,OK\n"
    )
