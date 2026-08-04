from pathlib import Path

import pytest

from shared.provenance import get_git_commit, sha256_file
from shared.reporting import (
    markdown_table,
    markdown_table_from_dataframe,
    union_fieldnames,
    write_csv_rows,
    write_json,
    write_text,
)


pytestmark = pytest.mark.fast


def test_sha256_file_hashes_file_contents(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("growthnet\n", encoding="utf-8")

    assert sha256_file(sample) == "ac4eba783c79ff2432067bb257e4004fccc50ff9c2010b1d524cad3b21c0f906"


def test_get_git_commit_returns_configured_unknown_for_non_repo(tmp_path: Path):
    missing = tmp_path / "missing"

    assert get_git_commit(missing, unknown="NO_GIT") == "NO_GIT"


def test_write_text_and_json_create_parent_directories(tmp_path: Path):
    text_path = tmp_path / "nested" / "report.md"
    json_path = tmp_path / "nested" / "payload.json"

    write_text(text_path, "# Report\n")
    write_json(json_path, {"b": 2, "a": 1})

    assert text_path.read_text(encoding="utf-8") == "# Report\n"
    assert json_path.read_text(encoding="utf-8").splitlines()[1].strip() == '"a": 1,'


def test_write_csv_rows_uses_explicit_schema(tmp_path: Path):
    csv_path = tmp_path / "out" / "rows.csv"

    write_csv_rows(
        csv_path,
        [{"case_id": "case-1", "value": 3}],
        ["case_id", "value"],
    )

    assert csv_path.read_text(encoding="utf-8") == "case_id,value\ncase-1,3\n"


def test_union_fieldnames_preserves_preferred_then_first_seen_order():
    rows = [
        {"b": 2, "a": 1},
        {"c": 3, "a": 4},
    ]

    assert union_fieldnames(rows, preferred=["case_id", "a"]) == ["case_id", "a", "b", "c"]


def test_markdown_table_is_deterministic():
    assert markdown_table(["File", "Verdict"], [["a.py", "KEEP"]]) == (
        "| File | Verdict |\n"
        "| --- | --- |\n"
        "| a.py | KEEP |"
    )


def test_markdown_table_from_dataframe_formats_floats():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [
            {"case_id": "a", "value": 1.23456789},
            {"case_id": "b", "value": float("nan")},
        ]
    )

    assert markdown_table_from_dataframe(frame) == (
        "| case_id | value |\n"
        "| --- | --- |\n"
        "| a | 1.23457 |\n"
        "| b |  |"
    )
