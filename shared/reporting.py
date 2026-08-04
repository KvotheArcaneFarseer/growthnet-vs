"""Shared report-writing helpers for local GrowthNet scripts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text, creating parent directories as needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    """Write a JSON payload with deterministic key ordering."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=indent, sort_keys=True), encoding="utf-8")


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV with an explicit schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> list[str]:
    """Return deterministic CSV fieldnames with preferred names first."""
    seen = set()
    fieldnames: list[str] = []
    for name in preferred:
        if name not in seen:
            fieldnames.append(str(name))
            seen.add(name)
    for row in rows:
        for name in row:
            if name not in seen:
                fieldnames.append(str(name))
                seen.add(name)
    return fieldnames


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Return a simple GitHub-flavored Markdown table."""
    header_cells = [str(header) for header in headers]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join("---" for _ in header_cells) + " |",
    ]
    for row in rows:
        cells = [str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown_table_from_dataframe(dataframe: Any, float_digits: int = 6) -> str:
    """Return a Markdown table from a pandas-like DataFrame.

    The helper intentionally uses duck typing so shared reporting does not need
    to import pandas at module import time.
    """
    if getattr(dataframe, "empty", False):
        return "_No rows._"

    display = dataframe.copy()
    for column in display.columns:
        values = display[column]
        dtype_kind = getattr(getattr(values, "dtype", None), "kind", "")
        if dtype_kind == "f":
            display[column] = values.map(
                lambda value: ""
                if not math.isfinite(float(value))
                else f"{float(value):.{float_digits}g}"
            )
        else:
            display[column] = values.astype(str)

    headers = [str(column) for column in display.columns]
    rows = [[record[column] for column in display.columns] for record in display.to_dict("records")]
    return markdown_table(headers, rows)
