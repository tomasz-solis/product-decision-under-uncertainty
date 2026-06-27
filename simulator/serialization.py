"""Deterministic JSON and markdown rendering shared across artifact generators.

Generated artifacts are content-addressed and diffed in CI, so every numeric value
must render identically on each run and platform. These helpers enforce that: floats
are rounded to a fixed precision, ``NaN`` is normalized away, and date-like values are
emitted as ISO strings. Centralizing them keeps ``reporting``, ``robustness``, and
``evidence`` from drifting apart with near-duplicate private copies.
"""

from __future__ import annotations

from datetime import date, datetime
from numbers import Integral, Real
from typing import Any

import pandas as pd


def stable_json_value(value: Any) -> Any:
    """Normalize floats and date-like values recursively before JSON serialization.

    Floats are rounded so generated artifacts diff cleanly, ``NaN`` collapses to
    ``None``, and ``date``/``datetime`` values become ISO strings. Mappings and lists
    are processed recursively; any other type is returned unchanged.
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if pd.isna(number):
            return None
        return round(number, 6)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: stable_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stable_json_value(item) for item in value]
    return value


def markdown_scalar(value: Any) -> str:
    """Return one stable markdown-cell string with fixed float precision."""

    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number = float(value)
        if pd.isna(number):
            return ""
        return f"{round(number, 6):.6f}".rstrip("0").rstrip(".")
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small markdown table without optional dependencies."""

    headers = [str(column) for column in frame.columns]
    rows = [[markdown_scalar(value) for value in row] for row in frame.values.tolist()]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
