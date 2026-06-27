"""Unit tests for the shared serialization and validation leaf modules."""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd
import pytest

from simulator.serialization import markdown_scalar, markdown_table, stable_json_value
from simulator.validation import normalize_iso_date, validate_allowed_keys


def test_stable_json_value_rounds_floats_and_normalizes_nan() -> None:
    assert stable_json_value(1.123456789) == round(1.123456789, 6)
    assert stable_json_value(float("nan")) is None
    assert stable_json_value(True) is True
    assert stable_json_value(7) == 7
    assert isinstance(stable_json_value(7), int)


def test_stable_json_value_handles_datelike_and_nested_structures() -> None:
    payload = {
        "as_of": date(2026, 6, 24),
        "stamp": datetime(2026, 6, 24, 13, 30),
        "values": [1.0000001, {"inner": 2.0000004}],
        "label": "unchanged",
    }
    result = stable_json_value(payload)
    assert result["as_of"] == "2026-06-24"
    assert result["stamp"] == "2026-06-24T13:30:00"
    assert result["values"][0] == round(1.0000001, 6)
    assert result["values"][1]["inner"] == round(2.0000004, 6)
    assert result["label"] == "unchanged"


def test_markdown_scalar_formats_each_type() -> None:
    assert markdown_scalar(None) == ""
    assert markdown_scalar(float("nan")) == ""
    assert markdown_scalar(True) == "True"
    assert markdown_scalar(5) == "5"
    assert markdown_scalar(1.5000000) == "1.5"
    assert markdown_scalar(2.0) == "2"
    assert markdown_scalar("verbatim") == "verbatim"


def test_markdown_table_renders_header_separator_and_rows() -> None:
    frame = pd.DataFrame([{"Option": "stabilize_core", "EV": 1.5}])
    rendered = markdown_table(frame).splitlines()
    assert rendered[0] == "| Option | EV |"
    assert rendered[1] == "| --- | --- |"
    assert rendered[2] == "| stabilize_core | 1.5 |"


def test_validate_allowed_keys_accepts_subset_and_rejects_unknown() -> None:
    validate_allowed_keys({"a": 1}, {"a", "b"}, "section")
    with pytest.raises(ValueError, match=r"Unknown field\(s\) in 'section': x, z\."):
        validate_allowed_keys({"a": 1, "z": 2, "x": 3}, {"a"}, "section")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2026, 6, 24), "2026-06-24"),
        (datetime(2026, 6, 24, 9, 0), "2026-06-24"),
        ("2026-06-24", "2026-06-24"),
    ],
)
def test_normalize_iso_date_accepts_valid_inputs(value: object, expected: str) -> None:
    assert normalize_iso_date(value, "ctx") == expected


@pytest.mark.parametrize("value", ["2026-13-01", "not-a-date", 42, math.nan])
def test_normalize_iso_date_rejects_invalid_inputs(value: object) -> None:
    with pytest.raises(ValueError, match="must be an ISO date string"):
        normalize_iso_date(value, "ctx")
