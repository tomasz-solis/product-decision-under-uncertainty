"""Tests for the strict public-evidence profiling seam."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from simulator.evidence import profile_public_evidence, write_public_evidence_outputs


def test_profile_public_evidence_with_clean_csv_fixture(tmp_path: Path) -> None:
    """The profiler should validate schema and emit the richer profile payload."""

    data_dir = tmp_path / "public"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "region": ["es", "de"],
            "orders": [10, 12],
            "failure_rate": [0.03, 0.04],
        }
    )
    frame.to_csv(data_dir / "checkout.csv", index=False)
    (data_dir / "sources.yaml").write_text(
        "sources:\n"
        "  - source_id: checkout_public_sample\n"
        "    file_name: checkout.csv\n"
        "    publication: Example source\n"
        "    license: CC-BY\n"
        "    extraction_date: 2026-04-05\n"
        "    grain: daily\n"
        "    expected_schema:\n"
        "      date: date\n"
        "      region: category\n"
        "      orders: integer\n"
        "      failure_rate: float\n"
        "    assumption_families:\n"
        "      - baseline_failure_rate\n",
        encoding="utf-8",
    )

    profile = profile_public_evidence(data_dir, data_dir / "sources.yaml")

    assert profile["status"] == "profiled"
    assert profile["file_count"] == 1
    source = profile["sources"][0]
    assert source["duplicate_row_count"] == 0
    assert source["row_count"] == 2
    assert source["date_ranges"]["date"]["min"].startswith("2026-01-01")


def test_profile_public_evidence_uses_null_for_all_nan_numeric_columns(tmp_path: Path) -> None:
    """Numeric summaries should stay strict JSON even when a numeric column is fully missing."""

    data_dir = tmp_path / "public"
    data_dir.mkdir()
    frame = pd.DataFrame({"metric": [None, None], "date": ["2026-01-01", "2026-01-02"]})
    frame.to_csv(data_dir / "missing.csv", index=False)
    (data_dir / "sources.yaml").write_text(
        "sources:\n"
        "  - source_id: all_nan_metric\n"
        "    file_name: missing.csv\n"
        "    publication: Example source\n"
        "    license: CC-BY\n"
        "    extraction_date: 2026-04-05\n"
        "    grain: daily\n"
        "    expected_schema:\n"
        "      metric: float\n"
        "      date: date\n"
        "    assumption_families:\n"
        "      - extension_loss_rate\n",
        encoding="utf-8",
    )

    profile = profile_public_evidence(data_dir, data_dir / "sources.yaml")
    assert profile["sources"][0]["numeric_summaries"]["metric"]["mean"] is None

    json_path = tmp_path / "profile.json"
    markdown_path = tmp_path / "profile.md"
    write_public_evidence_outputs(profile, json_path, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sources"][0]["numeric_summaries"]["metric"]["mean"] is None
    assert "NaN" not in json_path.read_text(encoding="utf-8")


def test_manifest_file_mismatch_fails_fast(tmp_path: Path) -> None:
    """The profiler should fail if the manifest and raw files disagree."""

    data_dir = tmp_path / "public"
    data_dir.mkdir()
    (data_dir / "sources.yaml").write_text(
        "sources:\n"
        "  - source_id: checkout_public_sample\n"
        "    file_name: checkout.csv\n"
        "    publication: Example source\n"
        "    license: CC-BY\n"
        "    extraction_date: 2026-04-05\n"
        "    grain: daily\n"
        "    expected_schema:\n"
        "      date: date\n"
        "    assumption_families:\n"
        "      - baseline_failure_rate\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match"):
        profile_public_evidence(data_dir, data_dir / "sources.yaml")
