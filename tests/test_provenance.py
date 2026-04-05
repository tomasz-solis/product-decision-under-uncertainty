"""Tests for parameter provenance and assumption-manifest coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from simulator.config import load_config
from simulator.provenance import (
    build_assumption_manifest,
    load_assumption_registry,
    load_parameter_registry,
    validate_assumption_registry,
    validate_parameter_registry,
)

CONFIG_PATH = Path("simulator/config.yaml")
PARAMETER_REGISTRY_PATH = Path("simulator/parameter_registry.yaml")
ASSUMPTION_REGISTRY_PATH = Path("simulator/assumption_registry.yaml")


def test_parameter_registry_covers_every_config_parameter() -> None:
    """Each configured parameter should have a registry entry."""

    cfg = load_config(CONFIG_PATH)
    registry = load_parameter_registry(PARAMETER_REGISTRY_PATH)

    validated = validate_parameter_registry(registry, cfg["params"])
    assert len(validated) == len(cfg["params"])


def test_parameter_registry_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    """Duplicate registry rows should fail on load instead of silently overwriting."""

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "baseline_failure_rate:\n"
        "  unit: share_of_volume\n"
        "  business_meaning: example\n"
        "  distribution_type: tri\n"
        "  low: 0.02\n"
        "  mode: 0.05\n"
        "  high: 0.10\n"
        "  source_type: synthetic_case_assumption\n"
        "  source_reference: example\n"
        "  reason_for_range: example\n"
        "  owner: test\n"
        "  last_updated: 2026-04-05\n"
        "baseline_failure_rate:\n"
        "  unit: share_of_volume\n"
        "  business_meaning: example\n"
        "  distribution_type: tri\n"
        "  low: 0.03\n"
        "  mode: 0.05\n"
        "  high: 0.10\n"
        "  source_type: synthetic_case_assumption\n"
        "  source_reference: example\n"
        "  reason_for_range: example\n"
        "  owner: test\n"
        "  last_updated: 2026-04-05\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate YAML key detected"):
        load_parameter_registry(registry_path)


def test_parameter_registry_rejects_unknown_fields(tmp_path: Path) -> None:
    """Registry rows should fail on extra unknown keys."""

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "baseline_failure_rate:\n"
        "  unit: share_of_volume\n"
        "  business_meaning: example\n"
        "  distribution_type: tri\n"
        "  low: 0.02\n"
        "  mode: 0.05\n"
        "  high: 0.10\n"
        "  source_type: synthetic_case_assumption\n"
        "  source_reference: example\n"
        "  reason_for_range: example\n"
        "  owner: test\n"
        "  last_updated: 2026-04-05\n"
        "  typo_field: nope\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys"):
        load_parameter_registry(registry_path)


def test_parameter_registry_rejects_missing_rows(tmp_path: Path) -> None:
    """Dropping a parameter row should fail validation."""

    cfg = load_config(CONFIG_PATH)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        "baseline_failure_rate:\n"
        "  unit: share_of_volume\n"
        "  business_meaning: example\n"
        "  distribution_type: tri\n"
        "  low: 0.02\n"
        "  mode: 0.05\n"
        "  high: 0.10\n"
        "  source_type: synthetic_case_assumption\n"
        "  source_reference: example\n"
        "  reason_for_range: example\n"
        "  owner: test\n"
        "  last_updated: 2026-04-05\n",
        encoding="utf-8",
    )

    registry = load_parameter_registry(registry_path)
    with pytest.raises(ValueError, match="missing config parameters"):
        validate_parameter_registry(registry, cfg["params"])


def test_assumption_registry_matches_current_config() -> None:
    """The assumption registry should cover policy, analysis, dependency, and scenario settings."""

    cfg = load_config(CONFIG_PATH)
    registry = load_assumption_registry(ASSUMPTION_REGISTRY_PATH)

    validated = validate_assumption_registry(registry, cfg, cfg["params"])

    assert len(validated["simulation"]) == 3
    assert len(validated["decision_policy"]) == 3
    assert len(validated["analysis"]) == 3
    assert len(validated["dependencies"]) == 4
    assert {row["scenario_name"] for row in validated["scenarios"]} == {
        "mid_range_pressure",
        "reliability_crisis",
        "growth_friendly_recovery",
    }
    crisis = next(
        row for row in validated["scenarios"] if row["scenario_name"] == "reliability_crisis"
    )
    assert crisis["simulation_overrides"][0]["setting_name"] == "annual_volume"
    assert crisis["simulation_overrides"][0]["value"] == 220000.0


def test_assumption_manifest_includes_non_parameter_sections() -> None:
    """The published manifest should include all modeled assumption families."""

    cfg = load_config(CONFIG_PATH)
    parameter_registry = load_parameter_registry(PARAMETER_REGISTRY_PATH)
    assumption_registry = load_assumption_registry(ASSUMPTION_REGISTRY_PATH)

    manifest = build_assumption_manifest(cfg, parameter_registry, assumption_registry)

    assert "params" in manifest
    assert "simulation" in manifest
    assert "decision_policy" in manifest
    assert "analysis" in manifest
    assert "dependencies" in manifest
    assert "scenarios" in manifest
