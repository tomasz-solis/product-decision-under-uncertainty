"""Validation tests for config parsing and full-model completeness."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import pytest

from simulator.config import (
    apply_scenario,
    get_analysis_settings,
    get_decision_policy,
    get_dependency_settings,
    get_simulation_settings,
    load_config,
    parse_param_specs,
    validate_config,
)

CONFIG_PATH = "simulator/config.yaml"


def test_parse_param_specs_supports_current_distribution_types() -> None:
    """The parser should accept the distributions used by the checked-in model."""

    parsed = parse_param_specs(load_config(CONFIG_PATH))

    assert parsed["regression_event_cost_eur"].median == 150000.0
    assert parsed["regression_event_cost_eur"].p95 == 300000.0
    assert parsed["stabilize_core_upfront_cost_eur"].value == 800000.0
    assert parsed["stabilize_core_launch_delay_months"].mode == 5.0


def test_parse_param_specs_rejects_invalid_tri_order() -> None:
    """Triangular specs should fail fast when the order is invalid."""

    cfg = load_config(CONFIG_PATH)
    cfg["params"]["baseline_failure_rate"] = {
        "dist": "tri",
        "low": 0.30,
        "mode": 0.20,
        "high": 0.40,
    }

    with pytest.raises(ValueError, match="low <= mode <= high"):
        parse_param_specs(cfg)


def test_get_simulation_settings_supports_legacy_volume_alias() -> None:
    """The loader should still accept the deprecated `volume` field."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        settings = get_simulation_settings({"simulation": {"n_worlds": 10, "volume": 123456}})

    assert settings["annual_volume"] == 123456
    assert any("deprecated" in str(warning.message).lower() for warning in caught)


def test_duplicate_yaml_keys_fail_fast(tmp_path: Path) -> None:
    """Duplicate keys should not be silently overwritten during config load."""

    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(
        "project:\n"
        "  seed: 42\n"
        "simulation:\n"
        "  n_worlds: 1000\n"
        "params:\n"
        "  baseline_failure_rate:\n"
        "    dist: tri\n"
        "    low: 0.01\n"
        "    mode: 0.05\n"
        "    high: 0.10\n"
        "  baseline_failure_rate:\n"
        "    dist: tri\n"
        "    low: 0.02\n"
        "    mode: 0.05\n"
        "    high: 0.10\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate YAML key detected"):
        load_config(config_path)


def test_simulation_settings_reject_float_to_int_truncation() -> None:
    """Count-like simulation fields should reject floats instead of truncating them."""

    with pytest.raises(ValueError, match="simulation.time_horizon_years"):
        get_simulation_settings(
            {"simulation": {"n_worlds": 1000, "annual_volume": 250000, "time_horizon_years": 1.9}}
        )


def test_analysis_settings_reject_float_to_int_truncation() -> None:
    """Count-like analysis fields should reject floats instead of truncating them."""

    with pytest.raises(ValueError, match="analysis.sensitivity_max_rows_per_option"):
        get_analysis_settings({"analysis": {"sensitivity_max_rows_per_option": 3.9}})


def test_simulation_settings_reject_boolean_numeric_fields() -> None:
    """Boolean values should not sneak through numeric settings."""

    with pytest.raises(ValueError, match="not a boolean"):
        get_simulation_settings({"simulation": {"n_worlds": True, "annual_volume": 250000}})


def test_parse_param_specs_rejects_unknown_fields() -> None:
    """Parameter specs should fail on unknown keys instead of ignoring typos."""

    cfg = load_config(CONFIG_PATH)
    cfg["params"]["baseline_failure_rate"]["mdoe"] = 0.05

    with pytest.raises(ValueError, match="Unknown field"):
        parse_param_specs(cfg)


def test_validate_config_rejects_missing_required_param() -> None:
    """Missing model inputs should fail before simulation starts."""

    cfg = load_config(CONFIG_PATH)
    del cfg["params"]["stabilize_core_launch_delay_months"]

    with pytest.raises(ValueError, match="missing required parameters"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_simulation_settings() -> None:
    """Impossible simulation settings should fail early."""

    with pytest.raises(ValueError, match="annual_volume"):
        get_simulation_settings({"simulation": {"n_worlds": 10, "annual_volume": 0}})


def test_validate_config_rejects_unknown_default_scenario() -> None:
    """The configured default scenario must exist."""

    cfg = load_config(CONFIG_PATH)
    cfg["simulation"]["scenario"] = "unknown_world"

    with pytest.raises(ValueError, match="Default scenario"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_policy_name() -> None:
    """Unknown recommendation policies should fail fast."""

    cfg = load_config(CONFIG_PATH)
    cfg["decision_policy"]["name"] = "mystery_policy"

    with pytest.raises(ValueError, match="decision_policy.name"):
        validate_config(cfg)


def test_validate_config_rejects_invalid_analysis_threshold() -> None:
    """Analysis settings should stay inside the documented bounds."""

    cfg = load_config(CONFIG_PATH)
    cfg["analysis"]["sensitivity_materiality_threshold_abs_spearman"] = 1.5

    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_config(cfg)


def test_dependency_settings_reject_unknown_parameters() -> None:
    """Dependency targets must reference known parameters."""

    cfg = load_config(CONFIG_PATH)
    cfg["dependencies"]["rank_correlations"]["baseline_failure_rate"]["unknown_param"] = 0.5

    with pytest.raises(ValueError, match="unknown parameter"):
        validate_config(cfg)


def test_scenarios_can_override_simulation_scalars() -> None:
    """Scenario application should update both parameters and simulation scalars."""

    base = load_config(CONFIG_PATH)
    crisis = apply_scenario(base, "reliability_crisis")
    recovery = apply_scenario(base, "growth_friendly_recovery")

    assert get_simulation_settings(crisis)["annual_volume"] == 220000
    assert get_simulation_settings(recovery)["annual_volume"] == 300000
    assert (
        crisis["params"]["baseline_failure_rate"]["mode"]
        >= base["params"]["baseline_failure_rate"]["mode"]
    )
    assert (
        recovery["params"]["baseline_failure_rate"]["mode"]
        <= base["params"]["baseline_failure_rate"]["mode"]
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("minimum_p05_value_eur", float("nan")),
        ("maximum_mean_regret_eur", float("inf")),
        ("ev_tolerance_eur", float("-inf")),
    ],
)
def test_decision_policy_rejects_non_finite_numbers(field_name: str, value: float) -> None:
    """Policy thresholds should reject NaN and infinity before simulation begins."""

    cfg = load_config(CONFIG_PATH)
    cfg["decision_policy"][field_name] = value

    with pytest.raises(ValueError, match="finite number"):
        get_decision_policy(cfg)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("maximum_mean_regret_eur", -1.0),
        ("ev_tolerance_eur", -1.0),
    ],
)
def test_decision_policy_rejects_negative_non_downside_thresholds(
    field_name: str, value: float
) -> None:
    """Regret and EV-tolerance thresholds must stay inside the valid policy domain."""

    cfg = load_config(CONFIG_PATH)
    cfg["decision_policy"][field_name] = value

    with pytest.raises(ValueError, match="non-negative"):
        get_decision_policy(cfg)


def test_dependency_settings_reject_non_finite_correlations() -> None:
    """Dependency targets should reject NaN before linear algebra is reached."""

    cfg = load_config(CONFIG_PATH)
    cfg["dependencies"]["rank_correlations"]["baseline_failure_rate"]["cost_per_failure_eur"] = (
        float("nan")
    )

    with pytest.raises(ValueError, match="finite number"):
        get_dependency_settings(cfg)


def test_validate_config_rejects_non_finite_discount_rate() -> None:
    """The timing model should reject malformed discount-rate inputs."""

    cfg = load_config(CONFIG_PATH)
    cfg["simulation"]["discount_rate_annual"] = float("inf")

    with pytest.raises(ValueError, match="finite number"):
        validate_config(cfg)


def test_validate_config_accepts_current_checked_in_config() -> None:
    """The committed config should remain self-consistent."""

    cfg = copy.deepcopy(load_config(CONFIG_PATH))

    validate_config(cfg)
