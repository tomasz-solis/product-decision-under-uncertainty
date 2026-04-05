"""Assumption-registry loading, validation, and export helpers."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from simulator.config import PARAM_VALUE_FIELDS
from simulator.yaml_utils import load_yaml_mapping

ALLOWED_SOURCE_TYPES = {
    "analytical_threshold",
    "elicited_range",
    "elicited_relationship",
    "illustrative_benchmark",
    "placeholder_for_real_telemetry",
    "portfolio_governance_assumption",
    "scenario_worldview",
    "synthetic_case_assumption",
}
PARAM_REQUIRED_FIELDS = {
    "unit",
    "business_meaning",
    "distribution_type",
    "source_type",
    "source_reference",
    "reason_for_range",
    "owner",
    "last_updated",
}
SCALAR_REQUIRED_FIELDS = {
    "unit",
    "business_meaning",
    "value",
    "source_type",
    "source_reference",
    "reason_for_choice",
    "owner",
    "last_updated",
}
DEPENDENCY_REQUIRED_FIELDS = {
    "left_parameter",
    "right_parameter",
    "rank_correlation",
    "business_meaning",
    "source_type",
    "source_reference",
    "reason_for_choice",
    "owner",
    "last_updated",
}
SCENARIO_REQUIRED_FIELDS = {
    "label",
    "description",
    "rationale",
    "source_type",
    "source_reference",
    "owner",
    "last_updated",
    "parameter_overrides",
    "simulation_overrides",
}
OPTIONAL_EVIDENCE_FIELDS = {"evidence_ids"}
EVIDENCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def load_parameter_registry(registry_path: str | Path) -> pd.DataFrame:
    """Load the parameter registry YAML into a dataframe."""

    raw = load_yaml_mapping(registry_path)
    if not raw:
        raise ValueError("Parameter registry must be a non-empty mapping keyed by parameter name.")

    rows: list[dict[str, Any]] = []
    for parameter_name, payload in raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Registry entry for '{parameter_name}' must be a mapping.")
        distribution_type = str(payload.get("distribution_type", ""))
        _validate_parameter_registry_keys(str(parameter_name), payload, distribution_type)
        row = {"parameter_name": str(parameter_name), **payload}
        _validate_source_type(str(row["source_type"]), str(parameter_name))
        _validate_non_empty(row["source_reference"], f"{parameter_name}.source_reference")
        _validate_non_empty(row["reason_for_range"], f"{parameter_name}.reason_for_range")
        _validate_non_empty(row["owner"], f"{parameter_name}.owner")
        row["last_updated"] = _normalize_iso_date(
            row["last_updated"], f"{parameter_name}.last_updated"
        )
        row["evidence_ids"] = _normalize_evidence_ids(
            row.get("evidence_ids", []),
            f"{parameter_name}.evidence_ids",
        )
        values = _extract_numeric_values(row, distribution_type, f"parameter '{parameter_name}'")
        row.update(values)
        row["chosen_range_or_value"] = _display_value(distribution_type, values)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("parameter_name").reset_index(drop=True)


def load_assumption_registry(registry_path: str | Path) -> dict[str, Any]:
    """Load the non-parameter assumption registry."""

    raw = load_yaml_mapping(registry_path)
    if not raw:
        raise ValueError("Assumption registry must be a non-empty mapping.")
    return raw


def validate_parameter_registry(
    registry: pd.DataFrame,
    config_params: dict[str, Any],
) -> pd.DataFrame:
    """Validate that every config parameter has a matching registry row."""

    registry_names = set(registry["parameter_name"])
    config_names = set(config_params)

    missing_from_registry = config_names - registry_names
    extra_in_registry = registry_names - config_names
    if missing_from_registry:
        missing_list = ", ".join(sorted(missing_from_registry))
        raise ValueError(f"Registry is missing config parameters: {missing_list}")
    if extra_in_registry:
        extra_list = ", ".join(sorted(extra_in_registry))
        raise ValueError(f"Registry has parameters not present in config.yaml: {extra_list}")

    for _, row in registry.iterrows():
        parameter_name = str(row["parameter_name"])
        config_spec = config_params[parameter_name]
        config_dist = str(config_spec["dist"])
        if config_dist != row["distribution_type"]:
            raise ValueError(
                f"Registry distribution for '{parameter_name}' does not match config: "
                f"{row['distribution_type']} vs {config_dist}"
            )
        expected_values = _extract_numeric_values(config_spec, config_dist, parameter_name)
        actual_row = {str(key): value for key, value in row.to_dict().items()}
        actual_values = _extract_numeric_values(actual_row, config_dist, parameter_name)
        if expected_values != actual_values:
            raise ValueError(
                f"Registry values for '{parameter_name}' do not match config: "
                f"{actual_values} vs {expected_values}"
            )
    return registry


def validate_assumption_registry(
    registry: dict[str, Any],
    cfg: dict[str, Any],
    config_params: dict[str, Any],
) -> dict[str, Any]:
    """Validate simulation, policy, dependency, analysis, and scenario assumptions."""

    _validate_allowed_keys(
        registry,
        {"simulation", "decision_policy", "analysis", "dependencies", "scenarios"},
        "assumption_registry",
    )
    return {
        "simulation": _validate_scalar_section(
            registry.get("simulation"),
            {
                "annual_volume": cfg.get("simulation", {}).get("annual_volume"),
                "time_horizon_years": cfg.get("simulation", {}).get("time_horizon_years"),
                "discount_rate_annual": cfg.get("simulation", {}).get("discount_rate_annual"),
            },
            section_name="simulation",
        ),
        "decision_policy": _validate_scalar_section(
            registry.get("decision_policy"),
            {
                "minimum_p05_value_eur": cfg.get("decision_policy", {}).get(
                    "minimum_p05_value_eur"
                ),
                "maximum_mean_regret_eur": cfg.get("decision_policy", {}).get(
                    "maximum_mean_regret_eur"
                ),
                "ev_tolerance_eur": cfg.get("decision_policy", {}).get("ev_tolerance_eur"),
            },
            section_name="decision_policy",
        ),
        "analysis": _validate_scalar_section(
            registry.get("analysis"),
            cfg.get("analysis", {}),
            section_name="analysis",
        ),
        "dependencies": _validate_dependency_section(
            registry.get("dependencies"),
            cfg.get("dependencies", {}),
        ),
        "scenarios": _validate_scenario_section(
            registry.get("scenarios"),
            cfg.get("scenarios", {}),
            cfg.get("simulation", {}),
            config_params=config_params,
        ),
    }


def build_assumption_manifest(
    cfg: dict[str, Any],
    parameter_registry: pd.DataFrame,
    assumption_registry: dict[str, Any],
) -> dict[str, Any]:
    """Return one unified, JSON-friendly assumption manifest."""

    validated_registry = validate_parameter_registry(parameter_registry, cfg["params"])
    validated_assumptions = validate_assumption_registry(
        assumption_registry,
        cfg,
        cfg["params"],
    )
    return {
        "params": validated_registry.to_dict(orient="records"),
        "simulation": validated_assumptions["simulation"],
        "decision_policy": validated_assumptions["decision_policy"],
        "analysis": validated_assumptions["analysis"],
        "dependencies": validated_assumptions["dependencies"],
        "scenarios": validated_assumptions["scenarios"],
    }


def _validate_scalar_section(
    section: Any,
    config_values: dict[str, Any],
    section_name: str,
) -> list[dict[str, Any]]:
    """Validate a scalar assumption section against config values."""

    if not isinstance(section, dict) or not section:
        raise ValueError(
            f"Assumption registry section '{section_name}' must be a non-empty mapping."
        )

    registry_names = set(section)
    config_names = set(config_values)
    if registry_names != config_names:
        missing = config_names - registry_names
        extra = registry_names - config_names
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"extra {sorted(extra)}")
        raise ValueError(
            f"Section '{section_name}' does not match config keys: {', '.join(details)}."
        )

    rows: list[dict[str, Any]] = []
    for key, payload in section.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Section '{section_name}.{key}' must be a mapping.")
        _validate_record_keys(
            payload,
            SCALAR_REQUIRED_FIELDS | OPTIONAL_EVIDENCE_FIELDS,
            f"{section_name}.{key}",
        )
        _validate_source_type(str(payload["source_type"]), f"{section_name}.{key}")
        _validate_non_empty(payload["source_reference"], f"{section_name}.{key}.source_reference")
        _validate_non_empty(payload["reason_for_choice"], f"{section_name}.{key}.reason_for_choice")
        _validate_non_empty(payload["owner"], f"{section_name}.{key}.owner")
        last_updated = _normalize_iso_date(
            payload["last_updated"], f"{section_name}.{key}.last_updated"
        )
        value = _coerce_finite_float(payload["value"], f"{section_name}.{key}.value")
        config_value = _coerce_finite_float(config_values[key], f"config.{section_name}.{key}")
        if value != config_value:
            raise ValueError(
                f"Section '{section_name}.{key}' does not match config: {value} vs {config_value}."
            )
        rows.append(
            {
                "assumption_name": str(key),
                "unit": str(payload["unit"]),
                "business_meaning": str(payload["business_meaning"]),
                "value": value,
                "source_type": str(payload["source_type"]),
                "source_reference": str(payload["source_reference"]),
                "reason_for_choice": str(payload["reason_for_choice"]),
                "owner": str(payload["owner"]),
                "last_updated": last_updated,
                "evidence_ids": _normalize_evidence_ids(
                    payload.get("evidence_ids", []),
                    f"{section_name}.{key}.evidence_ids",
                ),
            }
        )
    return rows


def _validate_dependency_section(
    section: Any,
    dependency_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate dependency provenance against config values."""

    if not isinstance(section, dict) or not section:
        raise ValueError("Assumption registry section 'dependencies' must be a non-empty mapping.")

    expected: dict[tuple[str, str], float] = {}
    rank_correlations = dependency_config.get("rank_correlations", {})
    if not isinstance(rank_correlations, dict):
        raise ValueError("Config dependencies must define a mapping of rank correlations.")
    for left_parameter, targets in rank_correlations.items():
        if not isinstance(targets, dict):
            raise ValueError(f"Config dependency row '{left_parameter}' must be a mapping.")
        for right_parameter, value in targets.items():
            expected[(str(left_parameter), str(right_parameter))] = _coerce_finite_float(
                value,
                f"dependencies.rank_correlations.{left_parameter}.{right_parameter}",
            )

    if len(section) != len(expected):
        raise ValueError(
            "Dependency registry does not match config relationships "
            f"({len(section)} rows vs {len(expected)} configured)."
        )

    rows: list[dict[str, Any]] = []
    for name, payload in section.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Dependency registry entry '{name}' must be a mapping.")
        _validate_record_keys(
            payload,
            DEPENDENCY_REQUIRED_FIELDS | OPTIONAL_EVIDENCE_FIELDS,
            f"dependencies.{name}",
        )
        _validate_source_type(str(payload["source_type"]), f"dependencies.{name}")
        _validate_non_empty(payload["source_reference"], f"dependencies.{name}.source_reference")
        _validate_non_empty(payload["reason_for_choice"], f"dependencies.{name}.reason_for_choice")
        _validate_non_empty(payload["owner"], f"dependencies.{name}.owner")
        last_updated = _normalize_iso_date(
            payload["last_updated"], f"dependencies.{name}.last_updated"
        )

        pair = (str(payload["left_parameter"]), str(payload["right_parameter"]))
        if pair not in expected:
            raise ValueError(f"Dependency registry entry '{name}' references unknown pair {pair}.")
        value = _coerce_finite_float(
            payload["rank_correlation"], f"dependencies.{name}.rank_correlation"
        )
        if value != expected[pair]:
            raise ValueError(
                f"Dependency registry entry '{name}' does not match config: "
                f"{value} vs {expected[pair]}."
            )
        rows.append(
            {
                "relationship_name": str(name),
                "left_parameter": pair[0],
                "right_parameter": pair[1],
                "rank_correlation": value,
                "business_meaning": str(payload["business_meaning"]),
                "source_type": str(payload["source_type"]),
                "source_reference": str(payload["source_reference"]),
                "reason_for_choice": str(payload["reason_for_choice"]),
                "owner": str(payload["owner"]),
                "last_updated": last_updated,
                "evidence_ids": _normalize_evidence_ids(
                    payload.get("evidence_ids", []),
                    f"dependencies.{name}.evidence_ids",
                ),
            }
        )
    return sorted(rows, key=lambda row: row["relationship_name"])


def _validate_scenario_section(
    section: Any,
    scenario_config: dict[str, Any],
    simulation_config: dict[str, Any],
    config_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate scenario provenance and override values."""

    if not isinstance(section, dict) or not section:
        raise ValueError("Assumption registry section 'scenarios' must be a non-empty mapping.")
    if set(section) != set(scenario_config):
        raise ValueError("Scenario registry keys do not match config scenario keys.")

    rows: list[dict[str, Any]] = []
    for scenario_name, payload in section.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Scenario registry entry '{scenario_name}' must be a mapping.")
        _validate_record_keys(
            payload,
            SCENARIO_REQUIRED_FIELDS | OPTIONAL_EVIDENCE_FIELDS,
            f"scenarios.{scenario_name}",
        )
        _validate_source_type(str(payload["source_type"]), f"scenarios.{scenario_name}")
        _validate_non_empty(
            payload["source_reference"], f"scenarios.{scenario_name}.source_reference"
        )
        _validate_non_empty(payload["rationale"], f"scenarios.{scenario_name}.rationale")
        _validate_non_empty(payload["owner"], f"scenarios.{scenario_name}.owner")
        last_updated = _normalize_iso_date(
            payload["last_updated"], f"scenarios.{scenario_name}.last_updated"
        )

        config_entry = scenario_config[scenario_name]
        if not isinstance(config_entry, dict):
            raise ValueError(f"Config scenario '{scenario_name}' must be a mapping.")
        if str(payload["label"]) != str(
            config_entry.get("label", str(scenario_name).replace("_", " ").title())
        ):
            raise ValueError(
                f"Scenario registry label for '{scenario_name}' does not match config."
            )
        if str(payload["description"]) != str(config_entry.get("description", "")):
            raise ValueError(
                f"Scenario registry description for '{scenario_name}' does not match config."
            )

        parameter_rows = _validate_parameter_overrides(
            scenario_name=scenario_name,
            registry_overrides=payload["parameter_overrides"],
            config_entry=config_entry,
            config_params=config_params,
        )
        simulation_rows = _validate_simulation_overrides(
            scenario_name=scenario_name,
            registry_overrides=payload["simulation_overrides"],
            config_entry=config_entry,
            simulation_config=simulation_config,
        )

        rows.append(
            {
                "scenario_name": str(scenario_name),
                "label": str(payload["label"]),
                "description": str(payload["description"]),
                "rationale": str(payload["rationale"]),
                "source_type": str(payload["source_type"]),
                "source_reference": str(payload["source_reference"]),
                "owner": str(payload["owner"]),
                "last_updated": last_updated,
                "evidence_ids": _normalize_evidence_ids(
                    payload.get("evidence_ids", []),
                    f"scenarios.{scenario_name}.evidence_ids",
                ),
                "parameter_overrides": parameter_rows,
                "simulation_overrides": simulation_rows,
            }
        )
    return sorted(rows, key=lambda row: row["scenario_name"])


def _validate_parameter_overrides(
    scenario_name: str,
    registry_overrides: Any,
    config_entry: dict[str, Any],
    config_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate scenario parameter overrides against config."""

    config_overrides = config_entry.get("parameter_overrides", {})
    if not isinstance(registry_overrides, dict) or not isinstance(config_overrides, dict):
        raise ValueError(f"Scenario '{scenario_name}' parameter_overrides must be mappings.")
    if set(registry_overrides) != set(config_overrides):
        raise ValueError(
            f"Scenario registry parameter_overrides for '{scenario_name}' do not match config."
        )

    rows = []
    for parameter_name, override_spec in registry_overrides.items():
        if parameter_name not in config_params:
            raise ValueError(
                "Scenario registry override "
                f"'{scenario_name}.{parameter_name}' references unknown parameter."
            )
        if not isinstance(override_spec, dict):
            raise ValueError(
                f"Scenario registry override '{scenario_name}.{parameter_name}' must be a mapping."
            )
        expected_dist = expected_distribution(
            str(config_params[parameter_name]["dist"]),
            config_overrides[parameter_name],
        )
        _validate_record_keys(
            override_spec,
            {"distribution_type", *PARAM_VALUE_FIELDS[expected_dist]},
            f"scenarios.{scenario_name}.{parameter_name}",
        )
        actual_dist = str(override_spec["distribution_type"])
        if actual_dist != expected_dist:
            raise ValueError(
                f"Scenario registry distribution for '{scenario_name}.{parameter_name}' "
                "does not match the effective override."
            )
        expected = _extract_numeric_values(
            {**config_params[parameter_name], **config_overrides[parameter_name]},
            expected_dist,
            f"scenarios.{scenario_name}.{parameter_name}",
        )
        actual = _extract_numeric_values(
            override_spec,
            actual_dist,
            f"scenarios.{scenario_name}.{parameter_name}",
        )
        if actual != expected:
            raise ValueError(
                f"Scenario registry values for '{scenario_name}.{parameter_name}' do not "
                f"match config: {actual} vs {expected}."
            )
        rows.append(
            {
                "parameter_name": str(parameter_name),
                "distribution_type": actual_dist,
                **actual,
            }
        )
    return sorted(rows, key=lambda row: row["parameter_name"])


def _validate_simulation_overrides(
    scenario_name: str,
    registry_overrides: Any,
    config_entry: dict[str, Any],
    simulation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate scenario simulation overrides against config."""

    config_overrides = config_entry.get("simulation_overrides", {})
    if not isinstance(registry_overrides, dict) or not isinstance(config_overrides, dict):
        raise ValueError(f"Scenario '{scenario_name}' simulation_overrides must be mappings.")
    if set(registry_overrides) != set(config_overrides):
        raise ValueError(
            f"Scenario registry simulation_overrides for '{scenario_name}' do not match config."
        )

    rows = []
    for setting_name, payload in registry_overrides.items():
        if not isinstance(payload, dict):
            raise ValueError(
                "Scenario registry simulation override "
                f"'{scenario_name}.{setting_name}' must be a mapping."
            )
        _validate_record_keys(
            payload,
            SCALAR_REQUIRED_FIELDS | OPTIONAL_EVIDENCE_FIELDS,
            f"scenarios.{scenario_name}.simulation_overrides.{setting_name}",
        )
        _validate_source_type(
            str(payload["source_type"]),
            f"scenarios.{scenario_name}.simulation_overrides.{setting_name}",
        )
        _validate_non_empty(
            payload["source_reference"],
            f"scenarios.{scenario_name}.simulation_overrides.{setting_name}.source_reference",
        )
        _validate_non_empty(
            payload["reason_for_choice"],
            f"scenarios.{scenario_name}.simulation_overrides.{setting_name}.reason_for_choice",
        )
        _validate_non_empty(
            payload["owner"],
            f"scenarios.{scenario_name}.simulation_overrides.{setting_name}.owner",
        )
        last_updated = _normalize_iso_date(
            payload["last_updated"],
            f"scenarios.{scenario_name}.simulation_overrides.{setting_name}.last_updated",
        )
        expected = _coerce_finite_float(
            config_overrides[setting_name],
            f"scenarios.{scenario_name}.simulation_overrides.{setting_name}",
        )
        actual = _coerce_finite_float(
            payload["value"],
            f"scenarios.{scenario_name}.simulation_overrides.{setting_name}.value",
        )
        if actual != expected:
            raise ValueError(
                f"Scenario registry simulation override '{scenario_name}.{setting_name}' "
                f"does not match config: {actual} vs {expected}."
            )
        base_value = _coerce_finite_float(
            simulation_config[setting_name],
            f"simulation.{setting_name}",
        )
        rows.append(
            {
                "setting_name": str(setting_name),
                "base_value": base_value,
                "value": actual,
                "unit": str(payload["unit"]),
                "business_meaning": str(payload["business_meaning"]),
                "source_type": str(payload["source_type"]),
                "source_reference": str(payload["source_reference"]),
                "reason_for_choice": str(payload["reason_for_choice"]),
                "owner": str(payload["owner"]),
                "last_updated": last_updated,
                "evidence_ids": _normalize_evidence_ids(
                    payload.get("evidence_ids", []),
                    f"scenarios.{scenario_name}.simulation_overrides.{setting_name}.evidence_ids",
                ),
            }
        )
    return sorted(rows, key=lambda row: row["setting_name"])


def _extract_numeric_values(
    payload: dict[str, Any],
    distribution_type: str,
    context: str,
) -> dict[str, float]:
    """Return the numeric values that define one assumption."""

    if distribution_type not in PARAM_VALUE_FIELDS:
        raise ValueError(f"{context} has unsupported distribution '{distribution_type}'.")
    values = {}
    for field_name in PARAM_VALUE_FIELDS[distribution_type]:
        if field_name not in payload:
            raise ValueError(
                f"{context} is missing numeric field '{field_name}' for distribution "
                f"'{distribution_type}'."
            )
        values[field_name] = _coerce_finite_float(payload[field_name], f"{context}.{field_name}")
    return values


def _display_value(distribution_type: str, values: dict[str, float]) -> str:
    """Return a human-readable value string from structured numeric fields."""

    if distribution_type == "constant":
        return f"{values['value']}"
    if distribution_type == "uniform":
        return f"{values['low']} / {values['high']}"
    if distribution_type == "tri":
        return f"{values['low']} / {values['mode']} / {values['high']}"
    return f"median {values['median']} / p95 {values['p95']}"


def _validate_parameter_registry_keys(
    parameter_name: str,
    payload: dict[str, Any],
    distribution_type: str,
) -> None:
    """Reject missing or extra keys in one parameter registry row."""

    if distribution_type not in PARAM_VALUE_FIELDS:
        raise ValueError(
            f"Registry entry for '{parameter_name}' has unsupported distribution "
            f"'{distribution_type}'."
        )
    _validate_record_keys(
        payload,
        PARAM_REQUIRED_FIELDS | PARAM_VALUE_FIELDS[distribution_type] | OPTIONAL_EVIDENCE_FIELDS,
        parameter_name,
    )


def _validate_record_keys(
    payload: dict[str, Any],
    allowed_keys: set[str],
    context: str,
) -> None:
    """Reject missing or extra keys in one registry payload."""

    missing = allowed_keys - OPTIONAL_EVIDENCE_FIELDS - set(payload)
    if missing:
        raise ValueError(f"Field set for '{context}' is missing keys: {sorted(missing)}.")
    extras = set(payload) - allowed_keys
    if extras:
        raise ValueError(f"Field set for '{context}' has unknown keys: {sorted(extras)}.")


def _validate_allowed_keys(
    payload: dict[str, Any],
    allowed_keys: set[str],
    context: str,
) -> None:
    """Reject unknown keys in one registry mapping."""

    unknown = sorted(str(key) for key in set(payload) - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown field(s) in '{context}': {', '.join(unknown)}.")


def _validate_source_type(source_type: str, context: str) -> None:
    """Reject unsupported provenance source types."""

    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"Field '{context}.source_type' must be one of {sorted(ALLOWED_SOURCE_TYPES)}."
        )


def _validate_non_empty(value: Any, context: str) -> None:
    """Reject empty provenance text fields."""

    if not str(value).strip():
        raise ValueError(f"Field '{context}' must be non-empty.")


def _normalize_iso_date(value: Any, context: str) -> str:
    """Return an ISO date string and reject malformed values."""

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f"Field '{context}' must be an ISO date string.")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:  # pragma: no cover - small branch
        raise ValueError(f"Field '{context}' must be an ISO date string.") from exc


def _normalize_evidence_ids(value: Any, context: str) -> list[str]:
    """Return a normalized evidence-id list and reject malformed values."""

    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError(f"Field '{context}' must be a list of evidence ids.")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise ValueError(f"Field '{context}' must not contain empty evidence ids.")
    if any(EVIDENCE_ID_RE.fullmatch(item) is None for item in normalized):
        raise ValueError(
            f"Field '{context}' must contain lowercase evidence ids with letters, digits, "
            "underscores, or hyphens only."
        )
    return normalized


def _coerce_finite_float(value: Any, context: str) -> float:
    """Return a finite float or raise a readable validation error."""

    if isinstance(value, bool):
        raise ValueError(f"Field '{context}' must be numeric, not a boolean.")
    number = float(value)
    if not pd.notna(number) or number in {float("inf"), float("-inf")}:
        raise ValueError(f"Field '{context}' must be a finite number.")
    return number


def expected_distribution(base_dist: str, override_spec: dict[str, Any]) -> str:
    """Return the effective distribution name after one override is applied."""

    if "dist" in override_spec:
        return str(override_spec["dist"])
    return base_dist
