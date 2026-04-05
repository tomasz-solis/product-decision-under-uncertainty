"""Config loading, validation, and normalization helpers for the simulator."""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

from simulator.yaml_utils import load_yaml_mapping

DistName = Literal["tri", "uniform", "constant", "lognormal"]
SUPPORTED_POLICY_NAMES = {"guardrailed_expected_value"}

TOP_LEVEL_KEYS = {
    "project",
    "simulation",
    "decision_policy",
    "analysis",
    "dependencies",
    "scenarios",
    "params",
}
PROJECT_KEYS = {"seed", "model_version"}
SIMULATION_KEYS = {
    "n_worlds",
    "annual_volume",
    "volume",
    "time_horizon_years",
    "discount_rate_annual",
    "scenario",
}
DECISION_POLICY_KEYS = {
    "name",
    "minimum_p05_value_eur",
    "maximum_mean_regret_eur",
    "ev_tolerance_eur",
}
ANALYSIS_KEYS = {
    "sensitivity_materiality_threshold_abs_spearman",
    "sensitivity_max_rows_per_option",
    "decision_boundary_top_parameters",
}
DEPENDENCY_KEYS = {"method", "rank_correlations"}
SCENARIO_KEYS = {"label", "description", "parameter_overrides", "simulation_overrides"}
SCENARIO_SIMULATION_OVERRIDE_KEYS = {"annual_volume"}
PARAM_VALUE_FIELDS = {
    "constant": {"value"},
    "uniform": {"low", "high"},
    "tri": {"low", "mode", "high"},
    "lognormal": {"median", "p95"},
}

REQUIRED_PARAMS = {
    "baseline_failure_rate",
    "failure_to_churn_rel",
    "cost_per_failure_eur",
    "value_per_success_eur",
    "stabilize_failure_reduction",
    "extension_uptake",
    "extension_exposure_reduction",
    "extension_value_per_uptake_eur",
    "extension_loss_rate",
    "new_capability_uplift",
    "regression_event_prob",
    "regression_event_cost_eur",
    "stabilize_core_regression_prob_multiplier",
    "feature_extension_regression_prob_multiplier",
    "new_capability_regression_prob_multiplier",
    "stabilize_core_upfront_cost_eur",
    "stabilize_core_annual_maintenance_cost_eur",
    "feature_extension_upfront_cost_eur",
    "feature_extension_annual_maintenance_cost_eur",
    "new_capability_upfront_cost_eur",
    "new_capability_annual_maintenance_cost_eur",
    "do_nothing_drift_cost_eur",
    "stabilize_core_launch_delay_months",
    "stabilize_core_benefit_ramp_months",
    "stabilize_core_cost_overrun_multiplier",
    "feature_extension_launch_delay_months",
    "feature_extension_benefit_ramp_months",
    "feature_extension_cost_overrun_multiplier",
    "new_capability_launch_delay_months",
    "new_capability_benefit_ramp_months",
    "new_capability_cost_overrun_multiplier",
}

PROPORTION_PARAMS = {
    "baseline_failure_rate",
    "failure_to_churn_rel",
    "stabilize_failure_reduction",
    "extension_uptake",
    "extension_exposure_reduction",
    "extension_loss_rate",
    "new_capability_uplift",
    "regression_event_prob",
}

NON_NEGATIVE_PARAMS = {
    "cost_per_failure_eur",
    "value_per_success_eur",
    "extension_value_per_uptake_eur",
    "regression_event_cost_eur",
    "stabilize_core_regression_prob_multiplier",
    "feature_extension_regression_prob_multiplier",
    "new_capability_regression_prob_multiplier",
    "stabilize_core_upfront_cost_eur",
    "stabilize_core_annual_maintenance_cost_eur",
    "feature_extension_upfront_cost_eur",
    "feature_extension_annual_maintenance_cost_eur",
    "new_capability_upfront_cost_eur",
    "new_capability_annual_maintenance_cost_eur",
    "do_nothing_drift_cost_eur",
    "stabilize_core_launch_delay_months",
    "stabilize_core_benefit_ramp_months",
    "stabilize_core_cost_overrun_multiplier",
    "feature_extension_launch_delay_months",
    "feature_extension_benefit_ramp_months",
    "feature_extension_cost_overrun_multiplier",
    "new_capability_launch_delay_months",
    "new_capability_benefit_ramp_months",
    "new_capability_cost_overrun_multiplier",
}


@dataclass(frozen=True)
class ParamSpec:
    """Validated parameter specification."""

    dist: DistName
    low: float | None = None
    mode: float | None = None
    high: float | None = None
    value: float | None = None
    median: float | None = None
    p95: float | None = None


@dataclass(frozen=True)
class DecisionPolicyConfig:
    """Decision-policy settings that govern recommendation selection."""

    name: str
    minimum_p05_value_eur: float
    maximum_mean_regret_eur: float
    ev_tolerance_eur: float


@dataclass(frozen=True)
class AnalysisConfig:
    """Presentation and analytical thresholds for reporting."""

    sensitivity_materiality_threshold_abs_spearman: float
    sensitivity_max_rows_per_option: int
    decision_boundary_top_parameters: int


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return its mapping root."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = load_yaml_mapping(path)
    _validate_allowed_keys(cfg, TOP_LEVEL_KEYS, "config")
    return cfg


def get_seed(cfg: dict[str, Any]) -> int:
    """Read the project seed, falling back to a stable default."""

    project = cfg.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("'project' must be a mapping if provided.")
    _validate_allowed_keys(project, PROJECT_KEYS, "project")
    if "seed" in project:
        return _coerce_strict_int(project["seed"], "project.seed")
    return 42


def get_declared_model_version(cfg: dict[str, Any]) -> str:
    """Return the declared model version string from config."""

    project = cfg.get("project", {})
    if not isinstance(project, dict):
        raise ValueError("'project' must be a mapping if provided.")
    _validate_allowed_keys(project, PROJECT_KEYS, "project")
    if "model_version" in project:
        return str(project["model_version"])
    return "unversioned"


def get_model_version(cfg: dict[str, Any]) -> str:
    """Return the declared model version string for backward compatibility."""

    return get_declared_model_version(cfg)


def get_simulation_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return normalized simulation settings from the config."""

    simulation = cfg.get("simulation", {})
    if not isinstance(simulation, dict):
        raise ValueError("'simulation' must be a mapping if provided.")
    _validate_allowed_keys(simulation, SIMULATION_KEYS, "simulation")

    annual_volume = simulation.get("annual_volume")
    legacy_volume = simulation.get("volume")
    if annual_volume is None:
        if legacy_volume is not None:
            warnings.warn(
                "'simulation.volume' is deprecated; use 'simulation.annual_volume' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            annual_volume = legacy_volume
        else:
            annual_volume = 250_000

    settings = {
        "n_worlds": _coerce_strict_int(simulation.get("n_worlds", 20_000), "simulation.n_worlds"),
        "annual_volume": _coerce_strict_int(annual_volume, "simulation.annual_volume"),
        "time_horizon_years": _coerce_strict_int(
            simulation.get("time_horizon_years", 2),
            "simulation.time_horizon_years",
        ),
        "discount_rate_annual": _coerce_finite_float(
            simulation.get("discount_rate_annual", 0.08),
            "simulation.discount_rate_annual",
        ),
        "scenario": str(simulation.get("scenario", "mid_range_pressure")),
    }
    _validate_simulation_settings(settings)
    return settings


def get_decision_policy(cfg: dict[str, Any]) -> DecisionPolicyConfig:
    """Return the configured decision-policy settings."""

    policy = cfg.get("decision_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("'decision_policy' must be a mapping if provided.")
    _validate_allowed_keys(policy, DECISION_POLICY_KEYS, "decision_policy")

    parsed = DecisionPolicyConfig(
        name=str(policy.get("name", "guardrailed_expected_value")),
        minimum_p05_value_eur=_coerce_finite_float(
            policy.get("minimum_p05_value_eur", -300000.0),
            "decision_policy.minimum_p05_value_eur",
        ),
        maximum_mean_regret_eur=_coerce_finite_float(
            policy.get("maximum_mean_regret_eur", 450000.0),
            "decision_policy.maximum_mean_regret_eur",
        ),
        ev_tolerance_eur=_coerce_finite_float(
            policy.get("ev_tolerance_eur", 100000.0),
            "decision_policy.ev_tolerance_eur",
        ),
    )
    if parsed.name not in SUPPORTED_POLICY_NAMES:
        supported = ", ".join(sorted(SUPPORTED_POLICY_NAMES))
        raise ValueError(f"'decision_policy.name' must be one of: {supported}.")
    if parsed.maximum_mean_regret_eur < 0.0:
        raise ValueError("'decision_policy.maximum_mean_regret_eur' must be non-negative.")
    if parsed.ev_tolerance_eur < 0.0:
        raise ValueError("'decision_policy.ev_tolerance_eur' must be non-negative.")
    return parsed


def get_analysis_settings(cfg: dict[str, Any]) -> AnalysisConfig:
    """Return analysis thresholds used in reporting and the app."""

    analysis = cfg.get("analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError("'analysis' must be a mapping if provided.")
    _validate_allowed_keys(analysis, ANALYSIS_KEYS, "analysis")

    parsed = AnalysisConfig(
        sensitivity_materiality_threshold_abs_spearman=_coerce_finite_float(
            analysis.get("sensitivity_materiality_threshold_abs_spearman", 0.10),
            "analysis.sensitivity_materiality_threshold_abs_spearman",
        ),
        sensitivity_max_rows_per_option=_coerce_strict_int(
            analysis.get("sensitivity_max_rows_per_option", 3),
            "analysis.sensitivity_max_rows_per_option",
        ),
        decision_boundary_top_parameters=_coerce_strict_int(
            analysis.get("decision_boundary_top_parameters", 3),
            "analysis.decision_boundary_top_parameters",
        ),
    )
    if not (0.0 <= parsed.sensitivity_materiality_threshold_abs_spearman <= 1.0):
        raise ValueError(
            "'analysis.sensitivity_materiality_threshold_abs_spearman' must be between 0 and 1."
        )
    if parsed.sensitivity_max_rows_per_option <= 0:
        raise ValueError("'analysis.sensitivity_max_rows_per_option' must be positive.")
    if parsed.decision_boundary_top_parameters <= 0:
        raise ValueError("'analysis.decision_boundary_top_parameters' must be positive.")
    return parsed


def get_dependency_settings(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return normalized dependency targets keyed by parameter name."""

    dependencies = cfg.get("dependencies", {})
    if dependencies in ({}, None):
        return {}
    if not isinstance(dependencies, dict):
        raise ValueError("'dependencies' must be a mapping if provided.")
    _validate_allowed_keys(dependencies, DEPENDENCY_KEYS, "dependencies")

    method = dependencies.get("method", "gaussian_copula")
    if method != "gaussian_copula":
        raise ValueError("Only 'gaussian_copula' dependency modeling is supported.")

    rank_correlations = dependencies.get("rank_correlations", {})
    if not isinstance(rank_correlations, dict):
        raise ValueError("'dependencies.rank_correlations' must be a mapping.")

    normalized: dict[str, dict[str, float]] = {}
    for left, targets in rank_correlations.items():
        if not isinstance(targets, dict):
            raise ValueError(f"Dependency row for '{left}' must be a mapping.")
        normalized[str(left)] = {}
        for right, value in targets.items():
            corr = _coerce_finite_float(
                value,
                f"dependencies.rank_correlations.{left}.{right}",
            )
            if abs(corr) >= 1.0:
                raise ValueError(
                    f"Dependency correlation for '{left}'/'{right}' must stay inside (-1, 1)."
                )
            normalized[str(left)][str(right)] = corr
    return normalized


def get_scenario_metadata(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return labels and descriptions for each configured scenario."""

    scenarios = cfg.get("scenarios", {})
    if not isinstance(scenarios, dict):
        return {}

    metadata: dict[str, dict[str, str]] = {}
    for scenario_name, entry in scenarios.items():
        _validate_scenario_entry(str(scenario_name), entry)
        metadata[str(scenario_name)] = {
            "label": str(entry.get("label", str(scenario_name).replace("_", " ").title())),
            "description": str(entry.get("description", "")),
        }
    return metadata


def get_scenario_descriptions(cfg: dict[str, Any]) -> dict[str, str]:
    """Return scenario descriptions for reporting and the app."""

    metadata = get_scenario_metadata(cfg)
    return {name: details["description"] for name, details in metadata.items()}


def apply_scenario(cfg: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    """Return a copy of the config with one scenario override applied."""

    new_cfg = copy.deepcopy(cfg)
    _get_scenario_entry(new_cfg, scenario_name)

    params = new_cfg.get("params")
    simulation = new_cfg.get("simulation")
    if not isinstance(params, dict) or not params:
        raise ValueError("Config must contain a non-empty 'params' mapping.")
    if not isinstance(simulation, dict):
        raise ValueError("Config must contain a 'simulation' mapping.")

    for name, override in extract_scenario_parameter_overrides(new_cfg, scenario_name).items():
        if name not in params:
            raise KeyError(f"Scenario override references unknown param '{name}'.")
        params[name] = {**params[name], **override}

    simulation_overrides = extract_scenario_simulation_overrides(new_cfg, scenario_name)
    simulation.update(simulation_overrides)
    return new_cfg


def extract_scenario_parameter_overrides(
    cfg: dict[str, Any],
    scenario_name: str,
) -> dict[str, dict[str, Any]]:
    """Return only the parameter override portion of one scenario entry."""

    entry = _get_scenario_entry(cfg, scenario_name)
    overrides = entry.get("parameter_overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError(f"Scenario '{scenario_name}' parameter_overrides must be a mapping.")
    return {str(name): value for name, value in overrides.items()}


def extract_scenario_simulation_overrides(
    cfg: dict[str, Any],
    scenario_name: str,
) -> dict[str, Any]:
    """Return only the simulation override portion of one scenario entry."""

    entry = _get_scenario_entry(cfg, scenario_name)
    overrides = entry.get("simulation_overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError(f"Scenario '{scenario_name}' simulation_overrides must be a mapping.")
    _validate_allowed_keys(
        overrides,
        SCENARIO_SIMULATION_OVERRIDE_KEYS,
        f"scenarios.{scenario_name}.simulation_overrides",
    )
    return {str(name): value for name, value in overrides.items()}


def parse_param_specs(cfg: dict[str, Any]) -> dict[str, ParamSpec]:
    """Parse and validate parameter specs from the config."""

    params = cfg.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("Config must contain a non-empty 'params' mapping.")

    parsed: dict[str, ParamSpec] = {}
    for name, raw_spec in params.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Param '{name}' must be a mapping.")
        dist = raw_spec.get("dist")
        if dist not in PARAM_VALUE_FIELDS:
            raise ValueError(
                f"Param '{name}' has unsupported dist '{dist}'. "
                "Use tri, uniform, constant, or lognormal."
            )
        _validate_param_spec_keys(str(name), raw_spec, str(dist), allow_missing_dist=False)
        parsed[str(name)] = _parse_param_spec(str(name), str(dist), raw_spec)

    missing = REQUIRED_PARAMS - parsed.keys()
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Config is missing required parameters: {missing_list}")
    return parsed


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the full config contract before simulation begins."""

    _validate_allowed_keys(cfg, TOP_LEVEL_KEYS, "config")
    project = cfg.get("project", {})
    if project not in ({}, None):
        if not isinstance(project, dict):
            raise ValueError("'project' must be a mapping if provided.")
        _validate_allowed_keys(project, PROJECT_KEYS, "project")

    simulation = get_simulation_settings(cfg)
    param_specs = parse_param_specs(cfg)
    scenarios = cfg.get("scenarios", {})
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("Config must define at least one scenario.")
    if simulation["scenario"] not in scenarios:
        raise ValueError(
            f"Default scenario '{simulation['scenario']}' is not defined under 'scenarios'."
        )

    get_decision_policy(cfg)
    get_analysis_settings(cfg)
    dependencies = get_dependency_settings(cfg)
    _validate_dependencies(param_specs, dependencies)

    for scenario_name, entry in scenarios.items():
        _validate_scenario_entry(str(scenario_name), entry)
        _validate_scenario_parameter_overrides(str(scenario_name), entry, param_specs)
        merged_cfg = apply_scenario(cfg, str(scenario_name))
        get_simulation_settings(merged_cfg)
        parse_param_specs(merged_cfg)

    return cfg


def _validate_dependencies(
    param_specs: dict[str, ParamSpec],
    dependencies: dict[str, dict[str, float]],
) -> None:
    """Check that dependency targets reference known, non-constant parameters."""

    for left, targets in dependencies.items():
        if left not in param_specs:
            raise ValueError(f"Dependency references unknown parameter '{left}'.")
        if param_specs[left].dist == "constant":
            raise ValueError(f"Dependency parameter '{left}' must not use a constant distribution.")
        for right in targets:
            if right not in param_specs:
                raise ValueError(f"Dependency references unknown parameter '{right}'.")
            if left == right:
                raise ValueError("Dependency definitions must not include self-correlations.")
            if param_specs[right].dist == "constant":
                raise ValueError(
                    f"Dependency parameter '{right}' must not use a constant distribution."
                )


def _validate_scenario_entry(scenario_name: str, entry: Any) -> None:
    """Validate the shape of one scenario entry."""

    if not isinstance(entry, dict):
        raise TypeError(f"Scenario '{scenario_name}' must be a mapping.")
    _validate_allowed_keys(entry, SCENARIO_KEYS, f"scenarios.{scenario_name}")
    parameter_overrides = entry.get("parameter_overrides", {})
    simulation_overrides = entry.get("simulation_overrides", {})
    if not isinstance(parameter_overrides, dict):
        raise TypeError(f"Scenario '{scenario_name}' parameter_overrides must be a mapping.")
    if not isinstance(simulation_overrides, dict):
        raise TypeError(f"Scenario '{scenario_name}' simulation_overrides must be a mapping.")
    _validate_allowed_keys(
        simulation_overrides,
        SCENARIO_SIMULATION_OVERRIDE_KEYS,
        f"scenarios.{scenario_name}.simulation_overrides",
    )


def _validate_scenario_parameter_overrides(
    scenario_name: str,
    entry: dict[str, Any],
    param_specs: dict[str, ParamSpec],
) -> None:
    """Validate override schemas before one scenario is applied."""

    parameter_overrides = entry.get("parameter_overrides", {})
    if not isinstance(parameter_overrides, dict):
        raise TypeError(f"Scenario '{scenario_name}' parameter_overrides must be a mapping.")

    for param_name, override in parameter_overrides.items():
        if param_name not in param_specs:
            raise ValueError(f"Scenario override references unknown param '{param_name}'.")
        if not isinstance(override, dict):
            raise TypeError(
                f"Scenario override for '{scenario_name}.{param_name}' must be a mapping."
            )
        _validate_param_spec_keys(
            name=f"{scenario_name}.{param_name}",
            raw_spec=override,
            dist=str(override.get("dist", param_specs[param_name].dist)),
            allow_missing_dist=True,
        )


def _parse_param_spec(name: str, dist: str, raw_spec: dict[str, Any]) -> ParamSpec:
    """Validate a single parameter spec."""

    if dist == "constant":
        value = _coerce_finite_float(raw_spec.get("value"), f"params.{name}.value")
        _validate_values(name, [value])
        return ParamSpec(dist="constant", value=value)

    if dist == "uniform":
        low = _coerce_finite_float(raw_spec.get("low"), f"params.{name}.low")
        high = _coerce_finite_float(raw_spec.get("high"), f"params.{name}.high")
        if low > high:
            raise ValueError(f"Param '{name}' must satisfy low <= high.")
        _validate_values(name, [low, high])
        return ParamSpec(dist="uniform", low=low, high=high)

    if dist == "tri":
        low = _coerce_finite_float(raw_spec.get("low"), f"params.{name}.low")
        mode = _coerce_finite_float(raw_spec.get("mode"), f"params.{name}.mode")
        high = _coerce_finite_float(raw_spec.get("high"), f"params.{name}.high")
        if not (low <= mode <= high):
            raise ValueError(f"Param '{name}' must satisfy low <= mode <= high.")
        _validate_values(name, [low, mode, high])
        return ParamSpec(dist="tri", low=low, mode=mode, high=high)

    median = _coerce_finite_float(raw_spec.get("median"), f"params.{name}.median")
    p95 = _coerce_finite_float(raw_spec.get("p95"), f"params.{name}.p95")
    if median <= 0.0 or p95 <= 0.0:
        raise ValueError(f"Param '{name}' lognormal parameters must be positive.")
    if p95 < median:
        raise ValueError(f"Param '{name}' must satisfy median <= p95.")
    _validate_values(name, [median, p95])
    return ParamSpec(dist="lognormal", median=median, p95=p95)


def _validate_param_spec_keys(
    name: str,
    raw_spec: dict[str, Any],
    dist: str,
    allow_missing_dist: bool,
) -> None:
    """Reject unknown keys in parameter specs and override specs."""

    if dist not in PARAM_VALUE_FIELDS:
        raise ValueError(
            f"Param '{name}' has unsupported dist '{dist}'. "
            "Use tri, uniform, constant, or lognormal."
        )
    allowed = set(PARAM_VALUE_FIELDS[dist])
    if allow_missing_dist:
        allowed.add("dist")
    else:
        allowed.add("dist")
    _validate_allowed_keys(raw_spec, allowed, f"params.{name}")


def _get_scenario_entry(cfg: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    """Return one validated scenario entry from config."""

    scenarios = cfg.get("scenarios", {})
    if scenario_name not in scenarios:
        raise KeyError(f"Unknown scenario '{scenario_name}'. Available: {list(scenarios.keys())}")
    entry = scenarios.get(scenario_name)
    _validate_scenario_entry(scenario_name, entry)
    return entry


def _coerce_strict_int(value: Any, field_name: str) -> int:
    """Return one integer field and reject floats, booleans, and strings."""

    if value is None:
        raise ValueError(f"Missing required integer field '{field_name}'.")
    if isinstance(value, bool):
        raise ValueError(f"Field '{field_name}' must be an integer, not a boolean.")
    if not isinstance(value, Integral):
        raise ValueError(f"Field '{field_name}' must be an integer.")
    return int(value)


def _coerce_finite_float(value: Any, field_name: str) -> float:
    """Return one numeric field and reject booleans, NaN, or infinity."""

    if value is None:
        raise ValueError(f"Missing required numeric field '{field_name}'.")
    if isinstance(value, bool):
        raise ValueError(f"Field '{field_name}' must be numeric, not a boolean.")
    if not isinstance(value, Real):
        raise ValueError(f"Field '{field_name}' must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Field '{field_name}' must be a finite number.")
    return number


def _validate_values(name: str, values: list[float]) -> None:
    """Apply lightweight domain validation for well-known parameter types."""

    if name in PROPORTION_PARAMS and not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Param '{name}' must stay between 0 and 1.")
    if name in NON_NEGATIVE_PARAMS and not all(value >= 0.0 for value in values):
        raise ValueError(f"Param '{name}' must be non-negative.")


def _validate_simulation_settings(settings: dict[str, Any]) -> None:
    """Fail early for impossible simulation settings."""

    if settings["n_worlds"] <= 0:
        raise ValueError("'simulation.n_worlds' must be positive.")
    if settings["annual_volume"] <= 0:
        raise ValueError("'simulation.annual_volume' must be positive.")
    if settings["time_horizon_years"] <= 0:
        raise ValueError("'simulation.time_horizon_years' must be positive.")
    if settings["discount_rate_annual"] < 0.0:
        raise ValueError("'simulation.discount_rate_annual' must be non-negative.")


def _validate_allowed_keys(
    payload: dict[str, Any],
    allowed_keys: set[str],
    context: str,
) -> None:
    """Reject unknown keys in one config mapping."""

    unknown = sorted(str(key) for key in payload.keys() - allowed_keys)
    if unknown:
        raise ValueError(f"Unknown field(s) in '{context}': {', '.join(unknown)}.")
