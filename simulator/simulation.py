"""Simulation engine for the decision case study."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from math import pi, sin
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from simulator.config import (
    ParamSpec,
    apply_scenario,
    get_decision_policy,
    get_dependency_settings,
    get_seed,
    get_simulation_settings,
    load_config,
    parse_param_specs,
    validate_config,
)

OPTION_COLUMNS = [
    "do_nothing",
    "stabilize_core",
    "feature_extension",
    "new_capability",
]

OPTION_LABELS = {
    "do_nothing": "Do Nothing",
    "stabilize_core": "Stabilize Core",
    "feature_extension": "Feature Extension",
    "new_capability": "New Capability",
}

ParamOverride = dict[str, float | str]


@dataclass(frozen=True)
class TimingSummary:
    """Discounted timing factors for one intervention option."""

    benefit_years: np.ndarray
    active_years: np.ndarray
    residual_drift_years: np.ndarray
    overrun_multiplier: np.ndarray


def sample_params(
    n: int,
    param_specs: Mapping[str, ParamSpec],
    seed: int,
    dependencies: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Sample one dataframe of uncertain parameters."""

    rng = np.random.default_rng(seed)
    normalized_dependencies = dependencies or {}
    dependent_names = _dependency_parameter_names(param_specs, normalized_dependencies)

    data: dict[str, np.ndarray] = {}
    if dependent_names:
        uniforms = _sample_dependent_uniforms(
            n=n,
            parameter_names=dependent_names,
            dependencies=normalized_dependencies,
            rng=rng,
        )
        for index, name in enumerate(dependent_names):
            data[name] = _inverse_cdf_param(param_specs[name], uniforms[:, index])

    for name, spec in param_specs.items():
        if name in data:
            continue
        data[name] = _sample_param(spec, rng, n)

    return pd.DataFrame(data)


def _sample_param(spec: ParamSpec, rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample a parameter spec into a NumPy array."""

    if spec.dist == "tri":
        low = _require_param_value(spec.low, "low", spec.dist)
        mode = _require_param_value(spec.mode, "mode", spec.dist)
        high = _require_param_value(spec.high, "high", spec.dist)
        return rng.triangular(low, mode, high, size=size).astype(float)

    if spec.dist == "uniform":
        low = _require_param_value(spec.low, "low", spec.dist)
        high = _require_param_value(spec.high, "high", spec.dist)
        return rng.uniform(low, high, size=size).astype(float)

    if spec.dist == "constant":
        value = _require_param_value(spec.value, "value", spec.dist)
        return np.full(size, value, dtype=float)

    if spec.dist == "lognormal":
        return _inverse_cdf_param(spec, rng.random(size=size))

    raise ValueError(f"Unknown dist: {spec.dist}")


def run_simulation(
    config_path: str | Path,
    n_worlds: int | None = None,
    seed: int | None = None,
    scenario: str | None = None,
    param_overrides: dict[str, ParamOverride] | None = None,
) -> pd.DataFrame:
    """Run one scenario and return parameters plus option outcomes per world."""

    cfg = load_config(config_path)
    validate_config(cfg)
    simulation = get_simulation_settings(cfg)

    if n_worlds is not None:
        simulation["n_worlds"] = int(n_worlds)
    if scenario is not None:
        simulation["scenario"] = str(scenario)

    cfg = apply_scenario(cfg, simulation["scenario"])
    simulation = get_simulation_settings(cfg)
    if n_worlds is not None:
        simulation["n_worlds"] = int(n_worlds)
    if param_overrides:
        cfg = _apply_param_overrides(cfg, param_overrides)

    chosen_seed = int(get_seed(cfg) if seed is None else seed)
    seed_sequence = np.random.SeedSequence(chosen_seed)
    param_seed, shared_risk_seed = seed_sequence.spawn(2)

    param_specs = parse_param_specs(cfg)
    dependencies = get_dependency_settings(cfg)
    params = sample_params(
        simulation["n_worlds"],
        param_specs,
        seed=int(param_seed.generate_state(1)[0]),
        dependencies=dependencies,
    )

    horizon_years = float(simulation["time_horizon_years"])
    annual_volume = float(simulation["annual_volume"])
    annual_discount_rate = float(simulation["discount_rate_annual"])

    shared_risk_rng = np.random.default_rng(int(shared_risk_seed.generate_state(1)[0]))
    shared_risk_draws = shared_risk_rng.random(simulation["n_worlds"])

    outcomes = pd.DataFrame(
        {
            "do_nothing": simulate_option_do_nothing(
                params,
                horizon_years=horizon_years,
                annual_discount_rate=annual_discount_rate,
            ),
            "stabilize_core": simulate_option_stabilize_core(
                params,
                annual_volume=annual_volume,
                horizon_years=horizon_years,
                annual_discount_rate=annual_discount_rate,
                risk_rng=None,
                shared_risk_draws=shared_risk_draws,
            ),
            "feature_extension": simulate_option_feature_extension(
                params,
                annual_volume=annual_volume,
                horizon_years=horizon_years,
                annual_discount_rate=annual_discount_rate,
                risk_rng=None,
                shared_risk_draws=shared_risk_draws,
            ),
            "new_capability": simulate_option_new_capability(
                params,
                annual_volume=annual_volume,
                horizon_years=horizon_years,
                annual_discount_rate=annual_discount_rate,
                risk_rng=None,
                shared_risk_draws=shared_risk_draws,
            ),
        }
    )
    outcomes["scenario"] = simulation["scenario"]
    return pd.concat([params, outcomes], axis=1)


def run_all_scenarios(
    config_path: str | Path,
    n_worlds: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run every declared scenario and return one summary row per option."""

    cfg = load_config(config_path)
    validate_config(cfg)
    scenarios = cfg.get("scenarios", {})
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("No scenarios found in config under 'scenarios'.")

    rows: list[dict[str, Any]] = []
    policy = get_decision_policy(cfg)
    for scenario_name in scenarios:
        from simulator.analytics import decision_diagnostics, summarize_results
        from simulator.policy import build_policy_eligibility_table, select_recommendation

        df = run_simulation(
            config_path,
            n_worlds=n_worlds,
            seed=seed,
            scenario=scenario_name,
        )
        summary = summarize_results(df)
        diagnostics = decision_diagnostics(df)
        recommendation = select_recommendation(summary, diagnostics, policy)
        eligibility = build_policy_eligibility_table(summary, diagnostics, policy)
        merged = summary.merge(diagnostics, on="option", how="left")
        merged = merged.merge(
            eligibility[["option", "eligible", "failure_reason"]],
            on="option",
            how="left",
        )
        merged.insert(0, "scenario", scenario_name)
        merged["selected_option"] = recommendation.selected_option
        merged["selected_option_label"] = OPTION_LABELS[recommendation.selected_option]
        merged["binding_constraint"] = recommendation.binding_constraint
        rows.extend(merged.to_dict("records"))

    return pd.DataFrame(rows)


def simulate_option_do_nothing(
    p: pd.DataFrame,
    horizon_years: float,
    annual_discount_rate: float = 0.0,
) -> np.ndarray:
    """Model the cost of carrying the current path forward."""

    discounted_years = _full_horizon_discounted_years(horizon_years, annual_discount_rate)
    return -(p["do_nothing_drift_cost_eur"].to_numpy() * discounted_years)


def simulate_option_stabilize_core(
    p: pd.DataFrame,
    annual_volume: float,
    horizon_years: float,
    annual_discount_rate: float,
    risk_rng: np.random.Generator | None,
    shared_risk_draws: np.ndarray | None = None,
) -> np.ndarray:
    """Model the core-stability investment with launch delay and ramp."""

    timing = _option_timing_summary(
        p,
        option_name="stabilize_core",
        horizon_years=horizon_years,
        annual_discount_rate=annual_discount_rate,
    )
    baseline_failure = p["baseline_failure_rate"].to_numpy()
    reduction = p["stabilize_failure_reduction"].to_numpy()
    improvement = baseline_failure * reduction
    recovered_value = (
        annual_volume * timing.benefit_years * improvement * p["value_per_success_eur"].to_numpy()
    )
    avoided_failure_cost = (
        annual_volume * timing.benefit_years * improvement * p["cost_per_failure_eur"].to_numpy()
    )
    avoided_churn_cost = (
        annual_volume
        * timing.benefit_years
        * improvement
        * p["failure_to_churn_rel"].to_numpy()
        * p["value_per_success_eur"].to_numpy()
    )
    regression_cost = _sample_regression_cost(
        p,
        option_name="stabilize_core",
        risk_rng=risk_rng,
        shared_risk_draws=shared_risk_draws,
    )
    upfront = p["stabilize_core_upfront_cost_eur"].to_numpy() * timing.overrun_multiplier
    maintenance = p["stabilize_core_annual_maintenance_cost_eur"].to_numpy() * timing.active_years
    residual_drift = p["do_nothing_drift_cost_eur"].to_numpy() * timing.residual_drift_years

    return (
        recovered_value
        + avoided_failure_cost
        + avoided_churn_cost
        - upfront
        - maintenance
        - residual_drift
        - regression_cost
    )


def simulate_option_feature_extension(
    p: pd.DataFrame,
    annual_volume: float,
    horizon_years: float,
    annual_discount_rate: float,
    risk_rng: np.random.Generator | None,
    shared_risk_draws: np.ndarray | None = None,
) -> np.ndarray:
    """Model the feature-extension option with adoption ramp."""

    timing = _option_timing_summary(
        p,
        option_name="feature_extension",
        horizon_years=horizon_years,
        annual_discount_rate=annual_discount_rate,
    )
    discounted_horizon_years = _full_horizon_discounted_years(horizon_years, annual_discount_rate)
    baseline_failure = p["baseline_failure_rate"].to_numpy()
    uptake = p["extension_uptake"].to_numpy()
    exposure_reduction = p["extension_exposure_reduction"].to_numpy()
    improvement = baseline_failure * uptake * exposure_reduction

    recovered_value = (
        annual_volume * timing.benefit_years * improvement * p["value_per_success_eur"].to_numpy()
    )
    avoided_failure_cost = (
        annual_volume * timing.benefit_years * improvement * p["cost_per_failure_eur"].to_numpy()
    )
    avoided_churn_cost = (
        annual_volume
        * timing.benefit_years
        * improvement
        * p["failure_to_churn_rel"].to_numpy()
        * p["value_per_success_eur"].to_numpy()
    )

    gross_extension_value = (
        annual_volume
        * timing.benefit_years
        * uptake
        * p["extension_value_per_uptake_eur"].to_numpy()
    )
    realized_extension_value = gross_extension_value * (1.0 - p["extension_loss_rate"].to_numpy())

    regression_cost = _sample_regression_cost(
        p,
        option_name="feature_extension",
        risk_rng=risk_rng,
        shared_risk_draws=shared_risk_draws,
    )
    upfront = p["feature_extension_upfront_cost_eur"].to_numpy() * timing.overrun_multiplier
    maintenance = (
        p["feature_extension_annual_maintenance_cost_eur"].to_numpy() * timing.active_years
    )
    ongoing_drift = p["do_nothing_drift_cost_eur"].to_numpy() * discounted_horizon_years

    return (
        recovered_value
        + avoided_failure_cost
        + avoided_churn_cost
        + realized_extension_value
        - upfront
        - maintenance
        - ongoing_drift
        - regression_cost
    )


def simulate_option_new_capability(
    p: pd.DataFrame,
    annual_volume: float,
    horizon_years: float,
    annual_discount_rate: float,
    risk_rng: np.random.Generator | None,
    shared_risk_draws: np.ndarray | None = None,
) -> np.ndarray:
    """Model the new-capability option with delayed launch and rollout."""

    timing = _option_timing_summary(
        p,
        option_name="new_capability",
        horizon_years=horizon_years,
        annual_discount_rate=annual_discount_rate,
    )
    discounted_horizon_years = _full_horizon_discounted_years(horizon_years, annual_discount_rate)

    base_successes = (
        annual_volume * timing.benefit_years * (1.0 - p["baseline_failure_rate"].to_numpy())
    )
    uplift_value = (
        base_successes
        * p["value_per_success_eur"].to_numpy()
        * p["new_capability_uplift"].to_numpy()
    )
    regression_cost = _sample_regression_cost(
        p,
        option_name="new_capability",
        risk_rng=risk_rng,
        shared_risk_draws=shared_risk_draws,
    )
    upfront = p["new_capability_upfront_cost_eur"].to_numpy() * timing.overrun_multiplier
    maintenance = p["new_capability_annual_maintenance_cost_eur"].to_numpy() * timing.active_years
    ongoing_drift = p["do_nothing_drift_cost_eur"].to_numpy() * discounted_horizon_years

    return uplift_value - upfront - maintenance - ongoing_drift - regression_cost


def _apply_param_overrides(
    cfg: dict[str, Any],
    param_overrides: dict[str, ParamOverride],
) -> dict[str, Any]:
    """Return a config copy with in-memory parameter overrides applied."""

    new_cfg = {
        **cfg,
        "params": {name: dict(spec) for name, spec in cfg.get("params", {}).items()},
    }
    params = new_cfg.get("params", {})
    if not isinstance(params, dict) or not params:
        raise ValueError("Config must contain a non-empty 'params' mapping.")

    for param_name, override in param_overrides.items():
        if param_name not in params:
            raise ValueError(f"Override references unknown param '{param_name}'.")
        if not isinstance(override, dict):
            raise ValueError(f"Override for '{param_name}' must be a mapping.")
        if "dist" in override:
            params[param_name] = dict(override)
        else:
            params[param_name] = {**params[param_name], **override}

    parse_param_specs(new_cfg)
    return new_cfg


def _sample_regression_cost(
    p: pd.DataFrame,
    option_name: str,
    risk_rng: np.random.Generator | None,
    shared_risk_draws: np.ndarray | None = None,
) -> np.ndarray:
    """Sample event-based regression costs with optional shared latent draws."""

    probability = p["regression_event_prob"].to_numpy()
    multiplier_name = f"{option_name}_regression_prob_multiplier"
    multiplier = p[multiplier_name].to_numpy() if multiplier_name in p.columns else np.ones(len(p))
    event_probability = np.clip(probability * multiplier, 0.0, 1.0)
    if shared_risk_draws is not None:
        if len(shared_risk_draws) != len(p):
            raise ValueError("Shared risk draws must match the number of simulated worlds.")
        event_draws = shared_risk_draws
    elif risk_rng is not None:
        event_draws = risk_rng.random(len(p))
    else:  # pragma: no cover - defensive branch
        raise ValueError("Provide either a risk RNG or shared risk draws.")
    event_happens = event_draws < event_probability
    severity = p["regression_event_cost_eur"].to_numpy()
    return event_happens.astype(float) * severity


def _option_timing_summary(
    p: pd.DataFrame,
    option_name: str,
    horizon_years: float,
    annual_discount_rate: float,
) -> TimingSummary:
    """Return discounted timing factors for one intervention option."""

    horizon_months = max(int(round(horizon_years * 12.0)), 1)
    weights = _discount_weights(horizon_months, annual_discount_rate)
    benefit_years = np.zeros(len(p), dtype=float)
    active_years = np.zeros(len(p), dtype=float)
    residual_drift_years = np.zeros(len(p), dtype=float)

    launch_delays = p[f"{option_name}_launch_delay_months"].to_numpy()
    ramp_months = p[f"{option_name}_benefit_ramp_months"].to_numpy()
    overrun_multiplier = p[f"{option_name}_cost_overrun_multiplier"].to_numpy()

    for index, (delay_months, ramp) in enumerate(zip(launch_delays, ramp_months, strict=True)):
        benefit_profile, active_profile = _monthly_timing_profiles(
            launch_delay_months=float(delay_months),
            benefit_ramp_months=float(ramp),
            horizon_months=horizon_months,
        )
        benefit_years[index] = float(np.dot(benefit_profile, weights) / 12.0)
        active_years[index] = float(np.dot(active_profile, weights) / 12.0)
        residual_drift_years[index] = float(np.dot(1.0 - benefit_profile, weights) / 12.0)

    return TimingSummary(
        benefit_years=benefit_years,
        active_years=active_years,
        residual_drift_years=residual_drift_years,
        overrun_multiplier=overrun_multiplier,
    )


def _monthly_timing_profiles(
    launch_delay_months: float,
    benefit_ramp_months: float,
    horizon_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return benefit and active profiles on a monthly grid."""

    month_end = np.arange(1, horizon_months + 1, dtype=float)
    active_profile = (month_end > launch_delay_months).astype(float)

    if benefit_ramp_months <= 0.0:
        benefit_profile = active_profile.copy()
    else:
        benefit_profile = np.clip(
            (month_end - launch_delay_months) / benefit_ramp_months,
            0.0,
            1.0,
        )
    return benefit_profile, active_profile


def _full_horizon_discounted_years(
    horizon_years: float,
    annual_discount_rate: float,
) -> float:
    """Return the discounted number of effective years across the full horizon."""

    horizon_months = max(int(round(horizon_years * 12.0)), 1)
    return float(_discount_weights(horizon_months, annual_discount_rate).sum() / 12.0)


def _discount_weights(horizon_months: int, annual_discount_rate: float) -> np.ndarray:
    """Return monthly discount weights for the evaluation horizon."""

    if annual_discount_rate == 0.0:
        return np.ones(horizon_months, dtype=float)

    monthly_discount_rate = (1.0 + annual_discount_rate) ** (1.0 / 12.0) - 1.0
    periods = np.arange(1, horizon_months + 1, dtype=float)
    return 1.0 / np.power(1.0 + monthly_discount_rate, periods)


def _dependency_parameter_names(
    param_specs: Mapping[str, ParamSpec],
    dependencies: dict[str, dict[str, float]],
) -> list[str]:
    """Return the sorted parameter names that participate in dependency modeling."""

    names: set[str] = set()
    for left, targets in dependencies.items():
        if left in param_specs:
            names.add(left)
        for right in targets:
            if right in param_specs:
                names.add(right)
    return sorted(names)


def _sample_dependent_uniforms(
    n: int,
    parameter_names: list[str],
    dependencies: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample dependent uniforms from a Gaussian copula."""

    requested_matrix = np.eye(len(parameter_names), dtype=float)
    index_by_name = {name: index for index, name in enumerate(parameter_names)}
    for left, targets in dependencies.items():
        for right, corr in targets.items():
            if left not in index_by_name or right not in index_by_name:
                continue
            i = index_by_name[left]
            j = index_by_name[right]
            gaussian_corr = 2.0 * sin((pi * corr) / 6.0)
            requested_matrix[i, j] = gaussian_corr
            requested_matrix[j, i] = gaussian_corr

    matrix = _nearest_correlation_matrix(requested_matrix)
    max_deviation = float(np.max(np.abs(matrix - requested_matrix)))
    if max_deviation > 0.05:
        warnings.warn(
            "Requested dependency matrix was adjusted to a usable correlation matrix. "
            f"Maximum absolute deviation: {max_deviation:.3f}.",
            RuntimeWarning,
            stacklevel=2,
        )

    normals = rng.multivariate_normal(mean=np.zeros(len(parameter_names)), cov=matrix, size=n)
    standard_normal = NormalDist()
    flattened = normals.reshape(-1)
    uniforms = np.array(
        [standard_normal.cdf(float(value)) for value in flattened],
        dtype=float,
    ).reshape(normals.shape)
    return np.clip(uniforms, 1e-9, 1 - 1e-9)


def _nearest_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the nearest usable correlation matrix."""

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, 1e-8, None)
    rebuilt = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(rebuilt))
    normalized = rebuilt / np.outer(diagonal, diagonal)
    np.fill_diagonal(normalized, 1.0)
    return normalized


def _inverse_cdf_param(spec: ParamSpec, uniform_draws: np.ndarray) -> np.ndarray:
    """Transform uniform draws into samples from one configured marginal."""

    if spec.dist == "constant":
        value = _require_param_value(spec.value, "value", spec.dist)
        return np.full(len(uniform_draws), value, dtype=float)

    if spec.dist == "uniform":
        low = _require_param_value(spec.low, "low", spec.dist)
        high = _require_param_value(spec.high, "high", spec.dist)
        return low + uniform_draws * (high - low)

    if spec.dist == "tri":
        low = _require_param_value(spec.low, "low", spec.dist)
        mode = _require_param_value(spec.mode, "mode", spec.dist)
        high = _require_param_value(spec.high, "high", spec.dist)
        if low == mode == high:
            return np.full(len(uniform_draws), low, dtype=float)
        threshold = (mode - low) / (high - low)
        low_side = low + np.sqrt(uniform_draws * (high - low) * (mode - low))
        high_side = high - np.sqrt((1.0 - uniform_draws) * (high - low) * (high - mode))
        return np.where(uniform_draws <= threshold, low_side, high_side)

    if spec.dist == "lognormal":
        median = _require_param_value(spec.median, "median", spec.dist)
        p95 = _require_param_value(spec.p95, "p95", spec.dist)
        standard_normal = NormalDist()
        sigma = np.log(p95 / median) / 1.6448536269514722
        mu = np.log(median)
        z_scores = np.array(
            [standard_normal.inv_cdf(float(value)) for value in uniform_draws],
            dtype=float,
        )
        return np.exp(mu + sigma * z_scores)

    raise ValueError(f"Unsupported dist: {spec.dist}")


def _require_param_value(value: float | None, field_name: str, dist_name: str) -> float:
    """Return a parameter value once the required field has been confirmed."""

    if value is None:
        raise ValueError(f"Missing '{field_name}' for distribution '{dist_name}'.")
    return value
