# simulator/mvp_simulator.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from simulator.config import (
    load_config,
    parse_param_specs,
    get_seed,
    get_simulation_settings,
    apply_scenario,
)


def _sample_param(spec, rng: np.random.Generator) -> float:
    if spec.dist == "tri":
        return float(rng.triangular(spec.low, spec.mode, spec.high))
    if spec.dist == "uniform":
        return float(rng.uniform(spec.low, spec.high))
    raise ValueError(f"Unknown dist: {spec.dist}")


def sample_params(n: int, param_specs, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {k: np.array([_sample_param(v, rng) for _ in range(n)], dtype=float) for k, v in param_specs.items()}
    return pd.DataFrame(data)


def _expected_regression_cost(p: pd.DataFrame) -> np.ndarray:
    return p["regression_prob"].to_numpy() * p["regression_cost_eur"].to_numpy()


def simulate_option_do_nothing(p: pd.DataFrame, volume: int) -> np.ndarray:
    """
    Status quo option: no deliberate change.
    We still pay the ongoing cost of failures and churn, plus a drift cost proxy.
    """
    failure = p["baseline_failure_rate"].to_numpy()

    successes = volume * (1.0 - failure)
    revenue = successes * p["rev_per_success_eur"].to_numpy()

    failure_cost = (volume * failure) * p["cost_per_failure_eur"].to_numpy()

    base_churn = p["base_churn"].to_numpy()
    churn = base_churn * (1.0 + p["failure_to_churn_rel"].to_numpy())
    churn_penalty = churn * revenue

    drift = p["do_nothing_drift_cost_eur"].to_numpy()

    # No regression cost because we are not shipping changes (by definition here).
    return revenue - failure_cost - churn_penalty - drift


def simulate_option_stabilize_core(p: pd.DataFrame, volume: int) -> np.ndarray:
    failure = p["baseline_failure_rate"].to_numpy()
    reduction = p["stabilize_failure_reduction"].to_numpy()
    new_failure = failure * (1.0 - reduction)

    successes = volume * (1.0 - new_failure)
    revenue = successes * p["rev_per_success_eur"].to_numpy()

    failure_cost = (volume * new_failure) * p["cost_per_failure_eur"].to_numpy()
    regression_cost = _expected_regression_cost(p)

    base_churn = p["base_churn"].to_numpy()
    churn = base_churn * (1.0 + (new_failure / np.maximum(failure, 1e-9)) * p["failure_to_churn_rel"].to_numpy())
    churn_penalty = churn * revenue  # MVP proxy

    return revenue - failure_cost - regression_cost - churn_penalty


def simulate_option_feature_extension(p: pd.DataFrame, volume: int) -> np.ndarray:
    failure = p["baseline_failure_rate"].to_numpy()
    exposure_reduction = p["extension_exposure_reduction"].to_numpy()
    effective_failure = failure * (1.0 - exposure_reduction)

    successes = volume * (1.0 - effective_failure)
    revenue = successes * p["rev_per_success_eur"].to_numpy()

    uptake = p["extension_uptake"].to_numpy()
    rev_per_uptake = p["extension_rev_per_uptake_eur"].to_numpy()
    extension_revenue = (volume * uptake) * rev_per_uptake

    loss_rate = p["extension_loss_rate"].to_numpy()
    extension_losses = extension_revenue * loss_rate

    failure_cost = (volume * effective_failure) * p["cost_per_failure_eur"].to_numpy()
    regression_cost = _expected_regression_cost(p)

    base_churn = p["base_churn"].to_numpy()
    churn = base_churn * (1.0 + (effective_failure / np.maximum(failure, 1e-9)) * p["failure_to_churn_rel"].to_numpy())
    churn_penalty = churn * (revenue + extension_revenue)

    return (revenue + extension_revenue) - failure_cost - extension_losses - regression_cost - churn_penalty


def simulate_option_new_capability(p: pd.DataFrame, volume: int) -> np.ndarray:
    failure = p["baseline_failure_rate"].to_numpy()
    successes = volume * (1.0 - failure)
    base_revenue = successes * p["rev_per_success_eur"].to_numpy()

    uplift = p["new_capability_uplift"].to_numpy()
    revenue = base_revenue * (1.0 + uplift)

    failure_cost = (volume * failure) * p["cost_per_failure_eur"].to_numpy()

    mult = p["new_capability_regression_multiplier"].to_numpy()
    regression_cost = _expected_regression_cost(p) * mult

    base_churn = p["base_churn"].to_numpy()
    churn = base_churn * (1.0 + p["failure_to_churn_rel"].to_numpy())
    churn_penalty = churn * revenue

    return revenue - failure_cost - regression_cost - churn_penalty


def run_simulation(config_path: str | Path) -> pd.DataFrame:
    cfg = load_config(config_path)
    sim = get_simulation_settings(cfg)

    # Apply scenario overrides before parsing params
    cfg = apply_scenario(cfg, sim["scenario"])

    seed = get_seed(cfg)
    param_specs = parse_param_specs(cfg)

    p = sample_params(sim["n_worlds"], param_specs, seed=seed)

    out = pd.DataFrame({
        "do_nothing": simulate_option_do_nothing(p, sim["volume"]),
        "stabilize_core": simulate_option_stabilize_core(p, sim["volume"]),
        "feature_extension": simulate_option_feature_extension(p, sim["volume"]),
        "new_capability": simulate_option_new_capability(p, sim["volume"]),
    })

    # Keep the scenario name for traceability
    out["scenario"] = sim["scenario"]

    return pd.concat([p, out], axis=1)


def run_all_scenarios(config_path: str | Path) -> pd.DataFrame:
    """
    Run the simulator for every scenario defined in config.yaml and return a long table:
    scenario x option with win_rate, mean, and p05.

    This stays intentionally simple and readable.
    """
    base_cfg = load_config(config_path)

    scenarios = base_cfg.get("scenarios", {})
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError("No scenarios found in config.yaml under 'scenarios'.")

    base_sim = get_simulation_settings(base_cfg)

    rows = []
    for scenario_name in scenarios.keys():
        # Use the same simulation settings, just swap the scenario
        cfg = load_config(config_path)
        cfg.setdefault("simulation", {})
        cfg["simulation"]["scenario"] = scenario_name

        sim = get_simulation_settings(cfg)
        cfg = apply_scenario(cfg, sim["scenario"])

        seed = get_seed(cfg)
        param_specs = parse_param_specs(cfg)
        p = sample_params(sim["n_worlds"], param_specs, seed=seed)

        out = pd.DataFrame({
            "do_nothing": simulate_option_do_nothing(p, sim["volume"]),
            "stabilize_core": simulate_option_stabilize_core(p, sim["volume"]),
            "feature_extension": simulate_option_feature_extension(p, sim["volume"]),
            "new_capability": simulate_option_new_capability(p, sim["volume"]),
        })

        best = out.max(axis=1)
        wins = out.eq(best, axis=0).mean()

        for opt in out.columns:
            rows.append({
                "scenario": scenario_name,
                "option": opt,
                "win_rate": float(wins[opt]),
                "mean_value_eur": float(out[opt].mean()),
                "p05_value_eur": float(out[opt].quantile(0.05)),
            })


    # stable ordering
    return (
        pd.DataFrame(rows)
        .sort_values(["scenario", "win_rate"], ascending=[True, False])
        .reset_index(drop=True)
    )


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    options = ["do_nothing", "stabilize_core", "feature_extension", "new_capability"]
    rows = []
    for opt in options:
        vals = df[opt].to_numpy()
        rows.append({
            "option": opt,
            "mean_value_eur": float(np.mean(vals)),
            "median_value_eur": float(np.median(vals)),
            "p05_value_eur": float(np.quantile(vals, 0.05)),
            "p95_value_eur": float(np.quantile(vals, 0.95)),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("mean_value_eur", ascending=False)
        .reset_index(drop=True)
    )


def decision_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    options = ["do_nothing", "stabilize_core", "feature_extension", "new_capability"]

    values = df[options].to_numpy()
    best = values.max(axis=1)

    rows = []
    for i, opt in enumerate(options):
        v = values[:, i]
        regret = best - v
        win_rate = float(np.mean(v == best))  # ties count as wins
        rows.append({
            "option": opt,
            "win_rate": win_rate,
            "mean_regret_eur": float(np.mean(regret)),
            "median_regret_eur": float(np.median(regret)),
            "p95_regret_eur": float(np.quantile(regret, 0.95)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("win_rate", ascending=False)
        .reset_index(drop=True)
    )


def sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman rank correlation between parameters and the advantage of
    Feature Extension over Stabilize Core.
    """
    options = ["do_nothing", "stabilize_core", "feature_extension", "new_capability", "scenario"]
    params = [c for c in df.columns if c not in options]

    delta = df["feature_extension"] - df["stabilize_core"]

    rows = []
    for p in params:
        corr = pd.Series(df[p]).corr(delta, method="spearman")
        rows.append({"parameter": p, "spearman_corr": float(corr)})

    return (
        pd.DataFrame(rows)
        .sort_values("spearman_corr", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    print("Note: these are simulated comparisons under uncertainty. They are not forecasts.\n")

    df = run_simulation("simulator/config.yaml")

    print("Summary (per option across plausible futures)")
    summary = summarize_results(df)
    print(summary.to_string(index=False))

    print("\nDecision diagnostics (comparative)")
    diag = decision_diagnostics(df)
    print(diag.to_string(index=False))

    print("\nSensitivity (feature_extension vs stabilize_core)")
    sens = sensitivity_analysis(df)
    print(sens.head(10).to_string(index=False))

    print("\nScenario comparison (per scenario, per option)")
    sc = run_all_scenarios("simulator/config.yaml")
    print(sc.to_string(index=False))
