"""Empirical checks for the configured dependency model."""

from __future__ import annotations

from pathlib import Path

import pytest

from simulator.analytics import spearman_rank_correlation
from simulator.config import get_dependency_settings, load_config, parse_param_specs
from simulator.simulation import sample_params

CONFIG_PATH = Path("simulator/config.yaml")


def test_dependency_sampling_hits_configured_rank_correlations() -> None:
    """The achieved rank correlations should stay near the configured targets."""

    cfg = load_config(CONFIG_PATH)
    specs = parse_param_specs(cfg)
    dependencies = get_dependency_settings(cfg)
    sampled = sample_params(12000, specs, seed=42, dependencies=dependencies)

    observed = {
        ("baseline_failure_rate", "cost_per_failure_eur"): spearman_rank_correlation(
            sampled["baseline_failure_rate"],
            sampled["cost_per_failure_eur"],
        ),
        ("baseline_failure_rate", "failure_to_churn_rel"): spearman_rank_correlation(
            sampled["baseline_failure_rate"],
            sampled["failure_to_churn_rel"],
        ),
        ("extension_uptake", "extension_value_per_uptake_eur"): spearman_rank_correlation(
            sampled["extension_uptake"],
            sampled["extension_value_per_uptake_eur"],
        ),
        ("regression_event_prob", "regression_event_cost_eur"): spearman_rank_correlation(
            sampled["regression_event_prob"],
            sampled["regression_event_cost_eur"],
        ),
    }

    targets = {
        ("baseline_failure_rate", "cost_per_failure_eur"): 0.55,
        ("baseline_failure_rate", "failure_to_churn_rel"): 0.60,
        ("extension_uptake", "extension_value_per_uptake_eur"): 0.40,
        ("regression_event_prob", "regression_event_cost_eur"): 0.45,
    }
    for pair, target in targets.items():
        assert observed[pair] == pytest.approx(target, abs=0.08)
