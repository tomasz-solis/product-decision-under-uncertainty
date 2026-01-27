import pandas as pd
import pytest
from pathlib import Path

from simulator.mvp_simulator import run_simulation, summarize_results, decision_diagnostics
from simulator.config import load_config


CONFIG_PATH = "simulator/config.yaml"


def test_full_simulation_flow():
    """Test complete simulation from config to results"""
    df = run_simulation(CONFIG_PATH, n_worlds=1000, seed=42, scenario="base")

    assert len(df) == 1000
    assert "do_nothing" in df.columns
    assert "stabilize_core" in df.columns
    assert "feature_extension" in df.columns
    assert "new_capability" in df.columns
    assert "scenario" in df.columns

    # All values should be finite
    assert df["do_nothing"].isna().sum() == 0
    assert df["stabilize_core"].isna().sum() == 0


def test_simulation_with_parameter_overrides():
    """Test simulation with manual parameter overrides"""
    overrides = {
        "baseline_failure_rate": {"low": 0.03, "mode": 0.05, "high": 0.08}
    }

    df = run_simulation(
        CONFIG_PATH,
        n_worlds=500,
        seed=7,
        scenario="base",
        param_overrides=overrides
    )

    assert len(df) == 500

    # Check that baseline_failure_rate is within our override range
    assert df["baseline_failure_rate"].min() >= 0.03
    assert df["baseline_failure_rate"].max() <= 0.08


def test_summarize_results_format():
    """Test that summary results have expected format"""
    df = run_simulation(CONFIG_PATH, n_worlds=500, seed=42)
    summary = summarize_results(df)

    assert isinstance(summary, pd.DataFrame)
    assert "option" in summary.columns
    assert "mean_value_eur" in summary.columns
    assert "p05_value_eur" in summary.columns
    assert "p95_value_eur" in summary.columns


def test_decision_diagnostics_format():
    """Test that decision diagnostics have expected format"""
    df = run_simulation(CONFIG_PATH, n_worlds=500, seed=42)
    diagnostics = decision_diagnostics(df)

    assert isinstance(diagnostics, pd.DataFrame)
    assert "option" in diagnostics.columns
    assert "win_rate" in diagnostics.columns
    assert "mean_regret_eur" in diagnostics.columns


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different simulations"""
    df1 = run_simulation(CONFIG_PATH, n_worlds=200, seed=1)
    df2 = run_simulation(CONFIG_PATH, n_worlds=200, seed=2)

    # Results should differ
    assert not df1["do_nothing"].equals(df2["do_nothing"])
    assert not df1["stabilize_core"].equals(df2["stabilize_core"])


def test_same_seed_produces_reproducible_results():
    """Test that same seed produces identical results"""
    df1 = run_simulation(CONFIG_PATH, n_worlds=200, seed=42)
    df2 = run_simulation(CONFIG_PATH, n_worlds=200, seed=42)

    # Results should be identical
    pd.testing.assert_series_equal(df1["do_nothing"], df2["do_nothing"])
    pd.testing.assert_series_equal(df1["stabilize_core"], df2["stabilize_core"])


def test_config_loads_without_errors():
    """Test that config file loads correctly"""
    cfg = load_config(CONFIG_PATH)

    assert isinstance(cfg, dict)
    assert "params" in cfg
    assert "simulation" in cfg
    assert "scenarios" in cfg


def test_all_options_produce_finite_results():
    """Test that all decision options produce finite values"""
    df = run_simulation(CONFIG_PATH, n_worlds=1000, seed=42)

    for col in ["do_nothing", "stabilize_core", "feature_extension", "new_capability"]:
        assert df[col].notna().all(), f"{col} has NaN values"
        assert (df[col] > -1e12).all(), f"{col} has unreasonably negative values"
        assert (df[col] < 1e12).all(), f"{col} has unreasonably large values"


def test_simulation_with_small_n():
    """Test simulation works with small world count"""
    df = run_simulation(CONFIG_PATH, n_worlds=10, seed=42)
    assert len(df) == 10


def test_parameter_override_validation():
    """Test that invalid overrides are handled"""
    overrides = {
        "nonexistent_param": {"low": 0.0, "mode": 0.5, "high": 1.0}
    }

    with pytest.raises(ValueError, match="unknown param"):
        run_simulation(CONFIG_PATH, n_worlds=100, seed=42, param_overrides=overrides)
