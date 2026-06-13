"""Tests for the value-of-information (EVPI / EVPPI) analytics.

These assert mathematical invariants of the estimator rather than specific
numeric outputs, so they hold regardless of the synthetic frame's exact draws:

- EVPI equals the mean regret of the expected-value-optimal option.
- Every EVPPI is bounded in ``[0, EVPI]``.
- A parameter that determines the best option carries near-maximal EVPPI,
  while a parameter independent of the ranking carries almost none.
- A constant parameter carries zero information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simulator.analytics import (
    OPTION_COLUMNS,
    decision_diagnostics,
    value_of_information,
)


def _crossing_frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Return a frame where one parameter decides which option is best.

    ``stabilize_core`` and ``feature_extension`` cross as ``x_driver`` moves, so
    knowing ``x_driver`` tells you which one wins. ``z_noise`` is independent of
    the ranking and ``c_const`` is constant. The two dominated options are held
    far below so they never win in any world.
    """

    rng = np.random.default_rng(seed)
    x_driver = np.linspace(0.0, 1.0, n)
    rng.shuffle(x_driver)
    return pd.DataFrame(
        {
            "x_driver": x_driver,
            "z_noise": rng.random(n),
            "c_const": np.full(n, 3.0),
            "do_nothing": np.full(n, -10_000.0),
            "stabilize_core": 100.0 * x_driver,
            "feature_extension": 100.0 * (1.0 - x_driver),
            "new_capability": np.full(n, -10_000.0),
        }
    )


def _evppi_lookup(result: dict[str, object]) -> dict[str, float]:
    """Return a parameter -> EVPPI map from a value-of-information result."""

    rows = result["evppi_rows"]
    assert isinstance(rows, list)
    return {str(row["parameter"]): float(row["evppi_eur"]) for row in rows}


def test_evpi_equals_mean_regret_of_ev_optimal_option() -> None:
    """EVPI must equal the mean regret of the expected-value-optimal option."""

    frame = _crossing_frame()
    result = value_of_information(frame)
    diagnostics = decision_diagnostics(frame)

    ev_optimal = result["ev_optimal_option"]
    assert ev_optimal in OPTION_COLUMNS
    mean_regret_leader = float(
        diagnostics.loc[diagnostics["option"] == ev_optimal, "mean_regret_eur"].iloc[0]
    )

    assert float(result["evpi_eur"]) == pytest.approx(mean_regret_leader, abs=1e-6)
    assert float(result["evpi_eur"]) > 0.0


def test_every_evppi_is_bounded_between_zero_and_evpi() -> None:
    """No single-parameter information can be negative or exceed full information."""

    result = value_of_information(_crossing_frame())
    evpi = float(result["evpi_eur"])

    for parameter, evppi in _evppi_lookup(result).items():
        assert evppi >= 0.0, f"EVPPI for {parameter} is negative"
        assert evppi <= evpi + 1e-6, f"EVPPI for {parameter} exceeds EVPI"


def test_decision_relevant_parameter_dominates_noise_parameter() -> None:
    """The ranking-determining parameter should carry most of the EVPI; noise almost none."""

    result = value_of_information(_crossing_frame())
    evpi = float(result["evpi_eur"])
    evppi = _evppi_lookup(result)

    assert evppi["x_driver"] > 0.5 * evpi
    assert evppi["z_noise"] < 0.25 * evpi
    assert evppi["x_driver"] > evppi["z_noise"]


def test_constant_parameter_carries_no_information() -> None:
    """A parameter with a single value cannot resolve any uncertainty."""

    result = value_of_information(_crossing_frame())
    assert _evppi_lookup(result)["c_const"] == 0.0


def test_value_of_information_is_deterministic() -> None:
    """The estimator uses no RNG, so repeated calls must match exactly."""

    frame = _crossing_frame()
    first = value_of_information(frame)
    second = value_of_information(frame)

    assert first["evpi_eur"] == second["evpi_eur"]
    assert _evppi_lookup(first) == _evppi_lookup(second)


def test_rows_are_sorted_by_descending_evppi() -> None:
    """Published rows should lead with the most decision-relevant parameter."""

    rows = value_of_information(_crossing_frame())["evppi_rows"]
    assert isinstance(rows, list)
    values = [float(row["evppi_eur"]) for row in rows]
    assert values == sorted(values, reverse=True)


def test_empty_frame_returns_zeroed_result() -> None:
    """An empty result set should degrade gracefully to zeros, not raise."""

    empty = pd.DataFrame({column: [] for column in [*OPTION_COLUMNS, "x_driver"]})
    result = value_of_information(empty)

    assert float(result["evpi_eur"]) == 0.0
    assert result["ev_optimal_option"] is None
    assert result["evppi_rows"] == []
