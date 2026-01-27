"""
Property-based tests using Hypothesis.

These tests check invariants that should always hold true,
regardless of input data.
"""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import column, data_frames

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import (_bootstrap_ci, _calculate_quality_score, _analyze_data_quality,
                 _test_normality, _calculate_correlation_matrix, _clamp_triplet)


# Property 1: Bootstrap CIs should have lower <= upper
@given(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=30, max_size=100))
@settings(max_examples=50, deadline=None)
def test_bootstrap_ci_ordering(data):
    """Bootstrap CI lower bound should always be <= upper bound."""
    arr = np.array(data)
    assume(len(arr) >= 30)  # Need reasonable sample size

    for percentile in [5, 50, 95]:
        lower, upper = _bootstrap_ci(arr, percentile, n_bootstrap=100)
        assert lower <= upper, f"CI bounds out of order: {lower} > {upper}"


# Property 2: Quality scores should always be 0-100
@given(
    n=st.integers(min_value=5, max_value=1000),
    mean=st.floats(min_value=1, max_value=1000, allow_nan=False),
    cv=st.floats(min_value=0, max_value=5, allow_nan=False),
    outlier_pct=st.floats(min_value=0, max_value=100, allow_nan=False),
    skewness=st.floats(min_value=-5, max_value=5, allow_nan=False)
)
def test_quality_score_bounds(n, mean, cv, outlier_pct, skewness):
    """Quality scores must always be between 0 and 100."""
    quality_metrics = {
        'n': n,
        'mean': mean,
        'cv': cv,
        'outlier_pct': outlier_pct,
        'skewness': skewness,
        'normality': {'is_normal': True}
    }

    score = _calculate_quality_score(quality_metrics)
    assert 0 <= score <= 100, f"Quality score out of bounds: {score}"
    assert isinstance(score, int), f"Quality score should be int, got {type(score)}"


# Property 3: Clamp triplet should always satisfy low <= mode <= high
@given(
    low=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    mode=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    high=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)
)
def test_clamp_triplet_ordering(low, mode, high):
    """Clamp triplet should always return ordered values."""
    clamped_low, clamped_mode, clamped_high = _clamp_triplet(low, mode, high)

    assert clamped_low <= clamped_mode, f"Low > Mode: {clamped_low} > {clamped_mode}"
    assert clamped_mode <= clamped_high, f"Mode > High: {clamped_mode} > {clamped_high}"
    assert clamped_low <= clamped_high, f"Low > High: {clamped_low} > {clamped_high}"


# Property 4: Correlation matrix should be symmetric
@given(data_frames([
    column('col1', elements=st.floats(min_value=-100, max_value=100, allow_nan=False)),
    column('col2', elements=st.floats(min_value=-100, max_value=100, allow_nan=False)),
    column('col3', elements=st.floats(min_value=-100, max_value=100, allow_nan=False)),
], index=st.range_indexes(min_size=10, max_size=50)))
def test_correlation_matrix_symmetric(df):
    """Correlation matrix should be symmetric."""
    corr_matrix, _ = _calculate_correlation_matrix(df)

    if not corr_matrix.empty:
        # Check symmetry
        np.testing.assert_array_almost_equal(
            corr_matrix.values,
            corr_matrix.values.T,
            decimal=10,
            err_msg="Correlation matrix is not symmetric"
        )

        # Diagonal should be 1 (or very close)
        diagonal = np.diag(corr_matrix.values)
        np.testing.assert_array_almost_equal(
            diagonal,
            np.ones(len(diagonal)),
            decimal=10,
            err_msg="Correlation matrix diagonal should be 1"
        )


# Property 5: Percentiles should be ordered
@given(st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False), min_size=20, max_size=100))
def test_percentiles_ordered(data):
    """p5 <= p50 <= p95 always."""
    arr = np.array(data)
    p05 = np.percentile(arr, 5)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)

    assert p05 <= p50, f"p5 > p50: {p05} > {p50}"
    assert p50 <= p95, f"p50 > p95: {p50} > {p95}"
    assert p05 <= p95, f"p5 > p95: {p05} > {p95}"
