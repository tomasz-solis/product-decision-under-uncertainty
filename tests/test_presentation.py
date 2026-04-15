"""Tests for presentation helpers."""

from __future__ import annotations

import pandas as pd

from simulator.presentation import driver_analysis_interpretation_note


def test_driver_analysis_interpretation_note_reads_like_plain_english() -> None:
    """Driver narration should turn ranked signals into business-readable text."""

    frame = pd.DataFrame(
        [
            {
                "option": "stabilize_core",
                "parameter": "baseline_failure_rate",
                "partial_rank_corr": 0.86,
            },
            {
                "option": "stabilize_core",
                "parameter": "stabilize_core_launch_delay_months",
                "partial_rank_corr": -0.62,
            },
        ]
    )

    result = driver_analysis_interpretation_note(frame, "stabilize_core", top_n=2)

    assert "Baseline Failure Rate (higher tends to help)" in result
    assert "Stabilize Core Launch Delay Months (lower tends to help)" in result
    assert "updated evidence" in result


def test_driver_analysis_interpretation_note_handles_empty_option() -> None:
    """The narration helper should stay calm when no rows exist for an option."""

    frame = pd.DataFrame(columns=["option", "parameter", "partial_rank_corr"])

    result = driver_analysis_interpretation_note(frame, "stabilize_core")

    assert result == "No material drivers identified for this option."
