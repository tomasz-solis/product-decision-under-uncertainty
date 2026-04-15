"""Table-formatting helpers for the app and generated outputs."""

from __future__ import annotations

import pandas as pd

from simulator.output_utils import (
    clean_label,
    driver_note,
    format_eur,
    format_pct,
    format_threshold_eur,
    labeled_option,
    material_driver_rows,
    material_sensitivity_rows,
    sensitivity_note,
)
from simulator.policy import (
    FAILURE_REASON_LABELS,
    FULL_FRONTIER_SWITCH_LABELS,
    RUNNER_UP_FRONTIER_STATUS_LABELS,
)


def summary_display_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted summary table for display."""

    table = summary.copy()
    table["Option"] = table["option"].map(labeled_option)
    table["Expected Value"] = table["mean_value_eur"].map(format_eur)
    table["P05"] = table["p05_value_eur"].map(format_eur)
    table["Median"] = table["median_value_eur"].map(format_eur)
    table["P95"] = table["p95_value_eur"].map(format_eur)
    return table[["Option", "Expected Value", "P05", "Median", "P95"]]


def diagnostics_display_table(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted diagnostics table for display."""

    table = diagnostics.copy()
    table["Option"] = table["option"].map(labeled_option)
    table["Win Rate"] = table["win_rate"].map(format_pct)
    table["Mean Regret"] = table["mean_regret_eur"].map(format_eur)
    table["P95 Regret"] = table["p95_regret_eur"].map(format_eur)
    return table[["Option", "Win Rate", "Mean Regret", "P95 Regret"]]


def eligibility_display_table(eligibility: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted guardrail-eligibility table for display."""

    table = eligibility.copy()
    table["Option"] = table["option"].map(labeled_option)
    table["Expected Value"] = table["mean_value_eur"].map(format_eur)
    table["P05"] = table["p05_value_eur"].map(format_eur)
    table["Downside Slack"] = table["downside_slack_eur"].map(format_eur)
    table["Mean Regret"] = table["mean_regret_eur"].map(format_eur)
    table["Regret Slack"] = table["regret_slack_eur"].map(format_eur)
    table["Eligible"] = table["eligible"].map(lambda value: "yes" if bool(value) else "no")
    table["Failure Reason"] = table["failure_reason"].map(
        lambda value: (
            "passes both guardrails"
            if pd.isna(value)
            else FAILURE_REASON_LABELS.get(str(value), str(value))
        )
    )
    return table[
        [
            "Option",
            "Expected Value",
            "P05",
            "Downside Slack",
            "Mean Regret",
            "Regret Slack",
            "Eligible",
            "Failure Reason",
        ]
    ]


def scenario_display_table(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted scenario table for display."""

    table = scenarios.copy()
    table["Selected Option"] = table["selected_option"].map(labeled_option)
    table["Option"] = table["option"].map(labeled_option)
    table["Expected Value"] = table["mean_value_eur"].map(format_eur)
    table["P05"] = table["p05_value_eur"].map(format_eur)
    table["Mean Regret"] = table["mean_regret_eur"].map(format_eur)
    table["Eligible"] = table["eligible"].map(lambda value: "yes" if bool(value) else "no")
    table["Win Rate"] = table["win_rate"].map(format_pct)
    table = table.rename(columns={"scenario": "Scenario"})
    return table[
        [
            "Scenario",
            "Selected Option",
            "Option",
            "Expected Value",
            "P05",
            "Mean Regret",
            "Eligible",
            "Win Rate",
        ]
    ]


def payoff_delta_display_table(delta_rows: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted payoff-delta diagnostic table for display."""

    if delta_rows.empty:
        return pd.DataFrame(columns=["Parameter", "Delta rho", "Sampled range", "Interpretation"])
    table = delta_rows.copy()
    table["Parameter"] = table["parameter"].map(lambda value: clean_label(str(value)))
    table["Delta rho"] = table["delta_spearman_corr"].map(lambda value: f"{value:+.2f}")
    table["Sampled range"] = table.apply(
        lambda row: f"{row['sampled_min_value']:,.3f} to {row['sampled_max_value']:,.3f}",
        axis=1,
    )
    table["Interpretation"] = table["interpretation_note"]
    return table[["Parameter", "Delta rho", "Sampled range", "Interpretation"]]


def frontier_display_table(frontier_rows: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted full-option policy-frontier table for display."""

    if frontier_rows.empty:
        return pd.DataFrame(
            columns=[
                "Threshold",
                "Current value",
                "Switching value",
                "Switching option",
                "Switch type",
            ]
        )
    table = frontier_rows.copy()
    table["Threshold"] = table["threshold_label"]
    table["Current value"] = table["current_value"].map(format_threshold_eur)
    table["Switching value"] = table["switching_value"].map(
        lambda value: "not observed" if pd.isna(value) else format_threshold_eur(float(value))
    )
    table["Switching option"] = table["first_switching_option"].map(
        lambda value: "not observed" if pd.isna(value) else labeled_option(str(value))
    )
    table["Switch type"] = table["switch_type"].map(
        lambda value: FULL_FRONTIER_SWITCH_LABELS.get(str(value), str(value))
    )
    table["Direction"] = table["switch_direction"].map(
        lambda value: "not observed" if pd.isna(value) else str(value).replace("_", " ")
    )
    table["Interpretation"] = table["interpretation_note"]
    return table[
        [
            "Threshold",
            "Current value",
            "Switching value",
            "Switching option",
            "Switch type",
            "Direction",
            "Interpretation",
        ]
    ]


def comparison_frontier_display_table(frontier_rows: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted secondary comparison table for display."""

    if frontier_rows.empty:
        return pd.DataFrame(columns=["Threshold", "Current value", "Comparison threshold", "Status"])
    table = frontier_rows.copy()
    table["Threshold"] = table["threshold_label"]
    table["Current value"] = table["current_value"].map(format_threshold_eur)
    table["Comparison threshold"] = table["switching_value"].map(
        lambda value: "not needed" if pd.isna(value) else format_threshold_eur(float(value))
    )
    table["Status"] = table["status"].map(
        lambda value: RUNNER_UP_FRONTIER_STATUS_LABELS.get(str(value), str(value))
    )
    table["Interpretation"] = table["interpretation_note"]
    return table[
        [
            "Threshold",
            "Current value",
            "Comparison threshold",
            "Status",
            "Interpretation",
        ]
    ]


def runner_up_frontier_display_table(frontier_rows: pd.DataFrame) -> pd.DataFrame:
    """Return the legacy alias for the comparison frontier display table."""

    return comparison_frontier_display_table(frontier_rows)


def stability_frequency_table(rows: list[dict[str, object]], key: str) -> pd.DataFrame:
    """Return a formatted frequency table for the published stability summary."""

    if not rows:
        return pd.DataFrame(columns=["Option", "Runs"])
    table = pd.DataFrame(rows).copy()
    table["Option"] = table[key].map(lambda value: labeled_option(str(value)))
    table["Runs"] = table["count"].astype(int)
    return table[["Option", "Runs"]]


def top_sensitivity_rows(
    sensitivity: pd.DataFrame,
    option: str,
    threshold: float,
    limit: int,
) -> pd.DataFrame:
    """Return the strongest material sensitivity rows for one option."""

    if "partial_rank_corr" in sensitivity.columns:
        table = material_driver_rows(
            driver_analysis=sensitivity,
            option=option,
            threshold=threshold,
            limit=limit,
        ).copy()
        if table.empty:
            return pd.DataFrame(columns=["Parameter", "Partial rank corr", "95% CI"])
        table["Parameter"] = table["parameter"].map(lambda value: clean_label(str(value)))
        table["Partial rank corr"] = table["partial_rank_corr"].map(lambda value: f"{value:+.2f}")
        table["95% CI"] = table.apply(
            lambda row: f"{float(row['ci_low']):+.2f} to {float(row['ci_high']):+.2f}",
            axis=1,
        )
        return table[["Parameter", "Partial rank corr", "95% CI"]]

    table = material_sensitivity_rows(
        sensitivity=sensitivity,
        option=option,
        threshold=threshold,
        limit=limit,
    ).copy()
    if table.empty:
        return pd.DataFrame(columns=["Parameter", "Spearman"])
    table["Parameter"] = table["parameter"].map(lambda value: clean_label(str(value)))
    table["Spearman"] = table["spearman_corr"].map(lambda value: f"{value:+.2f}")
    return table[["Parameter", "Spearman"]]


def sensitivity_summary_note(
    sensitivity: pd.DataFrame, option: str, threshold: float
) -> str | None:
    """Return a short note about material sensitivity coverage."""

    if "partial_rank_corr" in sensitivity.columns:
        return driver_note(sensitivity, option, threshold)
    return sensitivity_note(sensitivity, option, threshold)


def driver_analysis_interpretation_note(
    driver_df: pd.DataFrame,
    selected_option: str,
    top_n: int = 3,
) -> str:
    """Return a plain-English sentence summarizing the strongest driver findings."""

    option_rows = (
        driver_df.loc[driver_df["option"] == selected_option]
        .sort_values("partial_rank_corr", key=abs, ascending=False)
        .head(top_n)
    )
    if option_rows.empty:
        return "No material drivers identified for this option."

    parts: list[str] = []
    for _, row in option_rows.iterrows():
        name = clean_label(str(row["parameter"]))
        direction = "higher tends to help" if float(row["partial_rank_corr"]) > 0 else "lower tends to help"
        parts.append(f"{name} ({direction})")

    if len(parts) == 1:
        top_names = parts[0]
    else:
        top_names = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    return (
        "Holding the other modeled parameters fixed, the selected option is most sensitive to "
        f"{top_names}. These are the assumptions where updated evidence is most likely to move "
        "the recommendation."
    )
