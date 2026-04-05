"""Tests for truthful app wording around published results and reruns."""

from __future__ import annotations

from pathlib import Path

from app import PublishedGovernance, governance_warning_message, published_case_caption
from simulator.output_utils import build_run_context


def test_build_run_context_marks_exploratory_reruns() -> None:
    """Non-published settings should be labeled as exploratory reruns."""

    context = build_run_context(
        selected_settings={
            "scenario": "reliability_crisis",
            "scenario_label": "Reliability crisis",
            "seed": 7,
            "n_worlds": 5000,
        },
        published_settings={
            "scenario": "mid_range_pressure",
            "scenario_label": "Mid-range pressure",
            "seed": 42,
            "n_worlds": 20000,
        },
    )

    assert context["heading"] == "Run summary: Reliability crisis"
    assert context["matches_published"] is False
    assert "Exploratory rerun" in str(context["note"])
    assert "Mid-range pressure" in str(context["note"])


def test_app_source_uses_selected_run_language() -> None:
    """The app should separate the published case from interactive reruns."""

    source = Path("app.py").read_text(encoding="utf-8")

    assert "Base run summary" not in source
    assert "build_run_context" in source
    assert "published_case_caption()" in source
    assert "default scenario" in published_case_caption()
    assert "Guardrail eligibility" in source
    assert "Published-case stability" in source
    assert "Provenance and evidence" in source
    assert "Freshness status" in source
    assert "full-option switch frontier" in source
    assert "st.warning" in source
    assert 'st.title("Product Decision Under Uncertainty")' in source
    assert "use_container_width" not in source


def test_governance_warning_message_flags_stale_artifacts() -> None:
    """The app should surface a warning message when published artifacts are stale."""

    governance = PublishedGovernance(
        metadata={},
        stability_summary={},
        evidence_summary={},
        manifest_counts={},
        freshness_status="stale",
        freshness_message="Published governance artifacts do not match the current code.",
        stale_fields=("code_sha256",),
        evidence_note_path="artifacts/evidence/public_data_profile.md",
        frontier_semantics="full-option switch frontier",
    )

    assert "do not match" in str(governance_warning_message(governance))
