"""Tests that keep manual docs aligned with the current implementation."""

from __future__ import annotations

from pathlib import Path

from simulator.project_paths import PUBLIC_EVIDENCE_PROFILE_MARKDOWN


def test_evidence_workflow_docs_point_to_the_generated_profile_note() -> None:
    """Evidence workflow docs should reference the real generated evidence note."""

    expected_path = str(PUBLIC_EVIDENCE_PROFILE_MARKDOWN)
    for path in [
        Path("README.md"),
        Path("simulator/data_sources.md"),
        Path("simulator/parameter_provenance.md"),
        Path("data/public/README.md"),
    ]:
        content = path.read_text(encoding="utf-8")
        assert expected_path in content
        if path != Path("README.md"):
            assert "artifacts/evidence/README.md" not in content


def test_model_spec_uses_current_frontier_language() -> None:
    """The model spec should describe the current artifact split instead of boundary language."""

    content = Path("simulator/model_spec.md").read_text(encoding="utf-8")

    assert "decision-boundary view" not in content
    assert "full-option policy frontier" in content
    assert "selected-vs-runner-up payoff diagnostic" in content
