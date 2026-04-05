"""Smoke tests for secondary entrypoints."""

from __future__ import annotations

import subprocess
import sys

from simulator.mvp_simulator import main
from simulator.reporting import build_case_study_artifacts


def test_mvp_cli_uses_guardrail_and_frontier_language(capsys) -> None:
    """The CLI should mirror the current public reporting language."""

    artifacts = build_case_study_artifacts("simulator/config.yaml")

    main()

    captured = capsys.readouterr().out
    assert "Decision boundary" not in captured
    assert "Guardrail eligibility" in captured
    assert "Policy frontier" in captured
    assert "Published-case stability" in captured
    assert artifacts.recommendation.selected_option in captured


def test_streamlit_app_smoke_runs_without_exceptions() -> None:
    """The Streamlit app should boot in a headless test run."""

    code = (
        "from streamlit.testing.v1 import AppTest\n"
        "app = AppTest.from_file('app.py')\n"
        "app.run(timeout=30)\n"
        "assert [title.value for title in app.title] == ['Product Decision Under Uncertainty']\n"
        "assert 'Guardrail eligibility' in [item.value for item in app.subheader]\n"
        "assert 'Published-case stability' in [item.value for item in app.subheader]\n"
        "assert len(app.exception) == 0\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
