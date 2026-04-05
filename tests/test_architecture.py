"""Architecture and import-boundary tests."""

from __future__ import annotations

import importlib
import sys


def test_analytics_and_presentation_import_without_streamlit() -> None:
    """Utility modules should stay import-safe outside the app entrypoint."""

    sys.modules.pop("streamlit", None)
    analytics = importlib.import_module("simulator.analytics")
    presentation = importlib.import_module("simulator.presentation")
    output_utils = importlib.import_module("simulator.output_utils")

    assert hasattr(analytics, "summarize_results")
    assert hasattr(presentation, "summary_display_table")
    assert hasattr(output_utils, "build_run_context")
    assert "streamlit" not in sys.modules


def test_app_exposes_a_single_render_entrypoint() -> None:
    """The app module should export a callable render function."""

    app = importlib.import_module("app")
    assert callable(app.render_app)
    assert callable(app.published_case_caption)
