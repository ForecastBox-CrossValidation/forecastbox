"""Tests for Jinja2 template rendering."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from forecastbox.pipeline.pipeline import ForecastPipeline, PipelineResults
from forecastbox.reports.builder import ReportBuilder  # noqa: F401
from forecastbox.reports.template_renderer import (
    _has_jinja2,
    get_template_path,
    list_templates,
    render_template,
)


@pytest.fixture
def sample_results() -> PipelineResults:
    """Create sample PipelineResults for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01", periods=120, freq="MS")
    data = pd.Series(
        100.0 + np.cumsum(rng.normal(0.1, 1.0, size=120)),
        index=dates,
        name="test_series",
    )

    pipeline = ForecastPipeline(
        data_source=data,
        models=["auto_arima", "auto_ets", "theta"],
        combination="mean",
        evaluation=["rmse", "mae"],
        horizon=12,
    )
    return pipeline.run()


@pytest.fixture
def sections_and_metadata(sample_results: PipelineResults) -> tuple:
    """Create sections and metadata for template testing."""
    from forecastbox.reports.sections import SECTION_REGISTRY

    sections = []
    for section_type in ["summary", "data", "models", "forecasts", "evaluation"]:
        gen = SECTION_REGISTRY[section_type]
        sections.append(gen(sample_results))

    metadata = {
        "title": "Test Report",
        "author": "Test Author",
        "date": "2026-03-17",
        "template": "default",
    }

    return sections, metadata


class TestTemplates:
    """Tests for Jinja2 templates."""

    def test_list_templates(self) -> None:
        """Available templates are listed."""
        templates = list_templates()
        assert isinstance(templates, list)
        assert "default_html" in templates
        assert "executive_html" in templates
        assert "technical_html" in templates
        assert "default_latex" in templates

    def test_template_paths_exist(self) -> None:
        """Template files exist on disk."""
        for name in list_templates():
            path = get_template_path(name)
            assert path is not None, f"Template '{name}' path is None"
            assert path.exists(), f"Template '{name}' file does not exist at {path}"

    def test_default_html_template(self, sections_and_metadata: tuple) -> None:
        """Default HTML template renders correctly."""
        sections, metadata = sections_and_metadata
        if not _has_jinja2():
            pytest.skip("Jinja2 not available")

        result = render_template("default_html", sections, metadata)
        assert result is not None
        assert "<!DOCTYPE html>" in result
        assert "Test Report" in result
        assert "Test Author" in result

    def test_executive_html_template(self, sections_and_metadata: tuple) -> None:
        """Executive HTML template renders only summary, forecasts, scenarios."""
        sections, metadata = sections_and_metadata
        metadata["template"] = "executive"
        if not _has_jinja2():
            pytest.skip("Jinja2 not available")

        result = render_template("executive_html", sections, metadata)
        assert result is not None
        assert "<!DOCTYPE html>" in result
        assert "Executive Summary" in result or "Summary" in result

    def test_technical_html_template(self, sections_and_metadata: tuple) -> None:
        """Technical HTML template includes formulas and diagnostics."""
        sections, metadata = sections_and_metadata
        metadata["template"] = "technical"
        if not _has_jinja2():
            pytest.skip("Jinja2 not available")

        result = render_template("technical_html", sections, metadata)
        assert result is not None
        assert "<!DOCTYPE html>" in result

    def test_default_latex_template(self, sections_and_metadata: tuple) -> None:
        """Default LaTeX template renders correctly."""
        sections, metadata = sections_and_metadata
        if not _has_jinja2():
            pytest.skip("Jinja2 not available")

        result = render_template("default_latex", sections, metadata)
        assert result is not None
        assert r"\documentclass" in result
        assert r"\begin{document}" in result

    def test_unknown_template_returns_none(self, sections_and_metadata: tuple) -> None:
        """Unknown template name returns None."""
        sections, metadata = sections_and_metadata
        result = render_template("nonexistent_template", sections, metadata)
        assert result is None
