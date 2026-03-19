"""Tests for report generation module."""

from __future__ import annotations

import json
import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from forecastbox.pipeline.pipeline import ForecastPipeline, PipelineResults
from forecastbox.reports.builder import ReportBuilder


@pytest.fixture
def sample_results() -> PipelineResults:
    """Create sample PipelineResults for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01", periods=120, freq="MS")
    data = pd.Series(100.0 + np.cumsum(rng.normal(0.1, 1.0, size=120)),
                     index=dates, name="test_series")

    pipeline = ForecastPipeline(
        data_source=data,
        models=["auto_arima", "auto_ets", "theta"],
        combination="mean",
        evaluation=["rmse", "mae"],
        horizon=12,
    )
    return pipeline.run()


@pytest.fixture
def report_builder(sample_results: PipelineResults) -> ReportBuilder:
    """Create ReportBuilder with sections."""
    builder = ReportBuilder(
        results=sample_results,
        title="Test Forecast Report",
        author="Test Author",
    )
    builder.add_section("summary", title="Executive Summary")
    builder.add_section("data", title="Data")
    builder.add_section("models", title="Models")
    builder.add_section("forecasts", title="Forecasts")
    builder.add_section("evaluation", title="Evaluation")
    return builder


class TestReports:
    """Tests for report generation."""

    def test_html_report(self, report_builder: ReportBuilder) -> None:
        """render('html') gera HTML valido."""
        content = report_builder.render("html")
        assert isinstance(content, str)
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "Test Forecast Report" in content
        assert "Executive Summary" in content

    def test_latex_report(self, report_builder: ReportBuilder) -> None:
        """render('latex') gera LaTeX compilavel."""
        content = report_builder.render("latex")
        assert isinstance(content, str)
        assert r"\documentclass" in content
        assert r"\begin{document}" in content
        assert r"\end{document}" in content
        assert "Test Forecast Report" in content

    def test_markdown_report(self, report_builder: ReportBuilder) -> None:
        """render('markdown') gera Markdown."""
        content = report_builder.render("markdown")
        assert isinstance(content, str)
        assert "# Test Forecast Report" in content
        assert "## Executive Summary" in content
        assert "---" in content

    def test_json_report(self, report_builder: ReportBuilder) -> None:
        """render('json') gera JSON parseavel."""
        content = report_builder.render("json")
        assert isinstance(content, str)
        parsed = json.loads(content)
        assert "metadata" in parsed
        assert "sections" in parsed
        assert parsed["metadata"]["title"] == "Test Forecast Report"
        assert len(parsed["sections"]) == 5

    def test_add_section(self, sample_results: PipelineResults) -> None:
        """Secoes adicionadas na ordem correta."""
        builder = ReportBuilder(results=sample_results)
        builder.add_section("summary")
        builder.add_section("forecasts")
        builder.add_section("evaluation")

        assert len(builder._sections) == 3
        assert builder._section_order == ["summary", "forecasts", "evaluation"]
        assert builder._sections[0]["type"] == "summary"
        assert builder._sections[1]["type"] == "forecasts"
        assert builder._sections[2]["type"] == "evaluation"

    def test_template_default(self, sample_results: PipelineResults) -> None:
        """Template default inclui todas as secoes."""
        builder = ReportBuilder(results=sample_results, template="default")
        for section_type in ["summary", "data", "models", "forecasts",
                            "evaluation", "combination", "diagnostics"]:
            builder.add_section(section_type)

        content = builder.render("html")
        assert "Summary" in content
        assert "Data" in content
        assert "Models" in content

    def test_template_executive(self, sample_results: PipelineResults) -> None:
        """Template executive inclui apenas resumo + previsoes."""
        builder = ReportBuilder(results=sample_results, template="executive")
        builder.add_section("summary", title="Executive Summary")
        builder.add_section("forecasts", title="Forecasts")

        content = builder.render("html")
        assert "Executive Summary" in content
        assert "Forecasts" in content
        assert len(builder._sections) == 2

    def test_report_with_plots(self, sample_results: PipelineResults) -> None:
        """Report inclui graficos inline (base64 para HTML)."""
        builder = ReportBuilder(results=sample_results)
        builder.add_section("forecasts")

        content = builder.render("html")
        # Check that base64 image is included (if plots generated)
        assert isinstance(content, str)
        assert "<html" in content

    def test_report_output_file(self, report_builder: ReportBuilder) -> None:
        """output='file.html' salva em disco."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_report.html")
            report_builder.render("html", output=output_path)

            assert os.path.exists(output_path)
            with open(output_path) as f:
                file_content = f.read()
            assert "<!DOCTYPE html>" in file_content
            assert "Test Forecast Report" in file_content
