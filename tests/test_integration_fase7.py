"""End-to-end integration test for FASE 7: Pipeline -> Viz -> Reports."""

from __future__ import annotations

import json
import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from forecastbox.core.forecast import Forecast  # noqa: E402, F401
from forecastbox.pipeline.alerts import Alert, AlertRule, AlertSystem  # noqa: E402, F401
from forecastbox.pipeline.monitor import ForecastMonitor, MonitorReport  # noqa: E402
from forecastbox.pipeline.pipeline import ForecastPipeline, PipelineResults  # noqa: E402
from forecastbox.pipeline.recurring import RecurringForecast  # noqa: E402
from forecastbox.reports.builder import ReportBuilder  # noqa: E402
from forecastbox.viz._style import get_color_palette, set_nodesecon_style  # noqa: E402, F401
from forecastbox.viz.plotter import ForecastPlotter  # noqa: E402


@pytest.fixture
def sample_data() -> pd.Series:
    """Generate realistic time series for integration testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01", periods=180, freq="MS")
    trend = np.linspace(100, 130, 180)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(180) / 12)
    noise = rng.normal(0, 2.0, size=180)
    values = trend + seasonal + noise
    return pd.Series(values, index=dates, name="gdp_index")


class TestIntegrationEndToEnd:
    """Full integration test: Pipeline -> Viz -> Report."""

    def test_full_pipeline_to_report(self, sample_data: pd.Series) -> None:
        """Complete end-to-end: pipeline.run() -> plotter -> report."""

        # Step 1: Create and run pipeline
        pipeline = ForecastPipeline(
            data_source=sample_data,
            models=["auto_arima", "auto_ets", "theta"],
            combination="bma",
            evaluation=["rmse", "mae"],
            horizon=12,
            cv_type="expanding",
        )
        results = pipeline.run()

        # Verify pipeline results
        assert isinstance(results, PipelineResults)
        assert len(results.forecasts) == 3
        assert results.combination is not None
        assert not results.evaluation.empty
        assert len(results.execution_time) > 0

        # Step 2: Summary
        summary = results.summary()
        assert isinstance(summary, str)
        assert "PIPELINE RESULTS SUMMARY" in summary

        # Step 3: Best model
        best = results.best_model()
        assert isinstance(best, str)
        assert len(best) > 0

        # Step 4: Serialize
        result_dict = results.to_dict()
        assert isinstance(result_dict, dict)
        assert "forecasts" in result_dict

        # Step 5: Visualization
        plotter = ForecastPlotter(results, style="nodesecon")

        ax = plotter.forecast_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

        ax = plotter.fan_chart()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

        ax = plotter.comparison_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

        ax = plotter.accuracy_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

        ax = plotter.combination_weights_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

        ax = plotter.cv_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

        fig = plotter.residual_plot()
        assert isinstance(fig, plt.Figure)
        plt.close("all")

        fig = plotter.pipeline_dashboard()
        assert isinstance(fig, plt.Figure)
        plt.close("all")

        # Step 6: Reports in all formats
        builder = ReportBuilder(
            results=results,
            title="Integration Test Report",
            author="Test Suite",
        )
        builder.add_section("summary", title="Summary")
        builder.add_section("data", title="Data Description")
        builder.add_section("models", title="Model Specifications")
        builder.add_section("forecasts", title="Forecasts")
        builder.add_section("evaluation", title="Evaluation")
        builder.add_section("combination", title="Combination")
        builder.add_section("diagnostics", title="Diagnostics")

        # HTML
        html = builder.render("html")
        assert "<!DOCTYPE html>" in html
        assert "Integration Test Report" in html

        # LaTeX
        latex = builder.render("latex")
        assert r"\documentclass" in latex
        assert r"\begin{document}" in latex

        # Markdown
        md = builder.render("markdown")
        assert "# Integration Test Report" in md
        assert "##" in md

        # JSON
        json_str = builder.render("json")
        parsed = json.loads(json_str)
        assert "metadata" in parsed
        assert "sections" in parsed

        # Save HTML to file
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "report.html")
            builder.render("html", output=html_path)
            assert os.path.exists(html_path)

            md_path = os.path.join(tmpdir, "report.md")
            builder.render("markdown", output=md_path)
            assert os.path.exists(md_path)

    def test_recurring_and_monitor_flow(self, sample_data: pd.Series) -> None:
        """RecurringForecast -> Monitor -> Alerts integration."""

        # Step 1: Create recurring forecast
        pipeline = ForecastPipeline(
            data_source=sample_data,
            models=["auto_arima", "auto_ets"],
            combination="mean",
            horizon=6,
        )
        recurring = RecurringForecast(pipeline=pipeline)

        # Run 3 times
        for _ in range(3):
            recurring.run_once()

        assert len(recurring.history()) == 3

        # Forecast evolution
        evolution = recurring.forecast_evolution()
        assert isinstance(evolution, pd.DataFrame)
        assert len(evolution) == 3

        # Step 2: Monitor
        monitor = ForecastMonitor(pipeline)

        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01", periods=24, freq="MS")
        for i, date in enumerate(dates):
            actual = 120.0 + i * 0.3 + rng.normal(0, 1.0)
            forecast = 120.0 + i * 0.3 + rng.normal(0, 1.5)
            monitor.add_actual(date, actual)
            monitor.add_forecast(date, forecast, lower_95=forecast - 4, upper_95=forecast + 4)

        report = monitor.accuracy_report()
        assert isinstance(report, MonitorReport)
        assert "rmse" in report.overall_metrics

        rolling = monitor.rolling_accuracy(window=6)
        assert isinstance(rolling, pd.Series)

        bias = monitor.bias_tracker()
        assert isinstance(bias, pd.Series)

        # Step 3: Alerts
        alerts = AlertSystem(monitor)
        alerts.add_rule("test_rmse", metric="rmse", condition="above",
                       threshold=3.0, window=6, severity="warning")
        alerts.add_preset("bias_drift")

        triggered = alerts.check()
        assert isinstance(triggered, list)

        history = alerts.history()
        assert isinstance(history, list)

    def test_pipeline_with_preprocessing(self, sample_data: pd.Series) -> None:
        """Pipeline with preprocessing steps runs end-to-end."""
        pipeline = ForecastPipeline(
            data_source=sample_data.clip(lower=1.0),
            models=["auto_arima"],
            preprocess=["log", "diff"],
            horizon=6,
        )
        results = pipeline.run()
        assert len(results.forecasts) == 1
        assert "data" in results.execution_time
        assert "preprocess" in results.execution_time

    def test_pipeline_clone_and_modify(self, sample_data: pd.Series) -> None:
        """Clone pipeline, modify, and run independently."""
        pipeline = ForecastPipeline(
            data_source=sample_data,
            models=["auto_arima", "auto_ets"],
            horizon=12,
        )

        cloned = pipeline.clone()
        cloned.horizon = 6
        cloned.models = ["theta"]

        results_original = pipeline.run()
        results_cloned = cloned.run()

        assert len(results_original.forecasts) == 2
        assert len(results_cloned.forecasts) == 1
        assert list(results_original.forecasts.values())[0].horizon == 12
        assert list(results_cloned.forecasts.values())[0].horizon == 6

    def test_report_section_management(self, sample_data: pd.Series) -> None:
        """Test add, remove, reorder sections."""
        pipeline = ForecastPipeline(
            data_source=sample_data,
            models=["auto_arima"],
            horizon=6,
        )
        results = pipeline.run()

        builder = ReportBuilder(results=results, title="Test")
        builder.add_section("summary")
        builder.add_section("forecasts")
        builder.add_section("evaluation")
        builder.add_section("appendix")

        assert len(builder._sections) == 4

        # Remove
        builder.remove_section("appendix")
        assert len(builder._sections) == 3

        # Reorder
        builder.reorder_sections(["evaluation", "summary", "forecasts"])
        assert builder._section_order == ["evaluation", "summary", "forecasts"]

        # Render after modifications
        html = builder.render("html")
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_imports_from_top_level(self) -> None:
        """All key classes importable from expected paths."""
        from forecastbox.pipeline import (  # noqa: F811
            Alert,
            AlertRule,
            AlertSystem,  # noqa: F811
            ForecastMonitor,  # noqa: F811
            ForecastPipeline,  # noqa: F811
            MonitorReport,  # noqa: F811
            PipelineResults,  # noqa: F811
            RecurringForecast,  # noqa: F811
        )
        from forecastbox.reports import ReportBuilder  # noqa: F811
        from forecastbox.viz import (  # noqa: F811
            ForecastPlotter,  # noqa: F811
            get_color_palette,
            set_nodesecon_style,
        )

        assert ForecastPipeline is not None
        assert PipelineResults is not None
        assert RecurringForecast is not None
        assert ForecastMonitor is not None
        assert AlertSystem is not None
        assert Alert is not None
        assert AlertRule is not None
        assert MonitorReport is not None
        assert ForecastPlotter is not None
        assert set_nodesecon_style is not None
        assert get_color_palette is not None
        assert ReportBuilder is not None
