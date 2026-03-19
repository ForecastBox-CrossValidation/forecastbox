"""Tests for visualization module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from forecastbox.pipeline.pipeline import ForecastPipeline, PipelineResults
from forecastbox.viz._style import format_axis, get_color_palette, set_nodesecon_style
from forecastbox.viz.plotter import ForecastPlotter


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
def plotter(sample_results: PipelineResults) -> ForecastPlotter:
    """Create ForecastPlotter for testing."""
    return ForecastPlotter(sample_results, style="nodesecon")


class TestPlots:
    """Tests for visualization plots."""

    def test_forecast_plot(self, plotter: ForecastPlotter) -> None:
        """forecast_plot() executa sem erro."""
        ax = plotter.forecast_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_fan_chart_plot(self, plotter: ForecastPlotter) -> None:
        """fan_chart() executa sem erro."""
        ax = plotter.fan_chart()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_comparison_plot(self, plotter: ForecastPlotter) -> None:
        """comparison_plot() com 3 modelos."""
        ax = plotter.comparison_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_accuracy_plot(self, plotter: ForecastPlotter) -> None:
        """accuracy_plot() por horizonte."""
        ax = plotter.accuracy_plot(metric="rmse", by="model")
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_scenario_plot(self, plotter: ForecastPlotter) -> None:
        """scenario_plot() com 3 cenarios."""
        ax = plotter.scenario_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_nowcast_plot(self, plotter: ForecastPlotter) -> None:
        """nowcast_plot() executa sem erro."""
        ax = plotter.nowcast_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_combination_weights_plot(self, plotter: ForecastPlotter) -> None:
        """combination_weights_plot() executa sem erro."""
        ax = plotter.combination_weights_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_cv_plot(self, plotter: ForecastPlotter) -> None:
        """cv_plot() executa sem erro."""
        ax = plotter.cv_plot()
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_residual_plot(self, plotter: ForecastPlotter) -> None:
        """residual_plot() gera 4 paineis."""
        fig = plotter.residual_plot()
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close("all")

    def test_pipeline_dashboard(self, plotter: ForecastPlotter) -> None:
        """pipeline_dashboard() gera grid."""
        fig = plotter.pipeline_dashboard()
        assert isinstance(fig, plt.Figure)
        axes = fig.get_axes()
        assert len(axes) >= 4
        plt.close("all")

    def test_all_plots_return_axes(self, plotter: ForecastPlotter) -> None:
        """Todos retornam matplotlib axes ou figure."""
        # Axes-returning methods
        for method_name in [
            "forecast_plot",
            "fan_chart",
            "comparison_plot",
            "accuracy_plot",
            "scenario_plot",
            "nowcast_plot",
            "combination_weights_plot",
            "cv_plot",
        ]:
            method = getattr(plotter, method_name)
            result = method()
            assert isinstance(result, plt.Axes), f"{method_name} did not return Axes"
            plt.close("all")

        # Figure-returning methods
        for method_name in ["residual_plot", "pipeline_dashboard"]:
            method = getattr(plotter, method_name)
            result = method()
            assert isinstance(result, plt.Figure), f"{method_name} did not return Figure"
            plt.close("all")

    def test_custom_style(self) -> None:
        """Estilo NodesEcon aplicado corretamente."""
        set_nodesecon_style()
        assert plt.rcParams["figure.figsize"] == [12.0, 6.0]
        assert plt.rcParams["axes.grid"] is True

        colors = get_color_palette(3)
        assert len(colors) == 3
        assert colors[0] == "#1B3A5C"

        fig, ax = plt.subplots()
        ax = format_axis(ax, title="Test", xlabel="X", ylabel="Y")
        assert ax.get_title() == "Test"
        plt.close("all")
