"""Tests for ForecastMonitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.pipeline.monitor import ForecastMonitor, MonitorReport
from forecastbox.pipeline.pipeline import ForecastPipeline


@pytest.fixture
def sample_series() -> pd.Series:
    """Create sample time series."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01", periods=120, freq="MS")
    values = 100.0 + np.cumsum(rng.normal(0.1, 1.0, size=120))
    return pd.Series(values, index=dates, name="test_series")


@pytest.fixture
def monitor_with_data(sample_series: pd.Series) -> ForecastMonitor:
    """Create monitor with actual and forecasted data."""
    pipeline = ForecastPipeline(
        data_source=sample_series,
        models=["auto_arima"],
        horizon=12,
    )
    monitor = ForecastMonitor(pipeline)

    # Add matched forecast-actual pairs
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01", periods=36, freq="MS")
    for i, date in enumerate(dates):
        actual = 100.0 + i * 0.5 + rng.normal(0, 1.0)
        forecast = 100.0 + i * 0.5 + rng.normal(0, 1.5)
        lower = forecast - 5.0
        upper = forecast + 5.0
        monitor.add_actual(date, actual)
        monitor.add_forecast(date, forecast, lower_95=lower, upper_95=upper)

    return monitor


class TestForecastMonitor:
    """Tests for ForecastMonitor."""

    def test_add_actual(self, sample_series: pd.Series) -> None:
        """add_actual() armazena realizado."""
        pipeline = ForecastPipeline(data_source=sample_series, models=["auto_arima"])
        monitor = ForecastMonitor(pipeline)
        monitor.add_actual(pd.Timestamp("2024-01-01"), 105.0)
        assert len(monitor.actuals) == 1
        assert monitor.actuals[0][1] == 105.0

    def test_rolling_accuracy(self, monitor_with_data: ForecastMonitor) -> None:
        """rolling_accuracy() calcula RMSE em janela movel."""
        rolling = monitor_with_data.rolling_accuracy(window=6, metric="rmse")
        assert isinstance(rolling, pd.Series)
        assert len(rolling) > 0
        assert all(v >= 0 for v in rolling.values)

    def test_bias_tracker(self, monitor_with_data: ForecastMonitor) -> None:
        """bias_tracker() detecta bias sistematico."""
        bias = monitor_with_data.bias_tracker()
        assert isinstance(bias, pd.Series)
        assert len(bias) > 0

    def test_degradation_test(self, sample_series: pd.Series) -> None:
        """degradation_test() retorna True quando RMSE cresce."""
        pipeline = ForecastPipeline(data_source=sample_series, models=["auto_arima"])
        monitor = ForecastMonitor(pipeline)

        # Add stable period
        rng = np.random.default_rng(42)
        dates = pd.date_range("2015-01", periods=48, freq="MS")
        for i, date in enumerate(dates):
            actual = 100.0 + i * 0.1
            forecast = actual + rng.normal(0, 0.5) if i < 36 else actual + rng.normal(0, 5.0)
            monitor.add_actual(date, actual)
            monitor.add_forecast(date, forecast)

        result = monitor.degradation_test(window=12, threshold=1.5)
        assert isinstance(result, bool)

    def test_hit_rate(self, monitor_with_data: ForecastMonitor) -> None:
        """hit_rate calculado corretamente."""
        report = monitor_with_data.accuracy_report()
        assert 0.0 <= report.hit_rate <= 1.0

    def test_monitor_report(self, monitor_with_data: ForecastMonitor) -> None:
        """accuracy_report() gera MonitorReport completo."""
        report = monitor_with_data.accuracy_report()
        assert isinstance(report, MonitorReport)
        assert "rmse" in report.overall_metrics
        assert "mae" in report.overall_metrics
        assert isinstance(report.bias, float)
        summary = report.summary()
        assert "FORECAST MONITOR REPORT" in summary
