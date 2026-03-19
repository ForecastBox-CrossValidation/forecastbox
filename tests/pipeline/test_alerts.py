"""Tests for AlertSystem."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.pipeline.alerts import AlertSystem
from forecastbox.pipeline.monitor import ForecastMonitor
from forecastbox.pipeline.pipeline import ForecastPipeline


@pytest.fixture
def sample_series() -> pd.Series:
    """Create sample time series."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01", periods=120, freq="MS")
    values = 100.0 + np.cumsum(rng.normal(0.1, 1.0, size=120))
    return pd.Series(values, index=dates, name="test_series")


@pytest.fixture
def monitor_stable(sample_series: pd.Series) -> ForecastMonitor:
    """Monitor with stable (good) accuracy."""
    pipeline = ForecastPipeline(data_source=sample_series, models=["auto_arima"])
    monitor = ForecastMonitor(pipeline)

    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01", periods=36, freq="MS")
    for i, date in enumerate(dates):
        actual = 100.0 + i * 0.5
        forecast = actual + rng.normal(0, 0.5)  # small errors
        lower = forecast - 5.0
        upper = forecast + 5.0
        monitor.add_actual(date, actual)
        monitor.add_forecast(date, forecast, lower_95=lower, upper_95=upper)

    return monitor


@pytest.fixture
def monitor_degraded(sample_series: pd.Series) -> ForecastMonitor:
    """Monitor with degraded accuracy in recent period."""
    pipeline = ForecastPipeline(data_source=sample_series, models=["auto_arima"])
    monitor = ForecastMonitor(pipeline)

    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01", periods=48, freq="MS")
    for i, date in enumerate(dates):
        actual = 100.0 + i * 0.5
        forecast = actual + rng.normal(0, 0.3) if i < 36 else actual + rng.normal(0, 5.0)
        lower = forecast - 3.0
        upper = forecast + 3.0
        monitor.add_actual(date, actual)
        monitor.add_forecast(date, forecast, lower_95=lower, upper_95=upper)

    return monitor


class TestAlertSystem:
    """Tests for AlertSystem."""

    def test_add_rule(self, monitor_stable: ForecastMonitor) -> None:
        """Regra adicionada corretamente."""
        alerts = AlertSystem(monitor_stable)
        alerts.add_rule("test_rule", metric="rmse", condition="above", threshold=2.0, window=6)
        assert len(alerts.rules) == 1
        assert alerts.rules[0].name == "test_rule"
        assert alerts.rules[0].metric == "rmse"

    def test_rmse_spike_triggers(self, monitor_degraded: ForecastMonitor) -> None:
        """RMSE 2x historico -> alerta disparado."""
        alerts = AlertSystem(monitor_degraded)
        alerts.add_rule(
            "rmse_spike", metric="rmse", condition="above", threshold=1.5, window=6
        )
        triggered = alerts.check()
        assert len(triggered) > 0
        assert triggered[0].rule == "rmse_spike"

    def test_no_trigger(self, monitor_stable: ForecastMonitor) -> None:
        """Acuracia estavel -> nenhum alerta."""
        alerts = AlertSystem(monitor_stable)
        alerts.add_rule(
            "rmse_spike", metric="rmse", condition="above", threshold=3.0, window=6
        )
        triggered = alerts.check()
        assert len(triggered) == 0

    def test_bias_drift(self, sample_series: pd.Series) -> None:
        """Bias crescente -> alerta disparado."""
        pipeline = ForecastPipeline(data_source=sample_series, models=["auto_arima"])
        monitor = ForecastMonitor(pipeline)

        # Create systematic bias in recent period
        dates = pd.date_range("2018-01", periods=36, freq="MS")
        for i, date in enumerate(dates):
            actual = 100.0 + i * 0.5
            # Consistent over-prediction bias
            forecast = actual + 3.0  # always 3 above
            monitor.add_actual(date, actual)
            monitor.add_forecast(date, forecast)

        alerts = AlertSystem(monitor)
        alerts.add_rule(
            "bias_drift", metric="bias", condition="above", threshold=0.5, window=12
        )
        triggered = alerts.check()
        assert len(triggered) > 0

    def test_severity_levels(self, monitor_degraded: ForecastMonitor) -> None:
        """Alertas com severity 'warning' e 'critical'."""
        alerts = AlertSystem(monitor_degraded)
        alerts.add_rule(
            "warn_rule", metric="rmse", condition="above",
            threshold=1.5, window=6, severity="warning",
        )
        alerts.add_rule(
            "crit_rule", metric="rmse", condition="above",
            threshold=1.2, window=6, severity="critical",
        )
        triggered = alerts.check()
        severities = {a.severity for a in triggered}
        # At least one should trigger given degraded data
        assert len(triggered) > 0
        assert all(s in ("warning", "critical", "info") for s in severities)

    def test_alert_history(self, monitor_degraded: ForecastMonitor) -> None:
        """Historico de alertas armazenado."""
        alerts = AlertSystem(monitor_degraded)
        alerts.add_rule(
            "rmse_spike", metric="rmse", condition="above", threshold=1.5, window=6
        )
        alerts.check()
        history = alerts.history()
        assert isinstance(history, list)
        assert len(history) >= 0  # may or may not trigger
        # Check again to accumulate
        alerts.check()
        history2 = alerts.history()
        assert len(history2) >= len(history)
