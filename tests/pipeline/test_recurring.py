"""Tests for RecurringForecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.pipeline.pipeline import ForecastPipeline, PipelineResults
from forecastbox.pipeline.recurring import RecurringForecast


@pytest.fixture
def sample_series() -> pd.Series:
    """Create sample time series."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01", periods=120, freq="MS")
    values = 100.0 + np.cumsum(rng.normal(0.1, 1.0, size=120))
    return pd.Series(values, index=dates, name="test_series")


@pytest.fixture
def basic_pipeline(sample_series: pd.Series) -> ForecastPipeline:
    """Create basic pipeline."""
    return ForecastPipeline(
        data_source=sample_series,
        models=["auto_arima", "auto_ets"],
        combination="mean",
        horizon=6,
    )


class TestRecurringForecast:
    """Tests for RecurringForecast."""

    def test_run_once(self, basic_pipeline: ForecastPipeline) -> None:
        """run_once() executa pipeline e armazena resultado."""
        recurring = RecurringForecast(pipeline=basic_pipeline)
        result = recurring.run_once()
        assert isinstance(result, PipelineResults)
        assert len(recurring.history()) == 1

    def test_history(self, basic_pipeline: ForecastPipeline) -> None:
        """Apos 3 execucoes, history() tem 3 entradas."""
        recurring = RecurringForecast(pipeline=basic_pipeline)
        recurring.run_once()
        recurring.run_once()
        recurring.run_once()
        assert len(recurring.history()) == 3

    def test_forecast_evolution(self, basic_pipeline: ForecastPipeline) -> None:
        """forecast_evolution() mostra como previsao mudou."""
        recurring = RecurringForecast(pipeline=basic_pipeline)
        recurring.run_once()
        recurring.run_once()
        evolution = recurring.forecast_evolution()
        assert isinstance(evolution, pd.DataFrame)
        assert len(evolution) == 2

    def test_data_updater(self, sample_series: pd.Series) -> None:
        """data_updater callable e invocado a cada execucao."""
        call_count = {"value": 0}

        def updater() -> pd.Series:
            call_count["value"] += 1
            return sample_series

        pipeline = ForecastPipeline(
            data_source=sample_series,
            models=["auto_arima"],
            horizon=6,
        )
        recurring = RecurringForecast(
            pipeline=pipeline,
            data_updater=updater,
        )
        recurring.run_once()
        recurring.run_once()
        assert call_count["value"] == 2
