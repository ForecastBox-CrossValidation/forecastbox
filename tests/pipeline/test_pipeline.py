"""Tests for ForecastPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.pipeline.pipeline import ForecastPipeline, PipelineResults


@pytest.fixture
def sample_series() -> pd.Series:
    """Create sample time series for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2010-01", periods=120, freq="MS")
    values = 100.0 + np.cumsum(rng.normal(0.1, 1.0, size=120))
    return pd.Series(values, index=dates, name="test_series")


@pytest.fixture
def basic_pipeline(sample_series: pd.Series) -> ForecastPipeline:
    """Create basic pipeline for testing."""
    return ForecastPipeline(
        data_source=sample_series,
        models=["auto_arima", "auto_ets", "theta"],
        combination="mean",
        evaluation=["rmse", "mae"],
        horizon=12,
        cv_type="expanding",
    )


class TestPipeline:
    """Tests for ForecastPipeline."""

    def test_pipeline_end_to_end(self, basic_pipeline: ForecastPipeline) -> None:
        """Pipeline completo roda sem erro com dados sinteticos."""
        results = basic_pipeline.run()
        assert isinstance(results, PipelineResults)
        assert len(results.forecasts) == 3
        assert results.combination is not None

    def test_pipeline_results_complete(self, basic_pipeline: ForecastPipeline) -> None:
        """PipelineResults tem forecasts, evaluation, cv_results."""
        results = basic_pipeline.run()
        assert len(results.forecasts) > 0
        assert not results.evaluation.empty
        assert len(results.cv_results) > 0
        assert "auto_arima" in results.forecasts
        assert "auto_ets" in results.forecasts
        assert "theta" in results.forecasts

    def test_pipeline_custom_step(self, basic_pipeline: ForecastPipeline) -> None:
        """Adicionar etapa customizada funciona."""
        custom_called = {"value": False}

        def my_custom_step(results: PipelineResults) -> None:
            custom_called["value"] = True
            results.metadata["custom"] = "executed"

        basic_pipeline.add_step("custom_check", my_custom_step, after="evaluate")
        assert "custom_check" in basic_pipeline.steps()

        results = basic_pipeline.run()
        assert custom_called["value"] is True
        assert results.metadata.get("custom") == "executed"

    def test_pipeline_clone(self, basic_pipeline: ForecastPipeline) -> None:
        """clone() cria copia independente."""
        cloned = basic_pipeline.clone()
        assert cloned is not basic_pipeline
        assert cloned.models == basic_pipeline.models
        assert cloned.horizon == basic_pipeline.horizon

        # Modifying clone should not affect original
        cloned.horizon = 24
        assert basic_pipeline.horizon == 12

    def test_pipeline_summary(self, basic_pipeline: ForecastPipeline) -> None:
        """summary() produz string formatada sem erro."""
        results = basic_pipeline.run()
        summary = results.summary()
        assert isinstance(summary, str)
        assert "PIPELINE RESULTS SUMMARY" in summary
        assert "auto_arima" in summary

    def test_pipeline_preprocess(self, sample_series: pd.Series) -> None:
        """Preprocessamento log + diff aplicado corretamente."""
        pipeline = ForecastPipeline(
            data_source=sample_series.clip(lower=1.0),
            models=["auto_arima"],
            preprocess=["log", "diff"],
            horizon=6,
        )
        results = pipeline.run()
        assert isinstance(results, PipelineResults)
        assert len(results.forecasts) == 1

    def test_pipeline_combination(self, sample_series: pd.Series) -> None:
        """Pipeline com combination='bma' gera forecast combinado."""
        pipeline = ForecastPipeline(
            data_source=sample_series,
            models=["auto_arima", "auto_ets"],
            combination="bma",
            horizon=12,
        )
        results = pipeline.run()
        assert results.combination is not None
        assert "Combined" in results.combination.model_name

    def test_pipeline_execution_time(self, basic_pipeline: ForecastPipeline) -> None:
        """execution_time registrado para cada etapa."""
        results = basic_pipeline.run()
        assert len(results.execution_time) > 0
        assert "data" in results.execution_time
        assert "fit" in results.execution_time
        assert "forecast" in results.execution_time
        assert "evaluate" in results.execution_time
        for step, t in results.execution_time.items():
            assert t >= 0.0, f"Step {step} has negative time"
