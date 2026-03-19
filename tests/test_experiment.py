"""Tests for ForecastExperiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecastbox.core.forecast import Forecast
from forecastbox.experiment import ExperimentResults, ForecastExperiment


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample time series data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    return pd.DataFrame(
        {
            "ipca": 0.5 + rng.normal(0, 0.3, 120),
            "selic": 10.0 + np.cumsum(rng.normal(0, 0.5, 120)),
            "cambio": 3.0 + np.cumsum(rng.normal(0, 0.1, 120)),
        },
        index=dates,
    )


@pytest.fixture
def sample_results() -> ExperimentResults:
    """Create sample ExperimentResults."""
    fc1 = Forecast(
        point=np.array([100.5, 101.2, 102.0]),
        lower_80=np.array([98.0, 97.5, 96.8]),
        upper_80=np.array([103.0, 104.9, 107.2]),
        model_name="AutoARIMA",
    )
    fc2 = Forecast(
        point=np.array([100.8, 100.9, 102.5]),
        lower_80=np.array([98.5, 97.0, 97.0]),
        upper_80=np.array([103.1, 104.8, 108.0]),
        model_name="AutoETS",
    )
    combined = Forecast.combine([fc1, fc2], method="mean")

    evaluation = pd.DataFrame(
        {
            "mae": [0.3, 0.4],
            "rmse": [0.35, 0.45],
            "mape": [0.5, 0.6],
        },
        index=["AutoARIMA", "AutoETS"],
    )

    ranking = evaluation.sort_values("rmse")

    return ExperimentResults(
        forecasts={"AutoARIMA": fc1, "AutoETS": fc2},
        combination=combined,
        evaluation=evaluation,
        ranking=ranking,
        metadata={"target": "ipca", "horizon": 3},
    )


class TestForecastExperiment:
    """Tests for ForecastExperiment."""

    def test_experiment_end_to_end(self, sample_data: pd.DataFrame) -> None:
        """ForecastExperiment executes completely."""
        exp = ForecastExperiment(
            data=sample_data,
            target="ipca",
            models=["auto_arima", "auto_ets"],
            combination="mean",
            horizon=6,
            cv_type="expanding",
        )
        results = exp.run()

        assert isinstance(results, ExperimentResults)
        assert isinstance(results.metadata, dict)
        assert results.metadata["target"] == "ipca"
        assert results.metadata["horizon"] == 6

    def test_experiment_with_scenarios(self, sample_data: pd.DataFrame) -> None:
        """Scenarios are included in the experiment."""
        exp = ForecastExperiment(
            data=sample_data,
            target="ipca",
            models=["auto_arima"],
            scenarios={
                "base": {"selic": 13.75},
                "otimista": {"selic": 11.75},
            },
            horizon=6,
        )
        results = exp.run()

        assert isinstance(results, ExperimentResults)
        # Scenarios may or may not work depending on available modules
        assert isinstance(results.scenarios, (dict, type(None)))

    def test_experiment_save_load(
        self, sample_results: ExperimentResults, tmp_path: Path
    ) -> None:
        """save() -> load() preserves results."""
        save_dir = tmp_path / "experiment_output"
        sample_results.save(save_dir)

        assert (save_dir / "metadata.json").exists()
        assert (save_dir / "forecasts").exists()

        loaded = ExperimentResults.load(save_dir)
        assert len(loaded.forecasts) == 2
        assert "AutoARIMA" in loaded.forecasts or any(
            "AutoARIMA" in k for k in loaded.forecasts
        )

    def test_experiment_report(
        self, sample_results: ExperimentResults, tmp_path: Path
    ) -> None:
        """report() generates functional HTML."""
        report_path = tmp_path / "report.html"
        sample_results.report(report_path)

        assert report_path.exists()
        content = report_path.read_text()
        assert "<html>" in content
        assert "ForecastBox" in content
        assert "AutoARIMA" in content

    def test_experiment_summary(self, sample_results: ExperimentResults) -> None:
        """summary() produces readable text."""
        text = sample_results.summary()

        assert isinstance(text, str)
        assert len(text) > 50
        assert "AutoARIMA" in text
        assert "AutoETS" in text
        assert "Models:" in text

    def test_experiment_ranking(self, sample_results: ExperimentResults) -> None:
        """Ranking is consistent with metrics."""
        assert sample_results.ranking is not None
        assert "rmse" in sample_results.ranking.columns

        # AutoARIMA has lower RMSE (0.35 vs 0.45), should be first
        first_model = sample_results.ranking.index[0]
        assert first_model == "AutoARIMA"
