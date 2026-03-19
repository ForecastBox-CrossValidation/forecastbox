"""Tests for ForecastResults."""

from __future__ import annotations

import numpy as np

from forecastbox.core.forecast import Forecast
from forecastbox.core.results import ForecastResults


class TestForecastResults:
    """Tests for ForecastResults."""

    def _make_results(self) -> ForecastResults:
        """Helper to create a ForecastResults with 3 models."""
        actual = np.array([100.0, 101.0, 102.0])
        results = ForecastResults(actual=actual)

        # Model 1: good predictions
        good_fc = Forecast(
            point=np.array([100.1, 101.2, 101.8]), model_name="Good"
        )
        results.add_forecast("Good", good_fc)
        # Model 2: mediocre predictions
        medium_fc = Forecast(
            point=np.array([99.0, 102.5, 100.5]), model_name="Medium"
        )
        results.add_forecast("Medium", medium_fc)
        # Model 3: bad predictions
        results.add_forecast("Bad", Forecast(point=np.array([95.0, 105.0, 98.0]), model_name="Bad"))

        return results

    def test_add_forecast(self) -> None:
        """Add 3 forecasts, verify len(results.forecasts) == 3."""
        results = self._make_results()
        assert len(results.forecasts) == 3

    def test_evaluate(self) -> None:
        """Metrics calculated for all models."""
        results = self._make_results()
        df = results.evaluate(metrics=("mae", "rmse"))
        assert "mae" in df.columns
        assert "rmse" in df.columns
        assert len(df) == 3

    def test_rank(self) -> None:
        """Ranking correct by RMSE."""
        results = self._make_results()
        results.evaluate(metrics=("rmse",))
        ranking = results.rank("rmse")
        assert ranking[0] == "Good"
        assert ranking[-1] == "Bad"

    def test_best(self) -> None:
        """Best model has lowest RMSE."""
        results = self._make_results()
        results.evaluate(metrics=("rmse",))
        assert results.best("rmse") == "Good"

    def test_summary(self) -> None:
        """summary() produces formatted string without error."""
        results = self._make_results()
        s = results.summary()
        assert isinstance(s, str)
        assert "Good" in s
        assert "Bad" in s

    def test_to_dataframe(self) -> None:
        """All forecasts in long format DataFrame."""
        results = self._make_results()
        df = results.to_dataframe()
        assert "model" in df.columns
        assert "point" in df.columns
        assert len(df) == 9  # 3 models * 3 horizons

    def test_plot_comparison(self) -> None:
        """plot_comparison executes without error."""
        import matplotlib
        matplotlib.use("Agg")

        results = self._make_results()
        results.evaluate(metrics=("rmse",))
        ax = results.plot_comparison("rmse")
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_forecasts(self) -> None:
        """plot_forecasts executes without error."""
        import matplotlib
        matplotlib.use("Agg")

        results = self._make_results()
        ax = results.plot_forecasts()
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")
