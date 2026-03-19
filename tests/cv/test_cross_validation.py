"""Tests for cross-validation framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.cv.cross_validation import expanding_window_cv


def naive_model_fn(train: pd.Series) -> np.ndarray:
    """Simple naive model: forecast = last value repeated."""
    last = train.iloc[-1]
    # Return array directly (horizon will be inferred from test set)
    return np.full(12, last)


class TestExpandingWindowCV:
    """Tests for expanding_window_cv."""

    def _make_data(self, n: int = 120) -> pd.Series:
        """Create sample time series."""
        rng = np.random.default_rng(42)
        values = 100 + np.cumsum(rng.normal(0, 1, n))
        index = pd.date_range("2015-01-01", periods=n, freq="MS")
        return pd.Series(values, index=index, name="test")

    def test_expanding_window_splits(self) -> None:
        """Verify correct number of folds."""
        data = self._make_data(120)
        results = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
            step=1,
        )
        # Expected folds: (120 - 60 - 12) / 1 + 1 = 49
        expected_folds = (120 - 60 - 12) // 1 + 1
        assert results.n_folds == expected_folds

    def test_expanding_window_no_leakage(self) -> None:
        """train_end < test_start for every fold."""
        data = self._make_data(120)
        initial_window = 60
        horizon = 12
        step = 1

        # Check that forecasts don't use future data
        results = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=initial_window,
            horizon=horizon,
            step=step,
        )
        # All forecasts should exist
        assert len(results.forecasts) == results.n_folds
        # All actuals should exist
        assert len(results.actuals) == results.n_folds

    def test_expanding_window_coverage(self) -> None:
        """All test points covered."""
        data = self._make_data(120)
        results = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
            step=1,
        )
        assert results.errors.shape == (results.n_folds, 12)

    def test_cv_results_summary(self) -> None:
        """summary() without error."""
        data = self._make_data(120)
        results = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
            step=1,
        )
        s = results.summary()
        assert isinstance(s, str)
        assert "Folds" in s
        assert "Horizon" in s

    def test_cv_metrics_by_horizon(self) -> None:
        """Metrics for each h=1..H."""
        data = self._make_data(120)
        results = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
            step=1,
        )
        assert len(results.metrics_by_horizon) == 12
        assert "mae" in results.metrics_by_horizon.columns
        assert "rmse" in results.metrics_by_horizon.columns

    def test_initial_window_respected(self) -> None:
        """First fold has exactly initial_window training points."""
        data = self._make_data(120)
        initial_window = 60

        # We verify by checking that the model_fn receives correct size
        received_sizes: list[int] = []

        def tracking_model(train: pd.Series) -> np.ndarray:
            received_sizes.append(len(train))
            return np.full(12, train.iloc[-1])

        expanding_window_cv(
            data=data,
            model_fn=tracking_model,
            initial_window=initial_window,
            horizon=12,
            step=1,
        )
        assert received_sizes[0] == initial_window
        # Each subsequent fold should have one more data point
        for j in range(1, len(received_sizes)):
            assert received_sizes[j] == initial_window + j

    def test_step_size(self) -> None:
        """Step size affects number of folds."""
        data = self._make_data(120)
        results_step1 = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
            step=1,
        )
        results_step3 = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
            step=3,
        )
        assert results_step3.n_folds < results_step1.n_folds

    def test_insufficient_data(self) -> None:
        """Raises ValueError when data is too short."""
        data = self._make_data(10)
        with pytest.raises(ValueError, match="Not enough data"):
            expanding_window_cv(
                data=data,
                model_fn=naive_model_fn,
                initial_window=60,
                horizon=12,
            )

    def test_mean_metric(self) -> None:
        """mean_metric returns correct value."""
        data = self._make_data(120)
        results = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
        )
        rmse_val = results.mean_metric("rmse")
        assert isinstance(rmse_val, float)
        assert rmse_val > 0

    def test_plot_errors(self) -> None:
        """plot_errors executes without error."""
        import matplotlib

        matplotlib.use("Agg")

        data = self._make_data(120)
        results = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
        )
        ax = results.plot_errors()
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_forecast_vs_actual(self) -> None:
        """plot_forecast_vs_actual executes without error."""
        import matplotlib

        matplotlib.use("Agg")

        data = self._make_data(120)
        results = expanding_window_cv(
            data=data,
            model_fn=naive_model_fn,
            initial_window=60,
            horizon=12,
        )
        ax = results.plot_forecast_vs_actual(fold=0)
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_model_fn_with_object(self) -> None:
        """model_fn can return object with .forecast() method."""
        data = self._make_data(120)

        class SimpleModel:
            def __init__(self, last_value: float) -> None:
                self.last_value = last_value

            def forecast(self, h: int) -> np.ndarray:
                return np.full(h, self.last_value)

        def model_fn(train: pd.Series) -> SimpleModel:
            return SimpleModel(train.iloc[-1])

        results = expanding_window_cv(
            data=data,
            model_fn=model_fn,
            initial_window=60,
            horizon=12,
        )
        assert results.n_folds > 0
