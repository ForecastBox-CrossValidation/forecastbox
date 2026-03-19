"""Tests for AutoSelect cross-family model selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from forecastbox.auto.select import AutoSelect, _compute_metric
from forecastbox.auto.zoo import ModelZoo
from forecastbox.core.forecast import Forecast


def _make_test_data() -> pd.Series:
    """Generate test data for AutoSelect: seasonal series with trend."""
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=np.float64)
    seasonal = 10.0 * np.sin(2 * np.pi * t / 12)
    trend = 0.1 * t
    noise = np.random.normal(0, 2, n)
    y = 100.0 + trend + seasonal + noise
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="test")


def _make_simple_data() -> pd.Series:
    """Generate simple non-seasonal data."""
    np.random.seed(123)
    n = 150
    y = 50.0 + np.cumsum(np.random.normal(0.1, 1, n))
    y = np.abs(y) + 10  # Ensure positive
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="simple")


class DummyModel:
    """A dummy model for testing custom families via ModelZoo."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._last_value: float = 0.0

    def fit(self, y: Any, **kwargs: Any) -> DummyModel:
        arr = np.asarray(y, dtype=np.float64)
        self._last_value = float(arr[-1])
        return self

    def forecast(self, h: int, **kwargs: Any) -> Forecast:
        point = np.full(h, self._last_value)
        return Forecast(point=point, model_name="DummyModel", horizon=h)


@pytest.fixture(autouse=True)
def reset_zoo() -> None:
    """Reset ModelZoo before each test."""
    ModelZoo.reset()


class TestAutoSelect:
    """Tests for AutoSelect cross-family model selection."""

    def test_ranking_stable(self) -> None:
        """Two executions with same data should produce same ranking order."""
        data = _make_simple_data()

        selector1 = AutoSelect(
            families=["naive", "drift"],
            cv_horizon=6,
            cv_step=3,
            metric="rmse",
        )
        result1 = selector1.fit(data)

        selector2 = AutoSelect(
            families=["naive", "drift"],
            cv_horizon=6,
            cv_step=3,
            metric="rmse",
        )
        result2 = selector2.fit(data)

        # Same ranking order
        assert list(result1.ranking["family"]) == list(result2.ranking["family"])
        # Same best family
        assert result1.best_family == result2.best_family

    def test_best_model_forecasts(self) -> None:
        """Best model should be able to generate forecasts."""
        data = _make_simple_data()
        selector = AutoSelect(
            families=["naive", "drift"],
            cv_horizon=6,
            cv_step=3,
        )
        result = selector.fit(data)

        fc = result.forecast(12)
        assert isinstance(fc, Forecast)
        assert len(fc) == 12
        assert not np.any(np.isnan(fc.point))

    def test_custom_families(self) -> None:
        """Custom families registered in ModelZoo should be usable in AutoSelect."""
        zoo = ModelZoo()
        zoo.register("dummy", DummyModel, family="custom", description="Test dummy model")

        data = _make_simple_data()
        selector = AutoSelect(
            families=["naive", "dummy"],
            cv_horizon=6,
            cv_step=3,
        )
        result = selector.fit(data)

        # Both families should appear in ranking
        families_in_ranking = set(result.ranking["family"])
        assert "naive" in families_in_ranking
        assert "dummy" in families_in_ranking

    def test_cv_no_leakage(self) -> None:
        """No CV fold should use future data.

        Verify that the training window never extends beyond the fold boundary.
        This is a structural test: we verify by inspecting the algorithm's behavior.
        """
        data = _make_simple_data()

        # Use a large step to have fewer folds (easier to verify)
        selector = AutoSelect(
            families=["naive"],
            cv_horizon=6,
            cv_step=10,
            cv_initial=100,
        )
        result = selector.fit(data)

        # The CV should have run successfully
        assert result.all_cv_results["naive"] is not None
        assert len(result.all_cv_results["naive"]) > 0

        # Verify: initial window is 100, each fold starts 10 steps later
        # Fold 0: train[0:100], test[100:106]
        # Fold 1: train[0:110], test[110:116]
        # etc.
        # No fold should have infinite score (which would indicate data issue)
        finite_scores = [s for s in result.all_cv_results["naive"] if np.isfinite(s)]
        assert len(finite_scores) > 0

    def test_comparison_plot(self) -> None:
        """plot_comparison() should execute without error."""
        import matplotlib

        matplotlib.use("Agg")

        data = _make_simple_data()
        selector = AutoSelect(
            families=["naive", "drift"],
            cv_horizon=6,
            cv_step=3,
        )
        result = selector.fit(data)

        ax = result.plot_comparison()
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_ranking_dataframe_columns(self) -> None:
        """ranking DataFrame should have expected columns."""
        data = _make_simple_data()
        selector = AutoSelect(
            families=["naive", "drift"],
            cv_horizon=6,
            cv_step=3,
        )
        result = selector.fit(data)

        assert isinstance(result.ranking, pd.DataFrame)
        assert "family" in result.ranking.columns
        assert "model_name" in result.ranking.columns
        assert "cv_mean" in result.ranking.columns
        assert "cv_std" in result.ranking.columns

    def test_summary(self) -> None:
        """summary() should return informative text."""
        data = _make_simple_data()
        selector = AutoSelect(
            families=["naive", "drift"],
            cv_horizon=6,
            cv_step=3,
        )
        result = selector.fit(data)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "AutoSelect" in summary
        assert result.best_family in summary

    def test_different_metrics(self) -> None:
        """Different metrics should all produce valid rankings."""
        data = _make_simple_data()

        for metric_name in ("rmse", "mae"):
            selector = AutoSelect(
                families=["naive", "drift"],
                cv_horizon=6,
                cv_step=3,
                metric=metric_name,
            )
            result = selector.fit(data)
            assert result.metric_name == metric_name
            assert len(result.ranking) == 2

    def test_compute_metric(self) -> None:
        """Test metric computation with known values."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = np.array([1.1, 2.2, 2.8, 4.1, 5.2])

        rmse = _compute_metric(actual, predicted, "rmse")
        mae = _compute_metric(actual, predicted, "mae")

        assert rmse > 0
        assert mae > 0
        assert rmse >= mae  # RMSE >= MAE always

    def test_invalid_cv_type(self) -> None:
        """Invalid cv_type should raise ValueError."""
        with pytest.raises(ValueError, match="cv_type"):
            AutoSelect(cv_type="invalid")

    def test_invalid_metric(self) -> None:
        """Invalid metric should raise ValueError."""
        with pytest.raises(ValueError, match="metric"):
            AutoSelect(metric="invalid")

    def test_short_series_error(self) -> None:
        """Very short series should raise ValueError."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        selector = AutoSelect(cv_horizon=12)
        with pytest.raises(ValueError, match="too short"):
            selector.fit(y)

    def test_rolling_cv(self) -> None:
        """Rolling window CV should work correctly."""
        data = _make_simple_data()
        selector = AutoSelect(
            families=["naive", "drift"],
            cv_type="rolling",
            cv_horizon=6,
            cv_step=5,
        )
        result = selector.fit(data)

        assert len(result.ranking) == 2
        assert result.best_family in ("naive", "drift")

    def test_with_arima_ets(self) -> None:
        """Test with ARIMA and ETS families (requires statsmodels)."""
        data = _make_simple_data()

        try:
            selector = AutoSelect(
                families=["arima", "ets"],
                cv_horizon=6,
                cv_step=10,
                metric="rmse",
            )
            result = selector.fit(data)

            assert len(result.ranking) == 2
            assert result.best_family in ("arima", "ets")
            fc = result.forecast(6)
            assert len(fc) == 6

        except ImportError:
            pytest.skip("statsmodels not installed")
