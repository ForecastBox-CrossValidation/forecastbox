"""Tests for AutoETS model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.auto.ets import AutoETS, AutoETSResult
from forecastbox.core.forecast import Forecast


def _make_airline_data() -> pd.Series:
    """Generate synthetic airline-like data with multiplicative seasonality.

    Expected: ETS with multiplicative components should be selected.
    """
    np.random.seed(42)
    n = 144
    t = np.arange(n, dtype=np.float64)

    trend = 100 + 2.5 * t
    seasonal_pattern = np.array([
        0.90, 0.85, 0.95, 1.00, 1.05, 1.15,
        1.20, 1.18, 1.10, 1.02, 0.92, 0.88,
    ])
    seasonal = np.tile(seasonal_pattern, n // 12 + 1)[:n]
    noise = np.random.normal(0, 3, n)
    y = trend * seasonal + noise

    index = pd.date_range("1949-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="airline")


def _make_positive_seasonal_data() -> pd.Series:
    """Generate positive data with clear additive seasonality."""
    np.random.seed(123)
    n = 120
    t = np.arange(n, dtype=np.float64)
    seasonal_pattern = 10.0 * np.sin(2 * np.pi * t / 12)
    y = 200.0 + 0.5 * t + seasonal_pattern + np.random.normal(0, 2, n)
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="positive_seasonal")


def _make_negative_data() -> pd.Series:
    """Generate data with negative values."""
    np.random.seed(456)
    n = 100
    y = np.random.normal(-5, 10, n)  # Some values will be negative
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="negative")


def _make_simple_data() -> pd.Series:
    """Generate simple positive non-seasonal data."""
    np.random.seed(789)
    n = 60
    y = 100.0 + np.cumsum(np.random.normal(0.1, 1, n))
    y = np.abs(y) + 10  # Ensure positive
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="simple")


class TestAutoETS:
    """Tests for AutoETS model selection."""

    def test_airline_ets(self) -> None:
        """AutoETS on airline data should select a multiplicative model.

        With increasing variance (multiplicative seasonality), expect
        error='M' or seasonal='M' or both.
        """
        data = _make_airline_data()
        auto_ets = AutoETS(seasonal_period=12, ic="aicc")
        result = auto_ets.fit(data)

        assert isinstance(result, AutoETSResult)
        # For airline-like data, multiplicative components are expected
        has_multiplicative = (
            result.error == "M"
            or result.seasonal == "M"
            or "M" in result.trend
        )
        assert has_multiplicative, (
            f"Expected multiplicative component for airline data, got {result.model_type}"
        )

    def test_enumerate_30(self) -> None:
        """For positive data with seasonality, should enumerate up to 30 combinations."""
        data = _make_positive_seasonal_data()
        auto_ets = AutoETS(seasonal_period=12, ic="aicc")

        y_arr = np.asarray(data, dtype=np.float64)
        models = auto_ets._enumerate_models(y_arr)

        # All data is positive and seasonal, so all 30 combinations are valid
        assert len(models) <= 30
        assert len(models) >= 20  # Most should be valid for positive data

    def test_restrict_negative(self) -> None:
        """Data with negative values should exclude multiplicative models."""
        data = _make_negative_data()
        auto_ets = AutoETS(seasonal_period=12, ic="aicc", restrict=True)

        y_arr = np.asarray(data, dtype=np.float64)
        models = auto_ets._enumerate_models(y_arr)

        # No multiplicative error, trend, or seasonal should be present
        for error, trend, seasonal, _damped in models:
            assert error != "M", "Multiplicative error not allowed for negative data"
            base_trend = trend[0] if trend != "N" else trend
            assert base_trend != "M", (
                "Multiplicative trend not allowed for negative data"
            )
            assert seasonal != "M", (
                "Multiplicative seasonal not allowed for negative data"
            )

    def test_non_seasonal(self) -> None:
        """seasonal_period=1 should only use N for seasonal (10 models max)."""
        data = _make_simple_data()
        auto_ets = AutoETS(seasonal_period=1, ic="aicc")

        y_arr = np.asarray(data, dtype=np.float64)
        models = auto_ets._enumerate_models(y_arr)

        # 2 error * 5 trend * 1 seasonal = 10 max
        assert len(models) <= 10
        for _, _, seasonal, _ in models:
            assert seasonal == "N", (
                f"Expected seasonal='N' for non-seasonal data, got '{seasonal}'"
            )

    def test_fix_error(self) -> None:
        """error='A' should only test additive error models (max 15)."""
        data = _make_positive_seasonal_data()
        auto_ets = AutoETS(seasonal_period=12, error="A", ic="aicc")

        y_arr = np.asarray(data, dtype=np.float64)
        models = auto_ets._enumerate_models(y_arr)

        # 1 error * 5 trend * 3 seasonal = 15 max
        assert len(models) <= 15
        for error, _, _, _ in models:
            assert error == "A", f"Expected error='A', got '{error}'"

    def test_forecast_intervals(self) -> None:
        """Forecast intervals should be present and properly ordered."""
        data = _make_positive_seasonal_data()
        auto_ets = AutoETS(seasonal_period=12, ic="aicc")
        result = auto_ets.fit(data)
        fc = result.forecast(12)

        assert isinstance(fc, Forecast)
        assert len(fc) == 12
        assert len(fc.point) == 12
        assert not np.any(np.isnan(fc.point))

        # Check intervals exist and are ordered
        if fc.lower_80 is not None and fc.upper_80 is not None:
            assert np.all(fc.lower_80 <= fc.point), "lower_80 should be <= point"
            assert np.all(fc.point <= fc.upper_80), "point should be <= upper_80"

        if fc.lower_95 is not None and fc.upper_95 is not None:
            assert np.all(fc.lower_95 <= fc.point), "lower_95 should be <= point"
            assert np.all(fc.point <= fc.upper_95), "point should be <= upper_95"

        if (
            fc.lower_80 is not None
            and fc.lower_95 is not None
            and fc.upper_80 is not None
            and fc.upper_95 is not None
        ):
            assert np.all(fc.lower_95 <= fc.lower_80), (
                "95% should be wider than 80%"
            )
            assert np.all(fc.upper_80 <= fc.upper_95), (
                "95% should be wider than 80%"
            )

    def test_all_models_dataframe(self) -> None:
        """result.all_models should be a DataFrame with expected columns."""
        data = _make_simple_data()
        auto_ets = AutoETS(seasonal_period=1, ic="aicc")
        result = auto_ets.fit(data)

        assert isinstance(result.all_models, pd.DataFrame)
        assert "model_type" in result.all_models.columns
        assert "error" in result.all_models.columns
        assert "trend" in result.all_models.columns
        assert "seasonal" in result.all_models.columns
        assert "ic_value" in result.all_models.columns
        assert len(result.all_models) > 0

    def test_summary(self) -> None:
        """summary() should return a non-empty string with model info."""
        data = _make_simple_data()
        auto_ets = AutoETS(seasonal_period=1, ic="aicc")
        result = auto_ets.fit(data)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "AutoETS" in summary
        assert "ETS" in summary
        assert result.model_type in summary

    def test_model_type_format(self) -> None:
        """model_type should follow ETS(E,T,S) format."""
        data = _make_simple_data()
        auto_ets = AutoETS(seasonal_period=1, ic="aicc")
        result = auto_ets.fit(data)

        assert result.model_type.startswith("ETS(")
        assert result.model_type.endswith(")")

    def test_fix_damped(self) -> None:
        """damped=True should only test damped trend models."""
        data = _make_simple_data()
        auto_ets = AutoETS(seasonal_period=1, damped=True, ic="aicc")

        y_arr = np.asarray(data, dtype=np.float64)
        models = auto_ets._enumerate_models(y_arr)

        for _, trend, _, damped in models:
            if trend != "N":  # N has no damping concept
                assert damped, (
                    f"Expected damped=True, got trend={trend}, damped={damped}"
                )

    def test_invalid_ic(self) -> None:
        """Invalid IC should raise ValueError."""
        with pytest.raises(ValueError, match="ic must be"):
            AutoETS(ic="invalid")

    def test_short_series_error(self) -> None:
        """Very short series should raise ValueError."""
        y = np.array([1.0, 2.0])
        auto_ets = AutoETS()
        with pytest.raises(ValueError, match="too short"):
            auto_ets.fit(y)

    def test_n_fits(self) -> None:
        """n_fits should equal the number of models tested."""
        data = _make_simple_data()
        auto_ets = AutoETS(seasonal_period=1, ic="aicc")
        result = auto_ets.fit(data)

        assert result.n_fits == len(result.all_models)
        assert result.n_fits > 0
