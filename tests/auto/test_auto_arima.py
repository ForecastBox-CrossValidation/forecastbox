"""Tests for AutoARIMA model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.auto._stepwise import _determine_d, _determine_seasonal_d
from forecastbox.auto.arima import AutoARIMA, _compute_ic
from forecastbox.core.forecast import Forecast


def _make_airline_data() -> pd.Series:
    """Generate synthetic airline-like data with trend and multiplicative seasonality."""
    np.random.seed(42)
    n = 144  # 12 years of monthly data
    t = np.arange(n)

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


def _make_stationary_data() -> pd.Series:
    """Generate a stationary time series (AR(1) with phi=0.5)."""
    np.random.seed(123)
    n = 200
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + np.random.normal(0, 1)
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="stationary")


def _make_trend_data() -> pd.Series:
    """Generate a non-stationary series with linear trend."""
    np.random.seed(456)
    n = 200
    t = np.arange(n, dtype=np.float64)
    y = 50.0 + 0.5 * t + np.cumsum(np.random.normal(0, 1, n))
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="trend")


def _make_seasonal_data() -> pd.Series:
    """Generate a series with clear additive seasonality."""
    np.random.seed(789)
    n = 120  # 10 years
    t = np.arange(n, dtype=np.float64)
    seasonal_pattern = 10.0 * np.sin(2 * np.pi * t / 12)
    y = 100.0 + seasonal_pattern + np.random.normal(0, 1, n)
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="seasonal")


class TestAutoARIMA:
    """Tests for AutoARIMA model selection."""

    def test_airline_order(self) -> None:
        """AutoARIMA on airline-like data should select near ARIMA(0,1,1)(0,1,1)[12]."""
        data = _make_airline_data()
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True, ic="aicc")
        result = auto.fit(data)

        p, d, q = result.order
        _P, _D, _Q, m = result.seasonal_order

        assert d >= 1, f"Expected d >= 1, got d={d}"
        assert m == 12
        assert 0 <= p <= 3
        assert 0 <= q <= 3
        assert 0 <= _P <= 2
        assert 0 <= _Q <= 2

    def test_stepwise_fewer_fits(self) -> None:
        """Stepwise search should fit fewer models than grid search."""
        data = _make_airline_data()

        auto_sw = AutoARIMA(seasonal=True, m=12, stepwise=True, ic="aicc")
        result_sw = auto_sw.fit(data)

        auto_grid = AutoARIMA(
            seasonal=True, m=12, stepwise=False, ic="aicc",
            max_p=3, max_q=3, max_P=1, max_Q=1, max_order=5,
        )
        result_grid = auto_grid.fit(data)

        assert result_sw.n_fits < result_grid.n_fits, (
            f"Stepwise ({result_sw.n_fits}) should fit fewer models "
            f"than grid ({result_grid.n_fits})"
        )
        assert result_sw.n_fits < 30, (
            f"Stepwise should fit < 30 models, got {result_sw.n_fits}"
        )

    def test_d_determination(self) -> None:
        """Series with trend should get d >= 1; stationary series should get d = 0."""
        stationary = _make_stationary_data()
        trend = _make_trend_data()

        d_stationary = _determine_d(np.asarray(stationary))
        d_trend = _determine_d(np.asarray(trend))

        assert d_stationary == 0, f"Stationary series: expected d=0, got d={d_stationary}"
        assert d_trend >= 1, f"Trend series: expected d>=1, got d={d_trend}"

    def test_D_determination(self) -> None:
        """Series with seasonality should get D=1; non-seasonal should get D=0."""
        stationary = _make_stationary_data()

        d_nonseasonal = _determine_seasonal_d(np.asarray(stationary), m=12)
        assert d_nonseasonal == 0, f"Non-seasonal: expected D=0, got D={d_nonseasonal}"

    def test_non_seasonal(self) -> None:
        """AutoARIMA(seasonal=False) should not include P, Q."""
        data = _make_stationary_data()
        auto = AutoARIMA(seasonal=False, m=1, stepwise=True)
        result = auto.fit(data)

        big_p, big_d, big_q, m = result.seasonal_order
        assert big_p == 0, f"Non-seasonal: expected P=0, got P={big_p}"
        assert big_q == 0, f"Non-seasonal: expected Q=0, got Q={big_q}"
        assert big_d == 0, f"Non-seasonal: expected D=0, got D={big_d}"

    def test_forecast_shape(self) -> None:
        """forecast(12) should return Forecast with len == 12."""
        data = _make_airline_data()
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
        result = auto.fit(data)
        fc = result.forecast(12)

        assert isinstance(fc, Forecast)
        assert len(fc) == 12
        assert len(fc.point) == 12
        assert not np.any(np.isnan(fc.point)), "Forecast contains NaN values"

    def test_all_models_dataframe(self) -> None:
        """result.all_models should be a DataFrame with expected columns."""
        data = _make_airline_data()
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
        result = auto.fit(data)

        assert isinstance(result.all_models, pd.DataFrame)
        assert "order" in result.all_models.columns
        assert "seasonal" in result.all_models.columns
        assert "ic_value" in result.all_models.columns
        assert len(result.all_models) > 0

    def test_trace_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """trace=True should produce output to stdout."""
        data = _make_stationary_data()
        auto = AutoARIMA(seasonal=False, m=1, stepwise=True, trace=True)
        auto.fit(data)

        captured = capsys.readouterr()
        assert len(captured.out) > 0, "trace=True should produce output"
        assert "ARIMA" in captured.out

    def test_summary(self) -> None:
        """summary() should return a non-empty string with model info."""
        data = _make_stationary_data()
        auto = AutoARIMA(seasonal=False, m=1, stepwise=True)
        result = auto.fit(data)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "AutoARIMA" in summary
        assert "ARIMA" in summary
        assert str(result.n_fits) in summary

    def test_compute_ic(self) -> None:
        """Test information criterion computations."""
        aic = _compute_ic(log_likelihood=-100.0, n_params=3, n_obs=100, ic="aic")
        assert aic == pytest.approx(206.0)

        bic = _compute_ic(log_likelihood=-100.0, n_params=3, n_obs=100, ic="bic")
        expected_bic = 200.0 + 3 * np.log(100)
        assert bic == pytest.approx(expected_bic)

        aicc = _compute_ic(log_likelihood=-100.0, n_params=3, n_obs=100, ic="aicc")
        correction = (2 * 3 * 4) / (100 - 3 - 1)
        assert aicc == pytest.approx(206.0 + correction)

    def test_invalid_ic(self) -> None:
        """Invalid IC name should raise ValueError."""
        with pytest.raises(ValueError, match="ic must be"):
            AutoARIMA(ic="invalid")

    def test_short_series_error(self) -> None:
        """Very short series should raise ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        auto = AutoARIMA()
        with pytest.raises(ValueError, match="too short"):
            auto.fit(y)
