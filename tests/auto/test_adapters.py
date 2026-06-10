"""Tests for chronobox model adapters.

These tests are skipped entirely if chronobox is not installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.core.forecast import Forecast

chronobox = pytest.importorskip("chronobox")

# Imported after the importorskip: the module unconditionally imports chronobox
# at module load (it is omitted from coverage precisely for this reason).
from forecastbox.auto._adapters import (  # noqa: E402
    ARIMAAdapter,
    ETSAdapter,
    ThetaAdapter,
    VARAdapter,
)


def _make_series(n: int = 80) -> pd.Series:
    """Generate a deterministic random-walk series with a DatetimeIndex."""
    rng = np.random.RandomState(42)
    values = 100.0 + np.cumsum(rng.normal(0, 1, n))
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(values, index=index, name="y")


def _make_seasonal_series(n: int = 120, m: int = 12) -> pd.Series:
    """Generate a deterministic series with additive seasonality and trend."""
    rng = np.random.RandomState(123)
    t = np.arange(n, dtype=np.float64)
    trend = 100.0 + 0.5 * t
    seasonal = 10.0 * np.sin(2 * np.pi * t / m)
    values = trend + seasonal + rng.normal(0, 1, n)
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(values, index=index, name="y")


def _make_var_df(n: int = 120, k: int = 2) -> pd.DataFrame:
    """Generate a deterministic multivariate VAR(2) DataFrame."""
    rng = np.random.RandomState(7)
    y = np.zeros((n, k))
    for t in range(2, n):
        y[t, 0] = (
            0.5 * y[t - 1, 0]
            + 0.3 * y[t - 2, 0]
            + 0.2 * y[t - 1, 1]
            + rng.normal(0, 1)
        )
        y[t, 1] = (
            0.1 * y[t - 1, 0]
            + 0.4 * y[t - 1, 1]
            + 0.2 * y[t - 2, 1]
            + rng.normal(0, 1)
        )
    index = pd.date_range("2000-01", periods=n, freq="MS")
    columns = [f"y{i + 1}" for i in range(k)]
    return pd.DataFrame(y, index=index, columns=columns)


class TestARIMAAdapter:
    """Tests for ARIMAAdapter."""

    def test_arima_fit_returns_self(self) -> None:
        adapter = ARIMAAdapter(order=(1, 1, 1))
        fitted = adapter.fit(_make_series())
        assert fitted is adapter
        assert adapter._fitted is True

    def test_arima_forecast_shape_and_type(self) -> None:
        adapter = ARIMAAdapter(order=(1, 1, 1)).fit(_make_series())
        fc = adapter.forecast(h=6)
        assert isinstance(fc, Forecast)
        assert len(fc) == 6
        assert len(fc.point) == 6
        assert fc.point.dtype == np.float64
        assert not np.any(np.isnan(fc.point))

    def test_arima_model_name(self) -> None:
        adapter = ARIMAAdapter(order=(1, 1, 1)).fit(_make_series())
        fc = adapter.forecast(h=6)
        assert fc.model_name == f"ARIMA{(1, 1, 1)}"
        assert fc.horizon == 6

    def test_arima_seasonal_order(self) -> None:
        adapter = ARIMAAdapter(
            order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)
        ).fit(_make_seasonal_series())
        fc = adapter.forecast(12)
        assert isinstance(fc, Forecast)
        assert len(fc) == 12

    def test_arima_forecast_before_fit_raises(self) -> None:
        adapter = ARIMAAdapter(order=(1, 1, 1))
        with pytest.raises(RuntimeError, match="must be fit"):
            adapter.forecast(6)


class TestETSAdapter:
    """Tests for ETSAdapter."""

    def test_ets_forecast_shape_and_type(self) -> None:
        adapter = ETSAdapter(
            error="A", trend="A", seasonal="N", seasonal_period=1
        ).fit(_make_series())
        fc = adapter.forecast(6)
        assert isinstance(fc, Forecast)
        assert len(fc) == 6
        assert fc.point.dtype == np.float64
        assert not np.any(np.isnan(fc.point))

    def test_ets_model_name(self) -> None:
        adapter = ETSAdapter(
            error="A", trend="A", seasonal="N", seasonal_period=1
        ).fit(_make_series())
        fc = adapter.forecast(6)
        assert fc.model_name == "ETS(A,A,N)"

    def test_ets_seasonal(self) -> None:
        adapter = ETSAdapter(
            error="A", trend="A", seasonal="A", seasonal_period=12
        ).fit(_make_seasonal_series())
        fc = adapter.forecast(12)
        assert isinstance(fc, Forecast)
        assert len(fc) == 12

    def test_ets_forecast_before_fit_raises(self) -> None:
        adapter = ETSAdapter(error="A", trend="A", seasonal="N", seasonal_period=1)
        with pytest.raises(RuntimeError, match="must be fit"):
            adapter.forecast(6)


class TestVARAdapter:
    """Tests for VARAdapter."""

    def test_var_forecast_shape_and_type(self) -> None:
        adapter = VARAdapter(maxlags=2).fit(_make_var_df())
        fc = adapter.forecast(6)
        assert isinstance(fc, Forecast)
        point = np.asarray(fc.point)
        assert point.ndim == 2
        assert point.shape == (6, 2)
        assert point.dtype == np.float64
        assert len(fc) == 6

    def test_var_model_name(self) -> None:
        adapter = VARAdapter(maxlags=2).fit(_make_var_df())
        fc = adapter.forecast(6)
        assert fc.model_name == "VAR(2)"

    def test_var_forecast_before_fit_raises(self) -> None:
        adapter = VARAdapter(maxlags=2)
        with pytest.raises(RuntimeError, match="must be fit"):
            adapter.forecast(6)


class TestThetaAdapter:
    """Tests for ThetaAdapter."""

    def test_theta_forecast_shape_and_type(self) -> None:
        adapter = ThetaAdapter(theta=2.0).fit(_make_series())
        fc = adapter.forecast(6)
        assert isinstance(fc, Forecast)
        assert len(fc) == 6
        assert fc.point.dtype == np.float64
        assert not np.any(np.isnan(fc.point))

    def test_theta_model_name(self) -> None:
        adapter = ThetaAdapter(theta=2.0).fit(_make_series())
        fc = adapter.forecast(6)
        assert fc.model_name == "Theta"

    def test_theta_forecast_before_fit_raises(self) -> None:
        adapter = ThetaAdapter(theta=2.0)
        with pytest.raises(RuntimeError, match="must be fit"):
            adapter.forecast(6)


@pytest.mark.parametrize(
    "adapter_cls",
    [ARIMAAdapter, ETSAdapter, VARAdapter, ThetaAdapter],
)
def test_adapter_raises_importerror_when_chronobox_missing(
    adapter_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Instantiating any adapter without chronobox should raise ImportError."""
    monkeypatch.setattr("forecastbox.auto._adapters.HAS_CHRONOBOX", False)
    with pytest.raises(ImportError, match="chronobox is required"):
        adapter_cls()
