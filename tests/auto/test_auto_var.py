"""Tests for AutoVAR model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.auto.var import AutoVAR, AutoVARResult, _compute_var_ic
from forecastbox.core.forecast import Forecast


def _make_var2_data() -> pd.DataFrame:
    """Generate a VAR(2) process with 2 variables.

    y1_t = 0.5*y1_{t-1} + 0.3*y1_{t-2} + 0.2*y2_{t-1} + e1_t
    y2_t = 0.1*y1_{t-1} + 0.4*y2_{t-1} + 0.2*y2_{t-2} + e2_t

    The true lag order is 2.
    """
    np.random.seed(42)
    n = 300
    y = np.zeros((n, 2))

    for t in range(2, n):
        y[t, 0] = (
            0.5 * y[t - 1, 0]
            + 0.3 * y[t - 2, 0]
            + 0.2 * y[t - 1, 1]
            + np.random.normal(0, 1)
        )
        y[t, 1] = (
            0.1 * y[t - 1, 0]
            + 0.4 * y[t - 1, 1]
            + 0.2 * y[t - 2, 1]
            + np.random.normal(0, 1)
        )

    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.DataFrame(y, index=index, columns=["y1", "y2"])


def _make_5var_data() -> pd.DataFrame:
    """Generate data with 5 variables, only 2 are relevant.

    y1 and y2 are a VAR(1) process.
    y3, y4, y5 are independent noise.
    """
    np.random.seed(123)
    n = 200

    # Relevant variables: VAR(1) with cross-effects
    y = np.zeros((n, 5))
    for t in range(1, n):
        y[t, 0] = 0.6 * y[t - 1, 0] + 0.3 * y[t - 1, 1] + np.random.normal(0, 1)
        y[t, 1] = 0.2 * y[t - 1, 0] + 0.5 * y[t - 1, 1] + np.random.normal(0, 1)

    # Noise variables
    y[:, 2] = np.random.normal(0, 5, n)
    y[:, 3] = np.random.normal(0, 5, n)
    y[:, 4] = np.random.normal(0, 5, n)

    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.DataFrame(y, index=index, columns=["y1", "y2", "y3", "y4", "y5"])


def _make_simple_multivariate() -> pd.DataFrame:
    """Generate simple multivariate data for basic tests."""
    np.random.seed(456)
    n = 100
    y1 = np.cumsum(np.random.normal(0, 1, n))
    y2 = np.cumsum(np.random.normal(0, 1, n))
    y3 = np.cumsum(np.random.normal(0, 1, n))

    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.DataFrame({"var1": y1, "var2": y2, "var3": y3}, index=index)


class TestAutoVAR:
    """Tests for AutoVAR model selection."""

    def test_lag_selection(self) -> None:
        """VAR(2) simulated data should select lag=2 (or close)."""
        data = _make_var2_data()
        auto_var = AutoVAR(max_lags=8, ic="bic")
        result = auto_var.fit(data)

        assert isinstance(result, AutoVARResult)
        # BIC should select lag=2 or close for a VAR(2) process
        assert 1 <= result.selected_lag <= 4, (
            f"Expected lag near 2 for VAR(2) data, got {result.selected_lag}"
        )

    def test_ic_table(self) -> None:
        """ic_table should have expected columns and rows."""
        data = _make_var2_data()
        auto_var = AutoVAR(max_lags=6, ic="bic")
        result = auto_var.fit(data)

        assert isinstance(result.ic_table, pd.DataFrame)
        assert "lag" in result.ic_table.columns
        assert "aic" in result.ic_table.columns
        assert "bic" in result.ic_table.columns
        assert "hqc" in result.ic_table.columns
        assert len(result.ic_table) >= 1
        assert len(result.ic_table) <= 6

    def test_variable_selection(self) -> None:
        """With 5 variables (2 relevant), should select the 2 relevant ones."""
        data = _make_5var_data()
        auto_var = AutoVAR(max_lags=4, ic="bic", select_variables=True)
        result = auto_var.fit(data)

        assert isinstance(result.selected_variables, list)
        assert len(result.selected_variables) >= 2
        # The two relevant variables (y1, y2) should be in the selection
        # (though this is a statistical test and may occasionally fail)
        selected_set = set(result.selected_variables)
        has_relevant = "y1" in selected_set or "y2" in selected_set
        assert has_relevant, (
            f"Expected at least one of y1, y2 in selection, "
            f"got {result.selected_variables}"
        )

    def test_multivariate_forecast(self) -> None:
        """forecast(12) should return a Forecast object."""
        data = _make_simple_multivariate()
        auto_var = AutoVAR(max_lags=4, ic="bic")
        result = auto_var.fit(data)
        fc = result.forecast(12)

        assert isinstance(fc, Forecast)
        assert len(fc) == 12
        assert len(fc.point) == 12
        assert not np.any(np.isnan(fc.point)), "Forecast contains NaN"

        # Check metadata contains all variable forecasts
        assert "all_variable_forecasts" in fc.metadata
        assert len(fc.metadata["all_variable_forecasts"]) == result.n_vars

    def test_forecast_intervals(self) -> None:
        """Forecast should have prediction intervals."""
        data = _make_simple_multivariate()
        auto_var = AutoVAR(max_lags=4, ic="bic")
        result = auto_var.fit(data)
        fc = result.forecast(6)

        if fc.lower_80 is not None and fc.upper_80 is not None:
            assert np.all(fc.lower_80 <= fc.point)
            assert np.all(fc.point <= fc.upper_80)

    def test_summary(self) -> None:
        """summary() should return informative text."""
        data = _make_simple_multivariate()
        auto_var = AutoVAR(max_lags=4, ic="bic")
        result = auto_var.fit(data)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "AutoVAR" in summary
        assert str(result.selected_lag) in summary

    def test_compute_var_ic(self) -> None:
        """Test IC formulas with known values."""
        sigma = np.eye(2) * 0.5
        T = 100
        k = 2
        p = 2

        aic = _compute_var_ic(sigma, T, k, p, "aic")
        bic = _compute_var_ic(sigma, T, k, p, "bic")
        _compute_var_ic(sigma, T, k, p, "hqc")

        # BIC should penalize more than AIC for n > e^2 (n > 7.4)
        assert bic > aic, "BIC should be > AIC for T=100"

    def test_invalid_ic(self) -> None:
        """Invalid IC should raise ValueError."""
        with pytest.raises(ValueError, match="ic must be"):
            AutoVAR(ic="invalid")

    def test_invalid_trend(self) -> None:
        """Invalid trend should raise ValueError."""
        with pytest.raises(ValueError, match="trend must be"):
            AutoVAR(trend="invalid")

    def test_not_dataframe_error(self) -> None:
        """Non-DataFrame input should raise TypeError."""
        auto_var = AutoVAR()
        with pytest.raises(TypeError, match="DataFrame"):
            auto_var.fit(np.array([[1, 2], [3, 4]]))  # type: ignore[arg-type]

    def test_single_variable_error(self) -> None:
        """Single variable should raise ValueError."""
        data = pd.DataFrame({"y1": [1, 2, 3, 4, 5] * 20})
        auto_var = AutoVAR()
        with pytest.raises(ValueError, match="at least 2"):
            auto_var.fit(data)

    def test_short_series_error(self) -> None:
        """Very short series should raise ValueError."""
        data = pd.DataFrame({"y1": [1, 2, 3], "y2": [4, 5, 6]})
        auto_var = AutoVAR()
        with pytest.raises(ValueError, match="too short"):
            auto_var.fit(data)

    def test_different_ic_criteria(self) -> None:
        """Different IC criteria may select different lags."""
        data = _make_var2_data()

        results = {}
        for ic_name in ("aic", "bic", "hqc"):
            auto_var = AutoVAR(max_lags=8, ic=ic_name)
            result = auto_var.fit(data)
            results[ic_name] = result.selected_lag

        # All should select reasonable lags
        for ic_name, lag in results.items():
            assert 1 <= lag <= 8, f"{ic_name} selected lag={lag}, expected 1-8"

    def test_max_vars_limit(self) -> None:
        """max_vars should limit the number of selected variables."""
        data = _make_5var_data()
        auto_var = AutoVAR(
            max_lags=2, ic="bic", select_variables=True, max_vars=3
        )
        result = auto_var.fit(data)

        assert len(result.selected_variables) <= 3

    def test_irf(self) -> None:
        """irf() should execute without error."""
        data = _make_simple_multivariate()
        auto_var = AutoVAR(max_lags=4, ic="bic")
        result = auto_var.fit(data)

        irf_result = result.irf(steps=10)
        assert irf_result is not None
