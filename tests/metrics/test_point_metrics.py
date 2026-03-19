"""Tests for point forecast metrics."""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.metrics.point_metrics import mae, mape, mase, me, rmse


class TestPointMetrics:
    """Tests for point forecast metrics."""

    def test_mae_known(self) -> None:
        """actual=[3,0,2], pred=[2.5,0.5,2] -> MAE = 0.5."""
        actual = np.array([3.0, 0.0, 2.0])
        pred = np.array([2.5, 0.5, 2.0])
        # |3-2.5| + |0-0.5| + |2-2| = 0.5 + 0.5 + 0 = 1.0
        # MAE = 1.0/3
        result = mae(actual, pred)
        assert result == pytest.approx(1.0 / 3.0)

    def test_rmse_known(self) -> None:
        """actual=[3,0,2], pred=[2.5,0.5,2] -> RMSE = sqrt(0.5/3)."""
        actual = np.array([3.0, 0.0, 2.0])
        pred = np.array([2.5, 0.5, 2.0])
        # (0.25 + 0.25 + 0) / 3 = 0.5/3
        result = rmse(actual, pred)
        assert result == pytest.approx(np.sqrt(0.5 / 3))

    def test_mape_known(self) -> None:
        """actual=[100,200,300], pred=[110,190,310] -> MAPE ~ 5.56%."""
        actual = np.array([100.0, 200.0, 300.0])
        pred = np.array([110.0, 190.0, 310.0])
        # |10/100| + |10/200| + |10/300| = 0.10 + 0.05 + 0.0333 = 0.1833
        # MAPE = 100 * 0.1833 / 3 = 6.11%
        result = mape(actual, pred)
        expected = 100.0 * (0.10 + 0.05 + 10.0 / 300.0) / 3.0
        assert result == pytest.approx(expected)

    def test_mase_known(self) -> None:
        """MASE < 1 indicates better than naive."""
        train = np.array([100.0, 102.0, 101.0, 103.0, 102.0])
        actual = np.array([104.0, 105.0])
        pred = np.array([103.5, 104.8])
        result = mase(actual, pred, train)
        assert result < 1.0  # better than naive

    def test_mape_zero_warning(self) -> None:
        """actual containing 0 emits warning and returns inf."""
        actual = np.array([0.0, 1.0, 2.0])
        pred = np.array([0.5, 1.5, 2.5])
        with pytest.warns(UserWarning, match="zeros"):
            result = mape(actual, pred)
        assert result == np.inf

    def test_perfect_forecast(self) -> None:
        """All errors = 0 for perfect forecast."""
        actual = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        assert mae(actual, pred) == 0.0
        assert rmse(actual, pred) == 0.0
        assert me(actual, pred) == 0.0

    def test_mae_symmetric(self) -> None:
        """MAE(a,b) == MAE(b,a) — MAE is symmetric."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.5, 3.5])
        assert mae(a, b) == pytest.approx(mae(b, a))

    def test_me_bias(self) -> None:
        """ME detects systematic bias."""
        actual = np.array([100.0, 200.0, 300.0])
        pred = np.array([90.0, 190.0, 290.0])  # systematic under-prediction
        result = me(actual, pred)
        assert result > 0  # positive ME means under-prediction

    def test_me_unbiased(self) -> None:
        """ME = 0 for unbiased forecast."""
        actual = np.array([100.0, 200.0, 300.0])
        pred = np.array([110.0, 190.0, 300.0])  # errors cancel out
        result = me(actual, pred)
        assert result == pytest.approx(0.0)

    def test_invalid_inputs_nan(self) -> None:
        """NaN values raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            mae(np.array([1.0, np.nan]), np.array([1.0, 2.0]))

    def test_invalid_inputs_different_length(self) -> None:
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            mae(np.array([1.0, 2.0]), np.array([1.0]))

    def test_accepts_lists(self) -> None:
        """Functions accept plain lists."""
        result = mae([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
        assert result == pytest.approx(0.5)

    def test_accepts_series(self) -> None:
        """Functions accept pandas Series."""
        import pandas as pd

        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([1.5, 2.5, 3.5])
        result = mae(a, b)
        assert result == pytest.approx(0.5)
