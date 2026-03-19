"""Tests for the Giacomini-White test.

References
----------
Giacomini, R. & White, H. (2006). "Tests of Conditional Predictive Ability."
    Econometrica, 74(6), 1545-1578.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.evaluation.diebold_mariano import diebold_mariano
from forecastbox.evaluation.giacomini_white import giacomini_white


class TestGiacominiWhite:
    """Tests for the Giacomini-White test."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    def test_gw_rejects_conditional(self, rng: np.random.Generator) -> None:
        """In regime 1 f1 is better, in regime 2 f2 is better.
        DM should not reject (average zero), but GW rejects (conditional differs)."""
        T = 400

        actual = rng.normal(100, 10, size=T)

        # Create regime-switching forecasts
        forecast1 = actual.copy()
        forecast2 = actual.copy()

        # First half: f1 much better than f2
        forecast1[:T // 2] += rng.normal(0, 1, size=T // 2)
        forecast2[:T // 2] += rng.normal(0, 10, size=T // 2)

        # Second half: f2 much better than f1
        forecast1[T // 2:] += rng.normal(0, 10, size=T - T // 2)
        forecast2[T // 2:] += rng.normal(0, 1, size=T - T // 2)

        # DM should not reject (average difference ~ 0)
        dm_result = diebold_mariano(actual, forecast1, forecast2, h=1, loss="mse")
        assert dm_result.pvalue > 0.05

        # GW should reject (conditional ability differs)
        # Use a regime indicator as instrument
        regime = np.zeros(T)
        regime[T // 2:] = 1.0
        instruments = np.column_stack([np.ones(T), regime])

        gw_result = giacomini_white(
            actual, forecast1, forecast2, h=1, instruments=instruments, loss="mse"
        )
        assert gw_result.pvalue < 0.05

    def test_gw_constant_instrument(self, rng: np.random.Generator) -> None:
        """With h_t = [1], GW is approximately equivalent to DM squared."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast1 = actual + rng.normal(0, 2, size=T)
        forecast2 = actual + rng.normal(0, 5, size=T)

        # GW with only constant instrument
        instruments = np.ones((T, 1))
        gw_result = giacomini_white(
            actual, forecast1, forecast2, h=1, instruments=instruments, loss="mse"
        )

        # DM test
        dm_result = diebold_mariano(
            actual, forecast1, forecast2, h=1, loss="mse", hln_correction=False
        )

        # GW statistic should be approximately DM^2
        assert gw_result.statistic == pytest.approx(
            dm_result.statistic**2, rel=0.3
        )
        assert gw_result.df == 1

    def test_gw_chi_squared(self, rng: np.random.Generator) -> None:
        """Under H0, GW statistic ~ chi^2(q). Check df is correct."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        noise = rng.normal(0, 3, size=T)
        forecast1 = actual + noise
        forecast2 = actual + noise  # identical forecasts

        # Default instruments [1, d_{t-1}]
        gw_result = giacomini_white(
            actual, forecast1, forecast2, h=1, loss="mse"
        )

        assert gw_result.df == 2  # [1, d_{t-1}] has 2 instruments
        assert gw_result.statistic >= 0.0
        # Under H0, should not reject
        assert gw_result.pvalue > 0.01
        # Conclusion method works
        assert isinstance(gw_result.conclusion(), str)
