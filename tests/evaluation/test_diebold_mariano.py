"""Tests for the Diebold-Mariano test.

References
----------
Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
Harvey, D., Leybourne, S. & Newbold, P. (1997). "Testing the equality of
    prediction mean squared errors."
"""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.evaluation.diebold_mariano import diebold_mariano


class TestDieboldMariano:
    """Tests for the Diebold-Mariano test."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    def test_dm_rejects_dominant(self, rng: np.random.Generator) -> None:
        """Forecast 1 = actual (perfect), forecast 2 = actual + noise.
        DM must reject H0 with p < 0.05."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast1 = actual.copy()  # perfect forecast
        forecast2 = actual + rng.normal(0, 5, size=T)  # noisy forecast

        result = diebold_mariano(actual, forecast1, forecast2, h=1, loss="mse")
        assert result.pvalue < 0.05
        assert result.mean_loss_diff < 0  # forecast 1 is better

    def test_dm_not_rejects_equal(self, rng: np.random.Generator) -> None:
        """Two forecasts with same MSE. DM should not reject (p > 0.10)."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        noise1 = rng.normal(0, 3, size=T)
        noise2 = rng.normal(0, 3, size=T)
        forecast1 = actual + noise1
        forecast2 = actual + noise2

        result = diebold_mariano(actual, forecast1, forecast2, h=1, loss="mse")
        assert result.pvalue > 0.10

    def test_dm_statistic_sign(self, rng: np.random.Generator) -> None:
        """If forecast 1 is better, DM < 0 (d_bar < 0 with loss=mse)."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast1 = actual + rng.normal(0, 1, size=T)  # small noise
        forecast2 = actual + rng.normal(0, 10, size=T)  # large noise

        result = diebold_mariano(actual, forecast1, forecast2, h=1, loss="mse")
        assert result.statistic < 0
        assert result.mean_loss_diff < 0

    def test_dm_hln_correction(self, rng: np.random.Generator) -> None:
        """HLN correction reduces statistic in magnitude."""
        T = 50
        actual = rng.normal(100, 10, size=T)
        forecast1 = actual + rng.normal(0, 1, size=T)
        forecast2 = actual + rng.normal(0, 5, size=T)

        result_hln = diebold_mariano(
            actual, forecast1, forecast2, h=1, loss="mse", hln_correction=True
        )
        result_no_hln = diebold_mariano(
            actual, forecast1, forecast2, h=1, loss="mse", hln_correction=False
        )
        assert abs(result_hln.statistic) <= abs(result_no_hln.statistic)

    def test_dm_one_sided(self, rng: np.random.Generator) -> None:
        """one_sided=True returns p-value smaller than two-sided when forecast 1 better."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast1 = actual + rng.normal(0, 1, size=T)
        forecast2 = actual + rng.normal(0, 5, size=T)

        result_two = diebold_mariano(
            actual, forecast1, forecast2, h=1, loss="mse", one_sided=False
        )
        result_one = diebold_mariano(
            actual, forecast1, forecast2, h=1, loss="mse", one_sided=True
        )
        # One-sided p-value should be approximately half the two-sided
        assert result_one.pvalue < result_two.pvalue

    def test_dm_h_step(self, rng: np.random.Generator) -> None:
        """For h > 1, HAC uses truncation lag = h-1."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast1 = actual + rng.normal(0, 2, size=T)
        forecast2 = actual + rng.normal(0, 5, size=T)

        result_h1 = diebold_mariano(actual, forecast1, forecast2, h=1, loss="mse")
        result_h4 = diebold_mariano(actual, forecast1, forecast2, h=4, loss="mse")

        # Both should reject, but statistics may differ due to HAC correction
        assert result_h1.pvalue < 0.05
        assert result_h4.h == 4
        # The statistic values should differ because HAC lag differs
        assert result_h1.statistic != pytest.approx(result_h4.statistic, abs=0.01)

    def test_dm_mae_loss(self, rng: np.random.Generator) -> None:
        """loss='mae' is functional and produces valid results."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast1 = actual + rng.normal(0, 1, size=T)
        forecast2 = actual + rng.normal(0, 5, size=T)

        result = diebold_mariano(actual, forecast1, forecast2, h=1, loss="mae")
        assert result.loss == "mae"
        assert result.pvalue < 0.05
        assert result.mean_loss_diff < 0  # forecast 1 is better
        assert len(result.loss_differential) == T
        # Verify conclusion method works
        conclusion = result.conclusion(alpha=0.05)
        assert isinstance(conclusion, str)
        assert len(conclusion) > 0
