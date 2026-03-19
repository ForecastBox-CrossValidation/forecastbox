"""Tests for the forecast encompassing test.

References
----------
Harvey, D., Leybourne, S. & Newbold, P. (1998). "Tests for Forecast
    Encompassing."
"""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.evaluation.encompassing import encompassing_test


class TestEncompassing:
    """Tests for the forecast encompassing test."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    def test_f1_encompasses_f2(self, rng: np.random.Generator) -> None:
        """f2 is f1 plus extra noise -> f1 encompasses f2 (lambda ~ 1)."""
        T = 100
        actual = rng.normal(100, 10, size=T)
        forecast1 = actual + rng.normal(0, 2, size=T)
        # f2 = f1 + extra noise, so f1 contains all of f2's information
        forecast2 = forecast1 + rng.normal(0, 10, size=T)

        result = encompassing_test(actual, forecast1, forecast2)
        assert result.lambda_hat > 0.7
        assert result.f1_encompasses_f2
        assert not result.f2_encompasses_f1

    def test_f2_encompasses_f1(self, rng: np.random.Generator) -> None:
        """f1 is f2 plus extra noise -> f2 encompasses f1 (lambda ~ 0)."""
        T = 100
        actual = rng.normal(100, 10, size=T)
        forecast2 = actual + rng.normal(0, 2, size=T)
        # f1 = f2 + extra noise, so f2 contains all of f1's information
        forecast1 = forecast2 + rng.normal(0, 10, size=T)

        result = encompassing_test(actual, forecast1, forecast2)
        assert result.lambda_hat < 0.3
        assert result.f2_encompasses_f1
        assert not result.f1_encompasses_f2

    def test_neither_encompasses(self, rng: np.random.Generator) -> None:
        """Both have unique information -> 0 < lambda < 1."""
        T = 300
        actual = rng.normal(100, 10, size=T)

        # Each forecast captures different aspects
        signal1 = rng.normal(0, 5, size=T)
        signal2 = rng.normal(0, 5, size=T)
        forecast1 = actual + signal1
        forecast2 = actual + signal2

        # Add the signals back to actual so forecasts have unique info
        actual_combined = actual + 0.5 * signal1 + 0.5 * signal2 + rng.normal(0, 0.1, size=T)

        result = encompassing_test(actual_combined, forecast1, forecast2)
        # lambda should be between 0 and 1
        assert 0.1 < result.lambda_hat < 0.9
        assert result.neither_encompasses

    def test_both_encompassed(self, rng: np.random.Generator) -> None:
        """f1 = f2 -> test is inconclusive (identical forecasts)."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast = actual + rng.normal(0, 3, size=T)

        result = encompassing_test(actual, forecast, forecast)
        # Forecasts are identical, lambda is arbitrary
        assert result.pvalue == 1.0 or result.lambda_se == np.inf

        # Summary works
        summary = result.summary()
        assert "Encompassing" in summary
