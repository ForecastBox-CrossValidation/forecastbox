"""Tests for advanced forecast metrics.

References
----------
Gneiting, T. & Raftery, A.E. (2007). "Strictly Proper Scoring Rules."
"""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.metrics.advanced_metrics import (
    crps,
    crps_gaussian,
    mfe,
    smape,
    theil_u1,
    theil_u2,
)


class TestAdvancedMetrics:
    """Tests for advanced metrics."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    def test_mfe_bias(self, rng: np.random.Generator) -> None:
        """MFE positive for forecast with negative bias (under-forecast)."""
        n = 200
        actual = rng.normal(100, 10, size=n)
        predicted = actual - 3.0  # systematically under-forecasting

        result = mfe(actual, predicted)
        assert result > 0  # actual > predicted on average
        assert abs(result - 3.0) < 1.0  # close to the true bias

    def test_theil_u1_range(self, rng: np.random.Generator) -> None:
        """0 <= Theil U1 <= 1."""
        n = 100
        actual = rng.normal(100, 10, size=n)
        predicted = actual + rng.normal(0, 5, size=n)

        result = theil_u1(actual, predicted)
        assert 0.0 <= result <= 1.0

    def test_theil_u2_naive(self, rng: np.random.Generator) -> None:
        """Theil U2 < 1 indicates better than naive."""
        n = 200
        actual = np.cumsum(rng.normal(0, 1, size=n)) + 100  # random walk + drift
        # Good forecast: actual + small noise
        predicted = actual + rng.normal(0, 0.1, size=n)

        result = theil_u2(actual, predicted)
        assert result < 1.0

    def test_theil_u2_perfect(self, rng: np.random.Generator) -> None:
        """Theil U2 = 0 for perfect forecast."""
        actual = np.array([100.0, 102.0, 105.0, 103.0, 107.0])
        predicted = actual.copy()

        result = theil_u2(actual, predicted)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_smape_symmetric(self, rng: np.random.Generator) -> None:
        """sMAPE is symmetric in actual and predicted."""
        n = 100
        actual = rng.uniform(50, 150, size=n)
        predicted = actual + rng.normal(0, 10, size=n)

        s1 = smape(actual, predicted)
        s2 = smape(predicted, actual)

        assert s1 == pytest.approx(s2, rel=1e-10)

    def test_smape_range(self, rng: np.random.Generator) -> None:
        """0 <= sMAPE <= 200."""
        n = 100
        actual = rng.uniform(50, 150, size=n)
        predicted = actual + rng.normal(0, 10, size=n)

        result = smape(actual, predicted)
        assert 0.0 <= result <= 200.0

    def test_crps_perfect(self) -> None:
        """CRPS = 0 for ensemble concentrated on actual value."""
        actual = np.array([100.0, 200.0, 150.0])
        # Ensemble: all draws equal to actual
        ensemble = np.column_stack([actual] * 50)  # (3, 50)

        result = crps(actual, ensemble)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_crps_gaussian(self, rng: np.random.Generator) -> None:
        """crps_gaussian vs numerical crps should agree (tol=0.01)."""
        n = 50
        mu = rng.normal(100, 10, size=n)
        sigma = np.full(n, 5.0)
        actual = mu + rng.normal(0, 5, size=n)

        # Analytical CRPS
        crps_analytical = crps_gaussian(actual, mu, sigma)

        # Numerical CRPS via ensemble
        n_draws = 10000
        ensemble = np.zeros((n, n_draws))
        for t in range(n):
            ensemble[t] = rng.normal(mu[t], sigma[t], size=n_draws)

        crps_numerical = crps(actual, ensemble)

        assert crps_analytical == pytest.approx(crps_numerical, abs=0.05)

    def test_crps_wider_worse(self, rng: np.random.Generator) -> None:
        """Wider distribution -> higher (worse) CRPS."""
        n = 50
        actual = np.full(n, 100.0)
        mu = np.full(n, 100.0)  # centered on actual

        crps_narrow = crps_gaussian(actual, mu, np.full(n, 1.0))
        crps_wide = crps_gaussian(actual, mu, np.full(n, 10.0))

        assert crps_wide > crps_narrow
