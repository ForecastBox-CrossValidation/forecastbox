"""Tests for the Mincer-Zarnowitz regression.

References
----------
Mincer, J.A. & Zarnowitz, V. (1969). "The Evaluation of Economic Forecasts."
"""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.evaluation.mincer_zarnowitz import mincer_zarnowitz


class TestMincerZarnowitz:
    """Tests for the Mincer-Zarnowitz regression."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    def test_mz_unbiased(self, rng: np.random.Generator) -> None:
        """Forecast = actual + N(0,1). MZ should not reject efficiency."""
        T = 500
        actual = rng.normal(100, 10, size=T)
        forecast = actual + rng.normal(0, 1, size=T)

        result = mincer_zarnowitz(actual, forecast)
        # Should not reject H0: alpha=0, beta=1
        assert result.pvalue > 0.05
        assert result.is_efficient(alpha=0.05)

    def test_mz_biased(self, rng: np.random.Generator) -> None:
        """Forecast = actual + 5. MZ should detect alpha != 0."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast = actual + 5.0 + rng.normal(0, 0.5, size=T)

        result = mincer_zarnowitz(actual, forecast)
        # Should reject H0 (bias present)
        assert result.pvalue < 0.05
        assert not result.is_efficient(alpha=0.05)
        # Alpha should be significantly different from 0
        assert abs(result.alpha_tstat) > 2.0

    def test_mz_overreaction(self, rng: np.random.Generator) -> None:
        """Forecast = 2 * actual. MZ should detect beta != 1."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast = 2.0 * actual + rng.normal(0, 0.5, size=T)

        result = mincer_zarnowitz(actual, forecast)
        # Should reject H0 (overreaction)
        assert result.pvalue < 0.05
        # Beta should be significantly different from 1
        assert abs(result.beta_tstat) > 2.0

    def test_mz_r_squared(self, rng: np.random.Generator) -> None:
        """R^2 should be high for an accurate forecast."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecast = actual + rng.normal(0, 1, size=T)  # very accurate

        result = mincer_zarnowitz(actual, forecast)
        assert result.r_squared > 0.90

    def test_mz_alpha_zero_beta_one(self, rng: np.random.Generator) -> None:
        """For an efficient forecast, alpha ~ 0 and beta ~ 1."""
        T = 500
        actual = rng.normal(100, 10, size=T)
        forecast = actual + rng.normal(0, 0.5, size=T)

        result = mincer_zarnowitz(actual, forecast)
        assert abs(result.alpha) < 5.0  # alpha close to 0
        assert abs(result.beta - 1.0) < 0.1  # beta close to 1
        assert result.is_efficient(alpha=0.05)

        # Summary method works
        summary = result.summary()
        assert "Mincer-Zarnowitz" in summary
        assert "alpha" in summary
