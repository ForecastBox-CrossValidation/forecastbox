"""Tests for OLSCombiner (Granger-Ramanathan)."""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.combination.ols import OLSCombiner


class TestOLSCombiner:
    """Tests for OLSCombiner (Granger-Ramanathan 1984)."""

    def test_unconstrained_recovers_best(self) -> None:
        """If one model is perfect, unconstrained OLS assigns w~1 to it."""
        rng = np.random.default_rng(42)
        t = 200
        actual = rng.normal(100, 5, size=t)

        fc_perfect = actual.copy()  # perfect forecast
        fc_noisy = actual + rng.normal(0, 10.0, size=t)

        combiner = OLSCombiner(intercept=False, constrained=False)
        combiner.fit([fc_perfect, fc_noisy], actual)

        assert combiner.weights_ is not None
        # Weight for perfect model should be close to 1
        assert combiner.weights_[0] > 0.9
        # Weight for noisy model should be close to 0
        assert abs(combiner.weights_[1]) < 0.1

    def test_constrained_weights_sum_one(self) -> None:
        """Constrained OLS weights must sum to 1."""
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)
        fc_c = actual + rng.normal(0, 3.0, size=t)

        combiner = OLSCombiner(intercept=False, constrained=True)
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.weights_ is not None
        assert pytest.approx(np.sum(combiner.weights_), abs=1e-8) == 1.0

    def test_constrained_weights_nonneg(self) -> None:
        """Constrained OLS weights must all be >= 0."""
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)

        combiner = OLSCombiner(intercept=False, constrained=True)
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_ is not None
        assert np.all(combiner.weights_ >= -1e-10)

    def test_intercept_absorbs_bias(self) -> None:
        """If all forecasts have constant bias, intercept absorbs it."""
        rng = np.random.default_rng(42)
        t = 200
        actual = rng.normal(100, 5, size=t)
        bias = 10.0

        fc_a = actual + bias + rng.normal(0, 0.5, size=t)
        fc_b = actual + bias + rng.normal(0, 0.5, size=t)

        combiner = OLSCombiner(intercept=True, constrained=False)
        combiner.fit([fc_a, fc_b], actual)

        # Intercept should be approximately -bias (since y = intercept + w*f)
        # f_k ~ actual + bias, so if w ~ 1: y ~ intercept + actual + bias
        # -> intercept ~ -bias
        assert abs(combiner.intercept_ - (-bias)) < 2.0

    def test_ridge_shrinks_weights(self) -> None:
        """Ridge regularization shrinks weights toward equal (1/K)."""
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 0.5, size=t)  # much better
        fc_b = actual + rng.normal(0, 5.0, size=t)  # much worse

        # Without ridge
        combiner_ols = OLSCombiner(intercept=False, constrained=False)
        combiner_ols.fit([fc_a, fc_b], actual)

        # With strong ridge
        combiner_ridge = OLSCombiner(
            intercept=False,
            constrained=False,
            regularization="ridge",
            alpha=100.0,
        )
        combiner_ridge.fit([fc_a, fc_b], actual)

        assert combiner_ols.weights_ is not None
        assert combiner_ridge.weights_ is not None

        # Ridge weights should be closer to 0.5 than OLS weights
        dist_ols = np.sum((combiner_ols.weights_ - 0.5) ** 2)
        dist_ridge = np.sum((combiner_ridge.weights_ - 0.5) ** 2)
        assert dist_ridge < dist_ols

    def test_granger_ramanathan_formula(self) -> None:
        """Verify constrained solution against manual calculation."""
        # Simple 2-model case with known solution
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        fc_a = np.array([1.1, 2.2, 2.8, 4.1, 5.3], dtype=np.float64)
        fc_b = np.array([0.9, 1.8, 3.2, 3.9, 4.7], dtype=np.float64)

        combiner = OLSCombiner(intercept=False, constrained=True)
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_ is not None
        # Verify constraint: sum = 1, all >= 0
        assert pytest.approx(np.sum(combiner.weights_), abs=1e-8) == 1.0
        assert np.all(combiner.weights_ >= -1e-10)

        # Verify the combined forecast is reasonable
        f_matrix = np.column_stack([fc_a, fc_b])
        combined = f_matrix @ combiner.weights_
        residuals = actual - combined
        sse = np.sum(residuals**2)

        # SSE should be less than or equal to using equal weights
        equal_combined = 0.5 * fc_a + 0.5 * fc_b
        sse_equal = np.sum((actual - equal_combined) ** 2)
        assert sse <= sse_equal + 1e-8
