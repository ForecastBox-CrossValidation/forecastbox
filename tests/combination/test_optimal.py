"""Tests for OptimalCombiner (Bates-Granger 1969)."""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.combination.optimal import OptimalCombiner


class TestOptimalCombiner:
    """Tests for OptimalCombiner (Bates-Granger 1969)."""

    def test_two_models_formula(self) -> None:
        """For K=2, verify the analytic Bates-Granger formula.

        w1* = (sigma2^2 - rho*sigma1*sigma2)
              / (sigma1^2 + sigma2^2 - 2*rho*sigma1*sigma2)
        """
        rng = np.random.default_rng(42)
        t = 1000
        actual = rng.normal(100, 5, size=t)

        # Create errors with known covariance structure
        sigma1, sigma2 = 2.0, 4.0
        rho = 0.5
        # Generate correlated errors
        mean = [0, 0]
        cov = [
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ]
        errors = rng.multivariate_normal(mean, cov, size=t)

        fc_a = actual + errors[:, 0]
        fc_b = actual + errors[:, 1]

        combiner = OptimalCombiner(shrinkage=0.0, min_obs=20)
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_ is not None

        # Analytic formula
        denom = sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2
        w1_analytic = (sigma2**2 - rho * sigma1 * sigma2) / denom

        # Should be close (large T makes sample covariance close to true)
        assert abs(combiner.weights_[0] - w1_analytic) < 0.1
        assert abs(combiner.weights_[1] - (1 - w1_analytic)) < 0.1

    def test_weights_sum_one(self) -> None:
        """Optimal weights must sum to 1."""
        rng = np.random.default_rng(42)
        t = 200
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)
        fc_c = actual + rng.normal(0, 3.0, size=t)

        combiner = OptimalCombiner()
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.weights_ is not None
        assert pytest.approx(np.sum(combiner.weights_), abs=1e-8) == 1.0

    def test_combined_variance_leq_min(self) -> None:
        """Variance of combined forecast <= min individual variance.

        This is the fundamental property of optimal combination:
        Var(f_combined) = 1 / (iota' Sigma^{-1} iota) <= min(diag(Sigma))
        """
        rng = np.random.default_rng(42)
        t = 500
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 2.0, size=t)
        fc_b = actual + rng.normal(0, 3.0, size=t)
        fc_c = actual + rng.normal(0, 4.0, size=t)

        combiner = OptimalCombiner(shrinkage=0.0)
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.optimal_variance_ is not None
        assert combiner.individual_variances_ is not None

        min_individual_var = np.min(combiner.individual_variances_)
        assert combiner.optimal_variance_ <= min_individual_var + 1e-6

    def test_uncorrelated_inverse_variance(self) -> None:
        """If errors are uncorrelated (rho=0), weights proportional to 1/sigma_k^2."""
        rng = np.random.default_rng(42)
        t = 2000  # large sample for accurate estimation

        actual = rng.normal(100, 5, size=t)

        # Independent errors with known variances
        sigma_a, sigma_b = 1.0, 3.0
        fc_a = actual + rng.normal(0, sigma_a, size=t)
        fc_b = actual + rng.normal(0, sigma_b, size=t)

        combiner = OptimalCombiner(shrinkage=0.0)
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_ is not None

        # Expected: w_k proportional to 1/sigma_k^2
        inv_var = np.array([1.0 / sigma_a**2, 1.0 / sigma_b**2])
        expected_weights = inv_var / np.sum(inv_var)

        np.testing.assert_allclose(combiner.weights_, expected_weights, atol=0.05)

    def test_identical_models_equal_weights(self) -> None:
        """Identical models -> equal weights (1/K)."""
        rng = np.random.default_rng(42)
        t = 200
        actual = rng.normal(100, 5, size=t)
        noise = rng.normal(0, 2.0, size=t)

        # Three identical models (same errors)
        fc_a = actual + noise
        fc_b = actual + noise
        fc_c = actual + noise

        combiner = OptimalCombiner(shrinkage=0.0)
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.weights_ is not None
        np.testing.assert_allclose(combiner.weights_, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_shrinkage_toward_equal(self) -> None:
        """With shrinkage=1, covariance is diagonal -> weights closer to 1/sigma^2 ratios.

        Full shrinkage ignores correlations and weights by inverse variance.
        For similar variances, this approaches equal weights.
        """
        rng = np.random.default_rng(42)
        t = 200
        actual = rng.normal(100, 5, size=t)

        # Models with slightly different variances but high correlation
        base_noise = rng.normal(0, 2.0, size=t)
        fc_a = actual + base_noise + rng.normal(0, 0.1, size=t)
        fc_b = actual + base_noise + rng.normal(0, 0.1, size=t)

        # No shrinkage
        combiner_no = OptimalCombiner(shrinkage=0.0)
        combiner_no.fit([fc_a, fc_b], actual)

        # Full shrinkage (diagonal only)
        combiner_full = OptimalCombiner(shrinkage=1.0)
        combiner_full.fit([fc_a, fc_b], actual)

        assert combiner_no.weights_ is not None
        assert combiner_full.weights_ is not None

        # With full shrinkage and similar variances, weights should be near 0.5
        np.testing.assert_allclose(combiner_full.weights_, [0.5, 0.5], atol=0.05)
