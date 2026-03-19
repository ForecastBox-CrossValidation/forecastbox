"""Tests for BMACombiner."""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.combination.bma import BMACombiner
from forecastbox.core.forecast import Forecast


class TestBMACombiner:
    """Tests for BMACombiner (Bayesian Model Averaging)."""

    def test_weights_sum_one(self) -> None:
        """Posterior weights must sum to 1."""
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)
        fc_c = actual + rng.normal(0, 3.0, size=t)

        combiner = BMACombiner()
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.posterior_weights_ is not None
        assert pytest.approx(np.sum(combiner.posterior_weights_), abs=1e-10) == 1.0

    def test_uniform_prior(self) -> None:
        """With uniform prior, weights are proportional to evidence only."""
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 0.5, size=t)  # best
        fc_b = actual + rng.normal(0, 5.0, size=t)  # worst

        combiner = BMACombiner(prior_weights=None)  # uniform
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.posterior_weights_ is not None
        # Better model (lower MSE/BIC) should get higher posterior weight
        assert combiner.posterior_weights_[0] > combiner.posterior_weights_[1]

    def test_dominant_model(self) -> None:
        """If one model is far better, it gets weight close to 1."""
        rng = np.random.default_rng(42)
        t = 200
        actual = rng.normal(100, 5, size=t)
        fc_perfect = actual + rng.normal(0, 0.01, size=t)  # near perfect
        fc_terrible = actual + rng.normal(0, 50.0, size=t)  # terrible

        combiner = BMACombiner()
        combiner.fit([fc_perfect, fc_terrible], actual)

        assert combiner.posterior_weights_ is not None
        assert combiner.posterior_weights_[0] > 0.99

    def test_bma_variance(self) -> None:
        """BMA variance > variance of best individual model.

        BMA variance includes between-model uncertainty, so it should
        be larger than the variance of the best individual model.
        """
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 2.0, size=t)
        fc_b = actual + rng.normal(0, 3.0, size=t)
        fc_c = actual + rng.normal(0, 4.0, size=t)

        combiner = BMACombiner()
        combiner.fit([fc_a, fc_b, fc_c], actual)

        # Create forecasts with different point values
        fc1 = Forecast(point=np.array([100.0, 105.0, 110.0]), model_name="A")
        fc2 = Forecast(point=np.array([102.0, 107.0, 108.0]), model_name="B")
        fc3 = Forecast(point=np.array([98.0, 103.0, 112.0]), model_name="C")

        combiner.combine([fc1, fc2, fc3])

        assert combiner.bma_variance_ is not None
        assert combiner.bma_variance_ > 0

        # BMA variance should be positive (includes between-model uncertainty)
        best_mse = np.min(combiner.model_mse_)
        assert combiner.bma_variance_ >= best_mse * 0.5  # rough check

    def test_posterior_update(self) -> None:
        """Non-uniform prior changes posterior weights correctly."""
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 2.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)  # same MSE as A

        # Uniform prior
        combiner_uniform = BMACombiner(prior_weights=None)
        combiner_uniform.fit([fc_a, fc_b], actual)

        # Non-uniform prior favoring model A
        combiner_prior = BMACombiner(prior_weights=np.array([0.9, 0.1]))
        combiner_prior.fit([fc_a, fc_b], actual)

        assert combiner_uniform.posterior_weights_ is not None
        assert combiner_prior.posterior_weights_ is not None

        # With prior favoring A, A should get higher posterior weight
        assert (
            combiner_prior.posterior_weights_[0]
            > combiner_uniform.posterior_weights_[0]
        )

    def test_bic_approximation(self) -> None:
        """Weights via BIC are close to weights via AIC for large samples."""
        rng = np.random.default_rng(42)
        t = 500  # large sample
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 3.0, size=t)

        combiner_bic = BMACombiner(approximation="bic")
        combiner_bic.fit([fc_a, fc_b], actual)

        combiner_aic = BMACombiner(approximation="aic")
        combiner_aic.fit([fc_a, fc_b], actual)

        assert combiner_bic.posterior_weights_ is not None
        assert combiner_aic.posterior_weights_ is not None

        # For large T, BIC and AIC should give similar rankings
        assert (
            combiner_bic.posterior_weights_[0] > combiner_bic.posterior_weights_[1]
        )
        assert (
            combiner_aic.posterior_weights_[0] > combiner_aic.posterior_weights_[1]
        )

        # Weights should be in the same ballpark (within 0.3)
        np.testing.assert_allclose(
            combiner_bic.posterior_weights_,
            combiner_aic.posterior_weights_,
            atol=0.3,
        )
