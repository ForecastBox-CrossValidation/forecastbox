"""Tests for WeightedCombiner."""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.combination.weighted import WeightedCombiner
from forecastbox.core.forecast import Forecast


class TestWeightedCombiner:
    """Tests for WeightedCombiner (inverse_mse, aic_weights, bic_weights)."""

    def test_inverse_mse_better_model_higher_weight(self) -> None:
        """Model with lower MSE gets higher weight."""
        rng = np.random.default_rng(42)
        actual = rng.normal(100, 5, size=100)

        # Model A: very accurate (low noise)
        fc_a = actual + rng.normal(0, 0.5, size=100)
        # Model B: less accurate (high noise)
        fc_b = actual + rng.normal(0, 5.0, size=100)
        # Model C: medium accuracy
        fc_c = actual + rng.normal(0, 2.0, size=100)

        combiner = WeightedCombiner(method="inverse_mse")
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.weights_ is not None
        # Model A (index 0) should have highest weight
        assert combiner.weights_[0] > combiner.weights_[1]
        assert combiner.weights_[0] > combiner.weights_[2]

    def test_inverse_mse_weights_sum_one(self) -> None:
        """Inverse MSE weights must sum to 1."""
        rng = np.random.default_rng(42)
        actual = rng.normal(100, 5, size=60)
        fc_a = actual + rng.normal(0, 1.0, size=60)
        fc_b = actual + rng.normal(0, 2.0, size=60)
        fc_c = actual + rng.normal(0, 3.0, size=60)

        combiner = WeightedCombiner(method="inverse_mse")
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.weights_ is not None
        assert pytest.approx(np.sum(combiner.weights_), abs=1e-10) == 1.0

    def test_aic_weights_sum_one(self) -> None:
        """AIC weights must sum to 1."""
        rng = np.random.default_rng(42)
        actual = rng.normal(100, 5, size=60)
        fc_a = actual + rng.normal(0, 1.0, size=60)
        fc_b = actual + rng.normal(0, 2.0, size=60)

        combiner = WeightedCombiner(method="aic_weights")
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_ is not None
        assert pytest.approx(np.sum(combiner.weights_), abs=1e-10) == 1.0

    def test_aic_best_model_highest(self) -> None:
        """Model with lowest AIC gets highest weight."""
        rng = np.random.default_rng(42)
        actual = rng.normal(100, 5, size=100)
        fc_a = actual + rng.normal(0, 0.5, size=100)  # best
        fc_b = actual + rng.normal(0, 5.0, size=100)  # worst

        combiner = WeightedCombiner(method="aic_weights")
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_ is not None
        assert combiner.weights_[0] > combiner.weights_[1]

    def test_equal_mse_equal_weights(self) -> None:
        """Models with identical MSE get equal weights."""
        rng = np.random.default_rng(42)
        actual = np.ones(60) * 100.0
        noise = rng.normal(0, 1.0, size=60)

        # Two models with identical errors (same noise, just shifted)
        fc_a = actual + noise
        fc_b = actual + noise  # same errors

        combiner = WeightedCombiner(method="inverse_mse")
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_ is not None
        np.testing.assert_allclose(combiner.weights_[0], combiner.weights_[1], atol=1e-10)
        np.testing.assert_allclose(combiner.weights_, [0.5, 0.5], atol=1e-10)

    def test_combined_rmse_leq_worst(self) -> None:
        """RMSE of combined forecast <= RMSE of worst individual model."""
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)

        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 3.0, size=t)
        fc_c = actual + rng.normal(0, 5.0, size=t)

        combiner = WeightedCombiner(method="inverse_mse")
        combiner.fit([fc_a, fc_b, fc_c], actual)

        # Create Forecast objects for the combine step
        fc1 = Forecast(point=fc_a, model_name="A")
        fc2 = Forecast(point=fc_b, model_name="B")
        fc3 = Forecast(point=fc_c, model_name="C")

        combined = combiner.combine([fc1, fc2, fc3])

        # Compute RMSEs
        rmse_a = np.sqrt(np.mean((actual - fc_a) ** 2))
        rmse_b = np.sqrt(np.mean((actual - fc_b) ** 2))
        rmse_c = np.sqrt(np.mean((actual - fc_c) ** 2))
        rmse_combined = np.sqrt(np.mean((actual - combined.point) ** 2))

        worst_rmse = max(rmse_a, rmse_b, rmse_c)
        assert rmse_combined <= worst_rmse + 1e-10
