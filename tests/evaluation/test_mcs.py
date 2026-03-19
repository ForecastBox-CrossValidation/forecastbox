"""Tests for the Model Confidence Set.

References
----------
Hansen, P.R., Lunde, A. & Nason, J.M. (2011). "The Model Confidence Set."
    Econometrica, 79(2), 453-497.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.evaluation.mcs import model_confidence_set


class TestMCS:
    """Tests for Model Confidence Set."""

    @pytest.fixture
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(42)

    def test_mcs_includes_best(self, rng: np.random.Generator) -> None:
        """MCS always includes the best model (p-value = 1 for the best)."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecasts = {
            "best": actual + rng.normal(0, 0.5, size=T),
            "medium": actual + rng.normal(0, 3, size=T),
            "worst": actual + rng.normal(0, 10, size=T),
        }

        result = model_confidence_set(
            actual, forecasts, alpha=0.10, n_boot=1000, seed=42
        )
        assert "best" in result.included_models
        # Best model should have highest p-value
        assert result.pvalues["best"] >= result.pvalues["worst"]

    def test_mcs_excludes_bad(self, rng: np.random.Generator) -> None:
        """Clearly worse model is excluded (p-value < alpha)."""
        T = 300
        actual = rng.normal(100, 10, size=T)
        forecasts = {
            "good": actual + rng.normal(0, 1, size=T),
            "terrible": actual + rng.normal(0, 50, size=T),
        }

        result = model_confidence_set(
            actual, forecasts, alpha=0.10, n_boot=1000, seed=42
        )
        assert "terrible" in result.excluded_models
        assert result.pvalues["terrible"] < 0.10

    def test_mcs_all_equal(self, rng: np.random.Generator) -> None:
        """If all models are equal, MCS includes all."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        noise = rng.normal(0, 3, size=T)
        forecasts = {
            "A": actual + noise,
            "B": actual + noise,
            "C": actual + noise,
        }

        result = model_confidence_set(
            actual, forecasts, alpha=0.10, n_boot=1000, seed=42
        )
        assert len(result.included_models) == 3

    def test_mcs_pvalues_ordered(self, rng: np.random.Generator) -> None:
        """p-values are non-decreasing in order of elimination."""
        T = 200
        actual = rng.normal(100, 10, size=T)
        forecasts = {
            "best": actual + rng.normal(0, 1, size=T),
            "mid1": actual + rng.normal(0, 5, size=T),
            "mid2": actual + rng.normal(0, 8, size=T),
            "worst": actual + rng.normal(0, 15, size=T),
        }

        result = model_confidence_set(
            actual, forecasts, alpha=0.25, n_boot=1000, seed=42
        )

        if len(result.elimination_order) >= 2:
            for k in range(1, len(result.elimination_order)):
                prev_name = result.elimination_order[k - 1]
                curr_name = result.elimination_order[k]
                assert result.pvalues[curr_name] >= result.pvalues[prev_name] - 1e-10

    def test_mcs_reproducible(self, rng: np.random.Generator) -> None:
        """Same seed -> same result."""
        T = 100
        actual = rng.normal(100, 10, size=T)
        forecasts = {
            "A": actual + rng.normal(0, 2, size=T),
            "B": actual + rng.normal(0, 5, size=T),
            "C": actual + rng.normal(0, 10, size=T),
        }

        result1 = model_confidence_set(
            actual, forecasts, alpha=0.10, n_boot=500, seed=123
        )
        result2 = model_confidence_set(
            actual, forecasts, alpha=0.10, n_boot=500, seed=123
        )

        assert result1.included_models == result2.included_models
        assert result1.elimination_order == result2.elimination_order
        for name in result1.pvalues:
            assert result1.pvalues[name] == pytest.approx(
                result2.pvalues[name], abs=1e-10
            )

    def test_mcs_range_vs_sq(self, rng: np.random.Generator) -> None:
        """Statistic 'range' and 'semi_quadratic' can give different results."""
        T = 150
        actual = rng.normal(100, 10, size=T)
        forecasts = {
            "A": actual + rng.normal(0, 2, size=T),
            "B": actual + rng.normal(0, 4, size=T),
            "C": actual + rng.normal(0, 6, size=T),
        }

        result_range = model_confidence_set(
            actual, forecasts, alpha=0.10, statistic="range", n_boot=500, seed=42
        )
        result_sq = model_confidence_set(
            actual, forecasts, alpha=0.10, statistic="semi_quadratic", n_boot=500, seed=42
        )

        # Both should produce valid results; they may or may not differ
        assert isinstance(result_range.pvalues, dict)
        assert isinstance(result_sq.pvalues, dict)
        assert len(result_range.pvalues) == 3
        assert len(result_sq.pvalues) == 3
        # Verify summary method works
        summary = result_range.summary()
        assert "Model Confidence Set" in summary
