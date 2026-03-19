"""Tests for TimeVaryingCombiner."""

from __future__ import annotations

import numpy as np

from forecastbox.combination.time_varying import TimeVaryingCombiner


class TestTimeVaryingCombiner:
    """Tests for TimeVaryingCombiner (exponentially weighted time-varying)."""

    def test_initial_equal_weights(self) -> None:
        """At the beginning (before data overwhelms prior), weights are approximately equal."""
        rng = np.random.default_rng(42)
        t = 5  # very few observations
        actual = rng.normal(100, 1, size=t)
        fc_a = actual + rng.normal(0, 0.01, size=t)
        fc_b = actual + rng.normal(0, 0.01, size=t)
        fc_c = actual + rng.normal(0, 0.01, size=t)

        combiner = TimeVaryingCombiner(decay=0.99, initial_mse=1.0)
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.weights_history_ is not None
        initial_weights = combiner.weights_history_[0, :]

        # With high decay and few obs, initial weights should be close to equal
        np.testing.assert_allclose(initial_weights, [1 / 3, 1 / 3, 1 / 3], atol=0.15)

    def test_weights_adapt(self) -> None:
        """When model 1 deteriorates over time, its weight should decrease."""
        rng = np.random.default_rng(42)
        t = 200
        actual = np.ones(t) * 100.0

        # Model A: good in first half, bad in second half
        fc_a = np.zeros(t)
        fc_a[:100] = actual[:100] + rng.normal(0, 0.5, size=100)
        fc_a[100:] = actual[100:] + rng.normal(0, 10.0, size=100)

        # Model B: consistently moderate
        fc_b = actual + rng.normal(0, 2.0, size=t)

        combiner = TimeVaryingCombiner(decay=0.95)
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_history_ is not None

        # In first half, Model A should have higher weight
        avg_weight_a_first = np.mean(combiner.weights_history_[50:100, 0])
        # In second half, Model A should have lower weight
        avg_weight_a_second = np.mean(combiner.weights_history_[150:200, 0])

        assert avg_weight_a_first > avg_weight_a_second

    def test_decay_effect(self) -> None:
        """High decay -> weights change slowly. Low decay -> weights change fast."""
        np.random.default_rng(42)
        t = 100
        actual = np.ones(t) * 100.0

        # Model A: perfect first half, bad second half
        fc_a = np.zeros(t)
        fc_a[:50] = actual[:50]
        fc_a[50:] = actual[50:] + 20.0  # large bias

        # Model B: constant moderate error
        fc_b = actual + 2.0

        # High decay (slow adaptation)
        combiner_high = TimeVaryingCombiner(decay=0.99)
        combiner_high.fit([fc_a, fc_b], actual)

        # Low decay (fast adaptation)
        combiner_low = TimeVaryingCombiner(decay=0.80)
        combiner_low.fit([fc_a, fc_b], actual)

        assert combiner_high.weights_history_ is not None
        assert combiner_low.weights_history_ is not None

        # At end, low decay should have adapted more (lower weight for Model A)
        weight_a_high = combiner_high.weights_history_[-1, 0]
        weight_a_low = combiner_low.weights_history_[-1, 0]

        # Low decay adapts faster, so Model A weight should be lower
        assert weight_a_low < weight_a_high

    def test_weights_sum_one_always(self) -> None:
        """Weights must sum to 1 at every time step."""
        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)
        fc_c = actual + rng.normal(0, 3.0, size=t)

        combiner = TimeVaryingCombiner(decay=0.95)
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.weights_history_ is not None
        row_sums = np.sum(combiner.weights_history_, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(t), atol=1e-10)

    def test_weights_history_shape(self) -> None:
        """weights_history_ has shape (T, K)."""
        rng = np.random.default_rng(42)
        t = 80
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)

        combiner = TimeVaryingCombiner(decay=0.95)
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.weights_history_ is not None
        assert combiner.weights_history_.shape == (t, 2)
