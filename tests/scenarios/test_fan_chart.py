"""Tests for FanChart (Bank of England style)."""

from __future__ import annotations

import numpy as np

from forecastbox.scenarios.fan_chart import FanChart


class TestFanChart:
    """Tests for FanChart."""

    def test_quantiles_ordered(self) -> None:
        """Quantiles must be ordered: q_10 < q_20 < ... < q_90 for all horizons.

        This is a fundamental property of any valid quantile function.
        """
        rng = np.random.default_rng(42)
        H = 12
        N = 5000

        # Generate draws with increasing variance
        draws = np.zeros((N, H))
        for h in range(H):
            draws[:, h] = rng.normal(100, 1 + h * 0.5, size=N)

        fan = FanChart.from_ensemble(draws)

        quantile_levels = sorted(fan.quantiles.keys())
        for h in range(H):
            values = [fan.quantiles[q][h] for q in quantile_levels]
            for i in range(len(values) - 1):
                assert values[i] <= values[i + 1], (
                    f"At h={h}: q{quantile_levels[i]:.2f}={values[i]:.4f} > "
                    f"q{quantile_levels[i+1]:.2f}={values[i+1]:.4f}"
                )

    def test_from_gaussian(self) -> None:
        """Gaussian fan chart: median = mean, bands symmetric around mean.

        For a Gaussian distribution, q_50 = mean, and
        |q_50 - q_10| should approximately equal |q_90 - q_50|.
        """
        H = 8
        mean = np.linspace(100, 105, H)
        std = np.linspace(1.0, 3.0, H)

        fan = FanChart.from_gaussian(mean, std)

        # Median should equal mean
        np.testing.assert_allclose(fan.median, mean, atol=1e-10)

        # Check symmetry: |median - q10| ~ |q90 - median|
        q10 = fan.quantiles[0.10]
        q90 = fan.quantiles[0.90]
        diff_lower = fan.median - q10
        diff_upper = q90 - fan.median

        np.testing.assert_allclose(diff_lower, diff_upper, atol=1e-10)

        # q50 should equal mean for Gaussian
        if 0.50 in fan.quantiles:
            np.testing.assert_allclose(fan.quantiles[0.50], mean, atol=1e-10)

    def test_contains_coverage(self) -> None:
        """For simulated data, ~90% of points should fall in the 90% band.

        We generate data from a known distribution, create a fan chart,
        then check that the empirical coverage is close to the nominal level.
        """
        rng = np.random.default_rng(123)
        H = 20
        N = 10000

        mean = np.ones(H) * 100
        std = np.ones(H) * 5

        # Generate training draws for fan chart
        draws = rng.normal(mean, std, size=(N, H))
        fan = FanChart.from_ensemble(draws)

        # Generate test actuals from same distribution
        n_test = 1000
        coverage_count = 0.0

        for _ in range(n_test):
            actual = rng.normal(mean, std)
            contained = fan.contains(actual, level=0.80)
            coverage_count += np.mean(contained)

        empirical_coverage = coverage_count / n_test

        # Should be close to 0.80 (within 0.05 tolerance)
        assert abs(empirical_coverage - 0.80) < 0.05, (
            f"Expected ~80% coverage, got {empirical_coverage:.2%}"
        )

    def test_plot(self) -> None:
        """plot() executes without error, with and without history.

        Verifies that the plotting method runs and returns an Axes object.
        """
        import matplotlib

        matplotlib.use("Agg")

        rng = np.random.default_rng(42)
        H = 12
        N = 1000

        draws = rng.normal(100, 5, size=(N, H))
        history = rng.normal(100, 3, size=50)

        # Without history
        fan1 = FanChart.from_ensemble(draws)
        ax1 = fan1.plot(title="No History")
        assert ax1 is not None

        # With history
        fan2 = FanChart.from_ensemble(draws, history=history)
        ax2 = fan2.plot(title="With History")
        assert ax2 is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_width_at_horizon(self) -> None:
        """width_at_horizon should be monotonically increasing.

        As forecast uncertainty accumulates, the band width at each
        successive horizon should be at least as large as the previous.
        """
        rng = np.random.default_rng(42)
        H = 12
        N = 5000

        # Generate draws with linearly increasing variance
        draws = np.zeros((N, H))
        for h in range(H):
            draws[:, h] = rng.normal(100, 1 + h * 1.0, size=N)

        fan = FanChart.from_ensemble(draws)

        widths = [fan.width_at_horizon(h, level=0.80) for h in range(H)]

        # Check monotonically non-decreasing (with small tolerance for MC noise)
        for h in range(1, H):
            assert widths[h] >= widths[h - 1] * 0.95, (
                f"Width decreased at h={h}: {widths[h-1]:.4f} -> {widths[h]:.4f}"
            )

        # Overall: last >> first
        assert widths[-1] > widths[0], (
            f"Last width ({widths[-1]:.4f}) should be > first ({widths[0]:.4f})"
        )
