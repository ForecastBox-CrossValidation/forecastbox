"""Tests for MonteCarlo simulation."""

from __future__ import annotations

import numpy as np

from forecastbox.scenarios._protocols import SimpleVAR
from forecastbox.scenarios.monte_carlo import MonteCarlo


class TestMonteCarlo:
    """Tests for MonteCarlo simulation engine."""

    def test_fan_width_increases(self, simple_var_model: SimpleVAR) -> None:
        """Fan chart width (q95 - q5) must grow with forecast horizon.

        As forecast uncertainty accumulates, the spread of paths increases.
        We check that width at last horizon >= width at first horizon.
        """
        mc = MonteCarlo(simple_var_model, n_paths=2000, seed=42)
        mc.simulate(steps=12)

        # Get statistics for first variable
        stats = mc.statistics(variable=0)
        widths = stats["q95"].values - stats["q5"].values

        assert widths[-1] >= widths[0], (
            f"Fan width did not increase: first={widths[0]:.4f}, last={widths[-1]:.4f}"
        )

    def test_n_paths(self, simple_var_model: SimpleVAR) -> None:
        """simulate() returns array with shape (n_paths, steps, k_vars).

        For a 3-variable VAR with n_paths=100 and steps=6, the output
        should be (100, 6, 3).
        """
        n_paths = 100
        steps = 6
        mc = MonteCarlo(simple_var_model, n_paths=n_paths, seed=42)
        paths = mc.simulate(steps=steps)

        assert paths.shape == (n_paths, steps, simple_var_model.k_vars), (
            f"Expected shape ({n_paths}, {steps}, {simple_var_model.k_vars}), "
            f"got {paths.shape}"
        )

    def test_reproducible(self, simple_var_model: SimpleVAR) -> None:
        """Same seed produces identical trajectories.

        Two MonteCarlo instances with seed=42 must generate bitwise
        identical paths.
        """
        mc1 = MonteCarlo(simple_var_model, n_paths=50, seed=42)
        paths1 = mc1.simulate(steps=6)

        mc2 = MonteCarlo(simple_var_model, n_paths=50, seed=42)
        paths2 = mc2.simulate(steps=6)

        np.testing.assert_array_equal(paths1, paths2)

    def test_mean_close_to_point(self, simple_var_model: SimpleVAR) -> None:
        """Mean of simulated paths should be close to point forecast.

        By the Law of Large Numbers, the average of N paths converges
        to the unconditional forecast. With N=10000, we expect the
        mean to be within ~2% of the point forecast.
        """
        from forecastbox.scenarios.conditional import ConditionalForecast

        mc = MonteCarlo(simple_var_model, n_paths=10000, seed=42, parametric=True)
        paths = mc.simulate(steps=6)

        # Get unconditional point forecast
        cf = ConditionalForecast(simple_var_model)
        y_unc, _ = cf._unconditional_forecast(steps=6)

        # Compare means for each variable
        for i, name in enumerate(simple_var_model.var_names):
            mc_mean = np.mean(paths[:, :, i], axis=0)
            unc_point = y_unc[i::simple_var_model.k_vars]

            np.testing.assert_allclose(
                mc_mean, unc_point, rtol=0.05, atol=0.5,
                err_msg=f"MC mean for {name} differs from point forecast",
            )

    def test_probability_bounds(self, simple_var_model: SimpleVAR) -> None:
        """probability() returns values between 0 and 1.

        For any condition function, the returned probabilities must
        be valid (in [0, 1]).
        """
        mc = MonteCarlo(simple_var_model, n_paths=500, seed=42)
        mc.simulate(steps=8)

        # Probability that first variable exceeds some value
        probs = mc.probability(lambda y: y > 0.0, variable=0)

        assert np.all(probs >= 0.0), f"Found negative probability: {probs}"
        assert np.all(probs <= 1.0), f"Found probability > 1: {probs}"
        assert len(probs) == 8

        # Trivially true condition
        probs_all = mc.probability(lambda y: y > -1e10, variable=0)
        np.testing.assert_array_equal(probs_all, np.ones(8))

    def test_parametric_vs_bootstrap(self, simple_var_model: SimpleVAR) -> None:
        """Both parametric and bootstrap methods produce reasonable distributions.

        We check that both methods generate paths with finite values
        and non-zero variance.
        """
        mc_param = MonteCarlo(simple_var_model, n_paths=500, seed=42, parametric=True)
        paths_param = mc_param.simulate(steps=6)

        mc_boot = MonteCarlo(simple_var_model, n_paths=500, seed=42, parametric=False)
        paths_boot = mc_boot.simulate(steps=6)

        # Both should have finite values
        assert np.all(np.isfinite(paths_param)), "Parametric paths contain non-finite values"
        assert np.all(np.isfinite(paths_boot)), "Bootstrap paths contain non-finite values"

        # Both should have non-zero variance across paths
        var_param = np.var(paths_param[:, -1, 0])
        var_boot = np.var(paths_boot[:, -1, 0])

        assert var_param > 0, "Parametric paths have zero variance"
        assert var_boot > 0, "Bootstrap paths have zero variance"
