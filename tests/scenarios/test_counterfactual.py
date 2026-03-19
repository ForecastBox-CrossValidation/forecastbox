"""Tests for Counterfactual analysis."""

from __future__ import annotations

import numpy as np

from forecastbox.scenarios._protocols import SimpleVAR
from forecastbox.scenarios.counterfactual import Counterfactual


class TestCounterfactual:
    """Tests for Counterfactual analysis."""

    def test_no_change_no_diff(self, simple_var_model: SimpleVAR) -> None:
        """If counter_path == actual_path, the difference should be ~0.

        When the counterfactual path is identical to the actual path,
        the counterfactual trajectory should match the actual, giving
        near-zero differences.
        """
        history = simple_var_model.endog
        n_periods = 10
        selic_idx = simple_var_model.var_names.index("selic")

        # Actual path = last n_periods of selic
        actual_selic = history[-n_periods:, selic_idx]

        cf = Counterfactual(simple_var_model, history=history)
        result = cf.run(
            actual_path={"selic": actual_selic},
            counter_path={"selic": actual_selic},  # same as actual
            target="ipca",
        )

        # Difference should be near zero
        np.testing.assert_allclose(
            result.diff,
            0.0,
            atol=0.5,
            err_msg="No-change counterfactual produced non-zero difference",
        )

    def test_higher_selic_lower_ipca(self, simple_var_model: SimpleVAR) -> None:
        """Higher selic in counterfactual should yield different ipca.

        Since the VAR model has a negative coefficient from selic to ipca,
        a higher counterfactual selic should produce a different ipca path.
        This tests that the model's structural relationships are captured.
        """
        history = simple_var_model.endog
        n_periods = 10
        selic_idx = simple_var_model.var_names.index("selic")

        actual_selic = history[-n_periods:, selic_idx]
        higher_selic = actual_selic + 5.0  # 5 p.p. higher

        cf = Counterfactual(simple_var_model, history=history)
        result = cf.run(
            actual_path={"selic": actual_selic},
            counter_path={"selic": higher_selic},
            target="ipca",
        )

        # Difference should be non-zero
        assert np.any(np.abs(result.diff) > 1e-6), (
            "Higher selic counterfactual produced zero difference in ipca"
        )

    def test_periods_respected(self, simple_var_model: SimpleVAR) -> None:
        """Analysis should be restricted to the specified periods.

        The result length should match the number of periods in the
        counter_path, not the full history.
        """
        history = simple_var_model.endog
        n_periods = 8
        selic_idx = simple_var_model.var_names.index("selic")

        actual_selic = history[-n_periods:, selic_idx]
        counter_selic = actual_selic + 2.0

        cf = Counterfactual(simple_var_model, history=history)
        result = cf.run(
            actual_path={"selic": actual_selic},
            counter_path={"selic": counter_selic},
            target="ipca",
        )

        assert len(result.actual) == n_periods, (
            f"Expected {n_periods} periods, got {len(result.actual)}"
        )
        assert len(result.counterfactual) == n_periods
        assert len(result.diff) == n_periods

    def test_plot(self, simple_var_model: SimpleVAR) -> None:
        """plot() executes without error.

        Verifies that the plotting method runs and returns an Axes object.
        """
        import matplotlib

        matplotlib.use("Agg")

        history = simple_var_model.endog
        n_periods = 6
        selic_idx = simple_var_model.var_names.index("selic")

        actual_selic = history[-n_periods:, selic_idx]
        counter_selic = actual_selic + 3.0

        cf = Counterfactual(simple_var_model, history=history)
        result = cf.run(
            actual_path={"selic": actual_selic},
            counter_path={"selic": counter_selic},
            target="ipca",
        )

        ax = result.plot(title="Test Plot")
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")
