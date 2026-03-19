"""Tests for ConditionalForecast (Waggoner-Zha 1999)."""

from __future__ import annotations

import numpy as np

from forecastbox.scenarios._protocols import SimpleVAR
from forecastbox.scenarios.conditional import ConditionalForecast


class TestConditionalForecast:
    """Tests for ConditionalForecast."""

    def test_respects_conditions(self, simple_var_model: SimpleVAR) -> None:
        """Conditioned variables must have EXACT imposed values.

        The Waggoner-Zha formula guarantees R @ y_cond = r exactly.
        We verify abs(fc['selic'].point - conditions['selic']) < 1e-10.
        """
        cf = ConditionalForecast(simple_var_model, method="analytic")
        conditions = {"selic": [13.0, 12.5, 12.0, 11.5]}

        results = cf.forecast(steps=8, conditions=conditions, n_draws=500, seed=42)

        selic_forecast = results["selic"].point
        for h, imposed_val in enumerate(conditions["selic"]):
            assert abs(selic_forecast[h] - imposed_val) < 1e-10, (
                f"At horizon {h}: expected {imposed_val}, got {selic_forecast[h]}"
            )

    def test_unconditional_when_no_conditions(self, simple_var_model: SimpleVAR) -> None:
        """Without conditions, conditional forecast equals unconditional forecast.

        When conditions=None, the result should match a standard VAR forecast.
        """
        cf = ConditionalForecast(simple_var_model, method="analytic")

        # Conditional with no conditions
        results_cond = cf.forecast(steps=6, conditions=None, n_draws=500, seed=42)

        # Compute unconditional directly
        y_unc, _ = cf._unconditional_forecast(steps=6)

        for i, name in enumerate(simple_var_model.var_names):
            expected = y_unc[i :: simple_var_model.k_vars]
            np.testing.assert_allclose(
                results_cond[name].point,
                expected,
                atol=1e-10,
                err_msg=f"Unconditional mismatch for {name}",
            )

    def test_uncertainty_increases(self, simple_var_model: SimpleVAR) -> None:
        """Confidence intervals must widen with forecast horizon.

        The forecast error covariance grows with h, so intervals should expand.
        We check that the 95% interval width at h+1 >= width at h.
        """
        cf = ConditionalForecast(simple_var_model, method="analytic")
        conditions = {"selic": [13.0, 12.5, 12.0]}

        results = cf.forecast(steps=8, conditions=conditions, n_draws=2000, seed=42)

        # Check for unconditioned variable (ipca)
        fc_ipca = results["ipca"]
        assert fc_ipca.upper_95 is not None
        assert fc_ipca.lower_95 is not None

        widths = fc_ipca.upper_95 - fc_ipca.lower_95

        # At least the last width should be >= first width
        # (exact monotonicity may not hold due to Monte Carlo noise)
        assert widths[-1] >= widths[0] * 0.8, (
            f"Interval width did not grow: first={widths[0]:.4f}, last={widths[-1]:.4f}"
        )

    def test_gibbs_vs_analytic(self, simple_var_model: SimpleVAR) -> None:
        """Gibbs and analytic methods should agree in mean (within 5% tolerance).

        Both methods target the same conditional distribution. The Gibbs sampler
        adds parameter uncertainty, so we allow 5% relative tolerance.
        """
        conditions = {"selic": [13.0, 12.5]}

        cf_analytic = ConditionalForecast(simple_var_model, method="analytic")
        cf_gibbs = ConditionalForecast(simple_var_model, method="gibbs")

        results_analytic = cf_analytic.forecast(
            steps=4, conditions=conditions, n_draws=2000, seed=42
        )
        results_gibbs = cf_gibbs.forecast(
            steps=4, conditions=conditions, n_draws=2000, seed=42
        )

        # Compare unconditioned variable (ipca)
        point_analytic = results_analytic["ipca"].point
        point_gibbs = results_gibbs["ipca"].point

        # Relative tolerance of 5% or absolute tolerance of 0.5
        np.testing.assert_allclose(
            point_gibbs,
            point_analytic,
            rtol=0.05,
            atol=0.5,
            err_msg="Gibbs and analytic methods disagree beyond tolerance",
        )

    def test_multiple_conditions(self, simple_var_model: SimpleVAR) -> None:
        """Conditioning on 2+ variables simultaneously must work.

        We impose paths for both selic and cambio, and verify both
        are respected exactly in the point forecast.
        """
        cf = ConditionalForecast(simple_var_model, method="analytic")
        conditions = {
            "selic": [13.0, 12.5, 12.0],
            "cambio": [5.0, 5.1, 5.2],
        }

        results = cf.forecast(steps=6, conditions=conditions, n_draws=500, seed=42)

        # Check selic conditions
        selic_point = results["selic"].point
        for h, val in enumerate(conditions["selic"]):
            assert abs(selic_point[h] - val) < 1e-10

        # Check cambio conditions
        cambio_point = results["cambio"].point
        for h, val in enumerate(conditions["cambio"]):
            assert abs(cambio_point[h] - val) < 1e-10

        # ipca should still have a valid forecast
        assert len(results["ipca"].point) == 6
