"""Tests for StressTest and Shock."""

from __future__ import annotations

import numpy as np

from forecastbox.scenarios._protocols import SimpleVAR
from forecastbox.scenarios.stress_test import StressTest


class TestStressTest:
    """Tests for StressTest."""

    def test_shock_propagation(self, simple_var_model: SimpleVAR) -> None:
        """Shock in variable A must affect variable B (via VAR dynamics).

        A shock to 'cambio' should propagate to 'ipca' through the VAR
        coefficient matrix (exchange rate pass-through).
        """
        stress = StressTest(simple_var_model)
        stress.add_shock("cambio", magnitude=2.0, shock_type="std_dev", period=1)

        result = stress.run(steps=8, seed=42)

        # ipca should be affected (non-zero impact)
        ipca_impact = result.impact["ipca"]
        assert np.any(np.abs(ipca_impact) > 1e-6), (
            "Shock to cambio did not propagate to ipca"
        )

    def test_no_shock_equals_baseline(self, simple_var_model: SimpleVAR) -> None:
        """Without shocks, stressed forecast must equal baseline.

        StressTest with no shocks should produce identical baseline
        and stressed forecasts.
        """
        stress = StressTest(simple_var_model)
        # No shocks added
        result = stress.run(steps=6, seed=42)

        for name in simple_var_model.var_names:
            np.testing.assert_allclose(
                result.stressed[name].point,
                result.baseline[name].point,
                atol=1e-10,
                err_msg=f"Stressed != baseline for {name} with no shocks",
            )

    def test_absolute_shock(self, simple_var_model: SimpleVAR) -> None:
        """Absolute shock of +1 at period 1 increases the variable by exactly 1.

        The contemporaneous effect (Phi_0 = I_k) means an absolute shock
        of +1 to variable i at period 1 increases variable i by exactly 1
        at period 1.
        """
        stress = StressTest(simple_var_model)
        stress.add_shock(
            "selic", magnitude=1.0, shock_type="absolute", period=1, duration=1
        )

        result = stress.run(steps=6, seed=42)

        # At period 1, selic should increase by exactly 1.0
        selic_impact = result.impact["selic"]
        assert abs(selic_impact[0] - 1.0) < 1e-6, (
            f"Expected impact of 1.0 at h=0, got {selic_impact[0]}"
        )

    def test_std_dev_shock(self, simple_var_model: SimpleVAR) -> None:
        """Shock of 2 std_dev should have magnitude = 2 * sqrt(sigma_ii).

        The absolute magnitude of a 2 std_dev shock equals
        2 * sqrt(model.sigma_u[i, i]).
        """
        stress = StressTest(simple_var_model)
        var_idx = simple_var_model.var_names.index("selic")
        expected_std = np.sqrt(simple_var_model.sigma_u[var_idx, var_idx])

        stress.add_shock(
            "selic", magnitude=2.0, shock_type="std_dev", period=1, duration=1
        )
        result = stress.run(steps=6, seed=42)

        # At period 1, impact on selic should be ~2*std
        selic_impact = result.impact["selic"]
        expected_impact = 2.0 * expected_std

        assert abs(selic_impact[0] - expected_impact) < 1e-4, (
            f"Expected impact {expected_impact:.4f}, got {selic_impact[0]:.4f}"
        )

    def test_temporary_shock(self, simple_var_model: SimpleVAR) -> None:
        """Temporary shock (duration=1) should decay over time.

        After a one-period shock, the impact should diminish
        (not necessarily to zero due to VAR persistence).
        """
        stress = StressTest(simple_var_model)
        stress.add_shock(
            "selic", magnitude=3.0, shock_type="absolute", period=1, duration=1
        )

        result = stress.run(steps=8, seed=42)

        selic_impact = result.impact["selic"]
        # Impact at period 1 should be larger than at last period
        assert abs(selic_impact[0]) > abs(selic_impact[-1]), (
            f"Temporary shock did not decay: h=0={selic_impact[0]:.4f}, "
            f"h={len(selic_impact)-1}={selic_impact[-1]:.4f}"
        )

    def test_permanent_shock(self, simple_var_model: SimpleVAR) -> None:
        """Permanent shock (duration=H) persists throughout the forecast.

        With duration equal to the forecast horizon and no decay,
        the impact should persist at all horizons.
        """
        steps = 6
        stress = StressTest(simple_var_model)
        stress.add_shock(
            "selic",
            magnitude=2.0,
            shock_type="absolute",
            period=1,
            duration=steps,
            decay=0.0,
        )

        result = stress.run(steps=steps, seed=42)

        selic_impact = result.impact["selic"]
        # All periods should have non-trivial impact
        for h in range(steps):
            assert abs(selic_impact[h]) > 0.5, (
                f"Permanent shock has negligible impact at h={h}: {selic_impact[h]:.4f}"
            )

    def test_multiple_shocks(self, simple_var_model: SimpleVAR) -> None:
        """Two simultaneous shocks accumulate their effects.

        The combined impact should differ from each individual shock.
        """
        # Individual shock 1
        stress1 = StressTest(simple_var_model)
        stress1.add_shock("selic", magnitude=2.0, shock_type="absolute", period=1)
        result1 = stress1.run(steps=6, seed=42)

        # Individual shock 2
        stress2 = StressTest(simple_var_model)
        stress2.add_shock("cambio", magnitude=1.0, shock_type="absolute", period=1)
        result2 = stress2.run(steps=6, seed=42)

        # Combined shocks
        stress_both = StressTest(simple_var_model)
        stress_both.add_shock("selic", magnitude=2.0, shock_type="absolute", period=1)
        stress_both.add_shock("cambio", magnitude=1.0, shock_type="absolute", period=1)
        result_both = stress_both.run(steps=6, seed=42)

        # Due to linearity of VAR, combined should be approximately sum of individual
        ipca_combined = result_both.impact["ipca"]
        ipca_1 = result1.impact["ipca"]
        ipca_2 = result2.impact["ipca"]

        np.testing.assert_allclose(
            ipca_combined,
            ipca_1 + ipca_2,
            atol=1e-4,
            err_msg="Combined shock impact != sum of individual impacts",
        )
