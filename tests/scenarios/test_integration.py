"""End-to-end integration test for the scenarios module.

This test exercises the full pipeline: VAR model -> conditional forecast ->
scenarios -> Monte Carlo -> fan chart -> stress test -> counterfactual.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forecastbox.scenarios import (
    ConditionalForecast,
    Counterfactual,
    MonteCarlo,
    ScenarioBuilder,
    SimpleVAR,
    StressTest,
)


class TestEndToEnd:
    """End-to-end integration test for scenarios module."""

    @pytest.fixture
    def var_model(self) -> SimpleVAR:
        """Create a 3-variable VAR(2) model with synthetic data.

        Variables: ipca (inflation), selic (interest rate), cambio (exchange rate).
        Data has known structural relationships:
        - Higher selic -> lower ipca (Taylor rule)
        - Higher cambio -> higher ipca (pass-through)
        """
        rng = np.random.default_rng(42)
        T = 300
        k = 3

        A1 = np.array([
            [0.5, -0.1, 0.05],
            [0.1, 0.8, 0.0],
            [0.0, -0.05, 0.7],
        ])
        A2 = np.array([
            [0.2, -0.05, 0.02],
            [0.05, 0.1, 0.0],
            [0.0, -0.02, 0.15],
        ])
        c = np.array([0.3, 0.5, 0.2])
        Sigma = np.array([
            [0.04, 0.005, 0.002],
            [0.005, 0.09, -0.003],
            [0.002, -0.003, 0.06],
        ])

        data = np.zeros((T, k))
        data[0] = c
        data[1] = c + rng.multivariate_normal(np.zeros(k), Sigma)
        for t in range(2, T):
            data[t] = (
                c + A1 @ data[t - 1] + A2 @ data[t - 2]
                + rng.multivariate_normal(np.zeros(k), Sigma)
            )

        return SimpleVAR(data, p_order=2, var_names=["ipca", "selic", "cambio"])

    def test_full_pipeline(self, var_model: SimpleVAR) -> None:
        """Exercise the complete scenarios pipeline end-to-end.

        Steps:
        1. ConditionalForecast with Selic path imposed
        2. ScenarioBuilder with base/otimista/pessimista
        3. MonteCarlo simulation with 2000 paths
        4. FanChart from Monte Carlo draws
        5. StressTest with exchange rate shock
        6. Counterfactual analysis on selic

        All steps must complete without errors and produce valid results.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = 12

        # ============================================================
        # Step 1: Conditional Forecast (Waggoner-Zha)
        # ============================================================
        cf = ConditionalForecast(var_model, method="analytic")
        cond_results = cf.forecast(
            steps=steps,
            conditions={"selic": [13.0, 12.5, 12.0, 11.5]},
            n_draws=500,
            seed=42,
        )

        # Verify conditions are respected
        for h, val in enumerate([13.0, 12.5, 12.0, 11.5]):
            assert abs(cond_results["selic"].point[h] - val) < 1e-10

        # Verify all variables have forecasts
        for name in var_model.var_names:
            assert name in cond_results
            assert len(cond_results[name].point) == steps

        # ============================================================
        # Step 2: Scenario Builder
        # ============================================================
        builder = ScenarioBuilder(var_model)
        builder.add_scenario(
            "base",
            {"selic": [13.0, 13.0, 13.0]},
            description="Cenario base",
        )
        builder.add_scenario(
            "otimista",
            {"selic": [12.0, 11.0, 10.0]},
            description="Afrouxamento monetario",
        )
        builder.add_scenario(
            "pessimista",
            {"selic": [14.0, 15.0, 16.0]},
            description="Aperto monetario",
        )

        scenario_results = builder.run(steps=steps, n_draws=200, seed=42)

        # Verify scenarios produced different results
        ipca_base = scenario_results.get("base", "ipca").point
        ipca_otimista = scenario_results.get("otimista", "ipca").point
        ipca_pessimista = scenario_results.get("pessimista", "ipca").point

        assert not np.allclose(ipca_base, ipca_otimista)
        assert not np.allclose(ipca_base, ipca_pessimista)

        # Verify comparison DataFrame
        comparison = scenario_results.compare("ipca")
        assert "base" in comparison.columns
        assert "otimista" in comparison.columns
        assert len(comparison) == steps

        # Plot scenarios
        ax = scenario_results.plot_scenarios("ipca")
        assert ax is not None
        plt.close("all")

        # Summary
        summary = scenario_results.summary()
        assert "ipca" in summary
        assert "base" in summary.lower() or "base" in summary

        # ============================================================
        # Step 3: Monte Carlo Simulation
        # ============================================================
        mc = MonteCarlo(var_model, n_paths=2000, seed=42, parametric=True)
        paths = mc.simulate(steps=steps)

        assert paths.shape == (2000, steps, var_model.k_vars)
        assert np.all(np.isfinite(paths))

        # Statistics
        stats = mc.statistics(variable="ipca")
        assert "mean" in stats.columns
        assert "std" in stats.columns
        assert len(stats) == steps

        # Probability
        probs = mc.probability(lambda y: y > 2.0, variable="ipca")
        assert len(probs) == steps
        assert np.all(probs >= 0) and np.all(probs <= 1)

        # Expected shortfall
        es = mc.expected_shortfall(threshold=0, variable="ipca")
        assert len(es) == steps

        # ============================================================
        # Step 4: Fan Chart
        # ============================================================
        fan = mc.fan_chart(variable="ipca")
        assert fan is not None
        assert len(fan.median) == steps

        # Verify quantiles are ordered
        q_levels = sorted(fan.quantiles.keys())
        for h in range(steps):
            values = [fan.quantiles[q][h] for q in q_levels]
            for i in range(len(values) - 1):
                assert values[i] <= values[i + 1]

        # Width should generally increase
        widths = [fan.width_at_horizon(h, level=0.80) for h in range(steps)]
        assert widths[-1] >= widths[0] * 0.8  # allow some tolerance

        # Plot fan chart
        ax = fan.plot(title="IPCA Fan Chart")
        assert ax is not None
        plt.close("all")

        # DataFrame export
        df = fan.to_dataframe()
        assert "median" in df.columns
        assert len(df) == steps

        # ============================================================
        # Step 5: Stress Test
        # ============================================================
        stress = StressTest(var_model)
        stress.add_shock("cambio", magnitude=2.0, shock_type="std_dev", period=1)
        stress.add_shock("selic", magnitude=3.0, shock_type="absolute", period=1)

        stress_result = stress.run(steps=steps, seed=42)

        # Verify impact
        for name in var_model.var_names:
            assert name in stress_result.impact
            assert len(stress_result.impact[name]) == steps

        # Selic should have impact at period 1
        selic_impact = stress_result.impact["selic"]
        assert abs(selic_impact[0]) > 2.5  # at least 2.5 (3.0 absolute)

        # Plot impact
        ax = stress_result.plot_impact("ipca")
        assert ax is not None
        plt.close("all")

        # Plot comparison
        ax = stress_result.plot_comparison("ipca")
        assert ax is not None
        plt.close("all")

        # Summary
        summary = stress_result.summary()
        assert "STRESS TEST" in summary

        # ============================================================
        # Step 6: Counterfactual Analysis
        # ============================================================
        history = var_model.endog
        n_periods = 10
        selic_idx = var_model.var_names.index("selic")

        actual_selic = history[-n_periods:, selic_idx]
        higher_selic = actual_selic + 3.0

        counterfactual = Counterfactual(var_model, history=history)
        cf_result = counterfactual.run(
            actual_path={"selic": actual_selic},
            counter_path={"selic": higher_selic},
            target="ipca",
        )

        assert len(cf_result.actual) == n_periods
        assert len(cf_result.counterfactual) == n_periods
        assert len(cf_result.diff) == n_periods
        assert len(cf_result.cumulative_diff) == n_periods

        # Difference should be non-zero
        assert np.any(np.abs(cf_result.diff) > 1e-6)

        # Plot
        ax = cf_result.plot(title="Counterfactual: Selic +3 p.p.")
        assert ax is not None
        plt.close("all")

        # Summary
        summary = cf_result.summary()
        assert "COUNTERFACTUAL" in summary

    def test_imports(self) -> None:
        """All public classes must be importable from forecastbox.scenarios.

        Verifies that the __init__.py exports are complete and working.
        """
        from forecastbox.scenarios import (
            ConditionalForecast,
            Counterfactual,
            CounterfactualResult,
            FanChart,
            MonteCarlo,
            ScenarioBuilder,
            ScenarioResults,
            Shock,
            SimpleVAR,
            StressResult,
            StressTest,
        )

        # Verify all are classes/types
        assert callable(ConditionalForecast)
        assert callable(Counterfactual)
        assert callable(CounterfactualResult)
        assert callable(FanChart)
        assert callable(MonteCarlo)
        assert callable(ScenarioBuilder)
        assert callable(ScenarioResults)
        assert callable(Shock)
        assert callable(SimpleVAR)
        assert callable(StressResult)
        assert callable(StressTest)

    def test_yaml_workflow(self, var_model: SimpleVAR, tmp_path: Path) -> None:
        """Full YAML workflow: define scenarios, save, load, run.

        Tests that scenarios can be persisted to YAML and loaded back
        without loss of information, then executed successfully.
        """
        builder = ScenarioBuilder(var_model)
        builder.add_scenario("base", {"selic": [13.0, 13.0]}, description="Base")
        builder.add_scenario("stress", {"selic": [18.0, 19.0]}, description="Stress")

        yaml_path = tmp_path / "scenarios.yaml"
        builder.to_yaml(yaml_path)

        loaded = ScenarioBuilder.from_yaml(yaml_path, var_model)
        assert set(loaded.list_scenarios()) == {"base", "stress"}

        results = loaded.run(steps=6, n_draws=100, seed=42)
        assert "base" in results.scenarios
        assert "stress" in results.scenarios
