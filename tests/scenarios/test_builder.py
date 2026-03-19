"""Tests for ScenarioBuilder and ScenarioResults."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forecastbox.scenarios._protocols import SimpleVAR
from forecastbox.scenarios.builder import ScenarioBuilder, ScenarioResults


class TestScenarioBuilder:
    """Tests for ScenarioBuilder."""

    def test_add_remove_scenario(self, simple_var_model: SimpleVAR) -> None:
        """Adding and removing scenarios updates the internal list.

        After add_scenario('base', ...), list_scenarios() should include 'base'.
        After remove_scenario('base'), it should be gone.
        """
        builder = ScenarioBuilder(simple_var_model)

        builder.add_scenario("base", {"selic": [13.0, 13.0]}, description="Base case")
        builder.add_scenario("otimista", {"selic": [12.0, 11.0]}, description="Optimistic")

        assert "base" in builder.list_scenarios()
        assert "otimista" in builder.list_scenarios()
        assert len(builder.list_scenarios()) == 2

        builder.remove_scenario("base")
        assert "base" not in builder.list_scenarios()
        assert len(builder.list_scenarios()) == 1

        with pytest.raises(KeyError):
            builder.remove_scenario("nonexistent")

    def test_run_all_scenarios(self, simple_var_model: SimpleVAR) -> None:
        """run() generates forecasts for ALL registered scenarios.

        Each scenario should produce forecasts for all model variables.
        """
        builder = ScenarioBuilder(simple_var_model)
        builder.add_scenario("base", {"selic": [13.0, 13.0, 13.0]})
        builder.add_scenario("otimista", {"selic": [12.0, 11.5, 11.0]})
        builder.add_scenario("pessimista", {"selic": [14.0, 14.5, 15.0]})

        results = builder.run(steps=6, n_draws=200, seed=42)

        assert isinstance(results, ScenarioResults)
        assert "base" in results.scenarios
        assert "otimista" in results.scenarios
        assert "pessimista" in results.scenarios

        # Each scenario should have all variables
        for scenario_name in ["base", "otimista", "pessimista"]:
            for var_name in simple_var_model.var_names:
                fc = results.get(scenario_name, var_name)
                assert len(fc.point) == 6

    def test_scenarios_differ(self, simple_var_model: SimpleVAR) -> None:
        """Scenarios with different conditions produce different forecasts.

        'otimista' (lower selic) and 'pessimista' (higher selic) should
        yield different IPCA forecasts.
        """
        builder = ScenarioBuilder(simple_var_model)
        builder.add_scenario("otimista", {"selic": [10.0, 9.0, 8.0]})
        builder.add_scenario("pessimista", {"selic": [16.0, 17.0, 18.0]})

        results = builder.run(steps=6, n_draws=200, seed=42)

        ipca_otimista = results.get("otimista", "ipca").point
        ipca_pessimista = results.get("pessimista", "ipca").point

        # They should be different
        assert not np.allclose(ipca_otimista, ipca_pessimista), (
            "Different scenarios produced identical forecasts"
        )

    def test_plot_scenarios(self, simple_var_model: SimpleVAR) -> None:
        """plot_scenarios() executes without error.

        Verifies that the plotting method runs and returns an Axes object.
        """
        import matplotlib
        matplotlib.use("Agg")

        builder = ScenarioBuilder(simple_var_model)
        builder.add_scenario("base", {"selic": [13.0, 13.0]})
        builder.add_scenario("otimista", {"selic": [12.0, 11.0]})

        results = builder.run(steps=4, n_draws=100, seed=42)
        ax = results.plot_scenarios("ipca")

        assert ax is not None

        import matplotlib.pyplot as plt
        plt.close("all")

    def test_yaml_roundtrip(self, simple_var_model: SimpleVAR, tmp_path: Path) -> None:
        """to_yaml() -> from_yaml() preserves all scenarios.

        Scenarios saved to YAML and loaded back should have identical
        names, conditions, and descriptions.
        """
        builder = ScenarioBuilder(simple_var_model)
        builder.add_scenario(
            "base",
            {"selic": [13.0, 13.0, 13.0]},
            description="Cenario base",
        )
        builder.add_scenario(
            "otimista",
            {"selic": [12.0, 11.5, 11.0], "cambio": [5.0, 4.8, 4.6]},
            description="Cenario otimista",
        )

        yaml_path = tmp_path / "scenarios.yaml"
        builder.to_yaml(yaml_path)

        # Load back
        loaded = ScenarioBuilder.from_yaml(yaml_path, simple_var_model)

        assert set(loaded.list_scenarios()) == set(builder.list_scenarios())

        # Check conditions are preserved
        for name in builder.list_scenarios():
            orig_spec = builder._scenarios[name]
            loaded_spec = loaded._scenarios[name]

            assert orig_spec.description == loaded_spec.description
            assert set(orig_spec.conditions.keys()) == set(loaded_spec.conditions.keys())

            for var_name in orig_spec.conditions:
                np.testing.assert_allclose(
                    orig_spec.conditions[var_name],
                    loaded_spec.conditions[var_name],
                )
