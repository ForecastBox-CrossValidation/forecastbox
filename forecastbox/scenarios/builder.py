"""Scenario builder for named macroeconomic scenarios.

Provides a high-level interface for creating, managing, and comparing
multiple conditional forecast scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from forecastbox.core.forecast import Forecast
    from forecastbox.scenarios._protocols import VARModelProtocol


@dataclass
class ScenarioSpec:
    """Specification for a single scenario.

    Attributes
    ----------
    name : str
        Scenario name (e.g., 'base', 'otimista').
    conditions : dict[str, list[float]]
        Variable name -> imposed path values.
    description : str
        Human-readable description.
    """

    name: str
    conditions: dict[str, list[float]]
    description: str = ""


class ScenarioResults:
    """Container for scenario analysis results.

    Holds forecasts for all scenarios and provides comparison,
    visualization, and export methods.

    Attributes
    ----------
    scenarios : dict[str, dict[str, Forecast]]
        Mapping: scenario_name -> variable_name -> Forecast.
    specs : dict[str, ScenarioSpec]
        Original scenario specifications.
    steps : int
        Forecast horizon.
    var_names : list[str]
        Variable names from the model.
    """

    def __init__(
        self,
        scenarios: dict[str, dict[str, Forecast]],
        specs: dict[str, ScenarioSpec],
        steps: int,
        var_names: list[str],
    ) -> None:
        self.scenarios = scenarios
        self.specs = specs
        self.steps = steps
        self.var_names = var_names

    def get(self, scenario: str, variable: str) -> Forecast:
        """Get forecast for a specific scenario and variable.

        Parameters
        ----------
        scenario : str
            Scenario name.
        variable : str
            Variable name.

        Returns
        -------
        Forecast
            The forecast object.

        Raises
        ------
        KeyError
            If scenario or variable not found.
        """
        if scenario not in self.scenarios:
            available = list(self.scenarios.keys())
            msg = f"Unknown scenario '{scenario}'. Available: {available}"
            raise KeyError(msg)
        if variable not in self.scenarios[scenario]:
            available = list(self.scenarios[scenario].keys())
            msg = f"Unknown variable '{variable}'. Available: {available}"
            raise KeyError(msg)
        return self.scenarios[scenario][variable]

    def compare(
        self,
        variable: str,
        scenarios: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compare point forecasts across scenarios for a variable.

        Parameters
        ----------
        variable : str
            Variable to compare.
        scenarios : list[str] or None
            Scenarios to include. None means all.

        Returns
        -------
        pd.DataFrame
            DataFrame with horizons as rows and scenarios as columns.
        """
        scenario_names = scenarios if scenarios is not None else list(self.scenarios.keys())

        data: dict[str, NDArray[np.float64]] = {}
        for name in scenario_names:
            fc = self.get(name, variable)
            data[name] = fc.point

        index = [f"h={h + 1}" for h in range(self.steps)]
        return pd.DataFrame(data, index=index)

    def plot_scenarios(
        self,
        variable: str,
        scenarios: list[str] | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
        show_intervals: bool = True,
    ) -> plt.Axes:
        """Plot all scenarios for a given variable.

        Parameters
        ----------
        variable : str
            Variable to plot.
        scenarios : list[str] or None
            Scenarios to include. None means all.
        ax : matplotlib Axes or None
            Axes to plot on.
        title : str or None
            Plot title.
        show_intervals : bool
            Whether to show prediction intervals.

        Returns
        -------
        plt.Axes
            The matplotlib Axes.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        scenario_names = scenarios if scenarios is not None else list(self.scenarios.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_names)))

        x = np.arange(1, self.steps + 1)

        for idx, name in enumerate(scenario_names):
            fc = self.get(name, variable)
            color = colors[idx]

            ax.plot(x, fc.point, "-o", color=color, label=name, linewidth=2, markersize=4)

            if show_intervals and fc.lower_80 is not None and fc.upper_80 is not None:
                ax.fill_between(
                    x, fc.lower_80, fc.upper_80,
                    alpha=0.15, color=color,
                )

        ax.set_xlabel("Horizon")
        ax.set_ylabel(variable)
        ax.set_title(title or f"Scenario Comparison: {variable}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_fan(
        self,
        variable: str,
        scenario: str = "base",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot fan chart for a single scenario.

        Parameters
        ----------
        variable : str
            Variable to plot.
        scenario : str
            Scenario name.
        ax : matplotlib Axes or None
            Axes to plot on.

        Returns
        -------
        plt.Axes
            The matplotlib Axes.
        """
        fc = self.get(scenario, variable)
        return fc.plot(ax=ax, title=f"Fan Chart: {variable} ({scenario})")

    def summary(self) -> str:
        """Generate summary table of all scenarios.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("SCENARIO COMPARISON SUMMARY")
        lines.append("=" * 60)

        for var in self.var_names:
            lines.append(f"\nVariable: {var}")
            lines.append("-" * 40)

            df = self.compare(var)
            lines.append(df.to_string())
            lines.append("")

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all results to a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Multi-indexed DataFrame (scenario, horizon) x variables.
        """
        records: list[dict[str, Any]] = []

        for scenario_name, var_forecasts in self.scenarios.items():
            for h in range(self.steps):
                record: dict[str, Any] = {
                    "scenario": scenario_name,
                    "horizon": h + 1,
                }
                for var_name, fc in var_forecasts.items():
                    record[f"{var_name}_point"] = fc.point[h]
                    if fc.lower_80 is not None:
                        record[f"{var_name}_lower_80"] = fc.lower_80[h]
                    if fc.upper_80 is not None:
                        record[f"{var_name}_upper_80"] = fc.upper_80[h]
                    if fc.lower_95 is not None:
                        record[f"{var_name}_lower_95"] = fc.lower_95[h]
                    if fc.upper_95 is not None:
                        record[f"{var_name}_upper_95"] = fc.upper_95[h]
                records.append(record)

        return pd.DataFrame(records)


class ScenarioBuilder:
    """Builder for named macroeconomic scenarios.

    Provides a high-level interface for creating, managing, and executing
    conditional forecast scenarios using a fitted VAR model.

    Parameters
    ----------
    model : VARModelProtocol
        Fitted VAR model.

    Examples
    --------
    >>> builder = ScenarioBuilder(model)
    >>> builder.add_scenario('base', {'selic': [13.75, 13.75]})
    >>> builder.add_scenario('otimista', {'selic': [12.75, 11.75]})
    >>> results = builder.run(steps=12)
    >>> results.plot_scenarios('ipca')
    """

    def __init__(self, model: VARModelProtocol) -> None:
        self.model = model
        self._scenarios: dict[str, ScenarioSpec] = {}

    def add_scenario(
        self,
        name: str,
        conditions: dict[str, list[float] | NDArray[np.float64]],
        description: str = "",
    ) -> None:
        """Add a named scenario.

        Parameters
        ----------
        name : str
            Scenario name (e.g., 'base', 'otimista').
        conditions : dict[str, list[float]]
            Variable name -> imposed path values.
        description : str
            Human-readable description.

        Raises
        ------
        ValueError
            If variable name not recognized.
        """
        # Validate variable names
        for var_name in conditions:
            if var_name not in self.model.var_names:
                msg = f"Unknown variable '{var_name}'. Available: {self.model.var_names}"
                raise ValueError(msg)

        # Convert arrays to lists for serialization
        cond_lists: dict[str, list[float]] = {}
        for var_name, path in conditions.items():
            cond_lists[var_name] = [float(v) for v in np.asarray(path, dtype=np.float64)]

        self._scenarios[name] = ScenarioSpec(
            name=name,
            conditions=cond_lists,
            description=description,
        )

    def remove_scenario(self, name: str) -> None:
        """Remove a scenario by name.

        Parameters
        ----------
        name : str
            Scenario to remove.

        Raises
        ------
        KeyError
            If scenario not found.
        """
        if name not in self._scenarios:
            msg = f"Scenario '{name}' not found. Available: {list(self._scenarios.keys())}"
            raise KeyError(msg)
        del self._scenarios[name]

    def list_scenarios(self) -> list[str]:
        """List all registered scenario names.

        Returns
        -------
        list[str]
            Scenario names in insertion order.
        """
        return list(self._scenarios.keys())

    def run(
        self,
        steps: int,
        n_draws: int = 1000,
        seed: int | None = None,
    ) -> ScenarioResults:
        """Execute all scenarios and return results.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        n_draws : int
            Number of Monte Carlo draws for intervals.
        seed : int or None
            Random seed.

        Returns
        -------
        ScenarioResults
            Results container with forecasts for all scenarios.

        Raises
        ------
        ValueError
            If no scenarios are registered.
        """
        if not self._scenarios:
            msg = "No scenarios registered. Use add_scenario() first."
            raise ValueError(msg)

        from forecastbox.scenarios.conditional import ConditionalForecast

        cf = ConditionalForecast(self.model, method="analytic")

        results: dict[str, dict[str, Any]] = {}
        rng = np.random.default_rng(seed)

        for name, spec in self._scenarios.items():
            scenario_seed = rng.integers(0, 2**31)
            results[name] = cf.forecast(
                steps=steps,
                conditions=spec.conditions if spec.conditions else None,
                n_draws=n_draws,
                seed=int(scenario_seed),
            )

        return ScenarioResults(
            scenarios=results,
            specs=dict(self._scenarios),
            steps=steps,
            var_names=self.model.var_names,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save scenarios to YAML file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        try:
            import yaml
        except ImportError as err:
            msg = "PyYAML is required for YAML support. Install with: pip install pyyaml"
            raise ImportError(msg) from err

        data: dict[str, Any] = {"scenarios": {}}
        for name, spec in self._scenarios.items():
            data["scenarios"][name] = {
                "description": spec.description,
                "conditions": spec.conditions,
            }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        model: VARModelProtocol,
    ) -> ScenarioBuilder:
        """Load scenarios from YAML file.

        Parameters
        ----------
        path : str or Path
            YAML file path.
        model : VARModelProtocol
            Fitted VAR model.

        Returns
        -------
        ScenarioBuilder
            Builder with loaded scenarios.
        """
        try:
            import yaml
        except ImportError as err:
            msg = "PyYAML is required for YAML support. Install with: pip install pyyaml"
            raise ImportError(msg) from err

        with open(path) as f:
            data = yaml.safe_load(f)

        builder = cls(model)

        scenarios_data = data.get("scenarios", {})
        for name, spec_data in scenarios_data.items():
            conditions = spec_data.get("conditions", {})
            description = spec_data.get("description", "")
            builder.add_scenario(name, conditions, description)

        return builder
