"""Scenario analysis, conditional forecasts, and stress testing.

This module provides tools for:
- Conditional forecasting (Waggoner-Zha 1999)
- Named scenario analysis (base/optimistic/pessimistic/stress)
- Monte Carlo simulation for uncertainty quantification
- Fan chart visualization (Bank of England style)
- Stress testing with shock propagation via IRF
- Counterfactual analysis ('what if X had been different?')
"""

from forecastbox.scenarios._protocols import SimpleVAR, VARModelProtocol
from forecastbox.scenarios.builder import ScenarioBuilder, ScenarioResults
from forecastbox.scenarios.conditional import ConditionalForecast
from forecastbox.scenarios.counterfactual import Counterfactual, CounterfactualResult
from forecastbox.scenarios.fan_chart import FanChart
from forecastbox.scenarios.monte_carlo import MonteCarlo
from forecastbox.scenarios.stress_test import Shock, StressResult, StressTest

__all__ = [
    "ConditionalForecast",
    "Counterfactual",
    "CounterfactualResult",
    "FanChart",
    "MonteCarlo",
    "ScenarioBuilder",
    "ScenarioResults",
    "Shock",
    "SimpleVAR",
    "StressResult",
    "StressTest",
    "VARModelProtocol",
]
