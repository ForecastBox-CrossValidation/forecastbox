---
title: "Scenarios API"
description: "API reference for forecastbox.scenarios — ScenarioBuilder, ConditionalForecast, MonteCarloSimulation, FanChart, StressTest"
---

# Scenarios API Reference

!!! info "Module"
    **Import**: `from forecastbox.scenarios import ScenarioBuilder, ConditionalForecast, MonteCarloSimulation, FanChart, StressTest`
    **Source**: `forecastbox/scenarios/`

## Overview

The scenarios module provides tools for conditional forecasting, scenario analysis, stress testing, and uncertainty visualization. It supports both analytical and simulation-based approaches.

| Class | Purpose | Reference |
|-------|---------|-----------|
| [`ScenarioBuilder`](#scenariobuilder) | Fluent builder for scenario construction | — |
| [`ConditionalForecast`](#conditionalforecast) | Conditional forecasting in VAR models | Waggoner & Zha (1999) |
| [`MonteCarloSimulation`](#montecarlosimulation) | Monte Carlo simulation engine | — |
| [`FanChart`](#fanchart) | Fan chart construction from forecasts | — |
| [`StressTest`](#stresstest) | Stress testing framework | — |

**Data classes**: [`Scenario`](#scenario), [`SimulationResult`](#simulationresult), [`StressTestResult`](#stresstestresult)

---

## ScenarioBuilder

Fluent builder for constructing scenarios with variable paths, distributional assumptions, and shocks. Produces immutable `Scenario` objects consumed by `ConditionalForecast` and `MonteCarloSimulation`.

### Constructor

```python
ScenarioBuilder(
    name: str = "",
    horizon: int | None = None,
    base_date: str | pd.Timestamp | None = None,
) -> ScenarioBuilder
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `""` | Human-readable scenario name |
| `horizon` | `int \| None` | `None` | Forecast horizon (periods ahead) |
| `base_date` | `str \| pd.Timestamp \| None` | `None` | Reference date for the scenario |

### Methods

##### `set_variable(name, path)`

Define a deterministic path or distributional assumption for a variable.

```python
builder.set_variable(
    name: str,
    path: NDArray[np.float64] | str | dict[str, Any],
) -> ScenarioBuilder
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Variable name (must match model variable names) |
| `path` | `NDArray[np.float64] \| str \| dict[str, Any]` | *required* | Fixed path (array), distribution name (`"normal"`, `"t"`), or distribution config dict (`{"dist": "normal", "mean": 0.02, "std": 0.01}`) |

**Returns**: `self` (for method chaining)

##### `set_shock(name, type, magnitude)`

Apply a shock to a variable.

```python
builder.set_shock(
    name: str,
    type: str = "level",
    magnitude: float = 0.0,
    persistence: float = 1.0,
) -> ScenarioBuilder
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Variable to shock |
| `type` | `str` | `"level"` | Shock type: `"level"`, `"growth"`, `"std_dev"` |
| `magnitude` | `float` | `0.0` | Shock magnitude (interpretation depends on `type`) |
| `persistence` | `float` | `1.0` | Decay factor per period (`1.0` = permanent, `0.5` = halving each period) |

**Returns**: `self` (for method chaining)

##### `build()`

Validate and produce an immutable `Scenario` object.

```python
builder.build() -> Scenario
```

**Returns**: [`Scenario`](#scenario)

**Raises**: `ValueError` if the scenario is internally inconsistent (e.g., conflicting paths and shocks for the same variable).

### Example

```python
from forecastbox.scenarios import ScenarioBuilder

scenario = (
    ScenarioBuilder(name="Tightening", horizon=12)
    .set_variable("selic", path=np.linspace(0.1375, 0.15, 12))
    .set_shock("exchange_rate", type="std_dev", magnitude=2.0, persistence=0.8)
    .set_variable("cpi", path={"dist": "normal", "mean": 0.045, "std": 0.005})
    .build()
)

print(scenario)
# Scenario('Tightening', variables=3, horizon=12)
```

---

## ConditionalForecast

Conditional forecasting for VAR models. Given a set of conditions (fixed paths for a subset of variables), computes the forecast for the remaining variables consistent with the VAR dynamics.

### Constructor

```python
ConditionalForecast(
    method: str = "analytic",
    n_draws: int = 5000,
    confidence_levels: tuple[float, ...] = (0.80, 0.95),
    seed: int | None = None,
) -> ConditionalForecast
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"analytic"` | Computation method: `"analytic"` (closed-form, faster) or `"gibbs"` (Gibbs sampler, supports nonlinear constraints) |
| `n_draws` | `int` | `5000` | Number of posterior draws (only used when `method="gibbs"`) |
| `confidence_levels` | `tuple[float, ...]` | `(0.80, 0.95)` | Confidence levels for prediction intervals |
| `seed` | `int \| None` | `None` | Random seed for reproducibility (`"gibbs"` method) |

### Key Attributes (after `fit`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `draws_` | `NDArray[np.float64] \| None` | Posterior draws `(n_draws, H, K)` when `method="gibbs"` |
| `is_fitted_` | `bool` | Whether `fit()` has been called |
| `conditions_` | `dict[str, NDArray[np.float64]]` | Conditions used in the latest fit |

### Methods

##### `fit(y, conditions)`

Compute the conditional forecast.

```python
cforecast.fit(
    y: NDArray[np.float64] | pd.DataFrame,
    conditions: dict[str, NDArray[np.float64]] | Scenario,
    var_params: dict[str, Any] | None = None,
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `NDArray[np.float64] \| pd.DataFrame` | *required* | Multivariate time series `(T, K)` |
| `conditions` | `dict[str, NDArray[np.float64]] \| Scenario` | *required* | Fixed paths for conditioned variables, or a `Scenario` object |
| `var_params` | `dict[str, Any] \| None` | `None` | Pre-estimated VAR parameters. If `None`, a VAR is fitted internally |

**Returns**: [`Forecast`](core.md#forecast) with point forecasts and prediction intervals for unconditioned variables.

### Example

```python
from forecastbox.scenarios import ConditionalForecast
import numpy as np

# Condition: Selic rate follows a specific path
conditions = {
    "selic": np.array([0.1375, 0.14, 0.145, 0.15]),
}

cf = ConditionalForecast(method="gibbs", n_draws=10000, seed=42)
forecast = cf.fit(y=macro_data, conditions=conditions)

forecast.plot(variables=["gdp", "cpi"], title="Conditional on Selic Path")
```

!!! tip "Analytic vs. Gibbs"
    Use `method="analytic"` for speed when conditions are simple linear constraints on variable levels. Switch to `method="gibbs"` when you need nonlinear constraints, distributional assumptions, or full posterior uncertainty.

---

## MonteCarloSimulation

Monte Carlo simulation engine for generating forecast distributions from stochastic scenarios.

### Constructor

```python
MonteCarloSimulation(
    n_simulations: int = 10000,
    seed: int | None = None,
    n_jobs: int = 1,
) -> MonteCarloSimulation
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_simulations` | `int` | `10000` | Number of simulation paths |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `n_jobs` | `int` | `1` | Number of parallel workers (`-1` for all cores) |

### Methods

##### `simulate(model, scenario)`

Run the Monte Carlo simulation.

```python
mc.simulate(
    model: Any,
    scenario: Scenario,
    horizon: int | None = None,
) -> SimulationResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Any` | *required* | A fitted forecastbox model with a `.forecast()` method |
| `scenario` | `Scenario` | *required* | Scenario with distributional assumptions |
| `horizon` | `int \| None` | `None` | Override the scenario horizon |

**Returns**: [`SimulationResult`](#simulationresult)

### Example

```python
from forecastbox.scenarios import MonteCarloSimulation, ScenarioBuilder

scenario = (
    ScenarioBuilder(name="Uncertainty", horizon=8)
    .set_variable("oil_price", path={"dist": "normal", "mean": 80, "std": 15})
    .set_variable("exchange_rate", path={"dist": "t", "df": 5, "loc": 5.2, "scale": 0.3})
    .build()
)

mc = MonteCarloSimulation(n_simulations=50000, seed=42, n_jobs=-1)
result = mc.simulate(model=fitted_var, scenario=scenario)

print(result.summary())
#          mean    std     p5    p25    p50    p75    p95
# gdp     0.023  0.008  0.010  0.018  0.023  0.028  0.036
# cpi     0.042  0.006  0.032  0.038  0.042  0.046  0.052
```

---

## FanChart

Constructs fan chart data from forecast objects for uncertainty visualization.

### Constructor

```python
FanChart(
    confidence_levels: tuple[float, ...] = (0.50, 0.70, 0.90),
    distribution: str = "normal",
) -> FanChart
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_levels` | `tuple[float, ...]` | `(0.50, 0.70, 0.90)` | Probability bands for the fan chart |
| `distribution` | `str` | `"normal"` | Assumed distribution: `"normal"`, `"t"`, `"empirical"` |

### Methods

##### `from_forecast(forecast)`

Build fan chart data from a `Forecast` object.

```python
FanChart.from_forecast(
    forecast: Forecast,
    distribution: str = "normal",
    confidence_levels: tuple[float, ...] = (0.50, 0.70, 0.90),
) -> FanChartData
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | [`Forecast`](core.md#forecast) | *required* | Forecast with density draws or prediction intervals |
| `distribution` | `str` | `"normal"` | Distribution for band calculation |
| `confidence_levels` | `tuple[float, ...]` | `(0.50, 0.70, 0.90)` | Probability bands |

**Returns**: `FanChartData` — A data object with `bands`, `central`, and `index` attributes suitable for plotting.

##### `from_simulations(simulation_result)`

Build fan chart data from a `SimulationResult`.

```python
FanChart.from_simulations(
    simulation_result: SimulationResult,
    variable: str | None = None,
    confidence_levels: tuple[float, ...] = (0.50, 0.70, 0.90),
) -> FanChartData
```

### Example

```python
from forecastbox.scenarios import FanChart

fan = FanChart.from_forecast(
    forecast=gdp_forecast,
    distribution="t",
    confidence_levels=(0.50, 0.70, 0.90),
)

fan.plot(title="GDP Forecast Fan Chart", color="steelblue")
```

---

## StressTest

Framework for stress testing forecasts under historical, hypothetical, and reverse scenarios.

### Constructor

```python
StressTest(
    model: Any,
    baseline: Forecast | None = None,
) -> StressTest
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Any` | *required* | A fitted forecastbox model |
| `baseline` | [`Forecast`](core.md#forecast) `\| None` | `None` | Baseline forecast for comparison. If `None`, generated from the model |

### Methods

##### `add_historical(name, start, end)`

Add a stress scenario based on a historical episode.

```python
st.add_historical(
    name: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    variables: list[str] | None = None,
) -> StressTest
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Scenario label (e.g., `"2008 GFC"`) |
| `start` | `str \| pd.Timestamp` | *required* | Start of the historical episode |
| `end` | `str \| pd.Timestamp` | *required* | End of the historical episode |
| `variables` | `list[str] \| None` | `None` | Subset of variables to stress (`None` = all) |

**Returns**: `self`

##### `add_hypothetical(name, shocks)`

Add a hypothetical stress scenario.

```python
st.add_hypothetical(
    name: str,
    shocks: dict[str, float | NDArray[np.float64]],
) -> StressTest
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Scenario label |
| `shocks` | `dict[str, float \| NDArray[np.float64]]` | *required* | Variable-shock mapping. Scalar for constant shock, array for time-varying |

**Returns**: `self`

##### `add_reverse(name, target_variable, threshold)`

Add a reverse stress test: find the shocks that produce a given outcome.

```python
st.add_reverse(
    name: str,
    target_variable: str,
    threshold: float,
    direction: str = "below",
) -> StressTest
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Scenario label |
| `target_variable` | `str` | *required* | Variable whose outcome is constrained |
| `threshold` | `float` | *required* | Target threshold value |
| `direction` | `str` | `"below"` | `"below"` or `"above"` — direction of breach |

**Returns**: `self`

##### `run()`

Execute all registered stress scenarios.

```python
st.run() -> StressTestResult
```

**Returns**: [`StressTestResult`](#stresstestresult)

### Example

```python
from forecastbox.scenarios import StressTest

st = StressTest(model=fitted_var)

# Historical episode
st.add_historical("2008 GFC", start="2008-09-01", end="2009-03-01")

# Hypothetical shock
st.add_hypothetical("Oil Spike", shocks={"oil_price": 0.50, "exchange_rate": 0.15})

# Reverse stress: what shock causes GDP < -2%?
st.add_reverse("GDP Crash", target_variable="gdp", threshold=-0.02, direction="below")

result = st.run()
result.summary()
#               baseline  2008 GFC  Oil Spike  GDP Crash
# gdp             0.023    -0.031     -0.008     -0.020
# cpi             0.042     0.068      0.058      0.051
# exchange_rate   5.200     6.100      5.980      5.850
```

---

## Data Classes

### Scenario

Immutable data class representing a fully-specified scenario. Created via [`ScenarioBuilder.build()`](#build).

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Scenario name |
| `horizon` | `int` | Forecast horizon |
| `base_date` | `pd.Timestamp \| None` | Reference date |
| `variables` | `dict[str, NDArray[np.float64] \| dict]` | Variable paths or distribution configs |
| `shocks` | `dict[str, dict[str, Any]]` | Shock definitions |

### SimulationResult

Container for Monte Carlo simulation output.

| Attribute | Type | Description |
|-----------|------|-------------|
| `paths` | `NDArray[np.float64]` | Simulation paths `(n_simulations, H, K)` |
| `variable_names` | `list[str]` | Variable names |
| `scenario` | `Scenario` | Source scenario |
| `n_simulations` | `int` | Number of simulations run |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Mean, std, percentiles per variable |
| `percentile(q)` | `NDArray[np.float64]` | Paths at percentile `q` |
| `to_dataframe()` | `pd.DataFrame` | Long-form DataFrame of all paths |

### StressTestResult

Container for stress test output.

| Attribute | Type | Description |
|-----------|------|-------------|
| `baseline` | [`Forecast`](core.md#forecast) | Baseline forecast |
| `scenarios` | `dict[str, Forecast]` | Forecast per scenario |
| `impacts` | `pd.DataFrame` | Deviation from baseline per scenario and variable |
| `reverse_shocks` | `dict[str, dict[str, float]] \| None` | Implied shocks from reverse stress tests |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Summary table of all scenarios vs. baseline |
| `plot()` | `matplotlib.Figure` | Visual comparison across scenarios |

---

## See Also

- [Conditional Forecasting Theory](../theory/conditional-theory.md) — Mathematical framework
- [Scenarios User Guide](../user-guide/scenarios/index.md) — Tutorials and worked examples
- [Core API](core.md) — `Forecast` and `ForecastResults` data structures
