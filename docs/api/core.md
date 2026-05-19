---
title: "Core API"
description: "API reference for forecastbox.core — Forecast containers, ForecastResults, ForecastHorizon, DataVintage, and ForecastBoxConfig"
---

# Core API Reference

!!! info "Module"
    **Import**: `from forecastbox.core import Forecast, ForecastResults, ForecastHorizon, DataVintage`
    **Source**: `forecastbox/core/`

## Overview

The core module provides the fundamental data structures used throughout forecastbox:

- **`Forecast`** — Container for a single model's point, interval, and density forecasts
- **`ForecastResults`** — Collection of forecasts with evaluation and comparison tools
- **`ForecastHorizon`** / **`MultiHorizon`** — Forecast horizon specification
- **`DataVintage`** — Data vintage management for real-time evaluation
- **`ForecastBoxConfig`** — Global configuration

---

## Forecast

The primary forecast container. Holds point forecasts, prediction intervals, density draws, and metadata. Returned by all model `.forecast()` methods.

### Constructor

```python
Forecast(
    point: NDArray[np.float64],
    lower_80: NDArray[np.float64] | None = None,
    upper_80: NDArray[np.float64] | None = None,
    lower_95: NDArray[np.float64] | None = None,
    upper_95: NDArray[np.float64] | None = None,
    density: NDArray[np.float64] | None = None,
    index: pd.DatetimeIndex | None = None,
    model_name: str = "",
    horizon: int | None = None,
    metadata: dict[str, Any] | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `point` | `NDArray[np.float64]` | *required* | Point forecasts, array of shape `(H,)` |
| `lower_80` | `NDArray[np.float64] \| None` | `None` | Lower bound of 80% prediction interval |
| `upper_80` | `NDArray[np.float64] \| None` | `None` | Upper bound of 80% prediction interval |
| `lower_95` | `NDArray[np.float64] \| None` | `None` | Lower bound of 95% prediction interval |
| `upper_95` | `NDArray[np.float64] \| None` | `None` | Upper bound of 95% prediction interval |
| `density` | `NDArray[np.float64] \| None` | `None` | Density draws of shape `(H, N_draws)` for probabilistic forecasts |
| `index` | `pd.DatetimeIndex \| None` | `None` | Date index for forecast periods |
| `model_name` | `str` | `""` | Name of the model that generated this forecast |
| `horizon` | `int \| None` | `None` | Forecast horizon (inferred from `point` if not given) |
| `metadata` | `dict[str, Any] \| None` | `None` | Arbitrary metadata (model params, fit time, etc.) |

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `point` | `NDArray[np.float64]` | Point forecasts array |
| `lower_80` | `NDArray[np.float64] \| None` | Lower 80% interval |
| `upper_80` | `NDArray[np.float64] \| None` | Upper 80% interval |
| `lower_95` | `NDArray[np.float64] \| None` | Lower 95% interval |
| `upper_95` | `NDArray[np.float64] \| None` | Upper 95% interval |
| `density` | `NDArray[np.float64] \| None` | Density draws `(H, N_draws)` |
| `index` | `pd.DatetimeIndex \| None` | Forecast date index |
| `model_name` | `str` | Model identifier |
| `horizon` | `int` | Number of forecast steps |
| `created_at` | `datetime` | Timestamp of creation |
| `metadata` | `dict[str, Any]` | Additional metadata |

### Methods

##### `to_dataframe()`

Convert the forecast to a pandas DataFrame with columns for point, intervals, and index.

```python
df = forecast.to_dataframe()
```

**Returns**: `pd.DataFrame` with columns `['point', 'lower_80', 'upper_80', 'lower_95', 'upper_95']`.

##### `validate()`

Check internal consistency: array lengths match, lower < upper for intervals, no NaN in point forecasts.

```python
forecast.validate()  # raises ValueError if invalid
```

##### `plot(actual, ax, title, show_intervals)`

Plot the forecast with optional actuals and prediction intervals.

```python
forecast.plot(
    actual: NDArray[np.float64] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_intervals: bool = True,
) -> plt.Axes
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `NDArray[np.float64] \| None` | `None` | Realized values to overlay |
| `ax` | `plt.Axes \| None` | `None` | Matplotlib axes (creates new if `None`) |
| `title` | `str \| None` | `None` | Plot title |
| `show_intervals` | `bool` | `True` | Whether to show shaded prediction intervals |

**Returns**: `plt.Axes`

##### `save(path)`

Serialize the forecast to disk (JSON + numpy arrays).

```python
forecast.save("results/my_forecast.json")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | *required* | Output file path |

##### `from_distribution(draws, index, model_name, quantiles)` { .classmethod }

Construct a `Forecast` from posterior/simulated draws by computing point estimate (median) and quantile-based intervals.

```python
forecast = Forecast.from_distribution(
    draws: NDArray[np.float64],
    index: pd.DatetimeIndex | None = None,
    model_name: str = "",
    quantiles: tuple[float, ...] = (0.10, 0.90, 0.025, 0.975),
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `draws` | `NDArray[np.float64]` | *required* | Simulation draws of shape `(H, N_draws)` |
| `index` | `pd.DatetimeIndex \| None` | `None` | Forecast date index |
| `model_name` | `str` | `""` | Model name |
| `quantiles` | `tuple[float, ...]` | `(0.10, 0.90, 0.025, 0.975)` | Quantiles for 80% and 95% intervals |

**Returns**: `Forecast`

##### `load(path)` { .classmethod }

Load a previously saved forecast from disk.

```python
forecast = Forecast.load("results/my_forecast.json")
```

**Returns**: `Forecast`

##### `combine(forecasts, method)` { .staticmethod }

Quick combination of multiple forecasts using simple methods.

```python
combined = Forecast.combine(
    forecasts: list[Forecast],
    method: str = "mean",
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecasts` | `list[Forecast]` | *required* | List of forecasts to combine |
| `method` | `str` | `"mean"` | Combination method: `"mean"`, `"median"`, `"trimmed"` |

**Returns**: `Forecast`

### Example

```python
import numpy as np
import pandas as pd
from forecastbox.core import Forecast

# Create a 12-step-ahead forecast with intervals
index = pd.date_range("2025-01", periods=12, freq="MS")
point = np.array([2.1, 2.3, 2.5, 2.4, 2.6, 2.8, 2.7, 2.9, 3.0, 3.1, 3.2, 3.3])

forecast = Forecast(
    point=point,
    lower_95=point - 0.5,
    upper_95=point + 0.5,
    index=index,
    model_name="AutoARIMA",
    metadata={"order": (1, 1, 1), "ic": "aicc"},
)

print(f"Horizon: {forecast.horizon}")       # 12
print(f"Model: {forecast.model_name}")       # AutoARIMA

# Convert to DataFrame
df = forecast.to_dataframe()
print(df.head())

# Plot with actuals
actual = np.array([2.0, 2.4, 2.3, 2.5, 2.7, 2.6, 2.8, 2.9, 3.1, 3.0, 3.3, 3.2])
forecast.plot(actual=actual, title="GDP Growth Forecast")

# Save and reload
forecast.save("results/gdp_forecast.json")
loaded = Forecast.load("results/gdp_forecast.json")
```

---

## ForecastResults

Collection of multiple model forecasts with built-in evaluation and comparison. Typically constructed by `AutoSelect` or `ForecastExperiment`.

### Constructor

```python
ForecastResults(
    forecasts: dict[str, Forecast] | None = None,
    actual: np.ndarray | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecasts` | `dict[str, Forecast] \| None` | `None` | Named forecasts (`{model_name: Forecast}`) |
| `actual` | `np.ndarray \| None` | `None` | Realized values for evaluation |

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecasts` | `dict[str, Forecast]` | Dictionary of named forecasts |
| `actual` | `np.ndarray \| None` | Realized values |
| `metrics` | `dict[str, dict[str, float]]` | Computed metrics per model |
| `cv_results` | `dict \| None` | Cross-validation results (if computed) |
| `combination` | `Forecast \| None` | Combined forecast (if computed) |

### Methods

##### `add_forecast(name, forecast)`

Add a named forecast to the collection.

```python
results.add_forecast(name: str, forecast: Forecast) -> None
```

##### `set_actual(actual)`

Set the realized values for evaluation.

```python
results.set_actual(actual: np.ndarray) -> None
```

##### `evaluate(metrics)`

Compute error metrics for all forecasts against actuals.

```python
results.evaluate(
    metrics: tuple[str, ...] = ("mae", "rmse", "mape", "mase"),
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `tuple[str, ...]` | `("mae", "rmse", "mape", "mase")` | Metrics to compute |

**Returns**: `pd.DataFrame` with models as rows and metrics as columns.

##### `rank(metric)`

Rank models by a specific metric (ascending — lower is better).

```python
results.rank(metric: str = "rmse") -> list[str]
```

**Returns**: `list[str]` — Model names ordered from best to worst.

##### `best(metric)`

Return the name of the best model by a specific metric.

```python
results.best(metric: str = "rmse") -> str
```

**Returns**: `str` — Name of the best-performing model.

##### `summary()`

Print a formatted summary table with all models, metrics, and rankings.

```python
results.summary() -> str
```

**Returns**: `str` — Formatted summary.

##### `to_dataframe()`

Convert all forecasts to a single DataFrame (wide format).

```python
results.to_dataframe() -> pd.DataFrame
```

**Returns**: `pd.DataFrame` with one column per model.

##### `plot_comparison(metric)`

Bar chart comparing models by a metric.

```python
results.plot_comparison(metric: str = "rmse") -> plt.Axes
```

##### `plot_forecasts(ax)`

Overlay all forecasts on a single plot.

```python
results.plot_forecasts(ax: plt.Axes | None = None) -> plt.Axes
```

### Example

```python
from forecastbox.core import Forecast, ForecastResults

# Build a collection of forecasts
results = ForecastResults()
results.add_forecast("ARIMA", arima_forecast)
results.add_forecast("ETS", ets_forecast)
results.add_forecast("VAR", var_forecast)

# Set actuals and evaluate
results.set_actual(actual_values)
metrics_df = results.evaluate(metrics=("rmse", "mae", "mape"))
print(metrics_df)
#             rmse    mae   mape
# ARIMA      0.42   0.35   2.10
# ETS        0.45   0.37   2.30
# VAR        0.39   0.31   1.95

# Best model
print(results.best("rmse"))  # "VAR"
print(results.summary())
```

---

## ForecastHorizon

Specification of forecast horizon with optional date alignment.

### Constructor

```python
ForecastHorizon(
    h: int,
    freq: str | None = None,
    origin: str | pd.Timestamp | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h` | `int` | *required* | Number of steps ahead |
| `freq` | `str \| None` | `None` | Pandas frequency string (`"MS"`, `"QS"`, `"Y"`) |
| `origin` | `str \| pd.Timestamp \| None` | `None` | Forecast origin date |

### Methods

##### `to_index()`

Generate a `pd.DatetimeIndex` for the forecast horizon.

```python
horizon.to_index() -> pd.DatetimeIndex
```

**Returns**: `pd.DatetimeIndex` of length `h` starting after `origin`.

### Example

```python
from forecastbox.core import ForecastHorizon

horizon = ForecastHorizon(h=12, freq="MS", origin="2025-01-01")
print(len(horizon))        # 12
print(horizon.to_index())  # DatetimeIndex(['2025-02-01', ..., '2026-01-01'])
```

---

## MultiHorizon

Specification for multiple (possibly non-contiguous) forecast horizons.

### Constructor

```python
MultiHorizon(horizons: list[int])
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `horizons` | `list[int]` | *required* | List of horizon steps (e.g., `[1, 3, 6, 12]`) |

### Example

```python
from forecastbox.core import MultiHorizon

mh = MultiHorizon([1, 3, 6, 12])
print(len(mh))     # 4
print(6 in mh)     # True
```

---

## DataVintage

Manages data vintages for real-time forecast evaluation. Tracks how data revisions affect forecast accuracy.

### Constructor

```python
DataVintage(name: str)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Series name (e.g., `"GDP"`) |

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `release_dates` | `list[date]` | Available vintage dates |

### Methods

##### `add_vintage(release_date, data)`

Register a new data release.

```python
vintage.add_vintage(release_date: date, data: pd.Series) -> None
```

##### `get_vintage(release_date)`

Retrieve data as it was known on a given release date.

```python
vintage.get_vintage(release_date: date) -> pd.Series
```

##### `get_latest()`

Retrieve the most recent data vintage.

```python
vintage.get_latest() -> pd.Series
```

##### `get_revision(period, release1, release2)`

Compute the revision between two vintages for a specific period.

```python
vintage.get_revision(
    period: str,
    release1: date,
    release2: date,
) -> float
```

**Returns**: `float` — Revision magnitude (release2 value − release1 value).

### Example

```python
from datetime import date
import pandas as pd
from forecastbox.core import DataVintage

gdp = DataVintage("GDP")

# Add first release
gdp.add_vintage(
    release_date=date(2025, 3, 28),
    data=pd.Series([2.1, 2.3, 2.5], index=pd.period_range("2024Q2", periods=3, freq="Q")),
)

# Add revised release
gdp.add_vintage(
    release_date=date(2025, 6, 28),
    data=pd.Series([2.2, 2.4, 2.5, 2.7], index=pd.period_range("2024Q2", periods=4, freq="Q")),
)

# Check revision for 2024Q2
revision = gdp.get_revision("2024Q2", date(2025, 3, 28), date(2025, 6, 28))
print(revision)  # 0.1
```

---

## ForecastBoxConfig

Global configuration for forecastbox defaults.

### Constructor

```python
ForecastBoxConfig()
```

!!! note
    `ForecastBoxConfig` is typically accessed via `forecastbox.config` rather than instantiated directly.

---

## See Also

- [Auto-Forecast API](auto-forecast.md) — Automatic model selection that returns `Forecast` objects
- [Combination API](combination.md) — Forecast combination methods that operate on `Forecast` lists
- [Evaluation API](evaluation.md) — Statistical evaluation using `ForecastResults`
- [Getting Started: Core Concepts](../getting-started/core-concepts.md) — Conceptual introduction
