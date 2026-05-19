---
title: "Auto-Forecast API"
description: "API reference for forecastbox.auto — AutoARIMA, AutoETS, AutoVAR, AutoSelect, ModelZoo, and helper functions"
---

# Auto-Forecast API Reference

!!! info "Module"
    **Import**: `from forecastbox.auto import AutoARIMA, AutoETS, AutoVAR, AutoSelect, ModelZoo`
    **Source**: `forecastbox/auto/`

## Overview

The auto-forecast module provides automatic model selection across multiple forecasting families:

- **`AutoARIMA`** — Automatic ARIMA order selection (stepwise or grid search)
- **`AutoETS`** — Automatic Exponential Smoothing State Space model selection
- **`AutoVAR`** — Automatic VAR lag order selection
- **`AutoSelect`** — Cross-family model comparison and selection
- **`ModelZoo`** — Registry of available forecasting models
- **`auto_forecast()`** — One-line automatic forecasting
- **`compare_models()`** — Compare multiple models on the same data

| Class | Use Case | Key IC |
|-------|----------|--------|
| `AutoARIMA` | Univariate, trending/seasonal data | AICc, BIC |
| `AutoETS` | Univariate, level/trend/seasonal decomposition | AICc, BIC |
| `AutoVAR` | Multivariate, interdependent series | BIC, HQC |
| `AutoSelect` | Best model across all families | CV-based |

---

## AutoARIMA

Automatic ARIMA$(p, d, q)(P, D, Q)_m$ order selection using either Hyndman-Khandakar stepwise algorithm or exhaustive grid search. Implements unit root testing for differencing order and information criteria for model selection.

### Constructor

```python
AutoARIMA(
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    seasonal: bool = False,
    m: int = 12,
    stepwise: bool = True,
    ic: str = "aicc",
    approximation: bool = False,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_p` | `int` | `5` | Maximum AR order |
| `max_d` | `int` | `2` | Maximum differencing order |
| `max_q` | `int` | `5` | Maximum MA order |
| `seasonal` | `bool` | `False` | Whether to include seasonal component |
| `m` | `int` | `12` | Seasonal period (e.g., `12` for monthly, `4` for quarterly) |
| `stepwise` | `bool` | `True` | Use Hyndman-Khandakar stepwise search (`False` = grid search) |
| `ic` | `str` | `"aicc"` | Information criterion: `"aic"`, `"aicc"`, `"bic"` |
| `approximation` | `bool` | `False` | Use CSS approximation for faster fitting |

### Methods

##### `fit(data)`

Select the best ARIMA order and fit the model.

```python
result = model.fit(data: NDArray[np.float64] | list[float]) -> AutoARIMAResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `NDArray[np.float64] \| list[float]` | *required* | Univariate time series |

**Returns**: `AutoARIMAResult`

### Example

```python
import numpy as np
from forecastbox.auto import AutoARIMA

data = np.random.randn(120).cumsum() + 100  # simulated series

model = AutoARIMA(max_p=5, max_q=5, seasonal=True, m=12, ic="aicc")
result = model.fit(data)

print(result.summary())
# AutoARIMA Result
# Order: (1, 1, 1)(1, 0, 1)[12]
# AICc: 342.15
# Log-likelihood: -167.02

forecast = result.forecast(h=12, level=(80, 95))
print(forecast.point[:3])  # [101.2, 101.5, 101.8]
forecast.plot(title="AutoARIMA 12-step forecast")
```

---

## AutoARIMAResult

Result of `AutoARIMA.fit()`. Contains the selected order, fit statistics, and forecasting capability.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `order` | `tuple[int, int, int]` | Selected $(p, d, q)$ order |
| `seasonal_order` | `tuple[int, int, int, int]` | Selected $(P, D, Q, m)$ seasonal order |
| `ic_value` | `float` | Information criterion value of the selected model |
| `ic_name` | `str` | IC used for selection (`"aic"`, `"aicc"`, `"bic"`) |
| `log_likelihood` | `float` | Log-likelihood of the fitted model |
| `n_params` | `int` | Number of estimated parameters |
| `converged` | `bool` | Whether the optimizer converged |

### Methods

##### `forecast(h, level)`

Generate $h$-step-ahead forecasts with prediction intervals.

```python
result.forecast(
    h: int,
    level: tuple[int, ...] = (80, 95),
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h` | `int` | *required* | Forecast horizon |
| `level` | `tuple[int, ...]` | `(80, 95)` | Prediction interval levels (%) |

**Returns**: [`Forecast`](core.md#forecast)

##### `summary()`

Human-readable summary with order, IC, log-likelihood, and convergence status.

```python
result.summary() -> str
```

---

## AutoETS

Automatic selection of ETS (Error, Trend, Seasonal) state space models. Evaluates all valid combinations of error type (A/M), trend type (N/A/Ad/M/Md), and seasonal type (N/A/M).

!!! tip "Model Taxonomy"
    ETS models are denoted by three components — e.g., ETS(A, Ad, M) means Additive error, Damped-Additive trend, Multiplicative seasonality. See Hyndman et al. (2008) for the full taxonomy.

### Constructor

```python
AutoETS(
    seasonal_period: int = 12,
    ic: str = "aicc",
    restrict: bool = True,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seasonal_period` | `int` | `12` | Seasonal period (`12` = monthly, `4` = quarterly) |
| `ic` | `str` | `"aicc"` | Information criterion: `"aic"`, `"aicc"`, `"bic"` |
| `restrict` | `bool` | `True` | Restrict to stable models (avoids numerical issues with certain M-type combinations) |

### Methods

##### `fit(data)`

Select the best ETS model and fit it.

```python
result = model.fit(data: NDArray[np.float64] | list[float]) -> AutoETSResult
```

**Returns**: `AutoETSResult`

### Example

```python
from forecastbox.auto import AutoETS

model = AutoETS(seasonal_period=12, ic="aicc")
result = model.fit(monthly_sales)

print(result.summary())
# AutoETS Result
# Model: ETS(A, Ad, M)
# AICc: 1205.32
# SSE: 14523.7

forecast = result.forecast(h=6)
print(forecast.point)
```

---

## AutoETSResult

Result of `AutoETS.fit()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `error` | `str` | Error type: `"A"` (additive) or `"M"` (multiplicative) |
| `trend` | `str` | Trend type: `"N"`, `"A"`, `"Ad"` (damped), `"M"`, `"Md"` |
| `seasonal` | `str` | Seasonal type: `"N"`, `"A"`, `"M"` |
| `ic_value` | `float` | Information criterion value |
| `ic_name` | `str` | IC used for selection |
| `sse` | `float` | Sum of squared errors |
| `n_params` | `int` | Number of estimated parameters |
| `converged` | `bool` | Optimizer convergence flag |

### Methods

##### `forecast(h, level)`

Generate $h$-step-ahead forecasts.

```python
result.forecast(
    h: int,
    level: tuple[int, ...] = (80, 95),
) -> Forecast
```

**Returns**: [`Forecast`](core.md#forecast)

##### `summary()`

Human-readable summary with model type, IC, and SSE.

```python
result.summary() -> str
```

---

## AutoVAR

Automatic lag order selection for Vector Autoregressive models. Evaluates VAR($p$) for $p = 1, \ldots, \text{max\_lags}$ using information criteria. Optionally performs variable selection.

### Constructor

```python
AutoVAR(
    max_lags: int = 12,
    ic: str = "bic",
    variable_selection: bool = False,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_lags` | `int` | `12` | Maximum lag order to evaluate |
| `ic` | `str` | `"bic"` | Information criterion: `"aic"`, `"bic"`, `"hqc"`, `"fpe"` |
| `variable_selection` | `bool` | `False` | Whether to perform stepwise variable selection |

### Methods

##### `fit(data)`

Select the optimal lag order and fit the VAR model.

```python
result = model.fit(data: pd.DataFrame) -> AutoVARResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pd.DataFrame` | *required* | Multivariate time series (columns = variables, rows = observations) |

**Returns**: `AutoVARResult`

### Example

```python
import pandas as pd
from forecastbox.auto import AutoVAR

# Multivariate data: GDP growth, inflation, interest rate
data = pd.DataFrame({
    "gdp": gdp_series,
    "inflation": cpi_series,
    "interest": rate_series,
})

model = AutoVAR(max_lags=8, ic="bic")
result = model.fit(data)

print(result.summary())
# AutoVAR Result
# Selected lag: 2
# Variables: ['gdp', 'inflation', 'interest']
# BIC values: {1: 12.5, 2: 11.8, 3: 12.1, ...}

forecast = result.forecast(h=4)
```

---

## AutoVARResult

Result of `AutoVAR.fit()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `selected_lag` | `int` | Optimal lag order |
| `n_variables` | `int` | Number of endogenous variables |
| `selected_variables` | `list[str] \| None` | Selected variables (if `variable_selection=True`) |
| `ic_values` | `dict[int, float]` | IC value for each tested lag (`{lag: ic_value}`) |
| `ic_name` | `str` | IC used for selection |

### Methods

##### `forecast(h)`

Generate $h$-step-ahead multivariate forecasts.

```python
result.forecast(h: int) -> Forecast
```

**Returns**: [`Forecast`](core.md#forecast)

##### `summary()`

```python
result.summary() -> str
```

---

## AutoSelect

Cross-family automatic model selection. Fits models from multiple families (ARIMA, ETS, VAR, etc.) and selects the best one via time-series cross-validation.

### Constructor

```python
AutoSelect(
    families: list[str] | None = None,
    metric: str = "rmse",
    cv_horizon: int = 12,
    cv_method: str = "rolling",
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `families` | `list[str] \| None` | `None` | Model families to evaluate. Default: `["arima", "ets"]`. Options: `"arima"`, `"ets"`, `"var"`, `"theta"`, `"naive"` |
| `metric` | `str` | `"rmse"` | CV metric for comparison: `"rmse"`, `"mae"`, `"mape"`, `"mase"` |
| `cv_horizon` | `int` | `12` | Forecast horizon for CV evaluation |
| `cv_method` | `str` | `"rolling"` | CV strategy: `"rolling"`, `"expanding"`, `"blocked"` |

### Methods

##### `fit(data)`

Fit all model families and rank by cross-validation performance.

```python
result = model.fit(
    data: NDArray[np.float64] | list[float],
) -> AutoSelectResult
```

**Returns**: `AutoSelectResult`

### Example

```python
from forecastbox.auto import AutoSelect

selector = AutoSelect(
    families=["arima", "ets", "theta"],
    metric="rmse",
    cv_horizon=12,
    cv_method="expanding",
)
result = selector.fit(data)

print(result.summary())
# AutoSelect Result
# Best: AutoARIMA (ARIMA family)
# Ranking:
#   1. AutoARIMA   RMSE=0.42
#   2. Theta        RMSE=0.45
#   3. AutoETS      RMSE=0.48

# Forecast with the best model
forecast = result.forecast(h=12, level=(80, 95))
```

---

## AutoSelectResult

Result of `AutoSelect.fit()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ranking` | `pd.DataFrame` | Ranking table with columns `['family', 'model_name', 'cv_score']` |
| `best_model` | `Any` | The fitted best model object |
| `best_family` | `str` | Family of the best model (`"arima"`, `"ets"`, etc.) |
| `best_model_name` | `str` | Name of the best model |
| `all_cv_results` | `dict[str, list[float]]` | CV scores per model (`{name: [fold_scores]}`) |
| `metric_name` | `str` | Metric used for ranking |

### Methods

##### `forecast(h, level)`

Generate forecasts using the best-ranked model.

```python
result.forecast(
    h: int,
    level: tuple[int, ...] = (80, 95),
) -> Forecast
```

**Returns**: [`Forecast`](core.md#forecast)

##### `summary()`

```python
result.summary() -> str
```

---

## ModelZoo

Singleton registry of available forecasting models. Allows registering custom models that follow the `ForecastModel` protocol.

!!! info "Module"
    **Import**: `from forecastbox.auto import ModelZoo, ForecastModel`

### ForecastModel Protocol

Any class that implements this protocol can be registered in the `ModelZoo`:

```python
@runtime_checkable
class ForecastModel(Protocol):
    def fit(self, y: Any, **kwargs: Any) -> Any: ...
    def forecast(self, h: int, **kwargs: Any) -> Any: ...
```

### Methods

##### `register(name, model_class, family, default_params, description)`

Register a new model in the zoo.

```python
zoo.register(
    name: str,
    model_class: type,
    family: str = "custom",
    default_params: dict[str, Any] | None = None,
    description: str = "",
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Unique model identifier |
| `model_class` | `type` | *required* | Class implementing `ForecastModel` protocol |
| `family` | `str` | `"custom"` | Model family grouping |
| `default_params` | `dict[str, Any] \| None` | `None` | Default constructor parameters |
| `description` | `str` | `""` | Human-readable description |

##### `list_models(family)`

List registered model names, optionally filtered by family.

```python
zoo.list_models(family: str | None = None) -> list[str]
```

**Returns**: `list[str]` — Registered model names.

##### `create(name, **kwargs)`

Instantiate a registered model with optional parameter overrides.

```python
zoo.create(name: str, **kwargs: Any) -> Any
```

**Returns**: Instance of the registered model class.

##### `get(name)`

Retrieve the `ModelEntry` metadata for a registered model.

```python
zoo.get(name: str) -> ModelEntry
```

**Returns**: `ModelEntry`

### ModelEntry

Dataclass representing a registry entry:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Model identifier |
| `model_class` | `type` | Model class |
| `default_params` | `dict[str, Any]` | Default constructor parameters |
| `family` | `str` | Model family (`"arima"`, `"ets"`, `"custom"`, etc.) |
| `description` | `str` | Human-readable description |

### Example

```python
from forecastbox.auto import ModelZoo, ForecastModel

zoo = ModelZoo()

# List built-in models
print(zoo.list_models())
# ['auto_arima', 'auto_ets', 'auto_var', 'theta', 'naive', 'drift']

print(zoo.list_models(family="arima"))
# ['auto_arima']

# Register a custom model
class MyModel:
    def fit(self, y, **kwargs):
        self.mean_ = y.mean()
        return self

    def forecast(self, h, **kwargs):
        return np.full(h, self.mean_)

zoo.register(
    name="custom_mean",
    model_class=MyModel,
    family="custom",
    description="Simple historical mean forecast",
)

# Create and use
model = zoo.create("custom_mean")
model.fit(data)
pred = model.forecast(h=12)
```

---

## Helper Functions

### `auto_forecast()`

One-line automatic forecasting — selects the best model and produces forecasts.

```python
from forecastbox.auto import auto_forecast

forecast = auto_forecast(
    data: NDArray[np.float64] | list[float],
    h: int = 12,
    families: list[str] | None = None,
    ic: str = "aicc",
    level: tuple[int, ...] = (80, 95),
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `NDArray[np.float64] \| list[float]` | *required* | Univariate time series |
| `h` | `int` | `12` | Forecast horizon |
| `families` | `list[str] \| None` | `None` | Model families (default: `["arima", "ets"]`) |
| `ic` | `str` | `"aicc"` | Information criterion for selection |
| `level` | `tuple[int, ...]` | `(80, 95)` | Prediction interval levels (%) |

**Returns**: [`Forecast`](core.md#forecast)

```python
from forecastbox.auto import auto_forecast

forecast = auto_forecast(data, h=12)
forecast.plot(title="Automatic Forecast")
```

### `compare_models()`

Compare multiple models on the same dataset using cross-validation.

```python
from forecastbox.auto import compare_models

comparison = compare_models(
    data: NDArray[np.float64] | list[float],
    models: dict[str, Any] | None = None,
    metric: str = "rmse",
    cv_horizon: int = 12,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `NDArray[np.float64] \| list[float]` | *required* | Univariate time series |
| `models` | `dict[str, Any] \| None` | `None` | Named model instances to compare. Default: all built-in models |
| `metric` | `str` | `"rmse"` | Comparison metric |
| `cv_horizon` | `int` | `12` | CV forecast horizon |

**Returns**: `pd.DataFrame` with columns `['model', 'family', 'cv_score', 'rank']`.

```python
from forecastbox.auto import compare_models

df = compare_models(data, metric="mae", cv_horizon=6)
print(df)
#        model  family  cv_score  rank
# 0  AutoARIMA   arima      0.35     1
# 1    AutoETS     ets      0.38     2
# 2      Theta   theta      0.41     3
```

---

## See Also

- [Core API](core.md) — `Forecast` and `ForecastResults` containers
- [Combination API](combination.md) — Combine forecasts from multiple models
- [Evaluation API](evaluation.md) — Evaluate and compare forecast accuracy
- [User Guide: Auto-Forecast](../user-guide/auto-forecast/index.md) — Detailed usage guide
- [User Guide: ModelZoo](../user-guide/auto-forecast/model-zoo.md) — Extending the model registry
