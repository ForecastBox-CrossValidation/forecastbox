---
title: "Pipeline API"
description: "API reference for forecastbox.pipeline — ForecastPipeline, PipelineConfig, ForecastMonitor, Experiment"
---

# Pipeline API Reference

!!! info "Module"
    **Import**: `from forecastbox.pipeline import ForecastPipeline, PipelineConfig, ForecastMonitor, Experiment`
    **Source**: `forecastbox/pipeline/`

## Overview

The pipeline module provides end-to-end workflow orchestration for forecast production, monitoring, and experiment tracking. It supports both programmatic and YAML-based configuration.

| Class | Purpose | Description |
|-------|---------|-------------|
| [`ForecastPipeline`](#forecastpipeline) | Pipeline builder | Fluent API for composing forecast workflows |
| [`PipelineConfig`](#pipelineconfig) | YAML configuration | Declarative pipeline definition from config files |
| [`ForecastMonitor`](#forecastmonitor) | Monitoring | Track forecast performance and detect drift |
| [`Experiment`](#experiment) | Experiment tracking | Log, compare, and reproduce forecast experiments |

---

## ForecastPipeline

Fluent builder for composing end-to-end forecast pipelines. Each stage (data, preprocessing, models, combination, evaluation, output) is optional and can be added in any order.

### Constructor

```python
ForecastPipeline(
    name: str = "",
    verbose: bool = True,
) -> ForecastPipeline
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `""` | Pipeline identifier |
| `verbose` | `bool` | `True` | Print progress messages during execution |

### Methods

##### `add_data(y, X, freq)`

Register the input data.

```python
pipeline.add_data(
    y: NDArray[np.float64] | pd.Series | pd.DataFrame,
    X: NDArray[np.float64] | pd.DataFrame | None = None,
    freq: str | None = None,
) -> ForecastPipeline
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | `NDArray[np.float64] \| pd.Series \| pd.DataFrame` | *required* | Target variable(s) |
| `X` | `NDArray[np.float64] \| pd.DataFrame \| None` | `None` | Exogenous regressors |
| `freq` | `str \| None` | `None` | Frequency string (`"M"`, `"Q"`, `"Y"`). Inferred from index if not provided |

**Returns**: `self`

##### `add_preprocessing(steps)`

Add preprocessing transformations.

```python
pipeline.add_preprocessing(
    steps: list[str | tuple[str, dict[str, Any]]],
) -> ForecastPipeline
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `steps` | `list[str \| tuple[str, dict]]` | *required* | Preprocessing steps. Strings for defaults (e.g., `"log"`, `"diff"`, `"standardize"`, `"seasonal_adjust"`), or tuples for parameterized steps (e.g., `("diff", {"order": 2})`) |

**Returns**: `self`

##### `add_models(models)`

Register forecasting models.

```python
pipeline.add_models(
    models: list[str | tuple[str, dict[str, Any]] | Any],
) -> ForecastPipeline
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[str \| tuple[str, dict] \| Any]` | *required* | Model specifications. Strings for auto-configured models (e.g., `"arima"`, `"ets"`, `"var"`), tuples for parameterized (e.g., `("arima", {"order": (1,1,1)})`), or pre-instantiated model objects |

**Returns**: `self`

##### `add_combination(method, **kwargs)`

Add a combination stage.

```python
pipeline.add_combination(
    method: str = "mean",
    **kwargs: Any,
) -> ForecastPipeline
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"mean"` | Combination method (see [Combination API](combination.md)) |
| `**kwargs` | `Any` | — | Parameters passed to the combiner |

**Returns**: `self`

##### `add_evaluation(metrics, cv_strategy)`

Add an evaluation stage.

```python
pipeline.add_evaluation(
    metrics: tuple[str, ...] = ("rmse", "mae", "mape"),
    cv_strategy: str | None = None,
    cv_params: dict[str, Any] | None = None,
) -> ForecastPipeline
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `tuple[str, ...]` | `("rmse", "mae", "mape")` | Evaluation metrics |
| `cv_strategy` | `str \| None` | `None` | Cross-validation strategy: `"expanding"`, `"rolling"`, `"blocked"` |
| `cv_params` | `dict[str, Any] \| None` | `None` | CV parameters (e.g., `{"initial_window": 60, "horizon": 12}`) |

**Returns**: `self`

##### `add_output(format, path)`

Configure output format and destination.

```python
pipeline.add_output(
    format: str = "dataframe",
    path: str | None = None,
    include_diagnostics: bool = False,
) -> ForecastPipeline
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str` | `"dataframe"` | Output format: `"dataframe"`, `"csv"`, `"excel"`, `"json"` |
| `path` | `str \| None` | `None` | Output file path (required for `"csv"`, `"excel"`, `"json"`) |
| `include_diagnostics` | `bool` | `False` | Include diagnostic information in output |

**Returns**: `self`

##### `build()`

Validate and freeze the pipeline configuration.

```python
pipeline.build() -> ForecastPipeline
```

**Returns**: `self` (frozen — no further modifications allowed)

**Raises**: `ValueError` if required stages are missing or configuration is invalid.

##### `run()`

Execute the pipeline.

```python
pipeline.run() -> PipelineResult
```

**Returns**: `PipelineResult`

### PipelineResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecasts` | `dict[str, Forecast]` | Per-model forecasts |
| `combined` | [`Forecast`](core.md#forecast) `\| None` | Combined forecast (if combination stage was added) |
| `evaluation` | `pd.DataFrame \| None` | Evaluation metrics table |
| `cv_results` | `pd.DataFrame \| None` | Cross-validation results |
| `metadata` | `dict[str, Any]` | Pipeline execution metadata (timing, config, etc.) |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Summary of all models and metrics |
| `best_model()` | `str` | Name of the best model by primary metric |
| `plot()` | `matplotlib.Figure` | Visual comparison of forecasts |

### Example

```python
from forecastbox.pipeline import ForecastPipeline

result = (
    ForecastPipeline(name="GDP Forecast Q1-2025")
    .add_data(y=gdp_series, X=exog_indicators, freq="Q")
    .add_preprocessing(["log", "diff", "seasonal_adjust"])
    .add_models([
        "arima",
        ("ets", {"model": "ZZZ"}),
        ("var", {"maxlags": 4}),
    ])
    .add_combination(method="bma")
    .add_evaluation(
        metrics=("rmse", "mae", "mape", "mase"),
        cv_strategy="expanding",
        cv_params={"initial_window": 40, "horizon": 4},
    )
    .add_output(format="excel", path="output/gdp_forecast.xlsx")
    .build()
    .run()
)

print(result.summary())
#        rmse   mae  mape  mase
# ARIMA  0.42  0.35  2.10  0.88
# ETS    0.45  0.37  2.30  0.92
# VAR    0.39  0.31  1.95  0.81
# BMA    0.37  0.29  1.82  0.76

print(f"Best model: {result.best_model()}")
# Best model: BMA
```

---

## PipelineConfig

Declarative pipeline configuration from YAML files. An alternative to the programmatic `ForecastPipeline` builder.

### Class Methods

##### `from_yaml(path)`

Load pipeline configuration from a YAML file.

```python
PipelineConfig.from_yaml(
    path: str | Path,
) -> PipelineConfig
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | *required* | Path to the YAML configuration file |

**Returns**: `PipelineConfig`

##### `to_pipeline()`

Convert the configuration to a `ForecastPipeline`.

```python
config.to_pipeline() -> ForecastPipeline
```

**Returns**: [`ForecastPipeline`](#forecastpipeline) — Ready to call `.run()`.

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Pipeline name from the config |
| `data` | `dict[str, Any]` | Data stage configuration |
| `preprocessing` | `list[str \| dict]` | Preprocessing steps |
| `models` | `list[str \| dict]` | Model specifications |
| `combination` | `dict[str, Any] \| None` | Combination configuration |
| `evaluation` | `dict[str, Any] \| None` | Evaluation configuration |
| `output` | `dict[str, Any] \| None` | Output configuration |

### YAML Schema

```yaml
# pipeline_config.yaml
name: "GDP Forecast Pipeline"

data:
  target: "data/gdp.csv"
  exogenous: "data/indicators.csv"
  freq: "Q"

preprocessing:
  - log
  - diff:
      order: 1
  - seasonal_adjust

models:
  - arima
  - ets:
      model: "ZZZ"
  - var:
      maxlags: 4

combination:
  method: bma

evaluation:
  metrics: [rmse, mae, mape, mase]
  cv_strategy: expanding
  cv_params:
    initial_window: 40
    horizon: 4

output:
  format: excel
  path: "output/gdp_forecast.xlsx"
```

### Example

```python
from forecastbox.pipeline import PipelineConfig

config = PipelineConfig.from_yaml("pipeline_config.yaml")
pipeline = config.to_pipeline()
result = pipeline.run()
```

---

## ForecastMonitor

Tracks forecast performance over time and detects performance drift.

### Constructor

```python
ForecastMonitor(
    metrics: tuple[str, ...] = ("rmse", "mae"),
    window: int = 12,
    threshold: float = 2.0,
) -> ForecastMonitor
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `tuple[str, ...]` | `("rmse", "mae")` | Metrics to track |
| `window` | `int` | `12` | Rolling window for drift detection (periods) |
| `threshold` | `float` | `2.0` | Drift alert threshold (number of standard deviations above rolling mean) |

### Methods

##### `track(forecast, actual)`

Record a forecast-actual pair.

```python
monitor.track(
    forecast: Forecast | float,
    actual: float | NDArray[np.float64],
    date: str | pd.Timestamp | None = None,
    model_name: str = "",
) -> ForecastMonitor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | [`Forecast`](core.md#forecast) `\| float` | *required* | Forecast value or object |
| `actual` | `float \| NDArray[np.float64]` | *required* | Realized value |
| `date` | `str \| pd.Timestamp \| None` | `None` | Date of the observation |
| `model_name` | `str` | `""` | Model identifier |

**Returns**: `self`

##### `detect_drift()`

Check if recent performance deviates significantly from the historical baseline.

```python
monitor.detect_drift() -> dict[str, bool]
```

**Returns**: `dict[str, bool]` — `{metric_name: is_drifting}` for each tracked metric.

##### `alert(callback)`

Register a callback function triggered on drift detection.

```python
monitor.alert(
    callback: Callable[[str, float, float], None] | None = None,
) -> ForecastMonitor
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `callback` | `Callable \| None` | `None` | Function called with `(metric_name, current_value, threshold_value)`. If `None`, prints to stdout |

**Returns**: `self`

##### `history()`

Return the full tracking history.

```python
monitor.history() -> pd.DataFrame
```

**Returns**: `pd.DataFrame` with columns `date`, `model`, `forecast`, `actual`, and one column per tracked metric.

### Example

```python
from forecastbox.pipeline import ForecastMonitor

monitor = ForecastMonitor(metrics=("rmse", "mae"), window=12, threshold=2.0)
monitor.alert(callback=lambda m, v, t: print(f"⚠ {m} drift: {v:.4f} > {t:.4f}"))

# Monthly tracking loop
for date, forecast, actual in monthly_results:
    monitor.track(forecast=forecast, actual=actual, date=date)

# Check for performance degradation
drift = monitor.detect_drift()
if any(drift.values()):
    print("Model retraining recommended")

# Review performance history
print(monitor.history().tail())
#         date model   rmse    mae
# 2024-08  ...  VAR  0.041  0.033
# 2024-09  ...  VAR  0.038  0.030
# 2024-10  ...  VAR  0.055  0.048  <- drift detected
```

---

## Experiment

Experiment tracking for forecast workflows. Logs forecasts, metrics, and configurations for reproducibility and comparison.

### Constructor

```python
Experiment(
    name: str,
    storage: str | Path = ".experiments",
) -> Experiment
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Experiment name |
| `storage` | `str \| Path` | `".experiments"` | Directory for experiment logs |

### Methods

##### `log_forecast(forecast, model_name, tags)`

Log a forecast with metadata.

```python
exp.log_forecast(
    forecast: Forecast,
    model_name: str = "",
    tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | [`Forecast`](core.md#forecast) | *required* | Forecast to log |
| `model_name` | `str` | `""` | Model identifier |
| `tags` | `dict[str, str] \| None` | `None` | Tags for filtering (e.g., `{"target": "gdp", "vintage": "2024-03"}`) |
| `params` | `dict[str, Any] \| None` | `None` | Model hyperparameters |

**Returns**: `str` — Run ID.

##### `log_metric(run_id, metrics)`

Log evaluation metrics for a run.

```python
exp.log_metric(
    run_id: str,
    metrics: dict[str, float],
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_id` | `str` | *required* | Run ID from `log_forecast()` |
| `metrics` | `dict[str, float]` | *required* | Metric name-value pairs |

##### `compare(run_ids, metric)`

Compare runs by a given metric.

```python
exp.compare(
    run_ids: list[str] | None = None,
    metric: str = "rmse",
    tags_filter: dict[str, str] | None = None,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_ids` | `list[str] \| None` | `None` | Runs to compare (`None` = all) |
| `metric` | `str` | `"rmse"` | Primary metric for ranking |
| `tags_filter` | `dict[str, str] \| None` | `None` | Filter runs by tags |

**Returns**: `pd.DataFrame` — Runs sorted by the specified metric, with columns for model, params, tags, and all logged metrics.

##### `list_runs()`

List all runs in the experiment.

```python
exp.list_runs(
    tags_filter: dict[str, str] | None = None,
) -> pd.DataFrame
```

**Returns**: `pd.DataFrame` with run metadata.

### Example

```python
from forecastbox.pipeline import Experiment

exp = Experiment(name="gdp-nowcast-2025q1", storage="./experiments")

# Log different model runs
for model_name, forecast in model_forecasts.items():
    run_id = exp.log_forecast(
        forecast=forecast,
        model_name=model_name,
        tags={"target": "gdp", "horizon": "1Q"},
        params=model_params[model_name],
    )
    exp.log_metric(run_id, metrics={"rmse": rmse_val, "mae": mae_val})

# Compare all runs
comparison = exp.compare(metric="rmse")
print(comparison)
#        run_id model_name  rmse   mae  params
# 0  run_003        VAR  0.039  0.031  {'maxlags': 4}
# 1  run_001      ARIMA  0.042  0.035  {'order': (1,1,1)}
# 2  run_002        ETS  0.045  0.037  {'model': 'ZZZ'}

# Filter by tags
gdp_runs = exp.compare(tags_filter={"target": "gdp"}, metric="mae")
```

---

## See Also

- [Pipeline User Guide](../user-guide/pipeline/index.md) — Step-by-step pipeline tutorial
- [Experiment Tracking Guide](../user-guide/experiment.md) — Experiment management guide
- [Combination API](combination.md) — Forecast combination methods
- [Evaluation API](evaluation.md) — Statistical tests and metrics
