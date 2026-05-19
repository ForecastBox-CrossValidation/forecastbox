---
title: "Experiment API"
description: "API reference for forecastbox.experiment — Experiment tracking, comparison, and storage backends (SQLite, Parquet, Memory)"
---

# Experiment API Reference

!!! info "Module"
    **Import**: `from forecastbox.experiment import Experiment, ExperimentStore, SQLiteStore, ParquetStore, MemoryStore`
    **Import**: `from forecastbox.experiment import list_experiments, load_experiment, delete_experiment`
    **Source**: `forecastbox/experiment/`

## Overview

The experiment module provides structured tracking for forecast experiments — logging forecasts, metrics, and parameters across runs for reproducible comparison:

- **`Experiment`** — Context manager for tracking a single experiment run
- **`ExperimentStore`** — Backend abstraction for persistent storage
- **Store implementations** — `SQLiteStore`, `ParquetStore`, `MemoryStore`
- **Utilities** — `list_experiments()`, `load_experiment()`, `delete_experiment()`

| Class / Function | Description |
|------------------|-------------|
| [`Experiment`](#experiment) | Context manager for experiment tracking |
| [`ExperimentStore`](#experimentstore) | Abstract storage backend |
| [`SQLiteStore`](#sqlitestore) | SQLite-based persistent store |
| [`ParquetStore`](#parquetstore) | Parquet file-based store |
| [`MemoryStore`](#memorystore) | In-memory store (non-persistent) |
| [`list_experiments()`](#list_experiments) | List all saved experiments |
| [`load_experiment()`](#load_experiment) | Load an experiment by name |
| [`delete_experiment()`](#delete_experiment) | Delete an experiment by name |

---

## Experiment

Context manager for tracking forecasts, metrics, and parameters within a single experiment run. Each experiment has a unique name and stores all logged data to the configured backend.

### Constructor

```python
Experiment(
    name: str,
    store: ExperimentStore | None = None,
    tags: list[str] | None = None,
    description: str = "",
    overwrite: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Unique experiment name |
| `store` | `ExperimentStore \| None` | `None` | Storage backend (defaults to `SQLiteStore("experiments.db")`) |
| `tags` | `list[str] \| None` | `None` | Tags for filtering and grouping experiments |
| `description` | `str` | `""` | Free-text description of the experiment |
| `overwrite` | `bool` | `False` | If `True`, overwrite an existing experiment with the same name |

### Methods

##### `.log_forecast(name, forecast)`

Log a named forecast to the experiment.

```python
experiment.log_forecast(
    name: str,
    forecast: Forecast,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Forecast identifier (e.g., `"arima_h4"`, `"bma_baseline"`) |
| `forecast` | [`Forecast`](core.md#forecast) | *required* | Forecast object to log |

##### `.log_metric(name, value)`

Log a single named metric.

```python
experiment.log_metric(
    name: str,
    value: float,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Metric name (e.g., `"rmse"`, `"mae"`, `"log_score"`) |
| `value` | `float` | *required* | Metric value |

##### `.log_params(params_dict)`

Log a dictionary of model or experiment parameters.

```python
experiment.log_params(
    params_dict: dict[str, Any],
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params_dict` | `dict[str, Any]` | *required* | Key-value pairs of parameters (e.g., `{"order": (1,1,1), "horizon": 4}`) |

##### `.compare(metric)`

Compare all logged forecasts on a given metric.

```python
experiment.compare(
    metric: str = "rmse",
    sort: bool = True,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `str` | `"rmse"` | Metric to compare across forecasts |
| `sort` | `bool` | `True` | Sort by metric value (ascending) |

**Returns**: `pd.DataFrame` with forecast names as index and the metric as column.

##### `.best(metric)`

Return the name of the best forecast for a given metric.

```python
experiment.best(
    metric: str = "rmse",
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `str` | `"rmse"` | Metric to evaluate (lower is better) |

**Returns**: `str` — name of the best-performing forecast.

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Experiment name |
| `store` | `ExperimentStore` | Storage backend |
| `forecasts` | `dict[str, Forecast]` | Logged forecasts |
| `metrics` | `dict[str, float]` | Logged metrics |
| `params` | `dict[str, Any]` | Logged parameters |
| `tags` | `list[str]` | Experiment tags |
| `created_at` | `datetime` | Experiment creation timestamp |
| `duration` | `timedelta \| None` | Wall-clock duration (set on context exit) |

### Example — Full Experiment Workflow

```python
from forecastbox.experiment import Experiment
from forecastbox.core import Forecast
from forecastbox.metrics import rmse, mae

with Experiment("gdp_q4_2024", tags=["gdp", "quarterly"]) as exp:
    # Log parameters
    exp.log_params({
        "horizon": 4,
        "training_window": "2010Q1–2023Q4",
        "models": ["ARIMA", "ETS", "VAR", "BMA"],
    })

    # Log forecasts
    exp.log_forecast("arima", arima_forecast)
    exp.log_forecast("ets", ets_forecast)
    exp.log_forecast("var", var_forecast)
    exp.log_forecast("bma", bma_forecast)

    # Log metrics
    for name, fc in [("arima", arima_forecast), ("ets", ets_forecast),
                     ("var", var_forecast), ("bma", bma_forecast)]:
        exp.log_metric(f"{name}_rmse", rmse(y_test, fc.point))
        exp.log_metric(f"{name}_mae", mae(y_test, fc.point))

    # Compare
    print(exp.compare("rmse"))
    #        rmse
    # bma    0.31
    # var    0.35
    # arima  0.42
    # ets    0.45

    print(exp.best("rmse"))
    # bma
```

---

## ExperimentStore

Abstract base class for experiment storage backends. All stores implement the same interface.

```python
class ExperimentStore(ABC):
    def save(self, experiment: Experiment) -> None: ...
    def load(self, name: str) -> Experiment: ...
    def list(self) -> list[str]: ...
    def delete(self, name: str) -> None: ...
    def exists(self, name: str) -> bool: ...
```

---

### SQLiteStore

Default persistent store using a local SQLite database.

```python
SQLiteStore(
    path: str | Path = "experiments.db",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str \| Path` | `"experiments.db"` | Path to the SQLite database file |

!!! tip
    `SQLiteStore` is the default backend when no store is specified. Experiments are stored in a single portable `.db` file that can be shared across team members.

### ParquetStore

File-based store using Parquet files — one file per experiment, organized in a directory.

```python
ParquetStore(
    directory: str | Path = "experiments/",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `directory` | `str \| Path` | `"experiments/"` | Directory for Parquet files |

!!! note
    `ParquetStore` is well-suited for environments where experiments need to be queried with tools like DuckDB or pandas, or when you want to version-control experiment results alongside code.

### MemoryStore

Non-persistent in-memory store, useful for testing and interactive sessions.

```python
MemoryStore()
```

No parameters. Data is lost when the Python process exits.

---

## Utility Functions

### list_experiments()

List all saved experiment names in a store.

```python
list_experiments(
    store: ExperimentStore | None = None,
) -> list[str]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store` | `ExperimentStore \| None` | `None` | Storage backend (defaults to `SQLiteStore("experiments.db")`) |

**Returns**: `list[str]` — sorted list of experiment names.

```python
from forecastbox.experiment import list_experiments

print(list_experiments())
# ['gdp_q4_2024', 'inflation_dec_2024', 'exchange_rate_study']
```

### load_experiment()

Load a previously saved experiment by name.

```python
load_experiment(
    name: str,
    store: ExperimentStore | None = None,
) -> Experiment
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Experiment name |
| `store` | `ExperimentStore \| None` | `None` | Storage backend |

**Returns**: [`Experiment`](#experiment) with all logged forecasts, metrics, and parameters restored.

```python
from forecastbox.experiment import load_experiment

exp = load_experiment("gdp_q4_2024")
print(exp.best("rmse"))
# bma
```

### delete_experiment()

Delete a saved experiment.

```python
delete_experiment(
    name: str,
    store: ExperimentStore | None = None,
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Experiment name to delete |
| `store` | `ExperimentStore \| None` | `None` | Storage backend |

!!! warning
    This operation is irreversible. The experiment and all its logged data will be permanently deleted from the store.

---

## See Also

- [Core API](core.md) — `Forecast` objects logged in experiments
- [Evaluation API](evaluation.md) — Metrics used for comparison
- [Pipeline API](pipeline.md) — Integrate experiments into automated pipelines
- [CLI Reference](cli.md) — `forecastbox experiment list` and `forecastbox experiment compare`
