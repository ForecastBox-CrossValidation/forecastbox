# Core Concepts

## Forecast Container

The `Forecast` class is the fundamental data structure. It stores point forecasts,
prediction intervals, and optional density draws.

```python
from forecastbox import Forecast

fc = Forecast(
    point=np.array([100.5, 101.2, 102.0]),
    lower_80=np.array([98.0, 97.5, 96.8]),
    upper_80=np.array([103.0, 104.9, 107.2]),
    model_name='MyModel',
    horizon=3
)
```

## ForecastExperiment

The high-level API that combines multiple steps into a single workflow.

## Metrics

Point metrics (MAE, RMSE, MAPE, MASE) and interval metrics (CRPS, coverage).

## Cross-Validation

Time series cross-validation with expanding or sliding windows.

<!-- TODO: Add diagrams -->
<!-- TODO: Expand each concept -->
