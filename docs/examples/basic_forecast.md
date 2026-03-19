# Basic Forecast Example

## Overview

This example demonstrates a basic forecasting workflow.

```python
from forecastbox.auto.auto_arima import AutoARIMA
from forecastbox.datasets import load_dataset
from forecastbox.metrics import mae, rmse

# Load data
data = load_dataset('macro_brazil')
series = data['ipca']

# Split train/test
train = series[:-12]
test = series[-12:]

# Fit and forecast
model = AutoARIMA(seasonal_period=12)
model.fit(train)
fc = model.forecast(horizon=12)

# Evaluate
print(f"MAE: {mae(test.values, fc.point):.4f}")
print(f"RMSE: {rmse(test.values, fc.point):.4f}")

# Plot
fc.plot(actual=test.values)
```

<!-- TODO: Add output screenshots -->
<!-- TODO: Add more detailed explanations -->
