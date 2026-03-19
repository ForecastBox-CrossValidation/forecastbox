# Quickstart

## Basic Forecast

```python
import pandas as pd
from forecastbox.auto.auto_arima import AutoARIMA
from forecastbox.datasets import load_dataset

# Load data
data = load_dataset('macro_brazil')
series = data['ipca']

# Fit and forecast
model = AutoARIMA(seasonal_period=12)
model.fit(series)
fc = model.forecast(horizon=12)

# Plot
fc.plot()
```

## Forecast Combination

```python
from forecastbox.auto.auto_arima import AutoARIMA
from forecastbox.auto.auto_ets import AutoETS
from forecastbox.core.forecast import Forecast

fc1 = AutoARIMA().fit(series).forecast(12)
fc2 = AutoETS().fit(series).forecast(12)
combined = Forecast.combine([fc1, fc2], method='mean')
```

## CLI Usage

```bash
forecastbox forecast --data data.csv --target ipca --model auto_arima --horizon 12
```

<!-- TODO: Add more examples -->
<!-- TODO: Add evaluation example -->
<!-- TODO: Add nowcasting example -->
