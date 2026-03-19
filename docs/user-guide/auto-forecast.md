# Auto-Forecasting

## Overview

ForecastBox provides automatic model selection for several model families.

## AutoARIMA

Automatic ARIMA model selection using AIC/BIC criteria.

```python
from forecastbox.auto.auto_arima import AutoARIMA

model = AutoARIMA(
    max_p=5, max_d=2, max_q=5,
    seasonal=True, seasonal_period=12,
    ic='aic'
)
model.fit(series)
fc = model.forecast(horizon=12)
```

## AutoETS

Automatic Exponential Smoothing model selection.

<!-- TODO: Add AutoETS examples -->

## Theta Method

<!-- TODO: Add Theta examples -->

## AutoSelect

<!-- TODO: Add AutoSelect examples comparing all methods -->
