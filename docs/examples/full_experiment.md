# Full Experiment Example

## Overview

Run a complete forecasting experiment from data to report.

```python
from forecastbox import ForecastExperiment

exp = ForecastExperiment(
    data=data,
    target='ipca',
    models=['auto_arima', 'auto_ets', 'theta'],
    combination='bma',
    scenarios={
        'base': {'selic': 13.75},
        'otimista': {'selic': 11.75},
        'pessimista': {'selic': 15.75}
    },
    horizon=12,
    cv_type='expanding',
    report_format='html'
)
results = exp.run()
results.save('experiment_results/')
results.report('full_report.html')

print(results.summary())
```

<!-- TODO: Add step-by-step explanations -->
<!-- TODO: Add output screenshots -->
