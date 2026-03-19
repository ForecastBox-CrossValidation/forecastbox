# forecastbox

Forecast containers, evaluation metrics, and cross-validation for time series.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import pandas as pd
from forecastbox import Forecast
from forecastbox.metrics import mae, rmse
from forecastbox.datasets import load_dataset

# Create a forecast
fc = Forecast(
    point=np.array([100.5, 101.2, 102.0]),
    index=pd.date_range('2024-01', periods=3, freq='MS'),
    model_name='MyModel',
    horizon=3
)

# Evaluate
actual = np.array([100.8, 100.9, 103.1])
print(f"MAE: {mae(actual, fc.point):.2f}")
print(f"RMSE: {rmse(actual, fc.point):.2f}")

# Load dataset
data = load_dataset('macro_brazil')
print(data['ipca'].head())
```

## License

MIT
