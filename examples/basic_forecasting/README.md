# Basic Forecasting Examples

This directory contains introductory examples for time series forecasting using forecastbox.

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `01_metrics_introduction.ipynb` | MAE, RMSE, MAPE, MASE metrics |
| 2 | `02_cross_validation.ipynb` | Temporal cross-validation |
| 3 | `03_baseline_models.ipynb` | Naive, seasonal naive, drift models |

## Datasets

- `macro_brazil.csv` - Monthly Brazilian macro data (2010-2024): GDP growth, inflation, interest rate, unemployment, exchange rate
- `macro_us.csv` - Monthly US macro data (2010-2024): GDP growth, CPI inflation, Fed funds rate, unemployment

## Requirements

```bash
pip install forecastbox numpy pandas matplotlib statsmodels
```
