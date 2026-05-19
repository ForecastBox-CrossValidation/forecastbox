# Pipeline Examples

Examples demonstrating forecastbox pipeline automation and monitoring.

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `01_forecast_pipeline.ipynb` | ForecastPipeline and RecurringForecast |
| 2 | `02_monitoring_alerts.ipynb` | ForecastMonitor and alerting |

## Datasets

- `macro_brazil.csv` - Monthly Brazilian macro data (shared with basic_forecasting)

## Pipeline Components

- **ForecastPipeline**: End-to-end pipeline (data -> model -> forecast -> evaluate)
- **RecurringForecast**: Scheduled re-estimation and forecasting
- **ForecastMonitor**: Track forecast accuracy, detect degradation, alert
