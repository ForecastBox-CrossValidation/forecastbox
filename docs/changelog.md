# Changelog

## v0.1.0 (2026-03-17)

### Added

- Core: Forecast container, ForecastResults, ForecastHorizon, DataVintage
- Auto: AutoARIMA, AutoETS, Theta, AutoVAR, AutoSelect
- Combination: 7 methods (mean, median, inverse_mse, ols, bma, stacking, optimal)
- Evaluation: Diebold-Mariano, MCS, Giacomini-White, Mincer-Zarnowitz, Encompassing
- Metrics: MAE, RMSE, MAPE, MASE, CRPS, coverage
- Scenarios: Conditional forecasts, stress testing, fan charts
- Nowcasting: DFM, bridge equations, MIDAS, news decomposition
- Pipeline: ForecastPipeline, ForecastMonitor
- Visualization: Forecast plots, comparison plots
- Reports: HTML report generation
- CLI: 5 commands (forecast, evaluate, nowcast, monitor, combine)
- Datasets: 20 built-in datasets
- ForecastExperiment: High-level workflow API
- Cross-validation: Expanding and sliding window
- Documentation: MkDocs with ~37 pages

### Infrastructure

- Python >= 3.11
- Type hints throughout (pyright strict)
- Linting with ruff
- pytest with >= 90% coverage
