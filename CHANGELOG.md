# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-17

### Added

- **Core**: Forecast container, ForecastResults, ForecastHorizon, DataVintage
- **Auto-Forecasting**: AutoARIMA, AutoETS, Theta, AutoVAR, AutoSelect
- **Combination**: 7 methods (mean, median, inverse_mse, ols, bma, stacking, optimal)
- **Evaluation**: Diebold-Mariano, MCS, Giacomini-White, Mincer-Zarnowitz, Encompassing
- **Metrics**: MAE, RMSE, MAPE, MASE, CRPS, coverage
- **Scenarios**: Conditional forecasts, stress testing, fan charts
- **Nowcasting**: DFM, bridge equations, MIDAS, news decomposition
- **Pipeline**: ForecastPipeline, ForecastMonitor with alerts
- **Visualization**: Forecast plots, comparison plots, fan charts
- **Reports**: HTML/Markdown/JSON report generation
- **CLI**: 5 commands (forecast, evaluate, nowcast, monitor, combine)
- **Datasets**: 20 built-in datasets
- **ForecastExperiment**: High-level workflow API
- **Cross-Validation**: Expanding and sliding window
- **Documentation**: MkDocs with ~37 pages
