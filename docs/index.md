# ForecastBox

**Forecast engine for the NodesEcon ecosystem.**

ForecastBox is a comprehensive Python library for time series forecasting,
evaluation, nowcasting, monitoring, and forecast combination.

## Features

- **Auto-Forecasting**: AutoARIMA, AutoETS, Theta, AutoVAR with automatic model selection
- **Forecast Combination**: 7+ methods including BMA, inverse MSE, stacking
- **Statistical Evaluation**: Diebold-Mariano, Model Confidence Set, Giacomini-White
- **Nowcasting**: Dynamic Factor Models, bridge equations, MIDAS
- **Scenario Analysis**: Conditional forecasts, stress testing, fan charts
- **Pipeline & Monitoring**: Automated workflows with drift detection
- **CLI**: 5 commands for terminal-based workflows
- **20+ Built-in Datasets**: Macro, competition, and simulated data

## Quick Start

```python
from forecastbox import ForecastExperiment

exp = ForecastExperiment(
    data=data,
    target='ipca',
    models=['auto_arima', 'auto_ets', 'theta'],
    combination='bma',
    horizon=12,
)
results = exp.run()
results.report('report.html')
```

## Installation

```bash
pip install forecastbox
```

## License

MIT License. See [LICENSE](https://github.com/nodesecon/forecastbox/blob/main/LICENSE).
