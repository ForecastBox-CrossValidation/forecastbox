---
title: "API Reference"
description: "Complete API reference for forecastbox — auto-forecast, combination, evaluation, nowcasting, and pipeline modules"
---

# API Reference

!!! info "Package"
    **Import**: `import forecastbox`
    **Version**: `forecastbox.__version__`
    **Source**: [github.com/nodesecon/forecastbox](https://github.com/nodesecon/forecastbox)

## Overview

forecastbox is organized into modular components that can be used independently or composed into complete forecasting workflows. Each module follows a consistent API pattern: **configure → fit → predict → evaluate**.

## Module Map

| Module | Description | Key Classes / Functions |
|--------|-------------|------------------------|
| [`core`](core.md) | Forecast containers, results, horizons, vintages | `Forecast`, `ForecastResults`, `ForecastHorizon`, `DataVintage` |
| [`auto`](auto-forecast.md) | Automatic model selection | `AutoARIMA`, `AutoETS`, `AutoVAR`, `AutoSelect`, `ModelZoo` |
| [`combination`](combination.md) | Forecast combination methods | `SimpleCombiner`, `WeightedCombiner`, `OLSCombiner`, `BMACombiner`, `OptimalCombiner` |
| [`evaluation`](evaluation.md) | Statistical tests and metrics | `diebold_mariano()`, `model_confidence_set()`, `giacomini_white()` |
| [`scenarios`](scenarios.md) | Conditional forecasting and stress testing | `ConditionalForecaster`, `ScenarioBuilder`, `MonteCarloSimulator` |
| [`nowcasting`](nowcasting.md) | Real-time estimation with mixed-frequency data | `DFM`, `BridgeEquation`, `MIDAS`, `NewsDecomposition` |
| [`pipeline`](pipeline.md) | Production forecasting pipeline | `ForecastPipeline`, `PipelineMonitor` |
| [`diagnostics`](diagnostics.md) | Forecast diagnostic checks | `bias_test()`, `efficiency_test()`, `rationality_test()` |
| [`visualization`](visualization.md) | Plotting functions | `plot_forecast()`, `plot_comparison()`, `plot_combination()` |
| [`reports`](reports.md) | Report generation | `ForecastReport`, `generate_report()` |
| [`experiment`](experiment.md) | End-to-end experiment runner | `ForecastExperiment`, `ExperimentResults` |
| [`datasets`](datasets.md) | Built-in datasets for examples | `load_gdp()`, `load_inflation()`, `load_industrial()` |
| [`cli`](cli.md) | Command-line interface | `forecastbox run`, `forecastbox report` |

## API Conventions

### Naming

- **Classes**: `PascalCase` — `AutoARIMA`, `SimpleCombiner`, `ForecastResults`
- **Functions**: `snake_case` — `diebold_mariano()`, `model_confidence_set()`
- **Parameters**: `snake_case` — `max_lags`, `cv_horizon`, `trim_fraction`
- **Fitted attributes**: trailing underscore — `weights_`, `order_`, `is_fitted_`

### Type Hints

All public functions and methods use type annotations following PEP 484 / PEP 604:

```python
def diebold_mariano(
    actual: NDArray[np.float64],
    forecast1: NDArray[np.float64],
    forecast2: NDArray[np.float64],
    h: int = 1,
    loss: str = "mse",
) -> DMResult: ...
```

Common types used throughout the API:

| Type | Description |
|------|-------------|
| `NDArray[np.float64]` | NumPy array of floats (from `numpy.typing`) |
| `pd.DataFrame` | Pandas DataFrame |
| `pd.DatetimeIndex` | Pandas datetime index |
| `Forecast` | forecastbox forecast container |
| `str \| None` | Optional string (PEP 604 union) |

### Return Types

- **Model selection** returns dataclass results: `AutoARIMAResult`, `AutoETSResult`, `AutoVARResult`
- **Statistical tests** return dataclass results: `DMResult`, `MCSResult`, `GWResult`, `MZResult`
- **Combiners** return `Forecast` objects via `.combine()`
- **Evaluation** returns `pd.DataFrame` with metrics
- All result dataclasses have a `.summary()` method for human-readable output

### Fit / Predict Pattern

Most forecastbox objects follow the scikit-learn-inspired pattern:

```python
# Auto-forecast: configure → fit → forecast
model = AutoARIMA(max_p=5, ic="aicc")
result = model.fit(data)
forecast = result.forecast(h=12)

# Combination: configure → fit → combine
combiner = OLSCombiner(constrained=True)
combiner.fit(forecasts_train, actual)
combined = combiner.combine(forecasts)

# Evaluation: direct function call
dm = diebold_mariano(actual, f1, f2, h=1, loss="mse")
print(dm.conclusion())
```

### Versioning and Deprecation

- forecastbox follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH)
- Deprecated features raise `FutureWarning` for at least one minor version before removal
- Breaking changes are documented in the [Changelog](../contributing/changelog.md)

## Quick Links

- **Getting Started**: [Installation](../getting-started/installation.md) · [Quickstart](../getting-started/quickstart.md)
- **User Guide**: [Auto-Forecast](../user-guide/auto-forecast/index.md) · [Combination](../user-guide/combination/index.md) · [Evaluation](../user-guide/evaluation/index.md)
- **Theory**: [Combination](../theory/combination-theory.md) · [Evaluation](../theory/evaluation-theory.md) · [MCS](../theory/mcs-theory.md)
- **Tutorials**: [Fundamentals](../tutorials/fundamentals.md) · [Complete Workflow](../tutorials/complete-workflow.md)
