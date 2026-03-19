# Command-Line Interface

## Overview

ForecastBox provides 5 CLI commands.

## forecast

```bash
forecastbox forecast --data data.csv --target ipca --model auto_arima --horizon 12
```

## evaluate

```bash
forecastbox evaluate --forecasts fc1.json fc2.json --actual actual.csv --tests dm mcs
```

## nowcast

```bash
forecastbox nowcast --data panel.csv --target pib --method dfm --factors 2
```

## monitor

```bash
forecastbox monitor --pipeline pipeline.yaml --actual actual.csv --alerts
```

## combine

```bash
forecastbox combine --forecasts fc1.json fc2.json fc3.json --method bma
```

<!-- TODO: Add detailed option descriptions -->
<!-- TODO: Add output examples -->
