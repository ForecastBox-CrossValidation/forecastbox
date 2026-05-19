---
title: "CLI Reference"
description: "API reference for the forecastbox command-line interface — forecast, combine, evaluate, nowcast, pipeline, and experiment commands"
---

# CLI Reference

!!! info "Module"
    **Entry point**: `forecastbox` (installed via `pip install forecastbox`)
    **Source**: `forecastbox/cli/`

## Overview

The `forecastbox` CLI provides command-line access to all major workflows — forecasting, combination, evaluation, nowcasting, pipeline execution, and experiment management.

```
forecastbox <command> [options]
```

| Command | Description |
|---------|-------------|
| [`forecast`](#forecastbox-forecast) | Generate a forecast from data |
| [`combine`](#forecastbox-combine) | Combine multiple forecasts |
| [`evaluate`](#forecastbox-evaluate) | Evaluate forecast accuracy |
| [`nowcast`](#forecastbox-nowcast) | Generate a nowcast |
| [`pipeline run`](#forecastbox-pipeline-run) | Execute a pipeline from config |
| [`experiment list`](#forecastbox-experiment-list) | List saved experiments |
| [`experiment compare`](#forecastbox-experiment-compare) | Compare experiments |

### Global Options

| Flag | Description |
|------|-------------|
| `--version` | Show forecastbox version |
| `--verbose`, `-v` | Increase output verbosity (repeat for more: `-vv`) |
| `--quiet`, `-q` | Suppress non-error output |
| `--config <path>` | Path to a YAML/TOML config file for defaults |
| `--help`, `-h` | Show help for a command |

---

## forecastbox forecast

Generate a point forecast with optional prediction intervals.

```bash
forecastbox forecast \
    --data <path> \
    --model <model> \
    --horizon <int> \
    --output <path>
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data` | `path` | *required* | Path to input data (CSV or Parquet with datetime index) |
| `--model` | `str` | `"auto"` | Model name: `"arima"`, `"ets"`, `"var"`, `"theta"`, `"auto"` |
| `--horizon`, `-h` | `int` | *required* | Forecast horizon (number of periods) |
| `--output`, `-o` | `path` | `stdout` | Output path for forecast (CSV or Parquet) |
| `--column`, `-c` | `str` | `None` | Target column name (auto-detected if single column) |
| `--frequency` | `str` | `None` | Data frequency (`"M"`, `"Q"`, `"D"`) — auto-detected if not given |
| `--intervals` | `flag` | `False` | Include 80% and 95% prediction intervals |
| `--format` | `str` | `"csv"` | Output format: `"csv"`, `"parquet"`, `"json"` |

### Example

```bash
# Auto-select model and forecast 12 months ahead
forecastbox forecast \
    --data data/ipca.csv \
    --model auto \
    --horizon 12 \
    --intervals \
    --output forecasts/ipca_forecast.csv

# ARIMA forecast with specific column
forecastbox forecast \
    --data data/macro.csv \
    --column gdp_growth \
    --model arima \
    --horizon 4 \
    --output forecasts/gdp.parquet \
    --format parquet
```

---

## forecastbox combine

Combine multiple forecast files into a single combined forecast.

```bash
forecastbox combine \
    --forecasts <path1> <path2> ... \
    --method <method> \
    --output <path>
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--forecasts` | `path...` | *required* | Paths to forecast files (CSV or Parquet) |
| `--method`, `-m` | `str` | `"equal"` | Combination method: `"equal"`, `"inverse_mse"`, `"bma"`, `"aic"`, `"bic"`, `"granger_ramanathan"` |
| `--actual` | `path` | `None` | Path to actual values (required for `"inverse_mse"`, `"bma"`) |
| `--output`, `-o` | `path` | `stdout` | Output path for combined forecast |
| `--weights-output` | `path` | `None` | Save combination weights to this path |

### Example

```bash
# Equal-weight combination
forecastbox combine \
    --forecasts forecasts/arima.csv forecasts/ets.csv forecasts/var.csv \
    --method equal \
    --output forecasts/combined.csv

# BMA combination with actuals for weight estimation
forecastbox combine \
    --forecasts forecasts/arima.csv forecasts/ets.csv forecasts/var.csv \
    --method bma \
    --actual data/actual.csv \
    --output forecasts/bma.csv \
    --weights-output weights/bma_weights.csv
```

---

## forecastbox evaluate

Evaluate one or more forecasts against realized values.

```bash
forecastbox evaluate \
    --actual <path> \
    --forecasts <path1> <path2> ... \
    --metrics <metric1> <metric2> ...
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--actual` | `path` | *required* | Path to realized values |
| `--forecasts` | `path...` | *required* | Paths to forecast files |
| `--metrics` | `str...` | `rmse mae mape` | Metrics to compute: `rmse`, `mae`, `mape`, `mase`, `smape`, `crps` |
| `--output`, `-o` | `path` | `stdout` | Output path for evaluation results (CSV) |
| `--test` | `str` | `None` | Statistical test: `"dm"` (Diebold-Mariano), `"mcs"` (Model Confidence Set) |
| `--training` | `path` | `None` | Training series (required for `mase`) |

### Example

```bash
# Basic evaluation
forecastbox evaluate \
    --actual data/actual.csv \
    --forecasts forecasts/arima.csv forecasts/ets.csv forecasts/var.csv \
    --metrics rmse mae mape
# Output:
#        rmse   mae  mape
# arima  0.42  0.35  2.10
# ets    0.45  0.37  2.30
# var    0.39  0.31  1.95

# With Diebold-Mariano test (pairwise against first forecast)
forecastbox evaluate \
    --actual data/actual.csv \
    --forecasts forecasts/arima.csv forecasts/ets.csv \
    --test dm \
    --output evaluation/dm_results.csv
```

---

## forecastbox nowcast

Generate a nowcast using mixed-frequency data.

```bash
forecastbox nowcast \
    --data <path> \
    --target <path> \
    --model <model> \
    --output <path>
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data` | `path` | *required* | Path to high-frequency indicator panel (CSV or Parquet) |
| `--target` | `path` | *required* | Path to low-frequency target variable |
| `--model`, `-m` | `str` | `"dfm"` | Nowcasting model: `"dfm"`, `"midas"`, `"bridge"`, `"umidas"` |
| `--output`, `-o` | `path` | `stdout` | Output path for nowcast |
| `--n-factors` | `int` | `3` | Number of factors (DFM only) |
| `--lags` | `int` | `None` | Number of lags in the model |
| `--news` | `flag` | `False` | Compute news decomposition |
| `--news-output` | `path` | `None` | Output path for news decomposition |

### Example

```bash
# DFM nowcast with 4 factors
forecastbox nowcast \
    --data data/monthly_panel.csv \
    --target data/gdp_quarterly.csv \
    --model dfm \
    --n-factors 4 \
    --output nowcasts/gdp_nowcast.csv

# MIDAS with news decomposition
forecastbox nowcast \
    --data data/monthly_panel.csv \
    --target data/gdp_quarterly.csv \
    --model midas \
    --news \
    --news-output nowcasts/news.csv \
    --output nowcasts/midas_nowcast.csv
```

---

## forecastbox pipeline run

Execute a full forecasting pipeline from a YAML configuration file.

```bash
forecastbox pipeline run --config <path>
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | `path` | *required* | Path to pipeline YAML config file |
| `--dry-run` | `flag` | `False` | Validate config without executing |
| `--step` | `str` | `None` | Run only a specific step by name |
| `--output-dir` | `path` | `None` | Override the output directory |
| `--parallel` | `flag` | `False` | Run independent steps in parallel |

### Example

```bash
# Run full pipeline
forecastbox pipeline run --config pipelines/gdp_pipeline.yaml

# Dry run to validate config
forecastbox pipeline run --config pipelines/gdp_pipeline.yaml --dry-run

# Run only the evaluation step
forecastbox pipeline run --config pipelines/gdp_pipeline.yaml --step evaluate
```

!!! tip "Pipeline Config Format"
    See the [Pipeline API](pipeline.md) for the full YAML configuration schema, including step definitions, dependencies, and parameter overrides.

---

## forecastbox experiment list

List all saved experiments.

```bash
forecastbox experiment list [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--store` | `path` | `experiments.db` | Path to experiment store |
| `--tag` | `str` | `None` | Filter by tag |
| `--format` | `str` | `"table"` | Output format: `"table"`, `"json"`, `"csv"` |

### Example

```bash
forecastbox experiment list
# Name                Created      Tags           Best (RMSE)
# gdp_q4_2024        2024-11-15   gdp,quarterly  bma (0.31)
# inflation_dec_2024  2024-12-01   ipca,monthly   arima (0.18)

forecastbox experiment list --tag gdp --format json
```

---

## forecastbox experiment compare

Compare metrics across experiments.

```bash
forecastbox experiment compare <name1> <name2> ... [options]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `<names>` | `str...` | *required* | Experiment names to compare |
| `--metric` | `str` | `"rmse"` | Metric for comparison |
| `--output`, `-o` | `path` | `stdout` | Output path for comparison table |

### Example

```bash
forecastbox experiment compare gdp_q4_2024 gdp_q3_2024 --metric rmse
# Experiment      Best Model  RMSE
# gdp_q4_2024     bma         0.31
# gdp_q3_2024     var         0.35
```

---

## Integration with Cron and Shell Scripts

The CLI is designed for automation via cron jobs and shell scripts.

### Cron Example — Daily Nowcast Update

```bash
# crontab -e
# Run nowcast update every weekday at 9:00 AM
0 9 * * 1-5 forecastbox nowcast \
    --data /data/daily_panel.csv \
    --target /data/gdp.csv \
    --model dfm \
    --output /output/nowcast_$(date +\%Y\%m\%d).csv \
    --news \
    --news-output /output/news_$(date +\%Y\%m\%d).csv
```

### Shell Script Example — Monthly Forecast Pipeline

```bash
#!/bin/bash
# monthly_forecast.sh — Run full forecast pipeline and generate report
set -euo pipefail

DATA_DIR="/data/macro"
OUTPUT_DIR="/output/$(date +%Y%m)"
mkdir -p "$OUTPUT_DIR"

# Step 1: Generate forecasts
forecastbox forecast \
    --data "$DATA_DIR/ipca.csv" \
    --model auto \
    --horizon 12 \
    --intervals \
    --output "$OUTPUT_DIR/forecast.csv"

# Step 2: Evaluate against history
forecastbox evaluate \
    --actual "$DATA_DIR/actual.csv" \
    --forecasts "$OUTPUT_DIR/forecast.csv" \
    --metrics rmse mae mape \
    --output "$OUTPUT_DIR/evaluation.csv"

# Step 3: Run pipeline for combination
forecastbox pipeline run \
    --config pipelines/monthly_ipca.yaml \
    --output-dir "$OUTPUT_DIR"

echo "Pipeline complete. Results in $OUTPUT_DIR"
```

!!! note
    All CLI commands return exit code `0` on success and non-zero on failure, making them safe for `set -e` scripts and CI/CD pipelines.

---

## See Also

- [Core API](core.md) — Data structures used by all commands
- [Pipeline API](pipeline.md) — YAML config schema for `pipeline run`
- [Experiment API](experiment.md) — Programmatic experiment tracking
- [Datasets API](datasets.md) — Built-in data for testing CLI workflows
