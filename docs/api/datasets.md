---
title: "Datasets API"
description: "API reference for forecastbox.datasets — built-in example datasets for GDP, inflation, exchange rates, interest rates, and competition data"
---

# Datasets API Reference

!!! info "Module"
    **Import**: `from forecastbox.datasets import load_gdp, load_inflation, load_exchange_rate, load_interest_rate, load_macro_panel, load_m3_sample, load_nowcast_example`
    **Source**: `forecastbox/datasets/`

## Overview

The datasets module provides ready-to-use example datasets for tutorials, testing, and benchmarking. All loaders return pandas DataFrames (or dicts) with proper datetime indices and documented column names.

| Function | Series | Frequency | Source |
|----------|--------|-----------|--------|
| [`load_gdp()`](#load_gdp) | GDP growth (Brazil) | Quarterly | IBGE/SCN |
| [`load_inflation()`](#load_inflation) | IPCA inflation (Brazil) | Monthly | IBGE/SIDRA |
| [`load_interest_rate()`](#load_interest_rate) | Selic rate (Brazil) | Daily / Monthly | BCB/SGS |
| [`load_exchange_rate()`](#load_exchange_rate) | BRL/USD exchange rate | Daily | BCB/SGS |
| [`load_macro_panel()`](#load_macro_panel) | 20 macro indicators | Monthly | Various |
| [`load_m3_sample()`](#load_m3_sample) | M3 Competition sample | Mixed | Makridakis et al. (2000) |
| [`load_nowcast_example()`](#load_nowcast_example) | Nowcasting example data | Mixed | Simulated |

---

## load_gdp()

Load quarterly GDP growth rate for Brazil.

```python
load_gdp(
    start: str | None = None,
    end: str | None = None,
    real: bool = True,
    seasonally_adjusted: bool = True,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `str \| None` | `None` | Start date filter (e.g., `"2000Q1"`) — includes all if `None` |
| `end` | `str \| None` | `None` | End date filter (e.g., `"2024Q4"`) |
| `real` | `bool` | `True` | Real GDP (`True`) or nominal (`False`) |
| `seasonally_adjusted` | `bool` | `True` | Seasonally adjusted series |

**Returns**: `pd.DataFrame` with `PeriodIndex` (quarterly) and columns:

| Column | Type | Description |
|--------|------|-------------|
| `gdp_growth` | `float` | Quarter-over-quarter GDP growth rate (%) |
| `gdp_level` | `float` | GDP level index (2000Q1 = 100) |

### Example

```python
from forecastbox.datasets import load_gdp

df = load_gdp(start="2010Q1", end="2024Q4")
print(df.head())
#          gdp_growth  gdp_level
# 2010Q1        2.10     148.30
# 2010Q2        1.50     150.52
# 2010Q3        0.80     151.72
# 2010Q4        0.50     152.48
# 2011Q1        1.20     154.31
```

---

## load_inflation()

Load monthly IPCA inflation for Brazil.

```python
load_inflation(
    start: str | None = None,
    end: str | None = None,
    cumulative: bool = False,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `str \| None` | `None` | Start date filter (e.g., `"2000-01"`) |
| `end` | `str \| None` | `None` | End date filter |
| `cumulative` | `bool` | `False` | Include 12-month cumulative inflation column |

**Returns**: `pd.DataFrame` with `DatetimeIndex` (monthly) and columns:

| Column | Type | Description |
|--------|------|-------------|
| `ipca` | `float` | Monthly IPCA variation (%) |
| `ipca_12m` | `float` | 12-month cumulative IPCA (%) — only if `cumulative=True` |

### Example

```python
from forecastbox.datasets import load_inflation

df = load_inflation(start="2020-01", cumulative=True)
print(df.head())
#            ipca  ipca_12m
# 2020-01    0.21      4.19
# 2020-02    0.25      4.01
# 2020-03    0.07      3.30
# 2020-04   -0.31      2.40
# 2020-05   -0.38      1.88
```

---

## load_exchange_rate()

Load daily BRL/USD exchange rate (PTAX).

```python
load_exchange_rate(
    start: str | None = None,
    end: str | None = None,
    frequency: str = "daily",
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `str \| None` | `None` | Start date filter (e.g., `"2020-01-01"`) |
| `end` | `str \| None` | `None` | End date filter |
| `frequency` | `str` | `"daily"` | Frequency: `"daily"`, `"weekly"`, or `"monthly"` (averaged) |

**Returns**: `pd.DataFrame` with `DatetimeIndex` and columns:

| Column | Type | Description |
|--------|------|-------------|
| `ptax_buy` | `float` | PTAX buying rate (BRL per USD) |
| `ptax_sell` | `float` | PTAX selling rate (BRL per USD) |

### Example

```python
from forecastbox.datasets import load_exchange_rate

df = load_exchange_rate(start="2024-01-01", frequency="monthly")
print(df.head())
#            ptax_buy  ptax_sell
# 2024-01      4.92       4.93
# 2024-02      4.97       4.97
# 2024-03      4.98       4.98
# 2024-04      5.07       5.07
# 2024-05      5.12       5.12
```

---

## load_interest_rate()

Load the Brazilian Selic target rate.

```python
load_interest_rate(
    start: str | None = None,
    end: str | None = None,
    annualized: bool = True,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `str \| None` | `None` | Start date filter |
| `end` | `str \| None` | `None` | End date filter |
| `annualized` | `bool` | `True` | Annualized rate (`True`) or daily rate (`False`) |

**Returns**: `pd.DataFrame` with `DatetimeIndex` and columns:

| Column | Type | Description |
|--------|------|-------------|
| `selic` | `float` | Selic target rate (% p.a. if `annualized=True`) |

### Example

```python
from forecastbox.datasets import load_interest_rate

df = load_interest_rate(start="2020-01")
print(df.tail())
#            selic
# 2024-08    10.50
# 2024-09    10.75
# 2024-10    10.75
# 2024-11    11.25
# 2024-12    12.25
```

---

## load_macro_panel()

Load a panel of 20 monthly macroeconomic indicators for Brazil, suitable for nowcasting and factor model estimation.

```python
load_macro_panel(
    start: str | None = None,
    end: str | None = None,
    standardize: bool = False,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `str \| None` | `None` | Start date filter |
| `end` | `str \| None` | `None` | End date filter |
| `standardize` | `bool` | `False` | Standardize each series to zero mean and unit variance |

**Returns**: `pd.DataFrame` with `DatetimeIndex` (monthly) and 20 columns:

| Column | Description |
|--------|-------------|
| `ibc_br` | IBC-Br economic activity index |
| `industrial_production` | Industrial production index |
| `retail_sales` | Retail sales volume index |
| `services` | Services sector volume index |
| `employment` | Formal employment (CAGED) |
| `unemployment` | Unemployment rate (PNAD) |
| `ipca` | IPCA monthly variation |
| `igpm` | IGP-M monthly variation |
| `selic` | Selic target rate |
| `swap_360` | 360-day swap rate |
| `exchange_rate` | BRL/USD PTAX |
| `exports` | Exports (USD millions) |
| `imports` | Imports (USD millions) |
| `trade_balance` | Trade balance (USD millions) |
| `m1` | Monetary aggregate M1 |
| `credit` | Total credit / GDP |
| `confidence_consumer` | Consumer confidence index |
| `confidence_industry` | Industrial confidence index |
| `commodities` | CRB commodity index |
| `vix` | VIX volatility index |

!!! tip
    This panel is designed for use with `DynamicFactorModel` and `MIDASRegressor` in the [Nowcasting API](nowcasting.md). The ragged-edge pattern (different publication lags) makes it a realistic nowcasting exercise.

### Example

```python
from forecastbox.datasets import load_macro_panel

panel = load_macro_panel(start="2010-01", standardize=True)
print(panel.shape)
# (180, 20)

print(panel.columns.tolist()[:5])
# ['ibc_br', 'industrial_production', 'retail_sales', 'services', 'employment']
```

---

## load_m3_sample()

Load a sample of time series from the M3 Competition (Makridakis et al., 2000) for benchmarking.

```python
load_m3_sample(
    category: str | None = None,
    n: int = 50,
    seed: int = 42,
) -> dict[str, dict]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | `str \| None` | `None` | Filter by category: `"yearly"`, `"quarterly"`, `"monthly"`, `"other"` — all if `None` |
| `n` | `int` | `50` | Number of series to sample |
| `seed` | `int` | `42` | Random seed for reproducible sampling |

**Returns**: `dict[str, dict]` where each key is a series ID and each value is a dict:

| Key | Type | Description |
|-----|------|-------------|
| `"train"` | `NDArray[np.float64]` | Training portion of the series |
| `"test"` | `NDArray[np.float64]` | Holdout test portion |
| `"frequency"` | `str` | Series frequency (`"M"`, `"Q"`, `"Y"`) |
| `"category"` | `str` | M3 category |
| `"horizon"` | `int` | Official forecast horizon |

### Example

```python
from forecastbox.datasets import load_m3_sample

m3 = load_m3_sample(category="monthly", n=10)
print(list(m3.keys())[:3])
# ['M1001', 'M1002', 'M1003']

series = m3["M1001"]
print(f"Train: {len(series['train'])}, Test: {len(series['test'])}, H: {series['horizon']}")
# Train: 108, Test: 18, H: 18
```

---

## load_nowcast_example()

Load a pre-configured nowcasting example with mixed-frequency data and a ragged-edge structure.

```python
load_nowcast_example(
    vintage: str = "latest",
) -> dict[str, Any]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vintage` | `str` | `"latest"` | Data vintage: `"latest"`, `"2024-01"`, `"2024-02"`, `"2024-03"` |

**Returns**: `dict[str, Any]` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `"target"` | `pd.Series` | Quarterly GDP growth (target variable) |
| `"monthly"` | `pd.DataFrame` | Monthly indicators with ragged edges |
| `"daily"` | `pd.DataFrame` | Daily financial variables |
| `"calendar"` | `pd.DataFrame` | Release calendar (variable, publication date) |
| `"actual"` | `float` | Realized GDP growth for the nowcast quarter |

!!! note
    Different vintages simulate the real-time information flow within a quarter. `"2024-01"` has only January releases, `"2024-02"` adds February data, etc. This is designed to demonstrate nowcast evolution as in [`plot_nowcast_evolution()`](visualization.md#plot_nowcast_evolution).

### Example

```python
from forecastbox.datasets import load_nowcast_example

data = load_nowcast_example(vintage="2024-02")
print(data["monthly"].shape)
# (60, 15)

print(f"Target quarter actual: {data['actual']:.2f}%")
# Target quarter actual: 0.80%

# Use with DFM nowcaster
from forecastbox.nowcasting import DynamicFactorModel
dfm = DynamicFactorModel(n_factors=3)
dfm.fit(data["monthly"], target=data["target"])
nowcast = dfm.nowcast()
```

---

## See Also

- [Core API](core.md) — `Forecast` containers returned by models
- [Nowcasting API](nowcasting.md) — Models that consume `load_macro_panel()` and `load_nowcast_example()` data
- [Evaluation API](evaluation.md) — Evaluate forecasts against `load_m3_sample()` test sets
- [Tutorials](../tutorials/index.md) — Step-by-step guides using these datasets
