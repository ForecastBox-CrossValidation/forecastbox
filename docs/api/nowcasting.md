---
title: "Nowcasting API"
description: "API reference for forecastbox.nowcasting — DynamicFactorModel, BridgeEquation, MIDAS, NewsDecomposition, VintageManager"
---

# Nowcasting API Reference

!!! info "Module"
    **Import**: `from forecastbox.nowcasting import DynamicFactorModel, BridgeEquation, MIDAS, NewsDecomposition, VintageManager`
    **Source**: `forecastbox/nowcasting/`

## Overview

The nowcasting module implements methods for predicting the current state of the economy using mixed-frequency and real-time data. All models follow a **fit → nowcast** pattern.

| Class | Method | Reference |
|-------|--------|-----------|
| [`DynamicFactorModel`](#dynamicfactormodel) | Dynamic Factor Model via Kalman filter | Giannone, Reichlin & Small (2008) |
| [`BridgeEquation`](#bridgeequation) | Bridge equations for temporal aggregation | Baffigi, Golinelli & Parigi (2004) |
| [`MIDAS`](#midas) | Mixed Data Sampling regression | Ghysels, Santa-Clara & Valkanov (2004) |
| [`NewsDecomposition`](#newsdecomposition) | Nowcast revision decomposition | Bańbura & Modugno (2014) |
| [`VintageManager`](#vintagemanager) | Real-time data vintage management | — |

---

## DynamicFactorModel

Dynamic Factor Model (DFM) estimated via the EM algorithm with Kalman smoother. Extracts latent factors from a large panel of mixed-frequency indicators and projects them onto a target variable.

!!! warning "Dependency: kalmanbox"
    `DynamicFactorModel` requires the **kalmanbox** package for Kalman filter/smoother operations. Install it with:

    ```bash
    pip install kalmanbox
    ```

    If kalmanbox is not installed, importing `DynamicFactorModel` will raise an `ImportError` with installation instructions.

### Constructor

```python
DynamicFactorModel(
    n_factors: int = 2,
    factor_lags: int = 1,
    em_iterations: int = 100,
    em_tolerance: float = 1e-6,
    handle_missing: str = "em",
    standardize: bool = True,
) -> DynamicFactorModel
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_factors` | `int` | `2` | Number of latent factors to extract |
| `factor_lags` | `int` | `1` | Number of lags in the factor VAR transition equation |
| `em_iterations` | `int` | `100` | Maximum EM iterations for parameter estimation |
| `em_tolerance` | `float` | `1e-6` | Convergence tolerance for the EM log-likelihood |
| `handle_missing` | `str` | `"em"` | Missing data treatment: `"em"` (EM-based) or `"drop"` |
| `standardize` | `bool` | `True` | Standardize indicators before factor extraction |

### Key Attributes (after `fit`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `factors_` | `NDArray[np.float64]` | Extracted factors `(T, n_factors)` |
| `loadings_` | `NDArray[np.float64]` | Factor loadings matrix `(N, n_factors)` |
| `transition_` | `NDArray[np.float64]` | Factor VAR transition matrix |
| `log_likelihood_` | `float` | Log-likelihood at convergence |
| `n_iterations_` | `int` | EM iterations until convergence |
| `is_fitted_` | `bool` | Whether `fit()` has been called |

### Methods

##### `fit(X)`

Estimate the DFM parameters from the indicator panel.

```python
dfm.fit(
    X: NDArray[np.float64] | pd.DataFrame,
) -> DynamicFactorModel
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `NDArray[np.float64] \| pd.DataFrame` | *required* | Indicator panel `(T, N)`. May contain `NaN` for ragged edges |

**Returns**: `self` (for method chaining)

##### `nowcast(target, X_new)`

Produce a nowcast for the target variable using new indicator data.

```python
dfm.nowcast(
    target: NDArray[np.float64] | pd.Series,
    X_new: NDArray[np.float64] | pd.DataFrame | None = None,
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target` | `NDArray[np.float64] \| pd.Series` | *required* | Low-frequency target variable (e.g., quarterly GDP) |
| `X_new` | `NDArray[np.float64] \| pd.DataFrame \| None` | `None` | Updated indicator panel. If `None`, uses the latest available data from `fit()` |

**Returns**: [`Forecast`](core.md#forecast)

##### `get_factors()`

Return the estimated latent factors.

```python
dfm.get_factors() -> pd.DataFrame
```

**Returns**: `pd.DataFrame` with factors as columns.

### Example

```python
from forecastbox.nowcasting import DynamicFactorModel

# Panel of monthly indicators (some with missing recent data)
# indicators: pd.DataFrame with columns like PMI, industrial_prod, retail_sales, ...

dfm = DynamicFactorModel(n_factors=3, factor_lags=2, em_iterations=200)
dfm.fit(indicators)

# Nowcast quarterly GDP
nowcast = dfm.nowcast(target=gdp_quarterly, X_new=indicators_updated)

print(f"GDP nowcast: {nowcast.point[-1]:.3f}")
# GDP nowcast: 0.024
```

!!! tip "Choosing the number of factors"
    Use the Bai-Ng information criteria (IC~p1~, IC~p2~, IC~p3~) to determine `n_factors`. As a rule of thumb, 2–5 factors typically explain most of the common variation in macroeconomic panels.

---

## BridgeEquation

Bridge equations link a low-frequency target (e.g., quarterly GDP) to high-frequency indicators (e.g., monthly industrial production) via temporal aggregation.

### Constructor

```python
BridgeEquation(
    aggregation: str = "mean",
    indicator_forecast: str = "ar",
    ar_order: int | str = "aic",
    include_intercept: bool = True,
) -> BridgeEquation
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregation` | `str` | `"mean"` | Temporal aggregation rule: `"mean"` (flow/average), `"sum"` (cumulative flow), `"last"` (stock) |
| `indicator_forecast` | `str` | `"ar"` | Method to forecast missing months of indicators: `"ar"` (autoregressive), `"rw"` (random walk), `"none"` (assume available) |
| `ar_order` | `int \| str` | `"aic"` | AR order for indicator forecasting. `"aic"` or `"bic"` for automatic selection |
| `include_intercept` | `bool` | `True` | Include intercept in the bridge regression |

### Key Attributes (after `fit`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `coefficients_` | `NDArray[np.float64]` | Bridge equation regression coefficients |
| `r_squared_` | `float` | In-sample R² of the bridge equation |
| `indicator_models_` | `dict[str, Any]` | Fitted AR models for each indicator |
| `is_fitted_` | `bool` | Whether `fit()` has been called |

### Methods

##### `fit(y_quarterly, X_monthly)`

Estimate the bridge equation.

```python
bridge.fit(
    y_quarterly: NDArray[np.float64] | pd.Series,
    X_monthly: NDArray[np.float64] | pd.DataFrame,
) -> BridgeEquation
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_quarterly` | `NDArray[np.float64] \| pd.Series` | *required* | Quarterly target variable |
| `X_monthly` | `NDArray[np.float64] \| pd.DataFrame` | *required* | Monthly indicator(s) |

**Returns**: `self`

##### `nowcast(X_monthly_new)`

Produce a nowcast using updated monthly data.

```python
bridge.nowcast(
    X_monthly_new: NDArray[np.float64] | pd.DataFrame,
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_monthly_new` | `NDArray[np.float64] \| pd.DataFrame` | *required* | Updated monthly indicators (may have missing trailing months) |

**Returns**: [`Forecast`](core.md#forecast)

### Example

```python
from forecastbox.nowcasting import BridgeEquation

bridge = BridgeEquation(aggregation="mean", indicator_forecast="ar", ar_order="bic")
bridge.fit(y_quarterly=gdp_q, X_monthly=industrial_prod_m)

# Nowcast current quarter (only 2 of 3 months available)
nowcast = bridge.nowcast(X_monthly_new=industrial_prod_latest)

print(f"Q1 GDP nowcast: {nowcast.point[-1]:.3f}")
# Q1 GDP nowcast: 0.018
```

---

## MIDAS

Mixed Data Sampling regression for nowcasting with mixed-frequency data. Uses polynomial-distributed lag weighting functions to parsimoniously map high-frequency regressors to a low-frequency dependent variable.

### Constructor

```python
MIDAS(
    weight_function: str = "beta",
    n_lags: int = 12,
    polynomial_order: int = 2,
    optimizer: str = "L-BFGS-B",
    ar_terms: int = 0,
) -> MIDAS
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_function` | `str` | `"beta"` | Weighting function: `"beta"` (Beta polynomial), `"exp_almon"` (Exponential Almon), `"step"` (step function), `"unrestricted"` (U-MIDAS) |
| `n_lags` | `int` | `12` | Number of high-frequency lags |
| `polynomial_order` | `int` | `2` | Order of the polynomial in the weighting function |
| `optimizer` | `str` | `"L-BFGS-B"` | Optimizer for nonlinear weight estimation |
| `ar_terms` | `int` | `0` | Autoregressive terms for the low-frequency variable (MIDAS-AR) |

### Key Attributes (after `fit`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `NDArray[np.float64]` | Estimated high-frequency lag weights |
| `weight_params_` | `NDArray[np.float64]` | Polynomial parameters of the weighting function |
| `coefficients_` | `NDArray[np.float64]` | Regression coefficients (intercept, slope, AR) |
| `is_fitted_` | `bool` | Whether `fit()` has been called |

### Methods

##### `fit(y_low, X_high)`

Estimate the MIDAS regression.

```python
midas.fit(
    y_low: NDArray[np.float64] | pd.Series,
    X_high: NDArray[np.float64] | pd.DataFrame,
) -> MIDAS
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y_low` | `NDArray[np.float64] \| pd.Series` | *required* | Low-frequency dependent variable |
| `X_high` | `NDArray[np.float64] \| pd.DataFrame` | *required* | High-frequency regressor(s) |

**Returns**: `self`

##### `nowcast(X_high_new)`

Produce a nowcast using updated high-frequency data.

```python
midas.nowcast(
    X_high_new: NDArray[np.float64] | pd.DataFrame,
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_high_new` | `NDArray[np.float64] \| pd.DataFrame` | *required* | Updated high-frequency regressor(s) |

**Returns**: [`Forecast`](core.md#forecast)

##### `plot_weights()`

Visualize the estimated lag weighting function.

```python
midas.plot_weights() -> matplotlib.Figure
```

### MIDAS Weighting Functions

The weighting function $w(k; \boldsymbol{\theta})$ maps the $k$-th high-frequency lag to a weight:

=== "Beta"

    $$w(k; \theta_1, \theta_2) = \frac{x^{\theta_1 - 1}(1-x)^{\theta_2 - 1}}{\sum_{j=1}^{K} x_j^{\theta_1 - 1}(1-x_j)^{\theta_2 - 1}}, \quad x = \frac{k}{K}$$

=== "Exponential Almon"

    $$w(k; \theta_1, \theta_2) = \frac{\exp(\theta_1 k + \theta_2 k^2)}{\sum_{j=1}^{K} \exp(\theta_1 j + \theta_2 j^2)}$$

=== "U-MIDAS"

    $$w(k) = \beta_k \quad \text{(unrestricted, one parameter per lag)}$$

### Example

```python
from forecastbox.nowcasting import MIDAS

midas = MIDAS(weight_function="beta", n_lags=12, ar_terms=1)
midas.fit(y_low=gdp_quarterly, X_high=industrial_prod_monthly)

# Nowcast with latest monthly data
nowcast = midas.nowcast(X_high_new=industrial_prod_latest)

print(f"GDP nowcast: {nowcast.point[-1]:.4f}")
# GDP nowcast: 0.0215

# Inspect the learned lag structure
midas.plot_weights()
```

!!! note "U-MIDAS vs. restricted MIDAS"
    When the frequency mismatch is small (e.g., monthly to quarterly, `n_lags ≤ 12`), unrestricted MIDAS (`weight_function="unrestricted"`) can outperform polynomial specifications. For larger frequency ratios (e.g., daily to monthly), use `"beta"` or `"exp_almon"` to avoid overfitting.

---

## NewsDecomposition

Decomposes nowcast revisions into contributions from individual data releases ("news"). Quantifies how much each new data point moves the nowcast.

### Constructor

```python
NewsDecomposition(
    model: DynamicFactorModel | None = None,
) -> NewsDecomposition
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `DynamicFactorModel \| None` | `None` | A fitted DFM. If provided, uses its structure for decomposition |

### Methods

##### `decompose(nowcast_old, nowcast_new, data_old, data_new)`

Decompose the nowcast revision into news components.

```python
news.decompose(
    nowcast_old: Forecast,
    nowcast_new: Forecast,
    data_old: pd.DataFrame,
    data_new: pd.DataFrame,
) -> NewsResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nowcast_old` | [`Forecast`](core.md#forecast) | *required* | Previous nowcast |
| `nowcast_new` | [`Forecast`](core.md#forecast) | *required* | Updated nowcast |
| `data_old` | `pd.DataFrame` | *required* | Data vintage used for `nowcast_old` |
| `data_new` | `pd.DataFrame` | *required* | Data vintage used for `nowcast_new` |

**Returns**: `NewsResult`

### NewsResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `revision` | `float` | Total nowcast revision (`nowcast_new - nowcast_old`) |
| `contributions` | `pd.DataFrame` | Per-variable contribution to the revision |
| `surprises` | `pd.DataFrame` | Data surprises (actual minus expected) |
| `weights` | `pd.DataFrame` | Impact weights (how surprises map to the target) |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Sorted contributions from largest to smallest |
| `plot()` | `matplotlib.Figure` | Waterfall chart of contributions |

### Example

```python
from forecastbox.nowcasting import NewsDecomposition

news = NewsDecomposition(model=fitted_dfm)

result = news.decompose(
    nowcast_old=nowcast_march,
    nowcast_new=nowcast_april,
    data_old=vintage_march,
    data_new=vintage_april,
)

print(result.summary())
#                   surprise  weight  contribution
# industrial_prod     0.012   0.450         0.005
# retail_sales        0.008   0.320         0.003
# pmi                -0.003   0.180        -0.001
# Total                                     0.007

result.plot(title="Nowcast Revision: March → April")
```

---

## VintageManager

Manages real-time data vintages for pseudo real-time nowcasting exercises and revision analysis.

### Constructor

```python
VintageManager(
    name: str = "",
) -> VintageManager
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `""` | Label for the vintage collection |

### Methods

##### `add_vintage(date, data)`

Register a data vintage.

```python
vm.add_vintage(
    date: str | pd.Timestamp,
    data: pd.DataFrame,
) -> VintageManager
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date` | `str \| pd.Timestamp` | *required* | Vintage date (when the data was available) |
| `data` | `pd.DataFrame` | *required* | The data snapshot at that date |

**Returns**: `self`

##### `get_vintage(date)`

Retrieve a specific vintage.

```python
vm.get_vintage(
    date: str | pd.Timestamp,
) -> pd.DataFrame
```

**Returns**: `pd.DataFrame` — The data snapshot at the requested date.

**Raises**: `KeyError` if the vintage date is not found.

##### `compare(date1, date2)`

Compare two vintages and report revisions.

```python
vm.compare(
    date1: str | pd.Timestamp,
    date2: str | pd.Timestamp,
) -> pd.DataFrame
```

**Returns**: `pd.DataFrame` with columns `vintage_1`, `vintage_2`, `revision` for each variable and period.

##### `list_vintages()`

List all available vintage dates.

```python
vm.list_vintages() -> list[pd.Timestamp]
```

### Example

```python
from forecastbox.nowcasting import VintageManager

vm = VintageManager(name="GDP Indicators")

# Add monthly data snapshots
vm.add_vintage("2024-01-15", data=snapshot_jan)
vm.add_vintage("2024-02-15", data=snapshot_feb)
vm.add_vintage("2024-03-15", data=snapshot_mar)

# Compare revisions between vintages
revisions = vm.compare("2024-01-15", "2024-03-15")
print(revisions.head())
#                     vintage_1  vintage_2  revision
# industrial_prod_12    1.200      1.250     0.050
# retail_sales_12       0.800      0.780    -0.020

# Use in pseudo real-time evaluation
for date in vm.list_vintages():
    vintage = vm.get_vintage(date)
    dfm.fit(vintage)
    nowcast = dfm.nowcast(target=gdp_q)
```

---

## See Also

- [Nowcasting Theory](../theory/nowcasting-theory.md) — DFM, bridge equations, and nowcasting foundations
- [MIDAS Theory](../theory/midas-theory.md) — Mixed Data Sampling mathematical framework
- [News Decomposition Theory](../theory/news_decomposition.md) — News decomposition derivation
- [Nowcasting User Guide](../user-guide/nowcasting/index.md) — Tutorials and worked examples
- [Core API](core.md) — `Forecast` and `DataVintage` data structures
