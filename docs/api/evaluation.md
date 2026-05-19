---
title: "Evaluation API"
description: "API reference for forecastbox.evaluation and forecastbox.metrics — statistical tests, error metrics, and cross-validation"
---

# Evaluation API Reference

!!! info "Module"
    **Import**: `from forecastbox.evaluation import diebold_mariano, model_confidence_set, giacomini_white, mincer_zarnowitz, encompassing_test`
    **Import**: `from forecastbox.metrics import mae, rmse, mape, mase, smape, crps, log_score`
    **Import**: `from forecastbox.cv import expanding_window_cv, rolling_window_cv, blocked_cv`
    **Source**: `forecastbox/evaluation/`, `forecastbox/metrics/`, `forecastbox/cv/`

## Overview

The evaluation API spans three submodules:

- **`evaluation`** — Pairwise and multi-model statistical tests for forecast comparison
- **`metrics`** — Point and probabilistic error metrics
- **`cv`** — Time-series cross-validation strategies

| Function | Category | Reference |
|----------|----------|-----------|
| [`diebold_mariano()`](#diebold_mariano) | Pairwise test | Diebold & Mariano (1995), Harvey et al. (1997) |
| [`model_confidence_set()`](#model_confidence_set) | Multi-model test | Hansen, Lunde & Nason (2011) |
| [`giacomini_white()`](#giacomini_white) | Conditional test | Giacomini & White (2006) |
| [`mincer_zarnowitz()`](#mincer_zarnowitz) | Calibration test | Mincer & Zarnowitz (1969) |
| [`encompassing_test()`](#encompassing_test) | Encompassing test | Fair & Shiller (1990) |
| [`evaluate()`](#evaluate) | Multi-metric evaluation | — |

---

## evaluate()

Convenience function for evaluating one or more forecasts against actuals using multiple metrics.

```python
from forecastbox.evaluation import evaluate

results = evaluate(
    actual: NDArray[np.float64],
    forecasts: dict[str, NDArray[np.float64]],
    metrics: tuple[str, ...] = ("mae", "rmse", "mape", "mase"),
    training_series: NDArray[np.float64] | None = None,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `NDArray[np.float64]` | *required* | Realized values |
| `forecasts` | `dict[str, NDArray[np.float64]]` | *required* | Named forecasts (`{name: predictions}`) |
| `metrics` | `tuple[str, ...]` | `("mae", "rmse", "mape", "mase")` | Metrics to compute |
| `training_series` | `NDArray[np.float64] \| None` | `None` | Training data (required for `"mase"`) |

**Returns**: `pd.DataFrame` with models as rows and metrics as columns.

```python
from forecastbox.evaluation import evaluate

df = evaluate(
    actual=y_test,
    forecasts={"ARIMA": f1, "ETS": f2, "VAR": f3},
    metrics=("rmse", "mae", "mape"),
)
print(df)
#        rmse   mae  mape
# ARIMA  0.42  0.35  2.10
# ETS    0.45  0.37  2.30
# VAR    0.39  0.31  1.95
```

---

## Point Metrics

!!! info "Module"
    **Import**: `from forecastbox.metrics import mae, rmse, mape, mase, me, smape`
    **Source**: `forecastbox/metrics/point_metrics.py`

### `mae(actual, predicted)`

Mean Absolute Error.

$$\text{MAE} = \frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|$$

```python
mae(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float
```

### `rmse(actual, predicted)`

Root Mean Squared Error.

$$\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2}$$

```python
rmse(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float
```

### `mape(actual, predicted)`

Mean Absolute Percentage Error (in %).

$$\text{MAPE} = \frac{100}{T} \sum_{t=1}^{T} \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$

```python
mape(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float
```

!!! warning
    MAPE is undefined when any actual value equals zero. Use sMAPE or MASE for series that cross zero.

### `smape(actual, predicted)`

Symmetric Mean Absolute Percentage Error (in %).

$$\text{sMAPE} = \frac{100}{T} \sum_{t=1}^{T} \frac{2|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}$$

```python
smape(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float
```

### `mase(actual, predicted, training_series)`

Mean Absolute Scaled Error (Hyndman & Koehler, 2006). Scale-free metric comparing forecast error to in-sample naive forecast error.

$$\text{MASE} = \frac{\frac{1}{T}\sum_{t=1}^{T}|y_t - \hat{y}_t|}{\frac{1}{n-1}\sum_{i=2}^{n}|z_i - z_{i-1}|}$$

where $z_i$ is the training series.

```python
mase(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
    training_series: NDArray[np.float64] | list[float],
) -> float
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `NDArray[np.float64] \| list[float]` | *required* | Realized values |
| `predicted` | `NDArray[np.float64] \| list[float]` | *required* | Forecast values |
| `training_series` | `NDArray[np.float64] \| list[float]` | *required* | In-sample data for naive scaling |

### `me(actual, predicted)`

Mean Error (forecast bias).

$$\text{ME} = \frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)$$

```python
me(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float
```

### `mfe(actual, predicted)`

Mean Forecast Error (alias for ME).

```python
mfe(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float
```

### `theil_u1(actual, predicted)`

Theil U1 inequality coefficient. Ranges from 0 (perfect) to 1 (worst).

$$U_1 = \frac{\sqrt{\frac{1}{T}\sum(y_t - \hat{y}_t)^2}}{\sqrt{\frac{1}{T}\sum y_t^2} + \sqrt{\frac{1}{T}\sum \hat{y}_t^2}}$$

```python
theil_u1(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float
```

### `theil_u2(actual, predicted, naive)`

Theil U2 coefficient. Measures forecast accuracy relative to a naive (random walk) forecast. Values $< 1$ indicate the forecast beats the naive benchmark.

```python
theil_u2(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
    naive: NDArray[np.float64] | list[float] | None = None,
) -> float
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `NDArray[np.float64] \| list[float]` | *required* | Realized values |
| `predicted` | `NDArray[np.float64] \| list[float]` | *required* | Forecast values |
| `naive` | `NDArray[np.float64] \| list[float] \| None` | `None` | Naive forecast (random walk if `None`) |

### Example — Point Metrics

```python
import numpy as np
from forecastbox.metrics import mae, rmse, mape, mase, smape, theil_u2

actual = np.array([2.1, 2.3, 2.5, 2.4, 2.6])
predicted = np.array([2.0, 2.4, 2.3, 2.5, 2.7])
train = np.array([1.8, 1.9, 2.0, 2.1, 2.0, 2.1, 2.2])

print(f"MAE:      {mae(actual, predicted):.4f}")      # 0.1400
print(f"RMSE:     {rmse(actual, predicted):.4f}")      # 0.1549
print(f"MAPE:     {mape(actual, predicted):.2f}%")     # 5.21%
print(f"sMAPE:    {smape(actual, predicted):.2f}%")    # 5.18%
print(f"MASE:     {mase(actual, predicted, train):.4f}")  # 0.8750
print(f"Theil U2: {theil_u2(actual, predicted):.4f}")  # 0.7800
```

---

## Probabilistic Metrics

!!! info "Module"
    **Import**: `from forecastbox.metrics import crps, crps_gaussian, log_score`
    **Source**: `forecastbox/metrics/advanced_metrics.py`

### `crps(actual, draws)`

Continuous Ranked Probability Score. Evaluates the quality of a probabilistic forecast represented as simulation draws.

$$\text{CRPS}(F, y) = \mathbb{E}|X - y| - \frac{1}{2}\mathbb{E}|X - X'|$$

where $X, X' \sim F$ are independent draws from the forecast distribution.

```python
crps(
    actual: float,
    draws: NDArray[np.float64],
) -> float
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `float` | *required* | Single realized value |
| `draws` | `NDArray[np.float64]` | *required* | Simulation draws from forecast distribution |

### `crps_gaussian(actual, mean, std)`

Analytical CRPS for Gaussian forecast distributions.

```python
crps_gaussian(
    actual: float,
    mean: float,
    std: float,
) -> float
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `float` | *required* | Realized value |
| `mean` | `float` | *required* | Forecast mean |
| `std` | `float` | *required* | Forecast standard deviation |

### `log_score(actual, density)`

Logarithmic (probability) score. Evaluates the log-density of the realized value under the forecast distribution.

$$\text{LogScore} = -\log f(\hat{y} = y)$$

```python
log_score(
    actual: float,
    density: Callable[[float], float],
) -> float
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `float` | *required* | Realized value |
| `density` | `Callable[[float], float]` | *required* | Forecast density function (PDF) |

### Example — Probabilistic Metrics

```python
import numpy as np
from forecastbox.metrics import crps, crps_gaussian, log_score
from scipy.stats import norm

draws = np.random.normal(loc=2.5, scale=0.3, size=10000)
actual = 2.4

print(f"CRPS (draws):    {crps(actual, draws):.4f}")
print(f"CRPS (Gaussian): {crps_gaussian(actual, mean=2.5, std=0.3):.4f}")
print(f"LogScore:        {log_score(actual, norm(2.5, 0.3).pdf):.4f}")
```

---

## Statistical Tests

### diebold_mariano()

Diebold-Mariano test for equal predictive accuracy between two forecasts. Implements the Harvey, Leybourne & Newbold (1997) small-sample correction.

$H_0$: Both forecasts have equal predictive accuracy
$H_1$: Forecast 1 and Forecast 2 have different accuracy

```python
from forecastbox.evaluation import diebold_mariano

result = diebold_mariano(
    actual: NDArray[np.float64] | list[float],
    forecast1: NDArray[np.float64] | list[float],
    forecast2: NDArray[np.float64] | list[float],
    h: int = 1,
    loss: str = "mse",
    one_sided: bool = False,
    hln_correction: bool = True,
) -> DMResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `NDArray[np.float64] \| list[float]` | *required* | Realized values |
| `forecast1` | `NDArray[np.float64] \| list[float]` | *required* | Forecasts from model 1 |
| `forecast2` | `NDArray[np.float64] \| list[float]` | *required* | Forecasts from model 2 |
| `h` | `int` | `1` | Forecast horizon (for HAC variance estimation) |
| `loss` | `str` | `"mse"` | Loss function: `"mse"`, `"mae"`, `"mape"` |
| `one_sided` | `bool` | `False` | One-sided test ($H_1$: forecast1 is better) |
| `hln_correction` | `bool` | `True` | Apply Harvey-Leybourne-Newbold small-sample correction |

**Returns**: `DMResult`

#### DMResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `float` | DM test statistic |
| `pvalue` | `float` | p-value |
| `loss_differential` | `NDArray[np.float64]` | Loss differential series $d_t = L(e_{1t}) - L(e_{2t})$ |
| `mean_loss_diff` | `float` | Mean loss differential $\bar{d}$ |
| `h` | `int` | Forecast horizon used |
| `loss` | `str` | Loss function used |
| `hln_corrected` | `bool` | Whether HLN correction was applied |
| `one_sided` | `bool` | Whether test was one-sided |

##### `conclusion(alpha)`

Human-readable test conclusion.

```python
result.conclusion(alpha: float = 0.05) -> str
```

#### DM Test Statistic

$$\text{DM} = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}}, \quad \bar{d} = \frac{1}{T}\sum_{t=1}^{T} d_t$$

where $d_t = L(e_{1t}) - L(e_{2t})$ is the loss differential, and $\hat{V}(\bar{d})$ is the HAC variance estimator accounting for serial correlation up to horizon $h$.

The HLN correction modifies the statistic as:

$$\text{DM}^* = \text{DM} \cdot \left[\frac{T + 1 - 2h + h(h-1)/T}{T}\right]^{1/2}$$

#### Example

```python
from forecastbox.evaluation import diebold_mariano

dm = diebold_mariano(
    actual=y_test,
    forecast1=arima_preds,
    forecast2=ets_preds,
    h=1,
    loss="mse",
    hln_correction=True,
)

print(f"DM statistic: {dm.statistic:.3f}")
print(f"p-value:      {dm.pvalue:.4f}")
print(dm.conclusion(alpha=0.05))
# "Reject H0: forecasts have significantly different accuracy (p=0.023)"
```

---

### model_confidence_set()

Model Confidence Set (MCS) procedure of Hansen, Lunde & Nason (2011). Identifies the set of models that contains the best model with a given confidence level.

$H_0$: All models in the surviving set have equal predictive ability

```python
from forecastbox.evaluation import model_confidence_set

result = model_confidence_set(
    errors: dict[str, NDArray[np.float64]],
    alpha: float = 0.10,
    nsim: int = 10000,
    method: str = "TR",
    block_length: int | None = None,
    seed: int | None = None,
) -> MCSResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `errors` | `dict[str, NDArray[np.float64]]` | *required* | Forecast errors per model (`{name: error_array}`) |
| `alpha` | `float` | `0.10` | Significance level for elimination |
| `nsim` | `int` | `10000` | Number of bootstrap replications |
| `method` | `str` | `"TR"` | Test statistic: `"TR"` (range), `"SQ"` (semi-quadratic), `"R"` (max-$t$) |
| `block_length` | `int \| None` | `None` | Block bootstrap length (auto-selected if `None`) |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |

**Returns**: `MCSResult`

#### MCSResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `included_models` | `list[str]` | Models in the confidence set |
| `excluded_models` | `list[str]` | Models eliminated from the set |
| `pvalues` | `dict[str, float]` | MCS p-value per model |
| `elimination_order` | `list[str]` | Order in which models were eliminated |
| `alpha` | `float` | Significance level used |

##### `summary()`

```python
result.summary() -> str
```

#### Example

```python
from forecastbox.evaluation import model_confidence_set

errors = {
    "ARIMA": y_test - arima_preds,
    "ETS": y_test - ets_preds,
    "VAR": y_test - var_preds,
    "Theta": y_test - theta_preds,
    "Naive": y_test - naive_preds,
}

mcs = model_confidence_set(errors, alpha=0.10, nsim=10000, seed=42)

print(mcs.summary())
# Model Confidence Set (alpha=0.10)
# Included: ['ARIMA', 'VAR']
# Excluded: ['ETS', 'Theta', 'Naive']
# Elimination order: Naive → Theta → ETS

print(mcs.pvalues)
# {'ARIMA': 1.000, 'VAR': 0.342, 'ETS': 0.072, 'Theta': 0.031, 'Naive': 0.002}
```

---

### giacomini_white()

Giacomini-White test for conditional predictive ability. Unlike DM, this test allows the relative performance of forecasts to vary with the state of the economy.

$H_0$: Both forecasts have equal conditional predictive ability
$H_1$: Conditional predictive ability differs

```python
from forecastbox.evaluation import giacomini_white

result = giacomini_white(
    actual: NDArray[np.float64] | list[float],
    forecast1: NDArray[np.float64] | list[float],
    forecast2: NDArray[np.float64] | list[float],
    h: int = 1,
    loss: str = "mse",
    window_size: int | None = None,
) -> GWResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `NDArray[np.float64] \| list[float]` | *required* | Realized values |
| `forecast1` | `NDArray[np.float64] \| list[float]` | *required* | Forecasts from model 1 |
| `forecast2` | `NDArray[np.float64] \| list[float]` | *required* | Forecasts from model 2 |
| `h` | `int` | `1` | Forecast horizon |
| `loss` | `str` | `"mse"` | Loss function: `"mse"`, `"mae"`, `"mape"` |
| `window_size` | `int \| None` | `None` | Rolling window size for estimation (fixed scheme) |

**Returns**: `GWResult`

#### GWResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `float` | GW test statistic ($\chi^2$) |
| `pvalue` | `float` | p-value |
| `loss_differential` | `NDArray[np.float64]` | Loss differential series |
| `mean_loss_diff` | `float` | Mean loss differential |
| `h` | `int` | Forecast horizon used |
| `loss` | `str` | Loss function used |

##### `conclusion(alpha)`

```python
result.conclusion(alpha: float = 0.05) -> str
```

#### Example

```python
from forecastbox.evaluation import giacomini_white

gw = giacomini_white(
    actual=y_test,
    forecast1=arima_preds,
    forecast2=ets_preds,
    h=1,
    loss="mse",
)

print(f"GW statistic: {gw.statistic:.3f}")
print(f"p-value:      {gw.pvalue:.4f}")
print(gw.conclusion())
```

---

### mincer_zarnowitz()

Mincer-Zarnowitz regression for forecast calibration testing. Tests whether forecasts are unbiased and efficient by regressing actuals on forecasts.

$H_0$: $\alpha = 0$ and $\beta = 1$ (forecasts are well-calibrated)

The MZ regression is:

$$y_t = \alpha + \beta \hat{y}_t + \varepsilon_t$$

```python
from forecastbox.evaluation import mincer_zarnowitz

result = mincer_zarnowitz(
    actual: NDArray[np.float64] | list[float],
    forecast: NDArray[np.float64] | list[float],
) -> MZResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `NDArray[np.float64] \| list[float]` | *required* | Realized values |
| `forecast` | `NDArray[np.float64] \| list[float]` | *required* | Forecast values |

**Returns**: `MZResult`

#### MZResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `alpha` | `float` | Estimated intercept $\hat{\alpha}$ |
| `beta` | `float` | Estimated slope $\hat{\beta}$ |
| `alpha_se` | `float` | Standard error of $\hat{\alpha}$ |
| `beta_se` | `float` | Standard error of $\hat{\beta}$ |
| `alpha_tstat` | `float` | t-statistic for $H_0: \alpha = 0$ |
| `beta_tstat` | `float` | t-statistic for $H_0: \beta = 1$ |
| `pvalue_alpha` | `float` | p-value for $H_0: \alpha = 0$ |
| `pvalue_beta` | `float` | p-value for $H_0: \beta = 1$ |
| `r_squared` | `float` | Regression $R^2$ |

##### `conclusion(alpha_level)`

```python
result.conclusion(alpha_level: float = 0.05) -> str
```

#### Interpretation

| $\hat{\alpha}$ | $\hat{\beta}$ | Interpretation |
|:-:|:-:|:--|
| $\approx 0$ | $\approx 1$ | Well-calibrated — forecasts are unbiased and efficient |
| $\neq 0$ | $\approx 1$ | Systematic bias — forecast level is shifted |
| $\approx 0$ | $\neq 1$ | Inefficient — forecast variance is wrong |
| $\neq 0$ | $\neq 1$ | Both biased and inefficient |

#### Example

```python
from forecastbox.evaluation import mincer_zarnowitz

mz = mincer_zarnowitz(actual=y_test, forecast=arima_preds)

print(f"alpha: {mz.alpha:.4f} (p={mz.pvalue_alpha:.4f})")
print(f"beta:  {mz.beta:.4f} (p={mz.pvalue_beta:.4f})")
print(f"R²:    {mz.r_squared:.4f}")
print(mz.conclusion())
# "Fail to reject H0: forecasts are well-calibrated (alpha≈0, beta≈1)"
```

---

### encompassing_test()

Fair-Shiller forecast encompassing test. Tests whether one forecast contains all the useful information in another.

$H_0$: Forecast 1 encompasses Forecast 2 (i.e., Forecast 2 adds no information)

The encompassing regression is:

$$y_t = \lambda_1 \hat{y}_{1t} + \lambda_2 \hat{y}_{2t} + \varepsilon_t$$

If $\lambda_2 = 0$, then Forecast 1 encompasses Forecast 2.

```python
from forecastbox.evaluation import encompassing_test

result = encompassing_test(
    actual: NDArray[np.float64] | list[float],
    forecast1: NDArray[np.float64] | list[float],
    forecast2: NDArray[np.float64] | list[float],
) -> EncompassingResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `NDArray[np.float64] \| list[float]` | *required* | Realized values |
| `forecast1` | `NDArray[np.float64] \| list[float]` | *required* | Forecasts from model 1 |
| `forecast2` | `NDArray[np.float64] \| list[float]` | *required* | Forecasts from model 2 |

**Returns**: `EncompassingResult`

#### EncompassingResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `statistic` | `float` | Test statistic for $H_0: \lambda_2 = 0$ |
| `pvalue` | `float` | p-value |
| `coef_f1` | `float` | Estimated $\hat{\lambda}_1$ |
| `coef_f2` | `float` | Estimated $\hat{\lambda}_2$ |

##### `conclusion(alpha)`

```python
result.conclusion(alpha: float = 0.05) -> str
```

#### Example

```python
from forecastbox.evaluation import encompassing_test

enc = encompassing_test(
    actual=y_test,
    forecast1=arima_preds,
    forecast2=ets_preds,
)

print(f"lambda_1 (ARIMA): {enc.coef_f1:.3f}")
print(f"lambda_2 (ETS):   {enc.coef_f2:.3f}")
print(f"p-value:          {enc.pvalue:.4f}")
print(enc.conclusion())
# "Reject H0: ETS adds information beyond ARIMA (p=0.018)"
```

---

## Cross-Validation

!!! info "Module"
    **Import**: `from forecastbox.cv import expanding_window_cv, rolling_window_cv, blocked_cv, CVResults`
    **Source**: `forecastbox/cv/`

### expanding_window_cv()

Expanding (recursive) window cross-validation. The training window grows with each fold.

```
Fold 1: [====train====]--test--
Fold 2: [=====train=====]--test--
Fold 3: [======train======]--test--
```

```python
from forecastbox.cv import expanding_window_cv

results = expanding_window_cv(
    data: NDArray[np.float64] | list[float],
    horizon: int,
    forecast_fn: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    initial_size: int | None = None,
) -> CVResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `NDArray[np.float64] \| list[float]` | *required* | Full time series |
| `horizon` | `int` | *required* | Forecast horizon per fold |
| `forecast_fn` | `Callable[[NDArray, int], NDArray]` | *required* | Function that takes `(train_data, h)` and returns `h`-step forecasts |
| `initial_size` | `int \| None` | `None` | Initial training window (default: `len(data) - 3 * horizon`) |

**Returns**: `CVResults`

### rolling_window_cv()

Fixed-size rolling window cross-validation.

```
Fold 1: [====train====]--test--
Fold 2:  [====train====]--test--
Fold 3:   [====train====]--test--
```

```python
from forecastbox.cv import rolling_window_cv

results = rolling_window_cv(
    data: NDArray[np.float64] | list[float],
    horizon: int,
    forecast_fn: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    window_size: int,
    step: int = 1,
) -> CVResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `NDArray[np.float64] \| list[float]` | *required* | Full time series |
| `horizon` | `int` | *required* | Forecast horizon per fold |
| `forecast_fn` | `Callable[[NDArray, int], NDArray]` | *required* | Forecasting function |
| `window_size` | `int` | *required* | Fixed training window size |
| `step` | `int` | `1` | Step between folds |

**Returns**: `CVResults`

### blocked_cv()

Blocked cross-validation with gap between train and test to avoid information leakage.

```
Fold 1: [====train====]  gap  [test]
Fold 2:   [====train====]  gap  [test]
```

```python
from forecastbox.cv import blocked_cv

results = blocked_cv(
    data: NDArray[np.float64] | list[float],
    horizon: int,
    forecast_fn: Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    window_size: int,
    test_size: int,
    step: int = 1,
) -> CVResults
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `NDArray[np.float64] \| list[float]` | *required* | Full time series |
| `horizon` | `int` | *required* | Forecast horizon per fold |
| `forecast_fn` | `Callable[[NDArray, int], NDArray]` | *required* | Forecasting function |
| `window_size` | `int` | *required* | Training window size |
| `test_size` | `int` | *required* | Test block size |
| `step` | `int` | `1` | Step between folds |

**Returns**: `CVResults`

### CVResults

Container for cross-validation results.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `errors` | `NDArray[np.float64]` | Error matrix of shape `(n_folds, horizon)` |
| `forecasts` | `list[NDArray[np.float64]]` | Forecasts per fold |
| `actuals` | `list[NDArray[np.float64]]` | Actuals per fold |
| `n_folds` | `int` | Number of CV folds |
| `horizon` | `int` | Forecast horizon |
| `metrics_by_horizon` | `pd.DataFrame` | Metrics broken down by horizon step |
| `metrics_overall` | `dict` | Aggregated metrics across all folds |

##### `summary()`

```python
results.summary() -> str
```

### Example — Cross-Validation

```python
import numpy as np
from forecastbox.auto import AutoARIMA
from forecastbox.cv import expanding_window_cv, rolling_window_cv

data = np.random.randn(200).cumsum() + 100

# Define forecast function
def arima_forecast(train, h):
    model = AutoARIMA(ic="aicc")
    result = model.fit(train)
    return result.forecast(h).point

# Expanding window CV
cv_expanding = expanding_window_cv(
    data=data,
    horizon=12,
    forecast_fn=arima_forecast,
    initial_size=120,
)

print(cv_expanding.summary())
# Expanding Window CV (68 folds, h=12)
# Overall RMSE: 1.23
# Overall MAE:  0.98

# Rolling window CV
cv_rolling = rolling_window_cv(
    data=data,
    horizon=12,
    forecast_fn=arima_forecast,
    window_size=60,
    step=1,
)

print(cv_rolling.metrics_by_horizon)
#    h   rmse    mae
# 0  1  0.85   0.68
# 1  2  0.92   0.74
# ...
```

---

## EvaluationResult

Comprehensive evaluation result container aggregating metrics, tests, and rankings.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `metrics` | `pd.DataFrame` | Metrics table (models × metrics) |
| `dm_tests` | `dict[tuple[str, str], DMResult]` | Pairwise DM test results |
| `mcs` | `MCSResult \| None` | Model Confidence Set result |
| `rankings` | `pd.DataFrame` | Model rankings by each metric |
| `best_model` | `str` | Best model by primary metric |

---

## See Also

- [Core API](core.md) — `Forecast` and `ForecastResults` used with evaluation
- [Combination API](combination.md) — Combine forecasts before or after evaluation
- [User Guide: Evaluation](../user-guide/evaluation/index.md) — Detailed usage guide
- [User Guide: Cross-Validation](../user-guide/evaluation/cross-validation.md) — CV strategies guide
- [Theory: Evaluation](../theory/evaluation-theory.md) — Mathematical foundations
- [Theory: MCS](../theory/mcs-theory.md) — Model Confidence Set theory
