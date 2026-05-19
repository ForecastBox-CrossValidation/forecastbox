---
title: "Diagnostics API"
description: "API reference for forecastbox.diagnostics — bias, efficiency, rationality, encompassing, weight stability, news diagnostic, and real-time evaluation"
---

# Diagnostics API Reference

!!! info "Module"
    **Import**: `from forecastbox.diagnostics import bias_test, efficiency_test, rationality_test, weight_stability, encompassing_test, news_diagnostic, real_time_evaluation`
    **Source**: `forecastbox/diagnostics/`

## Overview

The diagnostics module provides statistical tests for evaluating forecast quality, combination stability, and real-time performance. All functions follow a consistent pattern: accept forecast/actual arrays and return typed result objects with `summary()` and `conclusion()` methods.

| Function | Category | Description |
|----------|----------|-------------|
| [`bias_test()`](#bias_test) | Calibration | Tests whether forecasts are systematically biased |
| [`efficiency_test()`](#efficiency_test) | Calibration | Tests whether forecasts efficiently use available information |
| [`rationality_test()`](#rationality_test) | Calibration | Joint test of unbiasedness and efficiency |
| [`weight_stability()`](#weight_stability) | Combination | Tests temporal stability of combination weights |
| [`encompassing_test()`](#encompassing_test) | Comparison | Tests whether one forecast encompasses another |
| [`news_diagnostic()`](#news_diagnostic) | Nowcasting | Diagnoses nowcast revision patterns |
| [`real_time_evaluation()`](#real_time_evaluation) | Real-time | Evaluates model performance across data vintages |

---

## bias_test()

Tests whether forecasts are systematically biased by regressing forecast errors on a constant.

$$H_0: \mathbb{E}[y_t - \hat{y}_t] = 0$$

The test regresses $e_t = y_t - \hat{y}_t$ on a constant and tests the significance of the intercept using HAC standard errors.

```python
from forecastbox.diagnostics import bias_test

result = bias_test(
    forecast: NDArray[np.float64] | pd.Series,
    actual: NDArray[np.float64] | pd.Series,
    alpha: float = 0.05,
    hac_lags: int | str = "auto",
) -> BiasResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | `NDArray[np.float64] \| pd.Series` | *required* | Forecast values |
| `actual` | `NDArray[np.float64] \| pd.Series` | *required* | Realized values |
| `alpha` | `float` | `0.05` | Significance level |
| `hac_lags` | `int \| str` | `"auto"` | HAC bandwidth. `"auto"` uses Newey-West automatic selection |

#### BiasResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `mean_error` | `float` | Mean forecast error |
| `t_statistic` | `float` | $t$-statistic for the bias test |
| `p_value` | `float` | $p$-value (two-sided) |
| `reject_null` | `bool` | Whether the null of no bias is rejected |
| `ci` | `tuple[float, float]` | Confidence interval for the mean error |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Test results table |
| `conclusion()` | `str` | Human-readable interpretation |

#### Example

```python
from forecastbox.diagnostics import bias_test

result = bias_test(forecast=f_arima, actual=y_test, alpha=0.05)
print(result.conclusion())
# "Forecasts show a statistically significant positive bias of 0.032
#  (t=2.41, p=0.018). The model systematically under-predicts."
```

---

## efficiency_test()

Tests forecast efficiency — whether forecast errors are orthogonal to past information (lagged errors and forecasts).

$$e_t = \alpha + \beta_1 e_{t-1} + \beta_2 \hat{y}_{t} + u_t, \quad H_0: \beta_1 = \beta_2 = 0$$

```python
from forecastbox.diagnostics import efficiency_test

result = efficiency_test(
    forecast: NDArray[np.float64] | pd.Series,
    actual: NDArray[np.float64] | pd.Series,
    n_lags: int = 1,
    alpha: float = 0.05,
) -> EfficiencyResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | `NDArray[np.float64] \| pd.Series` | *required* | Forecast values |
| `actual` | `NDArray[np.float64] \| pd.Series` | *required* | Realized values |
| `n_lags` | `int` | `1` | Number of lagged errors to include |
| `alpha` | `float` | `0.05` | Significance level |

#### EfficiencyResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `f_statistic` | `float` | $F$-statistic for joint significance |
| `p_value` | `float` | $p$-value of the $F$-test |
| `reject_null` | `bool` | Whether the null of efficiency is rejected |
| `coefficients` | `pd.DataFrame` | Regression coefficients with standard errors |
| `r_squared` | `float` | $R^2$ of the efficiency regression |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Test results table |
| `conclusion()` | `str` | Human-readable interpretation |

#### Example

```python
from forecastbox.diagnostics import efficiency_test

result = efficiency_test(forecast=f_arima, actual=y_test, n_lags=2)
print(result.conclusion())
# "Forecast errors are predictable from lagged information (F=4.12, p=0.008).
#  The forecaster does not efficiently incorporate available information."
```

---

## rationality_test()

Joint test of forecast rationality combining unbiasedness and efficiency. Based on the Mincer-Zarnowitz regression with additional information variables.

$$y_t = \alpha + \beta \hat{y}_t + \gamma' \mathbf{z}_t + u_t, \quad H_0: \alpha = 0, \; \beta = 1, \; \gamma = 0$$

```python
from forecastbox.diagnostics import rationality_test

result = rationality_test(
    forecast: NDArray[np.float64] | pd.Series,
    actual: NDArray[np.float64] | pd.Series,
    instruments: NDArray[np.float64] | pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> RationalityResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | `NDArray[np.float64] \| pd.Series` | *required* | Forecast values |
| `actual` | `NDArray[np.float64] \| pd.Series` | *required* | Realized values |
| `instruments` | `NDArray[np.float64] \| pd.DataFrame \| None` | `None` | Additional information variables ($\mathbf{z}_t$). If `None`, tests only $\alpha=0, \beta=1$ |
| `alpha` | `float` | `0.05` | Significance level |

#### RationalityResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `f_statistic` | `float` | $F$-statistic for the joint hypothesis |
| `p_value` | `float` | $p$-value of the joint test |
| `reject_null` | `bool` | Whether the null of rationality is rejected |
| `intercept` | `float` | Estimated $\alpha$ |
| `slope` | `float` | Estimated $\beta$ |
| `coefficients` | `pd.DataFrame` | Full coefficient table |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Test results table |
| `conclusion()` | `str` | Human-readable interpretation |

#### Example

```python
from forecastbox.diagnostics import rationality_test

result = rationality_test(
    forecast=f_arima,
    actual=y_test,
    instruments=oil_price_changes,
)
print(result.conclusion())
# "Forecasts fail the rationality test (F=5.32, p=0.002). The intercept (0.015)
#  and slope (0.87) deviate from (0, 1), and oil price changes carry additional
#  predictive information not incorporated by the forecaster."
```

---

## weight_stability()

Tests the temporal stability of forecast combination weights. Detects structural breaks in the optimal combination using fluctuation tests.

```python
from forecastbox.diagnostics import weight_stability

result = weight_stability(
    combination: Any,
    window: int | None = None,
    test: str = "cusum",
    alpha: float = 0.05,
) -> StabilityResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `combination` | `Any` | *required* | A fitted combiner object (e.g., `OLSCombiner`, `BMACombiner`) with time-varying weight history |
| `window` | `int \| None` | `None` | Rolling window for subsample analysis. `None` for full-sample fluctuation test |
| `test` | `str` | `"cusum"` | Test type: `"cusum"` (CUSUM), `"mosum"` (MOSUM), `"chow"` (Chow breakpoint) |
| `alpha` | `float` | `0.05` | Significance level |

#### StabilityResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `test_statistic` | `float` | Test statistic |
| `p_value` | `float` | $p$-value |
| `reject_null` | `bool` | Whether the null of stable weights is rejected |
| `break_date` | `pd.Timestamp \| None` | Estimated break date (if detected) |
| `weight_path` | `pd.DataFrame` | Time series of rolling weights |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Test results table |
| `conclusion()` | `str` | Human-readable interpretation |
| `plot()` | `matplotlib.Figure` | Weight evolution with stability bounds |

#### Example

```python
from forecastbox.diagnostics import weight_stability

result = weight_stability(combination=fitted_ols_combiner, test="cusum")
print(result.conclusion())
# "Combination weights are unstable (CUSUM=1.42, p=0.031). A structural
#  break is detected around 2020-03, likely due to the COVID-19 shock.
#  Consider using TimeVaryingCombiner for adaptive weighting."

result.plot()
```

---

## encompassing_test()

Tests whether forecast 1 encompasses forecast 2 — i.e., whether forecast 2 adds any useful information beyond what forecast 1 already provides.

$$y_t = \alpha + \lambda_1 \hat{y}_{1t} + \lambda_2 \hat{y}_{2t} + u_t, \quad H_0: \lambda_2 = 0$$

```python
from forecastbox.diagnostics import encompassing_test

result = encompassing_test(
    forecast1: NDArray[np.float64] | pd.Series,
    forecast2: NDArray[np.float64] | pd.Series,
    actual: NDArray[np.float64] | pd.Series,
    alpha: float = 0.05,
    hac_lags: int | str = "auto",
) -> EncompassingResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast1` | `NDArray[np.float64] \| pd.Series` | *required* | First forecast (tested as the encompassing model) |
| `forecast2` | `NDArray[np.float64] \| pd.Series` | *required* | Second forecast (tested as the encompassed model) |
| `actual` | `NDArray[np.float64] \| pd.Series` | *required* | Realized values |
| `alpha` | `float` | `0.05` | Significance level |
| `hac_lags` | `int \| str` | `"auto"` | HAC bandwidth |

#### EncompassingResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `lambda1` | `float` | Weight on forecast 1 |
| `lambda2` | `float` | Weight on forecast 2 |
| `t_statistic` | `float` | $t$-statistic for $\lambda_2 = 0$ |
| `p_value` | `float` | $p$-value (one-sided) |
| `reject_null` | `bool` | Whether the null of encompassing is rejected |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Test results table |
| `conclusion()` | `str` | Human-readable interpretation |

#### Interpretation

| $\lambda_2$ significant? | Interpretation |
|---------------------------|----------------|
| No | Forecast 1 encompasses forecast 2 — no gain from combining |
| Yes | Forecast 2 carries additional information — combination is beneficial |

#### Example

```python
from forecastbox.diagnostics import encompassing_test

result = encompassing_test(
    forecast1=f_arima,
    forecast2=f_var,
    actual=y_test,
)
print(result.conclusion())
# "ARIMA does NOT encompass VAR (t=2.85, p=0.003). The VAR forecast carries
#  additional information (λ₂=0.38). Combining both forecasts is recommended."
```

---

## news_diagnostic()

Diagnoses patterns in nowcast revisions. Tests whether revisions exhibit bias, serial correlation, or excessive volatility — signs of information inefficiency.

```python
from forecastbox.diagnostics import news_diagnostic

result = news_diagnostic(
    nowcast_old: Forecast | NDArray[np.float64],
    nowcast_new: Forecast | NDArray[np.float64],
    alpha: float = 0.05,
) -> NewsDiagnosticResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nowcast_old` | [`Forecast`](core.md#forecast) `\| NDArray[np.float64]` | *required* | Previous nowcast(s) |
| `nowcast_new` | [`Forecast`](core.md#forecast) `\| NDArray[np.float64]` | *required* | Updated nowcast(s) |
| `alpha` | `float` | `0.05` | Significance level |

#### NewsDiagnosticResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `mean_revision` | `float` | Average nowcast revision |
| `revision_bias_pvalue` | `float` | $p$-value for revision bias test |
| `serial_correlation` | `float` | First-order autocorrelation of revisions |
| `serial_corr_pvalue` | `float` | $p$-value for serial correlation test |
| `revision_volatility` | `float` | Standard deviation of revisions |
| `noise_to_signal` | `float` | Ratio of revision volatility to nowcast volatility |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | All diagnostic tests in one table |
| `conclusion()` | `str` | Human-readable interpretation |
| `plot()` | `matplotlib.Figure` | Revision histogram and time series |

#### Example

```python
from forecastbox.diagnostics import news_diagnostic

result = news_diagnostic(
    nowcast_old=nowcasts_t_minus_1,
    nowcast_new=nowcasts_t,
)
print(result.summary())
#                          value  p_value  reject
# revision_bias            0.003    0.420   False
# serial_correlation       0.312    0.008    True
# noise_to_signal          0.450      NaN     NaN

print(result.conclusion())
# "Nowcast revisions show no significant bias but exhibit serial correlation
#  (ρ=0.31, p=0.008), indicating that information is incorporated gradually
#  rather than in a single step. The noise-to-signal ratio (0.45) is moderate."
```

---

## real_time_evaluation()

Evaluates model performance across data vintages, replicating the forecaster's real-time information set. This is the gold standard for assessing forecast accuracy, as it avoids look-ahead bias from data revisions.

```python
from forecastbox.diagnostics import real_time_evaluation

result = real_time_evaluation(
    vintages: VintageManager,
    model: Any,
    target: str,
    metrics: tuple[str, ...] = ("rmse", "mae"),
    retrain: bool = True,
) -> RealTimeResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vintages` | [`VintageManager`](nowcasting.md#vintagemanager) | *required* | Collection of data vintages |
| `model` | `Any` | *required* | A forecastbox model with `fit()` and `forecast()` or `nowcast()` methods |
| `target` | `str` | *required* | Name of the target variable to evaluate |
| `metrics` | `tuple[str, ...]` | `("rmse", "mae")` | Evaluation metrics |
| `retrain` | `bool` | `True` | Whether to refit the model at each vintage |

#### RealTimeResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecasts` | `pd.DataFrame` | Forecast at each vintage date |
| `actuals` | `pd.Series` | Final-vintage realized values |
| `errors` | `pd.DataFrame` | Forecast errors at each vintage |
| `metrics` | `pd.DataFrame` | Metrics per vintage |
| `aggregate_metrics` | `dict[str, float]` | Overall metrics across all vintages |

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `pd.DataFrame` | Aggregate and per-vintage metrics |
| `conclusion()` | `str` | Human-readable interpretation |
| `plot()` | `matplotlib.Figure` | Forecast evolution across vintages |

#### Example

```python
from forecastbox.diagnostics import real_time_evaluation
from forecastbox.nowcasting import DynamicFactorModel, VintageManager

# Assume vm is a VintageManager with monthly snapshots
dfm = DynamicFactorModel(n_factors=3)

result = real_time_evaluation(
    vintages=vm,
    model=dfm,
    target="gdp",
    metrics=("rmse", "mae", "mape"),
    retrain=True,
)

print(result.summary())
#             rmse    mae   mape
# 2024-01    0.041  0.033  1.92
# 2024-02    0.038  0.030  1.78
# 2024-03    0.025  0.020  1.21
# Aggregate  0.035  0.028  1.64

print(result.conclusion())
# "Real-time RMSE (0.035) is 15% higher than pseudo out-of-sample RMSE (0.030),
#  indicating moderate data revision effects. Nowcast accuracy improves as
#  the quarter progresses (RMSE falls from 0.041 to 0.025)."

result.plot(title="GDP Nowcast Evolution Across Vintages")
```

---

## See Also

- [Evaluation API](evaluation.md) — Pairwise tests (Diebold-Mariano, MCS) and error metrics
- [Nowcasting API](nowcasting.md) — `VintageManager` and `NewsDecomposition`
- [Combination API](combination.md) — Combiners tested by `weight_stability()`
- [Evaluation Theory](../theory/evaluation-theory.md) — Statistical foundations
