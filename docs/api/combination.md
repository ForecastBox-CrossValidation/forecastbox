---
title: "Combination API"
description: "API reference for forecastbox.combination — SimpleCombiner, WeightedCombiner, OLSCombiner, StackingCombiner, BMACombiner, TimeVaryingCombiner, OptimalCombiner"
---

# Combination API Reference

!!! info "Module"
    **Import**: `from forecastbox.combination import SimpleCombiner, WeightedCombiner, OLSCombiner, StackingCombiner, BMACombiner, TimeVaryingCombiner, OptimalCombiner`
    **Source**: `forecastbox/combination/`

## Overview

The combination module implements forecast combination (ensemble) methods ranging from simple averaging to optimal Bayesian approaches. All combiners follow a consistent **fit → combine** pattern inheriting from `BaseCombiner`.

| Class | Method | Reference |
|-------|--------|-----------|
| [`SimpleCombiner`](#simplecombiner) | Mean, median, trimmed mean | — |
| [`WeightedCombiner`](#weightedcombiner) | Inverse MSE, AIC/BIC weights | Akaike (1978) |
| [`OLSCombiner`](#olscombiner) | OLS regression (Granger-Ramanathan) | Granger & Ramanathan (1984) |
| [`StackingCombiner`](#stackingcombiner) | Meta-learning (ridge, lasso, RF, GBM) | Wolpert (1992) |
| [`BMACombiner`](#bmacombiner) | Bayesian Model Averaging | Hoeting et al. (1999) |
| [`TimeVaryingCombiner`](#timevaryingcombiner) | Adaptive time-varying weights | Koop & Korobilis (2012) |
| [`OptimalCombiner`](#optimalcombiner) | Bates-Granger optimal | Bates & Granger (1969) |

---

## combine()

Convenience function for one-line forecast combination.

```python
from forecastbox.combination import combine

combined = combine(
    forecasts: list[Forecast],
    actual: NDArray[np.float64] | None = None,
    method: str = "mean",
    **kwargs: Any,
) -> Forecast
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecasts` | `list[Forecast]` | *required* | List of forecasts to combine |
| `actual` | `NDArray[np.float64] \| None` | `None` | Training actuals (required for fitted methods) |
| `method` | `str` | `"mean"` | Combination method: `"mean"`, `"median"`, `"trimmed"`, `"inverse_mse"`, `"ols"`, `"ols_constrained"`, `"stacking"`, `"bma"`, `"optimal"`, `"time_varying"` |
| `**kwargs` | `Any` | — | Additional parameters passed to the combiner constructor |

**Returns**: [`Forecast`](core.md#forecast)

```python
from forecastbox.combination import combine

combined = combine(forecasts, actual=train_actual, method="ols_constrained")
combined.plot(title="OLS Combined Forecast")
```

---

## BaseCombiner

Abstract base class for all combiners. Not instantiated directly.

### Key Attributes (after `fit`)

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `NDArray[np.float64] \| None` | Estimated combination weights |
| `is_fitted_` | `bool` | Whether `fit()` has been called |
| `n_models_` | `int` | Number of models being combined |

### Methods

##### `fit(forecasts_train, actual)`

Estimate combination weights from training data.

```python
combiner.fit(
    forecasts_train: list[NDArray[np.float64]],
    actual: NDArray[np.float64],
) -> BaseCombiner
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecasts_train` | `list[NDArray[np.float64]]` | *required* | Training forecasts, one array per model |
| `actual` | `NDArray[np.float64]` | *required* | Realized values for the training period |

**Returns**: `self` (for method chaining)

##### `combine(forecasts)`

Produce a combined forecast using the estimated weights.

```python
combiner.combine(forecasts: list[Forecast]) -> Forecast
```

**Returns**: [`Forecast`](core.md#forecast)

---

## SimpleCombiner

Forecast combination without weight estimation. Computes equal-weight averages across models.

### Constructor

```python
SimpleCombiner(
    method: str = "mean",
    trim_fraction: float = 0.1,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"mean"` | Aggregation method: `"mean"`, `"median"`, `"trimmed"` |
| `trim_fraction` | `float` | `0.1` | Fraction to trim from each tail (only used when `method="trimmed"`) |

!!! note
    `SimpleCombiner` does not require a `fit()` call — weights are equal by definition. Calling `fit()` is a no-op that sets `is_fitted_=True` for API consistency.

### Example

```python
from forecastbox.combination import SimpleCombiner

combiner = SimpleCombiner(method="median")
combined = combiner.combine([arima_fc, ets_fc, var_fc])

print(combined.model_name)  # "SimpleCombiner(median)"
combined.plot(title="Median Combination")
```

---

## WeightedCombiner

Performance-based weighting. Models with lower error receive higher weights. Supports inverse MSE and information-criteria-based weights.

### Constructor

```python
WeightedCombiner(
    method: str = "inverse_mse",
    n_params: list[int] | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"inverse_mse"` | Weighting scheme: `"inverse_mse"`, `"aic_weights"`, `"bic_weights"` |
| `n_params` | `list[int] \| None` | `None` | Number of parameters per model (required for `"aic_weights"` / `"bic_weights"`) |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `NDArray[np.float64]` | Estimated weights (sum to 1) |
| `mse_` | `NDArray[np.float64] \| None` | Per-model MSE values |
| `ic_values_` | `NDArray[np.float64] \| None` | Per-model IC values (if IC-based) |

### Weight Formulas

For **inverse MSE** weighting, the weight of model $i$ is:

$$w_i = \frac{1/\text{MSE}_i}{\sum_{j=1}^{K} 1/\text{MSE}_j}$$

For **AIC/BIC weights**, following Burnham & Anderson (2002):

$$w_i = \frac{\exp(-\Delta_i / 2)}{\sum_{j=1}^{K} \exp(-\Delta_j / 2)}, \quad \Delta_i = \text{IC}_i - \text{IC}_{\min}$$

### Example

```python
from forecastbox.combination import WeightedCombiner

combiner = WeightedCombiner(method="inverse_mse")
combiner.fit(forecasts_train=[f1_train, f2_train, f3_train], actual=y_train)

print(combiner.weights_)  # [0.45, 0.30, 0.25]

combined = combiner.combine([f1_test, f2_test, f3_test])
```

---

## OLSCombiner

Forecast combination via OLS regression following Granger & Ramanathan (1984). Supports three variants:

| Variant | `intercept` | `constrained` | Description |
|---------|:-----------:|:-------------:|-------------|
| **GR-1** | `False` | `True` | Weights constrained to sum to 1, no intercept |
| **GR-2** | `True` | `False` | Unconstrained with intercept (bias correction) |
| **GR-3** | `False` | `False` | Unconstrained without intercept |

### Constructor

```python
OLSCombiner(
    intercept: bool = False,
    constrained: bool = True,
    regularization: str | None = None,
    alpha: float = 0.01,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intercept` | `bool` | `False` | Include intercept term (GR-2 variant) |
| `constrained` | `bool` | `True` | Constrain weights to sum to 1 (GR-1 variant) |
| `regularization` | `str \| None` | `None` | Regularization: `None`, `"ridge"`, `"lasso"` |
| `alpha` | `float` | `0.01` | Regularization strength (only if `regularization` is set) |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `NDArray[np.float64]` | Estimated OLS weights |
| `intercept_` | `float` | Estimated intercept (0.0 if `intercept=False`) |
| `residuals_` | `NDArray[np.float64] \| None` | In-sample residuals |

### OLS Formulation

The combination regression is:

$$y_t = \alpha + \sum_{i=1}^{K} w_i \hat{y}_{it} + \varepsilon_t$$

where $y_t$ is the actual value, $\hat{y}_{it}$ is the forecast from model $i$, and $w_i$ are the combination weights. Under GR-1: $\alpha = 0$ and $\sum w_i = 1$.

### Example

```python
from forecastbox.combination import OLSCombiner

# GR-1: constrained weights, no intercept
combiner = OLSCombiner(intercept=False, constrained=True)
combiner.fit(forecasts_train=[f1, f2, f3], actual=y_train)

print(combiner.weights_)     # [0.52, 0.31, 0.17]
print(combiner.intercept_)   # 0.0

# GR-2: with intercept, unconstrained
combiner_gr2 = OLSCombiner(intercept=True, constrained=False)
combiner_gr2.fit(forecasts_train=[f1, f2, f3], actual=y_train)
print(combiner_gr2.intercept_)  # 0.05

# With ridge regularization
combiner_ridge = OLSCombiner(constrained=True, regularization="ridge", alpha=0.1)
combiner_ridge.fit(forecasts_train=[f1, f2, f3], actual=y_train)
combined = combiner_ridge.combine([f1_test, f2_test, f3_test])
```

---

## StackingCombiner

Meta-learning approach to forecast combination. Uses cross-validated predictions from individual models as features for a second-stage meta-learner.

### Constructor

```python
StackingCombiner(
    meta_learner: str | Any = "ridge",
    cv_folds: int = 5,
    use_cv_predictions: bool = True,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `meta_learner` | `str \| Any` | `"ridge"` | Meta-learner: `"ridge"`, `"lasso"`, `"rf"`, `"gbm"`, or a scikit-learn-compatible estimator |
| `cv_folds` | `int` | `5` | Number of CV folds for generating meta-features |
| `use_cv_predictions` | `bool` | `True` | Use CV predictions as meta-features (`True`) or raw training forecasts (`False`) |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `NDArray[np.float64]` | Effective weights (derived from meta-learner coefficients) |
| `meta_model_` | `Any` | Fitted meta-learner object |
| `feature_importances_` | `NDArray[np.float64] \| None` | Feature importances (for tree-based meta-learners) |

### Example

```python
from forecastbox.combination import StackingCombiner

# Ridge meta-learner (linear stacking)
combiner = StackingCombiner(meta_learner="ridge", cv_folds=5)
combiner.fit(forecasts_train=[f1, f2, f3, f4], actual=y_train)

print(combiner.weights_)  # [0.35, 0.28, 0.22, 0.15]
combined = combiner.combine([f1_test, f2_test, f3_test, f4_test])

# Gradient boosting meta-learner (nonlinear stacking)
combiner_gbm = StackingCombiner(meta_learner="gbm", cv_folds=3)
combiner_gbm.fit(forecasts_train=[f1, f2, f3, f4], actual=y_train)
print(combiner_gbm.feature_importances_)  # [0.40, 0.30, 0.20, 0.10]
```

---

## BMACombiner

Bayesian Model Averaging forecast combination. Weights are posterior model probabilities computed from information criteria or marginal likelihoods.

### Constructor

```python
BMACombiner(
    prior_weights: NDArray[np.float64] | None = None,
    approximation: str = "bic",
    n_params: list[int] | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior_weights` | `NDArray[np.float64] \| None` | `None` | Prior model probabilities (uniform if `None`) |
| `approximation` | `str` | `"bic"` | Approximation for marginal likelihood: `"bic"`, `"aic"`, `"loglike"` |
| `n_params` | `list[int] \| None` | `None` | Number of parameters per model (for IC computation) |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `NDArray[np.float64]` | Posterior model probabilities (sum to 1) |
| `posterior_weights_` | `NDArray[np.float64] \| None` | Same as `weights_` (explicit alias) |
| `bma_variance_` | `float \| None` | BMA predictive variance (accounts for model uncertainty) |
| `model_mse_` | `NDArray[np.float64] \| None` | Per-model MSE |
| `ic_values_` | `NDArray[np.float64] \| None` | Per-model IC values |

### BMA Formulation

The BMA combined forecast and its variance are:

$$\hat{y}_{\text{BMA}} = \sum_{i=1}^{K} P(M_i | \mathbf{y}) \, \hat{y}_i$$

$$\text{Var}_{\text{BMA}} = \sum_{i=1}^{K} P(M_i | \mathbf{y}) \left[ \sigma_i^2 + (\hat{y}_i - \hat{y}_{\text{BMA}})^2 \right]$$

where $P(M_i | \mathbf{y})$ is the posterior probability of model $M_i$ approximated via BIC:

$$P(M_i | \mathbf{y}) \propto P(M_i) \exp\left(-\tfrac{1}{2} \text{BIC}_i\right)$$

### Example

```python
from forecastbox.combination import BMACombiner

combiner = BMACombiner(approximation="bic")
combiner.fit(
    forecasts_train=[f1, f2, f3],
    actual=y_train,
    ic_values=np.array([342.1, 345.8, 348.2]),
)

print(combiner.posterior_weights_)  # [0.62, 0.25, 0.13]
print(combiner.bma_variance_)       # 0.18

combined = combiner.combine([f1_test, f2_test, f3_test])
```

---

## TimeVaryingCombiner

Adaptive forecast combination with time-varying weights. Weights evolve over time to capture changes in relative model performance. Implements exponential forgetting and recursive updating.

### Constructor

```python
TimeVaryingCombiner(
    method: str = "exponential",
    forgetting_factor: float = 0.95,
    window_size: int | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"exponential"` | Weight evolution method: `"exponential"`, `"rolling"`, `"recursive"` |
| `forgetting_factor` | `float` | `0.95` | Exponential forgetting factor $\lambda \in (0, 1]$. Lower values give more weight to recent performance |
| `window_size` | `int \| None` | `None` | Rolling window size (required for `method="rolling"`) |

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `NDArray[np.float64]` | Final (most recent) weights |
| `weights_history_` | `NDArray[np.float64]` | Weight path over time, shape `(T, K)` |

### Weight Evolution

For **exponential forgetting**, the weight of model $i$ at time $t$ is based on discounted cumulative MSE:

$$w_{i,t} = \frac{1 / \text{DMSE}_{i,t}}{\sum_{j=1}^{K} 1 / \text{DMSE}_{j,t}}, \quad \text{DMSE}_{i,t} = \sum_{s=1}^{t} \lambda^{t-s} (y_s - \hat{y}_{i,s})^2$$

where $\lambda$ is the forgetting factor.

### Example

```python
from forecastbox.combination import TimeVaryingCombiner

combiner = TimeVaryingCombiner(
    method="exponential",
    forgetting_factor=0.95,
)
combiner.fit(forecasts_train=[f1, f2, f3], actual=y_train)

print(combiner.weights_)  # final weights: [0.50, 0.30, 0.20]

# Inspect weight evolution
import matplotlib.pyplot as plt
plt.plot(combiner.weights_history_)
plt.legend(["Model 1", "Model 2", "Model 3"])
plt.title("Time-Varying Combination Weights")
plt.show()

combined = combiner.combine([f1_test, f2_test, f3_test])
```

---

## OptimalCombiner

Optimal forecast combination following Bates & Granger (1969). Computes weights that minimize the variance of the combined forecast error using the error covariance matrix.

### Constructor

```python
OptimalCombiner(
    shrinkage: float = 0.0,
    min_obs: int = 20,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shrinkage` | `float` | `0.0` | Shrinkage toward equal weights in $[0, 1]$. `0` = fully optimal, `1` = equal weights |
| `min_obs` | `int` | `20` | Minimum observations required for covariance estimation |

!!! warning
    The optimal combiner requires reliable estimation of the error covariance matrix. With many models and few observations, consider using `shrinkage > 0` to stabilize the weights.

### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weights_` | `NDArray[np.float64]` | Optimal combination weights (sum to 1) |
| `cov_matrix_` | `NDArray[np.float64] \| None` | Error covariance matrix, shape `(K, K)` |
| `optimal_variance_` | `float \| None` | Variance of the combined forecast error |
| `individual_variances_` | `NDArray[np.float64] \| None` | Per-model error variances (diagonal of `cov_matrix_`) |

### Optimal Weights

For $K = 2$ models, the Bates-Granger optimal weight for model 1 is:

$$w_1^* = \frac{\sigma_2^2 - \sigma_{12}}{\sigma_1^2 + \sigma_2^2 - 2\sigma_{12}}$$

For general $K$, the optimal weights solve:

$$\mathbf{w}^* = \frac{\boldsymbol{\Sigma}^{-1} \mathbf{1}}{\mathbf{1}' \boldsymbol{\Sigma}^{-1} \mathbf{1}}$$

where $\boldsymbol{\Sigma}$ is the error covariance matrix.

### Example

```python
from forecastbox.combination import OptimalCombiner

combiner = OptimalCombiner(shrinkage=0.1, min_obs=30)
combiner.fit(forecasts_train=[f1, f2, f3], actual=y_train)

print(combiner.weights_)              # [0.48, 0.32, 0.20]
print(combiner.optimal_variance_)     # 0.15
print(combiner.individual_variances_) # [0.25, 0.30, 0.42]

combined = combiner.combine([f1_test, f2_test, f3_test])
```

---

## CombinationResult

Result container returned by combination workflows. Holds the combined forecast, weights, and diagnostic information.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `forecast` | `Forecast` | The combined forecast |
| `weights` | `NDArray[np.float64]` | Final combination weights |
| `method` | `str` | Name of the combination method used |
| `n_models` | `int` | Number of models combined |
| `model_names` | `list[str]` | Names of individual models |
| `diagnostics` | `dict[str, Any]` | Method-specific diagnostics (residuals, IC, variance, etc.) |

---

## See Also

- [Core API](core.md) — `Forecast` container used by all combiners
- [Evaluation API](evaluation.md) — Evaluate combined vs. individual forecasts
- [User Guide: Combination](../user-guide/combination/index.md) — Detailed usage guide with decision tree
- [Theory: Combination](../theory/combination-theory.md) — Mathematical foundations
- [Theory: BMA](../theory/bma-theory.md) — Bayesian Model Averaging theory
