---
title: "Forecast Combination Theory"
description: "Mathematical foundations of forecast combination: Bates-Granger result, optimality conditions, and the forecast combination puzzle."
---

# Forecast Combination Theory

## Introduction

Forecast combination is the practice of aggregating predictions from multiple models into a single composite forecast. The idea, first formalized by **Bates and Granger (1969)**, rests on a simple but powerful insight: unless one model is uniformly superior, a weighted average of forecasts exploits the complementary information contained in each model and reduces overall prediction error.

This section develops the mathematical theory underlying the combination methods available in forecastbox.

---

## The Bates-Granger Framework

### Setup

Consider $N$ competing forecasts $\hat{y}_t^{(1)}, \hat{y}_t^{(2)}, \ldots, \hat{y}_t^{(N)}$ for a target variable $y_t$. Define the forecast error of model $i$ as:

$$
e_t^{(i)} = y_t - \hat{y}_t^{(i)}, \quad i = 1, \ldots, N
$$

A **linear combination** of forecasts takes the form:

$$
\hat{y}_t^{c} = \sum_{i=1}^{N} w_i \, \hat{y}_t^{(i)}, \quad \text{with} \quad \sum_{i=1}^{N} w_i = 1
$$

The combined forecast error is:

$$
e_t^{c} = y_t - \hat{y}_t^{c} = \sum_{i=1}^{N} w_i \, e_t^{(i)}
$$

### Two-Forecast Case

For $N = 2$ with weight $w$ on the first forecast:

$$
\hat{y}_t^{c} = w \, \hat{y}_t^{(1)} + (1 - w) \, \hat{y}_t^{(2)}
$$

The MSE of the combined forecast is:

$$
\text{MSE}(\hat{y}^{c}) = w^2 \sigma_1^2 + (1 - w)^2 \sigma_2^2 + 2w(1 - w) \rho \sigma_1 \sigma_2
$$

where $\sigma_i^2 = E[(e_t^{(i)})^2]$ and $\rho = \text{Corr}(e_t^{(1)}, e_t^{(2)})$.

### Optimal Weight Derivation

Minimizing $\text{MSE}(\hat{y}^{c})$ with respect to $w$:

$$
\frac{\partial \, \text{MSE}}{\partial w} = 2w \sigma_1^2 - 2(1 - w) \sigma_2^2 + 2(1 - 2w) \rho \sigma_1 \sigma_2 = 0
$$

Solving for $w$:

$$
w^* = \frac{\sigma_2^2 - \rho \sigma_1 \sigma_2}{\sigma_1^2 + \sigma_2^2 - 2\rho \sigma_1 \sigma_2}
$$

!!! note "Special cases"
    - **Uncorrelated errors** ($\rho = 0$): $w^* = \sigma_2^2 / (\sigma_1^2 + \sigma_2^2)$, i.e., inverse-variance weighting.
    - **Equal variances** ($\sigma_1 = \sigma_2$): $w^* = 1/2$, i.e., simple average regardless of $\rho$.

### Proof: Combination Never Increases MSE

**Theorem (Bates-Granger, 1969).** *Under the optimal weight $w^*$, the MSE of the combined forecast satisfies:*

$$
\text{MSE}(\hat{y}^{c}) \leq \min\!\big(\sigma_1^2, \, \sigma_2^2\big)
$$

*with equality if and only if one model encompasses the other.*

**Proof.** Substituting $w^*$ into the MSE expression:

$$
\text{MSE}^* = \frac{\sigma_1^2 \sigma_2^2 (1 - \rho^2)}{\sigma_1^2 + \sigma_2^2 - 2\rho \sigma_1 \sigma_2}
$$

We need to show $\text{MSE}^* \leq \sigma_1^2$. This is equivalent to:

$$
\frac{\sigma_1^2 \sigma_2^2 (1 - \rho^2)}{\sigma_1^2 + \sigma_2^2 - 2\rho \sigma_1 \sigma_2} \leq \sigma_1^2
$$

$$
\sigma_2^2 (1 - \rho^2) \leq \sigma_1^2 + \sigma_2^2 - 2\rho \sigma_1 \sigma_2
$$

$$
\sigma_2^2 - \rho^2 \sigma_2^2 \leq \sigma_1^2 + \sigma_2^2 - 2\rho \sigma_1 \sigma_2
$$

$$
0 \leq \sigma_1^2 - 2\rho \sigma_1 \sigma_2 + \rho^2 \sigma_2^2
$$

$$
0 \leq (\sigma_1 - \rho \sigma_2)^2
$$

The last inequality holds for all real values, with equality iff $\sigma_1 = \rho \sigma_2$, i.e., when forecast 1 is a noisy version of forecast 2 (encompassing). By symmetry, $\text{MSE}^* \leq \sigma_2^2$ as well. $\blacksquare$

---

## General $N$-Forecast Case

### Optimal Combination Weights

For $N$ forecasts with error covariance matrix $\boldsymbol{\Sigma} = E[\mathbf{e}_t \mathbf{e}_t']$, the constrained optimization problem is:

$$
\min_{\mathbf{w}} \; \mathbf{w}' \boldsymbol{\Sigma} \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}' \mathbf{1} = 1
$$

Using a Lagrange multiplier $\lambda$:

$$
\mathcal{L} = \mathbf{w}' \boldsymbol{\Sigma} \mathbf{w} - \lambda (\mathbf{w}' \mathbf{1} - 1)
$$

First-order conditions:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2 \boldsymbol{\Sigma} \mathbf{w} - \lambda \mathbf{1} = 0 \implies \mathbf{w} = \frac{\lambda}{2} \boldsymbol{\Sigma}^{-1} \mathbf{1}
$$

Applying the constraint:

$$
\mathbf{w}^* = \frac{\boldsymbol{\Sigma}^{-1} \mathbf{1}}{\mathbf{1}' \boldsymbol{\Sigma}^{-1} \mathbf{1}}
$$

The resulting MSE is:

$$
\text{MSE}^* = \frac{1}{\mathbf{1}' \boldsymbol{\Sigma}^{-1} \mathbf{1}}
$$

!!! warning "Estimation risk"
    The optimal weights require knowledge of $\boldsymbol{\Sigma}$, which must be estimated from data. With $N$ models and $T$ observations, the estimation error in $\hat{\boldsymbol{\Sigma}}$ can dominate the theoretical gains, especially when $N/T$ is not small. This is a key driver of the **forecast combination puzzle**.

### Bias-Corrected Combination

When forecasts may be biased, the unconstrained combination includes an intercept:

$$
\hat{y}_t^{c} = \alpha + \sum_{i=1}^{N} w_i \, \hat{y}_t^{(i)}
$$

This is equivalent to the **Granger-Ramanathan (1984)** regression:

$$
y_t = \alpha + \sum_{i=1}^{N} w_i \, \hat{y}_t^{(i)} + \varepsilon_t
$$

estimated by OLS. The weights are not constrained to sum to one, and $\alpha$ absorbs systematic bias.

---

## The Forecast Combination Puzzle

### Statement of the Puzzle

**Empirical finding** (Stock & Watson, 2004; Smith & Wallis, 2009; Genre et al., 2013): the simple average $w_i = 1/N$ frequently outperforms theoretically optimal combination methods in out-of-sample forecasting.

This is puzzling because the simple average ignores all information about relative model accuracy and error correlations.

### Explanations

#### 1. Estimation Error

The optimal weights depend on $\boldsymbol{\Sigma}$. **Smith and Wallis (2009)** showed that when the true weight vector is close to equal weights, the estimation error in $\hat{\mathbf{w}}^*$ is large relative to the potential MSE reduction:

$$
\text{MSE}(\hat{y}^{c}_{\text{opt}}) = \text{MSE}^* + \underbrace{E\!\left[(\hat{\mathbf{w}} - \mathbf{w}^*)' \boldsymbol{\Sigma} (\hat{\mathbf{w}} - \mathbf{w}^*)\right]}_{\text{estimation error term}}
$$

When the estimation error term exceeds $\text{MSE}(1/N) - \text{MSE}^*$, equal weights dominate.

#### 2. Structural Instability

If the data-generating process changes over time, the optimal weights are time-varying:

$$
\mathbf{w}_t^* = \frac{\boldsymbol{\Sigma}_t^{-1} \mathbf{1}}{\mathbf{1}' \boldsymbol{\Sigma}_t^{-1} \mathbf{1}}
$$

Weights estimated on historical data may be poorly suited for future periods. The simple average is immune to this problem since it does not depend on any estimated quantities.

#### 3. Finite-Sample Bias

**Timmermann (2006)** demonstrated that in finite samples, the OLS combination estimator is biased due to multicollinearity among forecasts. When individual forecasts are highly correlated (as is typical in macroeconomic applications), the design matrix is near-singular, inflating the variance of estimated weights.

#### 4. The Bias-Variance Tradeoff

The equal-weight combination can be understood through the bias-variance decomposition:

$$
\text{MSE}(\hat{\mathbf{w}}) = \underbrace{\left\|E[\hat{\mathbf{w}}] - \mathbf{w}^*\right\|_{\boldsymbol{\Sigma}}^2}_{\text{bias}^2} + \underbrace{\text{tr}\!\left(\boldsymbol{\Sigma} \, \text{Var}(\hat{\mathbf{w}})\right)}_{\text{variance}}
$$

Equal weights have maximum bias (unless $\mathbf{w}^* = \mathbf{1}/N$) but zero variance. Optimal weights have zero bias but maximum variance. The simple average wins when variance dominates.

---

## Optimality Conditions by Method

### Simple Average

Optimal when:

- Models are exchangeable (no prior reason to prefer one over another)
- Sample size is small relative to the number of models
- Structural breaks are suspected

### Inverse-MSE Weighting

Weights proportional to $1/\text{MSE}_i$:

$$
w_i = \frac{1/\text{MSE}_i}{\sum_{j=1}^N 1/\text{MSE}_j}
$$

Optimal when errors are **uncorrelated** and the diagonal of $\boldsymbol{\Sigma}$ is well estimated.

### OLS Combination (Granger-Ramanathan)

Optimal when:

- $T \gg N$ (enough data to estimate $N + 1$ parameters)
- Forecasts may be biased
- Error correlations carry useful information

### Constrained Least Squares

Imposing $w_i \geq 0$ and $\sum w_i = 1$. Optimal when:

- Negative weights are economically implausible
- Regularization from constraints reduces estimation error

### Time-Varying Weights

Using exponential discounting or regime-switching:

$$
w_{i,t} \propto \exp\!\left(-\eta \sum_{s=1}^{t-1} \delta^{t-1-s} L(e_s^{(i)})\right)
$$

Optimal when the relative performance of models changes over time.

---

## Combination Under Misspecification

When all models are misspecified, **Elliott (2011)** showed that the combination problem becomes:

$$
\min_{\mathbf{w}} \; E\!\left[\left(y_t - \mathbf{w}' \hat{\mathbf{y}}_t\right)^2\right] = \min_{\mathbf{w}} \; \mathbf{w}' \mathbf{M} \mathbf{w} - 2\mathbf{w}' \mathbf{m} + \sigma_y^2
$$

where $\mathbf{M} = E[\hat{\mathbf{y}}_t \hat{\mathbf{y}}_t']$ and $\mathbf{m} = E[y_t \hat{\mathbf{y}}_t]$. The solution is:

$$
\mathbf{w}^* = \mathbf{M}^{-1} \mathbf{m}
$$

which does **not** require $\sum w_i = 1$ and may include negative weights (short positions).

---

## Key References

- Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly*, 20(4), 451-468.
- Granger, C. W. J., & Ramanathan, R. (1984). Improved methods of combining forecasts. *Journal of Forecasting*, 3(2), 197-204.
- Timmermann, A. (2006). Forecast combinations. In G. Elliott, C. W. J. Granger, & A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 1, pp. 135-196). Elsevier.
- Smith, J., & Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. *Oxford Bulletin of Economics and Statistics*, 71(3), 331-355.
- Stock, J. H., & Watson, M. W. (2004). Combination forecasts of output growth in a seven-country data set. *Journal of Forecasting*, 23(6), 405-430.
- Elliott, G. (2011). Averaging and the optimal combination of forecasts. *Working Paper*, UC San Diego.
- Genre, V., et al. (2013). Combining expert forecasts: Can anything beat the simple average? *International Journal of Forecasting*, 29(1), 108-121.
