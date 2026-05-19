---
title: "Forecast Evaluation Theory"
description: "Theory of forecast evaluation: loss functions, their properties, consistency, robustness, and optimal forecasts under asymmetric loss."
---

# Forecast Evaluation Theory

## Introduction

Forecast evaluation determines which model produces the best predictions. The answer depends critically on the **loss function** used to measure forecast accuracy. This section develops the mathematical theory of loss functions, their properties, and their implications for model ranking and optimal forecasting.

---

## Loss Functions: Foundations

### Definition

A **loss function** $L(e_t)$ maps forecast errors $e_t = y_t - \hat{y}_t$ to non-negative real numbers:

$$
L : \mathbb{R} \to \mathbb{R}_{\geq 0}, \quad L(0) = 0
$$

The **expected loss** (risk) of a forecast is:

$$
R(\hat{y}) = E[L(y_t - \hat{y}_t)]
$$

The **optimal forecast** under loss $L$ is:

$$
\hat{y}_t^* = \arg\min_{\hat{y}} \; E[L(y_t - \hat{y}) \mid \mathcal{F}_{t-1}]
$$

### Standard Loss Functions

| Loss Function | Formula | Optimal Forecast |
|:---|:---|:---|
| Squared Error (SE) | $L(e) = e^2$ | Conditional mean |
| Absolute Error (AE) | $L(e) = \lvert e \rvert$ | Conditional median |
| Percentage Error (APE) | $L(e) = \lvert e/y \rvert$ | Depends on distribution |
| LINEX | $L(e) = \exp(ae) - ae - 1$ | $E[y] - \frac{a}{2}\text{Var}(y)$ (Gaussian) |
| Quantile | $L(e) = e(\tau - \mathbb{1}_{e < 0})$ | $\tau$-th quantile |

---

## Properties of Loss Functions

### Consistency (Patton, 2011)

A loss function $L$ is **consistent** for a functional $T$ of the conditional distribution $F_{t|t-1}$ if:

$$
E[L(y_t - T(F_{t|t-1}))] \leq E[L(y_t - g)] \quad \text{for all measurable } g
$$

!!! note "Why consistency matters"
    If a loss function is not consistent for the functional being evaluated, the ranking of models can be reversed. For example, evaluating mean forecasts with absolute error (consistent for the median) can select the wrong model.

**Theorem (Patton, 2011).** *A loss function $L$ is consistent for the conditional mean if and only if it can be written as:*

$$
L(e) = a(y)\, e^2 + b(y)\, e + c(y)
$$

*where $a(y) > 0$ and $b, c$ are arbitrary functions of $y$ alone.*

**Proof sketch.** The conditional mean minimizes $E[L(y - \hat{y}) \mid \mathcal{F}_{t-1}]$. Taking the first-order condition:

$$
\frac{\partial}{\partial \hat{y}} E[L(y - \hat{y})] = -E[L'(y - \hat{y})] = 0
$$

For this to hold at $\hat{y} = E[y \mid \mathcal{F}_{t-1}]$ for all conditional distributions, $L'$ must be an odd function centered at zero, which implies $L''$ is even and positive. The most general form satisfying this with arbitrary dependence on $y$ is the stated expression. $\blacksquare$

### Homogeneity

A loss function is **homogeneous of degree $p$** if:

$$
L(\lambda e) = \lambda^p L(e) \quad \text{for all } \lambda > 0
$$

- SE is homogeneous of degree 2
- AE is homogeneous of degree 1
- Homogeneity ensures the loss is scale-invariant up to a multiplicative constant

### Robustness

A loss function is **robust** if the implied optimal forecast is bounded:

$$
\sup_y \left|\frac{\partial L(y - \hat{y})}{\partial \hat{y}}\right| < \infty
$$

- SE is **not** robust (influence function is unbounded)
- AE is robust (bounded influence)
- Huber loss provides a compromise

---

## Asymmetric Loss Functions

### Motivation

In many economic applications, overprediction and underprediction carry different costs. For example:

- **Inflation forecasts**: underestimation may be costlier for central banks
- **GDP forecasts**: overestimation of growth may delay necessary policy action
- **Inventory management**: stockouts (underprediction) are costlier than excess inventory

### LINEX Loss

The **LINear-EXponential** loss function (Varian, 1975; Zellner, 1986):

$$
L_{\text{LINEX}}(e; a) = \exp(ae) - ae - 1, \quad a \neq 0
$$

**Properties:**

- For $a > 0$: penalizes positive errors (underprediction of $y$) more heavily
- For $a < 0$: penalizes negative errors (overprediction) more heavily
- As $a \to 0$: $L_{\text{LINEX}}(e) \to \frac{1}{2} e^2$ (symmetric, quadratic)

**Derivation of the limiting case.** Taylor expand around $a = 0$:

$$
\exp(ae) = 1 + ae + \frac{(ae)^2}{2} + O(a^3)
$$

$$
L_{\text{LINEX}} = 1 + ae + \frac{a^2 e^2}{2} + O(a^3) - ae - 1 = \frac{a^2 e^2}{2} + O(a^3)
$$

Dividing by $a^2/2$ gives $e^2$ as $a \to 0$.

**Optimal forecast under LINEX (Gaussian case).** If $y_t \mid \mathcal{F}_{t-1} \sim \mathcal{N}(\mu_t, \sigma_t^2)$:

$$
\hat{y}_t^* = \mu_t - \frac{a \sigma_t^2}{2}
$$

**Proof.** The expected LINEX loss is:

$$
E[L_{\text{LINEX}}] = E[\exp(a(y_t - \hat{y}_t))] - a E[y_t - \hat{y}_t] - 1
$$

Using the moment-generating function of the Gaussian:

$$
E[\exp(a(y_t - \hat{y}_t))] = \exp\!\left(a(\mu_t - \hat{y}_t) + \frac{a^2 \sigma_t^2}{2}\right)
$$

Taking the derivative with respect to $\hat{y}_t$ and setting to zero:

$$
-a \exp\!\left(a(\mu_t - \hat{y}_t) + \frac{a^2 \sigma_t^2}{2}\right) + a = 0
$$

$$
\exp\!\left(a(\mu_t - \hat{y}_t) + \frac{a^2 \sigma_t^2}{2}\right) = 1
$$

$$
a(\mu_t - \hat{y}_t) + \frac{a^2 \sigma_t^2}{2} = 0 \implies \hat{y}_t^* = \mu_t - \frac{a \sigma_t^2}{2} \quad \blacksquare
$$

### Double Power Loss

The **double power** family nests many common loss functions:

$$
L_{p,q}(e) = \begin{cases} c_1 |e|^p & \text{if } e \geq 0 \\ c_2 |e|^q & \text{if } e < 0 \end{cases}
$$

| Parameters | Loss Function |
|:---|:---|
| $p = q = 2$, $c_1 = c_2$ | MSE |
| $p = q = 1$, $c_1 = c_2$ | MAE |
| $p = q = 1$, $c_1 \neq c_2$ | Asymmetric linear (quantile-like) |
| $p = 2$, $q = 2$, $c_1 \neq c_2$ | Asymmetric quadratic |

---

## Loss Functions and Model Ranking

### The Ranking Problem

**Key result:** Different loss functions can produce different model rankings. If model A has lower MSE but model B has lower MAE, the choice between them depends on the decision-maker's loss function.

Formally, let $R_L^{(i)} = E[L(e_t^{(i)})]$ be the risk of model $i$ under loss $L$. The ranking:

$$
R_L^{(A)} < R_L^{(B)} \quad \text{does NOT imply} \quad R_{L'}^{(A)} < R_{L'}^{(B)} \text{ for } L' \neq L
$$

### Consistent Ranking

**Theorem (Patton, 2011).** *If two forecasts target the same functional (e.g., both target the conditional mean), then any loss function consistent for that functional produces the same ranking.*

This means that for comparing mean forecasts, MSE, MASE, and any other loss satisfying the consistency condition will agree on which model is better.

!!! warning "Implication for practice"
    When using a loss function to evaluate forecasts, ensure it is **consistent** for the type of forecast being produced. Using MAE to evaluate mean forecasts, or MSE to evaluate median forecasts, can lead to incorrect model selection.

### Encompassing and Ranking

If model A **encompasses** model B (i.e., the optimal combination weight on B is zero), then A dominates B under **all** loss functions consistent for the conditional mean. Encompassing implies universal superiority within the consistent loss class.

---

## Relative Loss and Comparison Tests

### Diebold-Mariano Framework

To compare models $A$ and $B$, define the **loss differential**:

$$
d_t = L(e_t^{(A)}) - L(e_t^{(B)})
$$

The null hypothesis of equal predictive ability is:

$$
H_0: E[d_t] = 0
$$

The Diebold-Mariano statistic:

$$
\text{DM} = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

where $\hat{V}(\bar{d})$ accounts for serial correlation in $d_t$ using HAC estimation.

### The Giacomini-White Extension

**Giacomini and White (2006)** extended the DM test to:

1. Allow for **nested models** (where DM has non-standard asymptotics)
2. Test **conditional** predictive ability: $E[d_t \mid \mathcal{F}_{t-1}] = 0$
3. Allow for **estimated parameters** (rolling/recursive estimation)

The test statistic is:

$$
\text{GW} = T \, \bar{\mathbf{h}}' \hat{\mathbf{V}}^{-1} \bar{\mathbf{h}} \xrightarrow{d} \chi^2(q)
$$

where $\mathbf{h}_t = \mathbf{z}_{t-1} d_t$ and $\mathbf{z}_{t-1}$ is a vector of test functions.

---

## Optimal Forecasts Under General Loss

### Characterization

Under a general loss function $L$, the optimal point forecast solves:

$$
\hat{y}_t^* = \arg\min_{\hat{y}} \int L(y - \hat{y}) \, f(y \mid \mathcal{F}_{t-1}) \, dy
$$

The first-order condition is:

$$
\int L'(y - \hat{y}_t^*) \, f(y \mid \mathcal{F}_{t-1}) \, dy = 0
$$

This defines a **generalized expectation** of the predictive distribution.

### Density Forecast Evaluation

When the full predictive density $\hat{f}_t(y)$ is available, evaluation uses the **logarithmic score**:

$$
S_{\text{log}}(\hat{f}_t, y_t) = -\log \hat{f}_t(y_t)
$$

or the **continuous ranked probability score** (CRPS):

$$
\text{CRPS}(\hat{F}_t, y_t) = \int_{-\infty}^{\infty} \left(\hat{F}_t(z) - \mathbb{1}_{z \geq y_t}\right)^2 dz
$$

The CRPS has the attractive property of reducing to the MAE when the predictive distribution is a point mass.

---

## Key References

- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246-256.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Giacomini, R., & White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545-1578.
- Varian, H. R. (1975). A Bayesian approach to real estate assessment. In S. E. Fienberg & A. Zellner (Eds.), *Studies in Bayesian Econometrics and Statistics* (pp. 195-208). North-Holland.
- Zellner, A. (1986). Bayesian estimation and prediction using asymmetric loss functions. *Journal of the American Statistical Association*, 81(394), 446-451.
- Gneiting, T. (2011). Making and evaluating point forecasts. *Journal of the American Statistical Association*, 106(494), 746-762.
