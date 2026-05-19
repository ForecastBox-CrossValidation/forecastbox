---
title: "Conditional Forecasting Theory"
description: "Mathematical foundations of conditional forecasting in VAR models: hard and soft restrictions, Gibbs sampling, and Bayesian approaches."
---

# Conditional Forecasting Theory

!!! abstract "Key Takeaway"
    Conditional forecasting generates predictions of some variables given assumed paths (scenarios) for others. In VAR models, this requires solving a system of linear restrictions on future shocks. Hard conditions impose exact paths; soft conditions impose distributional constraints. The Waggoner-Zha (1999) algorithm and Gibbs sampling provide efficient computation of conditional forecasts and their uncertainty bands.

## Motivation

Unconditional forecasts project all variables forward using the estimated model dynamics alone. In practice, forecasters often want to answer questions like:

- "What is the GDP forecast **if** the central bank raises rates by 50 basis points?"
- "What is inflation **conditional on** oil prices staying below $80?"
- "What is the output path **under** a fiscal stimulus scenario?"

These require **conditional forecasts**: projections of endogenous variables given assumed paths for a subset of variables or structural shocks.

---

## Framework: VAR Models

### The Reduced-Form VAR

Consider a VAR($p$) model for $n$ variables:

$$
\mathbf{y}_t = \mathbf{c} + \mathbf{B}_1 \mathbf{y}_{t-1} + \cdots + \mathbf{B}_p \mathbf{y}_{t-p} + \mathbf{u}_t, \quad \mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})
$$

The $h$-step-ahead forecast (unconditional) from origin $T$ is:

$$
\hat{\mathbf{y}}_{T+h|T} = E[\mathbf{y}_{T+h} \mid \mathbf{y}_T, \mathbf{y}_{T-1}, \ldots]
$$

computed recursively. The forecast error is:

$$
\mathbf{y}_{T+h} - \hat{\mathbf{y}}_{T+h|T} = \sum_{s=0}^{h-1} \boldsymbol{\Phi}_s \mathbf{u}_{T+h-s}
$$

where $\boldsymbol{\Phi}_s$ are the **impulse response matrices** (moving average coefficients) with $\boldsymbol{\Phi}_0 = \mathbf{I}_n$.

### Structural Form

Let $\boldsymbol{\Sigma} = \mathbf{A}_0^{-1} \mathbf{D} (\mathbf{A}_0^{-1})'$ be a decomposition of the error covariance. The structural shocks are:

$$
\boldsymbol{\varepsilon}_t = \mathbf{A}_0 \mathbf{u}_t, \quad \boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{D})
$$

The forecast error in terms of structural shocks:

$$
\mathbf{y}_{T+h} - \hat{\mathbf{y}}_{T+h|T} = \sum_{s=0}^{h-1} \boldsymbol{\Psi}_s \boldsymbol{\varepsilon}_{T+h-s}
$$

where $\boldsymbol{\Psi}_s = \boldsymbol{\Phi}_s \mathbf{A}_0^{-1}$ are the **structural impulse responses**.

---

## Hard Conditions

### Definition

A **hard condition** specifies the exact value of some variables at some future dates. Let $\mathcal{C} = \{(i_1, h_1), (i_2, h_2), \ldots, (i_c, h_c)\}$ be the set of conditions, where variable $i_j$ at horizon $h_j$ is constrained to equal $y_{i_j, T+h_j}^*$.

### The Waggoner-Zha (1999) Algorithm

Stack the $c$ conditions and the $nH$ future shocks (where $H$ is the maximum forecast horizon). The conditions can be written as a linear system:

$$
\mathbf{R} \boldsymbol{\varepsilon} = \mathbf{r} - \mathbf{R} \hat{\mathbf{y}}^u
$$

where:

- $\boldsymbol{\varepsilon} = (\boldsymbol{\varepsilon}_{T+1}', \ldots, \boldsymbol{\varepsilon}_{T+H}')'$ is the $nH \times 1$ vector of future structural shocks
- $\hat{\mathbf{y}}^u$ is the unconditional forecast path
- $\mathbf{R}$ is a $c \times nH$ **restriction matrix** derived from the structural impulse responses $\boldsymbol{\Psi}_s$
- $\mathbf{r}$ is the $c \times 1$ vector of target values

**Construction of $\mathbf{R}$:** For a condition on variable $i$ at horizon $h$, the corresponding row of $\mathbf{R}$ contains the elements $\Psi_{s,ij}$ for appropriate $(s, j)$ pairs. Specifically, the entry mapping shock $j$ at period $T+\ell$ to variable $i$ at period $T+h$ is $\Psi_{h-\ell, ij}$ for $\ell \leq h$.

### Minimum-Entropy Solution

The system $\mathbf{R} \boldsymbol{\varepsilon} = \mathbf{d}$ (where $\mathbf{d} = \mathbf{r} - \mathbf{R} \hat{\mathbf{y}}^u$) is underdetermined when $c < nH$. Among all solutions, Waggoner and Zha select the one that minimizes the Kullback-Leibler divergence from the unconditional distribution of shocks:

$$
\boldsymbol{\varepsilon}^* = \arg\min_{\boldsymbol{\varepsilon}} \; \boldsymbol{\varepsilon}' \mathbf{D}^{-1}_H \boldsymbol{\varepsilon} \quad \text{s.t.} \quad \mathbf{R} \boldsymbol{\varepsilon} = \mathbf{d}
$$

where $\mathbf{D}_H = \mathbf{I}_H \otimes \mathbf{D}$ is the block-diagonal covariance of all future shocks.

**Derivation.** Using Lagrange multipliers:

$$
\mathcal{L} = \boldsymbol{\varepsilon}' \mathbf{D}_H^{-1} \boldsymbol{\varepsilon} + \boldsymbol{\lambda}' (\mathbf{R} \boldsymbol{\varepsilon} - \mathbf{d})
$$

First-order conditions:

$$
2 \mathbf{D}_H^{-1} \boldsymbol{\varepsilon} + \mathbf{R}' \boldsymbol{\lambda} = \mathbf{0} \implies \boldsymbol{\varepsilon} = -\frac{1}{2} \mathbf{D}_H \mathbf{R}' \boldsymbol{\lambda}
$$

Substituting into the constraint:

$$
\mathbf{R} \left(-\frac{1}{2} \mathbf{D}_H \mathbf{R}' \boldsymbol{\lambda}\right) = \mathbf{d} \implies \boldsymbol{\lambda} = -2 (\mathbf{R} \mathbf{D}_H \mathbf{R}')^{-1} \mathbf{d}
$$

Therefore:

$$
\boldsymbol{\varepsilon}^* = \mathbf{D}_H \mathbf{R}' (\mathbf{R} \mathbf{D}_H \mathbf{R}')^{-1} \mathbf{d}
$$

The conditional forecast is:

$$
\hat{\mathbf{y}}^c = \hat{\mathbf{y}}^u + \boldsymbol{\Psi}_H \boldsymbol{\varepsilon}^*
$$

where $\boldsymbol{\Psi}_H$ collects the structural impulse responses.

!!! note "Interpretation"
    The minimum-entropy solution distributes the required shocks across all structural disturbances in proportion to their variance, choosing the most "likely" combination of shocks consistent with the conditions. It reduces to the Cholesky conditional forecast when $\mathbf{A}_0$ is lower-triangular and conditions are on contemporaneous variables.

### Conditional Forecast Error Bands

The conditional forecast is a **point** on the conditional distribution. The full conditional distribution of unconditioned variables is:

$$
\mathbf{y}^{\text{free}} \mid \mathbf{y}^{\text{cond}} = \mathbf{r} \sim \mathcal{N}\!\left(\hat{\mathbf{y}}^{c,\text{free}}, \; \boldsymbol{\Sigma}^{\text{free}|\text{cond}}\right)
$$

where $\boldsymbol{\Sigma}^{\text{free}|\text{cond}}$ is the Schur complement of the conditioned block in the unconditional forecast covariance. The conditional variance is **smaller** than the unconditional variance --- conditioning reduces uncertainty.

---

## Soft Conditions

### Definition

A **soft condition** specifies a distributional constraint rather than an exact value. Examples:

- "GDP growth will be between 1% and 3%" (interval)
- "Inflation is expected to be 2% with standard deviation 0.5%" (distributional)
- "The interest rate path follows a given distribution" (density)

Formally, soft conditions specify that certain future values follow:

$$
y_{i, T+h} \sim \mathcal{N}(\mu_{i,h}^*, (\sigma_{i,h}^*)^2)
$$

or more generally, $y_{i, T+h} \in [a_{i,h}, b_{i,h}]$ with specified probability.

### Gibbs Sampling for Soft Conditions

**Waggoner and Zha (1999)** proposed a Gibbs sampler to draw from the conditional forecast distribution under soft conditions.

**Algorithm:**

1. **Initialize:** draw $\boldsymbol{\varepsilon}^{(0)}$ from the unconditional distribution $\mathcal{N}(\mathbf{0}, \mathbf{D}_H)$

2. **For each iteration $s = 1, \ldots, S$:**

    a. Partition the shock vector: $\boldsymbol{\varepsilon} = (\boldsymbol{\varepsilon}^{\text{free}}, \boldsymbol{\varepsilon}^{\text{cond}})$, where $\boldsymbol{\varepsilon}^{\text{cond}}$ are the shocks that drive the conditioned variables

    b. **Draw conditioned shocks:** sample $\boldsymbol{\varepsilon}^{\text{cond},(s)}$ from a truncated normal distribution consistent with the soft constraints:

    $$
    \varepsilon_j^{\text{cond},(s)} \sim \mathcal{TN}\!\left(0, D_{jj}; \; a_j^{\text{eff}}, b_j^{\text{eff}}\right)
    $$

    where $a_j^{\text{eff}}, b_j^{\text{eff}}$ are the effective bounds implied by the soft conditions and the current values of other shocks

    c. **Draw free shocks:** sample $\boldsymbol{\varepsilon}^{\text{free},(s)} \sim \mathcal{N}(\mathbf{0}, \mathbf{D}^{\text{free}})$

    d. **Compute forecast:** $\mathbf{y}^{(s)} = \hat{\mathbf{y}}^u + \boldsymbol{\Psi}_H \boldsymbol{\varepsilon}^{(s)}$

3. **Discard burn-in** and use the remaining draws to characterize the conditional forecast distribution

!!! warning "Convergence"
    The Gibbs sampler requires sufficient iterations to converge. Convergence diagnostics (trace plots, Gelman-Rubin $\hat{R}$) should be checked. Tight soft conditions (small intervals) slow convergence due to high rejection rates in the truncated normal draws.

---

## Bayesian Conditional Forecasting

### Posterior Predictive Distribution

In a Bayesian VAR, parameter uncertainty is incorporated. The conditional forecast distribution is:

$$
p(\mathbf{y}^{\text{free}} \mid \mathbf{y}^{\text{cond}} = \mathbf{r}, \mathbf{Y}) = \int p(\mathbf{y}^{\text{free}} \mid \mathbf{y}^{\text{cond}} = \mathbf{r}, \boldsymbol{\theta}) \, p(\boldsymbol{\theta} \mid \mathbf{Y}) \, d\boldsymbol{\theta}
$$

where $p(\boldsymbol{\theta} \mid \mathbf{Y})$ is the posterior of the VAR parameters.

**Monte Carlo integration:**

1. Draw $\boldsymbol{\theta}^{(s)} \sim p(\boldsymbol{\theta} \mid \mathbf{Y})$ (from the posterior)
2. Conditional on $\boldsymbol{\theta}^{(s)}$, compute the conditional forecast $\mathbf{y}^{\text{free},(s)}$ using the Waggoner-Zha algorithm
3. Repeat $S$ times to obtain the posterior predictive distribution

### Minnesota Prior and Conditional Forecasts

Under the Minnesota (Litterman) prior, the posterior is available in closed form (Normal-Inverse-Wishart). The conditional forecast algorithm integrates naturally with posterior draws:

$$
\boldsymbol{\Sigma}^{(s)} \sim \text{IW}(\hat{\mathbf{S}}, \hat{\nu}), \quad \text{vec}(\mathbf{B}^{(s)}) \mid \boldsymbol{\Sigma}^{(s)} \sim \mathcal{N}(\text{vec}(\hat{\mathbf{B}}), \boldsymbol{\Sigma}^{(s)} \otimes \hat{\mathbf{V}})
$$

Each draw provides a different set of impulse responses $\boldsymbol{\Psi}_s^{(s)}$, propagating parameter uncertainty into the conditional forecast.

---

## Relationship to Granger Causality

### Conditional Forecasts and Causal Ordering

The choice of which shocks to adjust when imposing conditions depends on the **structural identification**. Different identification schemes (Cholesky, sign restrictions, narrative) lead to different conditional forecasts because they imply different $\mathbf{A}_0$ matrices.

### Granger Causality and Forecast Improvement

If variable $x$ **Granger-causes** $y$, then conditioning on the future path of $x$ will improve the forecast of $y$:

$$
\text{MSE}(\hat{y}_{T+h} \mid x_{T+1}, \ldots, x_{T+h}) < \text{MSE}(\hat{y}_{T+h})
$$

The converse is also true: if conditioning on $x$ does not improve the forecast of $y$, then $x$ does not Granger-cause $y$.

**Formal result.** Let $\boldsymbol{\Sigma}_{yy|x}$ be the conditional forecast variance of $y$ given the path of $x$, and $\boldsymbol{\Sigma}_{yy}$ the unconditional forecast variance. Then:

$$
\boldsymbol{\Sigma}_{yy} - \boldsymbol{\Sigma}_{yy|x} = \boldsymbol{\Sigma}_{yx} \boldsymbol{\Sigma}_{xx}^{-1} \boldsymbol{\Sigma}_{xy} \geq 0
$$

This is positive semidefinite, with equality iff $y$ and $x$ are conditionally independent given the information set. The magnitude of the variance reduction measures the **forecasting value** of conditioning on $x$.

---

## Computational Aspects

### Efficiency

The Waggoner-Zha algorithm has computational complexity $O(c^3 + cnH)$ per conditional forecast, where $c$ is the number of conditions, $n$ the number of variables, and $H$ the horizon. This is efficient even for large VAR systems.

### Numerical Stability

When conditions span many horizons, the restriction matrix $\mathbf{R}$ can be ill-conditioned. Remedies:

- **QR decomposition** of $\mathbf{R}$ instead of direct inversion
- **Scaling** the conditions to have comparable magnitudes
- **Pivoted Cholesky** for the conditional covariance

---

## Key References

- Waggoner, D. F., & Zha, T. (1999). Conditional forecasts in dynamic multivariate models. *Review of Economics and Statistics*, 81(4), 639-651.
- Doan, T., Litterman, R., & Sims, C. A. (1984). Forecasting and conditional projection using realistic prior distributions. *Econometric Reviews*, 3(1), 1-100.
- Banbura, M., Giannone, D., & Reichlin, L. (2015). Conditional forecasts and scenario analysis with vector autoregressions for large cross-sections. *International Journal of Forecasting*, 31(3), 739-756.
- Andersson, M. K., Palmqvist, S., & Waggoner, D. F. (2010). Density-conditional forecasts in dynamic multivariate models. *Journal of Business & Economic Statistics*, 28(2), 266-277.
- Jarocinski, M. (2010). Conditional forecasts and uncertainty about forecast revisions in vector autoregressions. *Economics Letters*, 108(3), 257-259.
- Luetkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. Chapter 5.

## See Also

- [Combination Theory](combination-theory.md) --- combining conditional forecasts from different scenarios
- [Nowcasting Theory](nowcasting-theory.md) --- nowcasting as conditional projection with partial data
- [BMA Theory](bma-theory.md) --- Bayesian model averaging for conditional forecasts
- [References](references.md) --- complete bibliography
