---
title: "Nowcasting Theory"
description: "Mathematical foundations of nowcasting: dynamic factor models, state-space representation, Kalman filter, EM algorithm, and bridge equations."
---

# Nowcasting Theory

## Introduction

**Nowcasting** is the prediction of the present or very recent past, when key economic indicators have not yet been released. The term, borrowed from meteorology, was formalized in the macroeconomic context by **Giannone, Reichlin, and Small (2008)**.

The core challenge is exploiting a large panel of mixed-frequency, irregularly released data to extract a timely signal about the state of the economy. This section develops the mathematical framework underlying the nowcasting methods in forecastbox.

---

## The Dynamic Factor Model

### Motivation

Macroeconomic variables are driven by a small number of common factors. The **dynamic factor model** (DFM) captures this structure:

$$
\mathbf{x}_t = \boldsymbol{\Lambda} \mathbf{f}_t + \boldsymbol{\xi}_t
$$

where:

- $\mathbf{x}_t \in \mathbb{R}^n$ is the vector of observed variables (standardized)
- $\mathbf{f}_t \in \mathbb{R}^r$ is the vector of latent common factors ($r \ll n$)
- $\boldsymbol{\Lambda} \in \mathbb{R}^{n \times r}$ is the factor loading matrix
- $\boldsymbol{\xi}_t \in \mathbb{R}^n$ is the idiosyncratic component

### Factor Dynamics

The factors follow a VAR($p$) process:

$$
\mathbf{f}_t = \mathbf{A}_1 \mathbf{f}_{t-1} + \mathbf{A}_2 \mathbf{f}_{t-2} + \cdots + \mathbf{A}_p \mathbf{f}_{t-p} + \boldsymbol{\eta}_t
$$

where $\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$.

The idiosyncratic components are assumed to follow independent AR processes:

$$
\xi_{i,t} = \rho_i \xi_{i,t-1} + \epsilon_{i,t}, \quad \epsilon_{i,t} \sim \mathcal{N}(0, \sigma_i^2)
$$

!!! note "Approximate factor model"
    The assumption of independent idiosyncratic components can be relaxed to allow for weak cross-sectional correlation (the **approximate** factor model of Chamberlain and Rothschild, 1983). In practice, forecastbox assumes diagonal or block-diagonal idiosyncratic covariance.

---

## State-Space Representation

### Companion Form

The DFM can be written in state-space form. Define the state vector:

$$
\boldsymbol{\alpha}_t = \begin{pmatrix} \mathbf{f}_t \\ \mathbf{f}_{t-1} \\ \vdots \\ \mathbf{f}_{t-p+1} \end{pmatrix} \in \mathbb{R}^{rp}
$$

**Transition equation:**

$$
\boldsymbol{\alpha}_t = \mathbf{T} \boldsymbol{\alpha}_{t-1} + \mathbf{R} \boldsymbol{\eta}_t
$$

where:

$$
\mathbf{T} = \begin{pmatrix}
\mathbf{A}_1 & \mathbf{A}_2 & \cdots & \mathbf{A}_{p-1} & \mathbf{A}_p \\
\mathbf{I}_r & \mathbf{0} & \cdots & \mathbf{0} & \mathbf{0} \\
\mathbf{0} & \mathbf{I}_r & \cdots & \mathbf{0} & \mathbf{0} \\
\vdots & & \ddots & & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \mathbf{I}_r & \mathbf{0}
\end{pmatrix}, \quad
\mathbf{R} = \begin{pmatrix} \mathbf{I}_r \\ \mathbf{0} \\ \vdots \\ \mathbf{0} \end{pmatrix}
$$

**Measurement equation:**

$$
\mathbf{x}_t = \mathbf{Z}_t \boldsymbol{\alpha}_t + \boldsymbol{\xi}_t
$$

where $\mathbf{Z}_t = (\boldsymbol{\Lambda} \; \mathbf{0} \; \cdots \; \mathbf{0})$ when all data are available. The **key feature** for nowcasting: when some variables are missing at time $t$, the corresponding rows of $\mathbf{Z}_t$ are simply removed.

### Handling Missing Data

Let $\mathcal{O}_t \subseteq \{1, \ldots, n\}$ denote the set of variables observed at time $t$. The measurement equation becomes:

$$
\mathbf{x}_t^{\mathcal{O}} = \mathbf{Z}_t^{\mathcal{O}} \boldsymbol{\alpha}_t + \boldsymbol{\xi}_t^{\mathcal{O}}
$$

where $\mathbf{x}_t^{\mathcal{O}}$, $\mathbf{Z}_t^{\mathcal{O}}$, and $\boldsymbol{\xi}_t^{\mathcal{O}}$ are the subsets corresponding to observed variables. This is the mechanism by which the DFM naturally handles the **ragged edge** of real-time data.

---

## The Kalman Filter

### Prediction Step

Given the filtered state $\boldsymbol{\alpha}_{t-1|t-1}$ and its covariance $\mathbf{P}_{t-1|t-1}$:

$$
\boldsymbol{\alpha}_{t|t-1} = \mathbf{T} \boldsymbol{\alpha}_{t-1|t-1}
$$

$$
\mathbf{P}_{t|t-1} = \mathbf{T} \mathbf{P}_{t-1|t-1} \mathbf{T}' + \mathbf{R} \mathbf{Q} \mathbf{R}'
$$

### Innovation

The prediction error and its covariance:

$$
\mathbf{v}_t = \mathbf{x}_t^{\mathcal{O}} - \mathbf{Z}_t^{\mathcal{O}} \boldsymbol{\alpha}_{t|t-1}
$$

$$
\mathbf{F}_t = \mathbf{Z}_t^{\mathcal{O}} \mathbf{P}_{t|t-1} (\mathbf{Z}_t^{\mathcal{O}})' + \mathbf{H}_t^{\mathcal{O}}
$$

where $\mathbf{H}_t^{\mathcal{O}} = \text{diag}(\sigma_i^2 : i \in \mathcal{O}_t)$.

### Update Step

The Kalman gain:

$$
\mathbf{K}_t = \mathbf{P}_{t|t-1} (\mathbf{Z}_t^{\mathcal{O}})' \mathbf{F}_t^{-1}
$$

The filtered state and covariance:

$$
\boldsymbol{\alpha}_{t|t} = \boldsymbol{\alpha}_{t|t-1} + \mathbf{K}_t \mathbf{v}_t
$$

$$
\mathbf{P}_{t|t} = (\mathbf{I} - \mathbf{K}_t \mathbf{Z}_t^{\mathcal{O}}) \mathbf{P}_{t|t-1}
$$

!!! tip "Interpretation of the Kalman gain"
    $\mathbf{K}_t$ determines how much the state estimate is revised in response to new data. When $\mathbf{F}_t$ is large (high uncertainty in the prediction), the gain is small and new data has little effect. When $\mathbf{P}_{t|t-1}$ is large (high uncertainty in the state), the gain is large and new data is influential.

### The Kalman Smoother

The **Rauch-Tung-Striebel** (RTS) smoother produces smoothed estimates using all available data $\mathbf{x}_1, \ldots, \mathbf{x}_T$:

**Backward recursion** (for $t = T-1, T-2, \ldots, 1$):

$$
\mathbf{J}_t = \mathbf{P}_{t|t} \mathbf{T}' \mathbf{P}_{t+1|t}^{-1}
$$

$$
\boldsymbol{\alpha}_{t|T} = \boldsymbol{\alpha}_{t|t} + \mathbf{J}_t (\boldsymbol{\alpha}_{t+1|T} - \boldsymbol{\alpha}_{t+1|t})
$$

$$
\mathbf{P}_{t|T} = \mathbf{P}_{t|t} + \mathbf{J}_t (\mathbf{P}_{t+1|T} - \mathbf{P}_{t+1|t}) \mathbf{J}_t'
$$

The smoother uses future observations to refine past state estimates, yielding lower variance than the filter: $\mathbf{P}_{t|T} \leq \mathbf{P}_{t|t}$ in the positive semidefinite sense.

---

## EM Algorithm for Parameter Estimation

### The Problem

The DFM parameters $\boldsymbol{\theta} = \{\boldsymbol{\Lambda}, \mathbf{A}_1, \ldots, \mathbf{A}_p, \mathbf{Q}, \sigma_1^2, \ldots, \sigma_n^2\}$ must be estimated from data with missing observations. Maximum likelihood estimation via direct numerical optimization is infeasible for large $n$. The **EM algorithm** (Dempster, Laird, and Rubin, 1977) provides an iterative solution.

### Log-Likelihood

The complete-data log-likelihood (if factors were observed) is:

$$
\ell(\boldsymbol{\theta}) = \ell_{\text{transition}}(\boldsymbol{\theta}) + \ell_{\text{measurement}}(\boldsymbol{\theta})
$$

$$
\ell_{\text{transition}} = -\frac{T}{2} \log |\mathbf{Q}| - \frac{1}{2} \sum_{t=1}^{T} (\mathbf{f}_t - \mathbf{A} \mathbf{f}_{t-1}^{+})' \mathbf{Q}^{-1} (\mathbf{f}_t - \mathbf{A} \mathbf{f}_{t-1}^{+})
$$

$$
\ell_{\text{measurement}} = -\frac{1}{2} \sum_{t=1}^{T} \sum_{i \in \mathcal{O}_t} \left[\log \sigma_i^2 + \frac{(x_{i,t} - \boldsymbol{\lambda}_i' \mathbf{f}_t)^2}{\sigma_i^2}\right]
$$

where $\mathbf{f}_{t-1}^{+} = (\mathbf{f}_{t-1}', \ldots, \mathbf{f}_{t-p}')' $ and $\mathbf{A} = (\mathbf{A}_1, \ldots, \mathbf{A}_p)$.

### E-Step

Compute the expected complete-data log-likelihood given observed data and current parameters $\boldsymbol{\theta}^{(k)}$:

$$
Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(k)}) = E\!\left[\ell(\boldsymbol{\theta}) \mid \mathbf{x}_1^{\mathcal{O}}, \ldots, \mathbf{x}_T^{\mathcal{O}}; \boldsymbol{\theta}^{(k)}\right]
$$

This requires the **sufficient statistics** from the Kalman smoother:

$$
\hat{\mathbf{f}}_{t|T} = E[\mathbf{f}_t \mid \mathbf{x}_1^{\mathcal{O}}, \ldots, \mathbf{x}_T^{\mathcal{O}}; \boldsymbol{\theta}^{(k)}]
$$

$$
\hat{\mathbf{P}}_{t|T} = \text{Cov}(\mathbf{f}_t \mid \mathbf{x}_1^{\mathcal{O}}, \ldots, \mathbf{x}_T^{\mathcal{O}}; \boldsymbol{\theta}^{(k)})
$$

$$
\hat{\mathbf{P}}_{t,t-1|T} = \text{Cov}(\mathbf{f}_t, \mathbf{f}_{t-1} \mid \mathbf{x}_1^{\mathcal{O}}, \ldots, \mathbf{x}_T^{\mathcal{O}}; \boldsymbol{\theta}^{(k)})
$$

Define the smoothed second-moment matrices:

$$
\mathbf{S}_{ff} = \sum_{t=1}^{T} (\hat{\mathbf{f}}_{t|T} \hat{\mathbf{f}}_{t|T}' + \hat{\mathbf{P}}_{t|T})
$$

$$
\mathbf{S}_{f^+f^+} = \sum_{t=1}^{T} (\hat{\mathbf{f}}_{t-1|T}^{+} (\hat{\mathbf{f}}_{t-1|T}^{+})' + \hat{\mathbf{P}}_{t-1|T}^{+})
$$

$$
\mathbf{S}_{ff^+} = \sum_{t=1}^{T} (\hat{\mathbf{f}}_{t|T} (\hat{\mathbf{f}}_{t-1|T}^{+})' + \hat{\mathbf{P}}_{t,t-1|T}^{+})
$$

### M-Step

Maximize $Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(k)})$ with respect to $\boldsymbol{\theta}$. The solutions are in closed form:

**Factor loadings:**

$$
\boldsymbol{\Lambda}^{(k+1)} = \left(\sum_{t=1}^{T} \mathbf{x}_t^{\mathcal{O}} \hat{\mathbf{f}}_{t|T}'\right) \left(\sum_{t=1}^{T} (\hat{\mathbf{f}}_{t|T} \hat{\mathbf{f}}_{t|T}' + \hat{\mathbf{P}}_{t|T})\right)^{-1}
$$

(applied row by row, using only time periods where variable $i$ is observed)

**VAR coefficients:**

$$
\mathbf{A}^{(k+1)} = \mathbf{S}_{ff^+} \, \mathbf{S}_{f^+f^+}^{-1}
$$

**State noise covariance:**

$$
\mathbf{Q}^{(k+1)} = \frac{1}{T} \left(\mathbf{S}_{ff} - \mathbf{A}^{(k+1)} \mathbf{S}_{ff^+}'\right)
$$

**Idiosyncratic variances:**

$$
(\sigma_i^2)^{(k+1)} = \frac{1}{|\{t : i \in \mathcal{O}_t\}|} \sum_{t: i \in \mathcal{O}_t} \left[(x_{i,t} - \boldsymbol{\lambda}_i' \hat{\mathbf{f}}_{t|T})^2 + \boldsymbol{\lambda}_i' \hat{\mathbf{P}}_{t|T} \boldsymbol{\lambda}_i\right]
$$

### Convergence

The EM algorithm guarantees $\ell(\boldsymbol{\theta}^{(k+1)}) \geq \ell(\boldsymbol{\theta}^{(k)})$ at each iteration. Convergence is declared when:

$$
|\ell(\boldsymbol{\theta}^{(k+1)}) - \ell(\boldsymbol{\theta}^{(k)})| < \epsilon
$$

for a tolerance $\epsilon$ (default: $10^{-4}$ in forecastbox).

!!! warning "Local optima"
    The EM algorithm converges to a local maximum. Initialization matters. forecastbox uses **principal components** on the balanced panel (observations with no missing data) for initial factor estimates, which provides a consistent starting point.

---

## Nowcasting as Conditional Projection

### The Nowcast

Given the model and all available data up to the current vintage $\Omega_v$, the nowcast of a target variable $y_{T+h}$ is:

$$
\hat{y}_{T+h|\Omega_v} = E[y_{T+h} \mid \Omega_v]
$$

If the target variable is linked to factors via:

$$
y_{T+h} = \boldsymbol{\beta}' \mathbf{f}_{T+h} + u_{T+h}
$$

then the nowcast is:

$$
\hat{y}_{T+h|\Omega_v} = \boldsymbol{\beta}' \hat{\mathbf{f}}_{T+h|\Omega_v}
$$

where $\hat{\mathbf{f}}_{T+h|\Omega_v}$ is the Kalman-filtered (or smoothed) factor estimate using all data in the vintage $\Omega_v$.

### News Decomposition

When new data arrive (vintage $\Omega_{v+1}$), the nowcast revision can be decomposed:

$$
\hat{y}_{T+h|\Omega_{v+1}} - \hat{y}_{T+h|\Omega_v} = \underbrace{\sum_{i \in \mathcal{N}} b_i \cdot \text{news}_i}_{\text{revision due to news}}
$$

where $\text{news}_i = x_{i,t_i} - E[x_{i,t_i} \mid \Omega_v]$ is the surprise in variable $i$, and $b_i$ is the **impact coefficient** measuring how much one unit of news in variable $i$ revises the nowcast.

---

## Bridge Equations

### As a Special Case

Bridge equations are a simplified alternative to the full DFM. They relate the target variable directly to high-frequency indicators:

$$
y_t^Q = \alpha + \sum_{i=1}^{k} \beta_i \bar{x}_{i,t}^Q + \varepsilon_t
$$

where $y_t^Q$ is the quarterly target and $\bar{x}_{i,t}^Q$ is the quarterly aggregate (e.g., 3-month average) of monthly indicator $i$.

### Relationship to DFM

Bridge equations can be viewed as a restricted DFM where:

- The factors are observed (the indicators themselves)
- The loading matrix $\boldsymbol{\Lambda}$ is an identity (one factor per indicator)
- There is no factor dynamics (static model)

The DFM generalizes bridge equations by:

1. Extracting **latent** factors from many indicators
2. Modeling **factor dynamics** for multi-step forecasting
3. Naturally handling the **ragged edge** through the Kalman filter

!!! tip "When to use bridge equations vs. DFM"
    Bridge equations are preferable when: (a) few high-quality indicators are available, (b) the relationship is direct and well-understood, (c) simplicity and interpretability are priorities. The DFM is preferable when: (a) many indicators are available, (b) the signal-to-noise ratio of individual indicators is low, (c) real-time data availability varies.

---

## Key References

- Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: The real-time informational content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665-676.
- Banbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). Now-casting and the real-time data flow. In G. Elliott & A. Timmermann (Eds.), *Handbook of Economic Forecasting* (Vol. 2, pp. 195-237). Elsevier.
- Doz, C., Giannone, D., & Reichlin, L. (2011). A two-step estimator for large approximate dynamic factor models based on Kalman filtering. *Journal of Econometrics*, 164(1), 188-205.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-38.
- Chamberlain, G., & Rothschild, M. (1983). Arbitrage, factor structure, and mean-variance analysis on large asset markets. *Econometrica*, 51(5), 1281-1304.
