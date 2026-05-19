---
title: "Model Confidence Set Theory"
description: "Mathematical framework of the Model Confidence Set (MCS) procedure: sequential elimination, test statistics, bootstrap inference, and asymptotic properties."
---

# Model Confidence Set Theory

## Introduction

The **Model Confidence Set** (MCS) procedure, introduced by **Hansen, Lunde, and Nason (2011)**, provides a framework for comparing multiple forecasting models simultaneously. Unlike pairwise tests (e.g., Diebold-Mariano), the MCS identifies the set of models that are statistically indistinguishable from the best model, with a given confidence level.

The MCS is to model comparison what a confidence interval is to parameter estimation: it quantifies uncertainty about which model is truly best.

---

## Formal Framework

### Setup

Let $\mathcal{M}_0 = \{1, 2, \ldots, m_0\}$ be the initial set of candidate models. For each model $i$ at time $t$, observe a loss $L_{i,t} = L(y_t, \hat{y}_{i,t})$.

Define the **loss differential** between models $i$ and $j$:

$$
d_{ij,t} = L_{i,t} - L_{j,t}, \quad i, j \in \mathcal{M}_0
$$

and its population mean:

$$
\mu_{ij} = E[d_{ij,t}]
$$

### The Superior Set of Models

The **superior set of models** $\mathcal{M}^*$ is defined as:

$$
\mathcal{M}^* = \{i \in \mathcal{M}_0 : \mu_{ij} \leq 0 \; \text{ for all } j \in \mathcal{M}_0\}
$$

In words, $\mathcal{M}^*$ contains all models whose expected loss is no greater than that of any other model in $\mathcal{M}_0$. This set is non-empty (it contains at least the best model) and may contain multiple models if they are equally good in population.

!!! note "Interpretation"
    Model $i \in \mathcal{M}^*$ means that $i$ is not significantly outperformed by any other model. The set $\mathcal{M}^*$ is the **smallest** subset of $\mathcal{M}_0$ such that no model outside the set is better than all models inside.

### Equivalence Hypothesis

For any subset $\mathcal{M} \subseteq \mathcal{M}_0$, define the **equivalence hypothesis**:

$$
H_{0,\mathcal{M}}: \mu_{ij} = 0 \quad \text{for all } i, j \in \mathcal{M}
$$

This states that all models in $\mathcal{M}$ have equal expected loss. The MCS procedure tests this hypothesis sequentially.

---

## Sequential Elimination Procedure

### Algorithm

The MCS is constructed by the following algorithm:

**Input:** Initial model set $\mathcal{M}_0$, significance level $\alpha$, loss series $\{L_{i,t}\}$.

1. Set $\mathcal{M} = \mathcal{M}_0$
2. Test $H_{0,\mathcal{M}}$ using a suitable test statistic
3. If $H_{0,\mathcal{M}}$ is **not rejected** at level $\alpha$: **stop**, return $\hat{\mathcal{M}}^*_{1-\alpha} = \mathcal{M}$
4. If $H_{0,\mathcal{M}}$ is **rejected**: identify the worst-performing model $e_\mathcal{M}$ using an elimination rule, set $\mathcal{M} \leftarrow \mathcal{M} \setminus \{e_\mathcal{M}\}$, and go to step 2

**Output:** $\hat{\mathcal{M}}^*_{1-\alpha}$, the model confidence set at level $1 - \alpha$.

### Elimination Rule

The model eliminated at each step is the one with the largest relative loss:

$$
e_\mathcal{M} = \arg\max_{i \in \mathcal{M}} \; \bar{d}_{i \cdot}
$$

where $\bar{d}_{i \cdot} = \frac{1}{|\mathcal{M}| - 1} \sum_{j \in \mathcal{M}, j \neq i} \bar{d}_{ij}$ is the average loss differential of model $i$ against all other models in $\mathcal{M}$, and $\bar{d}_{ij} = \frac{1}{T} \sum_{t=1}^T d_{ij,t}$.

### MCS p-values

For each model $i$, the **MCS p-value** $\hat{p}_i$ is defined as:

$$
\hat{p}_i = \max_{j \leq i} \; \hat{p}_{H_{0,\mathcal{M}_j}}
$$

where models are ordered by elimination step ($i = 1$ is eliminated first) and $\hat{p}_{H_{0,\mathcal{M}_j}}$ is the p-value for the equivalence test at step $j$. The max operator ensures monotonicity: earlier-eliminated models have lower p-values.

!!! tip "Reading MCS p-values"
    - $\hat{p}_i \geq \alpha$: model $i$ is in the MCS at confidence level $1 - \alpha$
    - $\hat{p}_i < \alpha$: model $i$ is eliminated from the MCS
    - Higher p-values indicate stronger evidence that the model belongs to the superior set

---

## Test Statistics

### The $T_R$ Statistic (Range)

The **range** statistic tests $H_{0,\mathcal{M}}$ by examining the maximum pairwise difference:

$$
T_R = \max_{i,j \in \mathcal{M}} \frac{|\bar{d}_{ij}|}{\sqrt{\widehat{\text{Var}}(\bar{d}_{ij})}}
$$

This is the maximum of a set of $t$-type statistics. Under $H_{0,\mathcal{M}}$, each $\bar{d}_{ij} / \sqrt{\widehat{\text{Var}}(\bar{d}_{ij})}$ is asymptotically $\mathcal{N}(0,1)$, but the maximum has a non-standard distribution that depends on the correlation structure.

### The $T_{SQ}$ Statistic (Sum of Squares)

The **semi-quadratic** statistic is based on the average loss differentials:

$$
T_{SQ} = \sum_{i \in \mathcal{M}} \frac{\bar{d}_{i \cdot}^2}{\widehat{\text{Var}}(\bar{d}_{i \cdot})}
$$

### Variance Estimation

The variance of $\bar{d}_{ij}$ is estimated using a HAC estimator. For the stationary bootstrap (described below), the variance is obtained directly from the bootstrap distribution.

For the asymptotic version:

$$
\widehat{\text{Var}}(\bar{d}_{ij}) = \frac{1}{T} \sum_{k=-(T-1)}^{T-1} \kappa\!\left(\frac{k}{S_T}\right) \hat{\gamma}_{ij}(k)
$$

where $\hat{\gamma}_{ij}(k) = \frac{1}{T} \sum_{t=|k|+1}^{T} (d_{ij,t} - \bar{d}_{ij})(d_{ij,t-|k|} - \bar{d}_{ij})$ and $\kappa(\cdot)$ is a kernel function with bandwidth $S_T$.

### Comparison of $T_R$ and $T_{SQ}$

| Property | $T_R$ | $T_{SQ}$ |
|:---|:---|:---|
| Power against | Single large deviation | Spread-out alternatives |
| Sensitivity | One bad model pair | Overall set heterogeneity |
| Computation | Maximum of statistics | Sum of statistics |
| Recommended when | Looking for the worst model | General-purpose comparison |

---

## Bootstrap Inference

### Why Bootstrap?

The distributions of $T_R$ and $T_{SQ}$ under $H_0$ are non-standard and depend on nuisance parameters (the correlation structure of loss differentials). The bootstrap provides a feasible way to approximate these distributions.

### Stationary Bootstrap (Politis and Romano, 1994)

The **stationary bootstrap** resamples blocks of data with random block lengths, preserving the dependence structure of the time series.

**Algorithm:**

1. Choose a smoothing parameter $q \in (0, 1)$ (the probability that a new block starts at each position). The expected block length is $1/q$.
2. For bootstrap replication $b = 1, \ldots, B$:
    - Generate bootstrap indices $\{t_1^*, t_2^*, \ldots, t_T^*\}$ as follows:
        - Draw $t_1^*$ uniformly from $\{1, \ldots, T\}$
        - For $s = 2, \ldots, T$: with probability $q$, draw $t_s^*$ uniformly from $\{1, \ldots, T\}$; with probability $1 - q$, set $t_s^* = t_{s-1}^* + 1 \pmod{T}$
    - Compute $d_{ij,t_s^*}^{*} = d_{ij,t_s^*} - \bar{d}_{ij}$ (center under $H_0$)
    - Compute $T_R^{*(b)}$ or $T_{SQ}^{*(b)}$ from the bootstrap sample
3. The bootstrap p-value is:

$$
\hat{p} = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}\!\left(T^{*(b)} \geq T^{\text{obs}}\right)
$$

!!! warning "Block length selection"
    The parameter $q$ (or equivalently the expected block length $1/q$) affects the performance of the bootstrap. Too short blocks fail to capture serial dependence; too long blocks reduce the effective number of bootstrap replications. **Politis and White (2004)** proposed an automatic selection procedure based on spectral density estimation, which is the default in forecastbox.

### Centering Under the Null

A crucial step is **centering** the bootstrap loss differentials:

$$
d_{ij,t}^* = d_{ij,t} - \bar{d}_{ij}
$$

This ensures that $E^*[d_{ij,t}^*] = 0$, imposing the null hypothesis $H_0: \mu_{ij} = 0$ in the bootstrap world. Without centering, the bootstrap would replicate the sample distribution rather than the null distribution.

---

## Asymptotic Properties

### Consistency

**Theorem (Hansen, Lunde, and Nason, 2011).** *Under regularity conditions (stationarity, mixing, finite moments), the MCS procedure is consistent:*

$$
\lim_{T \to \infty} P\!\left(\mathcal{M}^* \subseteq \hat{\mathcal{M}}^*_{1-\alpha}\right) = 1
$$

*That is, the probability that the MCS contains all models in the true superior set converges to 1.*

**Proof sketch.** The proof proceeds in two parts:

1. **No false exclusion:** For any $i \in \mathcal{M}^*$, the test statistic comparing $i$ to other models in $\mathcal{M}^*$ converges to a well-defined null distribution (since $\mu_{ij} = 0$ for all $j \in \mathcal{M}^*$). The probability of rejecting a model in $\mathcal{M}^*$ is bounded by $\alpha$ asymptotically.

2. **Correct exclusion:** For any $i \notin \mathcal{M}^*$, there exists $j \in \mathcal{M}^*$ with $\mu_{ij} > 0$. By the law of large numbers, $\bar{d}_{ij} \xrightarrow{p} \mu_{ij} > 0$, so the test statistic diverges and $i$ is eliminated with probability approaching 1.

The monotonicity correction on p-values ensures that if a model survives step $k$, it also survives all subsequent steps, maintaining the coherence of the procedure. $\blacksquare$

### Size Control

The MCS controls the **familywise error rate** (FWER):

$$
P\!\left(\exists \, i \in \mathcal{M}^* : i \notin \hat{\mathcal{M}}^*_{1-\alpha}\right) \leq \alpha + o(1)
$$

This is the probability of falsely excluding at least one model from the true superior set, which is bounded by $\alpha$ asymptotically.

The proof uses the **closure principle**: the sequential testing procedure with monotonicity correction is equivalent to a closed testing procedure, which is known to control the FWER.

### Power

The power of the MCS increases with:

- **Sample size** $T$: more data to detect true differences
- **Loss differential magnitude** $|\mu_{ij}|$: larger true differences are easier to detect
- **Lower serial dependence**: reduces the effective variance of $\bar{d}_{ij}$

The $T_R$ statistic has higher power when one model is clearly worst; $T_{SQ}$ has higher power when many models are slightly worse than the best.

---

## Practical Considerations

### Choosing $\alpha$

- $\alpha = 0.10$: larger MCS, more conservative (fewer false exclusions)
- $\alpha = 0.25$: smaller MCS, more aggressive (higher power)
- Hansen et al. (2011) recommend reporting MCS p-values rather than fixing a single $\alpha$

### Number of Bootstrap Replications

- $B = 5{,}000$ is standard for reliable p-values
- $B = 10{,}000$ or more for publications
- Computational cost scales linearly in $B$ and quadratically in $|\mathcal{M}_0|$

### Relationship to Other Procedures

| Procedure | Compares | Accounts for | Outcome |
|:---|:---|:---|:---|
| Diebold-Mariano | 2 models | Serial correlation | $p$-value |
| Reality Check (White, 2000) | $N$ vs. benchmark | Multiple testing | $p$-value |
| SPA (Hansen, 2005) | $N$ vs. benchmark | Multiple testing + power | $p$-value |
| **MCS** | All $N$ pairwise | Multiple testing + sequential | **Set of best models** |

---

## Key References

- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453-497.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365-380.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.
- Politis, D. N., & White, H. (2004). Automatic block-length selection for the dependent bootstrap. *Econometric Reviews*, 23(1), 53-70.
