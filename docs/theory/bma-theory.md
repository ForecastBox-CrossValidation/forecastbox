---
title: "Bayesian Model Averaging Theory"
description: "Mathematical foundations of Bayesian Model Averaging: posterior model probabilities, marginal likelihood, BIC approximation, and computational methods (MC3, RJMCMC)."
---

# Bayesian Model Averaging Theory

!!! abstract "Key Takeaway"
    Bayesian Model Averaging (BMA) addresses model uncertainty by averaging predictions over a set of candidate models, weighted by their posterior probabilities. Instead of selecting a single "best" model, BMA acknowledges that multiple models may be plausible given the data, producing forecasts that are better calibrated and prediction intervals with correct coverage. The key ingredients are prior model probabilities, marginal likelihoods (or their BIC approximation), and efficient computational methods for exploring the model space.

## Motivation

Model selection procedures (AIC, BIC, cross-validation) choose a single model and proceed as if that model were known to be true. This ignores **model uncertainty** --- the fact that the data may be consistent with several competing specifications. The consequences include:

- **Overconfident predictions**: intervals are too narrow because they condition on a single model
- **Instability**: small changes in the data can select a different model, causing large swings in forecasts
- **Omitted-variable bias**: the selected model may exclude relevant predictors

BMA addresses these issues by treating the model itself as a random variable with a posterior distribution.

---

## The Bayesian Framework

### Setup

Let $\mathcal{M} = \{M_1, M_2, \ldots, M_K\}$ be a set of candidate models. Each model $M_k$ is defined by:

- A likelihood function $p(\mathbf{y} \mid \boldsymbol{\theta}_k, M_k)$
- A parameter vector $\boldsymbol{\theta}_k \in \Theta_k$ with prior $p(\boldsymbol{\theta}_k \mid M_k)$
- A prior model probability $p(M_k)$

### Posterior Model Probability

By Bayes' theorem, the posterior probability of model $M_k$ given data $\mathbf{y}$ is:

$$
p(M_k \mid \mathbf{y}) = \frac{p(\mathbf{y} \mid M_k) \, p(M_k)}{\sum_{j=1}^{K} p(\mathbf{y} \mid M_j) \, p(M_j)}
$$

where $p(\mathbf{y} \mid M_k)$ is the **marginal likelihood** (or **evidence**) of model $M_k$.

### The BMA Predictive Distribution

The BMA predictive density for a future observation $\tilde{y}$ is:

$$
p(\tilde{y} \mid \mathbf{y}) = \sum_{k=1}^{K} p(\tilde{y} \mid \mathbf{y}, M_k) \, p(M_k \mid \mathbf{y})
$$

This is a **mixture** of model-specific predictive densities, weighted by posterior model probabilities.

**Key properties:**

- The BMA point forecast (posterior predictive mean) is:

$$
E[\tilde{y} \mid \mathbf{y}] = \sum_{k=1}^{K} E[\tilde{y} \mid \mathbf{y}, M_k] \, p(M_k \mid \mathbf{y})
$$

- The BMA predictive variance decomposes as:

$$
\text{Var}(\tilde{y} \mid \mathbf{y}) = \underbrace{\sum_{k=1}^{K} \text{Var}(\tilde{y} \mid \mathbf{y}, M_k) \, p(M_k \mid \mathbf{y})}_{\text{within-model uncertainty}} + \underbrace{\sum_{k=1}^{K} \left(E[\tilde{y} \mid \mathbf{y}, M_k] - E[\tilde{y} \mid \mathbf{y}]\right)^2 p(M_k \mid \mathbf{y})}_{\text{between-model uncertainty}}
$$

!!! note "Variance decomposition"
    The first term is the weighted average of within-model variances. The second term captures the **additional uncertainty** due to disagreement among models. This is the key advantage of BMA over model selection: prediction intervals account for both parameter and model uncertainty.

---

## Marginal Likelihood

### Definition and Computation

The marginal likelihood of model $M_k$ is obtained by integrating out the parameters:

$$
p(\mathbf{y} \mid M_k) = \int p(\mathbf{y} \mid \boldsymbol{\theta}_k, M_k) \, p(\boldsymbol{\theta}_k \mid M_k) \, d\boldsymbol{\theta}_k
$$

This integral is typically intractable in closed form, except for conjugate models.

### Conjugate Case: Normal Linear Regression

Consider the linear regression model $M_k$: $\mathbf{y} = \mathbf{X}_k \boldsymbol{\beta}_k + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$, with conjugate priors:

$$
\boldsymbol{\beta}_k \mid \sigma^2, M_k \sim \mathcal{N}\!\left(\boldsymbol{\mu}_0, \sigma^2 \mathbf{V}_0\right), \quad \sigma^2 \mid M_k \sim \text{IG}\!\left(\frac{\nu_0}{2}, \frac{\delta_0}{2}\right)
$$

The marginal likelihood has a closed-form expression:

$$
p(\mathbf{y} \mid M_k) = \frac{\Gamma\!\left(\frac{\nu_0 + T}{2}\right)}{\Gamma\!\left(\frac{\nu_0}{2}\right)} \cdot \frac{\delta_0^{\nu_0/2}}{(\delta_0 + S_k)^{(\nu_0 + T)/2}} \cdot \frac{|\mathbf{V}_0|^{1/2}}{|\mathbf{V}_k|^{1/2}} \cdot \pi^{-T/2}
$$

where $T$ is the sample size, $\mathbf{V}_k = (\mathbf{V}_0^{-1} + \mathbf{X}_k' \mathbf{X}_k)^{-1}$ is the posterior precision, and $S_k$ is the posterior sum of squares incorporating the prior.

**Derivation.** Start from:

$$
p(\mathbf{y} \mid M_k) = \int_0^\infty \int_{\mathbb{R}^{p_k}} p(\mathbf{y} \mid \boldsymbol{\beta}_k, \sigma^2) \, p(\boldsymbol{\beta}_k \mid \sigma^2) \, p(\sigma^2) \, d\boldsymbol{\beta}_k \, d\sigma^2
$$

The inner integral over $\boldsymbol{\beta}_k$ is Gaussian and yields a density proportional to $(\sigma^2)^{-T/2} |\mathbf{V}_k|^{1/2} |\mathbf{V}_0|^{-1/2} \exp(-S_k / (2\sigma^2))$, where $S_k$ absorbs the quadratic forms. The outer integral over $\sigma^2$ is then Inverse-Gamma, yielding the stated result. $\blacksquare$

### Zellner's $g$-Prior

A widely used prior for BMA is **Zellner's $g$-prior** (Zellner, 1986):

$$
\boldsymbol{\beta}_k \mid \sigma^2, g, M_k \sim \mathcal{N}\!\left(\mathbf{0}, g \sigma^2 (\mathbf{X}_k' \mathbf{X}_k)^{-1}\right)
$$

The hyperparameter $g$ controls the prior tightness:

- $g \to \infty$: diffuse prior, approaches OLS
- $g = T$: **unit information prior** (UIP), equivalent to one observation of information
- $g = 1$: tight prior, strong shrinkage

Under the $g$-prior, the marginal likelihood simplifies to:

$$
p(\mathbf{y} \mid M_k, g) = \frac{\Gamma(T/2)}{\pi^{T/2}} \cdot (1 + g)^{-(p_k)/2} \cdot \left[\mathbf{y}'\mathbf{y} - \frac{g}{1+g} \mathbf{y}' \mathbf{H}_k \mathbf{y}\right]^{-T/2}
$$

where $\mathbf{H}_k = \mathbf{X}_k (\mathbf{X}_k' \mathbf{X}_k)^{-1} \mathbf{X}_k'$ is the hat matrix and $p_k$ is the number of parameters in model $k$.

---

## BIC Approximation

### Derivation

When the marginal likelihood is intractable, the **Bayesian Information Criterion** (Schwarz, 1978) provides an asymptotic approximation. Using a Laplace approximation to the marginal likelihood integral:

$$
\log p(\mathbf{y} \mid M_k) \approx \log p(\mathbf{y} \mid \hat{\boldsymbol{\theta}}_k, M_k) - \frac{p_k}{2} \log T + O(1)
$$

where $\hat{\boldsymbol{\theta}}_k$ is the MLE and $p_k = \dim(\boldsymbol{\theta}_k)$.

This gives the BIC:

$$
\text{BIC}_k = -2 \log p(\mathbf{y} \mid \hat{\boldsymbol{\theta}}_k, M_k) + p_k \log T
$$

The approximate posterior model probability is:

$$
p(M_k \mid \mathbf{y}) \approx \frac{\exp(-\text{BIC}_k / 2) \, p(M_k)}{\sum_{j=1}^{K} \exp(-\text{BIC}_j / 2) \, p(M_j)}
$$

**Proof of the Laplace approximation.** Expand $\log p(\mathbf{y} \mid \boldsymbol{\theta}_k, M_k)$ around the MLE $\hat{\boldsymbol{\theta}}_k$:

$$
\log p(\mathbf{y} \mid \boldsymbol{\theta}_k) \approx \log p(\mathbf{y} \mid \hat{\boldsymbol{\theta}}_k) - \frac{1}{2} (\boldsymbol{\theta}_k - \hat{\boldsymbol{\theta}}_k)' \mathbf{J}_k (\boldsymbol{\theta}_k - \hat{\boldsymbol{\theta}}_k)
$$

where $\mathbf{J}_k = -\frac{\partial^2}{\partial \boldsymbol{\theta}_k \partial \boldsymbol{\theta}_k'} \log p(\mathbf{y} \mid \boldsymbol{\theta}_k) \big|_{\hat{\boldsymbol{\theta}}_k}$ is the observed information. Substituting:

$$
p(\mathbf{y} \mid M_k) \approx p(\mathbf{y} \mid \hat{\boldsymbol{\theta}}_k) \int \exp\!\left(-\frac{1}{2} (\boldsymbol{\theta}_k - \hat{\boldsymbol{\theta}}_k)' \mathbf{J}_k (\boldsymbol{\theta}_k - \hat{\boldsymbol{\theta}}_k)\right) p(\boldsymbol{\theta}_k) \, d\boldsymbol{\theta}_k
$$

The integral is a Gaussian with covariance $\mathbf{J}_k^{-1}$, yielding:

$$
p(\mathbf{y} \mid M_k) \approx p(\mathbf{y} \mid \hat{\boldsymbol{\theta}}_k) \cdot (2\pi)^{p_k/2} |\mathbf{J}_k|^{-1/2} \cdot p(\hat{\boldsymbol{\theta}}_k)
$$

Since $\mathbf{J}_k = O(T)$, we have $|\mathbf{J}_k| = O(T^{p_k})$, so $\log |\mathbf{J}_k| \approx p_k \log T$. Collecting terms and noting that $\log p(\hat{\boldsymbol{\theta}}_k) = O(1)$ gives the BIC approximation. $\blacksquare$

### Accuracy of the BIC Approximation

!!! warning "Limitations"
    The BIC approximation is accurate when: (a) the sample size $T$ is large, (b) the prior $p(\boldsymbol{\theta}_k)$ is smooth and non-degenerate at the MLE, (c) the model is well-identified. It can be poor when models are nested, when parameters are near the boundary of the parameter space, or when $T$ is small relative to $p_k$.

---

## Bayes Factors

### Definition

The **Bayes factor** comparing model $M_i$ to model $M_j$ is:

$$
\text{BF}_{ij} = \frac{p(\mathbf{y} \mid M_i)}{p(\mathbf{y} \mid M_j)}
$$

The posterior odds ratio relates to the Bayes factor via:

$$
\frac{p(M_i \mid \mathbf{y})}{p(M_j \mid \mathbf{y})} = \text{BF}_{ij} \cdot \frac{p(M_i)}{p(M_j)}
$$

### Interpretation Scale

Kass and Raftery (1995) proposed the following interpretation:

| $2 \log \text{BF}_{ij}$ | Evidence for $M_i$ |
|:---|:---|
| $0$ to $2$ | Not worth more than a bare mention |
| $2$ to $6$ | Positive |
| $6$ to $10$ | Strong |
| $> 10$ | Very strong |

### Properties

- **Transitivity:** $\text{BF}_{ij} = \text{BF}_{ik} \cdot \text{BF}_{kj}$
- **Consistency:** under regularity conditions, $\text{BF}_{ij} \to \infty$ if $M_i$ is true and $M_j$ is not, as $T \to \infty$
- **Sensitivity to priors:** the Bayes factor depends on the prior $p(\boldsymbol{\theta}_k \mid M_k)$, not just the posterior. Improper priors lead to indeterminate Bayes factors (the **Jeffreys-Lindley paradox**)

!!! warning "Jeffreys-Lindley Paradox"
    Improper (flat) priors on parameters make the marginal likelihood $p(\mathbf{y} \mid M_k)$ improper and Bayes factors meaningless. Always use **proper** priors for BMA. The $g$-prior and unit information prior are popular choices that provide proper marginal likelihoods with minimal subjective input.

---

## Prior Model Probabilities

### Uniform Prior

The simplest choice: $p(M_k) = 1/K$ for all $k$. This treats all models as equally plausible *a priori*.

When models are defined by inclusion/exclusion of $p$ predictors (so $K = 2^p$), the uniform prior over models implies that each predictor has inclusion probability $1/2$ and predictors are independent *a priori*.

### Beta-Binomial (Dilution) Prior

To address the **dilution problem** --- where adding correlated predictors dilutes posterior mass across many similar models --- **George (2010)** proposed placing a prior on the inclusion probability $\pi$:

$$
\pi \sim \text{Beta}(a, b)
$$

$$
p(M_k) = \int_0^1 \pi^{p_k} (1 - \pi)^{p - p_k} \, \text{Beta}(\pi; a, b) \, d\pi = \frac{B(a + p_k, b + p - p_k)}{B(a, b)}
$$

With $a = b = 1$ (uniform on $\pi$), this gives:

$$
p(M_k) = \frac{1}{(p+1) \binom{p}{p_k}}
$$

This prior assigns equal probability to each **model size** and then spreads probability uniformly within each size class. It favors parsimony compared to the uniform model prior.

### Fixed Inclusion Probability

Set each predictor's inclusion probability to $\pi_0 = \bar{p}/p$, where $\bar{p}$ is the expected model size:

$$
p(M_k) = \pi_0^{p_k} (1 - \pi_0)^{p - p_k}
$$

This allows the practitioner to express a prior belief about model complexity. With $\bar{p} = p/2$, this reduces to the uniform prior.

---

## BMA vs. Model Selection

### Theoretical Comparison

**Theorem (Raftery, 1995).** *Under regularity conditions, BMA provides optimal predictive performance in the sense that it minimizes the expected logarithmic predictive score:*

$$
E\!\left[-\log p(\tilde{y} \mid \mathbf{y})\right] \leq E\!\left[-\log p(\tilde{y} \mid \mathbf{y}, M_k)\right] \quad \text{for all } k
$$

*with equality iff $p(M_k \mid \mathbf{y}) = 1$, i.e., the data are perfectly informative about the true model.*

**Proof.** By Jensen's inequality and the concavity of $\log$:

$$
\log p(\tilde{y} \mid \mathbf{y}) = \log \sum_k p(\tilde{y} \mid \mathbf{y}, M_k) p(M_k \mid \mathbf{y}) \geq \sum_k p(M_k \mid \mathbf{y}) \log p(\tilde{y} \mid \mathbf{y}, M_k)
$$

Taking expectations and rearranging gives the result. Equality holds iff all weight is on one model. $\blacksquare$

### Practical Tradeoffs

| Aspect | BMA | Model Selection |
|:---|:---|:---|
| Point forecast | Weighted average | Single model |
| Prediction intervals | Accounts for model uncertainty | Conditional on selected model |
| Interpretation | Multiple models contribute | Single model is interpretable |
| Computation | Requires exploring model space | Requires fitting one model |
| Robustness | Robust to model misspecification | Sensitive to selection errors |
| When models are similar | Hedges across alternatives | Arbitrary choice |
| When one model dominates | Converges to best model | Correct selection |

!!! tip "When BMA helps most"
    BMA provides the largest gains over model selection when: (a) several models have similar fit, (b) the models give different predictions, (c) the sample size is moderate (model uncertainty is large). When one model clearly dominates ($p(M_k \mid \mathbf{y}) \approx 1$), BMA reduces to model selection.

---

## Computational Methods

### The Challenge

With $p$ potential predictors, the model space has $K = 2^p$ elements. For $p = 20$, $K \approx 10^6$; for $p = 30$, $K \approx 10^9$. Exhaustive enumeration is feasible only for $p \leq 25$ approximately.

### Markov Chain Monte Carlo Model Composition (MC$^3$)

**Madigan and York (1995)** proposed MC$^3$, which constructs a Markov chain that traverses the model space with stationary distribution equal to the posterior $p(M_k \mid \mathbf{y})$.

**Algorithm:**

1. **Initialize:** start at model $M^{(0)}$ (e.g., the full model or BIC-selected model)

2. **At iteration $s$:** given current model $M^{(s)}$:

    a. **Propose:** draw a candidate model $M'$ from a neighborhood $\text{nbd}(M^{(s)})$. The standard neighborhood consists of models obtained by adding, removing, or swapping one predictor.

    b. **Acceptance probability:**

    $$
    \alpha = \min\!\left(1, \; \frac{p(\mathbf{y} \mid M') \, p(M')}{p(\mathbf{y} \mid M^{(s)}) \, p(M^{(s)})} \cdot \frac{|\text{nbd}(M^{(s)})|}{|\text{nbd}(M')|}\right)
    $$

    c. **Accept/reject:** set $M^{(s+1)} = M'$ with probability $\alpha$, otherwise $M^{(s+1)} = M^{(s)}$

3. **Posterior estimates:** after burn-in, the posterior model probability is estimated by:

$$
\hat{p}(M_k \mid \mathbf{y}) = \frac{1}{S} \sum_{s=1}^{S} \mathbb{1}(M^{(s)} = M_k)
$$

The posterior inclusion probability for predictor $j$ is:

$$
\hat{p}(\gamma_j = 1 \mid \mathbf{y}) = \frac{1}{S} \sum_{s=1}^{S} \mathbb{1}(j \in M^{(s)})
$$

!!! note "Convergence"
    MC$^3$ explores the model space stochastically. Convergence requires sufficient iterations to visit the high-posterior-probability region of the model space. Multiple chains with different starting points and the Gelman-Rubin diagnostic are recommended.

### Reversible Jump MCMC (RJMCMC)

**Green (1995)** proposed RJMCMC, which jointly samples models and parameters in a single Markov chain that moves across spaces of different dimensions.

**Key idea:** at each iteration, the chain can:

1. **Within-model move:** update $\boldsymbol{\theta}_k$ conditional on the current model $M_k$ (standard Metropolis-Hastings)
2. **Between-model move:** propose a jump to model $M_{k'}$ with a different parameter dimension, using a dimension-matching transformation

**Dimension matching.** When jumping from $M_k$ (dimension $p_k$) to $M_{k'}$ (dimension $p_{k'}$ with $p_{k'} > p_k$), draw auxiliary variables $\mathbf{u} \in \mathbb{R}^{p_{k'} - p_k}$ and define a bijection:

$$
(\boldsymbol{\theta}_{k'}, \mathbf{u}') = g(\boldsymbol{\theta}_k, \mathbf{u})
$$

The acceptance probability is:

$$
\alpha = \min\!\left(1, \; \frac{p(\mathbf{y} \mid \boldsymbol{\theta}_{k'}, M_{k'}) \, p(\boldsymbol{\theta}_{k'} \mid M_{k'}) \, p(M_{k'})}{p(\mathbf{y} \mid \boldsymbol{\theta}_k, M_k) \, p(\boldsymbol{\theta}_k \mid M_k) \, p(M_k)} \cdot \frac{q(\mathbf{u}')}{q(\mathbf{u})} \cdot \left|\frac{\partial g}{\partial (\boldsymbol{\theta}_k, \mathbf{u})}\right|\right)
$$

where $q(\cdot)$ is the proposal density for the auxiliary variables and the last term is the Jacobian of the transformation.

### Comparison of Computational Methods

| Method | Model space | Parameters | Pros | Cons |
|:---|:---|:---|:---|:---|
| Exhaustive | All $2^p$ | Marginal likelihood | Exact | Infeasible for $p > 25$ |
| MC$^3$ | MCMC | Marginal likelihood | Simple, proven | Needs $p(\mathbf{y} \mid M_k)$ |
| RJMCMC | MCMC | MCMC | No need for marginal likelihood | Complex proposals |
| Stochastic search | Heuristic | Marginal likelihood | Fast exploration | No convergence guarantee |

---

## BMA for Forecasting

### Forecast Combination Interpretation

BMA can be viewed as a principled form of **forecast combination** where the weights are posterior model probabilities. Compared to the frequentist combination methods (see [Combination Theory](combination-theory.md)):

- Weights are derived from a coherent probabilistic framework
- Weights automatically adapt as more data arrive (posterior updating)
- The full predictive distribution (not just a point forecast) is combined

### Dynamic BMA

In time-varying environments, posterior model probabilities may shift. **Raftery, Karny, and Ettler (2010)** proposed Dynamic Model Averaging (DMA), where the model probabilities evolve via:

$$
p(M_k \mid \mathbf{y}_{1:t}) \propto p(y_t \mid \mathbf{y}_{1:t-1}, M_k) \cdot p(M_k \mid \mathbf{y}_{1:t-1})^\lambda
$$

The **forgetting factor** $\lambda \in (0, 1]$ controls the memory: smaller $\lambda$ discounts older evidence more aggressively. With $\lambda = 1$, standard BMA is recovered.

---

## Key References

- Hoeting, J. A., Madigan, D., Raftery, A. E., & Volinsky, C. T. (1999). Bayesian model averaging: A tutorial. *Statistical Science*, 14(4), 382-417.
- Raftery, A. E. (1995). Bayesian model selection in social research. *Sociological Methodology*, 25, 111-163.
- Kass, R. E., & Raftery, A. E. (1995). Bayes factors. *Journal of the American Statistical Association*, 90(430), 773-795.
- Madigan, D., & York, J. (1995). Bayesian graphical models for discrete data. *International Statistical Review*, 63(2), 215-232.
- Green, P. J. (1995). Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. *Biometrika*, 82(4), 711-732.
- Zellner, A. (1986). On assessing prior distributions and Bayesian regression analysis with $g$-prior distributions. In P. K. Goel & A. Zellner (Eds.), *Bayesian Inference and Decision Techniques* (pp. 233-243). Elsevier.
- Schwarz, G. (1978). Estimating the dimension of a model. *Annals of Statistics*, 6(2), 461-464.
- Raftery, A. E., Karny, M., & Ettler, P. (2010). Online prediction under model uncertainty via dynamic model averaging: Application to a cold rolling mill. *Technometrics*, 52(1), 52-66.
- Fernandez, C., Ley, E., & Steel, M. F. J. (2001). Benchmark priors for Bayesian model averaging. *Journal of Econometrics*, 100(2), 381-427.
- George, E. I. (2010). Dilution priors: Compensating for model space redundancy. In J. O. Berger, T. T. Cai, & I. M. Johnstone (Eds.), *Borrowing Strength: Theory Powering Applications*. IMS Collections.

## See Also

- [Combination Theory](combination-theory.md) --- frequentist forecast combination as a complement to BMA
- [Evaluation Theory](evaluation-theory.md) --- evaluating BMA forecasts with proper scoring rules
- [MCS Theory](mcs-theory.md) --- identifying the set of best models from a frequentist perspective
- [References](references.md) --- complete bibliography
