---
title: BMA (Bayesian Model Averaging)
description: Combinacao bayesiana de previsoes com posterior model probabilities, marginal likelihood e aproximacao BIC.
---

# BMA (Bayesian Model Averaging)

O Bayesian Model Averaging (BMA) trata a **incerteza sobre qual modelo e o correto**
como parte integral da previsao. Ao inves de escolher um unico modelo, o BMA pondera
todos os modelos candidatos pela sua **probabilidade posterior** — modelos mais plausiveis
dado os dados recebem peso maior.

---

## Framework Bayesiano

### Distribuicao Preditiva

A previsao BMA para uma quantidade de interesse $\Delta$ (ex: proxima observacao) e:

$$
p(\Delta \mid D) = \sum_{k=1}^{K} p(\Delta \mid M_k, D) \cdot p(M_k \mid D)
$$

onde:

- $p(\Delta \mid M_k, D)$ — previsao do modelo $k$ dado os dados
- $p(M_k \mid D)$ — probabilidade posterior do modelo $k$
- $D$ — dados observados
- $K$ — numero de modelos candidatos

!!! abstract "Ideia Central"

    O BMA e uma **media ponderada** das previsoes, onde os pesos sao as probabilidades
    posteriores dos modelos. Modelos que explicam melhor os dados recebem peso maior.
    A incerteza preditiva incorpora tanto a incerteza dentro de cada modelo quanto a
    incerteza sobre qual modelo usar.

### Probabilidade Posterior do Modelo

Pelo teorema de Bayes:

$$
p(M_k \mid D) = \frac{p(D \mid M_k) \cdot p(M_k)}{\sum_{j=1}^{K} p(D \mid M_j) \cdot p(M_j)}
$$

onde:

- $p(D \mid M_k)$ — **marginal likelihood** (verossimilhanca marginal) do modelo $k$
- $p(M_k)$ — prior do modelo (crenca a priori sobre a plausibilidade)

### Marginal Likelihood

A marginal likelihood integra sobre todos os parametros do modelo:

$$
p(D \mid M_k) = \int p(D \mid \theta_k, M_k) \cdot p(\theta_k \mid M_k)\, d\theta_k
$$

Esta integral penaliza naturalmente modelos complexos (**Occam's razor automatico**):
modelos com muitos parametros espalham a prior sobre um espaco grande, reduzindo a
verossimilhanca marginal.

---

## Aproximacao BIC

Calcular a marginal likelihood exata e geralmente intratavel. A **aproximacao BIC**
(Bayesian Information Criterion) oferece uma solucao pratica:

$$
\log p(D \mid M_k) \approx -\frac{1}{2} \text{BIC}_k = \ell_k(\hat{\theta}_k) - \frac{d_k}{2} \log T
$$

onde:

- $\ell_k(\hat{\theta}_k)$ — log-verossimilhanca maximizada do modelo $k$
- $d_k$ — numero de parametros do modelo $k$
- $T$ — numero de observacoes

A posterior do modelo com prior uniforme e aproximacao BIC:

$$
p(M_k \mid D) \approx \frac{\exp\!\left(-\frac{1}{2} \text{BIC}_k\right)}{\sum_{j=1}^{K} \exp\!\left(-\frac{1}{2} \text{BIC}_j\right)}
$$

!!! info "Por que BIC?"

    O BIC e uma aproximacao de Laplace para a log-marginal likelihood. Diferente
    do AIC, o BIC e **consistente** — com dados suficientes, atribui probabilidade
    1 ao modelo verdadeiro (se ele estiver no conjunto candidato).

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `prior` | `str` | `"uniform"` | Prior dos modelos: `"uniform"` ou `"custom"` |
| `prior_weights` | `list` | `None` | Pesos da prior customizada (com `prior="custom"`) |
| `approximation` | `str` | `"bic"` | Metodo para marginal likelihood: `"bic"`, `"bridge_sampling"` |

---

## Exemplos

### BMA com 4 Modelos

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.models import Theta, TBATS
from forecastbox.combine import combine

# Dados
y = pd.read_csv("ipca.csv", index_col="date", parse_dates=True)["ipca"]
y_train = y[:"2023-06"]
y_test = y["2023-07":"2023-12"]

# Ajustar 4 modelos
arima = AutoARIMA(seasonal=True, m=12).fit(y_train)
ets = AutoETS(seasonal_periods=12).fit(y_train)
theta = Theta().fit(y_train)
tbats = TBATS(seasonal_periods=[12]).fit(y_train)

fc_arima = arima.predict(horizon=6)
fc_ets = ets.predict(horizon=6)
fc_theta = theta.predict(horizon=6)
fc_tbats = tbats.predict(horizon=6)

# Combinar com BMA (prior uniforme, aproximacao BIC)
fc_bma = combine(
    forecasts=[fc_arima, fc_ets, fc_theta, fc_tbats],
    method="bma",
    prior="uniform",
    approximation="bic",
)
print(fc_bma.summary())
```

```text
Combination Summary
===================
Method: Bayesian Model Averaging (BIC approximation)
Models: 4
Prior: uniform

Posterior Model Probabilities:
  arima    0.412
  ets      0.298
  theta    0.187
  tbats    0.103

BIC Values:
  arima   -234.5
  ets     -231.8
  theta   -228.1
  tbats   -224.6

Effective Number of Models: 3.2
```

### BMA com Prior Informativa

```python
# Prior informativa: crenca de que ARIMA e ETS sao mais plausiveis
fc_bma_custom = combine(
    forecasts=[fc_arima, fc_ets, fc_theta, fc_tbats],
    method="bma",
    prior="custom",
    prior_weights=[0.35, 0.35, 0.15, 0.15],
)
print(fc_bma_custom.summary())
```

```text
Combination Summary
===================
Method: Bayesian Model Averaging (BIC approximation)
Models: 4
Prior: custom [0.35, 0.35, 0.15, 0.15]

Posterior Model Probabilities:
  arima    0.468
  ets      0.337
  theta    0.121
  tbats    0.074

Note: Prior shifted weight toward arima and ets.
```

### Intervalos de Confianca BMA

```python
# BMA produz intervalos que incorporam incerteza do modelo
fc_bma = combine(
    forecasts=[fc_arima, fc_ets, fc_theta, fc_tbats],
    method="bma",
)

# Intervalos de confianca (mais largos que modelos individuais)
print(fc_bma.prediction_interval(level=0.95))
```

```text
BMA Prediction Intervals (95%)
===============================
           point    lower    upper    width
2023-07    0.312    0.187    0.437    0.250
2023-08    0.298    0.154    0.442    0.288
2023-09    0.345    0.178    0.512    0.334
2023-10    0.321    0.142    0.500    0.358
2023-11    0.378    0.172    0.584    0.412
2023-12    0.356    0.138    0.574    0.436

Note: BMA intervals are wider than individual model intervals
because they incorporate model uncertainty.
```

!!! tip "Intervalos BMA vs Individuais"

    Os intervalos de confianca do BMA sao tipicamente **mais largos** que os de
    qualquer modelo individual, pois incorporam a incerteza sobre qual modelo e
    correto. Isso gera intervalos mais honestos e com melhor cobertura empirica.

### Evolucao das Probabilidades Posteriores

```python
# Monitorar como os pesos BMA mudam com dados adicionais
import matplotlib.pyplot as plt

posteriors = fc_bma.rolling_posteriors(window=36, step=1)
posteriors.plot(figsize=(10, 5))
plt.title("Evolucao das Probabilidades Posteriores BMA")
plt.ylabel("Posterior Probability")
plt.xlabel("Data")
plt.legend(loc="upper right")
plt.show()
```

```text
Rolling Posterior Probabilities (36-month window)
=================================================
         2021-01  2022-01  2023-01  2023-06
arima      0.31     0.38     0.42     0.41
ets        0.35     0.32     0.30     0.30
theta      0.22     0.19     0.18     0.19
tbats      0.12     0.11     0.10     0.10
```

---

## BMA vs Outros Metodos

| Aspecto | BMA | OLS | Stacking |
|:--------|:----|:----|:---------|
| Fundamento | Bayesiano | Frequentista | Machine Learning |
| Pesos | Probabilidades ($\sum = 1$, $w \geq 0$) | Coeficientes (sem restricao) | Depende do meta-learner |
| Incerteza | Intervalos incorporam incerteza do modelo | Nao | Nao |
| Selecao de modelos | Automatica (via posteriors) | Nao (exceto Lasso) | Com Lasso |
| Requer distribuicao | Sim (verossimilhanca) | Nao | Nao |

---

## Proximos Passos

- **[Time-Varying](time-varying.md)** — pesos que se adaptam ao longo do tempo
- **[Otima](optimal.md)** — combinacao otima de Bates-Granger
- **[Escolhendo Metodo](choosing.md)** — guia para selecionar a melhor estrategia

---

## Referencias

- **Raftery, A.E., Madigan, D. & Hoeting, J.A.** (1997). "Bayesian Model Averaging for Linear Regression Models." *Journal of the American Statistical Association*, 92(437), 179-191.
- **Hoeting, J.A., Madigan, D., Raftery, A.E. & Volinsky, C.T.** (1999). "Bayesian Model Averaging: A Tutorial." *Statistical Science*, 14(4), 382-417.
- **Fragoso, T.M., Bertoli, W. & Louzada, F.** (2018). "Bayesian Model Averaging: A Systematic Review and Conceptual Classification." *International Statistical Review*, 86(1), 1-28.
- **Timmermann, A.** (2006). "Forecast Combinations." *Handbook of Economic Forecasting*, Vol. 1, 135-196.
