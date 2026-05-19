---
title: Pesos Time-Varying
description: Combinacao com pesos adaptativos que evoluem ao longo do tempo via exponential forgetting, regime-switching e Kalman filter.
---

# Pesos Time-Varying

Em muitos contextos economicos, a **performance relativa dos modelos muda ao longo do
tempo**. Um modelo que domina em periodos de estabilidade pode falhar durante crises, e
vice-versa. Os metodos de pesos time-varying adaptam continuamente os pesos da combinacao
a medida que novos dados chegam.

---

## Motivacao

!!! abstract "Por que Pesos Fixos Podem Falhar"

    Pesos fixos (media simples, OLS, BMA estatico) assumem que a performance relativa
    dos modelos e **estacionaria**. Quando ha mudancas estruturais, quebras de regime
    ou evolucao gradual da economia, pesos fixos ficam desatualizados e podem
    prejudicar a combinacao.

Cenarios tipicos onde pesos adaptativos ajudam:

- **Crises financeiras**: modelos de volatilidade ganham importancia
- **Mudancas de politica**: modelos que capturam efeitos de politica se tornam mais relevantes
- **Evolucao tecnologica**: relacoes economicas mudam gradualmente

---

## Metodo 1: Exponential Forgetting

O metodo mais simples de pesos adaptativos: observacoes recentes recebem **peso
exponencialmente maior** na avaliacao dos modelos.

### Formulacao

Os pesos sao atualizados recursivamente:

$$
w_{i,t+1} = \frac{w_{i,t}^{\lambda} \cdot p(y_t \mid M_i)}{\sum_{j=1}^{N} w_{j,t}^{\lambda} \cdot p(y_t \mid M_j)}
$$

onde:

- $w_{i,t}$ — peso do modelo $i$ no tempo $t$
- $\lambda \in (0, 1]$ — **fator de esquecimento** (forgetting factor)
- $p(y_t \mid M_i)$ — verossimilhanca preditiva do modelo $i$ para $y_t$

!!! info "Interpretacao de $\\lambda$"

    - $\lambda = 1$: sem esquecimento — equivale ao BMA estatico
    - $\lambda = 0.99$: esquecimento lento — janela efetiva de ~100 observacoes
    - $\lambda = 0.95$: esquecimento moderado — janela efetiva de ~20 observacoes
    - $\lambda = 0.90$: esquecimento rapido — janela efetiva de ~10 observacoes

    Janela efetiva aproximada: $T_{\text{eff}} \approx \frac{1}{1 - \lambda}$

### Verossimilhanca Preditiva

Com erros gaussianos, a verossimilhanca preditiva e:

$$
p(y_t \mid M_i) \propto \exp\!\left(-\frac{(y_t - \hat{y}^{(i)}_t)^2}{2\hat{\sigma}^2_{i,t}}\right)
$$

---

## Metodo 2: Regime-Switching (Markov)

Os pesos dependem de um **regime latente** que evolui segundo uma cadeia de Markov.
Em cada regime, os modelos tem pesos fixos, mas o regime ativo muda ao longo do tempo.

### Formulacao

$$
w_{i,t} = \sum_{s=1}^{S} w_{i}^{(s)} \cdot P(S_t = s \mid D_t)
$$

onde:

- $S_t \in \{1, \ldots, S\}$ — regime no tempo $t$
- $w_i^{(s)}$ — peso do modelo $i$ no regime $s$
- $P(S_t = s \mid D_t)$ — probabilidade do regime dado os dados

A transicao entre regimes segue:

$$
P(S_t = s \mid S_{t-1} = r) = \pi_{rs}
$$

!!! tip "Dois Regimes"

    Na pratica, $S = 2$ regimes sao suficientes para a maioria das aplicacoes:
    um regime de **estabilidade** e um de **crise/turbulencia**.

---

## Metodo 3: Kalman Filter nos Pesos

Trata os pesos da combinacao como **estados latentes** que evoluem segundo um modelo
de espaco de estados, estimados recursivamente via filtro de Kalman.

### Formulacao

**Equacao de observacao:**

$$
y_t = \mathbf{f}_t' \mathbf{w}_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2_\varepsilon)
$$

**Equacao de transicao:**

$$
\mathbf{w}_t = \mathbf{w}_{t-1} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim N(\mathbf{0}, Q)
$$

onde:

- $\mathbf{f}_t = (\hat{y}^{(1)}_t, \ldots, \hat{y}^{(N)}_t)'$ — vetor de previsoes
- $\mathbf{w}_t$ — vetor de pesos no tempo $t$
- $Q$ — covariancia da inovacao dos pesos (controla velocidade de adaptacao)

O filtro de Kalman estima $\mathbf{w}_t$ recursivamente:

$$
\hat{\mathbf{w}}_{t|t} = \hat{\mathbf{w}}_{t|t-1} + K_t (y_t - \mathbf{f}_t' \hat{\mathbf{w}}_{t|t-1})
$$

onde $K_t$ e o **ganho de Kalman**.

!!! info "Kalman Filter via kalmanbox"

    O forecastbox utiliza o **kalmanbox** para a implementacao do filtro de Kalman.
    Os pesos evoluem suavemente, sem saltos abruptos, o que e ideal para
    mudancas graduais na performance dos modelos.

---

## Comparacao dos Metodos

| Aspecto | Exponential Forgetting | Regime-Switching | Kalman Filter |
|:--------|:----------------------|:-----------------|:--------------|
| Tipo de mudanca | Gradual | Abrupta (regimes) | Gradual |
| Parametros | $\lambda$ | $S$, $\pi_{rs}$ | $Q$, $\sigma^2_\varepsilon$ |
| Complexidade | Baixa | Alta | Moderada |
| Interpretabilidade | Alta | Alta (regimes) | Moderada |
| Melhor para | Evolucao lenta | Crises/quebras | Mudancas suaves |

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `method` | `str` | `"forgetting"` | Metodo: `"forgetting"`, `"regime_switching"`, `"kalman"` |
| `forgetting_factor` | `float` | `0.98` | Fator $\lambda$ para exponential forgetting |
| `window` | `int` | `None` | Janela movel para avaliacao (alternativa ao forgetting) |
| `n_regimes` | `int` | `2` | Numero de regimes (so para `"regime_switching"`) |

---

## Exemplos

### Exponential Forgetting

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.models import Theta
from forecastbox.combine import combine

# Dados com periodo de crise
y = pd.read_csv("pib.csv", index_col="date", parse_dates=True)["pib"]
y_train = y[:"2019-12"]
y_test = y["2020-01":"2021-12"]

# Ajustar modelos
arima = AutoARIMA(seasonal=True, m=4).fit(y_train)
ets = AutoETS(seasonal_periods=4).fit(y_train)
theta = Theta().fit(y_train)

fc_arima = arima.predict(horizon=8)
fc_ets = ets.predict(horizon=8)
fc_theta = theta.predict(horizon=8)

# Pesos time-varying com exponential forgetting
fc_tv = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="time_varying",
    tv_method="forgetting",
    forgetting_factor=0.95,
)
print(fc_tv.summary())
```

```text
Combination Summary
===================
Method: Time-Varying (Exponential Forgetting, lambda=0.95)
Models: 3
Effective Window: ~20 observations

Current Weights (last period):
  arima    0.521
  ets      0.312
  theta    0.167

Weight Range Over Time:
         min     max     mean
arima   0.298   0.581   0.445
ets     0.241   0.412   0.338
theta   0.097   0.312   0.217
```

### Regime-Switching

```python
# Pesos que mudam por regime (ex: crise vs estabilidade)
fc_rs = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="time_varying",
    tv_method="regime_switching",
    n_regimes=2,
)
print(fc_rs.summary())
```

```text
Combination Summary
===================
Method: Time-Varying (Regime-Switching, 2 regimes)
Models: 3

Regime 1 (Estabilidade, P=0.72):
  arima    0.412
  ets      0.398
  theta    0.190

Regime 2 (Turbulencia, P=0.28):
  arima    0.587
  ets      0.213
  theta    0.200

Transition Matrix:
  [[0.94, 0.06],
   [0.15, 0.85]]

Current Regime: 1 (P=0.89)
```

### Kalman Filter nos Pesos

```python
# Pesos que evoluem suavemente via Kalman filter
fc_kalman = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="time_varying",
    tv_method="kalman",
)
print(fc_kalman.summary())
```

```text
Combination Summary
===================
Method: Time-Varying (Kalman Filter)
Models: 3

Current Weights (filtered):
  arima    0.498
  ets      0.341
  theta    0.161

Weight Evolution (quarterly):
         2022-Q1  2022-Q3  2023-Q1  2023-Q3
arima      0.412    0.445    0.478    0.498
ets        0.378    0.362    0.351    0.341
theta      0.210    0.193    0.171    0.161
```

### Pesos Durante Crise Financeira

```python
# Visualizar como pesos mudam durante crise
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Painel superior: dados
y.plot(ax=ax1, color="black", linewidth=1)
ax1.axvspan("2008-09", "2009-06", alpha=0.2, color="red", label="Crise")
ax1.set_ylabel("PIB")
ax1.legend()

# Painel inferior: pesos time-varying
weights = fc_tv.weights_over_time()
weights.plot(ax=ax2)
ax2.axvspan("2008-09", "2009-06", alpha=0.2, color="red")
ax2.set_ylabel("Peso")
ax2.set_ylim(0, 1)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()
```

```text
Peso do ARIMA aumenta durante a crise (0.35 -> 0.58),
enquanto ETS e Theta perdem peso relativo.
Apos a crise, os pesos retornam gradualmente ao padrao anterior.
```

!!! tip "Escolhendo o Forgetting Factor"

    Use validacao cruzada temporal para selecionar $\lambda$. Valores tipicos:

    - $\lambda = 0.99$: economia estavel, mudancas lentas
    - $\lambda = 0.95$: volatilidade moderada
    - $\lambda = 0.90$: mudancas frequentes de regime

---

## Proximos Passos

- **[Otima](optimal.md)** — combinacao otima com pesos fixos (Bates-Granger)
- **[BMA](bma.md)** — pesos bayesianos (estaticos ou com forgetting)
- **[Escolhendo Metodo](choosing.md)** — guia para selecionar a melhor estrategia

---

## Referencias

- **Raftery, A.E., Karny, M. & Ettler, P.** (2010). "Online Prediction Under Model Uncertainty via Dynamic Model Averaging." *Technometrics*, 52(1), 52-66.
- **Koop, G. & Korobilis, D.** (2012). "Forecasting Inflation Using Dynamic Model Averaging." *International Economic Review*, 53(3), 867-886.
- **Timmermann, A.** (2006). "Forecast Combinations." *Handbook of Economic Forecasting*, Vol. 1, 135-196.
- **West, M. & Harrison, J.** (1997). *Bayesian Forecasting and Dynamic Models*. 2nd ed. Springer.
