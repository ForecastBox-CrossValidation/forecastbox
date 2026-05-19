---
title: Combinacao de Previsoes
description: Overview dos 7 metodos de combinacao de previsoes, da media simples ao BMA e pesos time-varying.
---

# Combinacao de Previsoes

Combinar previsoes de multiplos modelos e uma das estrategias mais robustas em
econometria aplicada. Desde o trabalho seminal de **Bates & Granger (1969)**, a
literatura demonstra que combinacoes frequentemente superam modelos individuais —
mesmo os melhores.

---

## Por que Combinar?

A intuicao e simples: modelos diferentes capturam aspectos distintos da serie temporal.
Ao combinar, **diversificamos o risco de escolher o modelo errado** e reduzimos a
variancia da previsao.

Formalmente, considere dois modelos com erros de previsao $e_1$ e $e_2$. A variancia
do erro da combinacao com pesos $w$ e $(1-w)$ e:

$$
\text{Var}(e_c) = w^2 \sigma_1^2 + (1-w)^2 \sigma_2^2 + 2w(1-w)\rho\sigma_1\sigma_2
$$

Quando a correlacao $\rho < 1$, existe um peso $w^*$ tal que
$\text{Var}(e_c) < \min(\sigma_1^2, \sigma_2^2)$ — a combinacao e **melhor que
qualquer modelo individual**.

!!! info "Forecast Combination Puzzle"

    Na pratica, a media simples ($w = 1/N$) frequentemente supera metodos
    sofisticados de estimacao de pesos. Esse fenomeno, documentado por
    Smith & Wallis (2009), ocorre porque a estimacao de pesos otimos introduce
    erro amostral que pode superar o ganho teorico.

---

## Equacao Geral

A previsao combinada e uma media ponderada das previsoes individuais:

$$
\hat{y}_{t+h|t}^c = \sum_{i=1}^{N} w_i \hat{y}_{t+h|t}^{(i)}, \quad \sum_{i=1}^{N} w_i = 1
$$

onde:

| Simbolo | Descricao |
|:--------|:----------|
| $\hat{y}_{t+h\|t}^c$ | Previsao combinada para $h$ passos a frente |
| $\hat{y}_{t+h\|t}^{(i)}$ | Previsao do modelo $i$ |
| $w_i$ | Peso atribuido ao modelo $i$ |
| $N$ | Numero de modelos no pool |

---

## Taxonomia dos Metodos

Os 7 metodos de combinacao do forecastbox se dividem em tres categorias:

### Metodos Simples

Pesos fixos, sem estimacao. Robustos e faceis de implementar.

### Metodos Estimados

Pesos estimados a partir dos dados. Potencialmente otimos, mas sujeitos a overfitting.

### Metodos Adaptativos

Pesos que variam no tempo. Capturam mudancas na performance relativa dos modelos.

---

## Tabela Comparativa

| Metodo | Categoria | Pesos | Pros | Contras |
|:-------|:----------|:------|:-----|:--------|
| **Media Simples** | Simples | $w_i = 1/N$ | Robusto, sem estimacao | Ignora diferenca de qualidade |
| **Pesos Fixos** | Simples | Definidos pelo usuario | Incorpora conhecimento do especialista | Subjetivo, nao se adapta |
| **OLS** | Estimado | Regressao linear | Otimo assintoticamente | Overfitting com poucos dados |
| **Stacking** | Estimado | Regressao regularizada | Controla overfitting | Requer validacao cruzada |
| **BMA** | Estimado | Probabilidades posteriores | Incorpora incerteza do modelo | Computacionalmente caro |
| **Time-Varying** | Adaptativo | Variam no tempo | Captura mudancas estruturais | Mais parametros, mais complexo |
| **Otima** | Estimado | Minimiza MSE | Teoricamente otimo | Sensivel a estimacao de $\Sigma$ |

---

## Fluxo de Trabalho

O fluxo de combinacao no forecastbox segue tres etapas:

```mermaid
graph LR
    A[Modelos Individuais] --> B[Forecast Containers]
    B --> C["combine()"]
    C --> D[Forecast Combinado]

    style C fill:#009688,stroke:#00796B,color:#fff
```

1. **Gerar previsoes individuais** com `AutoARIMA`, `AutoETS`, etc.
2. **Armazenar** cada previsao em um `Forecast` container
3. **Combinar** usando `combine()` com o metodo desejado

---

## Quick Start

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.combine import combine

# Carregar serie temporal
y = pd.read_csv("ipca.csv", index_col="date", parse_dates=True)["ipca"]

# Gerar previsoes individuais
arima = AutoARIMA(seasonal=True, m=12).fit(y)
ets = AutoETS(seasonal_periods=12).fit(y)

fc_arima = arima.predict(horizon=12)
fc_ets = ets.predict(horizon=12)

# Combinar com media simples
fc_combined = combine(
    forecasts=[fc_arima, fc_ets],
    method="simple",
)
print(fc_combined)
```

```text
Combination Method: Simple Average (N=2)

             point     lo95     hi95
2024-01     4.52     3.89     5.15
2024-02     4.48     3.72     5.24
2024-03     4.55     3.65     5.45
...
2024-12     4.61     3.12     6.10
```

---

## Metodos Disponiveis

<div class="grid cards" markdown>

-   :material-scale-balance:{ .lg .middle } **Media Simples**

    ---

    Pesos iguais para todos os modelos. Surpreendentemente forte baseline.

    [:octicons-arrow-right-24: Media Simples](simple.md)

-   :material-weight:{ .lg .middle } **Pesos Fixos**

    ---

    Pesos definidos pelo usuario ou inversamente proporcionais ao erro.

    [:octicons-arrow-right-24: Pesos Fixos](weighted.md)

-   :material-chart-scatter-plot:{ .lg .middle } **OLS (Granger-Ramanathan)**

    ---

    Combinacao por regressao linear com tres variantes classicas.

    [:octicons-arrow-right-24: OLS](ols.md)

-   :material-layers-triple:{ .lg .middle } **Stacking**

    ---

    Regressao regularizada (Ridge, Lasso) com validacao cruzada.

    [:octicons-arrow-right-24: Stacking](stacking.md)

-   :material-chart-bar-stacked:{ .lg .middle } **BMA**

    ---

    Bayesian Model Averaging com probabilidades posteriores.

    [:octicons-arrow-right-24: BMA](bma.md)

-   :material-chart-timeline-variant:{ .lg .middle } **Time-Varying**

    ---

    Pesos que se adaptam ao longo do tempo via Filtro de Kalman.

    [:octicons-arrow-right-24: Time-Varying](time-varying.md)

-   :material-star-circle:{ .lg .middle } **Otima**

    ---

    Combinacao otima minimizando MSE com estimacao de covariancias.

    [:octicons-arrow-right-24: Otima](optimal.md)

</div>

---

## Referencias

- **Bates, J.M. & Granger, C.W.J.** (1969). "The Combination of Forecasts." *Operational Research Quarterly*, 20(4), 451-468.
- **Smith, J. & Wallis, K.F.** (2009). "A Simple Explanation of the Forecast Combination Puzzle." *Oxford Bulletin of Economics and Statistics*, 71(3), 331-355.
- **Timmermann, A.** (2006). "Forecast Combinations." *Handbook of Economic Forecasting*, 1, 135-196.
