---
title: Auto-Forecast
description: Selecao automatica de modelos e hiperparametros com AutoARIMA, AutoETS, AutoVAR e ModelZoo.
---

# Auto-Forecast

O modulo **auto-forecast** automatiza a selecao de modelos e hiperparametros para
series temporais. Em vez de especificar manualmente a ordem de um ARIMA ou o tipo de
suavizacao exponencial, o forecastbox testa combinacoes de parametros e seleciona o
melhor modelo via criterios de informacao.

---

## O que e Auto-Forecast?

Auto-forecast e o processo de **selecao automatica de modelo e hiperparametros**.
Dado uma serie temporal, o algoritmo:

1. **Pre-processa** os dados (testes de raiz unitaria, deteccao de sazonalidade)
2. **Gera candidatos** via grid search ou busca stepwise
3. **Estima** cada candidato e calcula o criterio de informacao (AIC, BIC, AICc)
4. **Seleciona** o modelo com menor criterio
5. **Gera previsoes** com intervalos de confianca

```mermaid
graph LR
    A[Serie Temporal] --> B[Pre-processamento]
    B --> C[Gerar Candidatos]
    C --> D[Estimar Modelos]
    D --> E[Selecionar Melhor]
    E --> F[Forecast]

    style E fill:#009688,stroke:#00796B,color:#fff
```

---

## Modelos Disponiveis

| Modelo | Classe | Descricao |
|:-------|:-------|:----------|
| ARIMA | `AutoARIMA` | Modelos ARIMA sazonais com selecao automatica de ordens |
| ETS | `AutoETS` | Suavizacao exponencial com selecao de componentes |
| VAR | `AutoVAR` | Vetores autoregressivos para series multivariadas |
| Custom | `ModelZoo` | Registro de modelos customizados para auto-selecao |

---

## Quick Start

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS

# Carregar serie temporal mensal
y = pd.read_csv("pib_mensal.csv", index_col="date", parse_dates=True)["pib"]

# AutoARIMA com busca stepwise (padrao)
arima = AutoARIMA(seasonal=True, m=12)
arima.fit(y)
forecast_arima = arima.predict(horizon=12)

# AutoETS
ets = AutoETS(seasonal_periods=12)
ets.fit(y)
forecast_ets = ets.predict(horizon=12)
```

```text
AutoARIMA: ARIMA(1,1,1)(0,1,1)[12] — AIC: 1523.4
AutoETS:   ETS(M,Ad,M)             — AIC: 1498.7
```

---

## Auto-Forecast vs statsmodels/pmdarima

O **forecastbox** se diferencia das alternativas existentes em Python:

| Feature | statsmodels | pmdarima | forecastbox |
|:--------|:-----------|:---------|:------------|
| AutoARIMA | :material-close: Manual | :material-check: Stepwise | :material-check: Stepwise + Grid |
| AutoETS | :material-close: Nao | :material-close: Nao | :material-check: 30 combinacoes |
| AutoVAR | :material-close: Manual | :material-close: Nao | :material-check: Lag selection |
| Criterios | AIC/BIC | AIC/BIC/AICC | AIC/BIC/AICC/HQIC |
| Combinacao | :material-close: Nao | :material-close: Nao | :material-check: 7 metodos |
| Pipeline | :material-close: Nao | :material-close: Nao | :material-check: Producao |
| Integracao chronobox | :material-close: Nao | :material-close: Nao | :material-check: Nativo |

!!! info "Integracao com chronobox"

    O forecastbox usa objetos `TimeSeries` do **chronobox** como formato nativo.
    Transformacoes (log, diferenca, dessazonalizacao) sao aplicadas automaticamente
    e revertidas apos a previsao.

---

## AutoSelect

Para selecionar automaticamente entre **todos** os modelos disponiveis:

```python
from forecastbox.auto import AutoSelect

model = AutoSelect(
    models=["arima", "ets", "theta"],
    criterion="aicc",
    seasonal=True,
    m=12,
)
model.fit(y)
best = model.best_model_
print(f"Melhor modelo: {best.summary()}")
```

!!! tip "Quando usar AutoSelect"

    Use `AutoSelect` quando voce nao tem uma preferencia forte por um tipo de modelo.
    Ele compara ARIMA, ETS e outros modelos registrados no ModelZoo e retorna o
    melhor segundo o criterio de informacao escolhido.

---

## Proximos Passos

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **AutoARIMA**

    ---

    Algoritmo de selecao, parametros, testes de raiz unitaria e exemplos completos.

    [:octicons-arrow-right-24: AutoARIMA](auto-arima.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **AutoETS**

    ---

    Taxonomia ETS, 30 combinacoes, equacoes de estado e exemplos praticos.

    [:octicons-arrow-right-24: AutoETS](auto-ets.md)

</div>
