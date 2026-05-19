---
title: "Avaliacao de Previsoes"
description: "Overview completo da pipeline de avaliacao: metricas pontuais e probabilisticas, testes estatisticos de comparacao e cross-validation temporal."
---

# Avaliacao de Previsoes

!!! abstract "Key Takeaway"
    Uma boa previsao nao e apenas aquela com menor erro — e aquela que e **estatisticamente superior**, **nao-viesada** e **robusta** a diferentes amostras. O forecastbox oferece uma pipeline completa de avaliacao que vai de metricas simples a testes formais de comparacao.

## Por que avaliar formalmente?

Comparar previsoes apenas pelo RMSE pode levar a conclusoes erroneas. Diferencas pequenas podem ser fruto do acaso, e um modelo com menor erro medio pode ser inferior em determinados regimes. A avaliacao formal responde perguntas como:

- A diferenca de performance e **estatisticamente significativa**?
- A previsao e **nao-viesada** e **eficiente**?
- Qual o **conjunto de melhores modelos** com garantia estatistica?
- Um modelo **agrega informacao** alem do que outro ja captura?

## Tipos de Avaliacao

O forecastbox organiza a avaliacao em tres dimensoes:

| Dimensao | Pergunta | Ferramentas |
|----------|----------|-------------|
| **Pontual** | Quao proximo do valor realizado? | MSE, RMSE, MAE, MAPE, MASE |
| **Probabilistica** | A distribuicao preditiva e bem calibrada? | CRPS, Log Score, PIT, Winkler |
| **Relativa** | Modelo A e melhor que B? | DM, GW, MCS, Encompassing |

## Pipeline de Avaliacao

A avaliacao segue um fluxo estruturado:

```mermaid
graph LR
    A[Dados] --> B[Split Temporal]
    B --> C[Modelos]
    C --> D[Previsoes]
    D --> E[Metricas]
    E --> F[Testes Estatisticos]
    F --> G[Decisao]
```

### 1. Separacao temporal

```python
from forecastbox.evaluation import TimeSeriesSplit

splitter = TimeSeriesSplit(
    initial_window=100,
    step=1,
    horizon=12
)
```

### 2. Calculo de metricas

```python
from forecastbox.evaluation import forecast_metrics

metrics = forecast_metrics(
    actual=y_test,
    predicted=y_pred,
    metrics=["rmse", "mae", "mase", "crps"]
)
print(metrics)
```

```text
         RMSE      MAE     MASE     CRPS
Model  0.0342   0.0271   0.8914   0.0189
```

### 3. Testes de comparacao

```python
from forecastbox.evaluation import diebold_mariano, mcs

# Comparacao par-a-par
dm = diebold_mariano(errors_1, errors_2, h=1)
print(f"DM = {dm.statistic:.3f}, p = {dm.pvalue:.4f}")

# Comparacao multipla
mcs_result = mcs(loss_matrix, alpha=0.10)
print(f"Modelos no MCS: {mcs_result.superior_models}")
```

### 4. Diagnosticos de qualidade

```python
from forecastbox.evaluation import mincer_zarnowitz

mz = mincer_zarnowitz(actual=y_test, predicted=y_pred)
print(f"alpha={mz.alpha:.4f}, beta={mz.beta:.4f}")
print(f"Teste conjunto (p-valor): {mz.joint_pvalue:.4f}")
```

## Guia de Navegacao

<div class="grid cards" markdown>

- :material-chart-bar: **[Metricas](metrics.md)**

    Metricas pontuais, percentuais, escaladas e probabilisticas

- :material-scale-balance: **[Diebold-Mariano](diebold-mariano.md)**

    Teste de igualdade de poder preditivo entre dois modelos

- :material-select-group: **[Model Confidence Set](mcs.md)**

    Conjunto de modelos superiores com garantia estatistica

- :material-test-tube: **[Giacomini-White](giacomini-white.md)**

    Teste condicional de superioridade preditiva

- :material-target: **[Mincer-Zarnowitz](mincer-zarnowitz.md)**

    Regressao de vies e eficiencia da previsao

- :material-set-merge: **[Encompassing](encompassing.md)**

    Teste se um modelo agrega informacao sobre outro

- :material-refresh: **[Cross-Validation](cross-validation.md)**

    Estrategias de validacao cruzada temporal

</div>

## Exemplo Completo

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.evaluation import (
    forecast_metrics, diebold_mariano, mincer_zarnowitz,
    TimeSeriesSplit
)

# Dados
y = pd.read_csv("gdp.csv", index_col=0, parse_dates=True).squeeze()

# Split
train, test = y[:100], y[100:]

# Modelos
arima = AutoARIMA().fit(train).predict(len(test))
ets = AutoETS().fit(train).predict(len(test))

# Metricas
for name, pred in [("ARIMA", arima), ("ETS", ets)]:
    m = forecast_metrics(test, pred, metrics=["rmse", "mae", "mase"])
    print(f"{name}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, MASE={m['mase']:.4f}")

# Teste DM
dm = diebold_mariano(test - arima, test - ets, h=1)
print(f"\nDM test: stat={dm.statistic:.3f}, p={dm.pvalue:.4f}")

# Diagnostico MZ
mz = mincer_zarnowitz(test, arima)
print(f"MZ: alpha={mz.alpha:.4f}, beta={mz.beta:.4f}, p={mz.joint_pvalue:.4f}")
```
