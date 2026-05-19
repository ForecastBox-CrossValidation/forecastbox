---
title: "Cross-Validation Temporal"
description: "Estrategias de validacao cruzada para series temporais: expanding window, rolling window e blocked, com diagramas e exemplos completos."
---

# Cross-Validation Temporal

!!! abstract "Key Takeaway"
    Cross-validation temporal e a forma correta de avaliar previsoes out-of-sample. Diferente de CV classica, respeita a **ordenacao temporal** dos dados. O forecastbox implementa tres estrategias: **expanding window**, **rolling window** e **blocked CV**.

## Por que CV Temporal?

- CV classica (k-fold) **viola a causalidade** — usa dados futuros para prever o passado
- Um unico split treino/teste depende de **onde o corte foi feito**
- CV temporal gera **multiplas avaliacoes**, cada uma respeitando a ordem temporal

## Estrategias

### Expanding Window

A janela de treino **cresce** a cada iteracao. Usa toda a informacao disponivel ate o momento.

```text
Iteracao 1: [===TREINO===][TEST].....................
Iteracao 2: [====TREINO====][TEST]..................
Iteracao 3: [=====TREINO=====][TEST]...............
Iteracao 4: [======TREINO======][TEST]............
Iteracao 5: [=======TREINO=======][TEST].........
```

- **Vantagem**: maximiza dados de treino, estavel para amostras pequenas
- **Desvantagem**: assume que dados antigos continuam relevantes
- **Quando usar**: series estacionarias ou com tendencia estavel

```python
from forecastbox.evaluation import TimeSeriesSplit

cv = TimeSeriesSplit(
    strategy="expanding",
    initial_window=60,   # primeiros 60 obs. para treino
    step=1,              # avanca 1 obs. por iteracao
    horizon=12           # preve 12 passos a frente
)
```

### Rolling Window

A janela de treino tem **tamanho fixo** e desliza no tempo. Descarta dados antigos.

```text
Iteracao 1: [===TREINO===][TEST].....................
Iteracao 2: .[===TREINO===][TEST]..................
Iteracao 3: ..[===TREINO===][TEST]...............
Iteracao 4: ...[===TREINO===][TEST]............
Iteracao 5: ....[===TREINO===][TEST].........
```

- **Vantagem**: adapta-se a mudancas estruturais, dados recentes tem mais peso
- **Desvantagem**: descarta informacao potencialmente util
- **Quando usar**: series com quebras estruturais ou instabilidade de parametros

```python
cv = TimeSeriesSplit(
    strategy="rolling",
    initial_window=60,   # janela fixa de 60 obs.
    step=1,
    horizon=12
)
```

### Blocked CV

Divide a serie em **blocos contiguos** com gap entre treino e teste para evitar vazamento de informacao.

```text
Bloco 1: [==TREINO==][gap][TEST].....................
Bloco 2: ........[==TREINO==][gap][TEST]...........
Bloco 3: ................[==TREINO==][gap][TEST]...
```

- **Vantagem**: evita vazamento por autocorrelacao entre treino e teste
- **Desvantagem**: menos avaliacoes, menor uso dos dados
- **Quando usar**: series com alta autocorrelacao ou quando o horizonte e grande

```python
cv = TimeSeriesSplit(
    strategy="blocked",
    n_blocks=5,
    gap=6,         # gap de 6 obs. entre treino e teste
    horizon=12
)
```

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `strategy` | str | `"expanding"` | `"expanding"`, `"rolling"`, `"blocked"` |
| `initial_window` | int | — | Tamanho minimo (expanding) ou fixo (rolling) do treino |
| `step` | int | `1` | Quantas obs. avancar entre iteracoes |
| `horizon` | int | — | Horizonte de previsao ($h$) |
| `n_blocks` | int | — | Numero de blocos (blocked CV) |
| `gap` | int | `0` | Gap entre treino e teste |

## Multi-Step vs Direct Forecasting

Na CV temporal, a forma de gerar previsoes multi-step importa:

=== "Iterativo (Recursive)"

    Preve um passo a frente e usa a previsao como input para o proximo:

    $$
    \hat{y}_{t+1} = f(y_t, y_{t-1}, \ldots) \quad \rightarrow \quad \hat{y}_{t+2} = f(\hat{y}_{t+1}, y_t, \ldots)
    $$

    - Acumula erro ao longo do horizonte
    - Usa um unico modelo

    ```python
    cv_results = cv.evaluate(
        model=AutoARIMA(),
        data=y,
        forecast_method="recursive"
    )
    ```

=== "Direto (Direct)"

    Treina um modelo separado para cada horizonte:

    $$
    \hat{y}_{t+h} = f_h(y_t, y_{t-1}, \ldots), \quad h = 1, 2, \ldots, H
    $$

    - Nao acumula erro
    - Requer $H$ modelos separados (mais custoso)

    ```python
    cv_results = cv.evaluate(
        model=AutoARIMA(),
        data=y,
        forecast_method="direct"
    )
    ```

## Exemplo Completo

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.evaluation import TimeSeriesSplit, forecast_metrics

# Dados mensais
y = pd.read_csv("industrial_production.csv", index_col=0, parse_dates=True).squeeze()

# Configurar CV expanding
cv = TimeSeriesSplit(
    strategy="expanding",
    initial_window=120,  # 10 anos de treino inicial
    step=1,
    horizon=12           # prever 12 meses a frente
)

# Avaliar dois modelos
models = {
    "ARIMA": AutoARIMA(),
    "ETS": AutoETS(),
}

results = {}
for name, model in models.items():
    cv_result = cv.evaluate(model=model, data=y)
    results[name] = cv_result

# Resumo das metricas
for name, res in results.items():
    print(f"\n{name}:")
    print(f"  RMSE medio: {res.mean_metrics['rmse']:.4f}")
    print(f"  MAE medio:  {res.mean_metrics['mae']:.4f}")
    print(f"  MASE medio: {res.mean_metrics['mase']:.4f}")
    print(f"  N. folds:   {res.n_folds}")
```

```text
ARIMA:
  RMSE medio: 0.0342
  MAE medio:  0.0271
  MASE medio: 0.8914
  N. folds:   48

ETS:
  RMSE medio: 0.0389
  MAE medio:  0.0298
  MASE medio: 0.9812
  N. folds:   48
```

### Analise por horizonte

```python
# Metricas por horizonte de previsao
horizon_metrics = results["ARIMA"].metrics_by_horizon

print("\nRMSE por horizonte (ARIMA):")
print(horizon_metrics["rmse"])
```

```text
RMSE por horizonte (ARIMA):
  h=1:   0.0198
  h=3:   0.0267
  h=6:   0.0321
  h=12:  0.0456
```

### Comparacao por fold

```python
from forecastbox.evaluation import diebold_mariano

# Usar erros da CV para teste DM
e_arima = results["ARIMA"].all_errors
e_ets = results["ETS"].all_errors

dm = diebold_mariano(e_arima, e_ets, h=1)
print(f"\nDM test (CV errors): stat={dm.statistic:.3f}, p={dm.pvalue:.4f}")
```

## Comparacao das Estrategias

| Caracteristica | Expanding | Rolling | Blocked |
|----------------|-----------|---------|---------|
| Janela de treino | Cresce | Fixa | Variavel |
| Dados antigos | Mantidos | Descartados | Parcialmente |
| N. avaliacoes | Alto | Alto | Baixo |
| Robustez a quebras | Baixa | Alta | Media |
| Uso dos dados | Maximo | Moderado | Menor |
| Vazamento temporal | Possivel (autocorr.) | Possivel (autocorr.) | Controlado (gap) |

!!! tip "Recomendacoes"
    - **Padrao**: comece com `expanding` — e a estrategia mais utilizada em macroeconomia
    - **Instabilidade**: use `rolling` se suspeita de quebras estruturais ou mudanca de regime
    - **Alta autocorrelacao**: use `blocked` com `gap` para evitar vazamento
    - **Tamanho de `step`**: `step=1` da mais avaliacoes mas gera erros correlacionados; `step=horizon` garante erros independentes

## Ver Tambem

- [Metricas](metrics.md) — metricas calculadas em cada fold
- [Diebold-Mariano](diebold-mariano.md) — teste estatistico com erros da CV
- [Model Confidence Set](mcs.md) — selecao de modelos com resultados da CV
