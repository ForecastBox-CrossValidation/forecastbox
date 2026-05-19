---
title: Quickstart
description: Faca sua primeira previsao com o forecastbox em 5 minutos
---

# Quickstart

Este guia te leva de zero a uma previsao completa com auto-forecast, combinacao
de modelos e avaliacao -- tudo em menos de 5 minutos.

## O que voce vai aprender

- Carregar dados de exemplo
- Gerar uma previsao automatica com AutoARIMA
- Combinar previsoes de 3 modelos
- Avaliar com metricas basicas (RMSE, MAE, MAPE)
- Visualizar o resultado

---

## Step 1: Carregar Dados

O forecastbox inclui datasets de exemplo para experimentacao. Vamos usar
a serie de crescimento do PIB:

```python
from forecastbox.datasets import load_gdp

data = load_gdp()
print(f"Serie: {data.name}")
print(f"Periodos: {len(data)}")
print(data.tail())
```

```text
Serie: gdp_growth
Periodos: 80
2019Q1    1.1
2019Q2    2.0
2019Q3    2.1
2019Q4    2.1
2020Q1   -5.0
Freq: QS, Name: gdp_growth, dtype: float64
```

| Variavel | Descricao |
|----------|-----------|
| `gdp_growth` | Taxa de crescimento trimestral do PIB (%) |

!!! info "Datasets disponiveis"

    O forecastbox inclui varios datasets para experimentacao:
    `load_gdp()`, `load_inflation()`, `load_unemployment()`.
    Veja a [API de Datasets](../api/datasets.md) para a lista completa.

---

## Step 2: Auto-Forecast com AutoARIMA

O `AutoARIMA` seleciona automaticamente a melhor especificacao ARIMA
via criterios de informacao:

```python
from forecastbox import AutoARIMA

model = AutoARIMA()
result = model.fit_predict(data, horizon=4)

print(f"Modelo selecionado: {result.model_name}")
print(f"\nPrevisao (4 trimestres):")
print(result.forecast)
```

```text
Modelo selecionado: ARIMA(1,1,1)

Previsao (4 trimestres):
2020Q2   -3.2
2020Q3    1.8
2020Q4    2.3
2021Q1    2.1
Freq: QS, Name: gdp_growth, dtype: float64
```

!!! tip "Estrategias de busca"

    O `AutoARIMA` suporta duas estrategias de busca:

    - `strategy="stepwise"` (padrao) -- busca rapida via algoritmo stepwise
    - `strategy="exhaustive"` -- busca exaustiva em todo o espaco de modelos

    ```python
    model = AutoARIMA(strategy="exhaustive", ic="bic")
    ```

---

## Step 3: Combinar Previsoes

A combinacao de previsoes tipicamente supera modelos individuais. Vamos
combinar 3 modelos com **media simples**:

```python
from forecastbox import AutoARIMA, AutoETS, Theta, combine

# Ajustar 3 modelos
arima = AutoARIMA().fit_predict(data, horizon=4)
ets = AutoETS().fit_predict(data, horizon=4)
theta = Theta().fit_predict(data, horizon=4)

# Combinar via media simples
combined = combine(
    forecasts=[arima, ets, theta],
    method="simple_average",
)

print("Previsao combinada:")
print(combined.forecast)
print(f"\nPesos: {combined.weights}")
```

```text
Previsao combinada:
2020Q2   -2.9
2020Q3    1.5
2020Q4    2.1
2021Q1    2.0
Freq: QS, Name: combined, dtype: float64

Pesos: {'AutoARIMA': 0.333, 'AutoETS': 0.333, 'Theta': 0.333}
```

!!! note "Metodos de combinacao"

    O forecastbox oferece **7 metodos** de combinacao:

    | Metodo | Descricao |
    |--------|-----------|
    | `simple_average` | Media simples (igual peso) |
    | `weighted` | Pesos fixos definidos pelo usuario |
    | `ols` | Pesos estimados por OLS |
    | `stacking` | Stacking via regressao regularizada |
    | `bma` | Bayesian Model Averaging |
    | `time_varying` | Pesos que variam no tempo |
    | `optimal` | Combinacao otima (Bates-Granger) |

    Veja o [User Guide de Combinacao](../user-guide/combination/index.md) para detalhes.

---

## Step 4: Avaliar Previsoes

Use metricas basicas para avaliar a qualidade das previsoes. Vamos
comparar o modelo individual com a combinacao usando dados de teste:

```python
from forecastbox.evaluate import metrics

# Separar treino e teste
train, test = data[:-4], data[-4:]

# Re-estimar com dados de treino
arima_pred = AutoARIMA().fit_predict(train, horizon=4)
combined_pred = combine(
    forecasts=[
        AutoARIMA().fit_predict(train, horizon=4),
        AutoETS().fit_predict(train, horizon=4),
        Theta().fit_predict(train, horizon=4),
    ],
    method="simple_average",
)

# Calcular metricas
arima_metrics = metrics.compute(test, arima_pred.forecast)
combined_metrics = metrics.compute(test, combined_pred.forecast)

print("           RMSE     MAE    MAPE")
print(f"AutoARIMA  {arima_metrics.rmse:>6.2f}  {arima_metrics.mae:>6.2f}  {arima_metrics.mape:>5.1f}%")
print(f"Combinado  {combined_metrics.rmse:>6.2f}  {combined_metrics.mae:>6.2f}  {combined_metrics.mape:>5.1f}%")
```

```text
           RMSE     MAE    MAPE
AutoARIMA   1.85    1.42   12.3%
Combinado   1.52    1.18    9.8%
```

As metricas calculadas:

| Metrica | Formula | Interpretacao |
|---------|---------|---------------|
| **RMSE** | $$\sqrt{\frac{1}{h}\sum_{t=1}^{h}(y_t - \hat{y}_t)^2}$$ | Erro medio quadratico -- penaliza erros grandes |
| **MAE** | $$\frac{1}{h}\sum_{t=1}^{h}\lvert y_t - \hat{y}_t \rvert$$ | Erro medio absoluto -- mais robusto a outliers |
| **MAPE** | $$\frac{100}{h}\sum_{t=1}^{h}\left\lvert\frac{y_t - \hat{y}_t}{y_t}\right\rvert$$ | Erro percentual -- facilita comparacao entre series |

!!! tip "Avaliacao rigorosa"

    Para comparacoes estatisticamente rigorosas, use o **teste de Diebold-Mariano**:

    ```python
    from forecastbox.evaluate import diebold_mariano

    dm = diebold_mariano(test, arima_pred.forecast, combined_pred.forecast)
    print(dm)
    ```

    Veja o [User Guide de Avaliacao](../user-guide/evaluation/index.md) para detalhes.

---

## Step 5: Visualizar Resultado

Visualize a previsao combinada junto com os dados historicos:

```python
from forecastbox.viz import plot_forecast

fig = plot_forecast(
    actual=data,
    forecast=combined_pred.forecast,
    title="Previsao do PIB -- Combinacao de Modelos",
    ci=True,
)
fig.show()
```

O grafico exibe:

- **Linha azul**: dados historicos observados
- **Linha vermelha**: previsao combinada
- **Area sombreada**: intervalo de confianca de 95%
- **Linha vertical tracejada**: inicio do horizonte de previsao

!!! tip "Exportar grafico"

    Salve o grafico em diferentes formatos:

    ```python
    fig.savefig("forecast_gdp.png", dpi=150, bbox_inches="tight")
    ```

---

## Resumo

Em 5 minutos voce:

1. :material-database: Carregou dados de exemplo com `load_gdp()`
2. :material-auto-fix: Gerou uma previsao automatica com `AutoARIMA`
3. :material-set-merge: Combinou 3 modelos via media simples
4. :material-test-tube: Avaliou com RMSE, MAE e MAPE
5. :material-chart-line: Visualizou o resultado final

---

## Proximos Passos

<div class="grid cards" markdown>

- :material-book-open-variant: **[Conceitos Fundamentais](core-concepts.md)**

    Entenda a arquitetura do forecastbox: modelos, combinacao e avaliacao

- :material-auto-fix: **[Auto-Forecast](../user-guide/auto-forecast/index.md)**

    Guia completo de selecao automatica de modelos

- :material-set-merge: **[Combinacao](../user-guide/combination/index.md)**

    Explore os 7 metodos de combinacao de previsoes

- :material-test-tube: **[Avaliacao](../user-guide/evaluation/index.md)**

    Testes estatisticos rigorosos para comparacao de modelos

</div>
