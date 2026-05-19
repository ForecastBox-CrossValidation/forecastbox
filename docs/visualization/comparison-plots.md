---
title: Graficos de Comparacao
description: Visualizacao comparativa de modelos com metricas, rankings e distribuicao de erros
---

# Graficos de Comparacao

Funcoes para comparar visualmente o desempenho de multiplos modelos de previsao:
previsoes lado a lado, barras de metricas, heatmaps, distribuicao de erros,
erros acumulados e rankings.

---

## `plot_comparison`

Grafico de previsoes de N modelos sobrepostos com dados observados.
Permite comparacao visual direta da aderencia de cada modelo.

**Output visual**: Serie observada em preto (linha grossa), N linhas de previsao
em cores distintas, legenda com nomes dos modelos. Se `show_ci=True`, faixas
de confianca translucidas para cada modelo.

```python
from forecastbox.plot import plot_comparison

fig = plot_comparison(forecasts)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecasts` | `list[ForecastResult]` | *required* | Lista de previsoes a comparar |
| `actual` | `pd.Series \| None` | `None` | Valores observados (para out-of-sample) |
| `labels` | `list[str] \| None` | `None` | Nomes dos modelos |
| `show_ci` | `bool` | `False` | Exibir intervalos de confianca |
| `ci_level` | `float` | `0.95` | Nivel de confianca |
| `colors` | `list[str] \| None` | `None` | Paleta customizada |
| `figsize` | `tuple[float, float]` | `(12, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import AutoForecast
from forecastbox.plot import plot_comparison

# Gerar previsoes de multiplos modelos
models = {
    "ARIMA": AutoForecast(strategy="arima"),
    "ETS": AutoForecast(strategy="ets"),
    "Theta": AutoForecast(strategy="theta"),
    "VAR": AutoForecast(strategy="var"),
    "Best": AutoForecast(strategy="best"),
}

forecasts = []
labels = []
for name, model in models.items():
    fc = model.fit_predict(y_train, horizon=12)
    forecasts.append(fc)
    labels.append(name)

plot_comparison(
    forecasts,
    actual=y_test,
    labels=labels,
    title="IPCA - Comparacao de 5 Modelos",
    style="publication",
)
```

**Output**: Dados observados (teste) em preto com marcadores, 5 linhas de
previsao em cores distintas. Desvios visiveis entre modelos e observado
permitem identificar rapidamente os melhores ajustes.

---

## `plot_metrics_bar`

Grafico de barras comparando metricas de erro por modelo. Permite visualizar
rapidamente qual modelo tem menor erro em cada metrica.

**Output visual**: Barras agrupadas (uma cor por modelo) para cada metrica.
Valores exatos exibidos acima de cada barra. O melhor modelo (menor erro)
e destacado com borda ou asterisco.

```python
from forecastbox.plot import plot_metrics_bar

fig = plot_metrics_bar(results)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `results` | `EvaluationResult` | *required* | Resultado de avaliacao com metricas |
| `metrics` | `list[str] \| None` | `None` | Metricas a exibir (todas se `None`) |
| `sort_by` | `str \| None` | `None` | Metrica para ordenar modelos |
| `top_n` | `int \| None` | `None` | Exibir apenas top N modelos |
| `show_values` | `bool` | `True` | Exibir valores sobre as barras |
| `highlight_best` | `bool` | `True` | Destacar melhor modelo por metrica |
| `colors` | `list[str] \| None` | `None` | Paleta customizada |
| `orientation` | `str` | `"vertical"` | `"vertical"` ou `"horizontal"` |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import evaluate
from forecastbox.plot import plot_metrics_bar

results = evaluate.compare(
    actual=y_test,
    forecasts={"ARIMA": fc_arima, "ETS": fc_ets, "Theta": fc_theta},
    metrics=["rmse", "mae", "mape", "mase"],
)

plot_metrics_bar(
    results,
    metrics=["rmse", "mae", "mape"],
    sort_by="rmse",
    highlight_best=True,
    title="Metricas de Erro por Modelo",
    style="publication",
)
```

**Output**: 3 grupos de barras (RMSE, MAE, MAPE), cada um com 3 barras
(ARIMA, ETS, Theta). Modelo com menor RMSE aparece primeiro. Barra do
melhor modelo em cada metrica com borda destacada.

---

## `plot_metrics_heatmap`

Heatmap de metricas vs. modelos. Celulas coloridas por intensidade (verde =
melhor, vermelho = pior) com valores numericos no centro de cada celula.

**Output visual**: Matriz com modelos nas linhas, metricas nas colunas.
Gradiente de cor do verde (melhor) ao vermelho (pior), normalizado por coluna.
Valores numericos formatados dentro de cada celula.

```python
from forecastbox.plot import plot_metrics_heatmap

fig = plot_metrics_heatmap(results)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `results` | `EvaluationResult` | *required* | Resultado de avaliacao |
| `metrics` | `list[str] \| None` | `None` | Metricas a exibir |
| `normalize` | `bool` | `True` | Normalizar cores por coluna (0-1) |
| `annot` | `bool` | `True` | Exibir valores nas celulas |
| `fmt` | `str` | `".3f"` | Formato numerico |
| `cmap` | `str` | `"RdYlGn_r"` | Colormap (vermelho=pior, verde=melhor) |
| `sort_by` | `str \| None` | `None` | Metrica para ordenar linhas |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend: `"matplotlib"` ou `"seaborn"` |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import evaluate
from forecastbox.plot import plot_metrics_heatmap

results = evaluate.compare(
    actual=y_test,
    forecasts={
        "ARIMA": fc_arima,
        "ETS": fc_ets,
        "Theta": fc_theta,
        "VAR": fc_var,
        "Combinacao": fc_combined,
    },
    metrics=["rmse", "mae", "mape", "mase", "smape"],
)

plot_metrics_heatmap(
    results,
    sort_by="rmse",
    cmap="RdYlGn_r",
    title="Heatmap de Metricas - 5 Modelos",
    style="publication",
)
```

**Output**: Tabela 5x5 colorida. Modelo "Combinacao" com celulas verdes
(melhores metricas) no topo. Theta com celulas amarelas/vermelhas na base.
Formato visual imediato para identificar o melhor modelo por metrica.

---

## `plot_error_boxplot`

Boxplot da distribuicao de erros de previsao por modelo. Permite comparar
a dispersao, mediana e outliers de cada modelo.

**Output visual**: Boxplots lado a lado (um por modelo), ordenados por mediana.
Whiskers nos percentis 5-95%, caixa nos quartis 25-75%, mediana como linha,
outliers como pontos individuais.

Os erros de previsao sao definidos como:

$$
e_{t+h|t} = y_{t+h} - \hat{y}_{t+h|t}
$$

```python
from forecastbox.plot import plot_error_boxplot

fig = plot_error_boxplot(results)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `results` | `EvaluationResult` | *required* | Resultado de avaliacao com erros |
| `error_type` | `str` | `"raw"` | Tipo de erro: `"raw"`, `"absolute"`, `"squared"`, `"percentage"` |
| `sort_by` | `str` | `"median"` | Ordenar por: `"median"`, `"iqr"`, `"name"` |
| `show_mean` | `bool` | `True` | Exibir marcador da media |
| `show_points` | `bool` | `False` | Exibir pontos individuais (jitter) |
| `colors` | `list[str] \| None` | `None` | Paleta customizada |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_error_boxplot

plot_error_boxplot(
    results,
    error_type="absolute",
    sort_by="median",
    show_mean=True,
    title="Distribuicao de Erros Absolutos por Modelo",
    style="publication",
)
```

**Output**: 5 boxplots (ARIMA, ETS, Theta, VAR, Combinacao) ordenados pela
mediana do erro absoluto. O modelo Combinacao com boxplot mais compacto
(menor dispersao) e mediana mais baixa. Diamantes indicam a media.

---

## `plot_cumulative_error`

Erro acumulado ao longo do tempo para cada modelo. Revela se um modelo
comete erros concentrados em periodos especificos ou de forma distribuida.

**Output visual**: N linhas (uma por modelo) mostrando a soma acumulada
dos erros absolutos ao longo do horizonte de previsao. Modelo com linha
mais baixa tem menor erro acumulado. Cruzamentos indicam mudancas de
dominancia relativa.

O erro acumulado no horizonte $h$ e:

$$
CE_h = \sum_{j=1}^{h} |e_{t+j|t}|
$$

```python
from forecastbox.plot import plot_cumulative_error

fig = plot_cumulative_error(results)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `results` | `EvaluationResult` | *required* | Resultado de avaliacao |
| `error_type` | `str` | `"absolute"` | Tipo: `"absolute"`, `"squared"` |
| `labels` | `list[str] \| None` | `None` | Nomes dos modelos |
| `colors` | `list[str] \| None` | `None` | Paleta customizada |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_cumulative_error

plot_cumulative_error(
    results,
    error_type="absolute",
    title="Erro Absoluto Acumulado",
    style="publication",
)
```

**Output**: 5 linhas ascendentes. ARIMA e ETS proximos nos primeiros 6 meses,
depois ARIMA diverge para cima (erros crescentes no longo prazo). Combinacao
consistentemente abaixo dos modelos individuais.

!!! tip "Interpretacao"

    Inclinacoes acentuadas indicam periodos de erro elevado. Se um modelo
    tem erro acumulado baixo mas com degraus, seus erros estao concentrados
    em poucos periodos. Uma subida suave indica erros distribuidos uniformemente.

---

## `plot_ranking`

Ranking visual de modelos por metrica. Exibe posicoes de forma intuitiva
com cores indicando desempenho relativo.

**Output visual**: Grafico de pontos (lollipop) ou barras horizontais com
modelos no eixo y e valor da metrica no eixo x. Posicoes numeradas (1o, 2o, 3o)
com gradiente de cor do melhor (verde) ao pior (vermelho).

```python
from forecastbox.plot import plot_ranking

fig = plot_ranking(results, metric="rmse")
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `results` | `EvaluationResult` | *required* | Resultado de avaliacao |
| `metric` | `str` | `"rmse"` | Metrica para ranking |
| `top_n` | `int \| None` | `None` | Exibir apenas top N |
| `chart_type` | `str` | `"lollipop"` | Tipo: `"lollipop"`, `"bar"`, `"dot"` |
| `show_values` | `bool` | `True` | Exibir valores numericos |
| `colors` | `list[str] \| str \| None` | `None` | Paleta ou colormap |
| `figsize` | `tuple[float, float]` | `(8, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_ranking

plot_ranking(
    results,
    metric="rmse",
    chart_type="lollipop",
    title="Ranking de Modelos por RMSE",
    style="publication",
)
```

**Output**: 5 linhas horizontais com circulos na ponta (lollipop). Combinacao
no topo (menor RMSE = 0.342) em verde escuro, seguido por ETS (0.378) em
verde claro, ARIMA (0.401) em amarelo, Theta (0.445) em laranja e VAR (0.512)
em vermelho. Valores exatos a direita de cada ponto.

---

## `plot_error_by_horizon`

Erro medio por horizonte de previsao para cada modelo. Revela como o
desempenho degrada conforme o horizonte aumenta.

**Output visual**: Eixo x com horizonte (h=1, h=2, ..., h=H), eixo y com
metrica de erro. Uma linha por modelo. Tipicamente todas as linhas sobem,
mas a taxa de crescimento varia entre modelos.

```python
from forecastbox.plot import plot_error_by_horizon

fig = plot_error_by_horizon(results)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `results` | `EvaluationResult` | *required* | Resultado com erros por horizonte |
| `metric` | `str` | `"rmse"` | Metrica: `"rmse"`, `"mae"`, `"mape"` |
| `labels` | `list[str] \| None` | `None` | Nomes dos modelos |
| `colors` | `list[str] \| None` | `None` | Paleta |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_error_by_horizon

plot_error_by_horizon(
    results,
    metric="rmse",
    title="RMSE por Horizonte de Previsao",
    style="publication",
)
```

**Output**: 5 linhas ascendentes (h=1 a h=12). ETS melhor nos primeiros 3
meses, ARIMA domina no horizonte 4-8, Combinacao consistentemente proxima
ao melhor modelo em todos os horizontes.

---

## Exemplos Completos

### Dashboard de comparacao de 5 modelos

```python
import matplotlib.pyplot as plt
from forecastbox import AutoForecast, evaluate
from forecastbox.plot import (
    plot_comparison,
    plot_metrics_bar,
    plot_metrics_heatmap,
    plot_error_boxplot,
    plot_cumulative_error,
    plot_ranking,
    set_theme,
)

set_theme("publication")

# Dados e modelos
models = ["arima", "ets", "theta", "var", "best"]
forecasts = {}
for m in models:
    fc = AutoForecast(strategy=m).fit_predict(y_train, horizon=12)
    forecasts[m.upper()] = fc

# Avaliacao
results = evaluate.compare(
    actual=y_test,
    forecasts=forecasts,
    metrics=["rmse", "mae", "mape", "mase"],
)

# Dashboard 3x2
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

plot_comparison(list(forecasts.values()), actual=y_test,
               labels=list(forecasts.keys()), ax=axes[0, 0], show=False)
plot_metrics_bar(results, sort_by="rmse", ax=axes[0, 1], show=False)
plot_metrics_heatmap(results, ax=axes[1, 0], show=False)
plot_error_boxplot(results, error_type="absolute", ax=axes[1, 1], show=False)
plot_cumulative_error(results, ax=axes[2, 0], show=False)
plot_ranking(results, metric="rmse", ax=axes[2, 1], show=False)

fig.suptitle("Dashboard de Comparacao - IPCA", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("comparison_dashboard.pdf", dpi=300)
plt.show()
```

### Comparacao interativa com Plotly

```python
from forecastbox.plot import plot_comparison, plot_metrics_bar

# Comparacao interativa
fig = plot_comparison(
    list(forecasts.values()),
    actual=y_test,
    labels=list(forecasts.keys()),
    show_ci=True,
    backend="plotly",
    title="Comparacao Interativa de Modelos",
)
fig.write_html("comparison_interactive.html")

# Barras interativas
fig_bar = plot_metrics_bar(
    results,
    sort_by="rmse",
    backend="plotly",
    title="Metricas por Modelo",
)
fig_bar.write_html("metrics_bar.html")
```

---

## See Also

- :material-school: [Tutorial: Avaliacao Rigorosa](../tutorials/evaluation.md) — aprenda a comparar modelos com testes estatisticos
- [Graficos de Previsao](forecast-plots.md) — visualizacao de previsoes individuais
- [Graficos de Combinacao](combination-plots.md) — visualizacao de pesos e combinacao
- [User Guide - Avaliacao](../user-guide/evaluation/index.md) — testes e metricas
- [API Reference - Visualization](../api/visualization.md) — referencia completa
