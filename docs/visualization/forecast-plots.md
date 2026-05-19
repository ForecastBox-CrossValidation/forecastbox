---
title: Graficos de Previsao
description: Visualizacao de previsoes com intervalos de confianca, fan charts e diagnostico de residuos
---

# Graficos de Previsao

Funcoes para visualizar previsoes individuais: serie historica com projecao,
intervalos de confianca, fan charts e diagnostico de residuos.

---

## `plot_forecast`

Grafico principal de previsao: exibe a serie historica observada e a projecao
futura com intervalos de confianca sombreados.

**Output visual**: linha solida para dados historicos, linha tracejada para a
previsao, e faixas sombreadas (mais escuras no centro, mais claras nas bordas)
para os intervalos de confianca. Uma linha vertical pontilhada separa
o historico da previsao.

```python
from forecastbox.plot import plot_forecast

fig = plot_forecast(forecast)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | *required* | Objeto de previsao retornado por `fit_predict` |
| `ci_levels` | `list[float]` | `[0.80, 0.95]` | Niveis de confianca para os intervalos |
| `show_history` | `bool \| int` | `True` | Exibir historico. Se `int`, numero de periodos |
| `date_format` | `str` | `"%Y-%m"` | Formato de datas no eixo x |
| `colors` | `dict \| None` | `None` | Cores customizadas: `{"history", "forecast", "ci"}` |
| `title` | `str \| None` | `None` | Titulo (auto-gerado se `None`) |
| `ylabel` | `str` | `""` | Label do eixo y |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend: `"matplotlib"`, `"plotly"`, `"seaborn"` |
| `style` | `str` | `"light"` | Tema: `"light"`, `"dark"`, `"publication"`, `"presentation"` |
| `ax` | `Axes \| None` | `None` | Eixo matplotlib existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Previsao simples"

    ```python
    from forecastbox import AutoForecast
    from forecastbox.plot import plot_forecast

    model = AutoForecast(strategy="best")
    forecast = model.fit_predict(y, horizon=12)

    plot_forecast(forecast)
    ```

    **Output**: Serie historica completa em azul, previsao 12 meses em azul
    tracejado, faixas de 80% e 95% de confianca em azul translucido.

=== "Customizado"

    ```python
    plot_forecast(
        forecast,
        ci_levels=[0.50, 0.80, 0.95],
        show_history=60,  # ultimos 60 periodos
        date_format="%b %Y",
        colors={"history": "#2196F3", "forecast": "#F44336", "ci": "#FFCDD2"},
        title="IPCA - Previsao 12 meses",
        ylabel="% a.a.",
        style="publication",
        figsize=(8, 5),
    )
    ```

    **Output**: Ultimos 60 meses de historico, previsao em vermelho com 3 faixas
    de confianca em tons de rosa, formatacao academica.

=== "Sem historico"

    ```python
    plot_forecast(
        forecast,
        show_history=False,
        title="Projecao futura",
    )
    ```

    **Output**: Apenas a previsao e intervalos de confianca, sem dados historicos.

---

## `plot_forecast_interactive`

Versao interativa do grafico de previsao usando Plotly. Inclui hover com valores
exatos, zoom, pan e selecao de intervalo.

**Output visual**: Mesmo layout de `plot_forecast`, mas interativo. Hover exibe
data, valor observado/projetado e limites do intervalo de confianca. Toolbar
plotly permite zoom, pan, download PNG e reset de eixos.

```python
from forecastbox.plot import plot_forecast_interactive

fig = plot_forecast_interactive(forecast)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | *required* | Objeto de previsao |
| `ci_levels` | `list[float]` | `[0.80, 0.95]` | Niveis de confianca |
| `show_history` | `bool \| int` | `True` | Exibir historico |
| `title` | `str \| None` | `None` | Titulo |
| `height` | `int` | `500` | Altura em pixels |
| `width` | `int \| None` | `None` | Largura em pixels (`None` = responsivo) |
| `template` | `str` | `"plotly_white"` | Template plotly |

### Exemplo

```python
from forecastbox.plot import plot_forecast_interactive

fig = plot_forecast_interactive(
    forecast,
    ci_levels=[0.50, 0.80, 0.95],
    title="PIB - Previsao Interativa",
    height=600,
)
fig.show()

# Exportar como HTML
fig.write_html("pib_forecast.html", include_plotlyjs="cdn")
```

**Output**: Grafico interativo no browser. Hover sobre qualquer ponto mostra
`Data: Jan 2026 | Valor: 3.45 | IC 80%: [2.90, 4.00] | IC 95%: [2.50, 4.40]`.

---

## `plot_multi_forecast`

Sobreposicao de multiplas previsoes no mesmo grafico para comparacao visual
direta. Cada previsao recebe uma cor distinta com legenda automatica.

**Output visual**: Serie historica unica em preto, N linhas de previsao em
cores distintas, cada uma com sua faixa de confianca (sombreamento leve).
Legenda automatica com nomes dos modelos.

```python
from forecastbox.plot import plot_multi_forecast

fig = plot_multi_forecast([forecast1, forecast2, forecast3])
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecasts` | `list[ForecastResult]` | *required* | Lista de previsoes |
| `labels` | `list[str] \| None` | `None` | Nomes dos modelos (auto-detecta se `None`) |
| `ci_level` | `float \| None` | `0.95` | Nivel de confianca (`None` = sem intervalo) |
| `show_history` | `bool \| int` | `True` | Exibir historico |
| `colors` | `list[str] \| None` | `None` | Paleta customizada |
| `figsize` | `tuple[float, float]` | `(12, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import AutoForecast
from forecastbox.plot import plot_multi_forecast

# Gerar previsoes com diferentes modelos
arima = AutoForecast(strategy="arima").fit_predict(y, horizon=12)
ets = AutoForecast(strategy="ets").fit_predict(y, horizon=12)
theta = AutoForecast(strategy="theta").fit_predict(y, horizon=12)

plot_multi_forecast(
    [arima, ets, theta],
    labels=["ARIMA", "ETS", "Theta"],
    ci_level=0.80,
    show_history=48,
    title="IPCA - Comparacao de Modelos",
    style="publication",
)
```

**Output**: Ultimos 48 meses de historico em preto. Tres linhas de previsao
(ARIMA em azul, ETS em laranja, Theta em verde) com faixas de 80% de confianca
em tons translucidos. Legenda no canto superior direito.

---

## `plot_fan_chart`

Fan chart com multiplos quantis, ideal para comunicar incerteza de forma
graduada. Comumente utilizado por bancos centrais em relatorios de inflacao.

**Output visual**: Serie historica em linha solida. A previsao aparece como
faixas concentricas do quantil central (mais escuro) para os extremos (mais
claro), formando o formato de "leque". A mediana e exibida como linha solida.

O fan chart comunica a distribuicao preditiva completa:

$$
\hat{y}_{t+h|t}^{(q)} \quad \text{para} \quad q \in \{0.10, 0.25, 0.50, 0.75, 0.90\}
$$

```python
from forecastbox.plot import plot_fan_chart

fig = plot_fan_chart(forecast)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | *required* | Objeto de previsao com distribuicao quantilica |
| `quantiles` | `list[float]` | `[0.10, 0.25, 0.50, 0.75, 0.90]` | Quantis a exibir |
| `show_history` | `bool \| int` | `True` | Exibir historico |
| `colormap` | `str` | `"Blues"` | Colormap matplotlib para as faixas |
| `median_color` | `str` | `"#1565C0"` | Cor da linha mediana |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import AutoForecast
from forecastbox.plot import plot_fan_chart

model = AutoForecast(strategy="best")
forecast = model.fit_predict(y, horizon=24, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

plot_fan_chart(
    forecast,
    quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    colormap="RdYlBu_r",
    title="Inflacao - Fan Chart (Banco Central)",
    style="publication",
)
```

**Output**: Historico de inflacao em preto. Projecao 24 meses com 4 faixas
concentricas: 5-95% (mais claro), 10-90%, 25-75% (mais escuro), e mediana
como linha solida azul escuro. Formato classico de relatorio de banco central.

---

## `plot_residuals`

Painel de diagnostico de residuos com 4 graficos: serie temporal dos residuos,
histograma com curva normal, ACF e QQ-plot.

**Output visual**: Grid 2x2 com (1) residuos ao longo do tempo com linha zero,
(2) histograma com kernel density e curva normal sobreposta,
(3) autocorrelacao (ACF) com bandas de confianca, e
(4) QQ-plot contra distribuicao normal teorica.

```python
from forecastbox.plot import plot_residuals

fig = plot_residuals(forecast)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | *required* | Objeto de previsao com residuos |
| `lags` | `int` | `24` | Numero de lags para ACF/PACF |
| `bins` | `int` | `30` | Numero de bins do histograma |
| `qq_dist` | `str` | `"norm"` | Distribuicao teorica para QQ-plot |
| `figsize` | `tuple[float, float]` | `(12, 10)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import AutoForecast
from forecastbox.plot import plot_residuals

model = AutoForecast(strategy="arima")
forecast = model.fit_predict(y, horizon=12)

plot_residuals(forecast, lags=36, style="publication")
```

**Output**: Grid 2x2:

- **Superior esquerdo**: Residuos $\hat{e}_t = y_t - \hat{y}_t$ ao longo do tempo.
  Pontos dispersos ao redor de zero indicam bom ajuste.
- **Superior direito**: Histograma dos residuos com KDE e curva $\mathcal{N}(0, \hat{\sigma}^2)$.
  Simetria sugere normalidade.
- **Inferior esquerdo**: ACF com bandas de 95%. Barras dentro das bandas indicam
  ausencia de autocorrelacao (residuos white noise).
- **Inferior direito**: QQ-plot. Pontos sobre a diagonal indicam normalidade dos residuos.

---

## `plot_residuals_extended`

Painel estendido com 6 graficos de diagnostico: serie temporal, histograma,
ACF, PACF, QQ-plot e residuos ao quadrado (deteccao de heterocedasticidade).

**Output visual**: Grid 3x2 que inclui todos os graficos de `plot_residuals`
mais PACF e residuos ao quadrado com ACF para detectar efeitos ARCH.

```python
from forecastbox.plot import plot_residuals_extended

fig = plot_residuals_extended(forecast)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | *required* | Objeto de previsao com residuos |
| `lags` | `int` | `24` | Numero de lags para ACF/PACF |
| `bins` | `int` | `30` | Bins do histograma |
| `figsize` | `tuple[float, float]` | `(14, 12)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_residuals_extended

plot_residuals_extended(forecast, lags=48, style="publication")
```

**Output**: Grid 3x2:

- **Linha 1**: Residuos $\hat{e}_t$ + Histograma com KDE
- **Linha 2**: ACF + PACF com bandas de 95%
- **Linha 3**: QQ-plot + ACF de $\hat{e}_t^2$ (teste visual de efeitos ARCH)

!!! tip "Deteccao de heterocedasticidade"

    Se o grafico de $\hat{e}_t^2$ mostra autocorrelacao significativa nos
    primeiros lags, considere usar modelos GARCH (via archbox) para capturar
    a volatilidade condicional.

---

## `plot_forecast_decomposition`

Decomposicao visual da previsao em componentes: tendencia, sazonalidade e
residuo, alinhados verticalmente para inspecao.

**Output visual**: Painel vertical com 4 graficos alinhados (mesma escala
temporal no eixo x): serie original, componente de tendencia, componente
sazonal, e residuo. Uteis para entender as fontes de variacao.

```python
from forecastbox.plot import plot_forecast_decomposition

fig = plot_forecast_decomposition(forecast)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | *required* | Previsao com decomposicao |
| `components` | `list[str] \| None` | `None` | Componentes a exibir (todos se `None`) |
| `figsize` | `tuple[float, float]` | `(12, 10)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import AutoForecast
from forecastbox.plot import plot_forecast_decomposition

model = AutoForecast(strategy="ets")
forecast = model.fit_predict(y, horizon=12)

plot_forecast_decomposition(
    forecast,
    components=["trend", "seasonal", "residual"],
    title="IPCA - Decomposicao ETS",
    style="publication",
)
```

**Output**: 4 paineis verticais:

1. **Observado**: Serie original $y_t$
2. **Tendencia**: Componente $T_t$ extraida pelo modelo
3. **Sazonalidade**: Padrao sazonal $S_t$ (repetitivo)
4. **Residuo**: $e_t = y_t - T_t - S_t$

---

## Exemplos Completos

### Workflow de previsao com visualizacao

```python
import pandas as pd
from forecastbox import AutoForecast
from forecastbox.plot import (
    plot_forecast,
    plot_fan_chart,
    plot_residuals,
    plot_forecast_decomposition,
    set_theme,
)

# Configurar tema
set_theme("publication")

# Dados
y = pd.read_csv("ipca.csv", index_col="date", parse_dates=True)["ipca"]

# Modelo e previsao
model = AutoForecast(strategy="best")
forecast = model.fit_predict(y, horizon=12, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

# Grafico principal
plot_forecast(
    forecast,
    ci_levels=[0.80, 0.95],
    title="IPCA - Previsao AutoForecast",
    ylabel="% a.a.",
)

# Fan chart
plot_fan_chart(forecast, title="IPCA - Distribuicao Preditiva")

# Diagnostico de residuos
plot_residuals(forecast, lags=36)

# Decomposicao
plot_forecast_decomposition(forecast)
```

### Painel customizado com subplots

```python
import matplotlib.pyplot as plt
from forecastbox.plot import plot_forecast, plot_residuals, plot_fan_chart

fig = plt.figure(figsize=(16, 10))

# Layout customizado
ax1 = fig.add_subplot(2, 2, (1, 2))  # previsao ocupa topo inteiro
ax2 = fig.add_subplot(2, 2, 3)       # fan chart
ax3 = fig.add_subplot(2, 2, 4)       # residuos (serie temporal)

plot_forecast(forecast, ax=ax1, show=False, title="Previsao")
plot_fan_chart(forecast, ax=ax2, show=False, title="Fan Chart")
plot_residuals(forecast, ax=ax3, show=False)

fig.suptitle("Dashboard de Previsao - IPCA", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("dashboard.pdf", dpi=300)
plt.show()
```

---

## See Also

- :material-school: [Tutorial: Fundamentos de Previsao](../tutorials/fundamentals.md) — aprenda a gerar e visualizar previsoes
- [Comparacao de Modelos](comparison-plots.md) — graficos para comparar multiplos modelos
- [Combinacao](combination-plots.md) — graficos de pesos e combinacao
- [User Guide - Auto-Forecast](../user-guide/auto-forecast/index.md) — como gerar previsoes
- [API Reference - Visualization](../api/visualization.md) — referencia completa
