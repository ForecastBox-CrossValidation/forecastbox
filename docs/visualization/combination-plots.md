---
title: Graficos de Combinacao
description: Visualizacao de pesos de combinacao, evolucao temporal, posterior BMA e estabilidade
---

# Graficos de Combinacao

Funcoes para visualizar os resultados de combinacao de previsoes: pesos dos
modelos, evolucao temporal, posterior probabilities (BMA), estabilidade e
comparacao combinacao vs. modelos individuais.

---

## `plot_weights`

Grafico de barras com o peso atribuido a cada modelo na combinacao.
Permite identificar rapidamente quais modelos contribuem mais.

**Output visual**: Barras horizontais ou verticais, uma por modelo, ordenadas
por peso (maior para menor). Valor exato exibido junto a cada barra.
Barra do modelo com maior peso destacada em cor mais intensa.

Os pesos satisfazem a restricao:

$$
\sum_{i=1}^{M} w_i = 1, \quad w_i \geq 0
$$

```python
from forecastbox.plot import plot_weights

fig = plot_weights(combination)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `combination` | `CombinationResult` | *required* | Resultado de combinacao |
| `model_names` | `list[str] \| None` | `None` | Nomes dos modelos |
| `sort` | `bool` | `True` | Ordenar por peso |
| `highlight_top` | `int` | `1` | Destacar top N modelos |
| `show_values` | `bool` | `True` | Exibir valores nas barras |
| `orientation` | `str` | `"horizontal"` | `"horizontal"` ou `"vertical"` |
| `colors` | `list[str] \| None` | `None` | Paleta customizada |
| `figsize` | `tuple[float, float]` | `(8, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import combine
from forecastbox.plot import plot_weights

combination = combine(
    forecasts=[fc_arima, fc_ets, fc_theta, fc_var],
    method="bma",
    actual=y_train,
)

plot_weights(
    combination,
    model_names=["ARIMA", "ETS", "Theta", "VAR"],
    highlight_top=2,
    title="Pesos BMA",
    style="publication",
)
```

**Output**: 4 barras horizontais ordenadas. ARIMA (w=0.42) em azul escuro
(destaque), ETS (w=0.31) em azul medio (destaque), Theta (w=0.18) em cinza,
VAR (w=0.09) em cinza claro. Valores exatos a direita de cada barra.

---

## `plot_weights_evolution`

Grafico de area empilhada mostrando a evolucao temporal dos pesos de cada
modelo. Essencial para metodos time-varying (Kalman, forgetting factor).

**Output visual**: Area chart empilhado (soma = 1 em cada periodo). Cada
cor representa um modelo. A largura da faixa indica o peso do modelo
naquele periodo. Mudancas bruscas indicam instabilidade ou eventos.

Para metodos time-varying, os pesos sao funcao do tempo:

$$
w_{i,t} = f(e_{1,t}, \ldots, e_{M,t}), \quad \sum_{i=1}^{M} w_{i,t} = 1 \;\; \forall t
$$

```python
from forecastbox.plot import plot_weights_evolution

fig = plot_weights_evolution(combination)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `combination` | `CombinationResult` | *required* | Resultado com pesos temporais |
| `model_names` | `list[str] \| None` | `None` | Nomes dos modelos |
| `chart_type` | `str` | `"area"` | Tipo: `"area"`, `"line"`, `"bar"` |
| `highlight_model` | `str \| None` | `None` | Destacar modelo especifico |
| `date_format` | `str` | `"%Y-%m"` | Formato de datas |
| `colors` | `list[str] \| None` | `None` | Paleta customizada |
| `figsize` | `tuple[float, float]` | `(12, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Area chart (padrao)"

    ```python
    from forecastbox import combine
    from forecastbox.plot import plot_weights_evolution

    combination = combine(
        forecasts=[fc_arima, fc_ets, fc_theta],
        method="time_varying",
        actual=y_train,
        forgetting_factor=0.95,
    )

    plot_weights_evolution(
        combination,
        model_names=["ARIMA", "ETS", "Theta"],
        title="Evolucao dos Pesos - Time-Varying",
        style="publication",
    )
    ```

    **Output**: Area empilhada de Jan 2020 a Dez 2025. ARIMA domina (faixa larga)
    ate meados de 2022, quando ETS ganha peso apos mudanca estrutural. Theta
    relativamente estavel (~15-20%) ao longo de todo o periodo.

=== "Linhas individuais"

    ```python
    plot_weights_evolution(
        combination,
        model_names=["ARIMA", "ETS", "Theta"],
        chart_type="line",
        highlight_model="ETS",
        title="Evolucao dos Pesos (Linhas)",
    )
    ```

    **Output**: 3 linhas (soma nao necessariamente visivel). Linha do ETS
    mais espessa (destacada). Permite ver cruzamentos de pesos entre modelos.

---

## `plot_weights_heatmap`

Heatmap de pesos por modelo e periodo temporal. Cada celula mostra o peso
de um modelo em um periodo especifico.

**Output visual**: Matriz com modelos nas linhas, periodos nas colunas.
Gradiente de azul claro (peso baixo) a azul escuro (peso alto). Valores
numericos opcionais em cada celula.

```python
from forecastbox.plot import plot_weights_heatmap

fig = plot_weights_heatmap(combination)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `combination` | `CombinationResult` | *required* | Resultado com pesos temporais |
| `model_names` | `list[str] \| None` | `None` | Nomes dos modelos |
| `annot` | `bool` | `True` | Exibir valores nas celulas |
| `fmt` | `str` | `".2f"` | Formato numerico |
| `cmap` | `str` | `"Blues"` | Colormap |
| `date_freq` | `str \| None` | `None` | Frequencia de labels: `"Q"`, `"A"` |
| `figsize` | `tuple[float, float]` | `(14, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend: `"matplotlib"` ou `"seaborn"` |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_weights_heatmap

plot_weights_heatmap(
    combination,
    model_names=["ARIMA", "ETS", "Theta", "VAR"],
    cmap="YlOrRd",
    date_freq="Q",
    title="Heatmap de Pesos Trimestrais",
    style="publication",
)
```

**Output**: Matriz 4x24 (4 modelos x 24 trimestres). Celulas escuras indicam
peso alto naquele periodo. Padrao visual revela: ARIMA domina 2020-2021,
ETS ganha forca em 2022-2023, pesos mais dispersos em 2024-2025.

---

## `plot_combination_vs_individual`

Comparacao da previsao combinada contra os melhores modelos individuais.
Demonstra visualmente o ganho (ou nao) da combinacao.

**Output visual**: Serie observada em preto, previsao combinada em cor
destacada (linha mais grossa), modelos individuais em cores secundarias
(linhas mais finas). Legenda com RMSE de cada modelo para comparacao
quantitativa imediata.

```python
from forecastbox.plot import plot_combination_vs_individual

fig = plot_combination_vs_individual(combination)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `combination` | `CombinationResult` | *required* | Resultado de combinacao |
| `actual` | `pd.Series \| None` | `None` | Valores observados |
| `top_n` | `int` | `3` | Numero de modelos individuais a exibir |
| `show_metrics` | `bool` | `True` | Exibir RMSE na legenda |
| `metric` | `str` | `"rmse"` | Metrica para exibir e selecionar top N |
| `colors` | `dict \| None` | `None` | Cores: `{"combined": ..., "individual": [...]}` |
| `figsize` | `tuple[float, float]` | `(12, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import combine
from forecastbox.plot import plot_combination_vs_individual

combination = combine(
    forecasts=[fc_arima, fc_ets, fc_theta, fc_var, fc_ml],
    method="bma",
    actual=y_train,
)

plot_combination_vs_individual(
    combination,
    actual=y_test,
    top_n=3,
    title="BMA vs. Melhores Modelos Individuais",
    style="publication",
)
```

**Output**: Observado em preto. Combinacao BMA em vermelho (linha grossa,
RMSE=0.34). Top 3 individuais: ARIMA em azul (RMSE=0.39), ETS em verde
(RMSE=0.41), ML em laranja (RMSE=0.43). Combinacao consistentemente mais
proxima do observado.

!!! tip "Forecast combination puzzle"

    A combinacao frequentemente supera o melhor modelo individual, mesmo
    quando os pesos sao simples (media). Este fenomeno, conhecido como
    *forecast combination puzzle*, e robusto na literatura econometrica.

---

## `plot_posterior`

Distribuicao posterior das probabilidades dos modelos no Bayesian Model
Averaging (BMA). Mostra a incerteza sobre os pesos alem das estimativas
pontuais.

**Output visual**: Para cada modelo, barra com a probabilidade posterior
pontual e whiskers/intervalo representando o intervalo de credibilidade.
Opcionalmente, grafico de densidade posterior para cada modelo.

No BMA, a probabilidade posterior do modelo $k$ e:

$$
P(M_k | \mathbf{y}) = \frac{P(\mathbf{y} | M_k) \cdot P(M_k)}{\sum_{j=1}^{K} P(\mathbf{y} | M_j) \cdot P(M_j)}
$$

onde $P(\mathbf{y} | M_k)$ e a verossimilhanca marginal do modelo $k$.

```python
from forecastbox.plot import plot_posterior

fig = plot_posterior(bma_result)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `bma_result` | `CombinationResult` | *required* | Resultado BMA com posterior |
| `model_names` | `list[str] \| None` | `None` | Nomes dos modelos |
| `ci_level` | `float` | `0.95` | Nivel de credibilidade |
| `show_prior` | `bool` | `False` | Exibir prior junto ao posterior |
| `chart_type` | `str` | `"bar"` | Tipo: `"bar"`, `"density"`, `"forest"` |
| `colors` | `list[str] \| None` | `None` | Paleta |
| `figsize` | `tuple[float, float]` | `(8, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Barras com intervalos"

    ```python
    from forecastbox import combine
    from forecastbox.plot import plot_posterior

    bma = combine(
        forecasts=[fc_arima, fc_ets, fc_theta, fc_var],
        method="bma",
        actual=y_train,
    )

    plot_posterior(
        bma,
        model_names=["ARIMA", "ETS", "Theta", "VAR"],
        ci_level=0.95,
        title="Posterior Model Probabilities (BMA)",
        style="publication",
    )
    ```

    **Output**: 4 barras com whiskers. ARIMA: P=0.42 [0.35, 0.49], ETS: P=0.31
    [0.24, 0.38], Theta: P=0.18 [0.12, 0.24], VAR: P=0.09 [0.04, 0.14].

=== "Forest plot"

    ```python
    plot_posterior(
        bma,
        model_names=["ARIMA", "ETS", "Theta", "VAR"],
        chart_type="forest",
        show_prior=True,
        title="Forest Plot - BMA Posterior vs Prior",
    )
    ```

    **Output**: Linhas horizontais (forest plot) com ponto central (posterior)
    e intervalo de credibilidade. Prior uniforme exibido como linha vertical
    tracejada em P=0.25. Modelos com posterior > prior recebem destaque.

=== "Densidade posterior"

    ```python
    plot_posterior(
        bma,
        model_names=["ARIMA", "ETS", "Theta", "VAR"],
        chart_type="density",
        title="Densidades Posteriores (BMA)",
    )
    ```

    **Output**: 4 curvas de densidade (uma por modelo) sobrepostas. ARIMA com
    pico alto e estreito em 0.42, VAR com pico baixo e largo em 0.09
    (maior incerteza sobre o peso).

---

## `plot_weight_stability`

Boxplot da estabilidade dos pesos ao longo do tempo. Modelos com boxplots
compactos tem pesos estaveis; boxplots largos indicam instabilidade.

**Output visual**: Boxplot para cada modelo mostrando a distribuicao dos
pesos ao longo das janelas de estimacao. Mediana, quartis e outliers.
Linha horizontal tracejada em $1/M$ (peso uniforme) como referencia.

```python
from forecastbox.plot import plot_weight_stability

fig = plot_weight_stability(combination)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `combination` | `CombinationResult` | *required* | Resultado com pesos temporais |
| `model_names` | `list[str] \| None` | `None` | Nomes dos modelos |
| `show_uniform` | `bool` | `True` | Exibir linha de peso uniforme |
| `show_mean` | `bool` | `True` | Exibir marcador da media |
| `sort_by` | `str` | `"median"` | Ordenar: `"median"`, `"stability"`, `"name"` |
| `colors` | `list[str] \| None` | `None` | Paleta |
| `figsize` | `tuple[float, float]` | `(8, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_weight_stability

plot_weight_stability(
    combination,
    model_names=["ARIMA", "ETS", "Theta", "VAR"],
    sort_by="stability",
    title="Estabilidade dos Pesos - Time-Varying",
    style="publication",
)
```

**Output**: 4 boxplots ordenados por IQR (mais estavel primeiro). Theta com
boxplot compacto (peso estavel em ~0.18), ARIMA com boxplot largo (peso varia
de 0.25 a 0.55). Linha tracejada horizontal em 0.25 (peso uniforme).

!!! warning "Instabilidade de pesos"

    Pesos muito instaveis podem indicar overfitting na combinacao ou
    instabilidade estrutural nos modelos subjacentes. Considere usar
    metodos com regularizacao (shrinkage para media simples) ou impor
    restricoes nos pesos.

---

## `plot_weight_contribution`

Grafico de barras empilhadas mostrando a contribuicao de cada modelo para
a previsao combinada em cada periodo.

**Output visual**: Barras empilhadas (uma por periodo) onde a altura total
e a previsao combinada e cada segmento colorido representa a contribuicao
$w_i \cdot \hat{y}_{i,t}$ do modelo $i$.

A previsao combinada e:

$$
\hat{y}_{c,t} = \sum_{i=1}^{M} w_{i,t} \cdot \hat{y}_{i,t}
$$

```python
from forecastbox.plot import plot_weight_contribution

fig = plot_weight_contribution(combination)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `combination` | `CombinationResult` | *required* | Resultado de combinacao |
| `model_names` | `list[str] \| None` | `None` | Nomes dos modelos |
| `periods` | `int \| None` | `None` | Numero de periodos a exibir |
| `colors` | `list[str] \| None` | `None` | Paleta |
| `figsize` | `tuple[float, float]` | `(12, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_weight_contribution

plot_weight_contribution(
    combination,
    model_names=["ARIMA", "ETS", "Theta", "VAR"],
    periods=12,
    title="Contribuicao por Modelo na Previsao Combinada",
    style="publication",
)
```

**Output**: 12 barras empilhadas (uma por mes de previsao). ARIMA (azul)
contribui ~42% da altura em cada barra, ETS (laranja) ~31%, Theta (verde)
~18%, VAR (vermelho) ~9%. Proporcoes variam se pesos sao time-varying.

---

## Exemplos Completos

### Dashboard de combinacao BMA

```python
import matplotlib.pyplot as plt
from forecastbox import combine
from forecastbox.plot import (
    plot_weights,
    plot_posterior,
    plot_combination_vs_individual,
    plot_weight_stability,
    set_theme,
)

set_theme("publication")

# Combinacao BMA
bma = combine(
    forecasts=[fc_arima, fc_ets, fc_theta, fc_var, fc_ml],
    method="bma",
    actual=y_train,
)
names = ["ARIMA", "ETS", "Theta", "VAR", "ML"]

# Dashboard 2x2
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_weights(bma, model_names=names, ax=axes[0, 0], show=False,
            title="Pesos BMA")
plot_posterior(bma, model_names=names, ax=axes[0, 1], show=False,
             title="Posterior Probabilities")
plot_combination_vs_individual(bma, actual=y_test, ax=axes[1, 0], show=False,
                              title="BMA vs. Individuais")
plot_weight_stability(bma, model_names=names, ax=axes[1, 1], show=False,
                     title="Estabilidade dos Pesos")

fig.suptitle("Dashboard BMA - IPCA", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("bma_dashboard.pdf", dpi=300)
plt.show()
```

### Evolucao time-varying com Plotly

```python
from forecastbox import combine
from forecastbox.plot import plot_weights_evolution, plot_weights_heatmap

# Combinacao time-varying
tv = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="time_varying",
    actual=y_train,
    forgetting_factor=0.95,
)
names = ["ARIMA", "ETS", "Theta"]

# Area chart interativo
fig = plot_weights_evolution(
    tv,
    model_names=names,
    backend="plotly",
    title="Evolucao dos Pesos - Time-Varying (Interativo)",
)
fig.write_html("weights_evolution.html")

# Heatmap
plot_weights_heatmap(
    tv,
    model_names=names,
    date_freq="Q",
    title="Heatmap de Pesos Trimestrais",
    style="publication",
)
```

### Comparacao de metodos de combinacao

```python
from forecastbox import combine
from forecastbox.plot import plot_weights, plot_combination_vs_individual
import matplotlib.pyplot as plt

methods = {
    "Media Simples": "simple_mean",
    "BMA": "bma",
    "OLS": "ols",
    "Time-Varying": "time_varying",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
names = ["ARIMA", "ETS", "Theta", "VAR"]

for ax, (label, method) in zip(axes.flat, methods.items()):
    comb = combine(
        forecasts=[fc_arima, fc_ets, fc_theta, fc_var],
        method=method,
        actual=y_train,
    )
    plot_weights(comb, model_names=names, ax=ax, show=False, title=label)

plt.suptitle("Comparacao de Metodos de Combinacao", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## See Also

- :material-school: [Tutorial: Combinacao de Previsoes](../tutorials/combination.md) — aprenda a combinar modelos e diagnosticar pesos
- [Graficos de Previsao](forecast-plots.md) — visualizacao de previsoes individuais
- [Graficos de Comparacao](comparison-plots.md) — comparacao de modelos
- [User Guide - Combinacao](../user-guide/combination/index.md) — metodos de combinacao
- [Theory - Combinacao](../theory/combination-theory.md) — teoria de combinacao de previsoes
- [API Reference - Visualization](../api/visualization.md) — referencia completa
