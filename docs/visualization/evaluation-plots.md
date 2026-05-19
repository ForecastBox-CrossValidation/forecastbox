---
title: Graficos de Avaliacao
description: Visualizacao de testes estatisticos, diagnosticos de previsao, calibracao probabilistica e Model Confidence Set
---

# Graficos de Avaliacao

Funcoes para visualizar resultados de avaliacao de previsoes: testes de
habilidade preditiva (Diebold-Mariano), Model Confidence Set, validacao cruzada,
regressao Mincer-Zarnowitz, encompassing e calibracao probabilistica.

---

## `plot_dm_test`

Grafico de diferenciais de perda do teste Diebold-Mariano com bandas de confianca.
Permite avaliar visualmente se a diferenca de performance entre dois modelos e
estatisticamente significativa.

**Output visual**: Serie temporal dos diferenciais de perda $d_t = L(e_{1,t}) - L(e_{2,t})$
com uma linha horizontal em zero. Bandas de confianca sombreadas ao redor da media.
Anotacao com estatistica DM e p-valor.

A estatistica Diebold-Mariano e definida como:

$$
DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \xrightarrow{d} N(0, 1)
$$

onde $\bar{d} = T^{-1} \sum_{t=1}^{T} d_t$ e $\hat{V}(\bar{d})$ e um estimador
HAC da variancia.

```python
from forecastbox.plot import plot_dm_test

fig = plot_dm_test(dm_result)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `dm_result` | `DMTestResult` | *required* | Resultado do teste Diebold-Mariano |
| `loss_diff` | `bool` | `True` | Exibir serie de diferenciais de perda |
| `ci_level` | `float` | `0.95` | Nivel de confianca para as bandas |
| `significance_level` | `float` | `0.05` | Nivel de significancia para destacar |
| `annotate` | `bool` | `True` | Anotar estatistica DM e p-valor |
| `colors` | `dict \| None` | `None` | Cores: `{"positive", "negative", "ci"}` |
| `title` | `str \| None` | `None` | Titulo (auto-gerado se `None`) |
| `figsize` | `tuple[float, float]` | `(10, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend: `"matplotlib"`, `"plotly"`, `"seaborn"` |
| `style` | `str` | `"light"` | Tema: `"light"`, `"dark"`, `"publication"`, `"presentation"` |
| `ax` | `Axes \| None` | `None` | Eixo matplotlib existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Teste basico"

    ```python
    from forecastbox import evaluate
    from forecastbox.plot import plot_dm_test

    dm_result = evaluate.dm_test(
        actual=y_test,
        forecast_1=fc_arima,
        forecast_2=fc_ets,
        loss="mse",
    )

    plot_dm_test(dm_result)
    ```

    **Output**: Serie temporal de $d_t$ em azul, banda de 95% sombreada. Linha
    horizontal em zero (tracejada). Anotacao "DM = 2.34, p = 0.019" no canto
    superior. Regioes onde $d_t > 0$ indicam que o modelo 2 (ETS) foi melhor.

=== "Publicacao"

    ```python
    plot_dm_test(
        dm_result,
        ci_level=0.99,
        significance_level=0.01,
        title="Diebold-Mariano: ARIMA vs ETS",
        style="publication",
        figsize=(8, 4),
    )
    ```

    **Output**: Formatacao academica com fonte serif, eixos limpos. Bandas de
    99% de confianca. P-valor anotado com destaque se significativo a 1%.

=== "Interativo"

    ```python
    plot_dm_test(
        dm_result,
        backend="plotly",
        annotate=True,
    )
    ```

    **Output**: Versao interativa com hover mostrando data e valor do diferencial.
    Zoom e pan habilitados.

---

## `plot_mcs_inclusion`

Heatmap de inclusao no Model Confidence Set (MCS) por nivel de significancia.
Mostra quais modelos pertencem ao conjunto de confianca para diferentes valores
de $\alpha$.

**Output visual**: Heatmap onde linhas sao modelos e colunas sao valores de $\alpha$.
Celulas coloridas indicam inclusao (verde) e exclusao (vermelho). Modelos
ordenados pelo p-valor de eliminacao.

O MCS e definido como o menor conjunto $\hat{\mathcal{M}}^*_{1-\alpha}$ tal que:

$$
\hat{\mathcal{M}}^*_{1-\alpha} = \{ i : \hat{p}_i > \alpha \}
$$

onde $\hat{p}_i$ e o p-valor de eliminacao do modelo $i$.

```python
from forecastbox.plot import plot_mcs_inclusion

fig = plot_mcs_inclusion(mcs_result)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `mcs_result` | `MCSResult` | *required* | Resultado do Model Confidence Set |
| `alpha_range` | `list[float]` | `[0.01, 0.05, 0.10, 0.15, 0.20, 0.25]` | Niveis de $\alpha$ para exibir |
| `model_names` | `list[str] \| None` | `None` | Nomes dos modelos |
| `sort_by` | `str` | `"pvalue"` | Ordenacao: `"pvalue"`, `"name"`, `"rank"` |
| `annotate` | `bool` | `True` | Exibir p-valores nas celulas |
| `cmap` | `str` | `"RdYlGn"` | Colormap (verde=incluso, vermelho=excluido) |
| `significance_level` | `float` | `0.05` | Nivel destacado com borda |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "MCS basico"

    ```python
    from forecastbox import evaluate
    from forecastbox.plot import plot_mcs_inclusion

    mcs_result = evaluate.mcs(
        actual=y_test,
        forecasts={"ARIMA": fc_arima, "ETS": fc_ets, "Theta": fc_theta,
                    "VAR": fc_var, "BVAR": fc_bvar},
        loss="mse",
        alpha=0.25,
    )

    plot_mcs_inclusion(mcs_result)
    ```

    **Output**: Heatmap 5x6. Linhas: ARIMA, ETS, Theta, VAR, BVAR (ordenadas
    por p-valor). Colunas: $\alpha$ de 0.01 a 0.25. Celulas verdes para modelos
    inclusos, vermelhas para excluidos. P-valores anotados em cada celula.

=== "Customizado"

    ```python
    plot_mcs_inclusion(
        mcs_result,
        alpha_range=[0.05, 0.10, 0.20],
        sort_by="rank",
        significance_level=0.10,
        style="publication",
    )
    ```

    **Output**: Heatmap compacto com 3 colunas de $\alpha$. Coluna de $\alpha=0.10$
    destacada com borda. Modelos ordenados pelo ranking MCS.

---

## `plot_cv_results`

Performance por fold de validacao cruzada (expanding window ou rolling window).
Essencial para avaliar estabilidade do modelo ao longo do tempo.

**Output visual**: Grafico de linhas ou barras com a metrica de performance
em cada fold. Linha horizontal com a media. Faixas sombreadas para desvio padrao.

```python
from forecastbox.plot import plot_cv_results

fig = plot_cv_results(cv_result)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `cv_result` | `CVResult` | *required* | Resultado de validacao cruzada |
| `metric` | `str` | `"rmse"` | Metrica: `"rmse"`, `"mae"`, `"mape"`, `"mase"` |
| `chart_type` | `str` | `"line"` | Tipo: `"line"`, `"bar"`, `"box"` |
| `show_mean` | `bool` | `True` | Exibir linha de media |
| `show_std` | `bool` | `True` | Exibir faixa de desvio padrao |
| `show_train_size` | `bool` | `False` | Exibir tamanho da janela de treino |
| `annotate` | `bool` | `False` | Anotar valores em cada fold |
| `colors` | `dict \| None` | `None` | Cores: `{"metric", "mean", "std"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(10, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Expanding window"

    ```python
    from forecastbox import AutoForecast, evaluate
    from forecastbox.plot import plot_cv_results

    model = AutoForecast(strategy="best")
    cv_result = evaluate.time_series_cv(
        model=model, y=y, cv="expanding", n_splits=10, horizon=6,
    )

    plot_cv_results(cv_result, metric="rmse")
    ```

    **Output**: 10 pontos conectados por linha, cada um representando o RMSE de
    um fold. Linha tracejada horizontal na media. Faixa cinza de $\pm 1\sigma$.
    Eixo x mostra as datas de inicio de cada fold.

=== "Rolling window com barras"

    ```python
    plot_cv_results(
        cv_result,
        metric="mase",
        chart_type="bar",
        show_train_size=True,
        annotate=True,
        title="MASE por fold - Rolling Window",
    )
    ```

    **Output**: Barras verticais com MASE de cada fold. Valores anotados acima
    de cada barra. Eixo secundario mostra tamanho da janela de treino.

=== "Multiplas metricas"

    ```python
    from forecastbox.plot import plot_cv_results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric in zip(axes, ["rmse", "mae", "mape"]):
        plot_cv_results(cv_result, metric=metric, ax=ax, show=False)

    plt.tight_layout()
    plt.show()
    ```

    **Output**: Painel 1x3 com RMSE, MAE e MAPE por fold lado a lado.

---

## `plot_mincer_zarnowitz`

Scatter plot actual vs. forecast com regressao Mincer-Zarnowitz.
Avalia a eficiencia das previsoes testando se $\alpha = 0$ e $\beta = 1$.

**Output visual**: Scatter de pontos (forecast no eixo x, actual no eixo y),
linha de regressao estimada, e linha de 45 graus (previsao perfeita).
Anotacao com coeficientes e teste de hipotese conjunta.

A regressao Mincer-Zarnowitz e:

$$
y_t = \alpha + \beta \hat{y}_t + \varepsilon_t
$$

Previsoes eficientes satisfazem $H_0: \alpha = 0, \beta = 1$.

```python
from forecastbox.plot import plot_mincer_zarnowitz

fig = plot_mincer_zarnowitz(mz_result)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `mz_result` | `MZResult` | *required* | Resultado da regressao Mincer-Zarnowitz |
| `show_45_line` | `bool` | `True` | Exibir linha de 45 graus (previsao perfeita) |
| `show_regression` | `bool` | `True` | Exibir linha de regressao estimada |
| `show_ci` | `bool` | `True` | Exibir intervalo de confianca da regressao |
| `annotate` | `bool` | `True` | Anotar $\hat{\alpha}$, $\hat{\beta}$, R², p-valor |
| `significance_level` | `float` | `0.05` | Nivel de significancia |
| `point_size` | `float` | `20` | Tamanho dos pontos do scatter |
| `colors` | `dict \| None` | `None` | Cores: `{"points", "regression", "45_line", "ci"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(7, 7)` | Tamanho da figura (quadrado) |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "MZ basico"

    ```python
    from forecastbox import evaluate
    from forecastbox.plot import plot_mincer_zarnowitz

    mz_result = evaluate.mincer_zarnowitz(
        actual=y_test,
        forecast=fc_arima,
    )

    plot_mincer_zarnowitz(mz_result)
    ```

    **Output**: Scatter com pontos em azul, linha de regressao em vermelho
    (solida), linha de 45 graus em cinza (tracejada). Anotacao:
    "$\hat{\alpha} = 0.12$, $\hat{\beta} = 0.94$, R² = 0.87, p(F) = 0.23".
    Nao rejeita $H_0$: previsoes eficientes.

=== "Com intervalo de confianca"

    ```python
    plot_mincer_zarnowitz(
        mz_result,
        show_ci=True,
        significance_level=0.01,
        style="publication",
        title="Mincer-Zarnowitz: IPCA",
    )
    ```

    **Output**: Formatacao academica. Banda de confianca de 99% ao redor da
    regressao. Destaque visual se $H_0$ e rejeitada a 1%.

---

## `plot_encompassing`

Heatmap de resultados de testes de encompassing entre pares de modelos.
Identifica se um modelo engloba a informacao de outro.

**Output visual**: Heatmap quadrado (modelos x modelos). Celula $(i, j)$ exibe
o p-valor do teste de encompassing de $i$ sobre $j$. Diagonal em cinza.
Cores indicam rejeicao (vermelho) ou nao-rejeicao (verde).

O teste de forecast encompassing avalia:

$$
y_t = \lambda \hat{y}_{1,t} + (1 - \lambda) \hat{y}_{2,t} + \varepsilon_t
$$

Se $\lambda = 1$, o modelo 1 engloba o modelo 2 (informacao de 2 e redundante).

```python
from forecastbox.plot import plot_encompassing

fig = plot_encompassing(enc_results)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `enc_results` | `EncompassingResult` | *required* | Resultado de testes de encompassing |
| `model_names` | `list[str] \| None` | `None` | Nomes dos modelos |
| `significance_level` | `float` | `0.05` | Nivel de significancia |
| `annotate` | `bool` | `True` | Exibir p-valores nas celulas |
| `cmap` | `str` | `"RdYlGn"` | Colormap |
| `mask_diagonal` | `bool` | `True` | Mascarar diagonal |
| `figsize` | `tuple[float, float]` | `(8, 8)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox import evaluate
from forecastbox.plot import plot_encompassing

enc_results = evaluate.encompassing_test(
    actual=y_test,
    forecasts={"ARIMA": fc_arima, "ETS": fc_ets, "Theta": fc_theta},
)

plot_encompassing(
    enc_results,
    model_names=["ARIMA", "ETS", "Theta"],
    significance_level=0.10,
    style="publication",
)
```

**Output**: Heatmap 3x3. Diagonal cinza. Celula (ARIMA, ETS) com p=0.03
em vermelho (ARIMA engloba ETS). Celula (ETS, ARIMA) com p=0.41 em verde
(ETS nao engloba ARIMA). Valores anotados em cada celula.

---

## `plot_calibration`

Histograma PIT (Probability Integral Transform) para avaliacao de calibracao
probabilistica. Previsoes bem calibradas produzem um histograma PIT uniforme.

**Output visual**: Histograma dos valores PIT com linha de referencia uniforme
(horizontal). Barras acima da referencia indicam excesso de massa; abaixo indicam
deficit. Teste de uniformidade (Kolmogorov-Smirnov) anotado.

Os valores PIT sao definidos como:

$$
u_t = \hat{F}_t(y_t)
$$

onde $\hat{F}_t$ e a CDF da distribuicao preditiva. Se o modelo e bem
calibrado, $u_t \sim U(0, 1)$.

```python
from forecastbox.plot import plot_calibration

fig = plot_calibration(forecast)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | *required* | Previsao probabilistica com CDF preditiva |
| `n_bins` | `int` | `10` | Numero de bins do histograma |
| `show_uniform` | `bool` | `True` | Exibir linha de referencia uniforme |
| `show_ks_test` | `bool` | `True` | Exibir resultado do teste KS |
| `annotate` | `bool` | `True` | Anotar estatistica KS e p-valor |
| `significance_level` | `float` | `0.05` | Nivel de significancia do teste KS |
| `colors` | `dict \| None` | `None` | Cores: `{"bars", "uniform", "reject"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(8, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "PIT basico"

    ```python
    from forecastbox import AutoForecast
    from forecastbox.plot import plot_calibration

    model = AutoForecast(strategy="best", probabilistic=True)
    forecast = model.fit_predict(y, horizon=12)

    plot_calibration(forecast)
    ```

    **Output**: Histograma com 10 barras em azul. Linha horizontal tracejada
    em $1/10 = 0.10$ (referencia uniforme). Se bem calibrado, barras proximas
    da linha de referencia. Anotacao "KS = 0.08, p = 0.72".

=== "Diagnostico detalhado"

    ```python
    plot_calibration(
        forecast,
        n_bins=20,
        significance_level=0.01,
        title="PIT Histogram - IPCA",
        style="publication",
    )
    ```

    **Output**: Histograma com 20 bins para maior resolucao. Formato de
    publicacao. Teste KS a 1%.

!!! tip "Interpretacao do PIT"

    - **Histograma uniforme**: modelo bem calibrado
    - **Formato U**: caudas muito estreitas (subestima incerteza)
    - **Formato ∩ (invertido U)**: caudas muito largas (superestima incerteza)
    - **Inclinado para direita**: previsoes com vies negativo (subestima valores)
    - **Inclinado para esquerda**: previsoes com vies positivo (superestima valores)

---

## `plot_reliability`

Reliability diagram (diagrama de confiabilidade) para avaliacao de calibracao
de quantis. Compara frequencias empiricas com niveis de confianca nominais.

**Output visual**: Scatter + linha conectando frequencias empiricas (eixo y)
vs. niveis nominais (eixo x). Linha de 45 graus como referencia (calibracao
perfeita). Area sombreada indicando desvio da calibracao.

Para cada nivel de confianca nominal $\tau$, a frequencia empirica e:

$$
\hat{p}(\tau) = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\{y_t \leq \hat{q}_{\tau,t}\}
$$

Calibracao perfeita implica $\hat{p}(\tau) = \tau$ para todo $\tau$.

```python
from forecastbox.plot import plot_reliability

fig = plot_reliability(forecast)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | *required* | Previsao probabilistica com quantis |
| `quantiles` | `list[float] \| None` | `None` | Quantis a avaliar (default: 0.05 a 0.95) |
| `show_45_line` | `bool` | `True` | Exibir linha de calibracao perfeita |
| `show_ci` | `bool` | `True` | Exibir intervalo de confianca binomial |
| `annotate` | `bool` | `False` | Anotar frequencias empiricas |
| `fill_deviation` | `bool` | `True` | Sombrear area de desvio da calibracao |
| `significance_level` | `float` | `0.05` | Nivel para IC binomial |
| `colors` | `dict \| None` | `None` | Cores: `{"empirical", "reference", "ci", "deviation"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(7, 7)` | Tamanho da figura (quadrado) |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Reliability basico"

    ```python
    from forecastbox.plot import plot_reliability

    plot_reliability(forecast)
    ```

    **Output**: 19 pontos (quantis de 0.05 a 0.95 em passos de 0.05) conectados
    por linha azul. Linha de 45 graus tracejada em cinza. Pontos acima da diagonal
    indicam excesso de cobertura; abaixo, deficit.

=== "Com intervalo de confianca"

    ```python
    plot_reliability(
        forecast,
        quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        show_ci=True,
        fill_deviation=True,
        title="Reliability Diagram - IPCA",
        style="publication",
    )
    ```

    **Output**: 7 pontos com bandas de confianca binomiais. Area entre a curva
    empirica e a diagonal sombreada (azul translucido se sub-cobertura, vermelho
    se sobre-cobertura).

!!! info "Calibracao vs. Sharpness"

    O reliability diagram avalia **calibracao** (as probabilidades estao corretas?),
    mas nao **sharpness** (os intervalos sao estreitos?). Um modelo pode ser
    perfeitamente calibrado mas ter intervalos muito largos. Para avaliar sharpness,
    combine com analise de largura dos intervalos via `plot_forecast`.

---

## Painel de Avaliacao Completa

Para uma avaliacao abrangente, combine multiplos graficos em um painel:

```python
import matplotlib.pyplot as plt
from forecastbox.plot import (
    plot_dm_test,
    plot_mcs_inclusion,
    plot_mincer_zarnowitz,
    plot_calibration,
    plot_reliability,
    plot_cv_results,
)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

plot_dm_test(dm_result, ax=axes[0, 0], show=False)
plot_mcs_inclusion(mcs_result, ax=axes[0, 1], show=False)
plot_cv_results(cv_result, ax=axes[0, 2], show=False)
plot_mincer_zarnowitz(mz_result, ax=axes[1, 0], show=False)
plot_calibration(forecast, ax=axes[1, 1], show=False)
plot_reliability(forecast, ax=axes[1, 2], show=False)

fig.suptitle("Avaliacao Completa - ARIMA vs ETS", fontsize=14)
plt.tight_layout()
plt.savefig("evaluation_panel.png", dpi=300, bbox_inches="tight")
plt.show()
```

**Output**: Painel 2x3 com todas as perspectivas de avaliacao: habilidade
preditiva relativa (DM), ranking (MCS), estabilidade temporal (CV), eficiencia
(MZ), calibracao distribucional (PIT) e calibracao de quantis (reliability).

!!! tip "Boas praticas de avaliacao"

    - Use **DM test** para comparacao pareada (2 modelos)
    - Use **MCS** para selecao entre multiplos modelos
    - Use **CV** para avaliar estabilidade e overfitting
    - Use **Mincer-Zarnowitz** para testar eficiencia das previsoes
    - Use **PIT** e **Reliability** para avaliar previsoes probabilisticas
    - Combine **encompassing** para entender complementaridade entre modelos

---

## See Also

- :material-school: [Tutorial: Avaliacao Rigorosa](../tutorials/evaluation.md) — aprenda DM, MCS e Mincer-Zarnowitz na pratica
- [Graficos de Previsao](forecast-plots.md) — visualizacao de previsoes individuais
- [Graficos de Comparacao](comparison-plots.md) — comparacao visual de modelos
- [User Guide - Avaliacao](../user-guide/evaluation/index.md) — referencia completa de testes
- [Theory - Evaluation](../theory/evaluation-theory.md) — fundamentos teoricos
- [API Reference - Visualization](../api/visualization.md) — referencia completa da API
