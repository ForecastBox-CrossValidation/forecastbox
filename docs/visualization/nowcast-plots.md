---
title: Graficos de Nowcasting
description: Visualizacao de nowcasts, news decomposition, factor loadings, pesos MIDAS e ragged edge
---

# Graficos de Nowcasting

Funcoes para visualizar resultados de nowcasting: evolucao do nowcast ao longo
do trimestre, decomposicao de news por indicador, loadings dos fatores, pesos
MIDAS, visualizacao de dados disponiveis e comparacao de vintages.

---

## `plot_nowcast_evolution`

Evolucao do nowcast ao longo do trimestre (quarter). Mostra como a estimativa
converge para o valor realizado a medida que mais informacao fica disponivel.

**Output visual**: Linha conectando as estimativas do nowcast em diferentes
datas de referencia dentro do trimestre. Faixa de confianca ao redor de cada
estimativa. Linha horizontal tracejada com o valor realizado (quando disponivel).
Eixo x mostra as datas de referencia (vintage dates).

```python
from forecastbox.plot import plot_nowcast_evolution

fig = plot_nowcast_evolution(nowcasts)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `nowcasts` | `list[NowcastResult]` | *required* | Lista de nowcasts em diferentes vintage dates |
| `actual` | `float \| None` | `None` | Valor realizado (se disponivel) |
| `target_variable` | `str \| None` | `None` | Nome da variavel alvo (para titulo) |
| `show_ci` | `bool` | `True` | Exibir intervalos de confianca |
| `ci_level` | `float` | `0.95` | Nivel de confianca |
| `vintage_dates` | `list[str] \| None` | `None` | Labels das datas de referencia |
| `highlight_news` | `bool` | `False` | Destacar datas com news significativas |
| `colors` | `dict \| None` | `None` | Cores: `{"nowcast", "actual", "ci"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(12, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Evolucao do PIB"

    ```python
    from forecastbox.nowcast import DynamicFactorNowcast
    from forecastbox.plot import plot_nowcast_evolution

    dfm = DynamicFactorNowcast(n_factors=3)

    # Gerar nowcasts em diferentes datas
    nowcasts = []
    for date in vintage_dates:
        nc = dfm.nowcast(data, reference_date=date)
        nowcasts.append(nc)

    plot_nowcast_evolution(
        nowcasts,
        actual=pib_realizado,
        target_variable="PIB",
    )
    ```

    **Output**: Linha descendente/convergente do nowcast ao longo do trimestre.
    Primeira estimativa (2 meses antes) com intervalo largo, ultima estimativa
    (1 mes depois) com intervalo estreito. Linha horizontal tracejada no valor
    realizado do PIB. Convergencia visivel a medida que mais dados chegam.

=== "Com destaque de news"

    ```python
    plot_nowcast_evolution(
        nowcasts,
        actual=pib_realizado,
        highlight_news=True,
        style="publication",
        title="Evolucao do Nowcast - PIB T3/2024",
    )
    ```

    **Output**: Mesma evolucao com marcadores maiores nas datas onde houve
    news significativas (grandes revisoes). Formato de publicacao.

=== "Interativo"

    ```python
    plot_nowcast_evolution(
        nowcasts,
        actual=pib_realizado,
        backend="plotly",
    )
    ```

    **Output**: Versao interativa com hover mostrando vintage date, valor do
    nowcast e intervalo de confianca.

---

## `plot_news_waterfall`

Waterfall chart mostrando a contribuicao de cada indicador para a revisao
do nowcast entre duas vintage dates. Essencial para entender o que moveu
a estimativa.

**Output visual**: Barras horizontais cascateadas (waterfall). Ponto de
partida e o nowcast anterior. Barras verdes (positivas) e vermelhas (negativas)
mostram a contribuicao de cada indicador. Ponto de chegada e o nowcast
atualizado.

A decomposicao de news e definida como:

$$
\hat{y}_{t|v_2} - \hat{y}_{t|v_1} = \sum_{i=1}^{N} w_i \cdot \underbrace{(x_{i,v_2} - E_{v_1}[x_{i,v_2}])}_{\text{news}_i}
$$

onde $w_i$ e o peso do indicador $i$ e $\text{news}_i$ e a surpresa (diferenca
entre o dado observado e a expectativa do modelo).

```python
from forecastbox.plot import plot_news_waterfall

fig = plot_news_waterfall(news)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `news` | `NewsDecomposition` | *required* | Resultado da decomposicao de news |
| `top_n` | `int \| None` | `None` | Limitar aos N maiores contribuidores |
| `sort_by` | `str` | `"absolute"` | Ordenacao: `"absolute"`, `"positive_first"`, `"original"` |
| `show_total` | `bool` | `True` | Exibir barra de total ao final |
| `annotate` | `bool` | `True` | Anotar valores em cada barra |
| `colors` | `dict \| None` | `None` | Cores: `{"positive", "negative", "total", "connector"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(10, 7)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Waterfall basico"

    ```python
    from forecastbox.nowcast import DynamicFactorNowcast
    from forecastbox.plot import plot_news_waterfall

    dfm = DynamicFactorNowcast(n_factors=3)

    news = dfm.news_decomposition(
        data,
        vintage_old="2024-07-15",
        vintage_new="2024-08-15",
    )

    plot_news_waterfall(news)
    ```

    **Output**: Barra inicial "Nowcast Jul/2024 = 0.8%". Barras cascateadas:
    Producao Industrial (+0.15%, verde), Vendas Varejo (-0.05%, vermelho),
    PMI (+0.08%, verde), etc. Barra final "Nowcast Ago/2024 = 1.0%".

=== "Top 5 contribuidores"

    ```python
    plot_news_waterfall(
        news,
        top_n=5,
        sort_by="absolute",
        title="News Decomposition - PIB T3/2024",
        style="publication",
    )
    ```

    **Output**: Apenas os 5 indicadores com maior impacto absoluto. Barra
    "Outros" agrupando os demais. Formato de publicacao.

!!! info "Interpretacao do waterfall"

    - **Barras verdes**: surpresas positivas (dados melhores que esperado pelo modelo)
    - **Barras vermelhas**: surpresas negativas (dados piores que esperado)
    - **Indicadores sem barra**: nao tiveram dados novos entre as vintages
    - A soma das contribuicoes e igual a revisao total do nowcast

---

## `plot_factor_loadings`

Heatmap dos loadings dos fatores de um modelo de fatores dinamicos (DFM).
Mostra a relacao entre as variaveis observadas e os fatores latentes.

**Output visual**: Heatmap onde linhas sao variaveis observadas e colunas sao
fatores. Cores indicam magnitude e sinal dos loadings. Variaveis agrupadas
por setor ou tipo.

Os loadings $\Lambda$ conectam os fatores latentes $f_t$ as variaveis observadas:

$$
x_t = \Lambda f_t + e_t
$$

```python
from forecastbox.plot import plot_factor_loadings

fig = plot_factor_loadings(dfm)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `dfm` | `DynamicFactorNowcast` | *required* | Modelo DFM estimado |
| `variable_names` | `list[str] \| None` | `None` | Nomes das variaveis |
| `factor_names` | `list[str] \| None` | `None` | Nomes dos fatores |
| `sort_by` | `str \| None` | `None` | Ordenar variaveis por loading do fator especificado |
| `cluster` | `bool` | `False` | Agrupar variaveis por similaridade de loadings |
| `annotate` | `bool` | `True` | Exibir valores nas celulas |
| `cmap` | `str` | `"RdBu_r"` | Colormap divergente (azul=negativo, vermelho=positivo) |
| `figsize` | `tuple[float, float]` | `(8, 10)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Loadings basico"

    ```python
    from forecastbox.nowcast import DynamicFactorNowcast
    from forecastbox.plot import plot_factor_loadings

    dfm = DynamicFactorNowcast(n_factors=3)
    dfm.fit(data)

    plot_factor_loadings(dfm)
    ```

    **Output**: Heatmap com 30+ variaveis nas linhas e 3 fatores nas colunas.
    Producao Industrial com loading alto no Fator 1 (atividade real). Juros e
    credito com loading alto no Fator 2 (financeiro). Precos com loading no
    Fator 3 (inflacao).

=== "Agrupado por setor"

    ```python
    plot_factor_loadings(
        dfm,
        cluster=True,
        sort_by="Factor 1",
        title="Factor Loadings - DFM PIB",
        style="publication",
    )
    ```

    **Output**: Variaveis agrupadas por similaridade de loadings. Setores
    emergem naturalmente: variaveis de atividade juntas, financeiras juntas.

---

## `plot_midas_weights`

Visualizacao da funcao de pesos MIDAS (Mixed Data Sampling) que mapeia
dados de alta frequencia para baixa frequencia.

**Output visual**: Curva da funcao de pesos ao longo dos lags de alta
frequencia. Pesos maiores no inicio indicam que dados recentes sao mais
informativos. Tipo de funcao (Beta, Almon) indicado no titulo.

Os pesos MIDAS sao parametrizados por uma funcao de pesos $w(k; \theta)$:

=== "Beta (2 parametros)"

    $$
    w(k; \theta_1, \theta_2) = \frac{k^{\theta_1 - 1}(1-k)^{\theta_2 - 1}}{\sum_{j} j^{\theta_1 - 1}(1-j)^{\theta_2 - 1}}
    $$

=== "Almon exponencial"

    $$
    w(k; \theta_1, \theta_2) = \frac{\exp(\theta_1 k + \theta_2 k^2)}{\sum_{j} \exp(\theta_1 j + \theta_2 j^2)}
    $$

```python
from forecastbox.plot import plot_midas_weights

fig = plot_midas_weights(midas)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `midas` | `MIDASNowcast` | *required* | Modelo MIDAS estimado |
| `variable` | `str \| None` | `None` | Variavel especifica (se `None`, todas) |
| `weight_type` | `str \| None` | `None` | Tipo de peso para exibir (auto-detectado) |
| `show_cumulative` | `bool` | `False` | Exibir pesos cumulativos |
| `normalize` | `bool` | `True` | Normalizar pesos (soma = 1) |
| `colors` | `list[str] \| None` | `None` | Cores por variavel |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(10, 5)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Pesos Beta"

    ```python
    from forecastbox.nowcast import MIDASNowcast
    from forecastbox.plot import plot_midas_weights

    midas = MIDASNowcast(weight_function="beta")
    midas.fit(y_quarterly, X_monthly)

    plot_midas_weights(midas)
    ```

    **Output**: Curva decrescente dos pesos Beta ao longo de 3 meses (9 lags
    semanais ou 66 lags diarios). Peso maximo no lag mais recente, decaindo
    suavemente. Soma dos pesos = 1.

=== "Almon com cumulativo"

    ```python
    plot_midas_weights(
        midas,
        show_cumulative=True,
        title="Pesos MIDAS - Producao Industrial",
        style="publication",
    )
    ```

    **Output**: Dois eixos: pesos individuais (barras) no eixo esquerdo,
    pesos cumulativos (linha) no eixo direito. Permite ver quanto da
    informacao esta concentrada nos lags mais recentes.

=== "Multiplas variaveis"

    ```python
    plot_midas_weights(
        midas,
        variable=None,  # todas as variaveis
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    ```

    **Output**: Funcoes de peso sobrepostas para cada variavel de alta
    frequencia. Permite comparar quais variaveis tem decaimento mais rapido.

---

## `plot_ragged_edge`

Visualizacao dos dados disponiveis (ragged edge) mostrando quais variaveis
tem dados disponiveis em cada periodo. Essencial para entender o padrao
de publicacao dos indicadores.

**Output visual**: Heatmap binario (disponivel/missing) onde linhas sao variaveis
e colunas sao periodos. Celulas preenchidas indicam dados observados; vazias
indicam missing. O padrao escalonado (ragged edge) reflete as diferentes datas
de publicacao.

```python
from forecastbox.plot import plot_ragged_edge

fig = plot_ragged_edge(data)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `data` | `pd.DataFrame` | *required* | Painel de dados com NaN para missing |
| `variable_names` | `list[str] \| None` | `None` | Nomes das variaveis |
| `n_periods` | `int` | `12` | Numero de periodos recentes a exibir |
| `sort_by` | `str` | `"availability"` | Ordenacao: `"availability"`, `"name"`, `"publication_lag"` |
| `group_by` | `str \| None` | `None` | Agrupar por coluna (ex: setor) |
| `show_publication_lag` | `bool` | `True` | Mostrar defasagem de publicacao |
| `cmap` | `str` | `"Blues"` | Colormap |
| `figsize` | `tuple[float, float]` | `(14, 8)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplo

```python
from forecastbox.plot import plot_ragged_edge

plot_ragged_edge(
    data,
    n_periods=6,
    sort_by="publication_lag",
    title="Ragged Edge - Dados para Nowcast PIB",
    style="publication",
)
```

**Output**: Heatmap com ~30 variaveis e 6 meses. Variaveis de alta frequencia
(PMI, cambio, juros) preenchidas ate o mes atual. Variaveis com defasagem
(producao industrial, vendas) faltando o ultimo mes. PIB faltando os 2 ultimos
meses. Padrao escalonado claramente visivel.

!!! tip "Uso pratico do ragged edge"

    A visualizacao de ragged edge ajuda a:

    - Identificar quais indicadores ja foram publicados
    - Planejar o calendario de atualizacao do nowcast
    - Entender por que uma revisao do nowcast pode ser grande (muitos dados novos)
    - Diagnosticar problemas de dados (variaveis inesperadamente missing)

---

## `plot_vintage_comparison`

Comparacao de nowcasts produzidos em diferentes vintage dates para o mesmo
periodo de referencia. Mostra como a estimativa evoluiu ao longo do tempo.

**Output visual**: Barras ou linhas com o valor do nowcast em cada vintage date.
Cor codifica a distancia ate o valor realizado. Barra final com o valor realizado
(se disponivel).

```python
from forecastbox.plot import plot_vintage_comparison

fig = plot_vintage_comparison(vintages)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `vintages` | `dict[str, NowcastResult]` | *required* | Dicionario {vintage_date: nowcast} |
| `actual` | `float \| None` | `None` | Valor realizado |
| `target_variable` | `str \| None` | `None` | Nome da variavel alvo |
| `chart_type` | `str` | `"bar"` | Tipo: `"bar"`, `"line"`, `"lollipop"` |
| `show_error` | `bool` | `True` | Exibir erro de previsao |
| `vintage_dates` | `list[str] \| None` | `None` | Labels customizados para vintages |
| `highlight_news` | `bool` | `False` | Destacar vintages com news significativas |
| `colors` | `dict \| None` | `None` | Cores: `{"improving", "worsening", "actual"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(12, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Comparacao basica"

    ```python
    from forecastbox.plot import plot_vintage_comparison

    vintages = {
        "2024-07-01": nc_jul,
        "2024-08-01": nc_ago,
        "2024-09-01": nc_set,
        "2024-10-01": nc_out,
    }

    plot_vintage_comparison(
        vintages,
        actual=0.8,
        target_variable="PIB T3/2024",
    )
    ```

    **Output**: 4 barras azuis (Jul: 0.6%, Ago: 0.7%, Set: 0.9%, Out: 0.8%)
    convergindo para o valor realizado (linha tracejada em 0.8%). Ultima barra
    verde indicando acerto.

=== "Lollipop com erro"

    ```python
    plot_vintage_comparison(
        vintages,
        actual=0.8,
        chart_type="lollipop",
        show_error=True,
        style="publication",
    )
    ```

    **Output**: Formato lollipop com circulos nos valores. Eixo secundario
    mostrando erro absoluto em cada vintage. Erro decrescente ao longo do
    tempo.

---

## Painel de Nowcasting Completo

Combine os graficos de nowcasting em um painel para apresentacao:

```python
import matplotlib.pyplot as plt
from forecastbox.plot import (
    plot_nowcast_evolution,
    plot_news_waterfall,
    plot_ragged_edge,
    plot_factor_loadings,
)

fig = plt.figure(figsize=(16, 12))

ax1 = fig.add_subplot(2, 2, 1)
plot_nowcast_evolution(nowcasts, actual=pib, ax=ax1, show=False)

ax2 = fig.add_subplot(2, 2, 2)
plot_news_waterfall(news, top_n=8, ax=ax2, show=False)

ax3 = fig.add_subplot(2, 2, 3)
plot_ragged_edge(data, n_periods=6, ax=ax3, show=False)

ax4 = fig.add_subplot(2, 2, 4)
plot_factor_loadings(dfm, ax=ax4, show=False)

fig.suptitle("Nowcast Dashboard - PIB T3/2024", fontsize=14)
plt.tight_layout()
plt.savefig("nowcast_dashboard.png", dpi=300, bbox_inches="tight")
plt.show()
```

**Output**: Painel 2x2 com evolucao do nowcast (convergencia), news waterfall
(o que mudou), ragged edge (dados disponiveis) e factor loadings (estrutura
do modelo).

---

## See Also

- :material-school: [Tutorial: Nowcasting](../tutorials/nowcasting.md) — aprenda nowcasting com bridge, DFM e MIDAS
- :material-school: [Tutorial: MIDAS](../tutorials/midas.md) — MIDAS em detalhe com dados de multiplas frequencias
- [Graficos de Previsao](forecast-plots.md) — visualizacao de previsoes individuais
- [Graficos de Avaliacao](evaluation-plots.md) — testes de habilidade preditiva
- [User Guide - Nowcasting](../user-guide/nowcasting/index.md) — referencia completa
- [Theory - Nowcasting](../theory/nowcasting-theory.md) — fundamentos teoricos
- [API Reference - Visualization](../api/visualization.md) — referencia completa da API
