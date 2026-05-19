---
title: Visualization
description: Graficos prontos para publicacao com backends matplotlib, plotly e seaborn
---

# Visualization

O modulo `forecastbox.plot` oferece graficos prontos para publicacao com uma unica
linha de codigo. Todos os graficos seguem uma API unificada com suporte a multiplos
backends, temas e formatos de exportacao.

---

## Filosofia

> **Uma linha de codigo, um grafico publicavel.**

O forecastbox gera graficos que seguem boas praticas de visualizacao econometrica:
fontes legiveis, intervalos de confianca sombreados, legendas informativas e
paletas de cores acessiveis. Nao e necessario configurar eixos, rotacoes de labels
ou formatacao de datas manualmente.

---

## Backends Suportados

O forecastbox suporta tres backends de visualizacao, selecionaveis via parametro
`backend` em qualquer funcao `plot.*`:

| Backend | Tipo | Melhor Para | Formato |
|:--------|:-----|:------------|:--------|
| **matplotlib** | Estatico | Papers, relatorios PDF, apresentacoes | PNG, SVG, PDF |
| **plotly** | Interativo | Dashboards, exploracao, HTML | HTML |
| **seaborn** | Estatico | Visualizacoes estatisticas, heatmaps | PNG, SVG, PDF |

=== "matplotlib (padrao)"

    ```python
    from forecastbox.plot import plot_forecast

    fig = plot_forecast(forecast, backend="matplotlib")
    fig.savefig("forecast.png", dpi=300, bbox_inches="tight")
    ```

=== "plotly"

    ```python
    from forecastbox.plot import plot_forecast

    fig = plot_forecast(forecast, backend="plotly")
    fig.show()  # abre no browser
    fig.write_html("forecast.html")
    ```

=== "seaborn"

    ```python
    from forecastbox.plot import plot_forecast

    fig = plot_forecast(forecast, backend="seaborn")
    fig.savefig("forecast.svg")
    ```

!!! info "Importacao"

    Todas as funcoes de visualizacao estao disponiveis via namespace unificado:

    ```python
    from forecastbox.plot import (
        plot_forecast,
        plot_comparison,
        plot_weights,
        plot_metrics_heatmap,
        # ... todas as funcoes plot_*
    )
    ```

---

## Temas

O forecastbox inclui 4 temas pre-configurados, otimizados para diferentes contextos:

| Tema | Uso Recomendado | Caracteristicas |
|:-----|:----------------|:----------------|
| `"light"` | Painel de controle, web | Fundo branco, cores vibrantes |
| `"dark"` | Dashboards, apresentacoes com fundo escuro | Fundo escuro, cores neon |
| `"publication"` | Papers academicos, journals | Fundo branco, fontes serif, linhas finas |
| `"presentation"` | Slides, conferencias | Fontes grandes, alto contraste, linhas grossas |

```python
from forecastbox.plot import set_theme, plot_forecast

# Aplicar tema globalmente
set_theme("publication")

# Ou por grafico
fig = plot_forecast(forecast, style="presentation")
```

!!! tip "Tema para publicacao"

    O tema `"publication"` segue as diretrizes de journals como Econometrica,
    Journal of Econometrics e Review of Economics and Statistics: fontes Computer
    Modern, tamanho adequado para impressao em coluna unica (3.5") ou dupla (7"),
    e paleta em tons de cinza para compatibilidade com impressao P&B.

---

## Exportacao

Todos os graficos podem ser exportados nos seguintes formatos:

| Formato | Backend | Uso |
|:--------|:--------|:----|
| **PNG** | matplotlib, seaborn | Relatorios, apresentacoes |
| **SVG** | matplotlib, seaborn | Web, escalavel |
| **PDF** | matplotlib, seaborn | Papers, impressao |
| **HTML** | plotly | Dashboards interativos |

```python
from forecastbox.plot import plot_forecast, save_figure

fig = plot_forecast(forecast)

# Salvar em multiplos formatos
save_figure(fig, "forecast", formats=["png", "svg", "pdf"], dpi=300)
# Gera: forecast.png, forecast.svg, forecast.pdf
```

---

## API Unificada

Todas as funcoes seguem a mesma convencao de parametros:

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `backend` | `str` | `"matplotlib"` | Backend de renderizacao |
| `style` | `str` | `"light"` | Tema visual |
| `figsize` | `tuple[float, float]` | `(10, 6)` | Tamanho da figura em polegadas |
| `title` | `str \| None` | `None` | Titulo customizado (auto-gerado se `None`) |
| `ax` | `Axes \| None` | `None` | Eixo matplotlib existente (composicao) |
| `show` | `bool` | `True` | Exibir automaticamente |

!!! note "Composicao de graficos"

    Passe `ax` para compor multiplos graficos no mesmo painel:

    ```python
    import matplotlib.pyplot as plt
    from forecastbox.plot import plot_forecast, plot_residuals

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_forecast(forecast, ax=axes[0], show=False)
    plot_residuals(forecast, ax=axes[1], show=False)
    plt.tight_layout()
    plt.show()
    ```

---

## Funcoes Disponiveis

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **Graficos de Previsao**

    ---

    Serie historica, intervalos de confianca, fan charts, residuos
    e multiplas previsoes sobrepostas.

    [:octicons-arrow-right-24: Forecast Plots](forecast-plots.md)

-   :material-chart-bar:{ .lg .middle } **Graficos de Comparacao**

    ---

    Comparacao lado a lado, barras de metricas, heatmaps,
    boxplots de erro e rankings de modelos.

    [:octicons-arrow-right-24: Comparison Plots](comparison-plots.md)

-   :material-scale-balance:{ .lg .middle } **Graficos de Combinacao**

    ---

    Pesos de combinacao, evolucao temporal, posterior BMA,
    estabilidade e combinacao vs. individuais.

    [:octicons-arrow-right-24: Combination Plots](combination-plots.md)

-   :material-test-tube:{ .lg .middle } **Graficos de Avaliacao**

    ---

    Diebold-Mariano, Model Confidence Set, Mincer-Zarnowitz,
    calibracao probabilistica e encompassing.

    [:octicons-arrow-right-24: Evaluation Plots](evaluation-plots.md)

-   :material-pulse:{ .lg .middle } **Graficos de Nowcasting**

    ---

    Evolucao do nowcast, news waterfall, factor loadings,
    pesos MIDAS, ragged edge e comparacao de vintages.

    [:octicons-arrow-right-24: Nowcast Plots](nowcast-plots.md)

-   :material-pipe:{ .lg .middle } **Graficos de Pipeline**

    ---

    DAGs de pipeline, dashboards de monitoramento,
    deteccao de drift e comparacao de experimentos.

    [:octicons-arrow-right-24: Pipeline Plots](pipeline-plots.md)

-   :material-palette:{ .lg .middle } **Temas e Customizacao**

    ---

    5 temas built-in, paletas colorblind-safe, export para
    publicacao e criacao de temas customizados.

    [:octicons-arrow-right-24: Themes](themes.md)

</div>

---

## Catalogo Completo de Funcoes

### Previsao

| Funcao | Descricao |
|:-------|:----------|
| `plot_forecast` | Serie historica + previsao com intervalos de confianca |
| `plot_forecast_interactive` | Versao interativa (Plotly) com hover e zoom |
| `plot_multi_forecast` | Sobreposicao de multiplas previsoes |
| `plot_fan_chart` | Fan chart com multiplos quantis (estilo Banco Central) |
| `plot_residuals` | Painel 2x2 de diagnostico de residuos |
| `plot_residuals_extended` | Painel 3x2 com PACF e deteccao ARCH |
| `plot_forecast_decomposition` | Decomposicao em tendencia, sazonalidade e residuo |

### Comparacao

| Funcao | Descricao |
|:-------|:----------|
| `plot_comparison` | Previsoes de N modelos sobrepostas com observado |
| `plot_metrics_bar` | Barras comparando metricas de erro por modelo |
| `plot_metrics_heatmap` | Heatmap de metricas vs. modelos |
| `plot_error_boxplot` | Boxplot da distribuicao de erros por modelo |
| `plot_cumulative_error` | Erro acumulado ao longo do tempo |
| `plot_ranking` | Ranking visual de modelos por metrica |
| `plot_error_by_horizon` | Erro medio por horizonte de previsao |

### Combinacao

| Funcao | Descricao |
|:-------|:----------|
| `plot_weights` | Barras com peso de cada modelo na combinacao |
| `plot_weights_evolution` | Area empilhada da evolucao temporal dos pesos |
| `plot_weights_heatmap` | Heatmap de pesos por modelo e periodo |
| `plot_combination_vs_individual` | Combinacao vs. melhores modelos individuais |
| `plot_posterior` | Distribuicao posterior BMA |
| `plot_weight_stability` | Boxplot de estabilidade dos pesos |
| `plot_weight_contribution` | Contribuicao de cada modelo por periodo |

### Avaliacao

| Funcao | Descricao |
|:-------|:----------|
| `plot_dm_test` | Diferenciais de perda do teste Diebold-Mariano |
| `plot_mcs_inclusion` | Heatmap de inclusao no Model Confidence Set |
| `plot_cv_results` | Performance por fold de validacao cruzada |
| `plot_mincer_zarnowitz` | Scatter actual vs. forecast com regressao MZ |
| `plot_encompassing` | Heatmap de testes de encompassing |
| `plot_calibration` | Histograma PIT para calibracao probabilistica |
| `plot_reliability` | Reliability diagram para avaliacao de quantis |

### Nowcasting

| Funcao | Descricao |
|:-------|:----------|
| `plot_nowcast_evolution` | Evolucao do nowcast ao longo do trimestre |
| `plot_news_waterfall` | Waterfall de contribuicao por indicador |
| `plot_factor_loadings` | Heatmap de loadings do DFM |
| `plot_midas_weights` | Funcao de pesos MIDAS (Beta, Almon) |
| `plot_ragged_edge` | Heatmap de disponibilidade de dados |
| `plot_vintage_comparison` | Comparacao de nowcasts por vintage date |

### Pipeline e Monitoramento

| Funcao | Descricao |
|:-------|:----------|
| `plot_pipeline_dag` | Grafo DAG das etapas do pipeline |
| `plot_monitor_dashboard` | Dashboard de metricas de monitoramento |
| `plot_drift_detection` | CUSUM e deteccao de concept drift |
| `plot_experiment_comparison` | Radar/parallel charts de experimentos |
| `plot_model_lifecycle` | Timeline do ciclo de vida de modelos |

### Temas

| Funcao | Descricao |
|:-------|:----------|
| `set_theme` | Definir tema global |
| `get_theme` | Obter tema atual ou por nome |
| `reset_theme` | Restaurar tema padrao |
| `theme_context` | Context manager para tema temporario |
| `Theme` | Criar tema customizado |
| `register_theme` | Registrar tema para reutilizacao |

---

## See Also

- :material-school: [Tutorials](../tutorials/index.md) — tutoriais praticos com visualizacoes em cada etapa
- [User Guide - Avaliacao](../user-guide/evaluation/index.md) — metricas e testes estatisticos
- [User Guide - Combinacao](../user-guide/combination/index.md) — metodos de combinacao de previsoes
- [API Reference - Visualization](../api/visualization.md) — referencia completa da API
- [Temas](themes.md) — customizacao avancada de temas
