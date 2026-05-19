---
title: Graficos de Pipeline e Monitoramento
description: Visualizacao de DAGs de pipeline, dashboards de monitoramento, deteccao de drift e comparacao de experimentos
---

# Graficos de Pipeline e Monitoramento

Funcoes para visualizar a estrutura e o estado de pipelines de previsao:
grafos DAG, dashboards de metricas, deteccao de drift, comparacao de
experimentos e ciclo de vida de modelos.

---

## `plot_pipeline_dag`

Grafo direcionado aciclico (DAG) mostrando as etapas do pipeline de previsao
e suas dependencias. Estilo visual inspirado em diagramas mermaid.

**Output visual**: Grafo com nos representando etapas (data ingestion, transform,
train, evaluate, combine, forecast) e arestas indicando dependencias. Nos
coloridos por status (concluido, em execucao, pendente, erro). Tempo de
execucao anotado em cada no.

```python
from forecastbox.plot import plot_pipeline_dag

fig = plot_pipeline_dag(pipeline)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `pipeline` | `ForecastPipeline` | *required* | Pipeline configurado ou executado |
| `show_status` | `bool` | `True` | Colorir nos por status |
| `show_timing` | `bool` | `True` | Anotar tempo de execucao |
| `show_data_flow` | `bool` | `False` | Exibir shape dos dados entre etapas |
| `layout` | `str` | `"horizontal"` | Layout: `"horizontal"`, `"vertical"`, `"radial"` |
| `highlight_step` | `str \| None` | `None` | Destacar etapa especifica |
| `colors` | `dict \| None` | `None` | Cores: `{"completed", "running", "pending", "error"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(14, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "DAG basico"

    ```python
    from forecastbox import ForecastPipeline
    from forecastbox.plot import plot_pipeline_dag

    pipeline = ForecastPipeline(
        steps=[
            ("load", DataLoader(source="bcb")),
            ("transform", SeasonalAdjust()),
            ("train", AutoForecast(strategy="ensemble")),
            ("evaluate", Evaluate(metrics=["rmse", "mase"])),
            ("combine", Combine(method="bma")),
            ("forecast", Predict(horizon=12)),
        ]
    )

    plot_pipeline_dag(pipeline)
    ```

    **Output**: Grafo horizontal com 6 nos conectados sequencialmente.
    Cores: verde (concluido), amarelo (em execucao), cinza (pendente).
    Setas indicam fluxo de dados.

=== "Com fluxo de dados"

    ```python
    plot_pipeline_dag(
        pipeline,
        show_data_flow=True,
        layout="vertical",
        highlight_step="train",
        title="Pipeline de Previsao - IPCA",
    )
    ```

    **Output**: Layout vertical com shapes dos dados entre etapas
    (ex: "(120, 5)" na aresta load→transform). Etapa "train" destacada
    com borda mais espessa.

=== "Pipeline com erro"

    ```python
    # Apos execucao com falha
    pipeline.run()

    plot_pipeline_dag(
        pipeline,
        show_timing=True,
        style="dark",
    )
    ```

    **Output**: Etapas concluidas em verde com tempo (ex: "load: 2.3s").
    Etapa com erro em vermelho com mensagem resumida. Etapas pendentes
    em cinza.

---

## `plot_monitor_dashboard`

Dashboard consolidado de metricas de monitoramento em tempo real. Exibe
as principais metricas de performance do modelo ao longo do tempo.

**Output visual**: Grid de subplots com metricas selecionadas. Cada subplot
mostra a serie temporal da metrica com limites de alerta (warning/critical).
Fundo colorido nas regioes que excedem os limites.

```python
from forecastbox.plot import plot_monitor_dashboard

fig = plot_monitor_dashboard(monitor)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `monitor` | `ModelMonitor` | *required* | Monitor com historico de metricas |
| `metrics` | `list[str] \| None` | `None` | Metricas a exibir (default: todas) |
| `window` | `int \| None` | `None` | Janela de periodos recentes |
| `alerts` | `bool` | `True` | Exibir limites de alerta |
| `alert_levels` | `dict \| None` | `None` | Limites: `{"warning": float, "critical": float}` |
| `show_trend` | `bool` | `True` | Exibir tendencia (media movel) |
| `n_cols` | `int` | `2` | Numero de colunas no grid |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(14, 8)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Dashboard completo"

    ```python
    from forecastbox.monitor import ModelMonitor
    from forecastbox.plot import plot_monitor_dashboard

    monitor = ModelMonitor(model_id="ipca_v3")
    monitor.load_history()

    plot_monitor_dashboard(
        monitor,
        metrics=["rmse", "mae", "coverage_80", "bias"],
        window=52,  # ultimas 52 semanas
    )
    ```

    **Output**: Grid 2x2 com: RMSE (topo esquerda), MAE (topo direita),
    Cobertura 80% (baixo esquerda), Vies (baixo direita). Cada subplot
    com serie temporal, media movel (linha tracejada), e faixas de alerta
    (amarelo=warning, vermelho=critical).

=== "Com alertas customizados"

    ```python
    plot_monitor_dashboard(
        monitor,
        metrics=["rmse", "mape"],
        alerts=True,
        alert_levels={"warning": 1.5, "critical": 2.0},
        title="Monitoramento - IPCA",
        style="dark",
    )
    ```

    **Output**: Tema escuro. Faixas horizontais em amarelo (warning acima de
    1.5x baseline) e vermelho (critical acima de 2.0x baseline). Periodos
    em alerta destacados com fundo colorido.

---

## `plot_drift_detection`

Graficos de deteccao de drift (concept drift) usando estatisticas CUSUM
e indicadores de mudanca. Identifica quando o modelo perde aderencia.

**Output visual**: Painel com 2-3 subplots: (1) estatistica CUSUM acumulada
com limites de decisao, (2) erro de previsao ao longo do tempo com media
movel, (3) indicador binario de drift (detectado/nao detectado).

A estatistica CUSUM e definida como:

$$
S_t = \max(0, S_{t-1} + (e_t - \mu_0) - k)
$$

onde $e_t$ e o erro de previsao, $\mu_0$ e o erro medio no periodo de referencia,
e $k$ e o parametro de sensibilidade. Um alarme e disparado quando $S_t > h$.

```python
from forecastbox.plot import plot_drift_detection

fig = plot_drift_detection(monitor)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `monitor` | `ModelMonitor` | *required* | Monitor com deteccao de drift |
| `method` | `str` | `"cusum"` | Metodo: `"cusum"`, `"page_hinkley"`, `"ddm"`, `"adwin"` |
| `window` | `int \| None` | `None` | Janela de periodos |
| `show_statistic` | `bool` | `True` | Exibir estatistica do teste |
| `show_errors` | `bool` | `True` | Exibir serie de erros |
| `show_alarm` | `bool` | `True` | Exibir indicador de alarme |
| `threshold` | `float \| None` | `None` | Limiar de decisao (auto-calculado se `None`) |
| `annotate_drift` | `bool` | `True` | Anotar pontos de drift detectado |
| `colors` | `dict \| None` | `None` | Cores: `{"statistic", "threshold", "alarm", "error"}` |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(12, 8)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "CUSUM basico"

    ```python
    from forecastbox.monitor import ModelMonitor
    from forecastbox.plot import plot_drift_detection

    monitor = ModelMonitor(model_id="ipca_v3")
    monitor.detect_drift(method="cusum")

    plot_drift_detection(monitor, method="cusum")
    ```

    **Output**: Painel 3x1. Topo: estatistica CUSUM crescente, cruzando o limiar
    em Ago/2024 (drift detectado, anotado com seta vermelha). Meio: erros de
    previsao com media movel subindo. Base: indicador binario (0=ok, 1=drift)
    com transicao em Ago/2024.

=== "Page-Hinkley"

    ```python
    plot_drift_detection(
        monitor,
        method="page_hinkley",
        annotate_drift=True,
        title="Drift Detection - Modelo IPCA v3",
        style="publication",
    )
    ```

    **Output**: Formato de publicacao. Estatistica Page-Hinkley com limiar
    adaptativo. Pontos de drift anotados com datas especificas.

!!! warning "Drift nao e erro pontual"

    Um erro de previsao grande em um unico periodo nao configura drift.
    Drift e uma mudanca **sistematica** na relacao entre variaveis. Os
    metodos CUSUM e Page-Hinkley sao projetados para detectar mudancas
    persistentes, nao outliers isolados.

---

## `plot_experiment_comparison`

Comparacao visual de multiplos runs de experimentos. Permite identificar
o melhor modelo/configuracao em multiplas dimensoes.

**Output visual**: Grafico multi-dimensional comparando experiments em
metricas selecionadas. Pode ser radar chart, parallel coordinates ou
bar chart agrupado.

```python
from forecastbox.plot import plot_experiment_comparison

fig = plot_experiment_comparison(experiments)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `experiments` | `list[ExperimentResult]` | *required* | Lista de resultados de experimentos |
| `metrics` | `list[str] \| None` | `None` | Metricas a comparar |
| `chart_type` | `str` | `"radar"` | Tipo: `"radar"`, `"parallel"`, `"bar"`, `"table"` |
| `normalize` | `bool` | `True` | Normalizar metricas para [0, 1] |
| `highlight_best` | `bool` | `True` | Destacar melhor experimento por metrica |
| `show_params` | `bool` | `False` | Exibir hiperparametros no tooltip/legenda |
| `colors` | `list[str] \| None` | `None` | Cores por experimento |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(10, 8)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `ax` | `Axes \| None` | `None` | Eixo existente |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Radar chart"

    ```python
    from forecastbox import Experiment
    from forecastbox.plot import plot_experiment_comparison

    experiments = [
        Experiment("ARIMA", results_arima),
        Experiment("ETS", results_ets),
        Experiment("Ensemble", results_ensemble),
    ]

    plot_experiment_comparison(
        experiments,
        metrics=["rmse", "mae", "mape", "coverage_80", "crps"],
        chart_type="radar",
    )
    ```

    **Output**: Radar chart com 5 eixos (metricas normalizadas). 3 poligonos
    sobrepostos (ARIMA em azul, ETS em laranja, Ensemble em verde). Ensemble
    com area maior (melhor em mais metricas).

=== "Parallel coordinates"

    ```python
    plot_experiment_comparison(
        experiments,
        metrics=["rmse", "mae", "mape", "coverage_80", "crps"],
        chart_type="parallel",
        highlight_best=True,
        backend="plotly",
    )
    ```

    **Output**: Coordenadas paralelas interativas. Cada linha e um experimento.
    Eixos verticais sao metricas. Melhor valor por metrica destacado. Hover
    mostra detalhes do experimento.

=== "Barras agrupadas"

    ```python
    plot_experiment_comparison(
        experiments,
        metrics=["rmse", "mae"],
        chart_type="bar",
        normalize=False,
        show_params=True,
        style="publication",
    )
    ```

    **Output**: Barras agrupadas por metrica. Valores absolutos (nao
    normalizados). Hiperparametros na legenda.

---

## `plot_model_lifecycle`

Timeline do ciclo de vida de modelos: treino, deploy, monitoramento e retrain.
Mostra a historia operacional do modelo em producao.

**Output visual**: Timeline horizontal com marcos (treino, deploy, alertas,
retrain) ao longo do tempo. Barras horizontais indicam periodos de atividade
de cada versao do modelo. Icones para eventos (treino, alerta, retrain).

```python
from forecastbox.plot import plot_model_lifecycle

fig = plot_model_lifecycle(monitor)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `monitor` | `ModelMonitor` | *required* | Monitor com historico de lifecycle |
| `model_ids` | `list[str] \| None` | `None` | IDs dos modelos (default: todos) |
| `show_events` | `bool` | `True` | Exibir eventos (treino, deploy, retrain) |
| `show_metrics` | `bool` | `False` | Exibir metrica de performance abaixo |
| `metrics` | `list[str]` | `["rmse"]` | Metricas para subplot inferior |
| `window` | `str \| None` | `None` | Periodo: `"6M"`, `"1Y"`, `"2Y"` |
| `colors` | `dict \| None` | `None` | Cores por versao do modelo |
| `title` | `str \| None` | `None` | Titulo |
| `figsize` | `tuple[float, float]` | `(14, 6)` | Tamanho da figura |
| `backend` | `str` | `"matplotlib"` | Backend |
| `style` | `str` | `"light"` | Tema |
| `show` | `bool` | `True` | Exibir automaticamente |

### Exemplos

=== "Timeline basica"

    ```python
    from forecastbox.monitor import ModelMonitor
    from forecastbox.plot import plot_model_lifecycle

    monitor = ModelMonitor(model_id="ipca")
    monitor.load_history()

    plot_model_lifecycle(monitor)
    ```

    **Output**: Timeline horizontal mostrando:

    - `ipca_v1`: barra azul de Jan/2023 a Jul/2023, marco "deploy" no inicio,
      marco "retrain" no fim
    - `ipca_v2`: barra verde de Jul/2023 a Jan/2024, com alerta amarelo em
      Out/2023
    - `ipca_v3`: barra roxa de Jan/2024 ate presente

=== "Com metricas"

    ```python
    plot_model_lifecycle(
        monitor,
        show_metrics=True,
        metrics=["rmse", "bias"],
        window="2Y",
        style="publication",
    )
    ```

    **Output**: Painel 2x1. Topo: timeline de modelos. Base: RMSE e bias
    ao longo do tempo, com cores correspondentes a versao ativa. Transicoes
    de modelo visiveis como mudancas de cor.

!!! tip "Quando retreinar?"

    O grafico de lifecycle combinado com drift detection ajuda a responder
    quando retreinar:

    - **Drift detectado + metricas degradando**: retreinar imediatamente
    - **Drift detectado + metricas estaveis**: monitorar por mais tempo
    - **Sem drift + metricas degradando**: investigar mudancas nos dados
    - **Retreino periodico**: schedule fixo (ex: mensal) como baseline

---

## Pipeline Dashboard Completo

Combine os graficos de pipeline em um dashboard operacional:

```python
import matplotlib.pyplot as plt
from forecastbox.plot import (
    plot_pipeline_dag,
    plot_monitor_dashboard,
    plot_drift_detection,
    plot_model_lifecycle,
)

fig = plt.figure(figsize=(18, 14))

# DAG do pipeline (topo, largura total)
ax1 = fig.add_subplot(3, 2, (1, 2))
plot_pipeline_dag(pipeline, ax=ax1, show=False)

# Metricas de monitoramento
ax2 = fig.add_subplot(3, 2, 3)
plot_monitor_dashboard(monitor, metrics=["rmse"], ax=ax2, show=False)

# Drift detection
ax3 = fig.add_subplot(3, 2, 4)
plot_drift_detection(monitor, ax=ax3, show=False)

# Lifecycle (base, largura total)
ax4 = fig.add_subplot(3, 2, (5, 6))
plot_model_lifecycle(monitor, show_metrics=True, ax=ax4, show=False)

fig.suptitle("Pipeline Operations Dashboard", fontsize=14)
plt.tight_layout()
plt.savefig("pipeline_dashboard.png", dpi=300, bbox_inches="tight")
plt.show()
```

**Output**: Dashboard operacional completo com estrutura do pipeline (DAG),
metricas em tempo real, deteccao de drift e historico de modelos.

---

## See Also

- :material-school: [Tutorial: Pipeline de Producao](../tutorials/pipeline.md) — aprenda a criar pipelines automatizados
- :material-school: [Tutorial: Workflow Completo](../tutorials/complete-workflow.md) — projeto end-to-end com dashboard
- [Graficos de Previsao](forecast-plots.md) — visualizacao de previsoes individuais
- [Graficos de Avaliacao](evaluation-plots.md) — testes e metricas
- [User Guide - Pipeline](../user-guide/pipeline/pipeline.md) — referencia completa
- [API Reference - Visualization](../api/visualization.md) — referencia completa da API
