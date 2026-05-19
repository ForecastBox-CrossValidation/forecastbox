---
title: ForecastPipeline
description: Pipeline de previsao com builder pattern, configuracao YAML, execucao paralela e checkpointing para producao.
---

# ForecastPipeline

O `ForecastPipeline` e o orquestrador central do forecastbox para producao.
Ele encapsula todas as etapas — da leitura de dados ate a exportacao de
resultados — em um unico objeto configuravel, reprodutivel e paralelizavel.

---

## Conceito: Builder Pattern

O pipeline usa o **builder pattern** para construcao fluente. Cada metodo
`.add_*()` adiciona uma etapa, e `.build()` valida e compila o pipeline.

```mermaid
graph LR
    A[".add_data()"] --> B[".add_preprocessing()"]
    B --> C[".add_models()"]
    C --> D[".add_combination()"]
    D --> E[".add_evaluation()"]
    E --> F[".add_output()"]
    F --> G[".build()"]
    G --> H[".run()"]

    style A fill:#2E7D32,stroke:#1B5E20,color:#fff
    style B fill:#009688,stroke:#00796B,color:#fff
    style C fill:#1565C0,stroke:#0D47A1,color:#fff
    style D fill:#6A1B9A,stroke:#4A148C,color:#fff
    style E fill:#E65100,stroke:#BF360C,color:#fff
    style F fill:#F57F17,stroke:#F9A825,color:#000
    style G fill:#C62828,stroke:#B71C1C,color:#fff
    style H fill:#37474F,stroke:#263238,color:#fff
```

!!! tip "Ordem flexivel"

    As etapas `.add_preprocessing()` e `.add_combination()` sao opcionais.
    O unico requisito e que `.add_data()` venha primeiro e `.build()` venha
    por ultimo. O pipeline valida a consistencia na compilacao.

---

## Construcao via Codigo

### Pipeline basico

```python
from forecastbox.pipeline import ForecastPipeline

pipeline = (
    ForecastPipeline()
    .add_data(source="parquet", path="data/macro.parquet")
    .add_models(["AutoARIMA", "AutoETS"])
    .add_evaluation(metrics=["rmse"], cv_folds=3)
    .add_output(format="csv", path="output/forecast.csv")
    .build()
)

results = pipeline.run()
```

### Pipeline completo com combinacao

```python
from forecastbox.pipeline import ForecastPipeline

pipeline = (
    ForecastPipeline()
    # --- Data ---
    .add_data(
        source="parquet",
        path="data/macro.parquet",
        target_cols=["ipca", "pib", "selic"],
        date_col="date",
        freq="MS"
    )
    # --- Pre-processing ---
    .add_preprocessing(
        steps=["deseason", "detrend", "normalize"],
        deseason_method="stl",
        detrend_method="hp",
        normalize_method="zscore"
    )
    # --- Models ---
    .add_models(
        models=["AutoARIMA", "AutoETS", "AutoVAR"],
        horizon=12,
        n_jobs=4
    )
    # --- Combination ---
    .add_combination(
        method="bma",
        training_window=60
    )
    # --- Evaluation ---
    .add_evaluation(
        metrics=["rmse", "mase", "coverage"],
        cv_folds=5,
        cv_strategy="expanding"
    )
    # --- Output ---
    .add_output(
        format="excel",
        path="output/forecast_macro.xlsx",
        include_plots=True,
        include_diagnostics=True
    )
    .build()
)

results = pipeline.run()
```

---

## Configuracao via YAML

Para producao, e comum definir o pipeline em um arquivo YAML versionado no
repositorio. Isso permite revisao de codigo e reproducibilidade total.

=== "pipeline.yml"

    ```yaml
    pipeline:
      name: "macro_forecast_v2"
      description: "Pipeline de previsao macroeconomica mensal"

      data:
        source: parquet
        path: data/macro.parquet
        target_cols: [ipca, pib, selic, cambio, desemprego]
        date_col: date
        freq: MS

      preprocessing:
        steps: [deseason, detrend, normalize]
        deseason_method: stl
        detrend_method: hp
        normalize_method: zscore

      models:
        list: [AutoARIMA, AutoETS, AutoVAR]
        horizon: 12
        n_jobs: -1  # todos os cores

      combination:
        method: bma
        training_window: 60

      evaluation:
        metrics: [rmse, mase, coverage]
        cv_folds: 5
        cv_strategy: expanding

      output:
        format: excel
        path: output/forecast_macro.xlsx
        include_plots: true
        include_diagnostics: true
    ```

=== "Carregar YAML"

    ```python
    from forecastbox.pipeline import ForecastPipeline

    pipeline = ForecastPipeline.from_yaml("pipeline.yml")
    results = pipeline.run()
    ```

!!! warning "Validacao de YAML"

    O `ForecastPipeline.from_yaml()` valida o schema do arquivo YAML e
    levanta `PipelineConfigError` com mensagem detalhada se algum campo
    estiver faltando ou for invalido.

---

## Parametros de Execucao

O metodo `.run()` aceita parametros que controlam o comportamento da execucao:

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `n_jobs` | `int` | `1` | Numero de workers para execucao paralela. `-1` usa todos os cores |
| `cache` | `bool` | `True` | Cachear resultados intermediarios para evitar recomputacao |
| `verbose` | `int` | `1` | Nivel de verbosidade: `0`=silencioso, `1`=progresso, `2`=detalhado |
| `checkpoint_dir` | `str` | `None` | Diretorio para salvar checkpoints intermediarios |
| `dry_run` | `bool` | `False` | Valida o pipeline sem executar (util para CI/CD) |

### Execucao paralela

O forecastbox paraleliza a estimacao de modelos por serie e por modelo.
Com `n_jobs=-1`, todas as combinacoes serie x modelo sao distribuidas
entre os cores disponiveis.

```python
# Execucao paralela com cache e checkpointing
results = pipeline.run(
    n_jobs=-1,
    cache=True,
    verbose=2,
    checkpoint_dir="checkpoints/"
)
```

!!! info "Paralelismo inteligente"

    O pipeline agrupa modelos por tipo para minimizar overhead. Modelos
    univariados (ARIMA, ETS) sao paralelizados por serie. Modelos
    multivariados (VAR) sao executados uma vez e as previsoes distribuidas.

---

## Checkpointing e Resume

Para pipelines longos, o checkpointing permite retomar a execucao de onde parou
em caso de falha ou interrupcao.

```python
# Primeira execucao — salva checkpoints
results = pipeline.run(checkpoint_dir="checkpoints/")

# Se interrompido, retoma do ultimo checkpoint
results = pipeline.resume(checkpoint_dir="checkpoints/")
```

O checkpoint salva:

- **Dados pre-processados** — para nao reprocessar
- **Modelos estimados** — cada modelo e salvo apos fit
- **Resultados parciais** — metricas ja calculadas
- **Metadados** — timestamp, versao do pipeline, hash dos dados

```text
checkpoints/
  ├── data_preprocessed.pkl
  ├── models/
  │   ├── AutoARIMA_ipca.pkl
  │   ├── AutoARIMA_pib.pkl
  │   ├── AutoETS_ipca.pkl
  │   └── ...
  ├── evaluation_partial.pkl
  └── metadata.json
```

---

## Resultados do Pipeline

O objeto `PipelineResult` contem todos os outputs do pipeline:

```python
# Acessar previsoes
results.forecasts          # DataFrame com previsoes de todos os modelos
results.combined_forecast  # DataFrame com previsao combinada
results.best_model         # Melhor modelo por serie

# Acessar metricas
results.metrics            # DataFrame com metricas por modelo x serie
results.cv_results         # Resultados detalhados da validacao cruzada

# Acessar diagnosticos
results.diagnostics        # Testes de residuo, Mincer-Zarnowitz, etc.

# Resumo
print(results.summary())
```

```text
PipelineResult - macro_forecast_v2
====================================
Series: 5 | Models: 3 | Combination: BMA
CV Strategy: expanding (5 folds)

Per-series results:
  ipca:       Best=AutoETS    RMSE=0.231  MASE=0.845
  pib:        Best=AutoARIMA  RMSE=0.412  MASE=0.923
  selic:      Best=BMA        RMSE=0.189  MASE=0.712
  cambio:     Best=AutoARIMA  RMSE=0.567  MASE=1.023
  desemprego: Best=AutoETS    RMSE=0.298  MASE=0.867

Combined (BMA):
  RMSE (avg): 0.312 | MASE (avg): 0.874
  Coverage 95%: 93.2%

Duration: 47.3s | Cached: 0 steps | Workers: 8
```

---

## Exemplo Completo: 50 Series Macroeconomicas

Pipeline de producao para previsao mensal de 50 indicadores macroeconomicos
brasileiros:

```python
import pandas as pd
from forecastbox.pipeline import ForecastPipeline

# Carregar painel macro completo
data = pd.read_parquet("data/macro_br_50.parquet")
print(f"Series: {data.shape[1]}, Obs: {data.shape[0]}")
# Series: 50, Obs: 360

# Definir pipeline
pipeline = (
    ForecastPipeline()
    .add_data(
        source="dataframe",
        data=data,
        freq="MS"
    )
    .add_preprocessing(
        steps=["deseason", "detrend", "normalize"],
        deseason_method="stl",
        detrend_method="hp"
    )
    .add_models(
        models=["AutoARIMA", "AutoETS", "AutoVAR"],
        horizon=12,
        n_jobs=-1
    )
    .add_combination(
        method="bma",
        training_window=60
    )
    .add_evaluation(
        metrics=["rmse", "mase", "mape", "coverage"],
        cv_folds=5,
        cv_strategy="expanding"
    )
    .add_output(
        format="excel",
        path="output/macro_50_forecast.xlsx",
        include_plots=True,
        include_diagnostics=True
    )
    .build()
)

# Executar com checkpointing
results = pipeline.run(
    n_jobs=-1,
    cache=True,
    verbose=1,
    checkpoint_dir="checkpoints/macro_50/"
)

# Resumo
print(results.summary())

# Exportar metricas detalhadas
results.metrics.to_excel("output/macro_50_metrics.xlsx")
```

```text
ForecastPipeline - Run Complete
================================
Pipeline: macro_forecast_v2
Series: 50 | Models: 3 + BMA | Horizon: 12

Progress:
  [1/6] Data Ingestion ......... OK (0.3s)
  [2/6] Pre-Processing ......... OK (2.1s)
  [3/6] Modeling ............... OK (38.7s) [150 fits, 8 workers]
  [4/6] Combination ............ OK (4.2s)
  [5/6] Evaluation ............. OK (12.8s) [5-fold CV]
  [6/6] Output ................. OK (3.1s)

Best model distribution:
  AutoETS:   22/50 series (44%)
  AutoARIMA: 15/50 series (30%)
  BMA:       13/50 series (26%)

RMSE (avg): 0.387 | MASE (avg): 0.912
Duration: 61.2s
```

!!! tip "Boas praticas para producao"

    1. **Versione o YAML** — trate a config do pipeline como codigo
    2. **Use checkpointing** — para pipelines com mais de 20 series
    3. **Monitore apos deploy** — combine com `ForecastMonitor` para deteccao de drift
    4. **Registre experimentos** — use `Experiment` para comparar versoes do pipeline
    5. **Agende execucao** — use cron, Airflow ou Prefect para execucao periodica

---

## Parametros Completos

### `.add_data()`

| Parametro | Tipo | Descricao |
|:----------|:-----|:----------|
| `source` | `str` | Tipo de fonte: `"csv"`, `"parquet"`, `"sql"`, `"dataframe"` |
| `path` | `str` | Caminho do arquivo (para `csv` e `parquet`) |
| `data` | `DataFrame` | DataFrame direto (para `source="dataframe"`) |
| `target_cols` | `list[str]` | Colunas-alvo para previsao (default: todas numericas) |
| `date_col` | `str` | Nome da coluna de data (default: `"date"`) |
| `freq` | `str` | Frequencia dos dados: `"MS"`, `"QS"`, `"D"`, etc. |

### `.add_preprocessing()`

| Parametro | Tipo | Descricao |
|:----------|:-----|:----------|
| `steps` | `list[str]` | Etapas: `"deseason"`, `"detrend"`, `"normalize"`, `"outlier"` |
| `deseason_method` | `str` | Metodo: `"stl"`, `"x13"`, `"classical"` |
| `detrend_method` | `str` | Metodo: `"hp"`, `"linear"`, `"bk"` |
| `normalize_method` | `str` | Metodo: `"zscore"`, `"minmax"`, `"robust"` |

### `.add_models()`

| Parametro | Tipo | Descricao |
|:----------|:-----|:----------|
| `models` | `list[str]` | Modelos: `"AutoARIMA"`, `"AutoETS"`, `"AutoVAR"`, etc. |
| `horizon` | `int` | Horizonte de previsao (default: `12`) |
| `n_jobs` | `int` | Workers para paralelismo (default: `1`) |
| `model_params` | `dict` | Parametros customizados por modelo |

### `.add_combination()`

| Parametro | Tipo | Descricao |
|:----------|:-----|:----------|
| `method` | `str` | Metodo: `"simple"`, `"ols"`, `"bma"`, `"stacking"`, `"optimal"` |
| `training_window` | `int` | Janela de treinamento para pesos (default: `60`) |

### `.add_evaluation()`

| Parametro | Tipo | Descricao |
|:----------|:-----|:----------|
| `metrics` | `list[str]` | Metricas: `"rmse"`, `"mase"`, `"mape"`, `"coverage"` |
| `cv_folds` | `int` | Numero de folds para validacao cruzada |
| `cv_strategy` | `str` | Estrategia: `"expanding"`, `"rolling"`, `"fixed"` |

### `.add_output()`

| Parametro | Tipo | Descricao |
|:----------|:-----|:----------|
| `format` | `str` | Formato: `"csv"`, `"excel"`, `"parquet"`, `"sql"` |
| `path` | `str` | Caminho do arquivo de saida |
| `include_plots` | `bool` | Incluir graficos no output (default: `False`) |
| `include_diagnostics` | `bool` | Incluir diagnosticos no output (default: `False`) |
