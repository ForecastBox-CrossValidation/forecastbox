---
title: Experiment Tracking
description: Framework para registrar, comparar e reproduzir experimentos de previsao com log automatico de modelos, parametros e metricas.
---

# Experiment Tracking

Previsao e um processo **iterativo**: voce testa modelos, ajusta parametros, troca
metodos de combinacao, muda o horizonte. Sem um registro sistematico, e facil perder
o controle de o que funcionou e por que. O `Experiment` do forecastbox resolve isso
com logging automatico e comparacao estruturada.

---

## Conceito

O experiment tracking e similar ao **MLflow Tracking**, mas especializado
para previsao econometrica. Ele registra automaticamente:

- **Modelo** — tipo, hiperparametros, ordem selecionada
- **Dados** — series utilizadas, periodo, frequencia
- **Metricas** — RMSE, MASE, MAPE, cobertura para cada horizonte
- **Previsoes** — valores previstos e intervalos de confianca
- **Timestamp** — quando o experimento foi executado
- **Tags** — etiquetas customizadas para organizacao

!!! abstract "Key Takeaway"

    O `Experiment` funciona como um **caderno de laboratorio digital**
    para previsao. Cada run registra modelo, dados, metricas e previsoes.
    A comparacao entre runs permite escolher o melhor modelo com
    evidencia quantitativa, nao intuicao.

---

## Comparacao com MLflow Tracking

| Feature | **forecastbox Experiment** | MLflow Tracking |
|:--------|:-------------------------|:----------------|
| Foco | Previsao econometrica | ML generico |
| Log de forecast | :material-check: Nativo | :material-close: Manual |
| Metricas por horizonte | :material-check: Automatico | :material-close: Manual |
| Comparacao temporal | :material-check: Built-in | :material-close: Custom |
| Setup necessario | Nenhum (SQLite local) | Servidor MLflow |
| Integracao com combinacao | :material-check: Nativo | :material-close: |
| Integracao com CV expanding | :material-check: Nativo | :material-close: |

---

## Quick Start

```python
from forecastbox.experiment import Experiment
from forecastbox.auto import AutoARIMA, AutoETS

# Criar experimento
with Experiment("ipca_forecast_q1") as exp:
    # Run 1: AutoARIMA
    arima = AutoARIMA(max_p=5, max_q=5).fit(y_train)
    fc_arima = arima.predict(h=12)
    exp.log_run(
        name="AutoARIMA",
        model=arima,
        forecast=fc_arima,
        y_test=y_test
    )

    # Run 2: AutoETS
    ets = AutoETS(model="ZZZ").fit(y_train)
    fc_ets = ets.predict(h=12)
    exp.log_run(
        name="AutoETS",
        model=ets,
        forecast=fc_ets,
        y_test=y_test
    )

    # Comparar runs
    comparison = exp.compare(metric="rmse")
    print(comparison)
```

```text
Experiment: ipca_forecast_q1
=============================
Runs: 2 | Created: 2024-06-15 14:32:01

  Run          Model          RMSE    MASE    MAPE    AIC
  AutoARIMA    ARIMA(2,1,1)   0.231   0.845   3.21%   142.3
  AutoETS      ETS(M,A,M)     0.198   0.782   2.87%   138.7

Best: AutoETS (RMSE=0.198)
```

---

## Log Automatico vs Manual

### Log automatico

Quando voce passa o objeto `model`, o experiment extrai automaticamente
tipo, parametros, ordem selecionada e criterio de informacao:

```python
exp.log_run(
    name="AutoARIMA",
    model=arima,          # extrai parametros automaticamente
    forecast=fc_arima,
    y_test=y_test         # calcula metricas automaticamente
)
```

### Log manual

Para modelos externos ou metricas customizadas:

```python
exp.log_run(
    name="Prophet",
    params={"growth": "linear", "seasonality_mode": "multiplicative"},
    metrics={"rmse": 0.245, "mase": 0.891},
    forecast=fc_prophet,
    tags={"source": "external", "library": "prophet"}
)
```

---

## Armazenamento

O experiment tracking usa **SQLite local** por default, sem necessidade de
servidor ou infraestrutura adicional.

=== "SQLite (default)"

    ```python
    # Armazena em ~/.forecastbox/experiments.db
    with Experiment("meu_experimento") as exp:
        ...
    ```

=== "Caminho customizado"

    ```python
    with Experiment("meu_experimento", backend="sqlite",
                    db_path="experiments/tracking.db") as exp:
        ...
    ```

=== "Custom backend"

    ```python
    # Implementar interface ExperimentBackend
    class PostgresBackend(ExperimentBackend):
        def log_run(self, run_data: dict): ...
        def get_runs(self, experiment_name: str): ...

    with Experiment("meu_experimento",
                    backend=PostgresBackend(conn_string="...")) as exp:
        ...
    ```

---

## Comparacao de Experimentos

### Tabela comparativa

```python
# Comparar por metrica
comparison = exp.compare(metric="rmse")
print(comparison)

# Comparar multiplas metricas
comparison = exp.compare(metrics=["rmse", "mase", "mape"])
print(comparison)
```

```text
Experiment: pib_q1_2026
========================
  Run            RMSE    MASE    MAPE    Coverage
  AutoARIMA      0.412   0.923   4.12%   93.2%
  AutoETS        0.398   0.891   3.98%   94.1%
  AutoVAR        0.445   0.967   4.45%   91.8%
  BMA            0.378   0.856   3.67%   95.3%
  Stacking       0.385   0.869   3.78%   94.7%

Best by RMSE: BMA (0.378)
Best by MASE: BMA (0.856)
Best by Coverage: BMA (95.3%)
```

### Graficos de comparacao

```python
# Grafico de barras comparativo
exp.plot_comparison(metric="rmse")

# Grafico de previsoes sobrepostas
exp.plot_forecasts(include_actual=True)

# Grafico de metricas por horizonte
exp.plot_metrics_by_horizon(metric="rmse")
```

### Metricas por horizonte

Uma vantagem do experiment tracking especializado e a analise de como
a acuracia varia com o horizonte de previsao:

```python
horizon_analysis = exp.compare_by_horizon(metric="rmse")
print(horizon_analysis)
```

```text
RMSE by Horizon
================
  Horizon  AutoARIMA  AutoETS  BMA
  h=1      0.121      0.115    0.108
  h=3      0.234      0.218    0.201
  h=6      0.367      0.342    0.312
  h=12     0.512      0.498    0.478

Note: BMA consistently best across all horizons
```

---

## Exemplo Completo: Comparar Modelos para PIB

```python
import pandas as pd
from forecastbox.experiment import Experiment
from forecastbox.auto import AutoARIMA, AutoETS, AutoVAR
from forecastbox.combination import BMA, Stacking

# Dados
data = pd.read_parquet("data/pib_trimestral.parquet")
y = data["pib"]
y_train, y_test = y[:"2023-Q4"], y["2024-Q1":]

# Experimento completo
with Experiment("pib_q1_2026", tags={"target": "pib", "freq": "Q"}) as exp:

    # --- Modelos individuais ---
    models = {
        "AutoARIMA": AutoARIMA(max_p=4, max_q=4, ic="bic"),
        "AutoETS": AutoETS(model="ZZZ", ic="bic"),
        "AutoVAR": AutoVAR(max_lags=4, ic="aic"),
    }

    forecasts = {}
    for name, model in models.items():
        model.fit(y_train)
        fc = model.predict(h=4)
        forecasts[name] = fc
        exp.log_run(name=name, model=model, forecast=fc, y_test=y_test)

    # --- Combinacoes ---
    fc_matrix = pd.DataFrame(forecasts)

    # BMA
    bma = BMA().fit(fc_matrix, y_test)
    fc_bma = bma.predict(fc_matrix)
    exp.log_run(
        name="BMA",
        params={"method": "bma", "n_models": len(models)},
        forecast=fc_bma,
        y_test=y_test,
        tags={"type": "combination"}
    )

    # Stacking
    stack = Stacking(meta_learner="ridge").fit(fc_matrix, y_test)
    fc_stack = stack.predict(fc_matrix)
    exp.log_run(
        name="Stacking",
        params={"method": "stacking", "meta_learner": "ridge"},
        forecast=fc_stack,
        y_test=y_test,
        tags={"type": "combination"}
    )

    # --- Comparacao ---
    comparison = exp.compare(metrics=["rmse", "mase", "mape", "coverage"])
    print(comparison)

    # --- Graficos ---
    exp.plot_comparison(metric="rmse")
    exp.plot_forecasts(include_actual=True, title="PIB - Comparacao de Modelos")
    exp.plot_metrics_by_horizon(metric="rmse")

    # --- Exportar ---
    exp.to_dataframe().to_excel("output/experiment_pib.xlsx")
```

```text
Experiment: pib_q1_2026
========================
Runs: 5 | Tags: target=pib, freq=Q

  Run          Type          RMSE    MASE    MAPE     Coverage
  AutoARIMA    individual    0.412   0.923   4.12%    93.2%
  AutoETS      individual    0.398   0.891   3.98%    94.1%
  AutoVAR      individual    0.445   0.967   4.45%    91.8%
  BMA          combination   0.378   0.856   3.67%    95.3%
  Stacking     combination   0.385   0.869   3.78%    94.7%

Best overall: BMA
  RMSE: 0.378 (-5.0% vs AutoETS)
  MASE: 0.856 (-3.9% vs AutoETS)
  Coverage: 95.3% (target: 95%)
```

---

## Recuperar Experimentos Anteriores

```python
from forecastbox.experiment import Experiment

# Listar todos os experimentos
experiments = Experiment.list_all()
print(experiments)

# Carregar experimento anterior
exp = Experiment.load("pib_q1_2026")
print(exp.compare(metric="rmse"))

# Filtrar por tags
gdp_exps = Experiment.search(tags={"target": "pib"})
```

```text
Available Experiments
======================
  Name              Runs  Created             Tags
  pib_q1_2026       5     2024-06-15 14:32    target=pib, freq=Q
  ipca_mensal_v2    3     2024-06-10 09:15    target=ipca, freq=M
  selic_scenario    4     2024-06-08 16:45    target=selic, type=conditional
```

---

## Integracao com Pipeline e Monitor

O experiment tracking se integra naturalmente com os outros componentes:

```python
from forecastbox.pipeline import ForecastPipeline, ForecastMonitor
from forecastbox.experiment import Experiment

# 1. Registrar pipeline como experimento
with Experiment("macro_pipeline_v3") as exp:
    pipeline = ForecastPipeline.from_yaml("pipeline.yml")
    results = pipeline.run()
    exp.log_pipeline_run(pipeline, results)

# 2. Registrar re-estimacoes do monitor
monitor = ForecastMonitor(
    metrics=["rmse"],
    window=12,
    experiment="macro_pipeline_v3"  # log automatico de retrains
)
```

!!! tip "Workflow recomendado"

    1. **Experimente** — use `Experiment` para testar modelos e combinacoes
    2. **Escolha** — selecione o melhor modelo com base nas metricas
    3. **Deploy** — configure o `ForecastPipeline` com o modelo escolhido
    4. **Monitore** — use `ForecastMonitor` para vigiar em producao
    5. **Itere** — quando o monitor detectar drift, volte ao passo 1

---

## Parametros Completos

### `Experiment()`

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `name` | `str` | — | Nome do experimento (obrigatorio) |
| `backend` | `str` | `"sqlite"` | Backend de armazenamento |
| `db_path` | `str` | `"~/.forecastbox/experiments.db"` | Caminho do banco SQLite |
| `tags` | `dict` | `{}` | Tags para organizacao e busca |

### `exp.log_run()`

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `name` | `str` | — | Nome do run (obrigatorio) |
| `model` | `object` | `None` | Objeto do modelo (extrai parametros automaticamente) |
| `params` | `dict` | `None` | Parametros manuais (alternativa a `model`) |
| `forecast` | `Series/DataFrame` | `None` | Previsoes geradas |
| `y_test` | `Series` | `None` | Valores reais para calculo de metricas |
| `metrics` | `dict` | `None` | Metricas manuais (alternativa a `y_test`) |
| `tags` | `dict` | `{}` | Tags especificas do run |

### `exp.compare()`

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `metric` | `str` | `"rmse"` | Metrica para ordenacao |
| `metrics` | `list[str]` | `None` | Multiplas metricas para exibicao |
| `ascending` | `bool` | `True` | Ordenar do menor para o maior |
