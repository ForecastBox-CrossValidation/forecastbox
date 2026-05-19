---
title: "Pipeline de Producao"
description: "Tutorial pratico: definir pipeline via YAML, configurar modelos, avaliacao automatica, monitoramento e experiment tracking"
---

# Pipeline de Producao

!!! info "Sobre este tutorial"
    **Nivel**: :material-star: :material-star: Intermediario
    **Tempo estimado**: 60 minutos
    **Pre-requisitos**: Tutorial de [Fundamentos](fundamentals.md)
    **Dados**: Inflacao mensal (IPCA)

Em ambientes de producao, previsoes precisam ser **reprodutiveis, automatizadas
e monitoradas**. O modulo `pipeline` do forecastbox transforma seu workflow
de previsao ad hoc em um processo robusto e auditavel.

## O que voce vai aprender

- Definir um pipeline via configuracao YAML
- Configurar data source (parquet/CSV)
- Adicionar modelos e combinacao
- Configurar avaliacao automatica
- Executar pipeline e inspecionar resultados
- Adicionar monitoramento e alertas
- Automatizar com cron/scheduler
- Experiment tracking: comparar runs

---

## Etapa 1: Definir Pipeline via YAML

A forma mais limpa de configurar um pipeline e via arquivo YAML. Isso separa
**configuracao** de **codigo** e facilita versionamento com git.

```yaml
# pipeline_config.yaml
pipeline:
  name: "ipca_forecast"
  description: "Pipeline de previsao da inflacao IPCA"
  version: "1.0.0"

data:
  source: "data/ipca_mensal.parquet"
  target: "ipca"
  frequency: "MS"
  train_end: "2023-06-01"
  test_start: "2023-07-01"

models:
  - name: "auto_arima"
    params:
      max_p: 5
      max_q: 5
      seasonal: true
      m: 12
      ic: "aicc"
  - name: "auto_ets"
    params:
      restrict_models: true
      ic: "aicc"
  - name: "theta"
    params: {}
  - name: "naive"
    params:
      method: "seasonal"

combination:
  method: "bma"
  fallback: "mean"

evaluation:
  metrics: ["rmse", "mae", "mase", "crps"]
  tests: ["dm", "mcs"]
  cv:
    type: "expanding"
    initial_window: 120
    horizon: 12

output:
  format: "json"
  directory: "results/"
  save_forecasts: true
  save_report: true
```

!!! note "Por que YAML?"
    - **Versionavel**: `git diff` mostra exatamente o que mudou
    - **Reprodutivel**: qualquer pessoa pode replicar o pipeline
    - **Documentacao viva**: a configuracao e a documentacao
    - **Separacao de preocupacoes**: analistas ajustam YAML, engenheiros ajustam infra

---

## Etapa 2: Configurar Data Source

O pipeline aceita dados de diversas fontes. Vamos configurar com parquet:

```python
import pandas as pd
from forecastbox.datasets import load_inflation

# Preparar dados e salvar como parquet
data = load_inflation()
data.to_frame().to_parquet("data/ipca_mensal.parquet")

print(f"Dados salvos: data/ipca_mensal.parquet")
print(f"Serie: {data.name}")
print(f"Periodo: {data.index[0]:%Y-%m} a {data.index[-1]:%Y-%m}")
print(f"Observacoes: {len(data)}")
```

```text
Dados salvos: data/ipca_mensal.parquet
Serie: ipca
Periodo: 2004-01 a 2023-12
Observacoes: 240
```

```python
# Carregar pipeline a partir do YAML
from forecastbox.pipeline import ForecastPipeline

pipeline = ForecastPipeline.from_yaml("pipeline_config.yaml")

print(f"Pipeline: {pipeline.name}")
print(f"Target:   {pipeline.target}")
print(f"Modelos:  {pipeline.model_names}")
print(f"Combinacao: {pipeline.combination_method}")
```

```text
Pipeline: ipca_forecast
Target:   ipca
Modelos:  ['auto_arima', 'auto_ets', 'theta', 'naive']
Combinacao: bma
```

!!! tip "Fontes de dados suportadas"
    | Formato | Extensao | Exemplo |
    |---------|----------|---------|
    | Parquet | `.parquet` | `data/ipca.parquet` |
    | CSV | `.csv` | `data/ipca.csv` |
    | Excel | `.xlsx` | `data/ipca.xlsx` |
    | Pandas | em memoria | `pd.DataFrame(...)` |

---

## Etapa 3: Adicionar Modelos e Combinacao

Alem do YAML, voce pode configurar o pipeline programaticamente:

```python
from forecastbox.pipeline import ForecastPipeline
from forecastbox.datasets import load_inflation

# Pipeline programatico
data = load_inflation()

pipeline = ForecastPipeline(data=data, target="ipca")

# Adicionar modelos
pipeline.set_models(["auto_arima", "auto_ets", "theta", "naive"])

# Configurar combinacao
pipeline.set_combination(method="bma")

# Configurar avaliacao
pipeline.set_evaluation(tests=["dm", "mcs"])

print(f"Pipeline configurado:")
print(f"  Modelos:    {pipeline.model_names}")
print(f"  Combinacao: {pipeline.combination_method}")
print(f"  Testes:     {pipeline.evaluation_tests}")
```

```text
Pipeline configurado:
  Modelos:    ['auto_arima', 'auto_ets', 'theta', 'naive']
  Combinacao: bma
  Testes:     ['dm', 'mcs']
```

```python
# Adicionar steps customizados
import numpy as np

def remove_outliers(data, threshold=3.0):
    """Remove outliers via z-score."""
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores < threshold]

def log_transform(data):
    """Transformacao logaritmica."""
    return np.log(data)

pipeline.add_step("preprocess", remove_outliers, threshold=3.0)
pipeline.add_step("preprocess", log_transform)

print(f"Steps: {pipeline.steps}")
```

```text
Steps: ['preprocess: remove_outliers', 'preprocess: log_transform',
        'fit', 'forecast', 'combine', 'evaluate']
```

---

## Etapa 4: Configurar Avaliacao Automatica

A avaliacao automatica calcula metricas e testes estatisticos a cada execucao:

```python
# Configuracao detalhada de avaliacao
pipeline.set_evaluation(
    tests=["dm", "mcs", "mz"],
    metrics=["rmse", "mae", "mase", "crps"],
    cv_type="expanding",
    cv_initial=120,
    cv_horizon=12,
)

print("Avaliacao configurada:")
print(f"  Metricas: rmse, mae, mase, crps")
print(f"  Testes:   Diebold-Mariano, MCS, Mincer-Zarnowitz")
print(f"  CV:       Expanding window (T0=120, h=12)")
```

```text
Avaliacao configurada:
  Metricas: rmse, mae, mase, crps
  Testes:   Diebold-Mariano, MCS, Mincer-Zarnowitz
  CV:       Expanding window (T0=120, h=12)
```

---

## Etapa 5: Executar Pipeline e Inspecionar Resultados

Agora vamos executar o pipeline completo e inspecionar os resultados:

```python
# Executar pipeline
results = pipeline.run()

print(f"Pipeline executado com sucesso!")
print(f"Tempo total: {sum(results.execution_time.values()):.1f}s")
print(f"\nTempo por etapa:")
for step, time in results.execution_time.items():
    print(f"  {step:<20} {time:.2f}s")
```

```text
Pipeline executado com sucesso!
Tempo total: 12.3s

Tempo por etapa:
  preprocess           0.05s
  fit                  8.42s
  forecast             0.31s
  combine              0.18s
  evaluate             3.34s
```

```python
# Ranking de modelos
print("\nRanking de Modelos (RMSE):")
print(results.evaluation.sort_values("rmse"))
```

```text
Ranking de Modelos (RMSE):
           rmse    mae   mase   crps
BMA       0.152  0.118  0.812  0.098
auto_ets  0.165  0.128  0.881  0.105
auto_arima 0.172  0.135  0.929  0.112
theta     0.189  0.148  1.019  0.125
naive     0.231  0.182  1.253  0.158
```

```python
# Melhor modelo e MCS
print(f"\nMelhor modelo: {results.best_model()}")
print(f"\nModel Confidence Set (alpha=0.10):")
print(f"  Modelos superiores: {results.metadata['mcs_models']}")

# Resumo completo
print(results.summary())
```

```text
Melhor modelo: BMA

Model Confidence Set (alpha=0.10):
  Modelos superiores: ['BMA', 'auto_ets', 'auto_arima']

=== Pipeline Summary: ipca_forecast ===
Target: ipca | Horizon: 12 | Models: 4 + BMA
Best: BMA (RMSE: 0.152)
MCS: {BMA, auto_ets, auto_arima}
Mincer-Zarnowitz: BMA is unbiased (p=0.342)
CV RMSE: BMA=0.158, auto_ets=0.171, auto_arima=0.178
```

```python
# Acessar previsoes individuais
for name, fc in results.forecasts.items():
    print(f"{name:>12}: h=1 -> {fc.point[0]:.3f}  "
          f"[{fc.lower_80[0]:.3f}, {fc.upper_80[0]:.3f}]")
```

```text
  auto_arima: h=1 -> 0.452  [0.312, 0.592]
    auto_ets: h=1 -> 0.438  [0.298, 0.578]
       theta: h=1 -> 0.465  [0.305, 0.625]
       naive: h=1 -> 0.510  [0.310, 0.710]
         BMA: h=1 -> 0.445  [0.315, 0.575]
```

!!! example "Try it yourself"
    Adicione um quinto modelo (`var`) ao pipeline e verifique se ele
    entra no Model Confidence Set:

    ```python
    pipeline_5 = ForecastPipeline(data=data, target="ipca")
    pipeline_5.set_models(["auto_arima", "auto_ets", "theta", "naive", "var"])
    pipeline_5.set_combination(method="bma")
    pipeline_5.set_evaluation(tests=["dm", "mcs"])

    results_5 = pipeline_5.run()
    print(f"MCS com 5 modelos: {results_5.metadata['mcs_models']}")
    ```

---

## Etapa 6: Adicionar Monitoramento

O `ForecastMonitor` rastreia a acuracia ao longo do tempo e detecta degradacao:

```python
from forecastbox.pipeline import ForecastMonitor

# Configurar monitor
monitor = ForecastMonitor(
    forecasts=results.forecasts,
    actual=data[-12:],  # ultimos 12 meses realizados
)

# Avaliar acuracia atual
report = monitor.evaluate(metrics=("rmse", "mae"))
print("Monitoramento de Acuracia:")
print(report)
```

```text
Monitoramento de Acuracia:
           rmse    mae  status
auto_arima 0.172  0.135  OK
auto_ets   0.165  0.128  OK
theta      0.189  0.148  WARNING
naive      0.231  0.182  DEGRADED
BMA        0.152  0.118  OK
```

```python
# Detectar degradacao
degraded = monitor.detect_degradation(threshold=0.05)

if degraded:
    print(f"\n⚠ Modelos com degradacao detectada:")
    for model_name in degraded:
        print(f"  - {model_name}")
else:
    print("\nTodos os modelos dentro do limiar.")
```

```text
⚠ Modelos com degradacao detectada:
  - naive
```

```python
# Configurar alertas automaticos
from forecastbox.pipeline import AlertSystem, AlertRule

alerts = AlertSystem()

# Alerta se RMSE ultrapassar limiar
alerts.add_rule(AlertRule(
    name="rmse_threshold",
    metric="rmse",
    threshold=0.25,
    condition="above",
    severity="warning",
))

# Alerta se vies for significativo
alerts.add_rule(AlertRule(
    name="bias_check",
    metric="bias",
    threshold=0.05,
    condition="above",
    severity="critical",
))

# Verificar alertas
triggered = alerts.check(
    forecasts=results.forecasts,
    actual=data[-12:],
)

print(f"\nAlertas disparados: {len(triggered)}")
for alert in triggered:
    print(f"  [{alert.severity.upper()}] {alert.name}: "
          f"{alert.model} ({alert.metric}={alert.value:.3f})")
```

```text
Alertas disparados: 1
  [WARNING] rmse_threshold: naive (rmse=0.231)
```

---

## Etapa 7: Automatizar com Cron/Scheduler

Para previsoes recorrentes, use o `RecurringForecast`:

```python
from forecastbox.pipeline import RecurringForecast

# Configurar previsao recorrente
recurring = RecurringForecast(
    pipeline=pipeline,
    frequency="monthly",
)

# Gerar cron expression
cron_expr = recurring.schedule(cron_schedule="0 8 1 * *")  # 1o dia, 8h

print(f"Pipeline agendado:")
print(f"  Cron: {cron_expr.expression}")
print(f"  Descricao: Executa no dia 1 de cada mes as 08:00")
print(f"  Proximo run: {cron_expr.next_run}")
```

```text
Pipeline agendado:
  Cron: 0 8 1 * *
  Descricao: Executa no dia 1 de cada mes as 08:00
  Proximo run: 2024-04-01 08:00:00
```

Alternativamente, para integrar com um scheduler externo (cron do sistema,
Airflow, etc.), use o CLI:

```bash
# Executar pipeline via CLI
forecastbox run pipeline_config.yaml

# Executar e salvar resultados
forecastbox run pipeline_config.yaml --output results/2024-03/

# Validar configuracao sem executar
forecastbox validate pipeline_config.yaml
```

```text
$ forecastbox run pipeline_config.yaml --output results/2024-03/
[2024-03-01 08:00:01] Loading pipeline: ipca_forecast v1.0.0
[2024-03-01 08:00:01] Loading data: data/ipca_mensal.parquet
[2024-03-01 08:00:02] Fitting 4 models...
[2024-03-01 08:00:10] Combining with BMA...
[2024-03-01 08:00:10] Evaluating...
[2024-03-01 08:00:14] Results saved to results/2024-03/
[2024-03-01 08:00:14] Done! Best model: BMA (RMSE: 0.152)
```

!!! tip "Integracao com crontab"
    ```bash
    # Adicionar ao crontab do sistema
    crontab -e
    # 1o dia de cada mes, 8h
    0 8 1 * * cd /path/to/project && forecastbox run pipeline_config.yaml
    ```

!!! example "Try it yourself"
    Valide o arquivo YAML programaticamente e liste os passos do
    pipeline sem executa-lo:

    ```python
    from forecastbox.pipeline import ForecastPipeline

    pipeline_check = ForecastPipeline.from_yaml("pipeline_config.yaml")
    print(f"Pipeline valido: {pipeline_check.validate()}")
    print(f"Passos: {pipeline_check.steps}")
    print(f"Modelos: {pipeline_check.model_names}")
    print(f"Metricas: {pipeline_check.evaluation_metrics}")
    ```

---

## Etapa 8: Experiment Tracking -- Comparar Runs

O experiment tracking permite comparar diferentes execucoes do pipeline,
facilitando experimentacao e auditoria:

```python
from forecastbox.pipeline import ForecastPipeline

# Run 1: configuracao original
pipeline_v1 = ForecastPipeline.from_yaml("pipeline_config.yaml")
results_v1 = pipeline_v1.run()

# Run 2: trocar combinacao para media simples
pipeline_v2 = ForecastPipeline.from_yaml("pipeline_config.yaml")
pipeline_v2.set_combination(method="mean")
results_v2 = pipeline_v2.run()

# Run 3: adicionar mais modelos
pipeline_v3 = ForecastPipeline.from_yaml("pipeline_config.yaml")
pipeline_v3.set_models(["auto_arima", "auto_ets", "theta", "naive", "var"])
results_v3 = pipeline_v3.run()

# Comparar runs
print("Experiment Tracking -- Comparacao de Runs")
print(f"{'Run':<15} {'Combinacao':<12} {'#Modelos':>10} {'RMSE Best':>12}")
print("=" * 55)
runs = [
    ("v1_bma", "BMA", results_v1),
    ("v2_mean", "Media", results_v2),
    ("v3_extended", "BMA", results_v3),
]
for name, comb, res in runs:
    best = res.best_model()
    best_rmse = res.evaluation.loc[best, "rmse"]
    n_models = len(res.forecasts) - 1  # excluir combinacao
    print(f"{name:<15} {comb:<12} {n_models:>10} {best_rmse:>12.4f}")
```

```text
Experiment Tracking -- Comparacao de Runs
Run             Combinacao    #Modelos    RMSE Best
=======================================================
v1_bma          BMA                  4       0.1520
v2_mean         Media                4       0.1615
v3_extended     BMA                  5       0.1485
```

```python
# Salvar resultados para auditoria
results_v1.report(format="json", output="results/v1_bma.json")
results_v2.report(format="json", output="results/v2_mean.json")
results_v3.report(format="json", output="results/v3_extended.json")

# Exportar relatorio HTML
results_v3.report(format="html", output="results/report_v3.html")
print("Relatorio HTML salvo em results/report_v3.html")
```

```text
Relatorio HTML salvo em results/report_v3.html
```

---

## Resumo

| Componente | Funcao | Beneficio |
|------------|--------|-----------|
| **YAML Config** | Definir pipeline declarativamente | Reprodutibilidade, versionamento |
| **ForecastPipeline** | Orquestrar fit/forecast/combine/evaluate | Automacao end-to-end |
| **ForecastMonitor** | Rastrear acuracia ao longo do tempo | Detectar degradacao |
| **AlertSystem** | Regras de alerta automaticas | Proatividade |
| **RecurringForecast** | Agendar execucoes | Producao |
| **Experiment Tracking** | Comparar configuracoes | Auditoria |

## Proximos passos

- :material-map-marker-path: **[Workflow Completo](complete-workflow.md)** -- Tutorial end-to-end integrando todos os modulos
- :material-arrow-decision: **[Cenarios](scenarios.md)** -- Previsao condicional e stress testing
- :material-pulse: **[Nowcasting](nowcasting.md)** -- Previsao em tempo real
- :material-chart-bar: **[Graficos de Pipeline](../visualization/pipeline-plots.md)** -- Visualize DAGs, monitoramento e drift
- :material-book-open-variant: **[User Guide: Pipeline](../user-guide/pipeline/pipeline.md)** -- Referencia completa
- :material-school: **[Theory: Avaliacao](../theory/evaluation-theory.md)** -- Fundamentos teoricos de monitoramento e drift
