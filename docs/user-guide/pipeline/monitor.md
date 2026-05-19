---
title: ForecastMonitor
description: Monitoramento pos-deploy de modelos de previsao - deteccao de drift, alertas automaticos e re-estimacao adaptativa.
---

# ForecastMonitor

Modelos de previsao **degradam com o tempo**. A estrutura da serie muda, relacoes
entre variaveis se alteram, choques estruturais acontecem. O `ForecastMonitor`
vigia modelos em producao e detecta quando e hora de re-estimar.

---

## Por que Monitorar?

Um modelo que tinha RMSE de 0.30 no backtest pode ter RMSE de 0.80 seis meses
depois. Sem monitoramento, voce so descobre quando o relatorio ja esta errado.

!!! abstract "Key Takeaway"

    O `ForecastMonitor` calcula metricas de acuracia em **janela movel**,
    detecta **drift** nos dados e no modelo, e dispara **alertas** quando
    a performance cruza thresholds pre-definidos. Opcionalmente, pode
    iniciar **re-estimacao automatica** quando um trigger e ativado.

---

## Tipos de Drift

O monitoramento cobre dois tipos fundamentais de drift:

### Drift de Dados (Covariate Shift)

A distribuicao dos dados de entrada muda em relacao ao periodo de treinamento.

$$
P_{\text{train}}(\mathbf{X}) \neq P_{\text{prod}}(\mathbf{X})
$$

Detectado por:

- **Kolmogorov-Smirnov test** — compara distribuicoes empiricas
- **PSI (Population Stability Index)** — monitora estabilidade de distribuicoes
- **Media e variancia movel** — detecta mudancas de nivel e volatilidade

### Drift de Conceito (Concept Drift)

A relacao entre inputs e target muda, mesmo que os inputs permaneçam estaveis.

$$
P_{\text{train}}(Y | \mathbf{X}) \neq P_{\text{prod}}(Y | \mathbf{X})
$$

Detectado por:

- **RMSE rolling** — monitora acuracia em janela movel
- **Vies acumulado** — detecta erros sistematicos persistentes
- **CUSUM** — detecta mudancas de nivel no erro de previsao

---

## Quick Start

```python
from forecastbox.pipeline import ForecastMonitor

# Criar monitor
monitor = ForecastMonitor(
    metrics=["rmse", "bias", "coverage"],
    window=12,
    alert_threshold={"rmse": 1.5, "bias": 0.10}
)

# Registrar modelo e dados historicos
monitor.register_model(model, y_train=y_train)

# A cada novo dado observado, atualizar
monitor.update(y_true=0.82, y_pred=0.91, timestamp="2024-07-01")
monitor.update(y_true=0.75, y_pred=0.88, timestamp="2024-08-01")

# Checar status
status = monitor.check()
print(status)
```

```text
ForecastMonitor Status (2024-08-01)
====================================
Model: AutoARIMA | Window: 12 | Updates: 2

Metrics (rolling 12):
  RMSE:      0.112  [OK]     threshold: 0.450
  Bias:      0.060  [OK]     threshold: 0.100
  Coverage:  95.0%  [OK]     target: 95.0%

Drift Detection:
  Data drift:    NOT DETECTED (KS p=0.82)
  Concept drift: NOT DETECTED (CUSUM within bounds)

Status: HEALTHY
```

---

## Metricas Monitoradas

O monitor calcula metricas em janela movel (rolling window) para detectar
degradacao gradual:

| Metrica | Formula | Interpretacao |
|:--------|:--------|:-------------|
| **RMSE rolling** | $\text{RMSE}_t = \sqrt{\frac{1}{w}\sum_{i=t-w+1}^{t}(y_i - \hat{y}_i)^2}$ | Acuracia geral na janela recente |
| **Vies acumulado** | $\text{Bias}_t = \frac{1}{w}\sum_{i=t-w+1}^{t}(y_i - \hat{y}_i)$ | Erro sistematico persistente |
| **Cobertura** | $\text{Cov}_t = \frac{1}{w}\sum_{i=t-w+1}^{t}\mathbb{1}(y_i \in \text{IC}_i)$ | Proporcao de observacoes dentro do intervalo |
| **MAE rolling** | $\text{MAE}_t = \frac{1}{w}\sum_{i=t-w+1}^{t}|y_i - \hat{y}_i|$ | Erro absoluto medio recente |

---

## Alertas Automaticos

O sistema de alertas usa tres niveis baseados na razao entre a metrica
atual e o threshold:

```python
monitor = ForecastMonitor(
    metrics=["rmse", "bias", "coverage"],
    window=12,
    alert_threshold={
        "rmse": 1.5,      # 1.5x o RMSE do backtest
        "bias": 0.10,      # vies maximo tolerado
        "coverage": 0.85   # cobertura minima
    }
)
```

| Nivel | Condicao | Acao |
|:------|:---------|:-----|
| :material-check-circle:{ style="color: #4CAF50" } **HEALTHY** | Todas metricas OK | Nenhuma |
| :material-alert:{ style="color: #FF9800" } **WARNING** | Metrica entre 80-100% do threshold | Notificacao |
| :material-alert-octagon:{ style="color: #F44336" } **CRITICAL** | Metrica excede threshold | Re-estimacao |

### Notificacoes

```python
# Configurar notificacoes
monitor.set_alerts(
    on_warning=lambda msg: print(f"[WARN] {msg}"),
    on_critical=lambda msg: send_email(msg),  # funcao custom
)
```

!!! warning "Thresholds adequados"

    Thresholds muito baixos geram alertas excessivos (falsos positivos).
    Thresholds muito altos deixam o modelo degradar antes de reagir.
    Use o RMSE do backtest como referencia e calibre com 1.5x a 2.0x
    como ponto de partida.

---

## Deteccao de Drift

### Drift de dados

```python
# Verificar drift nos dados de entrada
drift_report = monitor.check_data_drift(X_new)
print(drift_report)
```

```text
Data Drift Report
==================
  Feature     KS-stat   p-value   PSI     Status
  ipca        0.087     0.823     0.032   OK
  selic       0.234     0.012     0.187   DRIFT DETECTED
  cambio      0.156     0.098     0.089   WARNING
  desemprego  0.045     0.967     0.011   OK

Summary: 1 feature with drift, 1 warning
```

### Drift de conceito

```python
# Verificar drift no modelo (relacao input-output)
concept_report = monitor.check_concept_drift()
print(concept_report)
```

```text
Concept Drift Report
=====================
  Test        Statistic   p-value   Status
  CUSUM       2.34        0.019     DRIFT DETECTED
  Page-Hinkley 0.87       0.384     OK

  RMSE trend: +0.023/month (increasing)
  Bias trend:  +0.008/month (increasing)

Recommendation: RE-ESTIMATE MODEL
```

---

## Re-estimacao Automatica

Quando o monitor detecta drift critico, pode disparar re-estimacao automatica:

```python
from forecastbox.pipeline import ForecastPipeline, ForecastMonitor

# Pipeline original
pipeline = ForecastPipeline.from_yaml("pipeline.yml")

# Monitor com retrain automatico
monitor = ForecastMonitor(
    metrics=["rmse", "bias"],
    window=12,
    alert_threshold={"rmse": 1.5, "bias": 0.10},
    auto_retrain=True,
    retrain_pipeline=pipeline
)

# Loop de producao
for timestamp, y_true, y_pred in production_stream:
    monitor.update(y_true=y_true, y_pred=y_pred, timestamp=timestamp)
    status = monitor.check()

    if status.retrained:
        print(f"[{timestamp}] Model re-estimated! New RMSE: {status.new_rmse:.3f}")
```

```text
[2024-06-01] Status: HEALTHY (RMSE=0.231)
[2024-07-01] Status: HEALTHY (RMSE=0.245)
[2024-08-01] Status: WARNING (RMSE=0.312)
[2024-09-01] Status: CRITICAL (RMSE=0.398)
[2024-09-01] Model re-estimated! New RMSE: 0.218
[2024-10-01] Status: HEALTHY (RMSE=0.224)
```

!!! tip "Retrain controlado"

    Configure `min_retrain_interval` para evitar re-estimacoes muito
    frequentes. Um intervalo minimo de 30 dias e uma boa pratica para
    series mensais.

---

## Integracao com Dashboards

O monitor pode exportar metricas para backends de monitoramento:

=== "Prometheus"

    ```python
    monitor = ForecastMonitor(
        metrics=["rmse", "bias", "coverage"],
        window=12,
        backend="prometheus",
        backend_config={
            "gateway": "http://localhost:9091",
            "job": "forecast_monitor"
        }
    )
    ```

=== "Grafana (via CSV)"

    ```python
    # Exportar historico de metricas para CSV
    monitor.export_metrics(
        path="metrics/monitor_history.csv",
        format="csv"
    )
    # Configurar Grafana para ler o CSV como datasource
    ```

=== "Custom Backend"

    ```python
    # Backend customizado
    def my_backend(metrics: dict, timestamp: str):
        # Enviar para banco de dados, API, etc.
        db.insert("forecast_metrics", metrics, timestamp)

    monitor = ForecastMonitor(
        metrics=["rmse", "bias"],
        window=12,
        backend="custom",
        backend_config={"callback": my_backend}
    )
    ```

---

## Exemplo Completo: Monitorar Inflacao Mensal

Cenario: modelo de previsao de IPCA mensal em producao, monitorado
continuamente com alertas e re-estimacao automatica.

```python
import pandas as pd
from forecastbox.auto import AutoARIMA
from forecastbox.pipeline import ForecastPipeline, ForecastMonitor

# --- Setup inicial ---
data = pd.read_parquet("data/ipca_mensal.parquet")
y_train = data.loc[:"2023-12", "ipca"]
y_test = data.loc["2024-01":, "ipca"]

# Estimar modelo inicial
model = AutoARIMA(max_p=5, max_q=5, ic="bic").fit(y_train)
forecasts = model.predict(h=len(y_test))

# --- Configurar monitor ---
monitor = ForecastMonitor(
    metrics=["rmse", "bias", "coverage"],
    window=6,                              # janela de 6 meses
    alert_threshold={
        "rmse": 0.45,                      # 1.5x RMSE do backtest (0.30)
        "bias": 0.10,                      # vies maximo de 10pp
        "coverage": 0.85                   # cobertura minima 85%
    },
    auto_retrain=True,
    retrain_pipeline=ForecastPipeline.from_yaml("pipeline_ipca.yml"),
    min_retrain_interval=90                # minimo 90 dias entre retrains
)

# Registrar modelo
monitor.register_model(model, y_train=y_train, backtest_rmse=0.30)

# --- Loop de producao ---
for i, (date, y_true) in enumerate(y_test.items()):
    y_pred = forecasts.iloc[i]
    monitor.update(y_true=y_true, y_pred=y_pred, timestamp=date)

    status = monitor.check()
    print(f"[{date.strftime('%Y-%m')}] {status.level}: "
          f"RMSE={status.metrics['rmse']:.3f}, "
          f"Bias={status.metrics['bias']:.3f}")

# --- Relatorio final ---
report = monitor.report()
print(report)
```

```text
[2024-01] HEALTHY: RMSE=0.198, Bias=0.021
[2024-02] HEALTHY: RMSE=0.215, Bias=0.034
[2024-03] HEALTHY: RMSE=0.267, Bias=0.045
[2024-04] WARNING: RMSE=0.312, Bias=0.067
[2024-05] WARNING: RMSE=0.389, Bias=0.082
[2024-06] CRITICAL: RMSE=0.467, Bias=0.112
  -> Auto-retrain triggered (RMSE > 0.45)
  -> New model: AutoARIMA(2,1,1) | RMSE=0.201
[2024-07] HEALTHY: RMSE=0.213, Bias=0.018

Monitor Report
===============
Period: 2024-01 to 2024-07
Total updates: 7
Retrains: 1 (at 2024-06-01)
Drift events: 1 concept drift
Avg RMSE: 0.294
```

---

## Parametros Completos

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `metrics` | `list[str]` | `["rmse"]` | Metricas a monitorar |
| `window` | `int` | `12` | Tamanho da janela movel |
| `alert_threshold` | `dict` | `{}` | Thresholds por metrica |
| `auto_retrain` | `bool` | `False` | Habilitar re-estimacao automatica |
| `retrain_pipeline` | `ForecastPipeline` | `None` | Pipeline para re-estimacao |
| `min_retrain_interval` | `int` | `30` | Dias minimos entre re-estimacoes |
| `backend` | `str` | `None` | Backend de metricas: `"prometheus"`, `"csv"`, `"custom"` |
| `backend_config` | `dict` | `{}` | Configuracao do backend |

---

## Boas Praticas

!!! tip "Monitoramento em producao"

    1. **Comece simples** — monitore RMSE rolling e vies antes de adicionar drift detection
    2. **Calibre thresholds** — use 1.5x o RMSE do backtest como ponto de partida
    3. **Evite over-retrain** — re-estimacao frequente pode introduzir instabilidade
    4. **Registre tudo** — combine com `Experiment` para historico de re-estimacoes
    5. **Monitore o monitor** — verifique se os alertas estao sendo gerados corretamente
    6. **Janela adequada** — para series mensais, use `window=6` a `12`; para diarias, `window=30` a `60`
