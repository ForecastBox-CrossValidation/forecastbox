---
title: FAQ Avancado
description: Perguntas avancadas sobre customizacao, performance, integracao e uso avancado do forecastbox
---

# FAQ Avancado

Perguntas avancadas sobre customizacao, performance e uso avancado do **forecastbox**.

---

## Como criar um modelo customizado para o ModelZoo?

Para registrar um modelo customizado, implemente a interface `BaseForecaster`:

```python
import numpy as np
import forecastbox as fb

class MeuModelo(fb.BaseForecaster):
    """Modelo customizado: media movel ponderada exponencialmente."""

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self._last_value = None
        self._fitted = False

    def fit(self, y, X=None):
        """Ajusta o modelo aos dados historicos."""
        self.y_train = np.asarray(y)
        weights = np.array([(1 - self.alpha) ** i
                           for i in range(len(y))])[::-1]
        weights /= weights.sum()
        self._last_value = np.dot(weights, self.y_train)
        self._fitted = True
        return self

    def forecast(self, h=1, X=None):
        """Gera previsoes h passos a frente."""
        if not self._fitted:
            raise ValueError("Modelo nao ajustado. Chame fit() primeiro.")
        return fb.ForecastResult(
            mean=np.full(h, self._last_value),
            model_name="MeuModelo"
        )

    def get_info(self):
        """Metadados do modelo."""
        return {"name": "MeuModelo", "alpha": self.alpha}

# Registrar no ModelZoo
fb.ModelZoo.register("ewma_custom", MeuModelo)

# Usar no AutoSelect
selector = fb.AutoSelect(
    models=["arima", "ets", "ewma_custom"],
    criterion="aic"
)
selector.fit(y)
best = selector.best_model()
```

!!! tip "Interface minima"
    Os metodos obrigatorios sao `fit(y, X=None)` e `forecast(h, X=None)`. O metodo `get_info()` e opcional mas recomendado.

---

## Qual metodo de combinacao usar quando tenho >20 modelos?

Com muitos modelos, metodos irrestritos (OLS, Stacking) tendem a **overfitting**. Recomendacoes:

| N. Modelos | Metodo Recomendado | Razao |
|-----------|-------------------|-------|
| 2-5 | OLS, Stacking, BMA | Poucos parametros, estimacao estavel |
| 5-10 | BMA, InverseMSE | BMA penaliza modelos ruins automaticamente |
| 10-20 | InverseMSE, Simple | Pesos simples sao robustos |
| >20 | Simple + MCS pre-selecao | Filtre primeiro, combine depois |

A estrategia recomendada para >20 modelos:

```python
import forecastbox as fb

# Passo 1: Filtrar com MCS
mcs = fb.ModelConfidenceSet(alpha=0.10)
mcs.fit(forecasts_dict, y_actual)
survivors = mcs.surviving_models()  # Tipicamente 5-10 modelos

# Passo 2: Combinar apenas os sobreviventes
combiner = fb.BMA()
combiner.fit(
    {k: v for k, v in forecasts_dict.items() if k in survivors},
    y_actual
)
combined = combiner.forecast(h=12)
```

!!! info "Regra pratica"
    A combinacao de previsoes sofre de uma **maldicao da dimensionalidade**: com $K$ modelos e $T$ observacoes, metodos irrestritos precisam de $T \gg K$ para estabilidade.

---

## Como interpretar pesos negativos na combinacao OLS?

Pesos negativos na combinacao OLS significam que o modelo esta sendo usado como **hedge** — sua previsao e subtraida para corrigir vieses dos outros modelos.

Causas comuns:

1. **Multicolinearidade**: previsoes muito correlacionadas entre si
2. **Overfitting**: poucos dados para muitos modelos
3. **Modelo redundante**: informacao ja capturada por outros

```python
import forecastbox as fb

combiner = fb.OLSCombination()
combiner.fit(forecasts_dict, y_actual)

# Verificar pesos
print(combiner.weights_)
# {'arima': 0.45, 'ets': 0.72, 'theta': -0.17}
#                                        ^^^^^ peso negativo

# Diagnostico: correlacao entre previsoes
print(combiner.correlation_matrix_)
```

**Solucoes:**

=== "Restringir pesos"

    ```python
    combiner = fb.OLSCombination(
        non_negative=True  # Restricao w >= 0
    )
    ```

=== "Usar BMA"

    ```python
    # BMA garante pesos em [0, 1] que somam 1
    combiner = fb.BMA()
    combiner.fit(forecasts_dict, y_actual)
    ```

=== "Usar InverseMSE"

    ```python
    # Pesos baseados em performance individual
    combiner = fb.InverseMSE()
    combiner.fit(forecasts_dict, y_actual)
    ```

---

## MCS esta demorando muito. Como acelerar?

O Model Confidence Set tem complexidade $O(K^2 \cdot B)$ onde $K$ e o numero de modelos e $B$ o numero de bootstrap replications. Estrategias para acelerar:

```python
import forecastbox as fb

# 1. Reduzir bootstrap replications (padrao: 5000)
mcs = fb.ModelConfidenceSet(
    alpha=0.10,
    n_boot=1000  # Menos replications = mais rapido
)

# 2. Usar estatistica Range em vez de Semi-quadratic
mcs = fb.ModelConfidenceSet(
    alpha=0.10,
    statistic="range"  # Mais rapido que "semi_quadratic"
)

# 3. Paralelizar
mcs = fb.ModelConfidenceSet(
    alpha=0.10,
    n_jobs=-1  # Usar todos os cores
)

# 4. Pre-filtrar modelos obviamente ruins
# Remover modelos com RMSE > 2x o melhor
from forecastbox.evaluation import rmse
scores = {k: rmse(y_actual, v) for k, v in forecasts_dict.items()}
best_rmse = min(scores.values())
filtered = {k: v for k, v in forecasts_dict.items()
            if scores[k] < 2 * best_rmse}

mcs.fit(filtered, y_actual)
```

| Parametro | Padrao | Rapido | Ultra-rapido |
|-----------|--------|--------|--------------|
| `n_boot` | 5000 | 1000 | 500 |
| `statistic` | `"semi_quadratic"` | `"range"` | `"range"` |
| `n_jobs` | 1 | -1 | -1 |
| Tempo (20 modelos) | ~120s | ~15s | ~5s |

!!! warning "Trade-off"
    Reduzir `n_boot` abaixo de 1000 pode tornar os p-valores instáveis. Para publicacoes academicas, use pelo menos `n_boot=5000`.

---

## Como fazer previsao condicional com restricoes soft?

Restricoes **hard** fixam o valor exato da variavel condicionante. Restricoes **soft** permitem desvios em torno de um valor central:

```python
import forecastbox as fb

model = fb.AutoVAR()
model.fit(df[["pib", "inflacao", "selic"]])

# Restricao hard: selic = 10.5% no proximo trimestre
hard = fb.ConditionalForecast(
    model=model,
    conditions={"selic": [10.5, 10.5, 10.5, 10.5]},
    h=4
)

# Restricao soft: selic em torno de 10.5% com desvio de 0.5pp
soft = fb.ConditionalForecast(
    model=model,
    conditions={"selic": [10.5, 10.5, 10.5, 10.5]},
    conditions_std={"selic": [0.5, 0.5, 0.5, 0.5]},
    h=4,
    method="soft"
)

# Comparar
print("Hard:", hard.mean["pib"])
print("Soft:", soft.mean["pib"])
print("Soft IC:", soft.lower["pib"], soft.upper["pib"])
```

A restricao soft e implementada como prior normal no filtro de Kalman:

$$
s_{t|t-1}^{cond} \sim \mathcal{N}(\bar{s}_t, \sigma_s^2)
$$

onde $\bar{s}_t$ e o valor da restricao e $\sigma_s^2$ controla a rigidez.

!!! tip "Quando usar soft vs hard"
    - **Hard**: quando a variavel e controlada pelo policymaker (meta Selic, teto de gastos)
    - **Soft**: quando ha incerteza no cenario (projecoes de cambio, preco de commodities)

---

## MIDAS vs Bridge: quando preferir cada?

Ambos lidam com dados de frequencias mistas, mas com abordagens distintas:

| Aspecto | Bridge Equations | MIDAS |
|---------|-----------------|-------|
| **Agregacao** | Agrega indicadores para freq. baixa | Usa dados na freq. original |
| **Flexibilidade** | Mais simples, linear | Permite ponderacao nao-linear |
| **N. indicadores** | Funciona bem com poucos (1-5) | Melhor com muitos indicadores |
| **Perda de info** | Perde info intra-periodo | Preserva dinamica de alta freq. |
| **Implementacao** | Mais facil de interpretar | Requer especificacao do polinomio |
| **U-MIDAS** | — | Versao irrestrita, mais robusta |

```python
import forecastbox as fb

# Bridge: agrega mensal -> trimestral, depois regressao
bridge = fb.BridgeEquation(
    target_freq="QS",
    indicators=["pmc", "pim", "confianca"],
    aggregation="mean"
)
bridge.fit(y_quarterly, X_monthly)

# MIDAS: usa dados mensais diretamente
midas = fb.MIDAS(
    target_freq="QS",
    source_freq="MS",
    polynomial="almon",  # Polinomio de Almon
    n_lags=3             # 3 meses de lag
)
midas.fit(y_quarterly, X_monthly)
```

!!! info "Recomendacao pratica"
    Comece com **Bridge** pela simplicidade. Migre para **MIDAS** se (a) a dinamica intra-trimestral importa, ou (b) tiver indicadores diarios/semanais onde a agregacao perde informacao.

---

## Como lidar com revisao de dados no nowcasting?

Dados macroeconomicos sao revisados apos a publicacao inicial. O `forecastbox` oferece o sistema de **vintages** para avaliacoes real-time:

```python
import forecastbox as fb

# Carregar dados vintage (cada coluna = uma data de publicacao)
vintages = fb.load_vintages("dados_vintage.csv")

# Avaliar nowcast com dados que estavam disponiveis em cada ponto
evaluator = fb.RealTimeEvaluator(
    model=fb.DFM(n_factors=2),
    vintages=vintages,
    target="pib",
    start="2015-01-01",
    end="2024-12-01"
)
results = evaluator.evaluate()

# Decomposicao: erro de revisao vs erro de modelo
print(results.revision_error)  # Erro por revisao de dados
print(results.model_error)     # Erro do modelo propriamente
print(results.total_error)     # Erro total
```

!!! warning "Armadilha comum"
    Avaliar nowcasting com dados **finais** (revisados) superestima a acuracia. Sempre use **pseudo out-of-sample com vintages** para resultados realistas.

---

## Como integrar forecastbox com Apache Airflow?

O `forecastbox` pode ser integrado em DAGs do Airflow via `PythonOperator`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def run_forecast(**kwargs):
    import forecastbox as fb
    import pandas as pd

    # Carregar dados atualizados
    data = pd.read_parquet("/data/series_economicas.parquet")

    # Pipeline automatizado
    pipeline = fb.Pipeline(
        models=["arima", "ets", "theta"],
        combination="bma",
        evaluation=True,
        h=12
    )
    results = pipeline.run(data)

    # Salvar resultados
    results.to_parquet(f"/data/forecasts/{kwargs['ds']}.parquet")

    # Alertar se performance degradou
    if results.rmse > results.historical_rmse * 1.5:
        raise ValueError(f"RMSE degradou: {results.rmse:.4f}")

dag = DAG(
    "forecastbox_pipeline",
    schedule_interval="0 8 1 * *",  # Dia 1 de cada mes as 8h
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

forecast_task = PythonOperator(
    task_id="run_forecast",
    python_callable=run_forecast,
    dag=dag,
)
```

!!! tip "Monitoramento"
    Use `fb.Monitor` para gerar relatorios automaticos e integre com sistemas de alerta (Slack, email) via callbacks do Airflow.

---

## Posso usar forecastbox em tempo real (streaming)?

O `forecastbox` nao e projetado para streaming de baixa latencia, mas suporta **atualizacao incremental**:

```python
import forecastbox as fb

# Ajuste inicial
model = fb.AutoARIMA()
model.fit(y_historical)

# Atualizar com novo dado (sem re-estimar do zero)
model.update(y_new=105.3)
forecast = model.forecast(h=12)

# Pipeline com atualizacao incremental
pipeline = fb.Pipeline(
    models=["arima", "ets"],
    combination="inverse_mse",
    update_mode="incremental"  # Atualiza sem re-estimar
)
pipeline.run(y_historical)

# Novo dado chega
pipeline.update(y_new=105.3)
new_forecast = pipeline.forecast(h=12)
```

Para latencias abaixo de 100ms, considere:

1. Pre-ajustar o modelo e serializar com `pickle` ou `joblib`
2. Usar `model.update()` em vez de `model.fit()` a cada novo dado
3. Servir previsoes via API (Flask/FastAPI) com modelo em memoria

---

## Como fazer selecao de variaveis para modelos de nowcasting?

Com muitos indicadores candidatos, a selecao de variaveis melhora a parcimonia e a performance:

```python
import forecastbox as fb

# Metodo 1: Selecao por correlacao cruzada com target
selector = fb.IndicatorSelector(
    method="cross_correlation",
    max_lag=6,
    threshold=0.3
)
selected = selector.select(X_monthly, y_quarterly)
print(f"Selecionados: {selected.columns.tolist()}")

# Metodo 2: LASSO adaptativo
selector = fb.IndicatorSelector(
    method="adaptive_lasso",
    alpha="bic"  # Selecao automatica via BIC
)
selected = selector.select(X_monthly, y_quarterly)

# Metodo 3: Selecao por importancia no DFM
dfm = fb.DFM(n_factors=3)
dfm.fit(X_monthly, y_quarterly)
importance = dfm.variable_importance()
print(importance.sort_values(ascending=False))
```

!!! info "Regra pratica para nowcasting"
    Para PIB brasileiro trimestral, 8-15 indicadores mensais sao tipicamente suficientes. Indicadores lideres (PMC, PIM-PF, confianca) geralmente dominam.

---

## Como exportar resultados para Excel ou LaTeX?

O `forecastbox` oferece exportacao direta em multiplos formatos:

```python
import forecastbox as fb

pipeline = fb.Pipeline(models=["arima", "ets", "theta"])
results = pipeline.run(y)

# Exportar para Excel
results.to_excel("resultados.xlsx", include_plots=True)

# Exportar tabela de metricas para LaTeX
metrics = results.evaluation.summary()
print(metrics.to_latex(
    caption="Comparacao de Modelos",
    label="tab:model_comparison",
    float_format="%.4f"
))

# Gerar relatorio completo em HTML
report = fb.Report(results)
report.save("relatorio_previsao.html")
```

A saida LaTeX gera tabelas formatadas para publicacao academica:

```latex
\begin{table}[htbp]
\centering
\caption{Comparacao de Modelos}
\label{tab:model_comparison}
\begin{tabular}{lrrr}
\toprule
Modelo & RMSE & MAE & MASE \\
\midrule
AutoARIMA & 1.234 & 0.987 & 0.876 \\
AutoETS   & 1.198 & 0.956 & 0.849 \\
Theta     & 1.267 & 1.012 & 0.899 \\
\bottomrule
\end{tabular}
\end{table}
```
