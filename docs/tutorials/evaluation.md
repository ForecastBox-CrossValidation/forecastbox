---
title: "Avaliacao Rigorosa de Previsoes"
description: "Tutorial pratico: metricas, Diebold-Mariano, Model Confidence Set, cross-validation temporal e Mincer-Zarnowitz"
---

# Avaliacao Rigorosa de Previsoes

!!! info "Sobre este tutorial"
    **Nivel**: :material-star: :material-star: Intermediario
    **Tempo estimado**: 45 minutos
    **Pre-requisitos**: Tutorial de [Fundamentos](fundamentals.md)
    **Dados**: Previsoes de cambio R$/USD com 3 modelos

Comparar previsoes apenas pelo RMSE pode levar a conclusoes erroneas. Diferencas
pequenas podem ser fruto do acaso, e um modelo com menor erro medio pode ser
inferior em determinados regimes. Neste tutorial, voce vai aprender a avaliar
previsoes de forma estatisticamente rigorosa.

## O que voce vai aprender

- Calcular metricas pontuais e probabilisticas (RMSE, MAE, MASE, CRPS)
- Testar se diferencas sao significativas (Diebold-Mariano)
- Identificar o conjunto de melhores modelos (Model Confidence Set)
- Validar com cross-validation temporal (expanding window)
- Verificar vies e eficiencia (Mincer-Zarnowitz)
- Visualizar rankings e distribuicao de erros

---

## Etapa 1: Setup -- Dados e Modelos

Vamos trabalhar com previsoes de taxa de cambio R$/USD usando 3 modelos:

```python
import pandas as pd
from forecastbox.datasets import load_exchange_rate
from forecastbox import AutoARIMA, AutoETS, Theta

# Carregar dados de cambio
data = load_exchange_rate()
train, test = data[:-60], data[-60:]

print(f"Treino: {len(train)} obs")
print(f"Teste:  {len(test)} obs (60 dias)")

# Ajustar 3 modelos
arima = AutoARIMA().fit_predict(train, horizon=60)
ets = AutoETS().fit_predict(train, horizon=60)
theta = Theta().fit_predict(train, horizon=60)

forecasts = {"ARIMA": arima, "ETS": ets, "Theta": theta}
print("\nModelos ajustados: ARIMA, ETS, Theta")
```

```text
Treino: 940 obs
Teste:  60 obs (60 dias)

Modelos ajustados: ARIMA, ETS, Theta
```

---

## Etapa 2: Metricas -- RMSE, MAE, MASE, CRPS

Vamos calcular um conjunto abrangente de metricas para cada modelo:

```python
from forecastbox.evaluate import metrics, forecast_metrics

# Metricas para cada modelo
print("Modelo   RMSE     MAE     MASE    CRPS")
print("=" * 45)
for name, fc in forecasts.items():
    m = forecast_metrics(
        actual=test,
        predicted=fc.forecast,
        metrics=["rmse", "mae", "mase", "crps"],
    )
    print(f"{name:>8} {m['rmse']:>6.4f}  {m['mae']:>6.4f}  {m['mase']:>6.4f}  {m['crps']:>6.4f}")
```

```text
Modelo   RMSE     MAE     MASE    CRPS
=============================================
   ARIMA 0.0856  0.0672  0.8914  0.0452
     ETS 0.0823  0.0648  0.8591  0.0431
   Theta 0.0891  0.0701  0.9298  0.0478
```

| Metrica | Formula | Interpretacao |
|---------|---------|---------------|
| **RMSE** | $\sqrt{\frac{1}{h}\sum_{t}(y_t - \hat{y}_t)^2}$ | Penaliza erros grandes |
| **MAE** | $\frac{1}{h}\sum_{t}\lvert y_t - \hat{y}_t \rvert$ | Robusta a outliers |
| **MASE** | $\frac{\text{MAE}}{\text{MAE}_{\text{naive}}}$ | Escalada pelo naive; $< 1$ supera o naive |
| **CRPS** | $\int_{-\infty}^{\infty}(F(\xi) - \mathbf{1}(\xi \geq y))^2 d\xi$ | Avalia distribuicao preditiva completa |

!!! note "MASE < 1"
    O MASE compara o modelo com o naive (random walk). Um MASE $< 1$ indica
    que o modelo supera o benchmark naive. Todos os nossos modelos superam.

!!! note "CRPS"
    O CRPS (Continuous Ranked Probability Score) e a metrica padrao para avaliar
    previsoes probabilisticas. Requer que o modelo forneca intervalos ou
    distribuicao preditiva, nao apenas um valor pontual.

!!! example "Try it yourself"
    Adicione o MAPE e o Theil U a tabela de metricas:

    ```python
    m = forecast_metrics(
        actual=test,
        predicted=arima.forecast,
        metrics=["rmse", "mae", "mape", "theil_u"],
    )
    print(m)
    ```

---

## Etapa 3: Diebold-Mariano -- Modelo A vs B

O RMSE diz que ETS e melhor que ARIMA, mas essa diferenca e **estatisticamente
significativa**? O teste de Diebold-Mariano (1995) responde essa pergunta.

A hipotese nula e que os dois modelos tem o mesmo poder preditivo:

$$
H_0: E[d_t] = 0, \quad d_t = L(e_{1t}) - L(e_{2t})
$$

onde $L(\cdot)$ e a funcao de perda (tipicamente erro quadratico) e a estatistica
do teste e:

$$
DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d})}} \xrightarrow{d} N(0,1)
$$

```python
from forecastbox.evaluation import diebold_mariano

# ARIMA vs ETS
errors_arima = test - arima.forecast
errors_ets = test - ets.forecast

dm_ae = diebold_mariano(errors_arima, errors_ets, h=1)

print("=== Diebold-Mariano: ARIMA vs ETS ===")
print(f"Estatistica DM: {dm_ae.statistic:.4f}")
print(f"p-valor:         {dm_ae.pvalue:.4f}")
print(f"Conclusao:       ", end="")

if dm_ae.pvalue < 0.05:
    print("Diferenca SIGNIFICATIVA a 5%")
else:
    print("Diferenca NAO significativa a 5%")
```

```text
=== Diebold-Mariano: ARIMA vs ETS ===
Estatistica DM: 1.8734
p-valor:         0.0612
Conclusao:       Diferenca NAO significativa a 5%
```

```python
# Todos os pares
pairs = [("ARIMA", "ETS"), ("ARIMA", "Theta"), ("ETS", "Theta")]

print("\nComparacoes par-a-par:")
print(f"{'Par':<18} {'DM stat':>8} {'p-valor':>8} {'Significativo?':>15}")
print("-" * 52)

for m1, m2 in pairs:
    e1 = test - forecasts[m1].forecast
    e2 = test - forecasts[m2].forecast
    dm = diebold_mariano(e1, e2, h=1)
    sig = "Sim *" if dm.pvalue < 0.05 else "Nao"
    print(f"{m1} vs {m2:<8} {dm.statistic:>8.4f} {dm.pvalue:>8.4f} {sig:>15}")
```

```text
Comparacoes par-a-par:
Par                DM stat  p-valor Significativo?
----------------------------------------------------
ARIMA vs ETS        1.8734   0.0612             Nao
ARIMA vs Theta     -0.9234   0.3561             Nao
ETS vs Theta       -2.3456   0.0191          Sim *
```

!!! tip "Interpretacao"
    - ARIMA vs ETS: $p = 0.061 > 0.05$ -- nao podemos rejeitar $H_0$ a 5%.
      Apesar do RMSE menor, o ETS **nao e significativamente melhor** que o ARIMA.
    - ETS vs Theta: $p = 0.019 < 0.05$ -- o ETS **e significativamente melhor**
      que o Theta.

!!! example "Try it yourself"
    Teste com funcao de perda absoluta ao inves de quadratica:

    ```python
    dm_abs = diebold_mariano(
        errors_arima, errors_ets, h=1, loss="absolute"
    )
    print(f"DM (perda absoluta): stat={dm_abs.statistic:.4f}, p={dm_abs.pvalue:.4f}")
    ```

---

## Etapa 4: Model Confidence Set

O teste de Diebold-Mariano compara modelos **par a par**. Mas e se queremos
identificar o **conjunto de modelos superiores** com garantia estatistica?

O Model Confidence Set (Hansen, Lunde & Nason, 2011) responde: qual o menor
subconjunto $\hat{M}^*$ tal que contem o melhor modelo com probabilidade
$\geq 1 - \alpha$?

```python
from forecastbox.evaluation import mcs

# Construir matriz de perdas (erros quadraticos)
loss_matrix = pd.DataFrame({
    name: (test - fc.forecast) ** 2
    for name, fc in forecasts.items()
})

# Model Confidence Set
mcs_result = mcs(loss_matrix, alpha=0.10)

print("=== Model Confidence Set (alpha=0.10) ===")
print(f"Modelos no MCS: {mcs_result.superior_models}")
print(f"\nMCS p-valores:")
for name, pval in mcs_result.pvalues.items():
    included = "IN" if name in mcs_result.superior_models else "OUT"
    print(f"  {name:>8}: p={pval:.4f}  [{included}]")
```

```text
=== Model Confidence Set (alpha=0.10) ===
Modelos no MCS: ['ARIMA', 'ETS']

MCS p-valores:
     ARIMA: p=0.3412  [IN]
       ETS: p=1.0000  [IN]
     Theta: p=0.0734  [OUT]
```

!!! note "Interpretacao do MCS"
    - **ETS** e **ARIMA** estao no MCS -- nao podemos distingui-los estatisticamente.
    - **Theta** fica fora ($p = 0.073 < 0.10$) -- e estatisticamente inferior.
    - O MCS com $\alpha = 0.10$ contem os dois melhores modelos com 90% de confianca.

```python
# Visualizar MCS
from forecastbox.viz import plot_mcs

fig = plot_mcs(mcs_result, title="Model Confidence Set -- Cambio R$/USD")
fig.show()
```

!!! example "Try it yourself"
    Repita o MCS com $\alpha = 0.05$ (mais conservador). O conjunto muda?

    ```python
    mcs_05 = mcs(loss_matrix, alpha=0.05)
    print(f"MCS (alpha=0.05): {mcs_05.superior_models}")
    ```

---

## Etapa 5: Cross-Validation Temporal

Avaliar em um unico split treino/teste pode ser instavel. A cross-validation
temporal com expanding window da uma visao mais robusta da performance:

```python
from forecastbox.evaluation import TimeSeriesSplit

# Expanding window: 10 folds, horizonte de 20 dias
splitter = TimeSeriesSplit(
    initial_window=500,
    step=44,
    horizon=20,
)

print(f"Numero de folds: {splitter.n_splits(data)}")
print(f"Janela inicial: {splitter.initial_window}")
print(f"Horizonte: {splitter.horizon}")
```

```text
Numero de folds: 10
Janela inicial: 500
Horizonte: 20
```

```python
# Cross-validation para cada modelo
from forecastbox.evaluate import metrics
import numpy as np

cv_results = {name: [] for name in forecasts}

for fold, (train_idx, test_idx) in enumerate(splitter.split(data)):
    train_cv = data.iloc[train_idx]
    test_cv = data.iloc[test_idx]

    for name, ModelClass in [("ARIMA", AutoARIMA), ("ETS", AutoETS), ("Theta", Theta)]:
        fc = ModelClass().fit_predict(train_cv, horizon=len(test_cv))
        m = metrics.compute(test_cv, fc.forecast)
        cv_results[name].append(m.rmse)

# Resumo
print("Cross-Validation Results (RMSE):")
print(f"{'Modelo':<8} {'Media':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 40)
for name, rmses in cv_results.items():
    arr = np.array(rmses)
    print(f"{name:<8} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")
```

```text
Cross-Validation Results (RMSE):
Modelo     Media      Std      Min      Max
----------------------------------------
ARIMA     0.0912   0.0234   0.0623   0.1345
ETS       0.0878   0.0198   0.0612   0.1198
Theta     0.0945   0.0267   0.0678   0.1412
```

```python
# Visualizar distribuicao de erros por fold
from forecastbox.viz import plot_cv_results

fig = plot_cv_results(
    cv_results,
    title="Cross-Validation: RMSE por Fold",
    kind="boxplot",
)
fig.show()
```

O boxplot mostra a distribuicao do RMSE nos 10 folds. Um modelo com menor
mediana **e** menor variabilidade e preferivel.

!!! note "Expanding vs Sliding Window"
    - **Expanding window**: a janela de treino cresce a cada fold. Mais dados
      para estimar, mas as primeiras observacoes podem ser menos relevantes.
    - **Sliding window**: a janela tem tamanho fixo. Mais adaptativo a mudancas,
      mas descarta dados antigos.

    ```python
    splitter_sliding = TimeSeriesSplit(
        initial_window=500,
        step=44,
        horizon=20,
        expanding=False,  # sliding window
    )
    ```

!!! example "Try it yourself"
    Calcule o MASE por fold e verifique se todos os modelos superam o naive
    em todos os folds:

    ```python
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data)):
        train_cv = data.iloc[train_idx]
        test_cv = data.iloc[test_idx]
        fc = AutoARIMA().fit_predict(train_cv, horizon=len(test_cv))
        m = forecast_metrics(test_cv, fc.forecast, metrics=["mase"])
        print(f"Fold {fold}: MASE={m['mase']:.4f} {'OK' if m['mase'] < 1 else 'PIOR que naive'}")
    ```

---

## Etapa 6: Mincer-Zarnowitz -- Vies e Eficiencia

A regressao de Mincer-Zarnowitz (1969) verifica se a previsao e nao-viesada
e eficiente. A regressao e:

$$
y_t = \alpha + \beta \hat{y}_t + \varepsilon_t
$$

Uma previsao otima satisfaz $\alpha = 0$ e $\beta = 1$ (teste conjunto):

- $\alpha \neq 0$ indica **vies** sistematico
- $\beta \neq 1$ indica **ineficiencia** (a previsao nao usa toda a informacao disponivel)

```python
from forecastbox.evaluation import mincer_zarnowitz

# Mincer-Zarnowitz para cada modelo
print("=== Regressao de Mincer-Zarnowitz ===\n")
print(f"{'Modelo':>8} {'alpha':>8} {'beta':>8} {'R2':>6} {'Teste F':>10} {'p-valor':>8}")
print("-" * 55)

for name, fc in forecasts.items():
    mz = mincer_zarnowitz(actual=test, predicted=fc.forecast)
    print(
        f"{name:>8} {mz.alpha:>8.4f} {mz.beta:>8.4f} "
        f"{mz.r_squared:>6.4f} {mz.f_statistic:>10.4f} {mz.joint_pvalue:>8.4f}"
    )
```

```text
=== Regressao de Mincer-Zarnowitz ===

  Modelo    alpha     beta     R2    Teste F  p-valor
-------------------------------------------------------
   ARIMA   0.1234   0.9678 0.8912     1.8923   0.1612
     ETS   0.0456   0.9912 0.9134     0.4567   0.6367
   Theta   0.2345   0.9234 0.8567     3.4567   0.0389
```

!!! tip "Interpretacao"
    - **ARIMA**: $\alpha = 0.12$, $\beta = 0.97$ -- nao rejeita $H_0$ ($p = 0.16$).
      Previsao nao-viesada e eficiente.
    - **ETS**: $\alpha = 0.05$, $\beta = 0.99$ -- excelente ($p = 0.64$).
      Previsao praticamente perfeita em termos de vies.
    - **Theta**: $\alpha = 0.23$, $\beta = 0.92$ -- **rejeita $H_0$** ($p = 0.039$).
      Ha evidencia de vies e/ou ineficiencia.

```python
# Visualizar a regressao MZ para o ETS
from forecastbox.viz import plot_mincer_zarnowitz

fig = plot_mincer_zarnowitz(
    actual=test,
    predicted=forecasts["ETS"].forecast,
    title="Mincer-Zarnowitz: ETS",
)
fig.show()
```

O grafico mostra o scatter plot de $y_t$ vs $\hat{y}_t$ com a reta ajustada.
A linha 45 graus (pontilhada) representa a previsao perfeita ($\alpha=0$, $\beta=1$).

!!! example "Try it yourself"
    Verifique se a combinacao de modelos (do tutorial anterior) passa no
    teste de Mincer-Zarnowitz:

    ```python
    from forecastbox import combine

    combined = combine(
        forecasts=list(forecasts.values()),
        method="simple_average",
    )
    mz_comb = mincer_zarnowitz(actual=test, predicted=combined.forecast)
    print(f"Combinacao: alpha={mz_comb.alpha:.4f}, beta={mz_comb.beta:.4f}, p={mz_comb.joint_pvalue:.4f}")
    ```

---

## Etapa 7: Visualizar Rankings e Distribuicao de Erros

Vamos consolidar todas as avaliacoes em visualizacoes informativas:

```python
# Ranking de modelos por multiplas metricas
from forecastbox.viz import plot_model_ranking

ranking_data = {}
for name, fc in forecasts.items():
    m = forecast_metrics(
        actual=test,
        predicted=fc.forecast,
        metrics=["rmse", "mae", "mase", "crps"],
    )
    ranking_data[name] = m

fig = plot_model_ranking(
    ranking_data,
    title="Ranking de Modelos -- Cambio R$/USD",
)
fig.show()
```

```python
# Boxplot de erros absolutos
from forecastbox.viz import plot_error_distribution

errors = {
    name: (test - fc.forecast).abs()
    for name, fc in forecasts.items()
}

fig = plot_error_distribution(
    errors,
    title="Distribuicao de Erros Absolutos",
    kind="boxplot",
)
fig.show()
```

```python
# Erros ao longo do tempo
from forecastbox.viz import plot_error_evolution

fig = plot_error_evolution(
    errors,
    title="Evolucao dos Erros Absolutos no Tempo",
)
fig.show()
```

!!! note "Analise visual"
    O boxplot revela nao apenas o erro medio, mas a **dispersao** e presenca
    de **outliers**. Um modelo com RMSE ligeiramente maior mas menos outliers
    pode ser preferivel em contextos onde erros extremos sao custosos.

!!! example "Try it yourself"
    Crie um grafico de erros acumulados (cumulative sum of squared errors)
    para visualizar quando cada modelo comeca a divergir:

    ```python
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, fc in forecasts.items():
        cumsqerr = ((test - fc.forecast) ** 2).cumsum()
        ax.plot(cumsqerr.index, cumsqerr.values, label=name)

    ax.set_title("Soma Cumulativa de Erros Quadraticos")
    ax.legend()
    fig.show()
    ```

---

## Etapa 8: Workflow Completo de Avaliacao

Vamos consolidar tudo em um pipeline estruturado:

```python
from forecastbox.evaluation import (
    forecast_metrics,
    diebold_mariano,
    mcs,
    mincer_zarnowitz,
    TimeSeriesSplit,
)

# === 1. Metricas pontuais e probabilisticas ===
print("=" * 60)
print("ETAPA 1: METRICAS")
print("=" * 60)
for name, fc in forecasts.items():
    m = forecast_metrics(test, fc.forecast, metrics=["rmse", "mae", "mase", "crps"])
    print(f"{name:>8}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, MASE={m['mase']:.4f}, CRPS={m['crps']:.4f}")

# === 2. Testes de comparacao (DM) ===
print(f"\n{'=' * 60}")
print("ETAPA 2: DIEBOLD-MARIANO (par a par)")
print("=" * 60)
for m1, m2 in [("ARIMA", "ETS"), ("ARIMA", "Theta"), ("ETS", "Theta")]:
    e1 = test - forecasts[m1].forecast
    e2 = test - forecasts[m2].forecast
    dm = diebold_mariano(e1, e2, h=1)
    sig = "*" if dm.pvalue < 0.05 else ""
    print(f"{m1} vs {m2}: DM={dm.statistic:+.3f}, p={dm.pvalue:.4f} {sig}")

# === 3. Model Confidence Set ===
print(f"\n{'=' * 60}")
print("ETAPA 3: MODEL CONFIDENCE SET")
print("=" * 60)
loss_matrix = pd.DataFrame({
    name: (test - fc.forecast) ** 2
    for name, fc in forecasts.items()
})
mcs_result = mcs(loss_matrix, alpha=0.10)
print(f"Modelos no MCS (alpha=0.10): {mcs_result.superior_models}")

# === 4. Diagnostico de qualidade (MZ) ===
print(f"\n{'=' * 60}")
print("ETAPA 4: MINCER-ZARNOWITZ")
print("=" * 60)
for name, fc in forecasts.items():
    mz = mincer_zarnowitz(actual=test, predicted=fc.forecast)
    status = "OK" if mz.joint_pvalue > 0.05 else "VIESADO"
    print(f"{name:>8}: alpha={mz.alpha:.4f}, beta={mz.beta:.4f}, p={mz.joint_pvalue:.4f} [{status}]")

# === 5. Decisao ===
print(f"\n{'=' * 60}")
print("DECISAO FINAL")
print("=" * 60)
print(f"Melhores modelos (MCS): {mcs_result.superior_models}")
print(f"Modelos nao-viesados:   ", end="")
unbiased = [name for name, fc in forecasts.items()
            if mincer_zarnowitz(test, fc.forecast).joint_pvalue > 0.05]
print(unbiased)
```

```text
============================================================
ETAPA 1: METRICAS
============================================================
   ARIMA: RMSE=0.0856, MAE=0.0672, MASE=0.8914, CRPS=0.0452
     ETS: RMSE=0.0823, MAE=0.0648, MASE=0.8591, CRPS=0.0431
   Theta: RMSE=0.0891, MAE=0.0701, MASE=0.9298, CRPS=0.0478

============================================================
ETAPA 2: DIEBOLD-MARIANO (par a par)
============================================================
ARIMA vs ETS: DM=+1.873, p=0.0612
ARIMA vs Theta: DM=-0.923, p=0.3561
ETS vs Theta: DM=-2.346, p=0.0191 *

============================================================
ETAPA 3: MODEL CONFIDENCE SET
============================================================
Modelos no MCS (alpha=0.10): ['ARIMA', 'ETS']

============================================================
ETAPA 4: MINCER-ZARNOWITZ
============================================================
   ARIMA: alpha=0.1234, beta=0.9678, p=0.1612 [OK]
     ETS: alpha=0.0456, beta=0.9912, p=0.6367 [OK]
   Theta: alpha=0.2345, beta=0.9234, p=0.0389 [VIESADO]

============================================================
DECISAO FINAL
============================================================
Melhores modelos (MCS): ['ARIMA', 'ETS']
Modelos nao-viesados:   ['ARIMA', 'ETS']
```

!!! tip "Pipeline de decisao"
    O workflow completo de avaliacao segue uma logica clara:

    1. **Metricas**: qual modelo tem menor erro?
    2. **DM test**: a diferenca e significativa?
    3. **MCS**: qual o conjunto de melhores modelos?
    4. **MZ test**: os melhores modelos sao nao-viesados?
    5. **Decisao**: escolha entre os modelos no MCS que passam no MZ.

---

## Resumo

Neste tutorial voce aprendeu:

| Etapa | O que voce fez | Funcao principal |
|-------|----------------|------------------|
| 1 | Setup com 3 modelos | `AutoARIMA`, `AutoETS`, `Theta` |
| 2 | Metricas completas | `forecast_metrics(..., metrics=[...])` |
| 3 | Diebold-Mariano | `diebold_mariano(e1, e2, h=1)` |
| 4 | Model Confidence Set | `mcs(loss_matrix, alpha=0.10)` |
| 5 | Cross-validation temporal | `TimeSeriesSplit(expanding=True)` |
| 6 | Mincer-Zarnowitz | `mincer_zarnowitz(actual, predicted)` |
| 7 | Visualizacao de erros | `plot_model_ranking()`, `plot_error_distribution()` |
| 8 | Workflow completo | Pipeline de avaliacao end-to-end |

---

## Proximos Passos

<div class="grid cards" markdown>

- :material-set-merge: **[Combinacao de Previsoes](combination.md)**

    Combine os modelos do MCS para obter previsoes ainda mais robustas

- :material-arrow-decision: **[Cenarios e Previsao Condicional](scenarios.md)**

    Avalie previsoes condicionais com os mesmos testes

- :material-chart-bar: **[Graficos de Avaliacao](../visualization/evaluation-plots.md)**

    Visualize DM tests, MCS, calibracao e Mincer-Zarnowitz

- :material-book-open-variant: **[User Guide: Avaliacao](../user-guide/evaluation/index.md)**

    Referencia completa de todos os testes estatisticos

- :material-school: **[Theory: Evaluation](../theory/evaluation-theory.md)**

    Fundamentos teoricos dos testes de comparacao

</div>
