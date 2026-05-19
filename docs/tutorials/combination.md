---
title: "Combinacao de Previsoes"
description: "Tutorial pratico: media simples, Inverse MSE, OLS, BMA e diagnostico de estabilidade de pesos"
---

# Combinacao de Previsoes

!!! info "Sobre este tutorial"
    **Nivel**: :material-star: :material-star: Intermediario
    **Tempo estimado**: 45 minutos
    **Pre-requisitos**: Tutorial de [Fundamentos](fundamentals.md)
    **Dados**: Inflacao mensal (IPCA) com 5 modelos pre-ajustados

Neste tutorial voce vai aprender a combinar previsoes de multiplos modelos para
obter previsoes mais robustas. A combinacao de previsoes e uma das tecnicas mais
poderosas da econometria aplicada -- desde Bates & Granger (1969), a literatura
mostra que combinacoes frequentemente superam modelos individuais.

## O que voce vai aprender

- Por que combinar previsoes (motivacao teorica e pratica)
- Media simples e Inverse MSE weights
- Combinacao OLS e BMA
- Comparar todos os metodos
- Diagnosticar estabilidade dos pesos
- Quando usar cada metodo

---

## Etapa 1: Motivacao -- Por que Combinar?

Modelos diferentes capturam aspectos distintos de uma serie temporal. Ao combinar,
**diversificamos o risco de escolher o modelo errado** e reduzimos a variancia
da previsao.

Formalmente, considere dois modelos com erros $e_1$ e $e_2$. A variancia do erro
da combinacao com pesos $w$ e $(1-w)$ e:

$$
\text{Var}(e_c) = w^2 \sigma_1^2 + (1-w)^2 \sigma_2^2 + 2w(1-w)\rho\sigma_1\sigma_2
$$

Quando $\rho < 1$, existe um peso $w^*$ tal que $\text{Var}(e_c) < \min(\sigma_1^2, \sigma_2^2)$.
A combinacao e **melhor que qualquer modelo individual**.

Vamos demonstrar isso na pratica com 5 modelos de inflacao:

```python
import pandas as pd
from forecastbox.datasets import load_inflation
from forecastbox import AutoARIMA, AutoETS, Theta, AutoVAR, Naive

# Carregar dados de inflacao mensal
data = load_inflation()
train, test = data[:-12], data[-12:]

print(f"Treino: {len(train)} obs")
print(f"Teste:  {len(test)} obs (12 meses)")

# Ajustar 5 modelos
models = {
    "ARIMA": AutoARIMA(),
    "ETS": AutoETS(),
    "VAR": AutoVAR(),
    "Theta": Theta(),
    "Naive": Naive(method="seasonal"),
}

forecasts = {}
for name, model in models.items():
    forecasts[name] = model.fit_predict(train, horizon=12)
    print(f"{name:>8}: fitted")
```

```text
Treino: 228 obs
Teste:  12 obs (12 meses)
   ARIMA: fitted
     ETS: fitted
     VAR: fitted
   Theta: fitted
   Naive: fitted
```

```python
# Metricas individuais
from forecastbox.evaluate import metrics

print("Modelo      RMSE     MAE    MAPE")
print("-" * 38)
for name, fc in forecasts.items():
    m = metrics.compute(test, fc.forecast)
    print(f"{name:>8}  {m.rmse:>6.4f}  {m.mae:>6.4f}  {m.mape:>5.2f}%")
```

```text
Modelo      RMSE     MAE    MAPE
--------------------------------------
   ARIMA  0.3421  0.2718  12.45%
     ETS  0.3189  0.2534  11.62%
     VAR  0.3567  0.2891  13.24%
   Theta  0.3312  0.2645  12.11%
   Naive  0.4102  0.3456  15.84%
```

!!! note "Observacao"
    O ETS e o melhor modelo individual, mas sera que uma combinacao pode
    supera-lo? Vamos testar.

---

## Etapa 2: Media Simples

O metodo mais simples e robusto: dar peso igual a todos os modelos.
$w_i = 1/N$ para todo $i$.

$$
\hat{y}_{t+h}^c = \frac{1}{N}\sum_{i=1}^{N}\hat{y}_{t+h}^{(i)}
$$

```python
from forecastbox import combine

# Combinacao via media simples
combined_simple = combine(
    forecasts=list(forecasts.values()),
    method="simple_average",
)

print("Pesos:")
for name, w in combined_simple.weights.items():
    print(f"  {name}: {w:.3f}")

print(f"\nPrevisao combinada:")
print(combined_simple.forecast)
```

```text
Pesos:
  ARIMA: 0.200
  ETS: 0.200
  VAR: 0.200
  Theta: 0.200
  Naive: 0.200

Previsao combinada:
2023-01    0.52
2023-02    0.48
2023-03    0.45
2023-04    0.41
2023-05    0.39
2023-06    0.37
2023-07    0.43
2023-08    0.40
2023-09    0.38
2023-10    0.36
2023-11    0.34
2023-12    0.42
Freq: MS, Name: combined, dtype: float64
```

```python
# Avaliar
m_simple = metrics.compute(test, combined_simple.forecast)
print(f"Media Simples: RMSE={m_simple.rmse:.4f}, MAE={m_simple.mae:.4f}, MAPE={m_simple.mape:.2f}%")
```

```text
Media Simples: RMSE=0.3012, MAE=0.2389, MAPE=10.95%
```

!!! tip "Forecast Combination Puzzle"
    A media simples frequentemente supera metodos sofisticados de combinacao.
    Isso ocorre porque a estimacao de pesos otimos introduz erro amostral
    que pode superar o ganho teorico (Smith & Wallis, 2009).

---

## Etapa 3: Inverse MSE Weights

Dar mais peso aos modelos com menor erro quadratico medio. Os pesos sao
inversamente proporcionais ao MSE de cada modelo:

$$
w_i = \frac{1/\text{MSE}_i}{\sum_{j=1}^{N} 1/\text{MSE}_j}
$$

```python
# Combinacao via Inverse MSE
combined_imse = combine(
    forecasts=list(forecasts.values()),
    method="inverse_mse",
    train_actual=train,
)

print("Pesos (Inverse MSE):")
for name, w in combined_imse.weights.items():
    print(f"  {name}: {w:.3f}")
```

```text
Pesos (Inverse MSE):
  ARIMA: 0.213
  ETS: 0.241
  VAR: 0.192
  Theta: 0.224
  Naive: 0.130
```

```python
# Avaliar
m_imse = metrics.compute(test, combined_imse.forecast)
print(f"Inverse MSE: RMSE={m_imse.rmse:.4f}, MAE={m_imse.mae:.4f}, MAPE={m_imse.mape:.2f}%")
```

```text
Inverse MSE: RMSE=0.2945, MAE=0.2341, MAPE=10.72%
```

!!! note "Pesos e performance"
    Note que o ETS recebeu o maior peso (0.241), consistente com sua melhor
    performance individual. O Naive recebeu o menor peso (0.130).

!!! example "Try it yourself"
    Tente usar o metodo `"optimal"` (Bates-Granger) que minimiza a variancia
    do erro da combinacao considerando correlacoes entre os modelos:

    ```python
    combined_opt = combine(
        forecasts=list(forecasts.values()),
        method="optimal",
        train_actual=train,
    )
    print("Pesos (Optimal):", combined_opt.weights)
    ```

---

## Etapa 4: Combinacao OLS

Os pesos sao estimados por Minimos Quadrados Ordinarios, regredindo os valores
realizados nas previsoes dos modelos:

$$
y_t = \beta_0 + \sum_{i=1}^{N}\beta_i \hat{y}_t^{(i)} + \varepsilon_t
$$

Os coeficientes $\beta_i$ sao usados como pesos. Note que esse metodo **nao
restringe** os pesos a somar 1, e permite intercepto (correcao de vies).

```python
# Combinacao via OLS
combined_ols = combine(
    forecasts=list(forecasts.values()),
    method="ols",
    train_actual=train,
)

print("Pesos (OLS):")
print(f"  Intercepto: {combined_ols.weights['intercept']:.4f}")
for name in models:
    print(f"  {name}: {combined_ols.weights[name]:.3f}")
```

```text
Pesos (OLS):
  Intercepto: 0.0234
  ARIMA: 0.185
  ETS: 0.312
  VAR: 0.078
  Theta: 0.256
  Naive: 0.142
```

```python
# Avaliar
m_ols = metrics.compute(test, combined_ols.forecast)
print(f"OLS: RMSE={m_ols.rmse:.4f}, MAE={m_ols.mae:.4f}, MAPE={m_ols.mape:.2f}%")
```

```text
OLS: RMSE=0.2867, MAE=0.2278, MAPE=10.43%
```

!!! warning "Risco de overfitting"
    Com muitos modelos e poucos dados, os pesos OLS podem sofrer de overfitting.
    Quando $N$ (numero de modelos) e grande relativo ao tamanho da amostra,
    considere usar `method="stacking"` com regularizacao:

    ```python
    combined_stack = combine(
        forecasts=list(forecasts.values()),
        method="stacking",
        train_actual=train,
        alpha=0.1,  # regularizacao L1
    )
    ```

---

## Etapa 5: Bayesian Model Averaging (BMA)

O BMA atribui pesos baseados nas probabilidades posteriores de cada modelo,
calculadas via criterios de informacao:

$$
w_i = P(M_i | \text{data}) = \frac{\exp(-\tfrac{1}{2}\Delta \text{BIC}_i)}{\sum_{j=1}^{N}\exp(-\tfrac{1}{2}\Delta \text{BIC}_j)}
$$

onde $\Delta \text{BIC}_i = \text{BIC}_i - \min_j(\text{BIC}_j)$.

```python
# Combinacao via BMA
combined_bma = combine(
    forecasts=list(forecasts.values()),
    method="bma",
    train_actual=train,
)

print("Probabilidades Posteriores (BMA):")
for name, w in combined_bma.weights.items():
    print(f"  {name}: {w:.3f}")
print(f"\nModelo mais provavel: {max(combined_bma.weights, key=combined_bma.weights.get)}")
```

```text
Probabilidades Posteriores (BMA):
  ARIMA: 0.178
  ETS: 0.342
  VAR: 0.089
  Theta: 0.261
  Naive: 0.130

Modelo mais provavel: ETS
```

```python
# Avaliar
m_bma = metrics.compute(test, combined_bma.forecast)
print(f"BMA: RMSE={m_bma.rmse:.4f}, MAE={m_bma.mae:.4f}, MAPE={m_bma.mape:.2f}%")
```

```text
BMA: RMSE=0.2901, MAE=0.2312, MAPE=10.58%
```

!!! note "BMA vs OLS"
    O BMA e mais parcimonioso que o OLS: os pesos somam 1, sao nao-negativos,
    e nao ha intercepto. Isso reduz o risco de overfitting, mas pode
    perder a correcao de vies que o OLS oferece.

!!! example "Try it yourself"
    Compare os pesos do BMA com os do Inverse MSE. Quais as diferencas?

    ```python
    import pandas as pd

    comparison = pd.DataFrame({
        "Inverse MSE": combined_imse.weights,
        "BMA": combined_bma.weights,
    })
    print(comparison)
    ```

---

## Etapa 6: Comparar Todas as Combinacoes

Vamos montar uma tabela comparativa de todos os metodos:

```python
# Tabela comparativa
results = {
    "Media Simples": m_simple,
    "Inverse MSE": m_imse,
    "OLS": m_ols,
    "BMA": m_bma,
}

# Incluir melhores modelos individuais para referencia
m_best = metrics.compute(test, forecasts["ETS"].forecast)
results["ETS (melhor ind.)"] = m_best

print("Metodo              RMSE     MAE    MAPE")
print("=" * 45)
for name, m in results.items():
    print(f"{name:<20} {m.rmse:>6.4f}  {m.mae:>6.4f}  {m.mape:>5.2f}%")
```

```text
Metodo              RMSE     MAE    MAPE
=============================================
Media Simples        0.3012  0.2389  10.95%
Inverse MSE          0.2945  0.2341  10.72%
OLS                  0.2867  0.2278  10.43%
BMA                  0.2901  0.2312  10.58%
ETS (melhor ind.)    0.3189  0.2534  11.62%
```

!!! tip "Resultado tipico"
    Todas as combinacoes superaram o melhor modelo individual (ETS).
    Neste caso, o OLS obteve o melhor resultado, mas as diferencas entre
    os metodos de combinacao sao pequenas -- um resultado comum na literatura.

```python
# Visualizar comparacao
from forecastbox.viz import plot_forecast_comparison

fig = plot_forecast_comparison(
    actual=data,
    forecasts={
        "Media Simples": combined_simple,
        "OLS": combined_ols,
        "BMA": combined_bma,
        "ETS (individual)": forecasts["ETS"],
    },
    test=test,
    title="Comparacao de Metodos de Combinacao -- Inflacao",
    ylabel="Inflacao (%)",
)
fig.show()
```

---

## Etapa 7: Diagnostico -- Estabilidade dos Pesos

Um diagnostico importante e verificar se os pesos sao estaveis ao longo do tempo.
Pesos instáveis sugerem que a performance relativa dos modelos muda, e metodos
time-varying podem ser mais adequados.

```python
from forecastbox.diagnostics import weight_stability

# Calcular pesos rolling (janela de 24 meses)
ws = weight_stability(
    forecasts=list(forecasts.values()),
    actual=train,
    method="inverse_mse",
    window=24,
)

print("Estabilidade dos pesos (ultimas 5 janelas):")
print(ws.rolling_weights.tail())
```

```text
Estabilidade dos pesos (ultimas 5 janelas):
            ARIMA    ETS    VAR  Theta  Naive
2021-08     0.205  0.248  0.188  0.228  0.131
2021-09     0.208  0.245  0.191  0.225  0.131
2021-10     0.210  0.243  0.193  0.223  0.131
2021-11     0.211  0.242  0.194  0.222  0.131
2021-12     0.213  0.241  0.192  0.224  0.130
```

```python
# Visualizar evolucao dos pesos
from forecastbox.viz import plot_weight_evolution

fig = plot_weight_evolution(
    ws.rolling_weights,
    title="Evolucao dos Pesos -- Inverse MSE (janela=24m)",
)
fig.show()
```

```python
# Teste formal de estabilidade
print(f"\nTeste de estabilidade:")
print(f"  Variancia media dos pesos: {ws.mean_variance:.6f}")
print(f"  Fluctuation test p-valor:  {ws.fluctuation_pvalue:.4f}")

if ws.fluctuation_pvalue < 0.05:
    print("  -> Pesos INSTÁVEIS: considere metodo time-varying")
else:
    print("  -> Pesos estaveis: metodo fixo e adequado")
```

```text
Teste de estabilidade:
  Variancia media dos pesos: 0.000234
  Fluctuation test p-valor:  0.3421
  -> Pesos estaveis: metodo fixo e adequado
```

!!! example "Try it yourself"
    Repita o diagnostico usando pesos OLS. Eles sao mais ou menos estaveis
    que os pesos Inverse MSE?

    ```python
    ws_ols = weight_stability(
        forecasts=list(forecasts.values()),
        actual=train,
        method="ols",
        window=24,
    )
    print(f"Fluctuation test (OLS): p={ws_ols.fluctuation_pvalue:.4f}")
    ```

---

## Etapa 8: Quando Usar Cada Metodo

| Metodo | Quando usar | Quando evitar |
|--------|-------------|---------------|
| **Media Simples** | Poucos dados, modelos similares, benchmark | Modelos com qualidade muito diferente |
| **Inverse MSE** | Boa diferenciacao entre modelos, amostra razoavel | Poucos dados out-of-sample |
| **OLS** | Amostra grande, suspeita de vies nos modelos | $N$ grande relativo a $T$, risco de overfitting |
| **BMA** | Incerteza sobre qual modelo e o correto | Modelos com BIC muito similar |
| **Time-Varying** | Pesos instáveis, mudancas estruturais | Amostra curta, pesos estaveis |

!!! tip "Regra pratica"
    Na duvida, comece com **media simples**. E surpreendentemente dificil de
    superar na pratica. Se a performance importa e voce tem dados suficientes,
    teste Inverse MSE ou BMA. Reserve OLS para quando ha suspeita de vies.

---

## Resumo

Neste tutorial voce aprendeu:

| Etapa | O que voce fez | Funcao principal |
|-------|----------------|------------------|
| 1 | Motivacao teorica | Por que $\text{Var}(e_c) < \min(\sigma_i^2)$ |
| 2 | Media simples | `combine(..., method="simple_average")` |
| 3 | Inverse MSE | `combine(..., method="inverse_mse")` |
| 4 | OLS | `combine(..., method="ols")` |
| 5 | BMA | `combine(..., method="bma")` |
| 6 | Comparacao completa | Tabela de RMSE, MAE, MAPE |
| 7 | Diagnostico de pesos | `weight_stability()`, `plot_weight_evolution()` |
| 8 | Guia de escolha | Quando usar cada metodo |

---

## Proximos Passos

<div class="grid cards" markdown>

- :material-test-tube: **[Avaliacao Rigorosa](evaluation.md)**

    Teste estatisticamente se a combinacao e realmente melhor que modelos individuais

- :material-scale-balance: **[Graficos de Combinacao](../visualization/combination-plots.md)**

    Visualize pesos, evolucao temporal e posterior BMA

- :material-book-open-variant: **[User Guide: Combinacao](../user-guide/combination/index.md)**

    Referencia completa dos 7 metodos de combinacao

- :material-school: **[Theory: Combinacao](../theory/combination-theory.md)**

    Fundamentos teoricos de Bates-Granger, BMA e shrinkage

- :material-arrow-decision: **[Cenarios e Previsao Condicional](scenarios.md)**

    Use combinacoes com cenarios condicionais para analise de risco

</div>
