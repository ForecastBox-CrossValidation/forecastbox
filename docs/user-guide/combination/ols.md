---
title: OLS (Granger-Ramanathan)
description: Combinacao por regressao OLS com tres variantes classicas de Granger-Ramanathan e regularizacao.
---

# OLS (Granger-Ramanathan)

A combinacao por regressao OLS estima os pesos otimos que minimizam o erro quadratico
medio da combinacao. **Granger & Ramanathan (1984)** propuseram tres variantes classicas
que diferem nas restricoes impostas aos pesos.

---

## Tres Variantes

### Variante 1: Pesos Restritos sem Intercepto

Estima pesos que somam 1, sem intercepto:

$$
y_t = \sum_{i=1}^{N} w_i \hat{y}_{t}^{(i)} + \varepsilon_t, \quad \text{s.a.} \quad \sum_{i=1}^{N} w_i = 1
$$

- Pesos representam proporcoes (podem ser negativos)
- Interpretavel: cada peso mede a contribuicao relativa do modelo
- Estimacao via OLS com restricao linear ou substituicao

!!! info "Estimacao com Restricao"

    A restricao $\sum w_i = 1$ pode ser implementada substituindo
    $w_N = 1 - \sum_{i=1}^{N-1} w_i$ e estimando $N-1$ parametros por OLS
    irrestrito na regressao transformada.

### Variante 2: Pesos Restritos com Intercepto (Bias-Corrected)

Adiciona um intercepto $\beta_0$ para corrigir vies sistematico:

$$
y_t = \beta_0 + \sum_{i=1}^{N} w_i \hat{y}_{t}^{(i)} + \varepsilon_t, \quad \text{s.a.} \quad \sum_{i=1}^{N} w_i = 1
$$

- Corrige vies quando **todos os modelos** subestimam ou superestimam
- Util quando os modelos individuais tem vies consistente
- $\beta_0 \neq 0$ indica que o pool de modelos tem vies medio

### Variante 3: OLS Irrestrito

Sem qualquer restricao nos pesos:

$$
y_t = \beta_0 + \sum_{i=1}^{N} w_i \hat{y}_{t}^{(i)} + \varepsilon_t
$$

- Pesos podem ter qualquer valor (negativos, soma diferente de 1)
- Maior flexibilidade, mas tambem maior risco de overfitting
- Equivalente a uma regressao linear padrao

---

## Propriedades dos Pesos OLS

Os pesos estimados por OLS tem propriedades importantes:

| Propriedade | Variante 1 | Variante 2 | Variante 3 |
|:------------|:-----------|:-----------|:-----------|
| $\sum w_i = 1$ | :material-check: | :material-check: | :material-close: |
| Intercepto | :material-close: | :material-check: | :material-check: |
| Pesos negativos | Possiveis | Possiveis | Possiveis |
| Correcao de vies | :material-close: | :material-check: | :material-check: |
| Risco de overfitting | Moderado | Moderado | Alto |
| Parametros | $N-1$ | $N$ | $N+1$ |

!!! warning "Pesos Negativos"

    Pesos negativos sao comuns e nao sao necessariamente problematicos. Um peso
    negativo indica que o modelo contribui como **hedge** — sua inclusao com peso
    negativo melhora a previsao combinada. Entretanto, pesos muito negativos
    podem indicar overfitting.

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `variant` | `int` | `1` | Variante Granger-Ramanathan: `1`, `2` ou `3` |
| `regularization` | `str` | `None` | Regularizacao: `None`, `"ridge"`, `"lasso"`, `"elastic_net"` |
| `alpha` | `float` | `1.0` | Forca da regularizacao (so com `regularization`) |
| `l1_ratio` | `float` | `0.5` | Ratio L1/L2 para elastic net (so com `"elastic_net"`) |

---

## Regularizacao

Com muitos modelos e poucos dados, os pesos OLS sofrem de **overfitting**. A
regularizacao adiciona uma penalidade ao tamanho dos pesos:

### Ridge (L2)

$$
\min_{\mathbf{w}} \| \mathbf{y} - \mathbf{F}\mathbf{w} \|^2 + \alpha \|\mathbf{w}\|_2^2
$$

- Encolhe todos os pesos em direcao a zero (shrinkage)
- Nenhum peso e exatamente zero
- Estabiliza a estimacao quando os modelos sao altamente correlacionados

### Lasso (L1)

$$
\min_{\mathbf{w}} \| \mathbf{y} - \mathbf{F}\mathbf{w} \|^2 + \alpha \|\mathbf{w}\|_1
$$

- Produz **pesos esparsos** (alguns exatamente zero)
- Funciona como selecao automatica de modelos
- Util quando poucos modelos sao realmente informativos

### Elastic Net

$$
\min_{\mathbf{w}} \| \mathbf{y} - \mathbf{F}\mathbf{w} \|^2 + \alpha \left[ \rho \|\mathbf{w}\|_1 + \frac{1-\rho}{2} \|\mathbf{w}\|_2^2 \right]
$$

- Combina Ridge e Lasso via parametro $\rho$ (`l1_ratio`)
- $\rho = 1$: Lasso puro; $\rho = 0$: Ridge puro
- Lida simultaneamente com multicolinearidade e selecao de modelos

---

## Exemplos

### OLS com 3 Modelos (Variante 1)

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.models import Theta
from forecastbox.combine import combine

# Dados: treino + teste
y = pd.read_csv("ipca.csv", index_col="date", parse_dates=True)["ipca"]
y_train = y[:"2023-06"]
y_test = y["2023-07":"2023-12"]

# Ajustar 3 modelos
arima = AutoARIMA(seasonal=True, m=12).fit(y_train)
ets = AutoETS(seasonal_periods=12).fit(y_train)
theta = Theta().fit(y_train)

fc_arima = arima.predict(horizon=6)
fc_ets = ets.predict(horizon=6)
fc_theta = theta.predict(horizon=6)

# Combinar com OLS (variante 1: pesos somam 1, sem intercepto)
fc_ols = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="ols",
    variant=1,
)
print(fc_ols.summary())
```

```text
Combination Summary
===================
Method: OLS (Granger-Ramanathan Variant 1)
Models: 3
Constraint: sum(w) = 1, no intercept

Estimated Weights:
  arima    0.482
  ets      0.374
  theta    0.144

R-squared (in-sample): 0.934
```

### Comparacao das 3 Variantes

```python
# Comparar as tres variantes
for v in [1, 2, 3]:
    fc = combine(
        forecasts=[fc_arima, fc_ets, fc_theta],
        method="ols",
        variant=v,
    )
    print(f"\nVariante {v}:")
    print(f"  Pesos: {fc.weights_}")
    if hasattr(fc, 'intercept_') and fc.intercept_ is not None:
        print(f"  Intercepto: {fc.intercept_:.4f}")
```

```text
Variante 1:
  Pesos: [0.482, 0.374, 0.144]

Variante 2:
  Pesos: [0.451, 0.362, 0.187]
  Intercepto: 0.0823

Variante 3:
  Pesos: [0.523, 0.341, 0.098]
  Intercepto: 0.1247
```

### OLS com Regularizacao Ridge

```python
# OLS com regularizacao Ridge (util com muitos modelos)
fc_ridge = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="ols",
    variant=3,
    regularization="ridge",
    alpha=0.1,
)
print(fc_ridge.summary())
```

```text
Combination Summary
===================
Method: OLS + Ridge (alpha=0.1)
Models: 3
Constraint: unrestricted

Estimated Weights:
  arima    0.401
  ets      0.352
  theta    0.198

Note: Ridge shrinks weights toward zero, reducing overfitting.
```

### OLS com Lasso (Selecao de Modelos)

```python
# Lasso para selecao automatica de modelos
fc_lasso = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="ols",
    variant=3,
    regularization="lasso",
    alpha=0.5,
)
print(fc_lasso.summary())
```

```text
Combination Summary
===================
Method: OLS + Lasso (alpha=0.5)
Models: 3 (2 active)
Constraint: unrestricted

Estimated Weights:
  arima    0.534
  ets      0.412
  theta    0.000  <-- eliminado pelo Lasso

Note: Lasso set 1 weight(s) to zero, selecting 2 of 3 models.
```

!!! tip "Escolhendo Alpha"

    Use validacao cruzada temporal (`TimeSeriesSplit`) para selecionar o
    parametro de regularizacao $\alpha$. O forecastbox integra com
    `sklearn.model_selection` para facilitar esse processo.

---

## Cuidados com OLS

!!! warning "Overfitting com Poucos Dados"

    O OLS estima $N$ ou $N+1$ parametros. Com poucos periodos de treino
    relativo ao numero de modelos, os pesos estimados serao **instaveis** e
    podem gerar previsoes piores que a media simples. Regra pratica: tenha
    ao menos $10 \times N$ observacoes para OLS irrestrito.

### Diagnostico de Overfitting

Sinais de que o OLS esta sobreajustando:

1. **Pesos muito grandes** (positivos ou negativos, ex: $w > 3$ ou $w < -2$)
2. **Alta variancia dos pesos** ao refazer com janelas rolling
3. **Performance in-sample muito superior** a out-of-sample
4. **Multicolinearidade** entre previsoes (VIF alto)

```python
# Diagnostico: verificar estabilidade dos pesos via rolling window
stability = fc_ols.weight_stability(window=24, step=1)
print(stability)
```

```text
Weight Stability Analysis (24-month rolling window)
====================================================
         mean    std     min     max     cv
arima   0.478  0.089   0.312   0.621  0.186
ets     0.381  0.072   0.245   0.498  0.189
theta   0.141  0.115  -0.087   0.342  0.816

Warning: theta has high coefficient of variation (0.816),
suggesting unstable weight estimation.
```

---

## Proximos Passos

- **[Stacking](stacking.md)** — OLS com regularizacao e validacao cruzada integrada
- **[BMA](bma.md)** — abordagem bayesiana para pesos com incerteza
- **[Media Simples](simple.md)** — quando OLS sobreajusta, volte ao baseline

---

## Referencias

- **Granger, C.W.J. & Ramanathan, R.** (1984). "Improved Methods of Combining Forecasts." *Journal of Forecasting*, 3(2), 197-204.
- **Elliott, G. & Timmermann, A.** (2004). "Optimal Forecast Combinations Under General Loss Functions and Forecast Error Distributions." *Journal of Econometrics*, 122(1), 47-79.
- **Hsiao, C. & Wan, S.K.** (2014). "Is There an Optimal Forecast Combination?" *Journal of Econometrics*, 178(2), 294-309.
