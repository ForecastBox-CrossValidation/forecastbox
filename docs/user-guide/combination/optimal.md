---
title: Combinacao Otima (Bates-Granger)
description: Combinacao otima que minimiza o MSE via pesos de Bates-Granger, com estimacao da matriz de covariancia e shrinkage.
---

# Combinacao Otima (Bates-Granger)

A combinacao otima, proposta por **Bates & Granger (1969)**, calcula os pesos que
**minimizam o erro quadratico medio** (MSE) da previsao combinada. Os pesos dependem
das variancias e correlacoes dos erros de previsao dos modelos individuais.

---

## Formulacao

### Caso com 2 Modelos

Para dois modelos com erros $e_1$ e $e_2$, o peso otimo do modelo 1 e:

$$
w_1^* = \frac{\sigma_2^2 - \rho\,\sigma_1\,\sigma_2}{\sigma_1^2 + \sigma_2^2 - 2\rho\,\sigma_1\,\sigma_2}
$$

onde:

- $\sigma_1^2, \sigma_2^2$ — variancias dos erros de previsao
- $\rho$ — correlacao entre os erros
- $w_2^* = 1 - w_1^*$

!!! info "Casos Especiais"

    - **Erros nao correlacionados** ($\rho = 0$): $w_1^* = \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2}$ — pesos inversamente proporcionais a variancia
    - **Variancias iguais** ($\sigma_1 = \sigma_2$): $w_1^* = \frac{1}{2}$ — media simples e otima
    - **Correlacao perfeita** ($\rho = 1$, $\sigma_1 = \sigma_2$): indeterminado — combinar nao ajuda

### Generalizacao para N Modelos

Para $N$ modelos, os pesos otimos sao:

$$
\mathbf{w}^* = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}'\Sigma^{-1}\mathbf{1}}
$$

onde:

- $\Sigma$ — matriz de covariancia dos erros de previsao ($N \times N$)
- $\mathbf{1}$ — vetor de uns ($N \times 1$)
- Restricao: $\mathbf{1}'\mathbf{w}^* = 1$ (pesos somam 1)

O MSE da combinacao otima e:

$$
\text{MSE}^* = \frac{1}{\mathbf{1}'\Sigma^{-1}\mathbf{1}}
$$

!!! abstract "Resultado Fundamental"

    A combinacao otima **nunca e pior** que o melhor modelo individual (em termos
    de MSE). Mesmo com apenas dois modelos de qualidade similar, a combinacao
    tipicamente reduz o MSE em 10-20%.

---

## O Problema da Estimacao de $\Sigma$

Na pratica, $\Sigma$ nao e conhecida e precisa ser estimada. Este e o **calcanhar de
Aquiles** da combinacao otima:

- Com poucos dados, $\hat{\Sigma}$ e ruidosa e instavel
- $\hat{\Sigma}$ pode ser singular ou mal-condicionada com muitos modelos
- Pesos estimados podem ser **muito diferentes** dos pesos verdadeiros

!!! warning "O Forecast Combination Puzzle"

    **Bates-Granger na pratica frequentemente perde para a media simples!**
    Este fenomeno, conhecido como o *forecast combination puzzle* (Smith & Wallis, 2009),
    ocorre porque o erro de estimacao de $\Sigma$ pode dominar o ganho teorico da
    combinacao otima. A media simples, ao nao estimar nada, evita esse problema.

### Shrinkage: A Solucao

O shrinkage combina a estimativa amostral $\hat{\Sigma}$ com um alvo estruturado,
reduzindo o ruido de estimacao:

$$
\hat{\Sigma}_{\text{shrunk}} = (1 - \delta)\,\hat{\Sigma} + \delta\,\Sigma_{\text{alvo}}
$$

onde $\delta \in [0, 1]$ controla a intensidade do shrinkage.

| Metodo | Alvo ($\Sigma_{\text{alvo}}$) | Descricao |
|:-------|:------------------------------|:----------|
| `"shrunk"` | $\text{diag}(\hat{\Sigma})$ | Encolhe correlacoes para zero |
| `"ledoit_wolf"` | $\frac{\text{tr}(\hat{\Sigma})}{N} I$ | Encolhe para identidade escalada (automatico) |
| `"mle"` | — | Sem shrinkage (estimativa amostral pura) |

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `shrinkage` | `str` | `"ledoit_wolf"` | Metodo de shrinkage: `"mle"`, `"shrunk"`, `"ledoit_wolf"` |
| `min_obs` | `int` | `30` | Minimo de observacoes para estimar $\Sigma$ |

---

## Exemplos

### Combinacao Otima com 2 Modelos

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.combine import combine

# Dados
y = pd.read_csv("ipca.csv", index_col="date", parse_dates=True)["ipca"]
y_train = y[:"2023-06"]
y_test = y["2023-07":"2023-12"]

# Ajustar 2 modelos
arima = AutoARIMA(seasonal=True, m=12).fit(y_train)
ets = AutoETS(seasonal_periods=12).fit(y_train)

fc_arima = arima.predict(horizon=6)
fc_ets = ets.predict(horizon=6)

# Combinacao otima (Bates-Granger)
fc_opt = combine(
    forecasts=[fc_arima, fc_ets],
    method="optimal",
    shrinkage="ledoit_wolf",
)
print(fc_opt.summary())
```

```text
Combination Summary
===================
Method: Optimal (Bates-Granger)
Models: 2
Shrinkage: Ledoit-Wolf

Error Covariance Matrix:
         arima     ets
arima    0.142   0.089
ets      0.089   0.178

Correlation: 0.560

Optimal Weights:
  arima    0.573
  ets      0.427

MSE (optimal):   0.098
MSE (arima):     0.142
MSE (ets):       0.178
Improvement:     31.0% vs best individual
```

### Comparacao: Otimo vs Media Simples

```python
from forecastbox.models import Theta

theta = Theta().fit(y_train)
fc_theta = theta.predict(horizon=6)

# Comparar otimo com media simples
fc_simple = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="simple",
)

fc_opt = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="optimal",
    shrinkage="ledoit_wolf",
)

fc_opt_mle = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="optimal",
    shrinkage="mle",  # sem shrinkage
)

# Avaliar out-of-sample
for name, fc in [("Media Simples", fc_simple),
                  ("Otimo (Ledoit-Wolf)", fc_opt),
                  ("Otimo (MLE)", fc_opt_mle)]:
    rmse = fc.evaluate(y_test, metric="rmse")
    print(f"{name:25s}  RMSE: {rmse:.4f}")
```

```text
Media Simples              RMSE: 0.3891
Otimo (Ledoit-Wolf)        RMSE: 0.3562
Otimo (MLE)                RMSE: 0.4123

Note: Optimal with Ledoit-Wolf beats simple average,
but optimal without shrinkage (MLE) is WORSE than simple average!
This is the forecast combination puzzle in action.
```

### Efeito do Shrinkage

```python
# Demonstrar efeito do shrinkage nos pesos
for method in ["mle", "shrunk", "ledoit_wolf"]:
    fc = combine(
        forecasts=[fc_arima, fc_ets, fc_theta],
        method="optimal",
        shrinkage=method,
    )
    print(f"\nShrinkage: {method}")
    print(f"  Pesos: {fc.weights_}")
```

```text
Shrinkage: mle
  Pesos: [0.612, 0.445, -0.057]

Shrinkage: shrunk
  Pesos: [0.498, 0.371, 0.131]

Shrinkage: ledoit_wolf
  Pesos: [0.467, 0.358, 0.175]
```

!!! tip "Regra Pratica"

    - Use sempre shrinkage (`"ledoit_wolf"` e o default e a escolha mais segura)
    - Com $T < 50$ ou $N > 5$: considere media simples ao inves de otimo
    - Se os pesos MLE tem valores extremos ($|w| > 2$), o shrinkage e essencial

### Sensitivity Analysis

```python
# Analisar sensibilidade dos pesos ao periodo de estimacao
sensitivity = fc_opt.weight_sensitivity(
    windows=[24, 36, 48, 60],
)
print(sensitivity)
```

```text
Weight Sensitivity to Estimation Window
========================================
Window   arima    ets     theta   MSE_oos
24       0.512   0.367   0.121   0.378
36       0.478   0.351   0.171   0.362
48       0.467   0.358   0.175   0.356
60       0.454   0.362   0.184   0.361

Note: Weights stabilize with longer estimation windows.
Optimal MSE plateaus around 48 observations.
```

---

## Quando Usar Combinacao Otima

| Cenario | Recomendacao |
|:--------|:-------------|
| 2-3 modelos, muito dado ($T > 100$) | **Otimo com shrinkage** |
| Muitos modelos, pouco dado | Media simples |
| Modelos muito correlacionados | Otimo ajuda (explora estrutura de $\Sigma$) |
| Performance muda no tempo | Time-varying ao inves de otimo |

---

## Proximos Passos

- **[Time-Varying](time-varying.md)** — quando os pesos otimos mudam ao longo do tempo
- **[Media Simples](simple.md)** — o benchmark que e difícil de bater
- **[Escolhendo Metodo](choosing.md)** — guia para selecionar a melhor estrategia

---

## Referencias

- **Bates, J.M. & Granger, C.W.J.** (1969). "The Combination of Forecasts." *Operational Research Quarterly*, 20(4), 451-468.
- **Smith, J. & Wallis, K.F.** (2009). "A Simple Explanation of the Forecast Combination Puzzle." *Oxford Bulletin of Economics and Statistics*, 71(3), 331-355.
- **Ledoit, O. & Wolf, M.** (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365-411.
- **Timmermann, A.** (2006). "Forecast Combinations." *Handbook of Economic Forecasting*, Vol. 1, 135-196.
