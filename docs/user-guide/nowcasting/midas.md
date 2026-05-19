---
title: MIDAS
description: Mixed Data Sampling - regressao com dados de frequencias mistas usando funcoes de ponderacao parametricas.
---

# MIDAS (Mixed Data Sampling)

MIDAS e uma abordagem de **regressao direta** que permite usar dados de alta
frequencia (mensais, semanais, diarios) para prever uma variavel de baixa
frequencia (trimestral) **sem agregar previamente**. Em vez de comprimir os
dados mensais em medias trimestrais, o MIDAS atribui pesos otimos a cada
lag de alta frequencia.

---

## Conceito

O problema fundamental: como incluir 90 observacoes diarias (ou 3 meses) em
uma regressao com apenas ~80 trimestres? A regressao convencional nao tem
graus de liberdade suficientes. O MIDAS resolve isso com **funcoes de
ponderacao parametricas** que comprimem dezenas de lags em poucos parametros.

!!! abstract "Intuicao"

    Em vez de estimar um coeficiente para cada lag ($\beta_1, \beta_2, \ldots, \beta_{90}$),
    o MIDAS assume que os pesos seguem uma funcao suave $w(j; \theta)$
    definida por 2-3 parametros. Isso reduz drasticamente a dimensionalidade
    mantendo flexibilidade.

---

## Modelo

### Equacao MIDAS

$$
y_t^Q = \alpha + \beta \sum_{j=0}^{K-1} w(j; \boldsymbol{\theta}) \, x_{t-j/m}^{M} + \varepsilon_t
$$

onde:

| Simbolo | Descricao |
|:--------|:----------|
| $y_t^Q$ | Target de baixa frequencia (trimestral) |
| $x_{t-j/m}^M$ | Indicador de alta frequencia no lag $j$ |
| $m$ | Razao de frequencias (e.g., $m=3$ para mensal/trimestral) |
| $K$ | Numero total de lags de alta frequencia |
| $w(j; \boldsymbol{\theta})$ | Funcao de ponderacao parametrica |
| $\beta$ | Coeficiente de escala |

Os pesos sao normalizados para somar 1: $\sum_{j=0}^{K-1} w(j; \boldsymbol{\theta}) = 1$.

### MIDAS com Multiplos Regressores

$$
y_t^Q = \alpha + \sum_{i=1}^{p} \beta_i \sum_{j=0}^{K_i-1} w_i(j; \boldsymbol{\theta}_i) \, x_{i,t-j/m_i}^{HF} + \varepsilon_t
$$

Cada regressor pode ter sua propria funcao de ponderacao, numero de lags e
razao de frequencia.

---

## Funcoes de Ponderacao

O forecastbox implementa quatro funcoes de ponderacao:

=== "Beta"

    A funcao Beta e a mais popular por sua flexibilidade:

    $$
    w(j; \theta_1, \theta_2) = \frac{j^{\theta_1 - 1}(1-j)^{\theta_2 - 1}}{\sum_{k=0}^{K-1} k^{\theta_1-1}(1-k)^{\theta_2-1}}
    $$

    onde $j$ e normalizado para $[0, 1]$.

    Propriedades:

    - $\theta_1 = 1, \theta_2 = 1$: pesos uniformes
    - $\theta_1 = 1, \theta_2 > 1$: pesos decrescentes (mais recente = mais peso)
    - $\theta_1 > 1, \theta_2 = 1$: pesos crescentes (hump-shaped)

    ```python
    midas = MIDAS(weight_function="beta")
    ```

=== "Almon Polinomial"

    Pesos seguem um polinomio exponencial:

    $$
    w(j; \theta_1, \theta_2) = \frac{\exp(\theta_1 j + \theta_2 j^2)}{\sum_{k=0}^{K-1} \exp(\theta_1 k + \theta_2 k^2)}
    $$

    Com `polynomial_order=3`, adiciona o termo $\theta_3 j^3$.

    ```python
    midas = MIDAS(weight_function="almon", polynomial_order=2)
    ```

=== "Exponential Almon"

    Variante da Almon com decaimento exponencial:

    $$
    w(j; \theta_1, \theta_2) = \frac{\exp(\theta_1 j + \theta_2 j^2)}{\sum_{k} \exp(\theta_1 k + \theta_2 k^2)}
    $$

    Restrita a $\theta_2 < 0$ para garantir decaimento.

    ```python
    midas = MIDAS(weight_function="exp_almon")
    ```

=== "Unrestricted (U-MIDAS)"

    Sem funcao de ponderacao — cada lag tem seu proprio coeficiente:

    $$
    y_t^Q = \alpha + \sum_{j=0}^{K-1} \beta_j \, x_{t-j/m}^M + \varepsilon_t
    $$

    !!! warning

        U-MIDAS so e viavel quando $K$ e pequeno (e.g., $K \leq 12$) ou
        com regularizacao. Com muitos lags, o modelo rapidamente perde
        graus de liberdade.

    ```python
    midas = MIDAS(weight_function="unrestricted", n_lags=9)
    ```

---

## Perfis de Peso

Visualizacao dos diferentes perfis de ponderacao:

```python
from forecastbox.nowcast import MIDAS

# Comparar funcoes de peso
midas_beta = MIDAS(weight_function="beta", n_lags=12).fit(data, target="pib")
midas_almon = MIDAS(weight_function="almon", n_lags=12).fit(data, target="pib")

midas_beta.plot_weights()
midas_almon.plot_weights()
```

```text
Weight Profile (Beta, θ₁=1.00, θ₂=3.42):

  Lag  0: ████████████████████  0.231
  Lag  1: ██████████████████    0.198
  Lag  2: ███████████████       0.164
  Lag  3: ████████████          0.131
  Lag  4: █████████             0.099
  ...
  Lag 11: █                     0.008
```

---

## Variantes

O forecastbox implementa tres variantes do MIDAS:

| Variante | Equacao | Uso |
|:---------|:--------|:----|
| **MIDAS** | $y_t^Q = \alpha + \beta \sum w(j;\theta) x_{t-j/m} + \varepsilon_t$ | Caso padrao |
| **MIDAS-AR** | $y_t^Q = \alpha + \rho y_{t-1}^Q + \beta \sum w(j;\theta) x_{t-j/m} + \varepsilon_t$ | Inclui lag do target |
| **U-MIDAS** | $y_t^Q = \alpha + \sum \beta_j x_{t-j/m} + \varepsilon_t$ | Sem restricao de peso |

```python
# MIDAS-AR com lag do target
midas_ar = MIDAS(
    weight_function="beta",
    n_lags=12,
    include_ar=True,
    ar_order=1,
)
midas_ar.fit(data, target="pib")
```

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `weight_function` | `str` | `"beta"` | Funcao de ponderacao: `"beta"`, `"almon"`, `"exp_almon"`, `"unrestricted"` |
| `n_lags` | `int` | `12` | Numero de lags de alta frequencia |
| `polynomial_order` | `int` | `2` | Ordem do polinomio (para Almon) |
| `include_ar` | `bool` | `False` | Incluir lag do target (MIDAS-AR) |
| `ar_order` | `int` | `1` | Ordem do componente AR |
| `frequency_ratio` | `int` | `None` | Razao de frequencias (auto-detectado se `None`) |
| `optimizer` | `str` | `"L-BFGS-B"` | Otimizador para NLS |

---

## Exemplo: MIDAS para PIB com Dados Mensais e Diarios

```python
import pandas as pd
from forecastbox.nowcast import MIDAS
from forecastbox.datasets import load_brazil_indicators

# Carregar indicadores de diferentes frequencias
monthly = load_brazil_indicators(frequency="M")  # prod. industrial, vendas, PMI
daily = load_brazil_indicators(frequency="D")     # energia, financeiro

# MIDAS com dados mensais (razao 3:1)
midas_monthly = MIDAS(
    weight_function="beta",
    n_lags=12,         # 12 meses = 4 trimestres
    include_ar=True,
)
midas_monthly.fit(monthly, target="pib")
nowcast_m = midas_monthly.predict(horizon=1)

# MIDAS com dados diarios (razao ~63:1)
midas_daily = MIDAS(
    weight_function="beta",
    n_lags=63,          # ~63 dias uteis por trimestre
    frequency_ratio=63,
)
midas_daily.fit(daily, target="pib")
nowcast_d = midas_daily.predict(horizon=1)

print(f"Nowcast (mensal): {nowcast_m.point:.2f}")
print(f"Nowcast (diario): {nowcast_d.point:.2f}")
```

```text
Nowcast (mensal): 0.82
Nowcast (diario): 0.79

MIDAS Summary:
  Weight function: Beta (θ₁=1.00, θ₂=2.87)
  Lags: 12 monthly | R²: 0.74
  AIC: -128.4 | BIC: -122.1
```

---

## Escolhendo a Funcao de Peso

| Funcao | Parametros | Flexibilidade | Quando Usar |
|:-------|:-----------|:--------------|:------------|
| **Beta** | 2 | Alta | Default — maioria dos casos |
| **Almon** | 2-3 | Media-Alta | Quando Beta nao converge |
| **Exp. Almon** | 2 | Media | Decaimento monotono garantido |
| **U-MIDAS** | $K$ | Maxima | Poucos lags ($K \leq 12$) |

!!! tip "Recomendacao"

    Comece com **Beta** (default). Se o perfil de peso estimado parecer
    irregular ou o otimizador nao convergir, tente **Almon**. Use **U-MIDAS**
    apenas quando a razao de frequencia e pequena (mensal→trimestral)
    e voce tem amostra longa.

---

## Referencias

- **Ghysels, E., Santa-Clara, P. & Valkanov, R.** (2004). "The MIDAS Touch: Mixed Data Sampling Regression Models." *CIRANO Working Paper*.
- **Ghysels, E., Sinko, A. & Valkanov, R.** (2007). "MIDAS Regressions: Further Results and New Directions." *Econometric Reviews*, 26(1), 53-90.
- **Foroni, C., Marcellino, M. & Schumacher, C.** (2015). "Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions with Unrestricted Lag Polynomials." *Journal of the Royal Statistical Society: Series A*, 178(1), 57-82.
- **Andreou, E., Ghysels, E. & Kourtellos, A.** (2013). "Should Macroeconomic Forecasters Use Daily Financial Data and How?" *Journal of Business & Economic Statistics*, 31(2), 240-251.
