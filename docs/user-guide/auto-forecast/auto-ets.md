---
title: AutoETS
description: Selecao automatica de modelos de suavizacao exponencial com taxonomia ETS, equacoes de estado e exemplos praticos.
---

# AutoETS

O `AutoETS` seleciona automaticamente o melhor modelo de **suavizacao exponencial**
(Exponential Smoothing - ETS) dentre 30 combinacoes possiveis de componentes de erro,
tendencia e sazonalidade. A selecao e feita via criterios de informacao.

---

## Taxonomia ETS

A familia ETS e parametrizada por tres componentes:

| Componente | Opcoes | Descricao |
|:-----------|:-------|:----------|
| **Error** (E) | A, M | Aditivo ou Multiplicativo |
| **Trend** (T) | N, A, A~d~, M, M~d~ | Nenhum, Aditivo, Aditivo Amortecido, Multiplicativo, Multiplicativo Amortecido |
| **Seasonal** (S) | N, A, M | Nenhum, Aditivo, Multiplicativo |

Isso gera $2 \times 5 \times 3 = 30$ combinacoes possiveis, cada uma representando
um modelo de suavizacao exponencial diferente.

### Modelos Classicos como Casos Especiais

| Modelo Classico | Notacao ETS | Componentes |
|:---------------|:-----------|:------------|
| Simple Exponential Smoothing | ETS(A,N,N) | Erro aditivo, sem tendencia, sem sazonalidade |
| Holt Linear | ETS(A,A,N) | Erro aditivo, tendencia aditiva |
| Holt-Winters Aditivo | ETS(A,A,A) | Todos os componentes aditivos |
| Holt-Winters Multiplicativo | ETS(M,A,M) | Erro e sazonalidade multiplicativos |
| Damped Trend | ETS(A,A~d~,N) | Tendencia aditiva amortecida |

---

## Equacoes de Estado

Cada modelo ETS e definido por um conjunto de **equacoes de estado** que descrevem
a evolucao do nivel ($\ell_t$), tendencia ($b_t$) e sazonalidade ($s_t$).

### ETS(A,A,A) — Holt-Winters Aditivo

$$
\begin{aligned}
y_t &= \ell_{t-1} + b_{t-1} + s_{t-m} + \varepsilon_t \\
\ell_t &= \ell_{t-1} + b_{t-1} + \alpha \varepsilon_t \\
b_t &= b_{t-1} + \beta \varepsilon_t \\
s_t &= s_{t-m} + \gamma \varepsilon_t
\end{aligned}
$$

onde $\alpha$, $\beta$, $\gamma$ sao os parametros de suavizacao para nivel, tendencia
e sazonalidade, respectivamente.

### ETS(M,A~d~,M) — Multiplicativo com Tendencia Amortecida

$$
y_t = (\ell_{t-1} + \phi b_{t-1}) s_{t-m} (1 + \varepsilon_t)
$$

Com equacoes de atualizacao:

$$
\begin{aligned}
\ell_t &= (\ell_{t-1} + \phi b_{t-1})(1 + \alpha \varepsilon_t) \\
b_t &= \phi b_{t-1} + \beta(\ell_{t-1} + \phi b_{t-1})\varepsilon_t \\
s_t &= s_{t-m}(1 + \gamma \varepsilon_t)
\end{aligned}
$$

onde $\phi \in (0,1)$ e o parametro de amortecimento. Quando $\phi = 1$, recupera-se
o modelo ETS(M,A,M) sem amortecimento.

!!! info "Erro Multiplicativo"

    Nos modelos com erro multiplicativo (M), $\varepsilon_t$ representa o erro
    relativo: $\varepsilon_t = (y_t - \hat{y}_t) / \hat{y}_t$. Isso implica que
    a variancia do erro e proporcional ao nivel da serie — adequado para series
    com variancia crescente (ex: vendas em crescimento).

### ETS(A,N,A) — Sazonal Aditivo sem Tendencia

$$
\begin{aligned}
y_t &= \ell_{t-1} + s_{t-m} + \varepsilon_t \\
\ell_t &= \ell_{t-1} + \alpha \varepsilon_t \\
s_t &= s_{t-m} + \gamma \varepsilon_t
\end{aligned}
$$

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `error_type` | `str \| None` | `None` | Tipo de erro: `"add"`, `"mul"` ou `None` (auto) |
| `trend_type` | `str \| None` | `None` | Tipo de tendencia: `"add"`, `"mul"`, `None` (auto) |
| `seasonal_type` | `str \| None` | `None` | Tipo de sazonalidade: `"add"`, `"mul"`, `None` (auto) |
| `damped` | `bool \| None` | `None` | Amortecimento da tendencia: `True`, `False` ou `None` (auto) |
| `seasonal_periods` | `int` | `1` | Periodo sazonal ($m$) |
| `information_criterion` | `str` | `"aicc"` | Criterio: `"aic"`, `"bic"`, `"aicc"` |
| `restrict` | `bool` | `True` | Restringir a modelos estaveis (parametros admissiveis) |
| `allow_multiplicative_trend` | `bool` | `False` | Permitir tendencia multiplicativa |
| `n_jobs` | `int` | `1` | Paralelismo na estimacao |

!!! note "Restricao de Modelos"

    Com `restrict=True` (padrao), o `AutoETS` descarta modelos instáveis onde os
    parametros de suavizacao ficam fora da regiao admissivel. Isso evita modelos
    com previsoes explosivas.

!!! warning "Tendencia Multiplicativa"

    A tendencia multiplicativa (`allow_multiplicative_trend=True`) pode gerar
    previsoes que crescem exponencialmente. Use com cautela e prefira a tendencia
    amortecida (`damped=True`) para horizontes longos.

---

## Selecao Automatica

Quando os parametros de tipo sao `None`, o `AutoETS` testa todas as combinacoes
aplicaveis e seleciona o melhor modelo via criterio de informacao:

```python
from forecastbox.auto import AutoETS

# Selecao totalmente automatica
ets = AutoETS(
    seasonal_periods=12,
    information_criterion="aicc",
)
ets.fit(y)

print(f"Modelo: ETS({ets.error_type_},{ets.trend_type_},{ets.seasonal_type_})")
print(f"Damped: {ets.damped_}")
print(f"AICc: {ets.aicc_:.1f}")
```

```text
Modelo: ETS(M,Ad,M)
Damped: True
AICc: 1498.7
```

### Selecao Parcialmente Restrita

Voce pode fixar um ou mais componentes e deixar os demais automaticos:

```python
# Fixar erro aditivo, selecionar tendencia e sazonalidade automaticamente
ets = AutoETS(
    error_type="add",        # fixo
    trend_type=None,         # auto
    seasonal_type=None,      # auto
    seasonal_periods=12,
)
ets.fit(y)
```

---

## Exemplos

### Serie de Vendas Mensais

```python
import pandas as pd
from forecastbox.auto import AutoETS

# Carregar serie de vendas no varejo
y = pd.read_csv("vendas_varejo.csv", index_col="date", parse_dates=True)["vendas"]

# AutoETS com sazonalidade mensal
model = AutoETS(seasonal_periods=12)
model.fit(y)

print(model.summary())
```

```text
AutoETS Summary
===============
Selected: ETS(M,Ad,M)
AICc: 2341.8  |  BIC: 2378.5

Smoothing Parameters:
  alpha (level):      0.312
  beta  (trend):      0.021
  gamma (seasonal):   0.187
  phi   (damping):    0.978

Initial States:
  l0: 1245.3
  b0: 12.4
  s0: [0.87, 0.92, 1.03, 1.08, 1.12, 1.15,
       1.11, 1.06, 0.98, 0.94, 0.89, 0.85]

Residual Diagnostics:
  Ljung-Box(12): Q=11.2, p=0.512
  MAPE(in-sample): 3.2%
```

```python
# Previsao 12 meses
forecast = model.predict(horizon=12, level=[80, 95])
print(forecast.head())
```

```text
             point     lo80     hi80     lo95     hi95
2024-01   1342.5   1278.4   1406.6   1245.2   1439.8
2024-02   1389.1   1312.7   1465.5   1273.1   1505.1
2024-03   1534.7   1441.2   1628.2   1393.0   1676.4
2024-04   1612.3   1504.8   1719.8   1449.5   1775.1
2024-05   1678.9   1558.1   1799.7   1496.0   1861.8
```

### Serie sem Tendencia

```python
# Serie de taxa de juros (estacionaria)
y_juros = pd.read_csv("selic.csv", index_col="date", parse_dates=True)["selic"]

model = AutoETS(
    seasonal_periods=12,
    allow_multiplicative_trend=False,
)
model.fit(y_juros)

print(f"Modelo: ETS({model.error_type_},{model.trend_type_},{model.seasonal_type_})")
```

```text
Modelo: ETS(A,N,A)
```

!!! tip "Sem Tendencia"

    Para series estacionarias (taxa de juros, spread, etc.), o `AutoETS` tipicamente
    seleciona modelos sem tendencia (N). Isso e esperado e indica que o algoritmo
    esta detectando corretamente a ausencia de tendencia.

### Serie com Sazonalidade Multiplicativa

```python
# Consumo de energia eletrica — amplitude sazonal cresce com o nivel
y_energia = pd.read_csv("consumo_energia.csv", index_col="date", parse_dates=True)["consumo"]

model = AutoETS(seasonal_periods=12)
model.fit(y_energia)

print(f"Modelo: ETS({model.error_type_},{model.trend_type_},{model.seasonal_type_})")
print(f"AICc: {model.aicc_:.1f}")
```

```text
Modelo: ETS(M,A,M)
AICc: 3456.2
```

!!! info "Quando usar Sazonalidade Multiplicativa"

    A sazonalidade multiplicativa e adequada quando a **amplitude sazonal** cresce
    proporcionalmente ao nivel da serie. Exemplos tipicos:

    - Consumo de energia eletrica
    - Vendas no varejo em crescimento
    - Trafego aereo de passageiros

    Se a amplitude sazonal e constante independente do nivel, prefira sazonalidade
    aditiva.

---

## Comparacao de Todos os Modelos

O `AutoETS` armazena os resultados de todos os modelos avaliados:

```python
# Tabela com todos os modelos avaliados
results = model.results_table_
print(results.head(10))
```

```text
        Model     AICc      BIC  Converged
0   ETS(M,Ad,M)  2341.8  2378.5       True
1   ETS(M,A,M)   2343.1  2377.9       True
2   ETS(A,Ad,A)  2348.5  2385.2       True
3   ETS(A,A,A)   2349.8  2384.6       True
4   ETS(M,Ad,A)  2352.3  2389.0       True
5   ETS(A,Ad,M)  2354.1  2390.8       True
6   ETS(M,N,M)   2367.4  2396.3       True
7   ETS(A,N,A)   2369.2  2398.1       True
8   ETS(M,Md,M)  2371.8  2408.5       True
9   ETS(A,N,M)   2378.6  2407.5       True
```

---

## ETS vs ARIMA

| Aspecto | ETS | ARIMA |
|:--------|:----|:------|
| **Abordagem** | Decomposicao em componentes | Autocorrelacao |
| **Interpretabilidade** | Alta — nivel, tendencia, sazonalidade | Media — coeficientes AR/MA |
| **Series nao estacionarias** | Via tendencia | Via diferenciacao |
| **Sazonalidade** | Aditiva ou multiplicativa | Via parte sazonal |
| **Previsao de longo prazo** | Tende a ser mais estavel com damping | Pode reverter a media rapido |
| **Melhor para** | Series com padroes claros de decomposicao | Series com autocorrelacao complexa |

!!! tip "Combine os Dois"

    Na pratica, combinar previsoes de AutoARIMA e AutoETS via
    [combinacao](../combination/index.md) frequentemente supera cada modelo
    individual. O modulo de combinacao do forecastbox facilita isso:

    ```python
    from forecastbox import combine

    combined = combine(
        forecasts=[forecast_arima, forecast_ets],
        method="bma",
        actual=y_test,
    )
    ```

---

## Proximos Passos

- **[AutoARIMA](auto-arima.md)** — selecao automatica de modelos ARIMA sazonais
- **[Combinacao](../combination/index.md)** — combine ETS com ARIMA e outros modelos
- **[Avaliacao](../evaluation/index.md)** — compare modelos com testes estatisticos
