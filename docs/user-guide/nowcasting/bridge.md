---
title: Bridge Equations
description: Nowcasting com bridge equations - regressao do target trimestral em indicadores mensais com agregacao temporal.
---

# Bridge Equations

Bridge equations sao o metodo mais intuitivo de nowcasting: uma **regressao linear**
do target trimestral (e.g., PIB) em indicadores mensais agregados temporalmente.
O nome vem da ideia de construir uma "ponte" entre a frequencia mensal dos
indicadores e a frequencia trimestral do target.

---

## Conceito

A logica e simples:

1. **Agregar** os indicadores mensais para frequencia trimestral
2. **Estimar** uma regressao do target nos indicadores agregados
3. **Prever** os meses faltantes dos indicadores (ragged edge)
4. **Projetar** o target usando a regressao estimada

!!! info "Vantagem Principal"

    Bridge equations sao **facilmente comunicaveis** para decisores e policy-makers.
    Cada coeficiente tem interpretacao direta: "um aumento de 1% na producao
    industrial esta associado a um aumento de $\beta$ pp no PIB trimestral."

---

## Modelo

A equacao de bridge relaciona o target trimestral aos indicadores mensais
agregados:

$$
y_t^Q = \alpha + \beta_1 \bar{x}_{1,t}^M + \beta_2 \bar{x}_{2,t}^M + \cdots + \beta_k \bar{x}_{k,t}^M + \varepsilon_t
$$

onde:

| Simbolo | Descricao |
|:--------|:----------|
| $y_t^Q$ | Target trimestral (e.g., crescimento do PIB) |
| $\bar{x}_{i,t}^M$ | Indicador $i$ agregado para frequencia trimestral |
| $\alpha$ | Intercepto |
| $\beta_i$ | Coeficiente do indicador $i$ |
| $k$ | Numero de indicadores |

---

## Agregacao Temporal

A agregacao transforma dados mensais em trimestrais. O forecastbox suporta
tres metodos:

=== "Media"

    $$\bar{x}_t^Q = \frac{1}{3}(x_{3t-2}^M + x_{3t-1}^M + x_{3t}^M)$$

    Indicado para **variaveis de fluxo em taxa** (inflacao, taxa de desemprego)
    e **indicadores de sentimento** (PMI, confianca).

    ```python
    bridge = BridgeEquation(aggregation="mean")
    ```

=== "Soma"

    $$\bar{x}_t^Q = x_{3t-2}^M + x_{3t-1}^M + x_{3t}^M$$

    Indicado para **variaveis de fluxo em nivel** (vendas em reais, producao
    em unidades).

    ```python
    bridge = BridgeEquation(aggregation="sum")
    ```

=== "Fim de Periodo"

    $$\bar{x}_t^Q = x_{3t}^M$$

    Indicado para **variaveis de estoque** (credito total, divida publica,
    reservas internacionais).

    ```python
    bridge = BridgeEquation(aggregation="last")
    ```

!!! note

    Voce pode especificar agregacoes diferentes para cada indicador usando
    um dicionario:

    ```python
    bridge = BridgeEquation(
        aggregation={
            "prod_industrial": "mean",
            "vendas_varejo": "sum",
            "credito": "last",
        }
    )
    ```

---

## Previsao dos Indicadores Faltantes

No ragged edge, alguns meses do trimestre corrente ainda nao foram publicados.
O forecastbox preve automaticamente os valores faltantes antes de agregar:

=== "AR"

    Previsto por um modelo AR(p) estimado na propria serie do indicador.

    ```python
    bridge = BridgeEquation(indicator_forecast="ar")
    ```

=== "Random Walk"

    O ultimo valor observado e repetido para os meses faltantes.

    ```python
    bridge = BridgeEquation(indicator_forecast="rw")
    ```

=== "Custom"

    O usuario fornece previsoes externas para os indicadores.

    ```python
    bridge = BridgeEquation(indicator_forecast="custom")
    bridge.fit(data, target="pib")
    nowcast = bridge.predict(
        indicator_forecasts={"prod_industrial": [102.5, 103.1]}
    )
    ```

---

## Selecao de Indicadores

Com muitos indicadores candidatos, o forecastbox oferece metodos de selecao:

| Metodo | Descricao | Parametro |
|:-------|:----------|:----------|
| **Forward** | Adiciona indicadores sequencialmente pelo maior ganho de R² | `selection="forward"` |
| **Backward** | Remove indicadores pelo menor impacto no R² | `selection="backward"` |
| **LASSO** | Regularizacao L1 para selecao esparsa | `selection="lasso"` |
| **BIC** | Seleciona o subconjunto que minimiza BIC | `selection="bic"` |
| **Manual** | O usuario define os indicadores | `selection=None` |

```python
bridge = BridgeEquation(
    selection="bic",
    max_indicators=5,
)
bridge.fit(data, target="pib")
print(bridge.selected_indicators_)
```

```text
Selected indicators (BIC):
  1. prod_industrial    (BIC: -124.5)
  2. vendas_varejo      (BIC: -131.2)
  3. pmi_industria      (BIC: -133.8)
  4. emprego_formal     (BIC: -134.1)
```

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `aggregation` | `str` ou `dict` | `"mean"` | Metodo de agregacao temporal |
| `indicator_forecast` | `str` | `"ar"` | Previsao de indicadores faltantes: `"ar"`, `"rw"`, `"custom"` |
| `selection` | `str` ou `None` | `None` | Metodo de selecao: `"forward"`, `"backward"`, `"lasso"`, `"bic"` |
| `max_indicators` | `int` | `10` | Maximo de indicadores na selecao automatica |
| `ar_order` | `int` | `4` | Ordem do AR para previsao de indicadores |
| `include_lags` | `bool` | `False` | Incluir lags do target na equacao |

---

## Exemplo: Bridge para PIB com Producao Industrial e Vendas

```python
import pandas as pd
from forecastbox.nowcast import BridgeEquation
from forecastbox.datasets import load_brazil_indicators

# Carregar dados
data = load_brazil_indicators()

# Configurar bridge equation
bridge = BridgeEquation(
    aggregation={
        "prod_industrial": "mean",
        "vendas_varejo": "mean",
        "pmi_industria": "mean",
    },
    indicator_forecast="ar",
    selection="bic",
)

# Estimar
bridge.fit(data, target="pib")

# Nowcast
nowcast = bridge.predict(horizon=1)
print(nowcast)
```

```text
Bridge Equation Nowcast (target=pib)

  Quarter    Nowcast    Lo95    Hi95
  2024-Q2      0.85    0.42    1.28

  Equation: pib = 0.12 + 0.45*prod_industrial + 0.31*vendas_varejo
  R² = 0.78 | BIC = -134.1
  Missing months filled: prod_industrial(1), vendas_varejo(2)
```

---

## Atualizacao Sequencial

A medida que novos dados sao publicados ao longo do trimestre, o nowcast e
atualizado automaticamente:

```python
# Primeira estimativa (apenas 1 mes do trimestre)
nowcast_m1 = bridge.predict(vintage="2024-04-15")

# Segunda estimativa (2 meses disponiveis)
nowcast_m2 = bridge.predict(vintage="2024-05-15")

# Terceira estimativa (trimestre completo)
nowcast_m3 = bridge.predict(vintage="2024-06-15")

print(f"M1: {nowcast_m1.point:.2f} | M2: {nowcast_m2.point:.2f} | M3: {nowcast_m3.point:.2f}")
```

```text
M1: 0.72 | M2: 0.85 | M3: 0.91
```

A precisao tipicamente melhora a medida que mais dados ficam disponiveis.

---

## Bridge vs DFM

| Aspecto | Bridge | DFM |
|:--------|:-------|:----|
| Interpretabilidade | Alta — coeficientes claros | Baixa — fatores latentes |
| Numero de indicadores | Poucos (3-8) | Muitos (20+) |
| Missing data | Requer previsao auxiliar | Kalman trata naturalmente |
| Facilidade de comunicacao | Facil | Dificil |
| Performance com painel grande | Limitada | Excelente |

!!! tip "Quando Escolher Bridge"

    Prefira bridge equations quando voce tem **poucos indicadores confiaves**
    e precisa **comunicar os resultados** para audiencias nao-tecnicas.
    A transparencia dos coeficientes e uma grande vantagem em bancos centrais
    e consultorias.

---

## Referencias

- **Baffigi, A., Golinelli, R. & Parigi, G.** (2004). "Bridge Models to Forecast the Euro Area GDP." *International Journal of Forecasting*, 20(3), 447-460.
- **Hahn, E. & Skudelny, F.** (2008). "Early Estimates of Euro Area Real GDP Growth: A Bottom Up Approach from the Production Side." *ECB Working Paper*, No. 975.
- **Kitchen, J. & Monaco, R.** (2003). "Real-Time Forecasting in Practice." *Business Economics*, 38(4), 10-19.
