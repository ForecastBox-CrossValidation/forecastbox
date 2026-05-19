---
title: "Teste Giacomini-White"
description: "Teste de superioridade preditiva condicional com instrumentos, robusto a estimacao de parametros e janelas rolling."
---

# Teste Giacomini-White

!!! abstract "Key Takeaway"
    O teste Giacomini-White (2006) estende o Diebold-Mariano para contextos mais realistas: funciona com **parametros estimados**, **janelas rolling**, e pode testar se a superioridade e **condicional** ao estado da economia.

## Motivacao

O teste DM assume que os modelos sao fixos (parametros conhecidos). Na pratica:

- Modelos sao **re-estimados** a cada ponto do tempo
- Janelas **rolling** ou **expanding** sao usadas
- A superioridade pode ser **condicional** — um modelo pode ser melhor em recessoes e pior em expansoes

O teste GW acomoda todas essas situacoes.

## Formulacao

### Teste Incondicional

Defina o diferencial de perda como no DM:

$$
d_t = L(e_{1t}) - L(e_{2t})
$$

O teste incondicional GW e identico ao DM, mas com validade formal para modelos com parametros estimados em janelas rolling de tamanho fixo $m$:

$$
H_0: E[d_t] = 0
$$

### Teste Condicional

A grande contribuicao do GW e o teste **condicional**:

$$
H_0: E[d_t \mid \mathcal{F}_{t-1}] = 0 \quad \text{quase certamente}
$$

onde $\mathcal{F}_{t-1}$ e o conjunto de informacao disponivel em $t-1$.

Isso e operacionalizado via regressao:

$$
d_t = \mathbf{z}_{t-1}' \boldsymbol{\delta} + u_t
$$

onde $\mathbf{z}_{t-1}$ e um vetor de **instrumentos** (variaveis observaveis em $t-1$).

### Estatistica de Teste

$$
\text{GW} = T \cdot R^2_{\text{reg}} \sim \chi^2(q)
$$

onde $q$ e o numero de instrumentos e $R^2_{\text{reg}}$ vem da regressao auxiliar de $d_t$ em $\mathbf{z}_{t-1}$.

Alternativamente, usando a formulacao de Wald:

$$
\text{GW} = T \cdot \hat{\boldsymbol{\delta}}' \hat{\mathbf{V}}^{-1} \hat{\boldsymbol{\delta}} \sim \chi^2(q)
$$

onde $\hat{\mathbf{V}}$ e um estimador HAC da variancia de $\hat{\boldsymbol{\delta}}$.

## Instrumentos

A escolha dos instrumentos determina o que o teste verifica:

=== "Constante"

    $$
    \mathbf{z}_{t-1} = [1]
    $$

    Equivalente ao teste DM incondicional.

    ```python
    gw = giacomini_white(e1, e2, instruments="constant")
    ```

=== "Loss passada"

    $$
    \mathbf{z}_{t-1} = [1, \, d_{t-1}]
    $$

    Testa se a superioridade relativa e previsivel pela perda recente.

    ```python
    gw = giacomini_white(e1, e2, instruments="lagged_loss")
    ```

=== "Variaveis exogenas"

    $$
    \mathbf{z}_{t-1} = [1, \, x_{1,t-1}, \, x_{2,t-1}, \ldots]
    $$

    Testa se a superioridade depende de variaveis macroeconomicas (e.g., indicadores de recessao, volatilidade).

    ```python
    gw = giacomini_white(
        e1, e2,
        instruments=instruments_df  # DataFrame com variaveis
    )
    ```

!!! note "Escolha de instrumentos"
    - **Constante**: quando quer apenas comparar performance media (robusto a estimacao)
    - **Loss passada**: quando suspeita que a superioridade e persistente
    - **Exogenas**: quando quer identificar *quando* cada modelo e melhor

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `e1` | array | — | Erros de previsao do modelo 1 |
| `e2` | array | — | Erros de previsao do modelo 2 |
| `loss` | str | `"squared"` | Funcao de perda |
| `instruments` | str/DataFrame | `"constant"` | Instrumentos para o teste |
| `test_type` | str | `"conditional"` | `"unconditional"` ou `"conditional"` |
| `h` | int | `1` | Horizonte de previsao |

## Exemplo: Superioridade Condicional

```python
import pandas as pd
from forecastbox.evaluation import giacomini_white

# Erros de previsao de dois modelos
e_arima = test.values - pred_arima.values
e_ets = test.values - pred_ets.values

# Teste incondicional
gw_unc = giacomini_white(
    e_arima, e_ets,
    test_type="unconditional",
    h=1
)
print(gw_unc)
```

```text
Giacomini-White Test (Unconditional)
====================================
Statistic:   4.612
P-value:     0.0318
df:          1
Instruments: constant

Conclusion: Rejeita H0 a 5%. Os modelos diferem em poder preditivo.
```

```python
# Teste condicional com indicador de recessao e volatilidade
instruments = pd.DataFrame({
    "const": 1,
    "recession": recession_indicator.shift(1),
    "volatility": realized_vol.shift(1)
}).dropna()

gw_cond = giacomini_white(
    e_arima, e_ets,
    instruments=instruments,
    test_type="conditional",
    h=1
)
print(gw_cond)
```

```text
Giacomini-White Test (Conditional)
===================================
Statistic:   11.287
P-value:     0.0103
df:          3
Instruments: const, recession, volatility

Conclusion: Rejeita H0 a 5%. A superioridade preditiva e condicional
            ao estado da economia.
```

### Interpretacao

- O teste incondicional rejeita: ETS e ARIMA diferem em performance media
- O teste condicional rejeita com estatistica *maior*: a superioridade nao e constante — depende do regime economico
- Os coeficientes da regressao auxiliar indicam *quais* variaveis explicam a alternancia de superioridade

```python
# Coeficientes da regressao auxiliar
print(gw_cond.coefficients)
```

```text
             coef     se      t     p-value
const      -0.012  0.005  -2.40     0.018
recession   0.031  0.012   2.58     0.011
volatility -0.008  0.004  -2.00     0.048
```

!!! info "Leitura dos coeficientes"
    - **recession > 0**: em recessao, ARIMA perde mais que ETS (ETS e melhor)
    - **volatility < 0**: em alta volatilidade, ARIMA perde menos (ARIMA e melhor)
    - Isso sugere que uma **combinacao condicional** pode explorar ambos

## GW vs DM

| Caracteristica | Diebold-Mariano | Giacomini-White |
|----------------|-----------------|-----------------|
| Parametros estimados | Nao valido formalmente | Valido |
| Janela rolling | Nao valido formalmente | Valido |
| Teste condicional | Nao | Sim |
| Modelos nested | Nao | Nao |
| Complexidade | Simples | Moderada |

## Ver Tambem

- [Diebold-Mariano](diebold-mariano.md) — teste mais simples quando parametros sao fixos
- [Model Confidence Set](mcs.md) — comparacao de muitos modelos simultaneamente
- [Encompassing](encompassing.md) — teste para modelos nested
- :material-stethoscope: [GW Test — Diagnostico](../../diagnostics/gw-test.md) — diagnostico pratico de superioridade condicional, instabilidade e escolha de instrumentos
