---
title: "Teste Diebold-Mariano"
description: "Teste de igualdade de poder preditivo entre dois modelos, com correcao HLN e diferentes funcoes de perda."
---

# Teste Diebold-Mariano

!!! abstract "Key Takeaway"
    O teste Diebold-Mariano (1995) responde formalmente a pergunta: **a diferenca de performance entre dois modelos e estatisticamente significativa?** Sem esse teste, comparar modelos apenas pelo RMSE pode levar a conclusoes erroneas.

## Formulacao

Dados dois conjuntos de erros de previsao $\{e_{1t}\}$ e $\{e_{2t}\}$, defina o diferencial de perda:

$$
d_t = L(e_{1t}) - L(e_{2t})
$$

onde $L(\cdot)$ e uma funcao de perda (e.g., quadratica, absoluta).

### Hipotese

$$
H_0: E[d_t] = 0 \quad \text{(Equal Predictive Ability)}
$$

$$
H_1: E[d_t] \neq 0 \quad \text{(bilateral)}
$$

### Estatistica de Teste

$$
\text{DM} = \frac{\bar{d}}{\hat{\sigma}_{\bar{d}}} \xrightarrow{d} N(0,1)
$$

onde:

- $\bar{d} = \frac{1}{T} \sum_{t=1}^{T} d_t$
- $\hat{\sigma}_{\bar{d}}^2 = \frac{1}{T} \left[ \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \hat{\gamma}_k \right]$
- $\hat{\gamma}_k = \frac{1}{T} \sum_{t=k+1}^{T} (d_t - \bar{d})(d_{t-k} - \bar{d})$
- $h$ e o horizonte de previsao

A variancia de longo prazo $\hat{\sigma}_{\bar{d}}^2$ usa um estimador HAC (Newey-West) com truncamento em $h-1$ para acomodar autocorrelacao nos erros multi-step.

## Correcao Harvey-Leybourne-Newbold

Harvey, Leybourne & Newbold (1997) mostraram que o teste DM original tem distorcao de tamanho em amostras finitas. A correcao ajusta a estatistica e usa a distribuicao $t$:

$$
\text{DM}^* = \sqrt{\frac{T + 1 - 2h + h(h-1)/T}{T}} \cdot \text{DM}
$$

$$
\text{DM}^* \sim t_{T-1}
$$

!!! tip "Use sempre a correcao HLN"
    Para amostras tipicas em macroeconomia ($T < 200$), a correcao HLN melhora substancialmente o controle de tamanho do teste. O forecastbox aplica a correcao por padrao.

## Funcoes de Perda

=== "Quadratica"

    $$
    L(e_t) = e_t^2
    $$

    A mais comum. Equivale a comparar MSE.

    ```python
    dm = diebold_mariano(e1, e2, loss="squared", h=1)
    ```

=== "Absoluta"

    $$
    L(e_t) = |e_t|
    $$

    Robusta a outliers. Equivale a comparar MAE.

    ```python
    dm = diebold_mariano(e1, e2, loss="absolute", h=1)
    ```

=== "LINEX (Assimetrica)"

    $$
    L(e_t) = \exp(a \cdot e_t) - a \cdot e_t - 1
    $$

    Para situacoes onde sub- e sobre-previsao tem custos diferentes. $a > 0$ penaliza mais sobre-previsao; $a < 0$ penaliza mais sub-previsao.

    ```python
    dm = diebold_mariano(e1, e2, loss="linex", a=0.5, h=1)
    ```

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `e1` | array | — | Erros de previsao do modelo 1 |
| `e2` | array | — | Erros de previsao do modelo 2 |
| `loss` | str | `"squared"` | Funcao de perda: `"squared"`, `"absolute"`, `"linex"` |
| `h` | int | `1` | Horizonte de previsao (para ajuste HAC) |
| `correction` | bool | `True` | Aplicar correcao HLN |
| `alternative` | str | `"two-sided"` | `"two-sided"`, `"less"`, `"greater"` |

## Exemplo: ARIMA vs ETS para PIB

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.evaluation import diebold_mariano

# Dados do PIB trimestral
y = pd.read_csv("gdp_quarterly.csv", index_col=0, parse_dates=True).squeeze()
train, test = y[:"2019"], y["2020":]

# Modelos
arima_pred = AutoARIMA().fit(train).predict(len(test))
ets_pred = AutoETS().fit(train).predict(len(test))

# Erros
e_arima = test.values - arima_pred.values
e_ets = test.values - ets_pred.values

# Teste DM (horizonte 1, perda quadratica)
dm = diebold_mariano(e_arima, e_ets, loss="squared", h=1)
print(dm)
```

```text
Diebold-Mariano Test
====================
Statistic (DM*):  -2.147
P-value:           0.0384
Loss function:     squared
Horizon (h):       1
HLN correction:    True
Alternative:       two-sided

Conclusion: Rejeita H0 a 5%. Modelo 2 (ETS) tem desempenho
            significativamente diferente do Modelo 1 (ARIMA).
```

### Interpretacao

- $\text{DM}^* < 0$: modelo 2 (ETS) tem menor perda media
- $p < 0.05$: a diferenca e significativa a 5%
- **Conclusao**: ETS supera ARIMA nesta amostra com significancia estatistica

### Teste unilateral

Para testar se ARIMA e *melhor* que ETS (nao apenas diferente):

```python
dm_one = diebold_mariano(e_arima, e_ets, loss="squared", h=1, alternative="less")
print(f"DM* = {dm_one.statistic:.3f}, p = {dm_one.pvalue:.4f}")
```

!!! warning "Cuidados"
    - O teste assume erros **estacionarios**. Se a serie tem quebras estruturais, considere o teste [Giacomini-White](giacomini-white.md).
    - Para $h > 1$, os erros sao autocorrelacionados por construcao — o ajuste HAC e essencial.
    - O teste **nao** e valido para modelos nested (e.g., AR(1) vs AR(2)). Use [Encompassing](encompassing.md) nesses casos.

## Ver Tambem

- [Model Confidence Set](mcs.md) — comparacao multipla (mais de 2 modelos)
- [Giacomini-White](giacomini-white.md) — alternativa que funciona com janelas rolling
- [Metricas](metrics.md) — funcoes de perda e metricas utilizadas
- :material-stethoscope: [DM Test — Diagnostico](../../diagnostics/dm-test.md) — workflow pratico de diagnostico com loss differentials, armadilhas e visualizacao
