---
title: "Teste de Encompassing"
description: "Teste Fair-Shiller de encompassing para verificar se um modelo agrega informacao preditiva sobre outro."
---

# Teste de Encompassing

!!! abstract "Key Takeaway"
    O teste de encompassing (Fair & Shiller, 1990) verifica se a previsao de um modelo **agrega informacao** alem do que outro modelo ja captura. Se o modelo 1 encompassa o modelo 2, nao ha ganho em combinar — caso contrario, a combinacao pode melhorar a previsao.

## Formulacao

### Versao Bivariada

Regride o valor realizado nas previsoes de ambos os modelos:

$$
y_t = \lambda \hat{y}_{1t} + (1 - \lambda) \hat{y}_{2t} + \varepsilon_t
$$

Equivalentemente, reescrevendo:

$$
y_t - \hat{y}_{2t} = \lambda (\hat{y}_{1t} - \hat{y}_{2t}) + \varepsilon_t
$$

### Hipoteses

| Hipotese | Significado |
|----------|-------------|
| $H_0: \lambda = 1$ | Modelo 1 **encompassa** modelo 2 (modelo 2 nao agrega nada) |
| $H_0: \lambda = 0$ | Modelo 2 **encompassa** modelo 1 (modelo 1 nao agrega nada) |
| $0 < \lambda < 1$ | **Nenhum modelo encompassa** o outro — combinacao agrega valor |

### Estatistica de Teste

Para testar $H_0: \lambda = 1$ (modelo 1 encompassa modelo 2):

$$
t = \frac{\hat{\lambda} - 1}{\hat{\sigma}_{\hat{\lambda}}} \sim t_{T-1}
$$

onde $\hat{\sigma}_{\hat{\lambda}}$ usa estimador HAC para erros autocorrelacionados.

!!! note "Relacao com combinacao"
    O $\hat{\lambda}$ otimo e exatamente o **peso otimo de combinacao** entre os dois modelos. Se $\hat{\lambda} \approx 1$, use apenas o modelo 1. Se $\hat{\lambda} \approx 0.5$, a media simples e uma boa aproximacao.

## Versao Multivariada

Para $K$ modelos, a regressao se estende a:

$$
y_t = \sum_{k=1}^{K} \lambda_k \hat{y}_{kt} + \varepsilon_t
$$

com a restricao $\sum_{k=1}^{K} \lambda_k = 1$.

O teste conjunto:

$$
H_0: \lambda_j = 0 \quad \text{(modelo } j \text{ nao agrega informacao)}
$$

e realizado via teste de Wald com estatistica $\chi^2(1)$.

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `actual` | array | — | Valores realizados $y_t$ |
| `pred_1` | array | — | Previsoes do modelo 1 |
| `pred_2` | array | — | Previsoes do modelo 2 |
| `cov_type` | str | `"HAC"` | Tipo de covariancia |
| `h` | int | `1` | Horizonte de previsao |

## Exemplo: Testar se Combinacao Agrega Valor

```python
import pandas as pd
from forecastbox.evaluation import encompassing_test

# Previsoes de dois modelos
actual = test_data.values
pred_arima = arima_forecast.values
pred_ets = ets_forecast.values

# Teste: ARIMA encompassa ETS?
enc = encompassing_test(actual, pred_arima, pred_ets, h=1)
print(enc)
```

```text
Forecast Encompassing Test (Fair-Shiller)
==========================================
y_t = lambda * y1_hat + (1 - lambda) * y2_hat + eps_t

lambda estimate:   0.6234
Std. Error:        0.1456

Test H0: lambda = 1 (Model 1 encompasses Model 2)
  t-statistic:    -2.586
  P-value:         0.0132

Test H0: lambda = 0 (Model 2 encompasses Model 1)
  t-statistic:     4.282
  P-value:         0.0001

Conclusion: Nenhum modelo encompassa o outro.
            Combinacao com peso ~0.62 para Modelo 1 e recomendada.
```

### Interpretacao

- $\hat{\lambda} = 0.62$: peso otimo de 62% para ARIMA, 38% para ETS
- $H_0: \lambda = 1$ rejeitada ($p = 0.013$): ETS agrega informacao alem do ARIMA
- $H_0: \lambda = 0$ rejeitada ($p < 0.001$): ARIMA agrega informacao alem do ETS
- **Conclusao**: nenhum modelo e suficiente sozinho — a combinacao e justificada

### Exemplo com encompassing

```python
# Agora testando Modelo A (forte) vs Modelo B (fraco)
enc2 = encompassing_test(actual, pred_strong, pred_weak, h=1)
print(enc2)
```

```text
Forecast Encompassing Test (Fair-Shiller)
==========================================
y_t = lambda * y1_hat + (1 - lambda) * y2_hat + eps_t

lambda estimate:   0.9512
Std. Error:        0.0834

Test H0: lambda = 1 (Model 1 encompasses Model 2)
  t-statistic:    -0.585
  P-value:         0.5621

Test H0: lambda = 0 (Model 2 encompasses Model 1)
  t-statistic:    11.405
  P-value:         0.0000

Conclusion: Modelo 1 encompassa Modelo 2.
            Nao ha ganho em combinar — use apenas Modelo 1.
```

!!! tip "Quando usar encompassing vs DM"
    - **DM**: "Qual modelo e melhor?" — comparacao de performance
    - **Encompassing**: "Combinar agrega valor?" — decisao sobre combinacao
    - Use encompassing *antes* de decidir se combina modelos

## Versao Multivariada — Exemplo

```python
from forecastbox.evaluation import encompassing_test_multi

# 4 modelos candidatos
preds = pd.DataFrame({
    "ARIMA": pred_arima,
    "ETS": pred_ets,
    "VAR": pred_var,
    "Theta": pred_theta,
})

enc_multi = encompassing_test_multi(actual, preds, h=1)
print(enc_multi)
```

```text
Multivariate Encompassing Test
===============================
y_t = sum(lambda_k * y_k_hat) + eps_t

Model      lambda    Std.Err    t-stat   p-value   Significant
───────────────────────────────────────────────────────────────
ARIMA       0.412     0.132     3.121    0.0032       *
ETS         0.351     0.118     2.975    0.0048       *
VAR         0.198     0.145     1.366    0.1801
Theta       0.039     0.098     0.398    0.6932

Conclusion: VAR e Theta nao agregam informacao significativa.
            Combinacao de ARIMA + ETS e suficiente.
```

## Ver Tambem

- [Diebold-Mariano](diebold-mariano.md) — comparacao de performance (nao encompassing)
- [Mincer-Zarnowitz](mincer-zarnowitz.md) — eficiencia individual (antes de testar encompassing)
- [Combinacao de Previsoes](../combination/index.md) — metodos de combinacao quando encompassing falha
- :material-stethoscope: [Encompassing — Diagnostico](../../diagnostics/encompassing-test.md) — diagnostico pratico com matriz de encompassing e teste de exclusao sequencial
