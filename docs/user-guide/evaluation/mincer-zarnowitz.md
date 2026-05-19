---
title: "Regressao Mincer-Zarnowitz"
description: "Teste de vies e eficiencia de previsoes via regressao Mincer-Zarnowitz, com interpretacao de resultados."
---

# Regressao Mincer-Zarnowitz

!!! abstract "Key Takeaway"
    A regressao Mincer-Zarnowitz testa se uma previsao e **nao-viesada** ($\alpha = 0$) e **eficiente** ($\beta = 1$). E o primeiro diagnostico a rodar em qualquer exercicio de previsao — antes de comparar modelos, verifique se cada um faz sentido individualmente.

## Formulacao

Regride os valores realizados nos valores previstos:

$$
y_t = \alpha + \beta \hat{y}_t + \varepsilon_t
$$

### Hipoteses

**Teste conjunto** (previsao otima):

$$
H_0: \alpha = 0 \text{ e } \beta = 1
$$

Sob $H_0$, a previsao e **nao-viesada** e **eficiente** — nao ha informacao sistematica nos erros que poderia melhorar a previsao.

**Testes individuais**:

| Hipotese | Interpretacao |
|----------|---------------|
| $H_0: \alpha = 0$ | Previsao nao tem vies de nivel |
| $H_0: \beta = 1$ | Previsao nao tem vies de escala |

### Interpretacao dos Coeficientes

| Resultado | Significado | Acao |
|-----------|-------------|------|
| $\alpha > 0$ | Previsao sistematicamente **baixa** | Adicionar constante de correcao |
| $\alpha < 0$ | Previsao sistematicamente **alta** | Subtrair constante de correcao |
| $\beta > 1$ | Previsao **subestima** a variacao | Ampliar a previsao |
| $\beta < 1$ | Previsao **superestima** a variacao | Comprimir a previsao em direcao a media |
| $\alpha = 0, \beta = 1$ | Previsao **otima** | Nenhuma correcao necessaria |

!!! note "Eficiencia fraca vs forte"
    A regressao MZ testa **eficiencia fraca** — a previsao e otima dado apenas ela mesma. Eficiencia forte requer que nenhuma informacao adicional (outras previsoes, variaveis macro) melhore a previsao. Para isso, use o teste de [Encompassing](encompassing.md).

## Estatistica do Teste Conjunto

O teste conjunto $H_0: \alpha = 0, \beta = 1$ e um teste de Wald:

$$
F = \frac{(\mathbf{R}\hat{\boldsymbol{\beta}} - \mathbf{r})' [\mathbf{R} \hat{\mathbf{V}} \mathbf{R}']^{-1} (\mathbf{R}\hat{\boldsymbol{\beta}} - \mathbf{r})}{q} \sim F(q, T-2)
$$

onde $\mathbf{R} = \mathbf{I}_2$, $\mathbf{r} = [0, 1]'$, $q = 2$, e $\hat{\mathbf{V}}$ e um estimador HAC da variancia para acomodar autocorrelacao.

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `actual` | array | — | Valores realizados $y_t$ |
| `predicted` | array | — | Valores previstos $\hat{y}_t$ |
| `cov_type` | str | `"HAC"` | Tipo de covariancia: `"OLS"`, `"HC"`, `"HAC"` |
| `max_lags` | int | `None` | Lags para HAC (auto se `None`) |

## Exemplo

```python
import pandas as pd
from forecastbox.evaluation import mincer_zarnowitz

# Valores realizados e previstos
actual = pd.Series([2.1, 1.8, 3.2, 2.5, 1.9, 3.1, 2.8, 2.0, 3.5, 2.3])
predicted = pd.Series([2.0, 1.9, 2.8, 2.4, 2.1, 2.7, 2.6, 2.2, 3.0, 2.4])

# Regressao Mincer-Zarnowitz
mz = mincer_zarnowitz(actual, predicted)
print(mz)
```

```text
Mincer-Zarnowitz Regression
============================
y_t = alpha + beta * y_hat_t + eps_t

           Estimate    Std.Err    t-stat    p-value
alpha       -0.2143     0.3521    -0.609     0.5601
beta         1.1071     0.1423     7.779     0.0001

Joint test (alpha=0, beta=1):
  F-statistic:  1.234
  P-value:      0.3412

R-squared: 0.8831

Conclusion: Nao rejeita H0 a 5%. A previsao e consistente com
            ausencia de vies e eficiencia.
```

### Interpretacao

- $\hat{\alpha} = -0.21$ nao e significativo ($p = 0.56$): sem vies de nivel
- $\hat{\beta} = 1.11$ nao e significativamente diferente de 1: sem vies de escala
- Teste conjunto $F = 1.23$ ($p = 0.34$): nao rejeita $H_0$
- **Conclusao**: nao ha evidencia de vies ou ineficiencia na previsao

### Exemplo com previsao viesada

```python
# Previsao sistematicamente baixa
predicted_biased = predicted - 0.5

mz_biased = mincer_zarnowitz(actual, predicted_biased)
print(mz_biased)
```

```text
Mincer-Zarnowitz Regression
============================
y_t = alpha + beta * y_hat_t + eps_t

           Estimate    Std.Err    t-stat    p-value
alpha        0.3214     0.3412     0.942     0.3741
beta         1.1071     0.1561     7.092     0.0001

Joint test (alpha=0, beta=1):
  F-statistic:  5.678
  P-value:      0.0289

R-squared: 0.8624

Conclusion: Rejeita H0 a 5%. A previsao apresenta vies ou
            ineficiencia sistematica.
```

!!! warning "Autocorrelacao nos erros"
    Para previsoes multi-step ($h > 1$), os erros $\varepsilon_t$ sao autocorrelacionados por construcao. Sempre use `cov_type="HAC"` (padrao) para obter inferencia valida.

## Visualizacao

```python
from forecastbox.visualization import plot_mincer_zarnowitz

# Scatter plot com linha de regressao e linha 45 graus
plot_mincer_zarnowitz(actual, predicted)
```

O grafico mostra:

- Pontos $(\\hat{y}_t, y_t)$
- Linha de regressao MZ (azul)
- Linha de 45 graus (cinza tracejada) — referencia para $\alpha=0, \beta=1$
- Desvio entre as duas linhas indica vies

## Ver Tambem

- [Encompassing](encompassing.md) — teste de eficiencia forte (com informacao adicional)
- [Metricas](metrics.md) — metricas complementares a regressao MZ
- [Diebold-Mariano](diebold-mariano.md) — comparar modelos apos verificar qualidade individual
- :material-stethoscope: [Diagnostico de Vies](../../diagnostics/bias.md) — diagnostico pratico de vies incondicional, condicional e tracking signal
