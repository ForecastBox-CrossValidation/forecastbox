---
title: "Teste de Eficiencia"
description: "Diagnostico de eficiencia de previsoes: testes de autocorrelacao (Ljung-Box, Breusch-Godfrey), regressao auxiliar e Mincer-Zarnowitz."
---

# Teste de Eficiencia

!!! abstract "Key Takeaway"
    Uma previsao eficiente nao desperdiça informacao: seus erros sao **imprevissiveis**. Se erros passados ou variaveis observaveis ajudam a prever o proximo erro, a previsao pode ser melhorada sem trocar de modelo.

## Conceito

Eficiencia significa que o forecaster utiliza toda a informacao disponivel de forma otima. Em termos formais, os erros de previsao devem ser **ortogonais** ao conjunto de informacao $\mathcal{F}_t$:

$$
E[e_{t+h} | \mathcal{F}_t] = 0
$$

Na pratica, testamos isso em dois niveis:

| Nivel | Definicao | Teste |
|-------|-----------|-------|
| **Eficiencia fraca** | Erros nao autocorrelacionados | Ljung-Box, Breusch-Godfrey |
| **Eficiencia semi-forte** | Erros ortogonais a variaveis observaveis | Regressao auxiliar |

## Eficiencia Fraca

### Teste de Ljung-Box

Testa se as autocorrelacoes dos erros sao **conjuntamente zero**:

$$
H_0: \rho_1 = \rho_2 = \cdots = \rho_m = 0
$$

A estatistica e:

$$
Q(m) = T(T+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{T-k} \sim \chi^2(m)
$$

onde $\hat{\rho}_k$ e a autocorrelacao amostral de lag $k$ e $T$ o tamanho da amostra.

!!! info "Escolha de $m$"
    Regra pratica: $m \approx \min(T/4, 2h)$ onde $h$ e o horizonte de previsao. Para previsoes mensais de 12 meses, $m = 12$ e razoavel.

### Teste de Breusch-Godfrey

Alternativa ao Ljung-Box que permite erros heteroscedasticos e regressores estocasticos. Estima a regressao auxiliar:

$$
e_t = \delta_0 + \delta_1 e_{t-1} + \delta_2 e_{t-2} + \cdots + \delta_p e_{t-p} + v_t
$$

$$
H_0: \delta_1 = \delta_2 = \cdots = \delta_p = 0
$$

A estatistica $LM = T \cdot R^2$ segue $\chi^2(p)$ sob $H_0$.

### Exemplo: eficiencia fraca de previsao do Focus

```python
import pandas as pd
from forecastbox.diagnostics import efficiency_test

# Erros de previsao do Focus (mediana) para IPCA
errors_focus = pd.Series(
    [0.08, 0.04, 0.06, -0.06, 0.05, 0.07,
     -0.07, -0.06, 0.06, 0.04, -0.04, 0.04],
    name="Erro Focus IPCA"
)

# Ljung-Box
lb = efficiency_test(errors_focus, test_type="ljung_box", max_lags=6)
print(f"Ljung-Box Q({lb.lags}): {lb.statistic:.3f}")
print(f"p-valor:              {lb.pvalue:.4f}")
print(f"Eficiente:            {not lb.reject}")
```

```text
Ljung-Box Q(6): 4.821
p-valor:              0.5672
Eficiente:            True
```

```python
# Breusch-Godfrey
bg = efficiency_test(errors_focus, test_type="breusch_godfrey", max_lags=4)
print(f"BG LM({bg.lags}):    {bg.statistic:.3f}")
print(f"p-valor:              {bg.pvalue:.4f}")
```

```text
BG LM(4):    3.152
p-valor:              0.5329
```

!!! tip "Resultado"
    Ambos os testes nao rejeitam $H_0$: os erros do Focus para o IPCA nao sao autocorrelacionados — a previsao e fracamente eficiente.

## Eficiencia Semi-Forte

Testa se variaveis **observaveis no momento da previsao** ajudam a prever o erro:

$$
e_t = \gamma_0 + \gamma_1 e_{t-1} + \gamma_2 x_t + v_t
$$

$$
H_0: \gamma_0 = \gamma_1 = \gamma_2 = 0
$$

onde $x_t$ pode incluir:

- Erro defasado $e_{t-1}$ (eficiencia fraca embutida)
- Variaveis macroeconomicas (cambio, Selic, commodities)
- A propria previsao $\hat{y}_t$ (teste de calibracao)

### Exemplo: eficiencia semi-forte com variaveis macro

```python
import pandas as pd
from forecastbox.diagnostics import efficiency_test

errors_focus = pd.Series(
    [0.08, 0.04, 0.06, -0.06, 0.05, 0.07,
     -0.07, -0.06, 0.06, 0.04, -0.04, 0.04]
)

# Variaveis disponiveis ao forecaster
variables = pd.DataFrame({
    "cambio_var": [0.5, -0.3, 1.2, -0.8, 0.2, 0.7,
                   -1.1, 0.4, -0.2, 0.6, -0.5, 0.3],
    "selic_nivel": [13.75, 13.75, 13.75, 13.75, 13.25, 13.25,
                    12.75, 12.75, 12.25, 12.25, 11.75, 11.75]
})

# Teste de eficiencia semi-forte
sf = efficiency_test(
    errors_focus,
    test_type="regression",
    variables=variables,
    max_lags=1
)

print(f"F-statistic:  {sf.statistic:.3f}")
print(f"p-valor:      {sf.pvalue:.4f}")
print(f"R² auxiliar:  {sf.r_squared:.4f}")
print(f"\nCoeficientes:")
for name, coef in sf.coefficients.items():
    print(f"  {name:12s}: {coef:+.4f}")
```

```text
F-statistic:  0.872
p-valor:      0.4981
R² auxiliar:  0.0823

Coeficientes:
  const       : +0.0134
  error_lag1  : -0.1823
  cambio_var  : +0.0091
  selic_nivel : -0.0012
```

!!! info "Interpretacao"
    $R^2$ baixo e p-valor alto indicam que as variaveis condicionantes nao explicam os erros — a previsao e semi-fortemente eficiente com respeito a essas variaveis.

## Mincer-Zarnowitz como Teste de Eficiencia

A regressao de Mincer-Zarnowitz tambem pode ser interpretada como um teste de eficiencia:

$$
y_t = \alpha + \beta \hat{y}_t + u_t
$$

Se a previsao e eficiente, entao $\alpha = 0$ e $\beta = 1$, o que implica que a previsao e a **melhor projecao linear** do realizado.

=== "Teste individual"

    Testar $\beta = 1$ isoladamente: a previsao captura a **escala** correta?

    ```python
    from forecastbox.evaluation import mincer_zarnowitz

    mz = mincer_zarnowitz(actual=actual, predicted=predicted)
    print(f"alpha: {mz.alpha:.4f} (p={mz.alpha_pvalue:.4f})")
    print(f"beta:  {mz.beta:.4f} (p={mz.beta_pvalue:.4f})")
    ```

=== "Teste conjunto"

    Testar $\alpha = 0, \beta = 1$ conjuntamente: eficiencia completa.

    ```python
    mz = mincer_zarnowitz(actual=actual, predicted=predicted)
    print(f"F-stat conjunto: {mz.joint_statistic:.3f}")
    print(f"p-valor:         {mz.joint_pvalue:.4f}")
    ```

!!! warning "Relacao com racionalidade"
    O teste conjunto de Mincer-Zarnowitz ($\alpha = 0, \beta = 1$) e **equivalente** ao teste de racionalidade sob perda quadratica. Veja [Racionalidade](rationality.md) para a extensao a funcoes de perda assimetricas.

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `errors` | `array-like` | — | Serie de erros de previsao |
| `test_type` | `str` | `"ljung_box"` | `"ljung_box"`, `"breusch_godfrey"` ou `"regression"` |
| `max_lags` | `int` | `None` | Numero de lags (auto se `None`: $\min(T/4, 2h)$) |
| `variables` | `DataFrame` | `None` | Variaveis condicionantes para teste semi-forte |
| `alpha` | `float` | `0.05` | Nivel de significancia |

## Resumo dos Testes

| Teste | $H_0$ | Quando usar |
|-------|--------|-------------|
| Ljung-Box | $\rho_1 = \cdots = \rho_m = 0$ | Primeiro teste de eficiencia fraca |
| Breusch-Godfrey | $\delta_1 = \cdots = \delta_p = 0$ | Erros heteroscedasticos ou regressores estocasticos |
| Regressao auxiliar | $\gamma_0 = \gamma_1 = \gamma_2 = 0$ | Testar se variaveis observaveis ajudam a prever erros |
| Mincer-Zarnowitz | $\alpha = 0, \beta = 1$ | Teste de eficiencia/calibracao da previsao |

## Proximos Passos

Se a previsao e nao-viesada e eficiente, o proximo passo e verificar se ela e **racional** — ou seja, se utiliza a informacao de forma otima dado o objetivo. Veja o [teste de racionalidade](rationality.md).

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [Avaliacao de Previsoes](../theory/evaluation-theory.md) — fundamentos teoricos dos testes de eficiencia
    - :material-notebook-edit: **User Guide**: [Metricas de Avaliacao](../user-guide/evaluation/metrics.md) — metricas complementares de performance
    - :material-arrow-right: **Proximo**: [Teste de Racionalidade](rationality.md) — teste conjunto de nao-vies e eficiencia
