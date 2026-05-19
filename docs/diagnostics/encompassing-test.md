---
title: "Encompassing Test — Diagnostico de Informacao Incremental"
description: "Teste de encompassing como ferramenta de diagnostico: verificar se um modelo contem toda a informacao de outro e decidir se a combinacao agrega valor."
---

# Encompassing Test — Diagnostico de Informacao Incremental

!!! abstract "Key Takeaway"
    O teste de encompassing responde a pergunta: **"modelo A contem toda a informacao relevante de modelo B?"** Se sim, modelo B e redundante na combinacao. Se nenhum modelo encompassa o outro, a combinacao agrega valor. Este diagnostico e fundamental para decidir **quais** modelos incluir antes de combinar.

## Conceito Fundamental

Dois modelos geram previsoes $\hat{y}_{A,t}$ e $\hat{y}_{B,t}$ para o mesmo alvo $y_t$. A pergunta de encompassing e:

> A previsao de A ja incorpora toda a informacao util contida em B?

Formalmente, considere a **regressao de combinacao**:

$$
y_t = (1 - \lambda) \hat{y}_{A,t} + \lambda \hat{y}_{B,t} + \varepsilon_t
$$

- Se $\lambda = 0$: A encompassa B — a previsao de B nao agrega informacao alem do que A ja fornece
- Se $\lambda = 1$: B encompassa A — a previsao de A e redundante dado B
- Se $0 < \lambda < 1$: nenhum encompassa o outro — ambos contribuem com informacao unica

### Formulacao Alternativa (Fair-Shiller)

A regressao de Fair-Shiller testa encompassing diretamente:

$$
y_t = \alpha + \beta_A \hat{y}_{A,t} + \beta_B \hat{y}_{B,t} + \varepsilon_t
$$

- $H_0: \beta_B = 0$ — A encompassa B (B nao agrega informacao)
- $H_0: \beta_A = 0$ — B encompassa A (A nao agrega informacao)

O teste e um **t-test** (ou F-test) sobre os coeficientes estimados por OLS, com erros-padrao robustos a heterocedasticidade e autocorrelacao (HAC).

!!! info "Teste bilateral vs unilateral"
    - **Unilateral** ($H_0: \beta_B = 0$ vs $H_1: \beta_B > 0$): testa se B agrega informacao **positiva** — mais conservador, recomendado na pratica
    - **Bilateral** ($H_0: \beta_B = 0$ vs $H_1: \beta_B \neq 0$): testa se B agrega informacao em qualquer direcao — pode captar contribuicoes espurias com sinal negativo

## Diagnostico Sequencial

O encompassing deve ser aplicado como um **processo sistematico de quatro passos** para decidir se a combinacao agrega valor:

```text
Passo 1: Testar se A encompassa B  (H0: beta_B = 0)
Passo 2: Testar se B encompassa A  (H0: beta_A = 0)
Passo 3: Interpretar os quatro cenarios possiveis
Passo 4: Decidir sobre combinacao vs selecao
```

### Os Quatro Cenarios

| Cenario | A encompassa B? | B encompassa A? | Interpretacao | Acao |
|---------|:-:|:-:|--------------|------|
| **1** | Sim | Nao | A contem toda a informacao de B | Usar apenas A |
| **2** | Nao | Sim | B contem toda a informacao de A | Usar apenas B |
| **3** | Nao | Nao | Ambos tem informacao unica | **Combinar** |
| **4** | Sim | Sim | Modelos redundantes | Qualquer um serve |

!!! tip "Cenario 3 e o mais interessante"
    Quando **nenhum modelo encompassa o outro**, a combinacao tem potencial real de melhorar a previsao. Este e o cenario que justifica o esforco de estimar pesos e manter multiplos modelos.

=== "Codigo"

    ```python
    from forecastbox.diagnostics import encompassing_test

    # Teste bilateral: A encompassa B? B encompassa A?
    result = encompassing_test(
        actual=actual,
        forecast_a=fc_arima,
        forecast_b=fc_ets,
        significance=0.05,
        one_sided=False,
    )

    print(result.summary())
    ```

=== "Output"

    ```text
    ====================================================================
                    Forecast Encompassing Test
    ====================================================================
    Model A: ARIMA | Model B: ETS
    Sample size: 120 | Significance: 5%

    Regression: y_t = alpha + beta_A * f_A,t + beta_B * f_B,t + eps_t

    Coefficient   Estimate   Std Error   t-stat   p-value
    ──────────────────────────────────────────────────────
    alpha         0.012      0.045       0.267    0.7901
    beta_A        0.648      0.112       5.786    0.0000 ***
    beta_B        0.341      0.098       3.480    0.0007 ***

    Tests:
      H0: beta_B = 0 (A encompasses B):  F = 12.11, p = 0.0007  REJECT
      H0: beta_A = 0 (B encompasses A):  F = 33.48, p = 0.0000  REJECT

    Conclusion: Neither model encompasses the other.
                Combination is justified.
    ====================================================================
    ```

### Interpretacao Passo a Passo

!!! info "Leitura do output"
    1. **$\hat{\beta}_A = 0.648$, significativo**: ARIMA contribui com informacao unica
    2. **$\hat{\beta}_B = 0.341$, significativo**: ETS tambem contribui com informacao unica
    3. **Ambos $H_0$ rejeitados**: nenhum encompassa o outro → **Cenario 3**
    4. **Os coeficientes sugerem pesos**: ARIMA merece ~65% e ETS ~35% na combinacao

## Encompassing Multivariado

Quando ha $K > 2$ modelos, o teste se estende para verificar se um modelo encompassa **todos os demais simultaneamente**.

### Regressao Multivariada

$$
y_t = \alpha + \sum_{i=1}^{K} \beta_i \hat{y}_{i,t} + \varepsilon_t
$$

O teste de encompassing do modelo $j$ sobre todos os demais e:

$$
H_0: \beta_i = 0 \quad \forall \, i \neq j
$$

Este e um **teste F de exclusao conjunta** — equivale a testar se remover todos os modelos exceto $j$ nao piora significativamente o ajuste.

$$
F = \frac{(SSR_R - SSR_U) / q}{SSR_U / (T - K - 1)}
$$

onde $SSR_R$ e a soma de residuos quadrados do modelo restrito (apenas modelo $j$), $SSR_U$ e a do modelo irrestrito (todos os modelos), e $q = K - 1$ e o numero de restricoes.

```python
from forecastbox.diagnostics import encompassing_test

# Teste multivariado: cada modelo vs todos os demais
forecasts = {
    "ARIMA": fc_arima,
    "ETS": fc_ets,
    "BVAR": fc_bvar,
    "RF": fc_rf,
}

result_multi = encompassing_test(
    actual=actual,
    forecasts=forecasts,
    significance=0.05,
    correction="hac",  # Newey-West HAC errors
)

print(result_multi.summary())
```

```text
====================================================================
             Multivariate Encompassing Test (HAC)
====================================================================
Models: ARIMA, ETS, BVAR, RF | Sample: 120 | Significance: 5%

Individual coefficients:
  Model     beta     Std Err   t-stat   p-value
  ARIMA     0.412    0.098     4.204    0.0001 ***
  ETS       0.287    0.085     3.376    0.0010 ***
  BVAR      0.198    0.112     1.768    0.0797  .
  RF        0.103    0.134     0.769    0.4436

Joint exclusion tests (F-test):
  H0: All except ARIMA = 0   F = 5.82, p = 0.0010  REJECT
  H0: All except ETS = 0     F = 7.14, p = 0.0002  REJECT
  H0: All except BVAR = 0    F = 9.23, p = 0.0000  REJECT
  H0: All except RF = 0      F = 11.87, p = 0.0000 REJECT

Conclusion: No single model encompasses all others.
             Full combination is justified.
====================================================================
```

!!! warning "Multicolinearidade"
    Se as previsoes dos modelos sao altamente correlacionadas, os coeficientes individuais podem ser insignificantes mesmo quando o modelo contribui — o teste F de exclusao conjunta e mais confiavel que os t-tests individuais neste caso.

## Matriz de Encompassing

Para $K$ modelos, a **matriz de encompassing** mostra todos os testes pareados em formato $K \times K$. A entrada $(i, j)$ indica se o modelo $i$ encompassa o modelo $j$.

=== "Codigo"

    ```python
    from forecastbox.diagnostics import encompassing_test

    forecasts = {
        "ARIMA": fc_arima,
        "ETS": fc_ets,
        "BVAR": fc_bvar,
        "RF": fc_rf,
    }

    # Matriz de encompassing completa
    matrix = encompassing_test(
        actual=actual,
        forecasts=forecasts,
        significance=0.05,
        correction="hac",
        output="matrix",
    )

    print(matrix.summary())
    ```

=== "Output"

    ```text
    ====================================================================
                 Encompassing Matrix (p-values)
    ====================================================================
    H0: Row model encompasses Column model
    Reject H0 (p < 0.05) means Column adds information beyond Row

              ARIMA    ETS      BVAR     RF
    ARIMA       —      0.003**  0.092    0.412
    ETS       0.000**    —      0.041**  0.287
    BVAR      0.000**  0.001**    —      0.156
    RF        0.000**  0.000**  0.008**    —

    Signif: ** p<0.05

    Interpretation:
      ARIMA: NOT encompassed by any model
      ARIMA does NOT encompass: ETS (p=0.003)
      ETS:   NOT encompassed by ARIMA/BVAR
      BVAR:  encompassed by ARIMA (p=0.092 > 0.05) — consider removing
      RF:    encompassed by ARIMA (p=0.412 > 0.05) — consider removing
    ====================================================================
    ```

=== "Visualizacao"

    ```python
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    matrix.plot_heatmap(
        ax=ax,
        cmap="RdYlGn_r",  # vermelho = rejeita (modelo agrega)
        annot=True,
        fmt=".3f",
        significance=0.05,
    )
    ax.set_title("Matriz de Encompassing (p-valores)")
    ax.set_xlabel("Modelo testado (coluna agrega info?)")
    ax.set_ylabel("Modelo base (linha encompassa coluna?)")
    plt.tight_layout()
    plt.show()
    ```

### Leitura da Matriz

!!! info "Como interpretar"
    - **Leia por linha**: "modelo da linha encompassa modelo da coluna?"
    - **p-valor baixo** (vermelho): coluna **agrega** informacao alem da linha → NAO encompassa
    - **p-valor alto** (verde): coluna **nao agrega** informacao → encompassa (ou redundante)
    - **Linha toda com p-valores altos**: o modelo da linha encompassa todos — pode ser suficiente sozinho
    - **Coluna toda com p-valores baixos**: o modelo da coluna agrega informacao sobre todos — essencial na combinacao

## Relacao com Teste F de Exclusao

O teste de encompassing esta intimamente ligado ao **teste F de exclusao** na regressao de combinacao. Na regressao irrestrita:

$$
y_t = \alpha + \beta_1 \hat{y}_{1,t} + \beta_2 \hat{y}_{2,t} + \cdots + \beta_K \hat{y}_{K,t} + \varepsilon_t
$$

Testar se o modelo $j$ pode ser excluido equivale a:

$$
H_0: \beta_j = 0 \quad \Longleftrightarrow \quad \text{todos os demais encompassam } j
$$

A estatistica F compara o modelo restrito (sem $j$) ao irrestrito:

$$
F = \frac{(SSR_{-j} - SSR_{\text{full}}) / 1}{SSR_{\text{full}} / (T - K - 1)} \sim F_{1, T-K-1}
$$

```python
# Teste F de exclusao sequencial
exclusion = encompassing_test(
    actual=actual,
    forecasts=forecasts,
    significance=0.05,
    output="exclusion",
)

print(exclusion.summary())
```

```text
Sequential Exclusion Test (F-test)
====================================================================
Step  Excluded  F-stat   p-value   Decision
────────────────────────────────────────────
1     RF        0.591    0.4436    EXCLUDE (p > 0.05)
2     BVAR      3.127    0.0797    EXCLUDE (p > 0.05)
3     ETS       11.41    0.0010    KEEP    (p < 0.05)

Final model set: {ARIMA, ETS}
Excluded (redundant): {RF, BVAR}
====================================================================
```

!!! tip "Encompassing como pre-selecao"
    Use o teste de exclusao sequencial para **reduzir o numero de modelos** antes de estimar pesos de combinacao. Menos modelos = pesos mais estaveis = combinacao mais robusta. Veja [Estabilidade de Pesos](weight-stability.md).

## Correcao para Multiplos Testes

Com $K$ modelos, a matriz de encompassing contem $K(K-1)$ testes. Sem correcao, a probabilidade de rejeitar falsamente pelo menos um teste cresce rapidamente.

$$
P(\text{pelo menos 1 falso positivo}) = 1 - (1 - \alpha)^{K(K-1)}
$$

Para $K = 5$ e $\alpha = 0.05$: $P = 1 - 0.95^{20} = 0.642$ — mais de 64% de chance de pelo menos um erro tipo I.

### Correcoes Disponiveis

| Correcao | Formula | Conservadorismo |
|----------|---------|----------------|
| **Bonferroni** | $\alpha^* = \alpha / m$ | Muito conservador |
| **Holm** | Sequencial ajustado | Moderado |
| **BH (FDR)** | Controla false discovery rate | Menos conservador |

```python
# Com correcao de Bonferroni
matrix_corrected = encompassing_test(
    actual=actual,
    forecasts=forecasts,
    significance=0.05,
    correction="bonferroni",
    output="matrix",
)

print(matrix_corrected.summary())
```

```text
Encompassing Matrix (Bonferroni-corrected, m=12)
Adjusted significance level: 0.0042

          ARIMA    ETS      BVAR     RF
ARIMA       —      0.003**  0.092    0.412
ETS       0.000**    —      0.041    0.287
BVAR      0.000**  0.001**    —      0.156
RF        0.000**  0.000**  0.008    —

Note: ETS->BVAR (p=0.041) no longer significant after correction
```

!!! warning "Bonferroni pode ser excessivo"
    Com muitos modelos, Bonferroni pode nao rejeitar nenhum teste, sugerindo que todos os modelos se encompassam mutuamente — o que raramente e verdade. Use **Holm** ou **BH** como alternativas menos conservadoras.

## Exemplo Completo: ARIMA vs ETS

```python
import numpy as np
from forecastbox.diagnostics import encompassing_test

# Simular dados
np.random.seed(42)
T = 120

actual = np.cumsum(np.random.randn(T) * 0.5) + 100
fc_arima = actual + np.random.randn(T) * 0.8 + 0.02
fc_ets = actual + np.random.randn(T) * 1.0 - 0.01

# ===== PASSO 1: Teste bilateral =====
result = encompassing_test(
    actual=actual,
    forecast_a=fc_arima,
    forecast_b=fc_ets,
    significance=0.05,
    one_sided=False,
)

print("=== Teste Bilateral ===")
print(result.summary())

# ===== PASSO 2: Teste unilateral (mais conservador) =====
result_one = encompassing_test(
    actual=actual,
    forecast_a=fc_arima,
    forecast_b=fc_ets,
    significance=0.05,
    one_sided=True,
)

print("\n=== Teste Unilateral ===")
print(result_one.summary())

# ===== PASSO 3: Interpretacao =====
if result.a_encompasses_b and not result.b_encompasses_a:
    print("\n-> ARIMA encompassa ETS: usar apenas ARIMA")
elif not result.a_encompasses_b and result.b_encompasses_a:
    print("\n-> ETS encompassa ARIMA: usar apenas ETS")
elif not result.a_encompasses_b and not result.b_encompasses_a:
    print("\n-> Nenhum encompassa o outro: COMBINAR")
else:
    print("\n-> Ambos se encompassam: modelos REDUNDANTES")
```

```text
=== Teste Bilateral ===
====================================================================
                Forecast Encompassing Test
====================================================================
Model A: forecast_a | Model B: forecast_b
Sample size: 120 | Significance: 5%

Regression: y_t = alpha + beta_A * f_A,t + beta_B * f_B,t + eps_t

Coefficient   Estimate   Std Error   t-stat   p-value
──────────────────────────────────────────────────────
alpha         0.008      0.041       0.195    0.8458
beta_A        0.672      0.105       6.400    0.0000 ***
beta_B        0.318      0.089       3.573    0.0005 ***

Tests:
  H0: beta_B = 0 (A encompasses B):  F = 12.77, p = 0.0005  REJECT
  H0: beta_A = 0 (B encompasses A):  F = 40.96, p = 0.0000  REJECT

Conclusion: Neither model encompasses the other.

=== Teste Unilateral ===
  H0: beta_B <= 0:  t = 3.573, p = 0.0003  REJECT
  H0: beta_A <= 0:  t = 6.400, p = 0.0000  REJECT

-> Nenhum encompassa o outro: COMBINAR
```

## Armadilhas Comuns

!!! danger "Erros frequentes"
    1. **Ignorar autocorrelacao**: se $\varepsilon_t$ e autocorrelacionado (comum em previsoes multi-step), use erros HAC (`correction="hac"`)
    2. **Amostra pequena**: com $T < 50$, o teste tem baixo poder — nao rejeitar encompassing nao significa que o modelo e de fato redundante
    3. **Previsoes colineares**: se $\hat{y}_A \approx \hat{y}_B$, os coeficientes sao instavel — o teste pode oscilar entre cenarios. Verifique a correlacao entre previsoes antes
    4. **Multiplos testes sem correcao**: com $K$ modelos, aplique correcao (Bonferroni, Holm ou BH)

!!! tip "Combine com estabilidade"
    O encompassing test e estatico — avalia a amostra inteira. Mas a relevancia de um modelo pode **mudar ao longo do tempo**. Combine com [Estabilidade de Pesos](weight-stability.md) para uma visao dinamica: um modelo pode encompassar outro na amostra completa mas nao em sub-periodos.

## Parametros de Referencia

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `actual` | `array-like` | — | Valores realizados |
| `forecast_a` | `array-like` | — | Previsao do modelo A (teste par) |
| `forecast_b` | `array-like` | — | Previsao do modelo B (teste par) |
| `forecasts` | `dict[str, array-like]` | — | Dicionario de previsoes (teste multivariado) |
| `significance` | `float` | `0.05` | Nivel de significancia |
| `one_sided` | `bool` | `False` | Teste unilateral ($\beta > 0$) vs bilateral |
| `correction` | `str` | `"hac"` | Correcao de erros: `"hac"`, `"white"`, `"none"` |
| `multiple_testing` | `str` | `None` | Correcao multiplos testes: `"bonferroni"`, `"holm"`, `"bh"` |
| `output` | `str` | `"summary"` | Formato: `"summary"`, `"matrix"`, `"exclusion"` |

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [Combinacao de Previsoes](../theory/combination-theory.md) — fundamentos teoricos do encompassing e combinacao
    - :material-notebook-edit: **User Guide**: [Teste de Encompassing](../user-guide/evaluation/encompassing.md) — formulacao Fair-Shiller e implementacao
    - :material-link-variant: **Relacionado**: [Estabilidade de Pesos](weight-stability.md) — diagnostico dinamico dos pesos de combinacao
    - :material-link-variant: **Relacionado**: [DM Test](dm-test.md) — testar superioridade preditiva
    - :material-link-variant: **Relacionado**: [MCS Diagnostic](mcs-diagnostic.md) — selecionar o conjunto de modelos superiores
