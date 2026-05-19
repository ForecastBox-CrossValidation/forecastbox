---
title: Dynamic Factor Model (DFM)
description: Modelo de fator dinamico via Filtro de Kalman para nowcasting com paineis grandes e missing data.
---

# Dynamic Factor Model (DFM)

O Dynamic Factor Model extrai **fatores latentes comuns** de um painel grande de
indicadores macroeconomicos. A ideia central e que a co-movimentacao de dezenas de
series pode ser resumida por poucos fatores nao-observaveis, que capturam o estado
da economia em tempo real.

!!! warning "Dependencia: kalmanbox"

    O DFM do forecastbox depende do pacote **kalmanbox** para o Filtro de Kalman
    e o EM algorithm. Instale com:

    ```bash
    pip install kalmanbox
    ```

    Sem o kalmanbox, apenas o metodo `method="pca"` (PCA estatico) esta disponivel.

---

## Modelo

O DFM e composto por duas equacoes em representacao estado-espaco:

### Equacao de Observacao

Relaciona os indicadores observados $\mathbf{x}_t$ aos fatores latentes
$\mathbf{f}_t$:

$$
\mathbf{x}_t = \Lambda \mathbf{f}_t + \mathbf{e}_t, \quad \mathbf{e}_t \sim N(\mathbf{0}, R)
$$

onde:

| Simbolo | Dimensao | Descricao |
|:--------|:---------|:----------|
| $\mathbf{x}_t$ | $n \times 1$ | Vetor de indicadores observados |
| $\Lambda$ | $n \times r$ | Matriz de factor loadings |
| $\mathbf{f}_t$ | $r \times 1$ | Vetor de fatores latentes |
| $\mathbf{e}_t$ | $n \times 1$ | Erros idiossincraticos (diagonal $R$) |
| $n$ | — | Numero de indicadores |
| $r$ | — | Numero de fatores |

### Equacao de Estado

Os fatores seguem um processo VAR(p):

$$
\mathbf{f}_t = A_1 \mathbf{f}_{t-1} + A_2 \mathbf{f}_{t-2} + \cdots + A_p \mathbf{f}_{t-p} + \mathbf{u}_t, \quad \mathbf{u}_t \sim N(\mathbf{0}, Q)
$$

onde $A_1, \ldots, A_p$ sao as matrizes de transicao e $Q$ e a covariancia dos
choques nos fatores.

---

## Estimacao

O forecastbox oferece tres metodos de estimacao:

=== "EM Algorithm"

    O metodo padrao e o **EM (Expectation-Maximization)**, que itera entre:

    - **E-step**: Filtro de Kalman + smoother para estimar $E[\mathbf{f}_t | \mathbf{x}_{1:T}]$
    - **M-step**: Atualizar $\Lambda$, $A$, $R$, $Q$ por MLE condicional

    ```python
    from forecastbox.nowcast import DFM

    dfm = DFM(
        n_factors=2,
        factor_lags=2,
        method="em",
        em_iterations=100,
        em_tol=1e-4,
    )
    dfm.fit(data)
    ```

    O EM garante convergencia para um maximo local da verossimilhanca e
    trata missing data naturalmente via Filtro de Kalman.

=== "Two-Step (PCA + Kalman)"

    Procedimento mais rapido em dois passos:

    1. Extrair fatores por **PCA** (completando missing com EM-PCA)
    2. Estimar dinamica dos fatores por **VAR** e refinar via Kalman

    ```python
    dfm = DFM(
        n_factors=2,
        factor_lags=2,
        method="two_step",
    )
    dfm.fit(data)
    ```

    Mais rapido que EM puro, mas pode ser menos eficiente com muitos missing values.

=== "PCA Estatico"

    Apenas PCA, sem Filtro de Kalman. Nao requer kalmanbox.

    ```python
    dfm = DFM(
        n_factors=2,
        method="pca",
    )
    dfm.fit(data)
    ```

    !!! note

        O metodo PCA estatico nao trata missing data no ragged edge.
        Indicadores incompletos sao excluidos ou imputados por media.
        Use este metodo apenas como baseline rapido.

---

## Tratamento de Missing Data

O principal diferencial do DFM para nowcasting e o tratamento natural de missing data
via Filtro de Kalman. No ragged edge, cada observacao missing simplesmente
**nao contribui para a atualizacao** do filtro naquele periodo.

Formalmente, se $x_{i,t}$ e missing, a linha $i$ da equacao de observacao e
removida no periodo $t$:

$$
\mathbf{x}_{t}^{obs} = \Lambda^{obs}_t \mathbf{f}_t + \mathbf{e}_{t}^{obs}
$$

onde $\Lambda^{obs}_t$ contem apenas as linhas correspondentes aos indicadores
observados em $t$.

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `n_factors` | `int` | `2` | Numero de fatores latentes |
| `factor_lags` | `int` | `2` | Ordem do VAR dos fatores |
| `method` | `str` | `"em"` | Metodo de estimacao: `"em"`, `"two_step"`, `"pca"` |
| `em_iterations` | `int` | `100` | Maximo de iteracoes do EM |
| `em_tol` | `float` | `1e-4` | Tolerancia de convergencia do EM |
| `standardize` | `bool` | `True` | Padronizar indicadores antes da estimacao |
| `target` | `str` | `None` | Variavel-alvo para nowcast (e.g., `"pib"`) |

---

## Selecao do Numero de Fatores

Escolher $r$ (numero de fatores) e crucial. O forecastbox implementa os
**criterios de Bai & Ng (2002)**, que generalizam o AIC/BIC para modelos fatoriais:

$$
IC(r) = \ln \hat{V}(r) + r \cdot g(n, T)
$$

onde $\hat{V}(r)$ e a variancia residual com $r$ fatores e $g(n, T)$ e uma
funcao de penalizacao que depende das dimensoes do painel.

=== "IC_p1"

    $$g(n,T) = \frac{n + T}{nT} \ln\left(\frac{nT}{n+T}\right)$$

=== "IC_p2"

    $$g(n,T) = \frac{n + T}{nT} \ln\left(\min(n, T)\right)$$

=== "IC_p3"

    $$g(n,T) = \frac{\ln(\min(n,T))}{\min(n,T)}$$

```python
from forecastbox.nowcast import select_n_factors

# Selecao automatica do numero de fatores
result = select_n_factors(data, max_factors=10, criterion="ic_p2")
print(result)
```

```text
Factor Selection (Bai-Ng IC_p2)

  r    IC_p2      Var. Explained
  1   -2.341      0.432
  2   -2.587      0.621  <-- selected
  3   -2.524      0.703
  4   -2.401      0.758
```

!!! tip "Regra Pratica"

    Para paineis macroeconomicos brasileiros com 15-30 indicadores mensais,
    **2 a 4 fatores** costumam ser suficientes. O primeiro fator tipicamente
    captura o ciclo economico geral; o segundo, um fator de precos ou financeiro.

---

## Exemplo: Nowcast do PIB Brasileiro

Nowcast do PIB trimestral usando ~20 indicadores mensais:

```python
import pandas as pd
from forecastbox.nowcast import DFM, VintageManager, select_n_factors
from forecastbox.datasets import load_brazil_indicators

# Carregar painel de indicadores macro brasileiros
data = load_brazil_indicators()
# Inclui: prod. industrial, vendas varejo, PMI, energia,
#         emprego formal, credito, exportacoes, ICC, etc.

# Selecionar numero de fatores
n_factors = select_n_factors(data, max_factors=8, criterion="ic_p2")
print(f"Fatores selecionados: {n_factors.selected}")

# Estimar DFM
dfm = DFM(
    n_factors=n_factors.selected,
    factor_lags=2,
    method="em",
    em_iterations=200,
    target="pib",
)
dfm.fit(data)

# Nowcast para o trimestre corrente
nowcast = dfm.predict(horizon=1)
print(nowcast)
```

```text
Fatores selecionados: 2

Nowcast (target=pib, method=DFM-EM)

  Quarter    Nowcast    Lo95    Hi95
  2024-Q2      0.78    0.32    1.24

  Factor loadings (top 5):
    prod_industrial     0.891
    vendas_varejo       0.823
    pmi_industria       0.798
    energia_eletrica    0.756
    emprego_formal      0.712
```

---

## Diagnosticos

Apos a estimacao, verifique a qualidade do modelo:

```python
# Factor loadings — quais indicadores mais contribuem
dfm.plot_loadings()

# Variancia explicada por fator
dfm.explained_variance()

# Fatores estimados ao longo do tempo
dfm.plot_factors()

# Log-verossimilhanca ao longo das iteracoes EM
dfm.plot_convergence()
```

---

## Quando Usar DFM

| Cenario | DFM e adequado? |
|:--------|:----------------|
| Painel grande (20+ indicadores) | Sim — DFM excele com muitas series |
| Poucos indicadores (3-5) | Nao — prefira bridge equations |
| Dados de frequencia mista (mensal + diario) | Parcial — use MIDAS para dados diarios |
| Missing data extensivo | Sim — Kalman trata naturalmente |
| Interpretabilidade e prioridade | Nao — bridge e mais transparente |

---

## Referencias

- **Stock, J.H. & Watson, M.W.** (2002). "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics*, 20(2), 147-162.
- **Bai, J. & Ng, S.** (2002). "Determining the Number of Factors in Approximate Factor Models." *Econometrica*, 70(1), 191-221.
- **Doz, C., Giannone, D. & Reichlin, L.** (2011). "A Two-Step Estimator for Large Approximate Dynamic Factor Models Based on Kalman Filtering." *Journal of Econometrics*, 164(1), 188-205.
- **Banbura, M. & Modugno, M.** (2014). "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics*, 29(1), 133-160.
