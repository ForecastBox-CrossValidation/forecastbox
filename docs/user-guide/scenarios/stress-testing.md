---
title: "Stress Testing"
description: "Stress testing para previsao econometrica: cenarios historicos, hipoteticos e reversos para avaliacao de resiliencia e gestao de risco."
---

# Stress Testing

!!! abstract "Key Takeaway"

    Stress testing avalia a **resiliencia de previsoes sob cenarios extremos**.
    O forecastbox implementa tres tipos de stress: **historico** (replicar crises
    passadas), **hipotetico** (cenarios ad-hoc) e **reverso** (encontrar o choque
    que causa um resultado especifico). Essencial para regulacao bancaria,
    gestao de risco e planejamento de contingencia.

---

## Conceito

Previsoes condicionais respondem "o que acontece se X?". Stress testing leva
essa pergunta ao extremo: **"o que acontece no pior caso?"**. A ideia e submeter
o modelo a choques severos — mas plausveis — para avaliar:

- **Magnitude do impacto**: quanto a variavel de interesse se desvia da baseline
- **Propagacao**: como o choque se transmite pelo sistema de variaveis
- **Resiliencia**: se o sistema converge de volta ao equilibrio ou diverge

!!! info "Stress Testing vs Cenarios Condicionais"

    Cenarios condicionais tipicamente usam hipoteses **moderadas** e **provaveis**.
    Stress testing deliberadamente usa cenarios **extremos** e **improvaveis**,
    mas **nao impossiveis**. A utilidade esta em revelar vulnerabilidades que
    cenarios "normais" nao expoe.

---

## Tipos de Stress

O forecastbox implementa tres abordagens complementares de stress testing:

### Stress Historico

Replica as condicoes de um episodio passado e aplica ao modelo atual.
Responde: **"o que aconteceria se vivessemos novamente a crise de 2008?"**

$$
\boxed{\Delta\mathbf{x}_{\text{stress}} = \mathbf{x}_{t_1:t_2}^{\text{historico}} - \bar{\mathbf{x}}_{t_1:t_2}^{\text{tendencia}}}
$$

onde $\mathbf{x}_{t_1:t_2}^{\text{historico}}$ sao os valores observados durante o episodio
e $\bar{\mathbf{x}}_{t_1:t_2}^{\text{tendencia}}$ e a tendencia contra-factual (o que teria
ocorrido sem a crise).

```python
from forecastbox.scenarios import StressTest

stress = StressTest(model)

# Replicar a crise financeira de 2008
stress.add_historical(
    "crise_2008",
    start="2008-09",
    end="2009-03",
    variables=["pib", "ipca", "selic", "cambio"],
)

# Replicar o choque COVID
stress.add_historical(
    "covid_2020",
    start="2020-03",
    end="2020-06",
    variables=["pib", "ipca", "selic", "cambio"],
)
```

!!! tip "Calibracao Historica"

    O forecastbox automaticamente extrai os **desvios da tendencia** durante o
    episodio e aplica como choques ao modelo. Voce nao precisa calcular os
    choques manualmente — apenas indique o periodo e as variaveis.

### Stress Hipotetico

Define cenarios ad-hoc baseados em julgamento do analista. Responde:
**"o que aconteceria se a Selic subisse para 20%?"**

$$
\mathbf{x}_{\text{stress}} = \bar{\mathbf{x}}_{\text{baseline}} + \boldsymbol{\delta}_{\text{hipotetico}}
$$

onde $\boldsymbol{\delta}_{\text{hipotetico}}$ e o vetor de choques definido pelo analista.

```python
# Cenario hipotetico: juros altos + cambio depreciado
stress.add_hypothetical(
    "juros_altos",
    shocks={"selic": 20.0, "cambio": 6.5},
)

# Cenario hipotetico: choque de commodities
stress.add_hypothetical(
    "commodities",
    shocks={"petroleo": 150.0, "minero": 200.0, "soja": 80.0},
)
```

!!! note "Valores Absolutos vs Choques"

    Os valores em `shocks` representam o **nivel absoluto** da variavel no cenario
    de stress. O forecastbox calcula automaticamente o desvio em relacao a
    baseline para cada horizonte.

### Stress Reverso

Parte do resultado indesejado e busca o choque que o causaria. Responde:
**"qual choque causaria uma recessao de -3%?"**

$$
\boxed{\boldsymbol{\delta}^* = \arg\min_{\boldsymbol{\delta}} \| g(\boldsymbol{\delta}) - y^{\text{target}} \|^2 \quad \text{s.t.} \quad \boldsymbol{\delta} \in \mathcal{D}}
$$

onde $g(\boldsymbol{\delta})$ e a funcao de previsao condicional ao choque $\boldsymbol{\delta}$,
$y^{\text{target}}$ e o resultado alvo, e $\mathcal{D}$ e o conjunto de choques
plausveis.

```python
# Stress reverso: qual choque causa PIB = -3%?
stress.add_reverse(
    "recessao",
    target_variable="pib",
    target_value=-3.0,
    shock_variables=["selic", "cambio", "petroleo"],
    horizon=4,
)

# Stress reverso: qual cambio causa inflacao > 8%?
stress.add_reverse(
    "inflacao_alta",
    target_variable="ipca",
    target_value=8.0,
    shock_variables=["cambio", "petroleo"],
    horizon=6,
)
```

!!! warning "Limitacoes do Stress Reverso"

    O stress reverso assume **linearidade local** do modelo. Para modelos VAR, a
    solucao e analitica. Para modelos nao-lineares, o forecastbox usa otimizacao
    numerica — o resultado depende do ponto de partida e pode encontrar apenas
    um minimo local. Verifique a plausibilidade dos choques encontrados.

---

## Framework Completo

### API do StressTest

```python
from forecastbox.auto import AutoVAR
from forecastbox.scenarios import StressTest

# Estimar modelo
data = pd.read_csv("macro_br.csv", index_col="date", parse_dates=True)
var = AutoVAR(max_lags=4, ic="aic").fit(data[["pib", "ipca", "selic", "cambio"]])

# Configurar stress test
stress = StressTest(var)

# Adicionar cenarios
stress.add_historical("crise_2008", start="2008-09", end="2009-03")
stress.add_hypothetical("juros_altos", shocks={"selic": 20.0, "cambio": 6.5})
stress.add_reverse("recessao", target_variable="pib", target_value=-3.0)

# Executar todos os cenarios
results = stress.run(horizon=12, baseline=True)
```

### Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `model` | estimado | — | Modelo estimado (VAR, ARIMA, etc.) |
| `horizon` | `int` | `12` | Horizonte de previsao |
| `baseline` | `bool` | `True` | Incluir cenario baseline (sem stress) |
| `confidence_level` | `float` | `0.95` | Nivel de confianca para intervalos |
| `n_simulations` | `int` | `5000` | Simulacoes Monte Carlo por cenario |
| `seed` | `int` | `None` | Semente para reproducibilidade |

---

## Metricas de Impacto

O forecastbox calcula metricas padrao de risco para cada cenario de stress:

### Desvio da Baseline

A metrica mais direta — quanto a previsao se desvia do cenario base:

$$
\Delta_{t+h} = \hat{y}_{t+h}^{\text{stress}} - \hat{y}_{t+h}^{\text{baseline}}
$$

### Value-at-Risk (VaR)

O quantil da distribuicao preditiva sob stress, ao nivel de confianca $\alpha$:

$$
\boxed{\text{VaR}_\alpha = -Q_\alpha\left(\{y_{t+h}^{(s)}\}_{s=1}^S\right)}
$$

onde $Q_\alpha$ e o quantil de ordem $\alpha$ da distribuicao de perdas simulada.
O VaR responde: **"qual a perda maxima com probabilidade $1-\alpha$?"**

### Expected Shortfall (CVaR)

A media das perdas que excedem o VaR — captura o **formato da cauda**:

$$
\boxed{\text{ES}_\alpha = -\frac{1}{|\{s : y^{(s)} \leq Q_\alpha\}|} \sum_{s : y^{(s)} \leq Q_\alpha} y_{t+h}^{(s)}}
$$

!!! info "VaR vs Expected Shortfall"

    O VaR indica o **limiar** da cauda, mas nao diz nada sobre o que acontece
    **alem** dele. O Expected Shortfall complementa o VaR ao informar a perda
    media condicional a estar na cauda. Para regulacao bancaria (Basileia III),
    o ES e preferido por ser uma **medida coerente de risco**.

### Resumo das Metricas

```python
# Acessar metricas
metrics = results.impact_metrics()
print(metrics)
```

```text
Stress Test Impact Metrics (horizon=12)

                        pib                       ipca
                  deviation   VaR95    ES95   deviation   VaR95    ES95
baseline              0.00    0.82    1.15        0.00    1.23    1.67
crise_2008           -2.85    3.91    4.52        1.42    2.78    3.21
juros_altos          -1.20    2.15    2.68       -0.85    1.05    1.42
recessao             -3.00    4.12    4.78        0.65    1.95    2.35
```

---

## Tabela de Resultados

O metodo `summary()` gera uma tabela comparativa de todos os cenarios:

```python
print(results.summary())
```

```text
Stress Test Summary (model=AutoVAR, horizon=12)

Scenario          Type          pib_h4   pib_h8  pib_h12  ipca_h4  ipca_h8  ipca_h12
─────────────────────────────────────────────────────────────────────────────────────
baseline          —              0.72     0.68     0.65     4.21     4.05     3.92
crise_2008        historico     -1.85    -2.42    -2.85     5.42     5.63     5.34
juros_altos       hipotetico    -0.45    -0.92    -1.20     3.52     3.28     3.07
recessao          reverso       -1.50    -2.35    -3.00     4.78     4.85     4.57

Reverse stress "recessao" solution:
  selic  = 18.5 (baseline: 11.75, delta: +6.75)
  cambio =  6.2 (baseline:  5.10, delta: +1.10)
```

---

## Exemplo Completo: Stress Test de Credito Bancario

Cenario tipico de regulacao bancaria: avaliar o impacto de cenarios adversos
na projecao de **credito inadimplente** (NPL ratio).

```python
import pandas as pd
from forecastbox.auto import AutoVAR
from forecastbox.scenarios import StressTest
from forecastbox.viz import stress_test_plot

# Carregar dados do sistema financeiro
data = pd.read_csv(
    "credito_bancario.csv",
    index_col="date",
    parse_dates=True,
)
variables = ["npl_ratio", "pib", "selic", "desemprego", "cambio"]
var = AutoVAR(max_lags=4, ic="aic").fit(data[variables])

# Configurar stress test
stress = StressTest(var)

# Cenario 1: Replicar crise 2015-2016
stress.add_historical(
    "recessao_2015",
    start="2015-06",
    end="2016-12",
    variables=variables,
)

# Cenario 2: Choque hipotetico severo
stress.add_hypothetical(
    "choque_severo",
    shocks={
        "selic": 20.0,
        "desemprego": 16.0,
        "cambio": 7.0,
    },
)

# Cenario 3: Qual choque causa NPL > 8%?
stress.add_reverse(
    "npl_critico",
    target_variable="npl_ratio",
    target_value=8.0,
    shock_variables=["selic", "desemprego", "cambio"],
    horizon=8,
)

# Executar
results = stress.run(
    horizon=12,
    baseline=True,
    n_simulations=10000,
    seed=42,
)

# Resumo
print(results.summary())
```

```text
Stress Test Summary (model=AutoVAR, horizon=12)

Scenario          Type         npl_h4  npl_h8  npl_h12  pib_h12  selic_h12
────────────────────────────────────────────────────────────────────────────
baseline          —             3.45    3.62     3.78     0.65     11.75
recessao_2015     historico     4.82    5.91     6.35    -1.42     14.25
choque_severo     hipotetico   5.15    6.42     7.18    -2.10     20.00
npl_critico       reverso      5.85    7.28     8.00    -1.85     17.80

Reverse stress "npl_critico" solution:
  selic      = 17.8 (baseline: 11.75, delta: +6.05)
  desemprego = 14.5 (baseline: 11.20, delta: +3.30)
  cambio     =  6.1 (baseline:  5.10, delta: +1.00)
```

### Metricas de Risco

```python
# Metricas de impacto para NPL
metrics = results.impact_metrics(variable="npl_ratio")
print(metrics)
```

```text
Impact Metrics — npl_ratio (horizon=12)

                  deviation   VaR95    ES95    max_impact
baseline              0.00    1.15     1.52         0.42
recessao_2015         2.57    4.28     5.12         3.85
choque_severo         3.40    5.05     6.23         4.52
npl_critico           4.22    5.78     6.95         5.15
```

### Visualizacao

```python
# Grafico comparativo de cenarios
fig = stress_test_plot(
    results,
    variable="npl_ratio",
    title="Stress Test — NPL Ratio (%)",
    show_baseline=True,
    show_bands=True,
)
fig.show()
```

!!! example "Descricao Visual do Stress Test"

    O grafico resultante exibe:

    - **Linha preta solida**: baseline (cenario sem stress)
    - **Linha vermelha**: cenario "recessao_2015" (historico)
    - **Linha laranja**: cenario "choque_severo" (hipotetico)
    - **Linha roxa**: cenario "npl_critico" (reverso)
    - **Bandas sombreadas**: intervalos de confianca 90% para cada cenario
    - **Linha horizontal tracejada**: limiar regulatorio de NPL (ex: 8%)

    O eixo X mostra o horizonte de projecao; o eixo Y mostra o NPL ratio (%).
    A divergencia entre cenarios se amplifica com o horizonte, revelando quais
    choques representam maior risco ao sistema.

---

## Interpretacao e Limitacoes

!!! warning "Linearidade e Nao-Linearidades"

    Modelos VAR sao **lineares** — os efeitos de stress sao proporcionais ao
    tamanho do choque. Na realidade, crises geram **nao-linearidades**:

    - **Efeitos de limiar**: um banco pode absorver perdas moderadas, mas entrar
      em espiral quando NPL ultrapassa um limite critico
    - **Contagio**: a falha de uma instituicao pode amplificar o choque no sistema
    - **Mudanca de regime**: politicas monetarias extremas alteram a dinamica estrutural

    Interprete os resultados como **aproximacoes lineares** do impacto, nao como
    previsoes precisas do cenario de stress.

!!! warning "Escolha dos Cenarios"

    A qualidade do stress test depende da **plausibilidade** dos cenarios. Evite:

    - Cenarios **impossiveis** (Selic a 100%, cambio a R$ 50)
    - Cenarios **muito leves** que nao testam resiliencia
    - **Poucos cenarios** — teste multiplas combinacoes de choques

    Um bom stress test combina cenarios historicos (calibrados por crises reais)
    com cenarios hipoteticos (calibrados por julgamento de especialistas).

!!! note "Regulacao Bancaria"

    Para uso em contexto regulatorio (CCAR, DFAST, Basileia III), os cenarios
    de stress geralmente sao definidos pelo regulador (BCB, Fed). O forecastbox
    permite importar cenarios padronizados via `StressTest.from_regulatory()`,
    mas a validacao dos resultados segue as diretrizes do regulador.

---

## Ver Tambem

- [Fan Charts](fan-charts.md) — visualizar a distribuicao preditiva sob stress
- [Monte Carlo](monte-carlo.md) — simulacao que gera os intervalos de stress
- [Scenario Builder](scenario-builder.md) — construir cenarios complexos
- [Previsao Condicional](conditional.md) — base da previsao condicionada
