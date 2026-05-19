---
title: Cenarios e Previsao Condicional
description: Overview da analise de cenarios no forecastbox - previsao condicional, scenario builder, simulacao Monte Carlo, fan charts e stress testing.
---

# Cenarios e Previsao Condicional

Previsoes pontuais respondem a pergunta "qual o valor mais provavel?". Cenarios
respondem a pergunta mais util: **"o que acontece se...?"**. O modulo de cenarios
do forecastbox permite condicionar previsoes a hipoteses sobre o futuro, gerar
distribuicoes de caminhos possiveis e testar resiliencia a choques extremos.

---

## O que sao Cenarios?

Um cenario e uma **previsao condicionada a hipoteses explicitas** sobre variaveis
exogenas ou endogenas. Em vez de projetar livremente, voce fixa trajetorias para
algumas variaveis e observa como o restante do sistema responde.

!!! abstract "Key Takeaway"

    Cenarios transformam a previsao de um exercicio passivo ("o que vai acontecer?")
    em uma ferramenta ativa de planejamento ("se X acontecer, qual o impacto em Y?").
    Sao essenciais para politica monetaria, gestao de risco e planejamento estrategico.

---

## Tipos de Cenarios

O forecastbox suporta tres categorias fundamentais de cenarios:

| Tipo | Pergunta | Metodo | Exemplo |
|:-----|:---------|:-------|:--------|
| **Condicional** | Dado $X = \bar{x}$, qual $Y$? | VAR condicional, Waggoner-Zha | Se Selic = 12%, qual a inflacao? |
| **Estocastico** | Qual a distribuicao de caminhos futuros? | Monte Carlo, bootstrap | 10.000 trajetorias para o PIB |
| **Stress** | O que acontece em cenarios extremos? | Choques calibrados | Depreciacao de 30% do cambio |

### Condicional

Fixa o caminho de uma ou mais variaveis e projeta as demais condicionalmente.
Baseado na teoria de previsao condicional em VAR (Waggoner & Zha, 1999).

### Estocastico

Gera $S$ caminhos futuros amostrando da distribuicao de erros de previsao.
Permite estimar intervalos de previsao empiricos e fan charts.

### Stress Testing

Define choques extremos (mas plausveis) e avalia o impacto no sistema.
Util para regulacao bancaria (CCAR, DFAST) e gestao de risco corporativo.

---

## Relacao com VAR Condicional e Fan Charts

Cenarios condicionais e estocasticos sao complementares:

- A **previsao condicional** fornece o caminho central dado as hipoteses
- A **simulacao Monte Carlo** fornece a incerteza ao redor desse caminho
- Os **fan charts** visualizam ambos simultaneamente

$$
\underbrace{E[\mathbf{y}_{2,t+h} | \mathbf{y}_1 = \bar{\mathbf{y}}_1]}_{\text{cenario condicional}} \pm \underbrace{q_{\alpha/2} \cdot \hat{\sigma}_{2|1}}_{\text{incerteza Monte Carlo}}
$$

---

## Pipeline de Cenarios

O fluxo completo de analise de cenarios segue quatro etapas:

```mermaid
graph LR
    A["Hipoteses"] --> B["ScenarioBuilder"]
    B --> C["Forecast Condicional"]
    C --> D["Visualizacao"]

    style A fill:#E65100,stroke:#BF360C,color:#fff
    style B fill:#009688,stroke:#00796B,color:#fff
    style C fill:#1565C0,stroke:#0D47A1,color:#fff
    style D fill:#6A1B9A,stroke:#4A148C,color:#fff
```

1. **Definir hipoteses** — trajetorias, distribuicoes ou choques para variaveis-chave
2. **Construir cenario** — usar `ScenarioBuilder` para compor as hipoteses
3. **Gerar previsao condicional** — projetar o sistema condicionado ao cenario
4. **Visualizar** — fan charts, comparacao de cenarios, distribuicoes

---

## Quick Start

```python
import pandas as pd
from forecastbox.auto import AutoVAR
from forecastbox.scenarios import ScenarioBuilder, conditional_forecast

# Estimar VAR com variaveis macroeconomicas
data = pd.read_csv("macro_br.csv", index_col="date", parse_dates=True)
var = AutoVAR(max_lags=4, ic="aic").fit(data[["pib", "ipca", "selic", "cambio"]])

# Construir cenario: Selic fixa em 12% por 6 meses
scenario = (
    ScenarioBuilder()
    .set_variable("selic", path=[12.0, 12.0, 12.0, 12.0, 12.0, 12.0])
    .build()
)

# Gerar previsao condicional
fc = conditional_forecast(var, scenario=scenario, horizon=12)
print(fc)
```

```text
Conditional Forecast (horizon=12, conditions=1)

             pib     ipca    selic   cambio
2024-01     0.82     4.21    12.00     5.15
2024-02     0.79     4.18    12.00     5.18
2024-03     0.75     4.12    12.00     5.21
...
2024-12     0.68     3.85    12.00     5.34
```

---

## Secoes Disponiveis

<div class="grid cards" markdown>

-   :material-target:{ .lg .middle } **Previsao Condicional**

    ---

    Projete variaveis dado que outras seguem trajetorias pre-definidas.
    Restricoes hard e soft em modelos VAR.

    [:octicons-arrow-right-24: Previsao Condicional](conditional.md)

-   :material-wrench:{ .lg .middle } **Scenario Builder**

    ---

    API fluent para construir cenarios complexos com trajetorias fixas,
    distribuicoes e choques compostos.

    [:octicons-arrow-right-24: Scenario Builder](scenario-builder.md)

-   :material-dice-multiple:{ .lg .middle } **Monte Carlo**

    ---

    Simulacao de N caminhos futuros para estimacao de intervalos de
    previsao empiricos e distribuicoes preditivas.

    [:octicons-arrow-right-24: Monte Carlo](monte-carlo.md)

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Fan Charts**

    ---

    Visualizacao de incerteza com bandas de probabilidade progressivas
    e comparacao de cenarios.

    [:octicons-arrow-right-24: Fan Charts](fan-charts.md)

-   :material-alert-octagon:{ .lg .middle } **Stress Testing**

    ---

    Choques extremos calibrados para avaliacao de resiliencia e
    cenarios regulatorios.

    [:octicons-arrow-right-24: Stress Testing](stress-testing.md)

</div>

---

## Referencias

- **Waggoner, D.F. & Zha, T.** (1999). "Conditional Forecasts in Dynamic Multivariate Models." *Review of Economics and Statistics*, 81(4), 639-651.
- **Banbura, M., Giannone, D. & Lenza, M.** (2015). "Conditional Forecasts and Scenario Analysis with Vector Autoregressions for Large Cross-Sections." *International Journal of Forecasting*, 31(3), 739-756.
- **Kilian, L. & Lutkepohl, H.** (2017). *Structural Vector Autoregressive Analysis*. Cambridge University Press.
