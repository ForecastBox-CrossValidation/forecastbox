---
title: "Model Confidence Set"
description: "Construcao do Model Confidence Set (MCS) para identificar o conjunto de melhores modelos com garantia estatistica."
---

# Model Confidence Set

!!! abstract "Key Takeaway"
    O Model Confidence Set (Hansen, Lunde & Nason, 2011) identifica o **conjunto de modelos que contem o melhor** com probabilidade $(1 - \alpha)$. Em vez de eleger um unico vencedor, o MCS reconhece que a evidencia pode ser insuficiente para distinguir modelos de performance similar.

## Motivacao

Quando comparamos muitos modelos:

- Testes par-a-par (DM) sofrem do problema de **comparacoes multiplas**
- O ranking por metrica nao indica se as diferencas sao significativas
- O MCS resolve ambos: testa todos simultaneamente com controle de tamanho

## Formulacao

Dado um conjunto de $M_0$ modelos e uma funcao de perda $L$, defina o diferencial relativo:

$$
d_{ij,t} = L_{i,t} - L_{j,t}, \quad i,j \in M_0
$$

### Hipotese

$$
H_0^{M}: E[d_{ij,t}] = 0, \quad \forall \, i,j \in M
$$

O teste verifica se **todos os modelos no conjunto $M$ tem performance igual**.

### Algoritmo Sequencial de Eliminacao

O MCS e construido por eliminacao iterativa:

```text
1. Iniciar com M = M_0 (todos os modelos)
2. Testar H_0^M: todos em M sao equivalentes
3. Se H_0^M nao e rejeitada → M e o MCS, PARAR
4. Se H_0^M e rejeitada:
   a. Identificar o pior modelo (maior perda relativa)
   b. Remover do conjunto M
   c. Voltar ao passo 2
```

### Estatisticas de Teste

=== "Range ($T_R$)"

    $$
    T_R = \max_{i,j \in M} \frac{|\bar{d}_{ij}|}{\sqrt{\widehat{\text{var}}(\bar{d}_{ij})}}
    $$

    Baseada na maior diferenca padronizada entre quaisquer dois modelos.

=== "Semi-quadratica ($T_{SQ}$)"

    $$
    T_{SQ} = \sum_{i \in M} \left( \frac{\bar{d}_{i\cdot}}{\sqrt{\widehat{\text{var}}(\bar{d}_{i\cdot})}} \right)^2
    $$

    onde $\bar{d}_{i\cdot} = \frac{1}{|M|} \sum_{j \in M} \bar{d}_{ij}$ e a perda relativa media do modelo $i$.

    Mais poderosa que $T_R$ quando varios modelos sao simultaneamente inferiores.

### Bootstrap para P-valores

Como a distribuicao de $T_R$ e $T_{SQ}$ nao tem forma fechada, os p-valores sao obtidos por **bootstrap em blocos**:

1. Gerar $B$ amostras bootstrap (com dependencia temporal preservada)
2. Calcular a estatistica de teste em cada amostra
3. O p-valor e a fracao de amostras bootstrap com estatistica $\geq$ observada

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `losses` | DataFrame | — | Matriz $T \times M$ de perdas por modelo |
| `alpha` | float | `0.10` | Nivel de significancia |
| `statistic` | str | `"semi_quadratic"` | `"range"` ou `"semi_quadratic"` |
| `bootstrap` | str | `"block"` | Tipo de bootstrap: `"block"`, `"stationary"` |
| `n_boot` | int | `5000` | Numero de replicacoes bootstrap |
| `block_size` | int | `None` | Tamanho do bloco (auto se `None`) |

## Exemplo: MCS com 10 Modelos

```python
import pandas as pd
import numpy as np
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.combination import SimpleAverage, OLSCombination
from forecastbox.evaluation import mcs

# Suponha que temos previsoes de 10 modelos para o periodo de teste
# loss_matrix: DataFrame (T x 10) com erros quadraticos
loss_matrix = pd.DataFrame({
    "ARIMA(1,1,1)": (test - pred_arima1)**2,
    "ARIMA(2,1,1)": (test - pred_arima2)**2,
    "ETS(AAN)":     (test - pred_ets1)**2,
    "ETS(MAN)":     (test - pred_ets2)**2,
    "VAR(1)":       (test - pred_var1)**2,
    "VAR(2)":       (test - pred_var2)**2,
    "Naive":        (test - pred_naive)**2,
    "SES":          (test - pred_ses)**2,
    "Theta":        (test - pred_theta)**2,
    "Comb_OLS":     (test - pred_comb)**2,
})

# Model Confidence Set
result = mcs(loss_matrix, alpha=0.10, statistic="semi_quadratic", n_boot=5000)
print(result)
```

```text
Model Confidence Set (alpha=0.10)
=================================
Statistic: semi_quadratic
Bootstrap: block (B=5000)

Model           MCS p-value   Status
─────────────────────────────────────
Comb_OLS          1.0000      IN
ARIMA(2,1,1)      0.6234      IN
ETS(AAN)          0.4521      IN
ARIMA(1,1,1)      0.3187      IN
Theta             0.1245      IN
ETS(MAN)          0.0876      OUT
VAR(1)            0.0654      OUT
SES               0.0432      OUT
VAR(2)            0.0198      OUT
Naive             0.0021      OUT

Models in MCS: 5 of 10
Superior models: ['Comb_OLS', 'ARIMA(2,1,1)', 'ETS(AAN)', 'ARIMA(1,1,1)', 'Theta']
```

### Interpretacao dos P-valores MCS

O **p-valor MCS** de um modelo e o menor nivel $\alpha$ no qual ele seria excluido do MCS:

- $p_{\text{MCS}} \geq \alpha$: modelo esta **no MCS** (nao pode ser rejeitado como inferior)
- $p_{\text{MCS}} < \alpha$: modelo esta **fora do MCS** (evidencia de inferioridade)
- $p_{\text{MCS}} = 1.0$: ultimo modelo a ser potencialmente eliminado (ranking mais alto)

!!! info "Interpretacao pratica"
    No exemplo acima, com $\alpha = 0.10$:

    - **5 modelos** sobrevivem no MCS — nao ha evidencia suficiente para distingui-los
    - **Comb_OLS** tem o maior p-valor (1.0) — e o modelo mais dificil de eliminar
    - **Naive** tem o menor p-valor (0.002) — forte evidencia de inferioridade
    - A **combinacao OLS** esta no conjunto superior, sugerindo que agregar modelos agrega valor

### Acessando resultados

```python
# Modelos no MCS
print(result.superior_models)
# ['Comb_OLS', 'ARIMA(2,1,1)', 'ETS(AAN)', 'ARIMA(1,1,1)', 'Theta']

# P-valores
print(result.pvalues)
# Series com p-valores MCS para cada modelo

# Modelos eliminados (em ordem de eliminacao)
print(result.eliminated)
# ['Naive', 'VAR(2)', 'SES', 'VAR(1)', 'ETS(MAN)']
```

## Range vs Semi-Quadratica

| Caracteristica | $T_R$ (Range) | $T_{SQ}$ (Semi-Quadratica) |
|----------------|---------------|----------------------------|
| Foco | Pior par de modelos | Performance agregada |
| Poder | Menor | Maior quando varios modelos sao inferiores |
| MCS resultante | Maior (mais conservador) | Menor (mais agressivo) |
| Recomendacao | Quando quer evitar excluir bons modelos | Quando quer um MCS mais enxuto |

!!! tip "Recomendacao"
    Use `"semi_quadratic"` como padrao. Use `"range"` quando o custo de excluir indevidamente um bom modelo e alto.

## Ver Tambem

- [Diebold-Mariano](diebold-mariano.md) — comparacao par-a-par (2 modelos)
- [Giacomini-White](giacomini-white.md) — teste condicional
- [Metricas](metrics.md) — funcoes de perda para construir a matriz de perdas
- :material-stethoscope: [MCS — Diagnostico](../../diagnostics/mcs-diagnostic.md) — interpretacao pratica, analise de sensibilidade e heatmap de inclusao
