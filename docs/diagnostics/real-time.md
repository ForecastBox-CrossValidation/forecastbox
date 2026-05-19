---
title: "Real-Time Diagnostic"
description: "Diagnostico de avaliacao em tempo real: vintage analysis, revision impact, timeliness, stability e metricas real-time vs pseudo-out-of-sample."
---

# Real-Time Diagnostic

!!! abstract "Key Takeaway"
    Avaliar um modelo de nowcasting com dados **revisados** e enganoso — o modelo nunca teve acesso a esses dados no momento da previsao. O diagnostico em tempo real compara a performance usando dados **como estavam disponiveis**, revelando o verdadeiro poder preditivo e o impacto das revisoes.

## Conceito

A avaliacao convencional de modelos usa dados **finais** (revisados), criando um **look-ahead bias**: o modelo e testado com informacao que nao existia quando a previsao foi feita.

$$
\underbrace{e_t^{\text{pseudo}}}_{\text{erro com dados revisados}} = y_t^{\text{final}} - \hat{y}_{t|t}^{\text{revisado}} \neq \underbrace{e_t^{\text{real-time}}}_{\text{erro verdadeiro}} = y_t^{\text{final}} - \hat{y}_{t|t}^{\text{real-time}}
$$

A diferenca pode ser substancial para variaveis como PIB, producao industrial e emprego, que sofrem revisoes significativas.

## Tipos de Dados

Tres conceitos fundamentais de avaliacao temporal:

=== "Real-Time Data"

    Dados **como estavam disponiveis** no momento da previsao. Requer um banco de vintages com historico completo.

    $$
    \hat{y}_{t|v_t} \text{ usando } \{x_{s,v_t}\}_{s \leq t}
    $$

    onde $v_t$ e o vintage disponivel no periodo $t$.

    !!! tip "Ouro da avaliacao"
        Esta e a unica avaliacao que reflete fielmente a experiencia do usuario. Tudo mais e aproximacao.

=== "Revised Data"

    Dados **finais** apos todas as revisoes. E o padrao da maioria das avaliacoes, mas introduz look-ahead bias.

    $$
    \hat{y}_{t|v_{\text{final}}} \text{ usando } \{x_{s,v_{\text{final}}}\}_{s \leq t}
    $$

    !!! warning "Cuidado"
        Um modelo que performa bem com dados revisados pode falhar em tempo real se depende de indicadores com grandes revisoes.

=== "Quasi-Real-Time"

    Dados **finais** mas com **timing real-time** — simula a disponibilidade temporal sem usar vintages historicos. Compromisso pratico quando nao ha banco de vintages.

    $$
    \hat{y}_{t|v_{\text{final}}} \text{ usando } \{x_{s,v_{\text{final}}}\}_{s \leq t} \text{ com ragged edges reais}
    $$

| Tipo | Dados | Timing | Requer Vintages? | Vies |
|------|-------|--------|-------------------|------|
| Real-time | Originais | Real | Sim | Nenhum |
| Revised | Finais | Real | Nao | Look-ahead |
| Quasi-real-time | Finais | Real | Nao | Parcial |

## Diagnosticos Real-Time

### 1. Vintage Analysis

Compara previsoes feitas com dados **reais** (vintage do momento) vs dados **revisados** (vintage final):

$$
\Delta_t = \hat{y}_{t|v_t}^{\text{real-time}} - \hat{y}_{t|v_{\text{final}}}^{\text{revisado}}
$$

Se $E[\Delta_t] \neq 0$, as revisoes de dados introduzem um vies sistematico na previsao.

```python
from forecastbox.diagnostics import real_time_diagnostic

diag = real_time_diagnostic(
    vintage_db=vintage_database,
    target="PIB",
    evaluation_vintage="real_time",
    metrics=["rmse", "mae", "bias"],
    period=("2020-Q1", "2025-Q4")
)

print("Vintage Analysis:")
print(f"{'Metrica':<15} {'Real-Time':>12} {'Revisado':>12} {'Diferenca':>12}")
print("-" * 55)
for m in diag.vintage_comparison:
    print(f"{m.name:<15} {m.real_time:>12.4f} {m.revised:>12.4f} "
          f"{m.difference:>+12.4f}")
```

```text
Vintage Analysis:
Metrica          Real-Time     Revisado    Diferenca
-------------------------------------------------------
RMSE                0.4523       0.3891      +0.0632
MAE                 0.3678       0.3102      +0.0576
Vies               +0.0845      +0.0123      +0.0722
```

!!! info "Interpretacao"
    O RMSE real-time (0.45) e **16% maior** que o pseudo-out-of-sample (0.39). Isso significa que avaliacoes convencionais **superestimam** a acuracia do modelo em 16%.

### 2. Revision Impact

Quantifica o quanto as revisoes de dados afetam a previsao. O **ratio de revisao** mede a proporcao da variancia do erro atribuivel a revisoes:

$$
\text{Revision Ratio} = \frac{\text{Var}(\hat{y}_{t|v_{\text{final}}} - \hat{y}_{t|v_t})}{\text{Var}(y_t - \hat{y}_{t|v_t})}
$$

| Revision Ratio | Interpretacao |
|----------------|---------------|
| $< 0.10$ | Revisoes tem impacto minimo — avaliacao convencional e aceitavel |
| $0.10 - 0.30$ | Impacto moderado — avaliacao real-time recomendada |
| $> 0.30$ | Impacto alto — avaliacao convencional e **enganosa** |

```python
print(f"\nRevision Impact:")
print(f"Var(revisao):       {diag.var_revision:.6f}")
print(f"Var(erro real-time):{diag.var_error_rt:.6f}")
print(f"Revision Ratio:     {diag.revision_ratio:.4f}")
print(f"Impacto:            {diag.revision_impact_level}")
```

```text
Revision Impact:
Var(revisao):       0.018734
Var(erro real-time):0.089456
Revision Ratio:     0.2094
Impacto:            moderado
```

#### Decomposicao por Indicador

Identifique quais indicadores tem as **maiores revisoes** e mais afetam a previsao:

```python
print(f"\nRevisao por Indicador:")
print(f"{'Indicador':<25} {'Revisao Media':>15} {'Std Revisao':>13} {'Max |Rev|':>11}")
print("-" * 68)
for ind in diag.indicator_revisions:
    print(f"{ind.name:<25} {ind.mean_revision:>+15.4f} "
          f"{ind.std_revision:>13.4f} {ind.max_abs_revision:>11.4f}")
```

```text
Revisao por Indicador:
Indicador               Revisao Media   Std Revisao   Max |Rev|
--------------------------------------------------------------------
PIB (1a estimativa)           +0.2340        0.4120      1.2300
Producao Industrial           +0.0890        0.2560      0.8900
Vendas Varejo                 +0.0230        0.0890      0.3100
PMI                           +0.0010        0.0120      0.0400
IPCA                          -0.0020        0.0150      0.0500
Taxa Desemprego               -0.0150        0.0670      0.2100
```

!!! warning "PIB e Producao Industrial"
    No Brasil, o PIB (1a estimativa do IBGE) e a Producao Industrial sofrem revisoes substanciais. Modelos que dependem fortemente destes indicadores podem ter performance real-time significativamente pior que a pseudo-out-of-sample.

### 3. Timeliness

O trade-off entre **velocidade** e **acuracia**: previsoes feitas mais cedo no trimestre usam menos dados, mas sao mais uteis para tomadores de decisao.

$$
\text{RMSE}(h) = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left(y_t - \hat{y}_{t|v_{t-h}}\right)^2}
$$

onde $h$ e o numero de semanas antes do fim do trimestre.

```python
print(f"\nTimeliness — RMSE por Antecedencia:")
print(f"{'Semanas antes':>15} {'RMSE':>8} {'N obs':>7} {'Ganho vs anterior':>20}")
print("-" * 54)
for t in diag.timeliness:
    gain = f"{t.gain_vs_previous:>+.4f}" if t.gain_vs_previous else "—"
    print(f"{t.weeks_ahead:>15} {t.rmse:>8.4f} {t.n_obs:>7} {gain:>20}")
```

```text
Timeliness — RMSE por Antecedencia:
  Semanas antes     RMSE   N obs   Ganho vs anterior
------------------------------------------------------
             12   0.6234      24                    —
              8   0.5123      24              -0.1111
              4   0.4012      24              -0.1111
              2   0.3456      24              -0.0556
              0   0.3102      24              -0.0354
```

```python
import matplotlib.pyplot as plt

weeks = [12, 8, 4, 2, 0]
rmse_values = [0.6234, 0.5123, 0.4012, 0.3456, 0.3102]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(weeks[::-1], rmse_values[::-1], "o-", color="teal",
        linewidth=2, markersize=8)
ax.fill_between(weeks[::-1], rmse_values[::-1], alpha=0.15, color="teal")
ax.set_xlabel("Semanas antes do fim do trimestre")
ax.set_ylabel("RMSE")
ax.set_title("Trade-off Timeliness: Velocidade vs Acuracia")
ax.invert_xaxis()
ax.set_xticks(weeks[::-1])
ax.set_xticklabels(["0\n(final)", "2", "4", "8", "12\n(inicio)"])
plt.tight_layout()
plt.show()
```

!!! tip "Regiao de maior ganho"
    O maior ganho marginal tipicamente ocorre nas **ultimas 4 semanas** do trimestre, quando dados de atividade real (producao industrial, vendas) sao publicados. Nas primeiras semanas, o nowcast depende mais de indicadores antecedentes (PMI, confianca).

### 4. Stability

A previsao muda **excessivamente** com pequenas revisoes de dados? Instabilidade indica overfitting ou sensibilidade excessiva a ruido:

$$
\text{Stability} = \frac{1}{V-1} \sum_{v=2}^{V} |\hat{y}_{t|v} - \hat{y}_{t|v-1}|
$$

O **ratio de instabilidade** normaliza pela variabilidade do erro:

$$
\text{Instability Ratio} = \frac{\text{Stability}}{\text{RMSE}_{\text{real-time}}}
$$

| Instability Ratio | Interpretacao |
|-------------------|---------------|
| $< 0.20$ | Estavel — revisoes pequenas nao afetam significativamente a previsao |
| $0.20 - 0.50$ | Moderadamente instavel — monitorar indicadores volateis |
| $> 0.50$ | Instavel — modelo pode estar overfitting ou excessivamente parametrizado |

```python
print(f"\nStability Diagnostic:")
print(f"Revisao media absoluta:  {diag.mean_abs_revision:.4f}")
print(f"Max revisao absoluta:    {diag.max_abs_revision:.4f}")
print(f"Instability Ratio:       {diag.instability_ratio:.4f}")
print(f"Estabilidade:            {diag.stability_level}")
```

```text
Stability Diagnostic:
Revisao media absoluta:  0.0823
Max revisao absoluta:    0.3412
Instability Ratio:       0.1820
Estabilidade:            estavel
```

```python
import matplotlib.pyplot as plt
import numpy as np

# Evolucao do nowcast para PIB Q4 2025 em diferentes vintages
vintages = ["Sep 15", "Oct 01", "Oct 15", "Nov 01", "Nov 15",
            "Dec 01", "Dec 15", "Jan 15"]
nowcast_q4 = [2.12, 2.15, 2.08, 2.14, 2.18, 2.22, 2.25, 2.28]
revisions = [0, 0.03, -0.07, 0.06, 0.04, 0.04, 0.03, 0.03]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                gridspec_kw={"height_ratios": [2, 1]},
                                sharex=True)

ax1.plot(vintages, nowcast_q4, "o-", color="teal", linewidth=2, markersize=8)
ax1.axhline(2.30, color="orange", linestyle="--", label="Realizado")
ax1.set_ylabel("Nowcast PIB (%)")
ax1.set_title("Estabilidade do Nowcast — PIB Q4 2025")
ax1.legend()

colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in revisions]
ax2.bar(vintages, revisions, color=colors, alpha=0.8, edgecolor="white")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Revisao (p.p.)")
ax2.set_xlabel("Vintage")

plt.tight_layout()
plt.show()
```

## Metricas Real-Time

### RMSE Real-Time vs Pseudo-Out-of-Sample

A comparacao central do diagnostico real-time:

$$
\text{RMSE}_{\text{RT}} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left(y_t^{\text{final}} - \hat{y}_{t|v_t}\right)^2}
$$

$$
\text{RMSE}_{\text{POoS}} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} \left(y_t^{\text{final}} - \hat{y}_{t|v_{\text{final}}}\right)^2}
$$

O **ratio de degradacao** mede quanto a performance piora em tempo real:

$$
\text{Degradation Ratio} = \frac{\text{RMSE}_{\text{RT}} - \text{RMSE}_{\text{POoS}}}{\text{RMSE}_{\text{POoS}}}
$$

### Ratio de Revisao

O ratio de revisao decompoe a variancia do erro de previsao em componentes:

$$
\text{Var}(e_t^{\text{RT}}) = \underbrace{\text{Var}(e_t^{\text{POoS}})}_{\text{erro do modelo}} + \underbrace{\text{Var}(r_t)}_{\text{efeito revisao}} + 2\text{Cov}(e_t^{\text{POoS}}, r_t)
$$

onde $r_t = \hat{y}_{t|v_{\text{final}}} - \hat{y}_{t|v_t}$ e a revisao da previsao devido a revisao dos dados.

$$
\text{Revision Ratio} = \frac{\text{Var}(r_t)}{\text{Var}(e_t^{\text{RT}})}
$$

!!! info "Interpretacao"
    Se o Revision Ratio e 0.20, **20% da variancia do erro real-time** e atribuivel a revisoes de dados — nao a falhas do modelo. Isso e informacao crucial para a tomada de decisao: o modelo nao pode ser melhorado para eliminar este componente.

## Exemplo Completo: Avaliacao Real-Time do PIB (2020-2025)

```python
import pandas as pd
from forecastbox.nowcasting import NowcastModel
from forecastbox.diagnostics import real_time_diagnostic
from forecastbox.data import VintageDatabase

# 1. Carregar banco de vintages
vintage_db = VintageDatabase.load("data/vintages_pib_2020_2025.parquet")

print(f"Vintages disponiveis: {vintage_db.n_vintages}")
print(f"Periodo: {vintage_db.start_date} a {vintage_db.end_date}")
print(f"Indicadores: {vintage_db.n_indicators}")

# 2. Carregar modelo
model = NowcastModel.load("models/pib_dfm.pkl")

# 3. Executar diagnostico completo
diag = real_time_diagnostic(
    vintage_db=vintage_db,
    target="PIB",
    model=model,
    evaluation_vintage="real_time",
    metrics=["rmse", "mae", "bias", "revision_ratio"],
    period=("2020-Q1", "2025-Q4")
)

# 4. Relatorio
print("=" * 65)
print("REAL-TIME DIAGNOSTIC — PIB Trimestral (2020-Q1 a 2025-Q4)")
print("=" * 65)

print(f"\n--- Performance Comparativa ---")
print(f"{'Metrica':<20} {'Real-Time':>12} {'Pseudo-OoS':>12} {'Degradacao':>12}")
print("-" * 60)
for m in diag.metrics_comparison:
    deg = f"{m.degradation_pct:>+.1f}%"
    print(f"{m.name:<20} {m.real_time:>12.4f} {m.pseudo_oos:>12.4f} {deg:>12}")

print(f"\n--- Revision Impact ---")
print(f"Revision Ratio:       {diag.revision_ratio:.4f}")
print(f"Impacto:              {diag.revision_impact_level}")
print(f"Var(erro modelo):     {diag.var_model_error:.6f}")
print(f"Var(revisao):         {diag.var_revision:.6f}")

print(f"\n--- Timeliness ---")
print(f"{'Antecedencia':<15} {'RMSE':>8}")
print("-" * 25)
for t in diag.timeliness:
    print(f"{t.label:<15} {t.rmse:>8.4f}")

print(f"\n--- Stability ---")
print(f"Instability Ratio:    {diag.instability_ratio:.4f}")
print(f"Nivel:                {diag.stability_level}")
```

```text
=================================================================
REAL-TIME DIAGNOSTIC — PIB Trimestral (2020-Q1 a 2025-Q4)
=================================================================

--- Performance Comparativa ---
Metrica               Real-Time   Pseudo-OoS   Degradacao
------------------------------------------------------------
RMSE                     0.4523       0.3891       +16.2%
MAE                      0.3678       0.3102       +18.6%
Vies                    +0.0845      +0.0123      +587.0%

--- Revision Impact ---
Revision Ratio:       0.2094
Impacto:              moderado
Var(erro modelo):     0.070722
Var(revisao):         0.018734

--- Timeliness ---
Antecedencia        RMSE
-------------------------
12 semanas        0.6234
8 semanas         0.5123
4 semanas         0.4012
2 semanas         0.3456
0 semanas         0.3102

--- Stability ---
Instability Ratio:    0.1820
Nivel:                estavel
```

!!! warning "Periodo COVID-19"
    Os trimestres de 2020 apresentam erros significativamente maiores tanto em tempo real quanto pseudo-out-of-sample. Considere reportar metricas **com e sem** o periodo pandemico para uma avaliacao mais robusta.

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `vintage_db` | `VintageDatabase` | — | Banco de dados de vintages |
| `target` | `str` | — | Variavel alvo (ex: `"PIB"`) |
| `model` | `NowcastModel` | — | Modelo de nowcasting |
| `evaluation_vintage` | `str` | `"real_time"` | `"real_time"`, `"revised"`, `"quasi_real_time"` |
| `metrics` | `list[str]` | `["rmse"]` | Metricas: `"rmse"`, `"mae"`, `"bias"`, `"revision_ratio"` |
| `period` | `tuple[str, str]` | `None` | Periodo de avaliacao `(inicio, fim)` |
| `timeliness_horizons` | `list[int]` | `[0, 2, 4, 8, 12]` | Semanas de antecedencia para timeliness |
| `alpha` | `float` | `0.05` | Nivel de significancia |

## Interpretacao e Decisoes

| Diagnostico | Resultado | Acao |
|------------|-----------|------|
| Degradation Ratio $> 20\%$ | Performance real-time muito pior | Investigar quais indicadores tem maiores revisoes; considerar modelo robusto a revisoes |
| Revision Ratio $> 0.30$ | Revisoes dominam o erro | Usar indicadores com menos revisoes; considerar quasi-real-time como benchmark |
| Timeliness — ganho marginal baixo nas ultimas semanas | Nowcast converge cedo | Modelo pode ser publicado mais cedo; dados tardios agregam pouco |
| Instability Ratio $> 0.50$ | Nowcast muito sensivel | Reduzir dimensionalidade; regularizar modelo; usar combinacao de modelos |

!!! note "Construindo um Vintage Database"
    O diagnostico real-time requer um banco de vintages. O forecastbox integra com o modulo de [Vintages](../user-guide/nowcasting/vintages.md) para construir e gerenciar bases de dados vintage a partir de fontes como IBGE, BCB e IPEA. Veja tambem o [SGS do Banco Central](https://www3.bcb.gov.br/sgspub/) para series com historico de revisoes.

## Proximos Passos

Com os diagnosticos de news e real-time completos, voce tem uma visao abrangente da qualidade do seu nowcast. Para um pipeline de diagnostico automatizado, veja a secao de [Pipeline](../user-guide/pipeline/index.md).

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [Nowcasting](../theory/nowcasting-theory.md) — fundamentos teoricos do nowcasting e avaliacao em tempo real
    - :material-notebook-edit: **User Guide**: [Vintages](../user-guide/nowcasting/vintages.md) — gestao de vintages de dados para avaliacao real-time
    - :material-link-variant: **Relacionado**: [News Diagnostic](news-diagnostic.md) — decomposicao de revisao por contribuicao de dados novos
    - :material-link-variant: **Relacionado**: [Pipeline](../user-guide/pipeline/index.md) — pipeline de diagnostico automatizado
