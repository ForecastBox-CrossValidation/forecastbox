---
title: "News Diagnostic"
description: "Diagnostico de news decomposition: por que a previsao mudou? Decomposicao da revisao em contribuicao de cada indicador, consistencia, concentracao, surpresa sistematica e impacto marginal."
---

# News Diagnostic

!!! abstract "Key Takeaway"
    A news decomposition responde a pergunta mais importante do nowcasting: **"por que minha previsao mudou?"**. Ela decompoe a revisao do nowcast na contribuicao de cada indicador, revelando quais dados novos moveram a previsao e em que direcao.

## Conceito

Quando novos dados economicos sao publicados, o nowcast e atualizado. A diferenca entre o nowcast antigo e o novo — a **revisao** — pode ser decomposta nas contribuicoes individuais de cada indicador:

$$
\underbrace{\hat{y}_{t|v_2} - \hat{y}_{t|v_1}}_{\text{revisao}} = \sum_{i=1}^{n} \underbrace{w_i}_{\text{peso}} \cdot \underbrace{(x_{i,v_2} - E[x_{i,v_2} | v_1])}_{\text{news}_i}
$$

onde:

- $\hat{y}_{t|v_j}$ e o nowcast do PIB no periodo $t$ baseado no vintage $v_j$
- $w_i$ e o **peso de impacto** do indicador $i$ (derivado do modelo DFM)
- $\text{news}_i = x_{i,v_2} - E[x_{i,v_2} | v_1]$ e a **surpresa** do indicador $i$ (dado observado menos a expectativa do modelo)

!!! warning "Pre-requisito"
    A news decomposition requer um **modelo DFM (Dynamic Factor Model) calibrado** com estrutura de fatores e equacao de observacao. O forecastbox implementa news decomposition integrada ao pipeline de nowcasting via `NowcastModel`.

## Diagnosticos de News

O forecastbox oferece quatro diagnosticos para avaliar a qualidade e o comportamento da news decomposition:

### 1. Consistencia

A soma das contribuicoes individuais deve ser **exatamente** igual a revisao total:

$$
\sum_{i=1}^{n} w_i \cdot \text{news}_i = \hat{y}_{t|v_2} - \hat{y}_{t|v_1}
$$

Se a diferenca excede uma tolerancia numerica ($\epsilon > 10^{-10}$), ha erro no modelo ou na implementacao.

```python
from forecastbox.diagnostics import news_diagnostic

diag = news_diagnostic(
    target="PIB",
    vintages=["2026-01-15", "2026-02-15"],
    model=nowcast_model
)

# Verificar consistencia
print(f"Revisao total:     {diag.revision:.6f}")
print(f"Soma contribuicoes: {diag.sum_contributions:.6f}")
print(f"Diferenca:          {diag.consistency_error:.2e}")
print(f"Consistente:        {diag.is_consistent}")
```

```text
Revisao total:     0.234500
Soma contribuicoes: 0.234500
Diferenca:          1.11e-16
Consistente:        True
```

### 2. Concentracao

Poucos indicadores podem dominar a revisao. O **indice de Herfindahl** dos pesos de contribuicao mede a concentracao:

$$
H = \sum_{i=1}^{n} s_i^2, \quad s_i = \frac{|w_i \cdot \text{news}_i|}{\sum_{j=1}^{n} |w_j \cdot \text{news}_j|}
$$

| $H$ | Interpretacao |
|-----|---------------|
| $H \approx 1/n$ | Contribuicoes uniformes — diversificacao maxima |
| $H > 0.25$ | Concentracao moderada — poucos indicadores dominam |
| $H > 0.50$ | Alta concentracao — revisao dependente de 1-2 indicadores |

```python
print(f"Herfindahl Index: {diag.herfindahl:.4f}")
print(f"N efetivo:        {diag.effective_n:.1f}")
print(f"Concentracao:     {diag.concentration_level}")
```

```text
Herfindahl Index: 0.3142
N efetivo:        3.2
Concentracao:     moderada
```

!!! info "N efetivo"
    O numero efetivo de indicadores e $N_{\text{eff}} = 1/H$. Se voce tem 10 indicadores mas $N_{\text{eff}} = 2$, apenas 2 indicadores efetivamente explicam a revisao.

### 3. Surpresa Sistematica

Indicadores que **sempre** surpreendem na mesma direcao revelam um problema de calibracao — o modelo sistematicamente subestima ou superestima aquele indicador:

$$
\text{sign\_ratio}_i = \frac{1}{V} \sum_{v=1}^{V} \mathbb{1}[\text{news}_{i,v} > 0]
$$

Se $\text{sign\_ratio}_i \approx 1$ ou $\text{sign\_ratio}_i \approx 0$ em multiplos vintages, o indicador $i$ tem surpresa sistematica.

```python
# Analise de surpresa sistematica (multiplos vintages)
vintages = [
    "2025-07-15", "2025-08-15", "2025-09-15",
    "2025-10-15", "2025-11-15", "2025-12-15",
    "2026-01-15", "2026-02-15"
]

sys_diag = news_diagnostic(
    target="PIB",
    vintages=vintages,
    model=nowcast_model,
    decomposition="systematic"
)

print("Surpresa Sistematica por Indicador:")
print(f"{'Indicador':<20} {'Sign Ratio':>12} {'Media News':>12} {'Alerta':>8}")
print("-" * 56)
for ind in sys_diag.systematic_surprise:
    flag = "***" if ind.is_systematic else ""
    print(f"{ind.name:<20} {ind.sign_ratio:>12.2f} {ind.mean_news:>+12.4f} {flag:>8}")
```

```text
Surpresa Sistematica por Indicador:
Indicador             Sign Ratio   Media News   Alerta
--------------------------------------------------------
Producao Industrial         0.88      +0.0234      ***
PMI Manufatura              0.75      +0.0156
Vendas Varejo               0.50      -0.0023
Confianca Consumidor        0.38      -0.0089
IPCA                        0.25      -0.0112      ***
Taxa Desemprego             0.50      +0.0045
Exportacoes                 0.62      +0.0078
Credito                     0.75      +0.0134
```

!!! warning "Indicador com surpresa sistematica"
    Se a Producao Industrial **sempre** surpreende para cima (sign ratio = 0.88), o modelo DFM provavelmente subestima a dinamica deste indicador. Considere re-estimar o modelo ou revisar a transformacao da serie.

### 4. Impacto Marginal

Qual indicador tem **maior impacto por unidade de surpresa**? O impacto marginal isola o efeito do peso $w_i$, independente do tamanho da surpresa:

$$
\text{impacto\_marginal}_i = |w_i| \cdot \sigma_{\text{news}_i}
$$

onde $\sigma_{\text{news}_i}$ e o desvio-padrao historico das surpresas do indicador $i$.

```python
print("Impacto Marginal por Indicador:")
print(f"{'Indicador':<20} {'|w_i|':>8} {'σ(news)':>10} {'Impacto':>10}")
print("-" * 52)
for ind in sorted(diag.marginal_impact, key=lambda x: x.impact, reverse=True):
    print(f"{ind.name:<20} {ind.abs_weight:>8.4f} {ind.news_std:>10.4f} {ind.impact:>10.4f}")
```

```text
Impacto Marginal por Indicador:
Indicador              |w_i|    σ(news)    Impacto
----------------------------------------------------
Producao Industrial    0.3210     0.0456     0.0146
PMI Manufatura         0.2870     0.0389     0.0112
Credito                0.1950     0.0512     0.0100
Vendas Varejo          0.2140     0.0345     0.0074
Exportacoes            0.1560     0.0423     0.0066
Confianca Consumidor   0.1230     0.0398     0.0049
IPCA                   0.0890     0.0312     0.0028
Taxa Desemprego        0.0650     0.0287     0.0019
```

## Visualizacoes

### Waterfall Chart: Contribuicao de cada indicador

O waterfall chart mostra como cada indicador contribui para a revisao total, partindo do nowcast antigo ate o novo:

```python
from forecastbox.visualization import plot_news_waterfall

fig = plot_news_waterfall(
    diag,
    title="Decomposicao da Revisao do Nowcast — PIB Q1 2026",
    figsize=(12, 6)
)
```

```python
import matplotlib.pyplot as plt
import numpy as np

# Dados da decomposicao
indicators = [
    "Nowcast\nAnterior", "Prod.\nIndustrial", "PMI", "Credito",
    "Vendas\nVarejo", "Exportacoes", "Confianca", "IPCA",
    "Desemprego", "Nowcast\nAtual"
]
contributions = [0, 0.0980, 0.0620, 0.0510, 0.0340, 0.0280,
                 -0.0150, -0.0120, -0.0115, 0]
base_value = 2.10  # nowcast anterior

# Construir waterfall
values = [base_value]
for c in contributions[1:-1]:
    values.append(values[-1] + c)
values.append(values[-1])

bottoms = [0] + [min(values[i], values[i] + contributions[i])
                 for i in range(1, len(values)-1)] + [0]
heights = [base_value] + [abs(c) for c in contributions[1:-1]] + [values[-1]]
colors = ["steelblue"] + ["#2ecc71" if c > 0 else "#e74c3c"
          for c in contributions[1:-1]] + ["steelblue"]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(indicators, heights, bottom=bottoms, color=colors, edgecolor="white")

# Conectar barras com linhas
for i in range(len(values)-1):
    ax.plot([i - 0.4, i + 1.4], [values[i], values[i]],
            color="gray", linewidth=0.8, linestyle="--")

ax.set_ylabel("Nowcast PIB (%)")
ax.set_title("Decomposicao da Revisao — Nowcast PIB Q1 2026")
ax.axhline(base_value, color="gray", linewidth=0.5, linestyle=":")
plt.tight_layout()
plt.show()
```

### Evolucao do Nowcast com News Bars

Acompanhe como o nowcast evolui ao longo do trimestre, com barras mostrando a contribuicao dos news em cada atualizacao:

```python
from forecastbox.visualization import plot_nowcast_evolution

fig = plot_nowcast_evolution(
    target="PIB",
    vintages=[
        "2025-10-15", "2025-11-01", "2025-11-15",
        "2025-12-01", "2025-12-15", "2026-01-15",
        "2026-01-31", "2026-02-15"
    ],
    model=nowcast_model,
    show_news_bars=True,
    title="Evolucao do Nowcast — PIB Q1 2026"
)
```

```python
import matplotlib.pyplot as plt
import numpy as np

vintages = ["Oct 15", "Nov 01", "Nov 15", "Dec 01",
            "Dec 15", "Jan 15", "Jan 31", "Feb 15"]
nowcasts = [1.85, 1.92, 2.01, 2.05, 2.08, 2.15, 2.28, 2.34]
revisions = [0, 0.07, 0.09, 0.04, 0.03, 0.07, 0.13, 0.06]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                gridspec_kw={"height_ratios": [2, 1]},
                                sharex=True)

# Nowcast evolution
ax1.plot(vintages, nowcasts, "o-", color="teal", linewidth=2, markersize=8)
ax1.axhline(2.30, color="orange", linestyle="--", label="Realizado (2.30%)")
ax1.fill_between(vintages, [n - 0.3 for n in nowcasts],
                 [n + 0.3 for n in nowcasts], alpha=0.15, color="teal")
ax1.set_ylabel("Nowcast PIB (%)")
ax1.set_title("Evolucao do Nowcast — PIB Q1 2026")
ax1.legend()

# News bars
colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in revisions]
ax2.bar(vintages, revisions, color=colors, edgecolor="white", alpha=0.8)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_ylabel("Revisao (p.p.)")
ax2.set_xlabel("Vintage")

plt.tight_layout()
plt.show()
```

### Scatter: Surpresa vs Impacto

O scatter plot relaciona o tamanho da surpresa com o impacto na revisao para cada indicador:

```python
from forecastbox.visualization import plot_news_scatter

fig = plot_news_scatter(
    diag,
    title="Surpresa vs Impacto — Nowcast PIB Q1 2026",
    annotate=True
)
```

```python
import matplotlib.pyplot as plt
import numpy as np

# Dados por indicador
names = ["Prod. Industrial", "PMI", "Credito", "Vendas Varejo",
         "Exportacoes", "Confianca", "IPCA", "Desemprego"]
surpresa = [0.305, 0.216, 0.261, 0.159, 0.179, -0.122, -0.135, -0.177]
impacto = [0.098, 0.062, 0.051, 0.034, 0.028, -0.015, -0.012, -0.012]
abs_weight = [0.321, 0.287, 0.195, 0.214, 0.156, 0.123, 0.089, 0.065]

fig, ax = plt.subplots(figsize=(10, 8))

# Tamanho proporcional ao peso
sizes = [w * 800 for w in abs_weight]
colors = ["#2ecc71" if i > 0 else "#e74c3c" for i in impacto]

scatter = ax.scatter(surpresa, impacto, s=sizes, c=colors,
                     alpha=0.7, edgecolors="white", linewidth=1.5)

# Anotacoes
for i, name in enumerate(names):
    ax.annotate(name, (surpresa[i], impacto[i]),
                textcoords="offset points", xytext=(10, 5),
                fontsize=9, alpha=0.8)

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Surpresa (news)")
ax.set_ylabel("Impacto na revisao")
ax.set_title("Surpresa vs Impacto — Nowcast PIB Q1 2026")

plt.tight_layout()
plt.show()
```

!!! tip "Leitura do scatter"
    - **Quadrante superior direito**: indicadores com surpresa positiva e impacto positivo (dado melhor que o esperado, revisao para cima)
    - **Quadrante inferior esquerdo**: indicadores com surpresa negativa e impacto negativo
    - **Tamanho do circulo**: proporcional ao peso $|w_i|$ do indicador no modelo
    - Indicadores **longe da reta** $y = w_i \cdot x$ tem peso atipico

## Exemplo Completo: Revisao do Nowcast do PIB Q1 2026

```python
import pandas as pd
from forecastbox.nowcasting import NowcastModel
from forecastbox.diagnostics import news_diagnostic
from forecastbox.visualization import (
    plot_news_waterfall, plot_nowcast_evolution, plot_news_scatter
)

# 1. Carregar modelo DFM calibrado
model = NowcastModel.load("models/pib_dfm_q1_2026.pkl")

# 2. Definir vintages para comparacao
vintage_old = "2026-01-15"  # antes dos dados de producao industrial
vintage_new = "2026-02-15"  # apos dados de jan/2026

# 3. Executar news diagnostic
diag = news_diagnostic(
    target="PIB",
    vintages=[vintage_old, vintage_new],
    model=model,
    indicators=[
        "producao_industrial", "pmi_manufatura", "vendas_varejo",
        "confianca_consumidor", "ipca", "taxa_desemprego",
        "exportacoes", "credito_pf"
    ],
    decomposition="full"
)

# 4. Resumo
print("=" * 60)
print(f"NEWS DIAGNOSTIC — PIB Q1 2026")
print(f"Vintage: {vintage_old} -> {vintage_new}")
print("=" * 60)
print(f"\nNowcast anterior:    {diag.nowcast_old:.2f}%")
print(f"Nowcast atual:       {diag.nowcast_new:.2f}%")
print(f"Revisao total:       {diag.revision:+.4f} p.p.")
print(f"\nConsistencia:        {'OK' if diag.is_consistent else 'FALHA'}")
print(f"Herfindahl:          {diag.herfindahl:.4f} ({diag.concentration_level})")
print(f"N efetivo:           {diag.effective_n:.1f} de {diag.n_indicators}")

# 5. Contribuicoes
print(f"\n{'Indicador':<20} {'News':>10} {'Peso':>8} {'Contrib.':>10} {'%':>8}")
print("-" * 60)
for c in diag.contributions:
    pct = 100 * abs(c.contribution) / abs(diag.revision)
    print(f"{c.name:<20} {c.news:>+10.4f} {c.weight:>8.4f} "
          f"{c.contribution:>+10.4f} {pct:>7.1f}%")

# 6. Visualizacoes
plot_news_waterfall(diag, title="Decomposicao — PIB Q1 2026")
plot_news_scatter(diag, title="Surpresa vs Impacto — PIB Q1 2026")
```

```text
============================================================
NEWS DIAGNOSTIC — PIB Q1 2026
Vintage: 2026-01-15 -> 2026-02-15
============================================================

Nowcast anterior:    2.10%
Nowcast atual:       2.34%
Revisao total:       +0.2345 p.p.

Consistencia:        OK
Herfindahl:          0.3142 (moderada)
N efetivo:           3.2 de 8

Indicador                  News     Peso    Contrib.        %
------------------------------------------------------------
Producao Industrial     +0.3050   0.3210    +0.0979    41.8%
PMI Manufatura          +0.2160   0.2870    +0.0620    26.4%
Credito                 +0.2615   0.1950    +0.0510    21.7%
Vendas Varejo           +0.1590   0.2140    +0.0340    14.5%
Exportacoes             +0.1790   0.1560    +0.0279    11.9%
Confianca Consumidor    -0.1220   0.1230    -0.0150     6.4%
IPCA                    -0.1350   0.0890    -0.0120     5.1%
Taxa Desemprego         -0.1770   0.0650    -0.0115     4.9%
```

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `target` | `str` | — | Variavel alvo (ex: `"PIB"`, `"IPCA"`) |
| `vintages` | `list[str]` | — | Par de vintages `[antigo, novo]` ou lista para analise temporal |
| `model` | `NowcastModel` | — | Modelo DFM calibrado |
| `indicators` | `list[str]` | `None` | Indicadores a incluir (default: todos do modelo) |
| `decomposition` | `str` | `"full"` | `"full"` (completa), `"systematic"` (surpresa sistematica) |
| `tolerance` | `float` | `1e-10` | Tolerancia para teste de consistencia |
| `alpha` | `float` | `0.05` | Nivel de significancia para testes de surpresa sistematica |

## Interpretacao e Decisoes

| Diagnostico | Resultado | Acao |
|------------|-----------|------|
| Consistencia falha | $\sum w_i \cdot \text{news}_i \neq \text{revisao}$ | Verificar modelo DFM — possivel bug ou dados corrompidos |
| Alta concentracao ($H > 0.5$) | 1-2 indicadores dominam | Monitorar esses indicadores com cuidado extra; considerar modelo com mais fatores |
| Surpresa sistematica | Sign ratio $> 0.8$ ou $< 0.2$ | Re-estimar modelo; revisar transformacao do indicador |
| Impacto marginal alto | Indicador com peso e volatilidade altos | Priorizar qualidade e tempestividade desse dado |

!!! note "Relacao com o User Guide"
    A pagina [News Decomposition](../user-guide/nowcasting/news.md) no User Guide explica **como** usar a news decomposition no pipeline de nowcasting. Esta pagina de diagnostico foca em **avaliar a qualidade** da decomposicao e identificar problemas.

## Proximos Passos

Apos diagnosticar as revisoes via news, avalie se as previsoes se mantem validas com dados em **tempo real** — veja [Real-Time Diagnostic](real-time.md).

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [News Decomposition](../theory/nowcasting-theory.md) — fundamentos teoricos da decomposicao de revisoes
    - :material-notebook-edit: **User Guide**: [News Decomposition](../user-guide/nowcasting/news.md) — como usar news no pipeline de nowcasting
    - :material-link-variant: **Relacionado**: [Real-Time Diagnostic](real-time.md) — diagnostico com dados em tempo real e vintages
    - :material-link-variant: **Relacionado**: [Vintages](../user-guide/nowcasting/vintages.md) — gestao de vintages de dados
