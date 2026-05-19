---
title: News Decomposition
description: Decomposicao da revisao do nowcast na contribuicao de cada novo dado - interpretacao economica em tempo real.
---

# News Decomposition

A news decomposition responde a pergunta mais importante para o analista de
conjuntura: **"por que o nowcast mudou?"**. Quando um novo dado e publicado
e o nowcast e atualizado, a decomposicao identifica **quanto cada novo dado
contribuiu** para a revisao.

---

## Conceito

Considere dois momentos no tempo:

- **Vintage $v_1$**: conjunto de dados disponivel antes da publicacao
- **Vintage $v_2$**: conjunto de dados apos a publicacao de novos dados

A revisao do nowcast pode ser decomposta:

$$
\underbrace{\hat{y}_{t|v_2} - \hat{y}_{t|v_1}}_{\text{revisao total}} = \sum_{i=1}^{N} \underbrace{w_i}_{\text{peso}} \times \underbrace{(x_{i,v_2} - E[x_i | v_1])}_{\text{news}_i}
$$

onde:

| Componente | Descricao |
|:-----------|:----------|
| $\hat{y}_{t\|v_2} - \hat{y}_{t\|v_1}$ | Revisao total do nowcast |
| $w_i$ | Peso (impacto marginal) do indicador $i$ no nowcast |
| $x_{i,v_2}$ | Valor observado do indicador $i$ na vintage $v_2$ |
| $E[x_i \| v_1]$ | Valor esperado de $x_i$ dado a informacao em $v_1$ |
| $\text{news}_i$ | Surpresa = observado $-$ esperado |

!!! abstract "Intuicao Economica"

    Se a producao industrial veio **acima do esperado** pelo modelo (news positiva)
    e o peso da producao industrial no nowcast do PIB e alto, isso **puxa o
    nowcast para cima**. A decomposicao quantifica exatamente quanto.

---

## Formalizacao (Banbura & Modugno, 2014)

No contexto do DFM, a decomposicao de news e derivada das propriedades do
Filtro de Kalman. Seja $\Omega_{v_1}$ e $\Omega_{v_2}$ os conjuntos de
informacao nas vintages $v_1$ e $v_2$.

### News

O vetor de news e a surpresa dos dados novos, condicionada na informacao antiga:

$$
\mathbf{n} = \mathbf{x}_{v_2}^{new} - E[\mathbf{x}^{new} | \Omega_{v_1}]
$$

### Peso

O peso de cada news no nowcast e:

$$
\mathbf{w} = \text{Cov}(y_t, \mathbf{x}^{new} | \Omega_{v_1}) \cdot \text{Var}(\mathbf{x}^{new} | \Omega_{v_1})^{-1}
$$

### Decomposicao

A revisao total se decompooe exatamente como:

$$
\hat{y}_{t|v_2} - \hat{y}_{t|v_1} = \mathbf{w}' \mathbf{n}
$$

Cada componente $w_i \cdot n_i$ representa a contribuicao do indicador $i$
para a revisao do nowcast.

---

## Uso no forecastbox

```python
from forecastbox.nowcast import DFM, NewsDecomposition, VintageManager

# Configurar vintages
vm = VintageManager(frequency="M")
vm.add_vintage("2024-04-15", data_april)
vm.add_vintage("2024-05-15", data_may)

# Estimar DFM
dfm = DFM(n_factors=2, factor_lags=2).fit(vm.get_latest())

# Decompor news entre duas vintages
news = NewsDecomposition(
    model=dfm,
    vintage_old="2024-04-15",
    vintage_new="2024-05-15",
    target="pib",
)
result = news.decompose(vm)
print(result)
```

```text
News Decomposition (target=pib)

  Nowcast (old):  0.72
  Nowcast (new):  0.85
  Revision:      +0.13

  Indicator            News    Weight   Contribution
  ─────────────────────────────────────────────────────
  prod_industrial     +1.2%     0.06       +0.072
  vendas_varejo       +0.8%     0.04       +0.032
  pmi_industria       -0.3%     0.03       -0.009
  emprego_formal      +0.5%     0.05       +0.025
  energia_eletrica    +0.2%     0.02       +0.004
  outros (17)            —        —        +0.006
  ─────────────────────────────────────────────────────
  Total                                    +0.130
```

---

## Interpretacao

A decomposicao permite narrativas economicas precisas:

!!! example "Exemplo de Narrativa"

    "O nowcast do PIB do 2T24 subiu de 0.72% para 0.85% (+0.13 pp) apos a
    publicacao dos dados de abril. A **producao industrial** foi o principal
    driver (+0.07 pp), vindo 1.2% acima do esperado pelo modelo. **Vendas no
    varejo** tambem surpreenderam positivamente (+0.03 pp). O **PMI** veio
    levemente abaixo do esperado, mas com impacto limitado (-0.01 pp)."

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `model` | `DFM` | — | Modelo DFM estimado |
| `vintage_old` | `str` | — | Data da vintage anterior |
| `vintage_new` | `str` | — | Data da vintage atualizada |
| `target` | `str` | — | Variavel-alvo do nowcast |
| `groupby` | `str` | `None` | Agrupar contribuicoes por categoria |

---

## Visualizacao

O forecastbox oferece graficos especializados para news:

```python
# Grafico de barras com contribuicoes
result.plot_contributions()

# Evolucao do nowcast com decomposicao ao longo do tempo
result.plot_waterfall()

# Heatmap de news por indicador e vintage
result.plot_news_heatmap()
```

### Waterfall

O grafico waterfall mostra como o nowcast evolui a cada nova publicacao:

```python
from forecastbox.nowcast import NewsDecomposition, VintageManager

# Decomposicao sequencial ao longo do trimestre
vintages = ["2024-04-01", "2024-04-15", "2024-05-01",
            "2024-05-15", "2024-06-01", "2024-06-15"]

waterfall = NewsDecomposition.sequential(
    model=dfm,
    vintages=vintages,
    target="pib",
    vintage_manager=vm,
)
waterfall.plot_waterfall()
```

```text
Nowcast Evolution (target=pib, 2024-Q2)

  0.60 ─────────────┐
                     ├── +0.05 (prod_ind Apr)
  0.65 ──────────────┤
                     ├── +0.07 (vendas Apr)
  0.72 ──────────────┤
                     ├── +0.08 (prod_ind May, emprego)
  0.80 ──────────────┤
                     ├── +0.05 (PMI, energia)
  0.85 ──────────────┤
                     ├── +0.06 (dados Jun)
  0.91 ──────────────┘
         Apr-01  Apr-15  May-01  May-15  Jun-01  Jun-15
```

---

## Agrupamento por Categoria

Agrupe contribuicoes por setor economico para uma visao macro:

```python
# Definir categorias
categories = {
    "Atividade": ["prod_industrial", "vendas_varejo", "servicos"],
    "Sentimento": ["pmi_industria", "icc", "ice"],
    "Mercado de Trabalho": ["emprego_formal", "desemprego"],
    "Financeiro": ["credito", "spread_bancario"],
}

result = news.decompose(vm, groupby=categories)
result.plot_contributions(by_group=True)
```

```text
Contributions by Category:
  Atividade             +0.09 pp  ██████████████
  Sentimento            -0.01 pp  █ (negative)
  Mercado de Trabalho   +0.03 pp  █████
  Financeiro            +0.02 pp  ███
```

---

## News com Revisao de Dados

A decomposicao distingue dois tipos de mudanca entre vintages:

1. **News**: dados novos que nao existiam na vintage anterior
2. **Revisao**: dados ja publicados que foram revisados

$$
\hat{y}_{t|v_2} - \hat{y}_{t|v_1} = \underbrace{\mathbf{w}_{news}' \mathbf{n}_{news}}_{\text{dados novos}} + \underbrace{\mathbf{w}_{rev}' \mathbf{n}_{rev}}_{\text{revisoes}}
$$

```python
result = news.decompose(vm, separate_revisions=True)
print(f"Contribuicao de dados novos: {result.news_contribution:+.3f}")
print(f"Contribuicao de revisoes:    {result.revision_contribution:+.3f}")
```

---

## Ver Tambem

- :material-stethoscope: [News Diagnostic](../../diagnostics/news-diagnostic.md) — diagnostico de consistencia, concentracao, surpresa sistematica e impacto marginal
- [Vintages](vintages.md) — gestao de vintages de dados para nowcasting
- [Real-Time Diagnostic](../../diagnostics/real-time.md) — avaliacao de performance em tempo real

## Referencias

- **Banbura, M. & Modugno, M.** (2014). "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics*, 29(1), 133-160.
- **Banbura, M., Giannone, D. & Reichlin, L.** (2011). "Nowcasting." *Oxford Handbook of Economic Forecasting*, 193-224.
- **Banbura, M., Giannone, D., Modugno, M. & Reichlin, L.** (2013). "Now-casting and the Real-Time Data Flow." *Handbook of Economic Forecasting*, 2, 195-237.
