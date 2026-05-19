---
title: Nowcasting
description: Overview do nowcasting no forecastbox - Dynamic Factor Models, bridge equations, MIDAS, news decomposition e gestao de vintages.
---

# Nowcasting

Nowcasting e a arte de prever o **presente e o futuro proximo** usando dados de alta
frequencia que chegam em tempo real. Enquanto a previsao tradicional projeta meses
ou trimestres a frente, o nowcasting responde a pergunta mais urgente para policy-makers
e analistas: **"onde estamos agora?"**

---

## O que e Nowcasting?

O PIB trimestral, por exemplo, e publicado com **defasagem de 2-3 meses**. Mas
indicadores mensais (producao industrial, vendas no varejo, PMI) e diarios
(consumo de energia, transacoes financeiras) ja contem informacao sobre o trimestre
corrente. O nowcasting extrai essa informacao para antecipar o dado oficial.

!!! abstract "Key Takeaway"

    Nowcasting combina dados de **frequencias mistas** (diarios, mensais,
    trimestrais) que chegam com **defasagens distintas** para produzir uma
    estimativa em tempo real do estado da economia. A cada novo dado, o
    nowcast e atualizado e a contribuicao de cada informacao pode ser
    decomposta via **news decomposition**.

---

## O Desafio do Ragged Edge

O problema central do nowcasting e o **ragged edge**: ao final de cada periodo,
os dados disponiveis formam um painel irregular — alguns indicadores ja foram
publicados, outros nao.

```text
                Jan   Fev   Mar   Abr   Mai   Jun
PIB (Q)         ───────────┐
                           Q1          ???
Prod. Industrial  ✓     ✓     ✓     ✓     ·     ·
Vendas Varejo     ✓     ✓     ✓     ·     ·     ·
PMI               ✓     ✓     ✓     ✓     ✓     ·
Energia (diario)  ✓     ✓     ✓     ✓     ✓     ✓
```

Os modelos de nowcasting sao projetados para lidar com esse painel incompleto,
extraindo o maximo de informacao dos dados disponiveis.

---

## Modelos Disponiveis

O forecastbox oferece tres abordagens complementares para nowcasting:

| Modelo | Abordagem | Vantagem | Quando Usar |
|:-------|:----------|:---------|:------------|
| **DFM** | Fator latente via Kalman | Muitos indicadores, missing data | Painel grande (20+ series) |
| **Bridge** | Regressao com agregacao | Simplicidade, interpretabilidade | Poucos indicadores-chave |
| **MIDAS** | Ponderacao de alta frequencia | Dados diarios/semanais | Frequencias muito diferentes |

### Dynamic Factor Model (DFM)

Extrai fatores latentes comuns de um painel grande de indicadores usando o
**Filtro de Kalman**. Trata missing data naturalmente e permite nowcasting
com dezenas de series.

### Bridge Equations

Regressao simples do target trimestral em indicadores mensais **agregados
temporalmente**. Indicadores faltantes sao previstos por modelos auxiliares
(AR, random walk). Facil de interpretar e comunicar.

### MIDAS

Regressao direta com dados de frequencias mistas, usando **funcoes de
ponderacao parametricas** (Beta, Almon) para comprimir os lags de alta
frequencia. Ideal para combinar dados trimestrais com mensais ou diarios.

---

## Pipeline de Nowcasting

O fluxo completo de nowcasting no forecastbox segue cinco etapas:

```mermaid
graph LR
    A["Vintages"] --> B["Alinhamento"]
    B --> C["Modelo"]
    C --> D["Nowcast"]
    D --> E["News"]

    style A fill:#2E7D32,stroke:#1B5E20,color:#fff
    style B fill:#009688,stroke:#00796B,color:#fff
    style C fill:#1565C0,stroke:#0D47A1,color:#fff
    style D fill:#6A1B9A,stroke:#4A148C,color:#fff
    style E fill:#E65100,stroke:#BF360C,color:#fff
```

1. **Coletar vintages** — armazenar snapshots dos dados em cada ponto do tempo
2. **Alinhar frequencias** — tratar ragged edge e alinhar dados de frequencias mistas
3. **Estimar modelo** — DFM, bridge ou MIDAS
4. **Gerar nowcast** — previsao do periodo corrente com intervalo de confianca
5. **Decompor news** — identificar a contribuicao de cada novo dado para a revisao do nowcast

---

## Quick Start

```python
import pandas as pd
from forecastbox.nowcast import DFM, VintageManager

# Carregar painel de indicadores mensais
data = pd.read_csv("indicadores_br.csv", index_col="date", parse_dates=True)

# Criar vintage manager
vm = VintageManager(frequency="M")
vm.add_vintage("2024-04-15", data)

# Estimar DFM com 2 fatores
dfm = DFM(n_factors=2, factor_lags=2).fit(data)

# Nowcast do PIB trimestral
nowcast = dfm.predict(target="pib", horizon=1)
print(nowcast)
```

```text
Nowcast (target=pib, vintage=2024-04-15)

  Quarter    Nowcast    Lo95    Hi95
  2024-Q2      0.82    0.41    1.23

  Factors: 2 | Indicators: 22 | Missing: 8.3%
```

---

## Secoes Disponiveis

<div class="grid cards" markdown>

-   :material-blur:{ .lg .middle } **Dynamic Factor Model**

    ---

    Modelo de fator dinamico via Filtro de Kalman para paineis grandes
    com missing data. Selecao de fatores por Bai-Ng.

    [:octicons-arrow-right-24: DFM](dfm.md)

-   :material-bridge:{ .lg .middle } **Bridge Equations**

    ---

    Regressao do target trimestral em indicadores mensais agregados.
    Simples, interpretavel e eficaz.

    [:octicons-arrow-right-24: Bridge Equations](bridge.md)

-   :material-transfer-down:{ .lg .middle } **MIDAS**

    ---

    Mixed Data Sampling com funcoes de ponderacao parametricas
    para dados de frequencias mistas.

    [:octicons-arrow-right-24: MIDAS](midas.md)

-   :material-newspaper-variant:{ .lg .middle } **News Decomposition**

    ---

    Decomposicao da revisao do nowcast na contribuicao de cada
    novo dado — interpretacao economica do nowcast.

    [:octicons-arrow-right-24: News Decomposition](news.md)

-   :material-history:{ .lg .middle } **Vintages**

    ---

    Gestao de snapshots de dados para analise real-time,
    reconstrucao de nowcasts e revisao de dados.

    [:octicons-arrow-right-24: Vintages](vintages.md)

</div>

---

## Referencias

- **Banbura, M., Giannone, D., Modugno, M. & Reichlin, L.** (2013). "Now-casting and the Real-Time Data Flow." *Handbook of Economic Forecasting*, 2, 195-237.
- **Giannone, D., Reichlin, L. & Small, D.** (2008). "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics*, 55(4), 665-676.
- **Stock, J.H. & Watson, M.W.** (2002). "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics*, 20(2), 147-162.
