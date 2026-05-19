---
title: Vintages
description: Gestao de vintages de dados para nowcasting em tempo real - VintageManager, armazenamento e reconstrucao de nowcasts.
---

# Gestao de Vintages

Um **vintage** e um snapshot dos dados macroeconomicos em um ponto especifico do
tempo. Vintages sao fundamentais para nowcasting porque capturam exatamente
**quais dados estavam disponiveis quando** — permitindo reconstruir nowcasts
historicos, avaliar performance em tempo real e entender o impacto de revisoes.

---

## Por que Vintages?

Dados macroeconomicos sao **revisados apos a publicacao**. O PIB do 1T24
publicado em maio pode ser diferente do valor revisado em agosto. Sem gestao
de vintages, e impossivel:

- Reconstruir o que o modelo sabia em cada momento
- Avaliar a performance do nowcast de forma justa (pseudo real-time)
- Distinguir entre news e revisao de dados
- Replicar analises passadas

!!! warning "Armadilha do Look-Ahead Bias"

    Usar dados revisados para avaliar um modelo de nowcasting e **look-ahead
    bias** — o modelo nunca teria acesso a esses dados no momento da previsao.
    Sempre avalie com vintages para resultados confiáveis.

---

## Conceito: Painel 3D

Vintages formam um **painel tridimensional**:

```text
              Variavel
              ┌───────────────────────┐
              │  PInd  Vendas  PMI    │
              │                       │
   Tempo ─────│  Jan    ✓      ✓   ✓ │──── Vintage 1 (Abr-15)
              │  Fev    ✓      ✓   ✓ │
              │  Mar    ✓      ·   ✓ │
              │  Abr    ·      ·   · │
              │                       │
              │  Jan    ✓      ✓   ✓ │──── Vintage 2 (Mai-15)
              │  Fev    ✓      ✓   ✓ │
              │  Mar    ✓      ✓   ✓ │
              │  Abr    ✓      ·   ✓ │
              └───────────────────────┘
```

Cada "fatia" ao longo da dimensao vintage mostra o estado dos dados em um
momento especifico, incluindo o ragged edge daquele momento.

---

## VintageManager

O `VintageManager` e a interface central para gerenciar vintages no forecastbox:

```python
from forecastbox.nowcast import VintageManager
import pandas as pd

# Criar manager
vm = VintageManager(frequency="M", storage="memory")

# Adicionar vintages
data_apr = pd.read_csv("dados_2024-04-15.csv", index_col="date", parse_dates=True)
data_may = pd.read_csv("dados_2024-05-15.csv", index_col="date", parse_dates=True)
data_jun = pd.read_csv("dados_2024-06-15.csv", index_col="date", parse_dates=True)

vm.add_vintage("2024-04-15", data_apr)
vm.add_vintage("2024-05-15", data_may)
vm.add_vintage("2024-06-15", data_jun)

# Consultar
print(vm.summary())
```

```text
VintageManager (frequency=M, storage=memory)

  Vintages: 3
  Variables: 22
  Date range: 2020-01 to 2024-06
  Latest vintage: 2024-06-15

  Vintage       Obs     Missing%    New data
  2024-04-15    1,936     8.3%      —
  2024-05-15    1,974     5.1%      38
  2024-06-15    2,012     2.8%      38
```

---

## Operacoes Principais

### Recuperar Vintage

```python
# Dados como estavam em 15/abril/2024
data = vm.get_vintage("2024-04-15")

# Dados mais recentes
latest = vm.get_latest()
```

### Comparar Vintages

```python
# Diferencas entre duas vintages
diff = vm.compare("2024-04-15", "2024-05-15")
print(diff)
```

```text
Vintage Comparison (2024-04-15 → 2024-05-15)

  New observations:
    prod_industrial  2024-04    102.3
    vendas_varejo    2024-03     98.7
    pmi_industria    2024-04     51.2

  Revisions:
    prod_industrial  2024-03    101.1 → 101.4  (+0.3)
    vendas_varejo    2024-02     97.2 →  97.5  (+0.3)
```

### Identificar Ragged Edge

```python
# Visualizar o ragged edge de uma vintage
vm.plot_ragged_edge("2024-05-15")
```

```text
Ragged Edge (vintage=2024-05-15, target_quarter=2024-Q2)

  Variable          Jan  Fev  Mar  Abr  Mai  Jun
  prod_industrial    ✓    ✓    ✓    ✓    ·    ·
  vendas_varejo      ✓    ✓    ✓    ·    ·    ·
  pmi_industria      ✓    ✓    ✓    ✓    ✓    ·
  emprego_formal     ✓    ✓    ✓    ✓    ·    ·
  energia_eletrica   ✓    ✓    ✓    ✓    ✓    ✓
```

---

## Armazenamento

O `VintageManager` suporta tres backends de armazenamento:

| Backend | Uso | Persistencia | Performance |
|:--------|:----|:-------------|:------------|
| `"memory"` | Prototipacao, analise interativa | Nao | Rapido |
| `"parquet"` | Producao, pipelines | Sim (arquivo) | Rapido |
| `"sql"` | Sistemas grandes, multi-usuario | Sim (banco) | Medio |

=== "Memory"

    ```python
    vm = VintageManager(storage="memory")
    ```

    Dados ficam em memoria. Ideal para prototipacao e notebooks.

=== "Parquet"

    ```python
    vm = VintageManager(
        storage="parquet",
        path="./vintages/",
    )
    ```

    Cada vintage e salva como um arquivo Parquet. Compressao eficiente e
    leitura rapida.

=== "SQL"

    ```python
    vm = VintageManager(
        storage="sql",
        connection_string="sqlite:///vintages.db",
    )
    ```

    Armazena em banco relacional. Suporta SQLite, PostgreSQL e MySQL.

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `frequency` | `str` | `"M"` | Frequencia dos dados: `"D"`, `"W"`, `"M"`, `"Q"` |
| `storage` | `str` | `"memory"` | Backend: `"memory"`, `"parquet"`, `"sql"` |
| `path` | `str` | `None` | Caminho para armazenamento Parquet |
| `connection_string` | `str` | `None` | String de conexao para SQL |
| `validate` | `bool` | `True` | Validar consistencia entre vintages |

---

## Exemplo: Reconstruir Evolucao do Nowcast

Reconstrua como o nowcast evoluiu ao longo de um trimestre inteiro:

```python
from forecastbox.nowcast import DFM, VintageManager, NewsDecomposition

# Carregar vintages quinzenais do 2T24
vm = VintageManager(storage="parquet", path="./vintages_2024q2/")

# Datas de vintage ao longo do trimestre
vintage_dates = [
    "2024-04-01", "2024-04-15",
    "2024-05-01", "2024-05-15",
    "2024-06-01", "2024-06-15",
    "2024-07-01",  # dados completos do trimestre
]

# Reconstruir nowcast em cada vintage
results = []
for vdate in vintage_dates:
    data = vm.get_vintage(vdate)
    dfm = DFM(n_factors=2, factor_lags=2).fit(data)
    nowcast = dfm.predict(target="pib", horizon=1)
    results.append({"vintage": vdate, "nowcast": nowcast.point})

# Visualizar evolucao
import pandas as pd
evolution = pd.DataFrame(results)
print(evolution)
```

```text
      vintage  nowcast
  2024-04-01     0.60
  2024-04-15     0.65
  2024-05-01     0.72
  2024-05-15     0.80
  2024-06-01     0.85
  2024-06-15     0.91
  2024-07-01     0.88

  PIB oficial (1a publicacao): 0.87
```

---

## Revisao de Dados e Impacto nas Previsoes

Vintages permitem medir o impacto de revisoes nos nowcasts:

```python
# Impacto de revisoes vs. news
for i in range(1, len(vintage_dates)):
    news = NewsDecomposition(
        model=dfm,
        vintage_old=vintage_dates[i-1],
        vintage_new=vintage_dates[i],
        target="pib",
    )
    result = news.decompose(vm, separate_revisions=True)
    print(
        f"{vintage_dates[i]}: "
        f"news={result.news_contribution:+.3f}, "
        f"revision={result.revision_contribution:+.3f}"
    )
```

```text
2024-04-15: news=+0.042, revision=+0.008
2024-05-01: news=+0.061, revision=+0.009
2024-05-15: news=+0.070, revision=+0.010
2024-06-01: news=+0.043, revision=+0.007
2024-06-15: news=+0.051, revision=+0.009
2024-07-01: news=-0.022, revision=-0.008
```

!!! tip "Diagnostico de Revisoes"

    Se revisoes consistentemente afetam os nowcasts em uma direcao, isso
    indica **vies sistematico** nos dados preliminares. Considere usar
    modelos que incorporam a estrutura de revisao (e.g., state-space
    com equacao de revisao).

---

## Construindo um Painel de Vintages

Para analises de larga escala, construa vintages automaticamente:

```python
from forecastbox.nowcast import VintageManager

# Construir vintages a partir de calendario de publicacao
vm = VintageManager(storage="parquet", path="./vintages/")

# Registrar calendario de publicacao
vm.register_release_calendar({
    "prod_industrial": {"lag_days": 45, "frequency": "M"},
    "vendas_varejo": {"lag_days": 50, "frequency": "M"},
    "pmi_industria": {"lag_days": 5, "frequency": "M"},
    "pib": {"lag_days": 75, "frequency": "Q"},
})

# Gerar vintages automaticamente com base no calendario
vm.build_vintages(
    data=historical_data,
    start="2020-01-01",
    end="2024-06-30",
    frequency="15D",  # vintage a cada 15 dias
)

print(f"Vintages gerados: {vm.n_vintages}")
```

```text
Vintages gerados: 109
```

---

## Ver Tambem

- :material-stethoscope: [Real-Time Diagnostic](../../diagnostics/real-time.md) — diagnostico de avaliacao em tempo real com vintage analysis, revision impact e timeliness
- :material-stethoscope: [News Diagnostic](../../diagnostics/news-diagnostic.md) — decomposicao da revisao do nowcast por contribuicao de cada indicador
- [News Decomposition](news.md) — como usar news no pipeline de nowcasting

## Referencias

- **Croushore, D.** (2011). "Frontiers of Real-Time Data Analysis." *Journal of Economic Literature*, 49(1), 72-100.
- **Croushore, D. & Stark, T.** (2001). "A Real-Time Data Set for Macroeconomists." *Journal of Econometrics*, 105(1), 111-130.
- **Jacobs, J.P.A.M. & van Norden, S.** (2011). "Modeling Data Revisions: Measurement Error and Dynamics of 'True' Values." *Journal of Econometrics*, 161(2), 101-109.
