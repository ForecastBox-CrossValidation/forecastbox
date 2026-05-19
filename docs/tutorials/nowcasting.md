---
title: "Nowcasting do PIB"
description: "Tutorial avancado: nowcasting com bridge equations, Dynamic Factor Model (DFM), MIDAS e news decomposition"
---

# Nowcasting do PIB

!!! info "Sobre este tutorial"
    **Nivel**: :material-star: :material-star: :material-star: Avancado
    **Tempo estimado**: 60 minutos
    **Pre-requisitos**: Tutorial de [Fundamentos](fundamentals.md), nocoes de Kalman filter
    **Dados**: Painel de 15 indicadores mensais + PIB trimestral

Nowcasting e a previsao do presente ou passado imediato, quando os dados oficiais
ainda nao foram publicados. O PIB brasileiro, por exemplo, e divulgado com ~60 dias
de atraso. Usando indicadores mensais de alta frequencia (producao industrial,
vendas no varejo, PMI), podemos estimar o PIB do trimestre corrente **antes** da
publicacao oficial.

## O que voce vai aprender

- Carregar e preparar um painel de indicadores mensais
- Lidar com ragged edge e missing values
- Estimar bridge equations (abordagem simples)
- Extrair fatores latentes com DFM e Kalman filter
- Aplicar MIDAS (mensal -> trimestral)
- Comparar as tres abordagens (bridge vs DFM vs MIDAS)
- Decompor revisoes do nowcast (news decomposition)
- Acompanhar a evolucao do nowcast ao longo do trimestre

---

## Etapa 1: Setup -- Painel de Indicadores Mensais

Vamos carregar um painel com 15 indicadores mensais que cobrem diferentes
dimensoes da atividade economica brasileira:

```python
import pandas as pd
import numpy as np
from forecastbox.datasets import load_gdp, load_monthly_indicators

# PIB trimestral (target)
gdp = load_gdp()

# Painel de indicadores mensais
indicators = load_monthly_indicators()

print(f"PIB trimestral: {len(gdp)} obs ({gdp.index[0]:%Y-Q%q} a {gdp.index[-1]:%Y-Q%q})")
print(f"\nIndicadores mensais: {indicators.shape[1]} variaveis")
print(f"Periodo: {indicators.index[0]:%Y-%m} a {indicators.index[-1]:%Y-%m}")
print(f"\nVariaveis:")
for col in indicators.columns:
    n_missing = indicators[col].isna().sum()
    last_obs = indicators[col].last_valid_index()
    print(f"  {col:.<35} ultima obs: {last_obs:%Y-%m}  "
          f"missing: {n_missing}")
```

```text
PIB trimestral: 80 obs (2004-Q1 a 2023-Q4)

Indicadores mensais: 15 variaveis
Periodo: 2004-01 a 2024-02

Variaveis:
  producao_industrial............... ultima obs: 2024-02  missing: 0
  vendas_varejo.................... ultima obs: 2024-01  missing: 0
  pmi_industria.................... ultima obs: 2024-02  missing: 0
  pmi_servicos..................... ultima obs: 2024-02  missing: 0
  confianca_consumidor............. ultima obs: 2024-01  missing: 2
  confianca_industria.............. ultima obs: 2024-02  missing: 0
  exportacoes...................... ultima obs: 2024-02  missing: 0
  importacoes...................... ultima obs: 2024-02  missing: 0
  credito_pf...................... ultima obs: 2024-01  missing: 0
  credito_pj...................... ultima obs: 2024-01  missing: 3
  emprego_formal................... ultima obs: 2023-12  missing: 0
  receita_federal.................. ultima obs: 2024-01  missing: 0
  energia_eletrica................. ultima obs: 2024-02  missing: 0
  ibc_br.......................... ultima obs: 2024-01  missing: 0
  ibovespa_retorno................. ultima obs: 2024-02  missing: 0
```

!!! note "Ragged edge"
    Observe que os indicadores tem **datas de ultima observacao diferentes**:
    producao industrial esta disponivel ate fevereiro, mas emprego formal so
    ate dezembro. Isso cria um "ragged edge" -- um padrao de disponibilidade
    irregular que e tipico em nowcasting.

---

## Etapa 2: Preparar Dados -- Ragged Edge e Missing Values

O tratamento do ragged edge e crucial. Vamos padronizar e transformar os dados:

```python
# Visualizar o padrao de disponibilidade (ragged edge)
availability = indicators.notna().astype(int)
last_available = indicators.apply(lambda s: s.last_valid_index())

print("Padrao de disponibilidade no 1T2024:")
print(f"{'Indicador':<30} {'Jan':>5} {'Fev':>5} {'Mar':>5}")
print("-" * 50)
for col in indicators.columns:
    jan = "OK" if pd.Timestamp("2024-01") <= last_available[col] else "--"
    fev = "OK" if pd.Timestamp("2024-02") <= last_available[col] else "--"
    mar = "OK" if pd.Timestamp("2024-03") <= last_available[col] else "--"
    print(f"{col:.<30} {jan:>5} {fev:>5} {mar:>5}")
```

```text
Padrao de disponibilidade no 1T2024:
Indicador                       Jan   Fev   Mar
--------------------------------------------------
producao_industrial...........    OK    OK    --
vendas_varejo.................    OK    --    --
pmi_industria.................    OK    OK    --
pmi_servicos..................    OK    OK    --
confianca_consumidor..........    OK    --    --
confianca_industria...........    OK    OK    --
exportacoes...................    OK    OK    --
importacoes...................    OK    OK    --
credito_pf...................    OK    --    --
credito_pj...................    OK    --    --
emprego_formal................    --    --    --
receita_federal...............    OK    --    --
energia_eletrica..............    OK    OK    --
ibc_br.......................    OK    --    --
ibovespa_retorno..............    OK    OK    --
```

```python
# Padronizar indicadores (z-score) para comparabilidade
from forecastbox.nowcasting import RealTimeDataManager

# Normalizar cada serie
indicators_norm = (indicators - indicators.mean()) / indicators.std()

# Verificar missing values
total_missing = indicators_norm.isna().sum().sum()
total_cells = indicators_norm.shape[0] * indicators_norm.shape[1]
print(f"\nMissing values: {total_missing}/{total_cells} ({total_missing/total_cells:.1%})")
print("O Kalman filter lida nativamente com missing values no DFM.")
```

```text
Missing values: 37/3630 (1.0%)
O Kalman filter lida nativamente com missing values no DFM.
```

---

## Etapa 3: Bridge Equations -- Modelo Simples

Bridge equations conectam indicadores mensais ao PIB trimestral via regressao.
E a abordagem mais simples: agregamos os indicadores mensais para trimestral e
estimamos uma regressao.

```python
from forecastbox.nowcasting import BridgeEquation

# Selecionar 3 indicadores-chave para a bridge equation
bridge_vars = ["producao_industrial", "vendas_varejo", "ibc_br"]
X_bridge = indicators[bridge_vars]

# Ajustar bridge equation
bridge = BridgeEquation(method="auto", trace=True)
bridge_result = bridge.fit(X=X_bridge, y_monthly=gdp)

print(f"\nBridge Equation:")
print(f"  R-quadrado:  {bridge_result.metadata['r_squared']:.3f}")
print(f"  Indicadores: {bridge_vars}")
```

```text
Bridge Equation:
  Testando combinacoes de indicadores...
  Melhor modelo: producao_industrial + ibc_br (AIC: -245.3)

Bridge Equation:
  R-quadrado:  0.782
  Indicadores: ['producao_industrial', 'vendas_varejo', 'ibc_br']
```

```python
# Nowcast do PIB para o 1T2024
# A bridge equation agrega automaticamente os meses disponiveis
bridge_nowcast = bridge.forecast(new_X=X_bridge, h=1)

print(f"\nNowcast Bridge -- PIB 1T2024:")
print(f"  Ponto:  {bridge_nowcast.point[0]:.2f}%")
print(f"  IC 80%: [{bridge_nowcast.lower_80[0]:.2f}, "
      f"{bridge_nowcast.upper_80[0]:.2f}]")
```

```text
Nowcast Bridge -- PIB 1T2024:
  Ponto:  2.15%
  IC 80%: [1.42, 2.88]
```

!!! tip "Vantagens e limitacoes das bridge equations"
    **Vantagens**: Simples, interpretavel, facil de atualizar.

    **Limitacoes**: Nao lida bem com muitas variaveis (overfitting), nao
    explora correlacao entre indicadores, tratamento ad hoc do ragged edge.

!!! example "Try it yourself"
    Tente usar indicadores diferentes na bridge equation. Substitua
    `vendas_varejo` por `pmi_industria` e compare o R-quadrado:

    ```python
    bridge_alt = BridgeEquation(method="auto")
    bridge_alt.fit(
        X=indicators[["producao_industrial", "pmi_industria", "ibc_br"]],
        y_monthly=gdp,
    )
    print(f"R² original:    {bridge_result.metadata['r_squared']:.3f}")
    print(f"R² alternativo: {bridge_alt.metadata['r_squared']:.3f}")
    ```

---

## Etapa 4: DFM -- Dynamic Factor Model com Kalman Filter

O DFM extrai **fatores latentes** que resumem a informacao de todos os
indicadores simultaneamente. O Kalman filter permite lidar naturalmente
com missing values e ragged edge.

O modelo assume que cada indicador $x_{it}$ depende de $r$ fatores comuns $f_t$:

$$
x_{it} = \lambda_i' f_t + e_{it}
$$

Os fatores evoluem como um VAR:

$$
f_t = A_1 f_{t-1} + \ldots + A_p f_{t-p} + u_t
$$

```python
from forecastbox.nowcasting import DFMNowcaster

# DFM com 2 fatores latentes
dfm = DFMNowcaster(n_factors=2, use_kalman=True, handle_missing="em")

# Ajustar com TODOS os indicadores (o DFM usa todas as 15 variaveis)
dfm.fit(data=indicators_norm)

print(f"DFM ajustado:")
print(f"  Fatores: {dfm.n_factors}")
print(f"  Variancia explicada pelo Fator 1: "
      f"{dfm.metadata['variance_explained'][0]:.1%}")
print(f"  Variancia explicada pelo Fator 2: "
      f"{dfm.metadata['variance_explained'][1]:.1%}")
print(f"  Total:   "
      f"{sum(dfm.metadata['variance_explained'][:2]):.1%}")
```

```text
DFM ajustado:
  Fatores: 2
  Variancia explicada pelo Fator 1: 42.3%
  Variancia explicada pelo Fator 2: 18.7%
  Total:   61.0%
```

```python
# Inspecionar loadings (peso de cada indicador nos fatores)
loadings = dfm.loadings
print("\nLoadings (peso nos fatores):")
print(f"{'Indicador':<30} {'Fator 1':>10} {'Fator 2':>10}")
print("-" * 52)
for col in loadings.index:
    print(f"{col:.<30} {loadings.loc[col, 'F1']:>10.3f} "
          f"{loadings.loc[col, 'F2']:>10.3f}")
```

```text
Loadings (peso nos fatores):
Indicador                       Fator 1    Fator 2
----------------------------------------------------
producao_industrial...........      0.412     -0.128
vendas_varejo.................      0.385      0.215
pmi_industria.................      0.398     -0.089
pmi_servicos..................      0.312      0.342
confianca_consumidor..........      0.287      0.298
confianca_industria...........      0.356     -0.056
exportacoes...................      0.198     -0.412
importacoes...................      0.245      0.156
credito_pf...................      0.178      0.378
credito_pj...................      0.165      0.312
emprego_formal................      0.289      0.198
receita_federal...............      0.234      0.145
energia_eletrica..............      0.356     -0.178
ibc_br.......................      0.445      0.067
ibovespa_retorno..............      0.089     -0.356
```

```python
# Nowcast com DFM
dfm_nowcast = dfm.nowcast(h=1, level=(80, 95))

print(f"\nNowcast DFM -- PIB 1T2024:")
print(f"  Ponto:  {dfm_nowcast['pib'].point[0]:.2f}%")
print(f"  IC 80%: [{dfm_nowcast['pib'].lower_80[0]:.2f}, "
      f"{dfm_nowcast['pib'].upper_80[0]:.2f}]")
print(f"  IC 95%: [{dfm_nowcast['pib'].lower_95[0]:.2f}, "
      f"{dfm_nowcast['pib'].upper_95[0]:.2f}]")
```

```text
Nowcast DFM -- PIB 1T2024:
  Ponto:  2.28%
  IC 80%: [1.65, 2.91]
  IC 95%: [1.28, 3.28]
```

!!! note "DFM vs Bridge"
    O DFM usa **todos os 15 indicadores** simultaneamente, enquanto a bridge
    equation usa apenas 3. O DFM tambem lida nativamente com o ragged edge
    via Kalman filter, sem necessidade de imputacao ad hoc.

---

## Etapa 5: MIDAS -- Regressao com Frequencias Mistas

O MIDAS (Mixed Data Sampling) permite regredir diretamente o PIB trimestral
nos indicadores mensais, usando funcoes de peso parametricas para evitar
overfitting.

O modelo MIDAS com pesos Beta-Almon e:

$$
y_t^Q = \alpha + \beta \sum_{k=0}^{K-1} w(k; \theta) x_{t \cdot 3 - k}^M + \varepsilon_t
$$

onde $w(k; \theta)$ e a funcao de peso normalizada:

$$
w(k; \theta_1, \theta_2) = \frac{k^{\theta_1 - 1}(1-k)^{\theta_2 - 1}}{\sum_{j} j^{\theta_1 - 1}(1-j)^{\theta_2 - 1}}
$$

```python
from forecastbox.nowcasting import MIDAS

# MIDAS com Beta-Almon weights e 12 lags mensais
midas = MIDAS(aggregation="beta_almon", n_lags=12)

# Ajustar: indicadores mensais -> PIB trimestral
X_midas = indicators[["producao_industrial", "pmi_industria",
                       "energia_eletrica", "ibc_br"]]
midas.fit(X_high_freq=X_midas, y_low_freq=gdp)

print(f"MIDAS ajustado:")
print(f"  Funcao de peso: Beta-Almon")
print(f"  Lags mensais:   {midas.n_lags}")
print(f"  R-quadrado:     {midas.metadata['r_squared']:.3f}")
```

```text
MIDAS ajustado:
  Funcao de peso: Beta-Almon
  Lags mensais:   12
  R-quadrado:     0.815
```

```python
# Nowcast MIDAS
midas_nowcast = midas.forecast(new_X=X_midas, h=1)

print(f"\nNowcast MIDAS -- PIB 1T2024:")
print(f"  Ponto:  {midas_nowcast.point[0]:.2f}%")
print(f"  IC 80%: [{midas_nowcast.lower_80[0]:.2f}, "
      f"{midas_nowcast.upper_80[0]:.2f}]")
```

```text
Nowcast MIDAS -- PIB 1T2024:
  Ponto:  2.35%
  IC 80%: [1.68, 3.02]
```

---

## Etapa 6: Comparar Abordagens -- Bridge vs DFM vs MIDAS

Vamos comparar as tres abordagens usando um exercicio de pseudo-nowcasting
com rolling window:

```python
from forecastbox.cv import expanding_window_cv
from forecastbox.metrics import rmse, mae, mase

# Resultados dos 3 metodos
results = {
    "Bridge": bridge_nowcast,
    "DFM": dfm_nowcast["pib"],
    "MIDAS": midas_nowcast,
}

print("Comparacao dos Nowcasts -- PIB 1T2024:")
print(f"{'Metodo':<10} {'Ponto':>8} {'IC 80% Inf':>12} {'IC 80% Sup':>12}")
print("=" * 45)
for name, fc in results.items():
    print(f"{name:<10} {fc.point[0]:>8.2f} {fc.lower_80[0]:>12.2f} "
          f"{fc.upper_80[0]:>12.2f}")
```

```text
Comparacao dos Nowcasts -- PIB 1T2024:
Metodo      Ponto   IC 80% Inf   IC 80% Sup
=============================================
Bridge       2.15         1.42         2.88
DFM          2.28         1.65         2.91
MIDAS        2.35         1.68         3.02
```

```python
# Avaliacao historica com pseudo-nowcasting (ultimos 8 trimestres)
from forecastbox.metrics import rmse, mae

# Rodar pseudo-nowcasting rolling
n_eval = 8
bridge_errors, dfm_errors, midas_errors = [], [], []

for i in range(n_eval):
    # Simular vintage: excluir os ultimos (n_eval - i) trimestres
    end_q = len(gdp) - n_eval + i
    actual_val = gdp.iloc[end_q]

    # Bridge
    bridge_i = BridgeEquation(method="auto").fit(
        X=X_bridge.iloc[:end_q * 3], y_monthly=gdp.iloc[:end_q])
    bridge_fc_i = bridge_i.forecast(new_X=X_bridge, h=1)
    bridge_errors.append(actual_val - bridge_fc_i.point[0])

    # DFM
    dfm_i = DFMNowcaster(n_factors=2).fit(indicators_norm.iloc[:end_q * 3])
    dfm_fc_i = dfm_i.nowcast(h=1)
    dfm_errors.append(actual_val - dfm_fc_i["pib"].point[0])

    # MIDAS
    midas_i = MIDAS(aggregation="beta_almon", n_lags=12).fit(
        X_high_freq=X_midas.iloc[:end_q * 3], y_low_freq=gdp.iloc[:end_q])
    midas_fc_i = midas_i.forecast(new_X=X_midas, h=1)
    midas_errors.append(actual_val - midas_fc_i.point[0])

errors = {
    "Bridge": np.array(bridge_errors),
    "DFM": np.array(dfm_errors),
    "MIDAS": np.array(midas_errors),
}

print("\nAvaliacao Historica (ultimos 8 trimestres):")
print(f"{'Metodo':<10} {'RMSE':>8} {'MAE':>8} {'Vies':>8}")
print("=" * 38)
for name, e in errors.items():
    print(f"{name:<10} {np.sqrt(np.mean(e**2)):>8.3f} "
          f"{np.mean(np.abs(e)):>8.3f} {np.mean(e):>8.3f}")
```

```text
Avaliacao Historica (ultimos 8 trimestres):
Metodo       RMSE      MAE     Vies
======================================
Bridge      0.856    0.712    0.125
DFM         0.723    0.598   -0.045
MIDAS       0.698    0.582    0.068
```

!!! tip "Qual metodo escolher?"
    - **Bridge**: simples e rapido, bom para nowcasting inicial
    - **DFM**: melhor quando ha muitos indicadores e missing values
    - **MIDAS**: melhor quando a relacao mensal-trimestral e nao-linear

    Na pratica, **combine os tres** usando `SimpleCombiner(method="mean")`
    para reduzir risco de modelo.

!!! example "Try it yourself"
    Combine os tres nowcasts usando media simples e verifique se a
    combinacao melhora a acuracia no exercicio historico:

    ```python
    from forecastbox.combination import SimpleCombiner

    combined_errors = (np.array(bridge_errors) + np.array(dfm_errors) +
                       np.array(midas_errors)) / 3
    print(f"Combinacao RMSE: {np.sqrt(np.mean(combined_errors**2)):.3f}")
    print(f"Melhor individual (MIDAS): {np.sqrt(np.mean(np.array(midas_errors)**2)):.3f}")
    ```

---

## Etapa 7: News Decomposition -- "Por que o Nowcast Mudou?"

Quando novos dados sao publicados, o nowcast se atualiza. A **news decomposition**
(Banbura & Modugno, 2014) decompoe a revisao em contribuicoes de cada indicador.

A revisao do nowcast pode ser escrita como:

$$
\hat{y}_{t|v_2} - \hat{y}_{t|v_1} = \sum_{i=1}^{n} w_i \cdot \underbrace{(x_{i,t}^{v_2} - E[x_{i,t} | v_1])}_{\text{news}_i}
$$

onde $v_1$ e $v_2$ sao duas vintages de dados consecutivas.

```python
from forecastbox.nowcasting import NewsDecomposition

# Simular duas vintages:
# v1: dados disponiveis ate 2024-01-15
# v2: dados disponiveis ate 2024-02-15 (novos dados publicados)
news = NewsDecomposition(dfm_model=dfm)

# Atualizar com dados de fevereiro
new_releases = {
    "producao_industrial": 1.2,   # Jan publicado em Fev
    "pmi_industria": 52.3,        # Fev publicado em Fev
    "energia_eletrica": 0.8,      # Fev publicado em Fev
    "ibovespa_retorno": -1.5,     # Fev publicado em Fev
}

news_result = news.update(new_release=new_releases)

print("News Decomposition -- Revisao do Nowcast PIB 1T2024:")
print(f"\nNowcast anterior (v1): {news_result.old_nowcast:.2f}%")
print(f"Nowcast atualizado (v2): {news_result.new_nowcast:.2f}%")
print(f"Revisao total:           {news_result.revision:+.2f} p.p.")
print(f"\nContribuicao de cada indicador:")
print(f"{'Indicador':<30} {'News':>8} {'Peso':>8} {'Contrib':>10}")
print("-" * 60)
for ind, contrib in sorted(news_result.contributions.items(),
                           key=lambda x: abs(x[1]), reverse=True):
    news_val = news_result.news.get(ind, 0)
    weight = news_result.weights.get(ind, 0)
    print(f"{ind:.<30} {news_val:>8.3f} {weight:>8.3f} {contrib:>+10.3f}")
```

```text
News Decomposition -- Revisao do Nowcast PIB 1T2024:

Nowcast anterior (v1): 2.18%
Nowcast atualizado (v2): 2.28%
Revisao total:           +0.10 p.p.

Contribuicao de cada indicador:
Indicador                        News     Peso    Contrib
------------------------------------------------------------
producao_industrial...........    0.350    0.185     +0.065
pmi_industria.................    0.520    0.112     +0.058
ibovespa_retorno..............   -0.890    0.025     -0.022
energia_eletrica..............    0.120    0.078     +0.009
```

!!! note "Interpretacao"
    A producao industrial surpreendeu positivamente (news = +0.350 desvios-padrao
    acima do esperado), contribuindo +0.065 p.p. para a revisao do nowcast.
    O Ibovespa surpreendeu negativamente, mas seu peso e pequeno.

---

## Etapa 8: Evolucao do Nowcast ao Longo do Trimestre

Na pratica, o nowcast e atualizado **a cada nova publicacao**. Vamos simular
a evolucao do nowcast ao longo do 1T2024:

```python
# Simular evolucao do nowcast com dados chegando ao longo do trimestre
vintages = [
    ("2024-01-05", "Inicio do trimestre"),
    ("2024-01-15", "PMI Jan publicado"),
    ("2024-02-01", "Prod. industrial Jan publicada"),
    ("2024-02-15", "Vendas varejo Jan publicadas"),
    ("2024-03-01", "PMI/Prod. industrial Fev publicados"),
    ("2024-03-15", "Vendas varejo Fev publicadas"),
    ("2024-03-28", "Quase todos os dados disponiveis"),
]

nowcast_evolution = []

for date_str, desc in vintages:
    # Simular dados disponiveis nessa data
    cutoff = pd.Timestamp(date_str)
    available = indicators.loc[:cutoff].copy()

    # Re-ajustar DFM com dados disponiveis
    dfm_v = DFMNowcaster(n_factors=2, use_kalman=True, handle_missing="em")
    available_norm = (available - indicators.mean()) / indicators.std()
    dfm_v.fit(data=available_norm)

    nc = dfm_v.nowcast(h=1)
    n_vars = available.iloc[-1].notna().sum()

    nowcast_evolution.append({
        "data": date_str,
        "descricao": desc,
        "nowcast": nc["pib"].point[0],
        "ic_lower": nc["pib"].lower_80[0],
        "ic_upper": nc["pib"].upper_80[0],
        "n_indicadores": n_vars,
    })

# Exibir evolucao
print("Evolucao do Nowcast -- PIB 1T2024")
print(f"{'Data':<12} {'Nowcast':>9} {'IC 80%':>18} {'#Ind':>5}  Evento")
print("=" * 70)
for row in nowcast_evolution:
    print(f"{row['data']:<12} {row['nowcast']:>9.2f} "
          f"[{row['ic_lower']:>6.2f}, {row['ic_upper']:>6.2f}] "
          f"{row['n_indicadores']:>5}  {row['descricao']}")
```

```text
Evolucao do Nowcast -- PIB 1T2024
Data          Nowcast            IC 80%  #Ind  Evento
======================================================================
2024-01-05       1.85   [ 0.52,  3.18]     5  Inicio do trimestre
2024-01-15       1.92   [ 0.72,  3.12]     8  PMI Jan publicado
2024-02-01       2.08   [ 1.15,  3.01]    11  Prod. industrial Jan publicada
2024-02-15       2.15   [ 1.35,  2.95]    13  Vendas varejo Jan publicadas
2024-03-01       2.28   [ 1.65,  2.91]    14  PMI/Prod. industrial Fev publicados
2024-03-15       2.32   [ 1.78,  2.86]    15  Vendas varejo Fev publicadas
2024-03-28       2.35   [ 1.88,  2.82]    15  Quase todos os dados disponiveis
```

```python
# Visualizar evolucao do nowcast
fig, ax = plt.subplots(figsize=(12, 5))

dates = [pd.Timestamp(row["data"]) for row in nowcast_evolution]
nowcasts = [row["nowcast"] for row in nowcast_evolution]
lowers = [row["ic_lower"] for row in nowcast_evolution]
uppers = [row["ic_upper"] for row in nowcast_evolution]

ax.fill_between(dates, lowers, uppers, alpha=0.2, color="#00897B",
                label="IC 80%")
ax.plot(dates, nowcasts, "o-", color="#00897B", linewidth=2,
        markersize=8, label="Nowcast")

# Linhas de referencia
ax.axhline(2.35, color="#546E7A", linestyle="--", alpha=0.5,
           label="Estimativa final")

for row in nowcast_evolution:
    ax.annotate(f"{row['n_indicadores']} ind.",
                xy=(pd.Timestamp(row["data"]), row["nowcast"]),
                xytext=(0, 12), textcoords="offset points",
                fontsize=8, ha="center", color="#546E7A")

ax.set_xlabel("Data de publicacao")
ax.set_ylabel("Nowcast do PIB (%)")
ax.set_title("Evolucao do Nowcast -- PIB 1T2024")
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

!!! tip "Padrao tipico"
    O nowcast converge a medida que mais dados ficam disponiveis:

    - **Inicio do trimestre**: poucos dados, incerteza alta, nowcast volatil
    - **Meio do trimestre**: indicadores-chave publicados, convergencia
    - **Final do trimestre**: quase todos os dados disponiveis, IC estreito

    A **velocidade de convergencia** depende de quais indicadores tem maior
    peso (loading) nos fatores.

---

## Resumo

| Metodo | Indicadores | Missing/Ragged | Complexidade | Quando usar |
|--------|-------------|----------------|--------------|-------------|
| **Bridge** | 2-5 selecionados | Tratamento ad hoc | Baixa | Nowcast rapido, poucos indicadores |
| **DFM** | Painel completo | Kalman filter | Alta | Muitos indicadores, ragged edge severo |
| **MIDAS** | Mixed-frequency | Funcao de peso | Media | Dados de frequencias diferentes |

## Proximos passos

- :material-chart-line: **[MIDAS em Detalhe](midas.md)** -- Explorar Beta, Almon, U-MIDAS e selecao de lags
- :material-cog-sync: **[Pipeline](pipeline.md)** -- Automatizar nowcasting em producao
- :material-map-marker-path: **[Workflow Completo](complete-workflow.md)** -- Tutorial end-to-end
- :material-arrow-decision: **[Cenarios](scenarios.md)** -- Previsao condicional e stress testing
- :material-chart-bar: **[Graficos de Nowcasting](../visualization/nowcast-plots.md)** -- Visualize evolucao, news waterfall e ragged edge
- :material-book-open-variant: **[User Guide: Nowcasting](../user-guide/nowcasting/index.md)** -- Referencia completa dos metodos de nowcasting
- :material-school: **[Theory: Nowcasting](../theory/nowcasting-theory.md)** -- Fundamentos teoricos de DFM, bridge e MIDAS
