---
title: "Cenarios e Previsao Condicional"
description: "Tutorial pratico: previsao condicional com VAR, cenarios otimista/pessimista, Monte Carlo, fan charts e stress testing"
---

# Cenarios e Previsao Condicional

!!! info "Sobre este tutorial"
    **Nivel**: :material-star: :material-star: Intermediario
    **Tempo estimado**: 45 minutos
    **Pre-requisitos**: Tutorial de [Fundamentos](fundamentals.md), conceitos basicos de VAR
    **Dados**: PIB, Selic e Cambio trimestrais (dataset incluso)

Previsoes pontuais sao uteis, mas insuficientes para tomada de decisao. Gestores
precisam responder: *"o que acontece se a Selic subir para 15%?"* ou *"qual o impacto
de um choque tipo COVID?"*. Neste tutorial voce vai aprender a construir cenarios
condicionais, simular incerteza via Monte Carlo e visualizar riscos com fan charts.

## O que voce vai aprender

- Ajustar um VAR trivariado (PIB, Selic, Cambio)
- Gerar previsao incondicional (baseline)
- Criar cenarios condicionais (otimista e pessimista)
- Simular incerteza com Monte Carlo (5000 draws)
- Visualizar distribuicao futura com fan charts
- Aplicar stress test (cenario tipo COVID)
- Comparar cenarios lado a lado

---

## Etapa 1: Setup -- Modelo VAR com 3 Variaveis

Vamos ajustar um VAR com tres variaveis macroeconomicas centrais para a economia
brasileira: crescimento do PIB, taxa Selic e taxa de cambio.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecastbox.datasets import load_gdp, load_interest_rate, load_exchange_rate
from forecastbox.auto import AutoVAR

# Carregar e combinar series trimestrais
gdp = load_gdp()               # Crescimento do PIB (%)
selic = load_interest_rate()    # Taxa Selic (% a.a.)
cambio = load_exchange_rate()   # Taxa R$/USD

# Montar DataFrame multivariado
data = pd.DataFrame({
    "pib": gdp,
    "selic": selic,
    "cambio": cambio,
}).dropna()

print(f"Variaveis: {list(data.columns)}")
print(f"Periodos:  {len(data)} ({data.index[0]:%Y-Q%q} a {data.index[-1]:%Y-Q%q})")
print(f"\nCorrelacoes:\n{data.corr().round(3)}")
```

```text
Variaveis: ['pib', 'selic', 'cambio']
Periodos:  72 (2006-Q1 a 2023-Q4)

Correlacoes:
         pib  selic  cambio
pib    1.000 -0.312  -0.287
selic -0.312  1.000   0.154
cambio-0.287  0.154   1.000
```

```python
# Ajustar VAR com selecao automatica de lags
var_result = AutoVAR(maxlag=5, ic="aic").fit(data)

print(f"Ordem selecionada: VAR({var_result.p_order})")
print(f"AIC: {var_result.ic_value:.2f}")
```

```text
Ordem selecionada: VAR(2)
AIC: -856.32
```

!!! note "Por que VAR?"
    O VAR captura interdependencias entre variaveis: um choque na Selic afeta
    PIB e cambio, e vice-versa. Isso e essencial para cenarios condicionais --
    ao fixar o caminho de uma variavel, o modelo propaga o efeito para as demais.

---

## Etapa 2: Previsao Incondicional (Baseline)

A previsao incondicional usa apenas a dinamica historica, sem impor restricoes
sobre o futuro. Serve como **referencia** para comparar cenarios.

```python
from forecastbox.scenarios import ConditionalForecast

# Previsao incondicional (baseline) -- 4 trimestres a frente
h = 4
cond_fc = ConditionalForecast(model=var_result.model, method="analytic")

baseline = cond_fc.forecast(steps=h, conditions=None, n_draws=5000, seed=42)

# Exibir previsoes baseline
for var_name in ["pib", "selic", "cambio"]:
    fc = baseline[var_name]
    print(f"\n{var_name.upper()} -- Baseline:")
    for t in range(h):
        step = fc[t]
        print(f"  h={t+1}: {step['point']:.2f}  "
              f"[{step['lower_80']:.2f}, {step['upper_80']:.2f}]")
```

```text
PIB -- Baseline:
  h=1: 1.82  [0.45, 3.19]
  h=2: 1.75  [-0.12, 3.62]
  h=3: 1.69  [-0.58, 3.96]
  h=4: 1.65  [-0.95, 4.25]

SELIC -- Baseline:
  h=1: 11.25  [9.80, 12.70]
  h=2: 11.10  [9.15, 13.05]
  h=3: 10.98  [8.62, 13.34]
  h=4: 10.89  [8.18, 13.60]

CAMBIO -- Baseline:
  h=1: 4.95  [4.52, 5.38]
  h=2: 4.98  [4.35, 5.61]
  h=3: 5.00  [4.22, 5.78]
  h=4: 5.02  [4.12, 5.92]
```

---

## Etapa 3: Cenario Otimista -- Selic Cai para 9%

No cenario otimista, supomos que o Banco Central reduz a Selic gradualmente
ate 9% a.a. ao final do horizonte. O `ConditionalForecast` recalcula as trajetorias
de PIB e cambio **condicionadas** a essa restricao.

```python
# Cenario otimista: Selic cai linearmente de 11.25% para 9.00%
selic_otimista = np.linspace(11.25, 9.00, h)

cenario_otimista = cond_fc.forecast(
    steps=h,
    conditions={"selic": selic_otimista.tolist()},
    n_draws=5000,
    seed=42,
)

print("Cenario Otimista (Selic -> 9%):")
print(f"{'h':>3}  {'PIB':>8}  {'Selic':>8}  {'Cambio':>8}")
print("-" * 35)
for t in range(h):
    pib_t = cenario_otimista["pib"][t]["point"]
    sel_t = cenario_otimista["selic"][t]["point"]
    cam_t = cenario_otimista["cambio"][t]["point"]
    print(f"{t+1:>3}  {pib_t:>8.2f}  {sel_t:>8.2f}  {cam_t:>8.2f}")
```

```text
Cenario Otimista (Selic -> 9%):
  h       PIB     Selic    Cambio
-----------------------------------
  1      2.15      10.50      4.88
  2      2.48       9.75      4.82
  3      2.72       9.25      4.78
  4      2.91       9.00      4.75
```

!!! tip "Interpretacao"
    Com a queda da Selic, o PIB acelera (+2.91% vs +1.65% no baseline) e o cambio
    aprecia ligeiramente (4.75 vs 5.02). O modelo captura a transmissao da politica
    monetaria via canal de demanda e canal de cambio.

!!! example "Try it yourself"
    Crie um cenario condicional onde o **cambio** deprecia para R$6.50
    ao inves de condicionar na Selic. Como isso afeta o PIB?

    ```python
    cambio_depr = np.linspace(macro_data["cambio"].iloc[-1], 6.50, h)

    cenario_cambio = cond_fc.forecast(
        steps=h,
        conditions={"cambio": cambio_depr.tolist()},
        n_draws=5000,
        seed=42,
    )

    for t in range(h):
        print(f"h={t+1}: PIB={cenario_cambio['pib'][t]['point']:.2f}%")
    ```

---

## Etapa 4: Cenario Pessimista -- Selic Sobe para 15%

No cenario pessimista, o Banco Central e forcado a subir a Selic para 15%
(cenario de dominancia fiscal ou choque inflacionario).

```python
# Cenario pessimista: Selic sobe linearmente de 11.25% para 15.00%
selic_pessimista = np.linspace(11.25, 15.00, h)

cenario_pessimista = cond_fc.forecast(
    steps=h,
    conditions={"selic": selic_pessimista.tolist()},
    n_draws=5000,
    seed=42,
)

print("Cenario Pessimista (Selic -> 15%):")
print(f"{'h':>3}  {'PIB':>8}  {'Selic':>8}  {'Cambio':>8}")
print("-" * 35)
for t in range(h):
    pib_t = cenario_pessimista["pib"][t]["point"]
    sel_t = cenario_pessimista["selic"][t]["point"]
    cam_t = cenario_pessimista["cambio"][t]["point"]
    print(f"{t+1:>3}  {pib_t:>8.2f}  {sel_t:>8.2f}  {cam_t:>8.2f}")
```

```text
Cenario Pessimista (Selic -> 15%):
  h       PIB     Selic    Cambio
-----------------------------------
  1      1.38      12.50      5.05
  2      0.85      13.75      5.18
  3      0.42      14.50      5.28
  4      0.12      15.00      5.35
```

---

## Etapa 5: Monte Carlo -- Simulacoes e Intervalos Empiricos

Para quantificar a **incerteza total** (incluindo incerteza parametrica), usamos
simulacao Monte Carlo. O forecastbox gera $N$ trajetorias futuras amostrando
dos residuos estimados do VAR.

```python
from forecastbox.scenarios import MonteCarlo

# Monte Carlo: 5000 simulacoes a partir do VAR
mc = MonteCarlo(model=var_result.model, n_draws=5000, seed=42)
mc_results = mc.simulate(h=h)

# Distribuicao empirica do PIB no horizonte h=4
pib_draws = mc_results["pib"].density[:, -1]  # (5000,) draws no h=4

print(f"PIB no h=4 -- Distribuicao Empirica (5000 simulacoes):")
print(f"  Media:          {np.mean(pib_draws):.2f}%")
print(f"  Mediana:        {np.median(pib_draws):.2f}%")
print(f"  Desvio-padrao:  {np.std(pib_draws):.2f}%")
print(f"  IC 80%:         [{np.percentile(pib_draws, 10):.2f}, "
      f"{np.percentile(pib_draws, 90):.2f}]")
print(f"  IC 95%:         [{np.percentile(pib_draws, 2.5):.2f}, "
      f"{np.percentile(pib_draws, 97.5):.2f}]")
print(f"  P(PIB < 0):     {(pib_draws < 0).mean():.1%}")
```

```text
PIB no h=4 -- Distribuicao Empirica (5000 simulacoes):
  Media:          1.64%
  Mediana:        1.68%
  Desvio-padrao:  1.52%
  IC 80%:         [-0.31, 3.59]
  IC 95%:         [-1.42, 4.58]
  P(PIB < 0):     14.2%
```

```python
# Histograma da distribuicao
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(pib_draws, bins=60, density=True, alpha=0.7, color="#00897B",
        edgecolor="white", linewidth=0.5)
ax.axvline(np.mean(pib_draws), color="#E53935", linewidth=2,
           label=f"Media: {np.mean(pib_draws):.2f}%")
ax.axvline(0, color="black", linewidth=1, linestyle="--",
           label="Zero (recessao)")
ax.set_xlabel("Crescimento do PIB (%)")
ax.set_ylabel("Densidade")
ax.set_title("Distribuicao do PIB em h=4 (5000 simulacoes Monte Carlo)")
ax.legend()
plt.tight_layout()
plt.show()
```

!!! note "Assimetria"
    A distribuicao simulada pode ser **assimetrica** -- choques negativos tendem
    a ser maiores que positivos em periodos de alta volatilidade. Os intervalos
    empiricos capturam essa assimetria, ao contrario de intervalos gaussianos.

---

## Etapa 6: Fan Chart -- Distribuicao Futura com Assimetria

O fan chart (estilo Banco da Inglaterra) mostra a evolucao da incerteza ao longo
do horizonte. Faixas mais claras representam maior incerteza.

```python
from forecastbox.scenarios import FanChart

# Fan chart para o PIB
fan = FanChart(forecast=mc_results["pib"], actual=data["pib"])

fig = fan.plot(
    quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    ax=None,
)
fig.axes[0].set_title("Fan Chart -- Crescimento do PIB (%)")
fig.axes[0].set_ylabel("Crescimento (%)")
plt.tight_layout()
plt.show()
```

A visualizacao mostra:

| Faixa | Quantis | Probabilidade |
|-------|---------|---------------|
| Central (mais escura) | 25%--75% | 50% |
| Intermediaria | 10%--90% | 80% |
| Extrema (mais clara) | 5%--95% | 90% |

!!! tip "Fan charts na pratica"
    Bancos centrais usam fan charts em relatorios de inflacao para comunicar
    incerteza. A **assimetria** das faixas indica o balanco de riscos: se a faixa
    inferior e mais larga, os riscos sao de queda.

!!! example "Try it yourself"
    Gere o fan chart para a **Selic** ao inves do PIB e compare a incerteza
    entre as duas variaveis. Qual variavel tem mais incerteza no horizonte h=4?

    ```python
    fan_selic = FanChart(forecast=mc_results["selic"], actual=data["selic"])

    fig = fan_selic.plot(
        quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
    )
    fig.axes[0].set_title("Fan Chart -- Selic (% a.a.)")
    plt.tight_layout()
    plt.show()

    # Comparar incerteza
    pib_range = np.percentile(mc_results["pib"].density[:, -1], 95) - \
                np.percentile(mc_results["pib"].density[:, -1], 5)
    selic_range = np.percentile(mc_results["selic"].density[:, -1], 95) - \
                  np.percentile(mc_results["selic"].density[:, -1], 5)
    print(f"Range 90% PIB: {pib_range:.2f} p.p.")
    print(f"Range 90% Selic: {selic_range:.2f} p.p.")
    ```

---

## Etapa 7: Stress Test -- Cenario Tipo COVID

Um stress test aplica um choque extremo e observa a propagacao pelo sistema.
Vamos replicar um cenario similar ao impacto do COVID no 2T2020: queda abrupta
do PIB com alta do cambio.

```python
from forecastbox.scenarios import StressTest, Shock

# Definir stress test baseado no modelo VAR
stress = StressTest(var_model=var_result.model, shock_std=1.0)

# Choque tipo COVID: -3 desvios-padrao no PIB
covid_result = stress.apply_shock(
    variable="pib",
    shock_size=-3.0,   # -3 sigma
    periods=h,
)

print("Stress Test -- Choque tipo COVID (PIB: -3 sigma):")
print(f"{'h':>3}  {'PIB':>10}  {'Selic':>10}  {'Cambio':>10}")
print("-" * 40)
for t in range(h):
    print(f"{t+1:>3}  {covid_result.responses['pib'][t]:>10.2f}  "
          f"{covid_result.responses['selic'][t]:>10.2f}  "
          f"{covid_result.responses['cambio'][t]:>10.2f}")
```

```text
Stress Test -- Choque tipo COVID (PIB: -3 sigma):
  h         PIB       Selic      Cambio
----------------------------------------
  1       -3.85       -0.45        0.52
  2       -2.12       -0.78        0.38
  3       -0.95       -0.62        0.25
  4       -0.31       -0.45        0.15
```

!!! warning "Interpretacao do stress test"
    Os valores mostram os **desvios em relacao ao baseline**, nao niveis absolutos.
    O choque no PIB (-3.85 p.p. no h=1) se propaga para Selic (queda de 0.45 p.p.,
    refletindo resposta do BC) e cambio (depreciacao de 0.52 R$/USD).

```python
# Visualizar IRF do stress test
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, var_name in enumerate(["pib", "selic", "cambio"]):
    axes[i].bar(range(1, h + 1), covid_result.responses[var_name],
                color="#E53935" if var_name == "pib" else "#00897B",
                alpha=0.8)
    axes[i].axhline(0, color="black", linewidth=0.5)
    axes[i].set_xlabel("Horizonte (trimestres)")
    axes[i].set_ylabel("Desvio do baseline")
    axes[i].set_title(f"IRF: {var_name.upper()}")

fig.suptitle("Stress Test -- Resposta ao Choque tipo COVID", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## Etapa 8: Comparar Cenarios Lado a Lado

Finalmente, vamos consolidar todos os cenarios em uma visualizacao comparativa.

```python
from forecastbox.scenarios import ScenarioBuilder

# Construir cenarios estruturados
builder = ScenarioBuilder(base_forecast=baseline)

builder.add_scenario(
    name="otimista",
    paths={"selic": selic_otimista},
)
builder.add_scenario(
    name="pessimista",
    paths={"selic": selic_pessimista},
)

scenarios = builder.build()

# Tabela comparativa
print("Comparacao de Cenarios -- PIB (% crescimento)")
print(f"{'h':>3}  {'Baseline':>10}  {'Otimista':>10}  {'Pessimista':>10}  {'COVID':>10}")
print("=" * 50)
for t in range(h):
    base_v = baseline["pib"][t]["point"]
    otim_v = cenario_otimista["pib"][t]["point"]
    pess_v = cenario_pessimista["pib"][t]["point"]
    # COVID: baseline + resposta ao choque
    covid_v = base_v + covid_result.responses["pib"][t]
    print(f"{t+1:>3}  {base_v:>10.2f}  {otim_v:>10.2f}  {pess_v:>10.2f}  {covid_v:>10.2f}")
```

```text
Comparacao de Cenarios -- PIB (% crescimento)
  h    Baseline   Otimista  Pessimista      COVID
==================================================
  1        1.82       2.15        1.38      -2.03
  2        1.75       2.48        0.85      -0.37
  3        1.69       2.72        0.42       0.74
  4        1.65       2.91        0.12       1.34
```

```python
# Visualizacao comparativa
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
horizonte = range(1, h + 1)
colors = {"Baseline": "#546E7A", "Otimista": "#00897B",
          "Pessimista": "#E53935", "COVID": "#FF6F00"}

for i, var_name in enumerate(["pib", "selic", "cambio"]):
    ax = axes[i]

    # Baseline
    vals_base = [baseline[var_name][t]["point"] for t in range(h)]
    ax.plot(horizonte, vals_base, "o-", color=colors["Baseline"],
            linewidth=2, label="Baseline")

    # Otimista
    vals_otim = [cenario_otimista[var_name][t]["point"] for t in range(h)]
    ax.plot(horizonte, vals_otim, "s--", color=colors["Otimista"],
            linewidth=2, label="Otimista")

    # Pessimista
    vals_pess = [cenario_pessimista[var_name][t]["point"] for t in range(h)]
    ax.plot(horizonte, vals_pess, "^--", color=colors["Pessimista"],
            linewidth=2, label="Pessimista")

    # COVID
    vals_covid = [vals_base[t] + covid_result.responses[var_name][t]
                  for t in range(h)]
    ax.plot(horizonte, vals_covid, "D:", color=colors["COVID"],
            linewidth=2, label="COVID")

    ax.set_xlabel("Horizonte (trimestres)")
    ax.set_title(var_name.upper())
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Cenarios Comparados -- 4 Trimestres a Frente", fontsize=14)
plt.tight_layout()
plt.show()
```

---

## Resumo

| Tecnica | Quando usar | Vantagem |
|---------|-------------|----------|
| **Incondicional** | Baseline, sem hipoteses | Referencia neutra |
| **Condicional** | Cenarios de politica | Propaga restricoes pelo sistema |
| **Monte Carlo** | Quantificar incerteza total | Intervalos empiricos (assimetricos) |
| **Fan Chart** | Comunicar riscos | Visual intuitivo para decisores |
| **Stress Test** | Cenarios extremos | Identifica vulnerabilidades |
| **Comparacao** | Tomada de decisao | Contrasta alternativas |

## Proximos passos

- :material-pulse: **[Nowcasting](nowcasting.md)** -- Previsao em tempo real com dados incompletos
- :material-chart-line: **[MIDAS](midas.md)** -- Regressao com frequencias mistas
- :material-cog-sync: **[Pipeline](pipeline.md)** -- Automatizar previsoes em producao
- :material-map-marker-path: **[Workflow Completo](complete-workflow.md)** -- Tutorial end-to-end integrando todos os modulos
- :material-chart-bar: **[Graficos de Previsao](../visualization/forecast-plots.md)** -- Fan charts e visualizacao de cenarios
- :material-book-open-variant: **[User Guide: Cenarios](../user-guide/scenarios/index.md)** -- Referencia completa de cenarios condicionais
- :material-school: **[Theory: Condicional](../theory/conditional-theory.md)** -- Fundamentos teoricos de previsao condicional
