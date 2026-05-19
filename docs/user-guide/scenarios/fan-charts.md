---
title: "Fan Charts"
description: "Fan charts para visualizacao de incerteza: bandas de probabilidade, distribuicoes assimetricas e two-piece normal para balanco de riscos."
---

# Fan Charts

!!! abstract "Key Takeaway"

    Fan charts (graficos de leque) visualizam a **distribuicao futura** como bandas
    de probabilidade progressivas ao redor da previsao central. Inspirados no
    *Inflation Report* do Bank of England, permitem comunicar incerteza e
    **assimetria de riscos** de forma intuitiva para decisores.

---

## Conceito

Uma previsao pontual esconde a incerteza. Intervalos de confianca tradicionais
assumem simetria. O fan chart resolve ambos os problemas: exibe **bandas de
quantis** que se alargam com o horizonte, e permite que a distribuicao seja
**assimetrica** — refletindo o julgamento do analista sobre o balanco de riscos.

A construcao segue tres etapas:

1. **Previsao central** — mediana ou media do modelo
2. **Bandas de quantis** — faixas de probabilidade (ex: 10%, 25%, 50%, 75%, 90%)
3. **Ajuste de assimetria** — modo $\neq$ mediana quando ha riscos desbalanceados

```mermaid
graph LR
    A["Modelo Estimado"] --> B["Previsao Central"]
    B --> C["Distribuicao Preditiva"]
    C --> D["Quantis por Horizonte"]
    D --> E["Fan Chart"]

    style A fill:#E65100,stroke:#BF360C,color:#fff
    style C fill:#009688,stroke:#00796B,color:#fff
    style E fill:#1565C0,stroke:#0D47A1,color:#fff
```

---

## Distribuicoes Suportadas

O forecastbox suporta quatro distribuicoes para construcao de fan charts, cada
uma adequada a diferentes hipoteses sobre a forma da incerteza:

| Distribuicao | Parametros | Assimetria | Uso Recomendado |
|:-------------|:-----------|:-----------|:----------------|
| `"normal"` | $\mu, \sigma$ | Nao | Incerteza simetrica padrao |
| `"t"` | $\mu, \sigma, \nu$ | Nao | Caudas pesadas, amostras pequenas |
| `"two-piece-normal"` | $\mu, \sigma_1, \sigma_2$ | Sim | Balanco de riscos assimetrico |
| `"skew-normal"` | $\mu, \sigma, \alpha$ | Sim | Assimetria moderada e continua |

---

## Formulacao Matematica

### Distribuicao Normal

O caso base: bandas simetricas ao redor da previsao central.

$$
f(y) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(y - \mu)^2}{2\sigma^2}\right)
$$

Os quantis sao dados por $Q_p = \mu + \sigma \cdot z_p$, onde $z_p$ e o quantil
da normal padrao. As bandas se alargam linearmente com $\sigma_{t+h}$, que cresce
com o horizonte de previsao.

### Distribuicao t-Student

Para caudas mais pesadas que a normal, com $\nu$ graus de liberdade:

$$
f(y) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)\sigma} \left(1 + \frac{1}{\nu}\left(\frac{y-\mu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}}
$$

!!! info "Quando usar t-Student?"

    Use quando os residuos do modelo apresentam **excesso de curtose** ($\kappa > 3$).
    Valores tipicos: $\nu = 5$ a $10$ para dados macroeconomicos.
    Conforme $\nu \to \infty$, a distribuicao converge para a normal.

### Two-Piece Normal (Distribuicao Bimoda)

A distribuicao central para fan charts com **balanco de riscos assimetrico**.
Construida unindo duas metades de normais com desvios-padrao distintos:

$$
\boxed{f(y) = \begin{cases} \dfrac{2}{\sigma_1 + \sigma_2}\,\phi\!\left(\dfrac{y-\mu}{\sigma_1}\right) & y \leq \mu \\[10pt] \dfrac{2}{\sigma_1 + \sigma_2}\,\phi\!\left(\dfrac{y-\mu}{\sigma_2}\right) & y > \mu \end{cases}}
$$

onde $\phi(\cdot)$ e a densidade da normal padrao, e:

- $\mu$ e a **moda** da distribuicao (ponto de pico)
- $\sigma_1$ controla a dispersao para **baixo** (riscos de queda)
- $\sigma_2$ controla a dispersao para **cima** (riscos de alta)

As propriedades da two-piece normal sao:

$$
\text{Media} = \mu + \sqrt{\frac{2}{\pi}}(\sigma_2 - \sigma_1)
$$

$$
\text{Variancia} = \left(1 - \frac{2}{\pi}\right)(\sigma_2 - \sigma_1)^2 + \sigma_1 \sigma_2
$$

$$
\text{Assimetria} = \sqrt{\frac{2}{\pi}}(\sigma_2 - \sigma_1)\left[\left(\frac{4}{\pi} - 1\right)(\sigma_2 - \sigma_1)^2 + \sigma_1\sigma_2\right]
$$

!!! tip "Interpretacao do Balanco de Riscos"

    - $\sigma_1 = \sigma_2$: riscos simetricos (equivale a normal)
    - $\sigma_1 < \sigma_2$: **risco assimetrico para cima** — cauda direita mais longa
    - $\sigma_1 > \sigma_2$: **risco assimetrico para baixo** — cauda esquerda mais longa

    O Bank of England usa essa parametrizacao para comunicar o julgamento do
    comite de politica monetaria sobre o balanco de riscos da inflacao.

### Skew-Normal

Alternativa a two-piece normal com parametrizacao mais suave:

$$
f(y) = \frac{2}{\sigma}\,\phi\!\left(\frac{y - \mu}{\sigma}\right)\,\Phi\!\left(\alpha\,\frac{y - \mu}{\sigma}\right)
$$

onde $\Phi(\cdot)$ e a CDF da normal padrao e $\alpha$ controla a assimetria:

- $\alpha = 0$: distribuicao normal (simetrica)
- $\alpha > 0$: assimetria positiva (cauda direita mais pesada)
- $\alpha < 0$: assimetria negativa (cauda esquerda mais pesada)

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `forecast` | `ForecastResult` | — | Resultado de previsao (pontual ou Monte Carlo) |
| `quantiles` | `list[float]` | `[0.10, 0.25, 0.50, 0.75, 0.90]` | Quantis para as bandas do fan chart |
| `distribution` | `str` | `"normal"` | Distribuicao para as bandas: `"normal"`, `"t"`, `"two-piece-normal"`, `"skew-normal"` |
| `skew` | `float` | `0.0` | Parametro de assimetria (para `"skew-normal"`) |
| `mode_adjustment` | `float` | `0.0` | Deslocamento da moda em relacao a media (para `"two-piece-normal"`) |
| `sigma_lower` | `float` | `None` | $\sigma_1$ — dispersao para baixo (para `"two-piece-normal"`) |
| `sigma_upper` | `float` | `None` | $\sigma_2$ — dispersao para cima (para `"two-piece-normal"`) |
| `df` | `int` | `5` | Graus de liberdade (para `"t"`) |

---

## Construcao do Fan Chart

### A Partir de Monte Carlo

A forma mais direta: usar as simulacoes Monte Carlo para computar os quantis
empiricos por horizonte.

```python
from forecastbox.auto import AutoARIMA
from forecastbox.scenarios import monte_carlo_forecast
from forecastbox.viz import fan_chart

# Estimar modelo
model = AutoARIMA(seasonal=True, m=12).fit(ipca)

# Simulacao Monte Carlo
fc = monte_carlo_forecast(
    model=model,
    horizon=12,
    n_simulations=10000,
    error_distribution="empirical",
    seed=42,
)

# Fan chart direto dos quantis empiricos
fig = fan_chart(
    fc,
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    title="Projecao IPCA — Fan Chart",
)
fig.show()
```

### A Partir de Distribuicao Parametrica

Para incorporar julgamento sobre o balanco de riscos, especifique a distribuicao
e seus parametros:

```python
from forecastbox.viz import fan_chart

# Fan chart com two-piece normal (risco assimetrico para cima)
fig = fan_chart(
    fc,
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    distribution="two-piece-normal",
    sigma_lower=0.8,   # dispersao para baixo (menor)
    sigma_upper=1.2,    # dispersao para cima (maior)
    title="Projecao IPCA — Risco Assimetrico para Cima",
)
fig.show()
```

---

## Exemplo Completo: Inflacao com Risco Assimetrico

Cenario tipico de banco central: a inflacao tem previsao central de 4.5%,
mas os riscos estao assimetricos para cima devido a pressoes cambiais.

```python
import pandas as pd
from forecastbox.auto import AutoARIMA
from forecastbox.scenarios import monte_carlo_forecast
from forecastbox.viz import fan_chart

# Carregar dados mensais de IPCA
ipca = pd.read_csv("ipca_mensal.csv", index_col="date", parse_dates=True)["ipca"]

# Estimar modelo
model = AutoARIMA(seasonal=True, m=12).fit(ipca)

# Simulacao Monte Carlo (base para o fan chart)
fc = monte_carlo_forecast(
    model=model,
    horizon=12,
    n_simulations=10000,
    error_distribution="empirical",
    seed=42,
)

# Fan chart com balanco de riscos assimetrico para cima
fig = fan_chart(
    fc,
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    distribution="two-piece-normal",
    sigma_lower=0.7,    # riscos de queda contidos
    sigma_upper=1.3,    # riscos de alta ampliados (pressao cambial)
    title="Fan Chart — IPCA 12 meses (risco assimetrico)",
)
fig.show()
```

**Grafico esperado:**

!!! example "Descricao Visual do Fan Chart"

    O fan chart resultante exibe:

    - **Linha central solida** (preta): previsao pontual mediana (~4.5% a.a.)
    - **Banda 50%** (teal escuro, opacidade 0.4): intervalo entre quantis 25% e 75%
    - **Banda 80%** (teal medio, opacidade 0.3): intervalo entre quantis 10% e 90%
    - **Banda 90%** (teal claro, opacidade 0.2): caudas extremas

    **Assimetria visivel**: a banda superior (riscos de alta) e mais larga que
    a banda inferior (riscos de queda), refletindo $\sigma_2 > \sigma_1$.
    O "leque" se abre mais para cima, comunicando que o comite avalia riscos
    de inflacao acima da meta como mais provaveis que riscos de queda.

    **Eixo X**: horizonte de previsao (jan/2024 a dez/2024).
    **Eixo Y**: inflacao acumulada 12 meses (%).
    **Legenda**: identifica cada banda por nivel de probabilidade.

### Acessar Quantis Numericos

```python
# Quantis por horizonte
quantis = fc.quantiles([0.10, 0.25, 0.50, 0.75, 0.90])
print(quantis)
```

```text
Fan Chart Quantiles (h=1..12)

              q10     q25     q50     q75     q90
2024-01      4.12    4.28    4.48    4.68    4.84
2024-02      3.95    4.18    4.45    4.75    4.98
2024-03      3.78    4.08    4.42    4.82    5.12
...
2024-12      2.85    3.52    4.35    5.28    6.05
```

---

## Configuracao Visual

### Cores e Opacidade

```python
fig = fan_chart(
    fc,
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    colors={
        "central": "#000000",       # linha central preta
        "bands": "#009688",         # cor base das bandas (teal)
    },
    opacity={
        "50%": 0.4,                 # banda mais interna
        "80%": 0.3,                 # banda intermediaria
        "90%": 0.2,                 # banda mais externa
    },
    show_legend=True,
    show_history=24,                # mostrar ultimos 24 meses de dados observados
    title="Fan Chart — IPCA",
)
```

### Comparacao de Cenarios

Sobreponha fan charts de cenarios distintos para visualizar o impacto de
hipoteses alternativas:

```python
from forecastbox.viz import fan_chart_comparison

fig = fan_chart_comparison(
    forecasts={
        "Baseline": fc_baseline,
        "Selic alta": fc_selic_alta,
        "Choque cambial": fc_choque_cambial,
    },
    quantiles=[0.25, 0.75],         # apenas banda central para clareza
    title="Comparacao de Cenarios — IPCA",
)
fig.show()
```

---

## Interpretacao e Limitacoes

!!! warning "Interpretacao Cuidadosa"

    Fan charts comunicam **incerteza condicional ao modelo**. Eles **nao** capturam:

    - **Incerteza de modelo**: a possibilidade de o modelo estar errado
    - **Eventos sem precedente**: cisnes negros fora da distribuicao estimada
    - **Mudancas estruturais**: quebras de regime que invalidam os parametros

    A largura das bandas reflete a incerteza **historica** do modelo, nao
    necessariamente a incerteza **futura** real.

!!! warning "Assimetria Subjetiva"

    Quando se usa `distribution="two-piece-normal"` com $\sigma_1 \neq \sigma_2$,
    o balanco de riscos reflete **julgamento do analista**, nao uma propriedade
    estatistica do modelo. E essencial documentar as premissas:

    - Por que os riscos sao assimetricos?
    - Qual a magnitude da assimetria?
    - O que mudaria para reverter a direcao dos riscos?

!!! note "Fan Charts vs Intervalos de Confianca"

    Fan charts e intervalos de confianca sao complementares:

    - **Intervalos de confianca**: derivados de propriedades assintotiticas do modelo
    - **Fan charts**: derivados da distribuicao preditiva (empirica ou parametrica)

    Fan charts sao mais flexiveis porque permitem assimetria e nao dependem
    de hipoteses distributivas especificas.

---

## Ver Tambem

- [Monte Carlo](monte-carlo.md) — gerar a distribuicao preditiva empirica
- [Stress Testing](stress-testing.md) — cenarios extremos para as caudas do fan chart
- [Scenario Builder](scenario-builder.md) — construir cenarios condicionais
- [Previsao Condicional](conditional.md) — base teorica da previsao condicionada
