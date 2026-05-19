---
title: "Simulacao Monte Carlo"
description: "Simulacao Monte Carlo para previsao: bootstrap parametrico e nao-parametrico, intervalos de previsao empiricos e distribuicoes preditivas."
---

# Simulacao Monte Carlo

!!! abstract "Key Takeaway"

    A simulacao Monte Carlo gera $S$ caminhos futuros amostrando da distribuicao
    de erros de previsao. Permite estimar **intervalos de previsao empiricos** sem
    assumir normalidade, capturando toda a incerteza do processo gerador de dados.

---

## Conceito

A previsao pontual fornece o caminho mais provavel. Mas decisores precisam saber
**quao incerto** e esse caminho. A simulacao Monte Carlo responde isso gerando
milhares de trajetorias alternativas a partir da estrutura de erros do modelo.

Para cada simulacao $s = 1, \ldots, S$:

$$
\boxed{y_{t+h}^{(s)} = \hat{y}_{t+h} + \varepsilon_{t+h}^{(s)}, \quad s = 1, \ldots, S}
$$

onde $\varepsilon_{t+h}^{(s)}$ e amostrado da distribuicao estimada dos erros de previsao.

A colecao $\{y_{t+h}^{(1)}, \ldots, y_{t+h}^{(S)}\}$ forma a **distribuicao preditiva
empirica** para o horizonte $h$.

---

## Formulacao Matematica

### Modelo Geral

Considere um modelo de previsao com representacao:

$$
y_{t+h} = g(\mathbf{y}_t, \mathbf{y}_{t-1}, \ldots; \boldsymbol{\theta}) + \varepsilon_{t+h}
$$

A simulacao Monte Carlo propaga a incerteza em duas dimensoes:

1. **Incerteza dos erros futuros** ($\varepsilon$): choques aleatorios que afetam a trajetoria
2. **Incerteza dos parametros** ($\boldsymbol{\theta}$): os parametros estimados tem variancia

### Propagacao Multi-Passo

Para horizontes $h > 1$, os erros se acumulam via representacao MA($\infty$):

$$
y_{t+h}^{(s)} = \hat{y}_{t+h} + \sum_{j=0}^{h-1} \boldsymbol{\Phi}_j \varepsilon_{t+h-j}^{(s)}
$$

onde $\boldsymbol{\Phi}_j$ sao as matrizes de resposta ao impulso do modelo. A incerteza
**cresce com o horizonte** porque os erros se acumulam.

---

## Metodos de Bootstrap

O forecastbox implementa dois metodos para amostrar os erros:

=== "Parametric Bootstrap"

    Assume que os erros seguem uma distribuicao conhecida (tipicamente normal):

    $$
    \varepsilon_{t+h}^{(s)} \sim N(0, \hat{\sigma}^2)
    $$

    ```python
    from forecastbox.scenarios import monte_carlo_forecast

    fc = monte_carlo_forecast(
        model=var,
        horizon=12,
        n_simulations=10000,
        error_distribution="normal",  # parametric bootstrap
        seed=42,
    )
    ```

    **Vantagens**: simples, rapido, suave.

    **Desvantagens**: assume normalidade — pode subestimar caudas pesadas.

=== "Non-Parametric Bootstrap"

    Reamostra diretamente dos residuos do modelo — nao assume distribuicao:

    $$
    \varepsilon_{t+h}^{(s)} \sim \hat{F}_{\varepsilon} \quad \text{(distribuicao empirica dos residuos)}
    $$

    ```python
    fc = monte_carlo_forecast(
        model=var,
        horizon=12,
        n_simulations=10000,
        error_distribution="empirical",  # non-parametric bootstrap
        seed=42,
    )
    ```

    **Vantagens**: captura assimetria, caudas pesadas, sem hipotese distributiva.

    **Desvantagens**: limitado a choques historicos — nao gera eventos ineditos.

---

## Distribuicoes de Erros Disponiveis

| Distribuicao | Parametro | Descricao |
|:-------------|:----------|:----------|
| `"normal"` | `"normal"` | Gaussiana com $\mu=0$, $\sigma = \hat{\sigma}$ |
| `"t"` | `"t"` | t-Student com $\nu$ graus de liberdade estimados |
| `"empirical"` | `"empirical"` | Reamostragem dos residuos observados |
| `"skew-t"` | `"skew-t"` | t assimetrica (Hansen, 1994) |

```python
# Comparar distribuicoes
for dist in ["normal", "t", "empirical"]:
    fc = monte_carlo_forecast(
        model=model,
        horizon=12,
        n_simulations=10000,
        error_distribution=dist,
        seed=42,
    )
    print(f"{dist:>10s}: IC90 = [{fc.quantile(0.05):.2f}, {fc.quantile(0.95):.2f}]")
```

```text
    normal: IC90 = [3.21, 5.79]
         t: IC90 = [3.05, 5.95]
 empirical: IC90 = [3.12, 5.88]
```

!!! info "Escolha da Distribuicao"

    Use `"empirical"` como default — e o metodo mais robusto e nao requer
    hipoteses distributivas. Use `"t"` ou `"skew-t"` se os residuos apresentam
    caudas pesadas ou assimetria documentada.

---

## Intervalos de Previsao Empiricos

A partir das $S$ simulacoes, os intervalos de previsao sao estimados diretamente
pelos quantis empiricos:

$$
IC_{1-\alpha}(y_{t+h}) = \left[ Q_{\alpha/2}\left(\{y_{t+h}^{(s)}\}_{s=1}^S\right), \; Q_{1-\alpha/2}\left(\{y_{t+h}^{(s)}\}_{s=1}^S\right) \right]
$$

onde $Q_p$ e o quantil de ordem $p$ da distribuicao empirica.

```mermaid
graph LR
    A["Modelo Estimado"] --> B["Amostrar erros (S vezes)"]
    B --> C["Gerar S trajetorias"]
    C --> D["Calcular quantis"]
    D --> E["Intervalos de Previsao"]

    style A fill:#E65100,stroke:#BF360C,color:#fff
    style C fill:#009688,stroke:#00796B,color:#fff
    style E fill:#1565C0,stroke:#0D47A1,color:#fff
```

!!! tip "Quantas Simulacoes?"

    - $S = 1{,}000$: suficiente para intervalos de 90%
    - $S = 5{,}000$: recomendado para intervalos de 95%
    - $S = 10{,}000$: necessario para intervalos de 99% ou estimacao de caudas
    - $S = 50{,}000+$: para VaR e Expected Shortfall

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `model` | estimado | — | Modelo estimado (VAR, ARIMA, etc.) |
| `horizon` | `int` | — | Numero de periodos a simular |
| `n_simulations` | `int` | `5000` | Numero de trajetorias Monte Carlo |
| `error_distribution` | `str` | `"empirical"` | Distribuicao dos erros |
| `seed` | `int` | `None` | Semente para reproducibilidade |
| `include_parameter_uncertainty` | `bool` | `False` | Incluir incerteza dos parametros |
| `confidence_levels` | `list[float]` | `[0.50, 0.80, 0.90, 0.95]` | Niveis de confianca para intervalos |

---

## Exemplo Completo: Projecao de Inflacao

Simular 10.000 trajetorias para a inflacao (IPCA) nos proximos 12 meses:

```python
import pandas as pd
from forecastbox.auto import AutoARIMA
from forecastbox.scenarios import monte_carlo_forecast

# Carregar dados mensais de IPCA
ipca = pd.read_csv("ipca_mensal.csv", index_col="date", parse_dates=True)["ipca"]

# Estimar modelo
model = AutoARIMA(seasonal=True, m=12).fit(ipca)

# Simulacao Monte Carlo
fc = monte_carlo_forecast(
    model=model,
    horizon=12,
    n_simulations=10000,
    error_distribution="empirical",
    seed=42,
    confidence_levels=[0.50, 0.80, 0.90, 0.95],
)

print(fc.summary())
```

```text
Monte Carlo Forecast (S=10000, horizon=12)

             point    lo95    lo90    lo80    lo50    hi50    hi80    hi90    hi95
2024-01      0.52    0.28    0.32    0.37    0.44    0.60    0.67    0.72    0.76
2024-02      0.48    0.18    0.23    0.30    0.39    0.57    0.66    0.73    0.78
2024-03      0.55    0.15    0.22    0.31    0.43    0.67    0.79    0.88    0.95
...
2024-12      0.45   -0.12    0.01    0.15    0.32    0.58    0.75    0.89    1.02
```

### Acessar Simulacoes Individuais

```python
# Matriz de simulacoes: (n_simulations, horizon)
paths = fc.paths
print(f"Shape: {paths.shape}")  # (10000, 12)

# Estatisticas
print(f"Media h=12: {paths[:, -1].mean():.4f}")
print(f"Std h=12:   {paths[:, -1].std():.4f}")
print(f"Skewness:   {paths[:, -1].skew():.4f}")
print(f"Kurtosis:   {paths[:, -1].kurtosis():.4f}")
```

```text
Shape: (10000, 12)
Media h=12: 0.4512
Std h=12:   0.2891
Skewness:   0.1234
Kurtosis:   3.4567
```

### Probabilidades de Eventos

```python
# Probabilidade de inflacao mensal > 0.75% em dezembro
prob = (paths[:, -1] > 0.75).mean()
print(f"P(IPCA_dez > 0.75%) = {prob:.1%}")

# Probabilidade de deflacao em algum mes
prob_deflacao = (paths < 0).any(axis=1).mean()
print(f"P(deflacao em algum mes) = {prob_deflacao:.1%}")
```

```text
P(IPCA_dez > 0.75%) = 18.3%
P(deflacao em algum mes) = 12.7%
```

---

## Monte Carlo com Cenario Condicional

Combine previsao condicional com simulacao Monte Carlo para obter a **distribuicao
preditiva condicional**:

```python
from forecastbox.scenarios import ScenarioBuilder, conditional_forecast

# Cenario: Selic fixa em 12%
scenario = (
    ScenarioBuilder()
    .set_variable("selic", path=[12.0] * 12)
    .build()
)

# Previsao condicional com Monte Carlo
fc = conditional_forecast(
    model=var,
    scenario=scenario,
    horizon=12,
    method="gibbs",
    n_draws=10000,
    seed=42,
)

# Agora fc contem a distribuicao condicional completa
print(f"PIB medio (h=4): {fc['pib'].mean(axis=0)[3]:.2f}")
print(f"PIB IC90 (h=4):  [{fc['pib'].quantile(0.05)[3]:.2f}, "
      f"{fc['pib'].quantile(0.95)[3]:.2f}]")
```

```text
PIB medio (h=4): 0.72
PIB IC90 (h=4):  [0.31, 1.13]
```

---

## Incerteza dos Parametros

Por default, o Monte Carlo captura apenas a incerteza dos **erros futuros**.
Para incluir a incerteza dos **parametros estimados**, ative `include_parameter_uncertainty`:

```python
# Sem incerteza de parametros (default)
fc_errors = monte_carlo_forecast(
    model=model,
    horizon=12,
    n_simulations=10000,
    include_parameter_uncertainty=False,
    seed=42,
)

# Com incerteza de parametros
fc_full = monte_carlo_forecast(
    model=model,
    horizon=12,
    n_simulations=10000,
    include_parameter_uncertainty=True,
    seed=42,
)

print(f"IC95 (apenas erros):     [{fc_errors.quantile(0.025)[-1]:.2f}, "
      f"{fc_errors.quantile(0.975)[-1]:.2f}]")
print(f"IC95 (erros + params):   [{fc_full.quantile(0.025)[-1]:.2f}, "
      f"{fc_full.quantile(0.975)[-1]:.2f}]")
```

```text
IC95 (apenas erros):     [3.15, 5.85]
IC95 (erros + params):   [2.88, 6.12]
```

!!! info "Quando incluir incerteza de parametros?"

    - **Amostras pequenas** ($T < 100$): a incerteza dos parametros e relevante
    - **Horizontes longos** ($h > 8$): a incerteza se acumula
    - **Modelos complexos** (muitos parametros): mais parametros = mais incerteza
    - **Producao rapida**: desative para ganhar velocidade quando $T$ e grande

---

## Performance

| $S$ | $H$ | Variaveis | Tempo Aprox. |
|:----|:----|:----------|:-------------|
| 1.000 | 12 | 1 | ~0.1s |
| 10.000 | 12 | 1 | ~0.5s |
| 10.000 | 12 | 4 (VAR) | ~2s |
| 50.000 | 24 | 4 (VAR) | ~15s |

!!! tip "Otimizacao"

    Para simulacoes grandes, o forecastbox usa NumPy vectorizado internamente.
    A operacao mais custosa e a propagacao multi-passo em VAR — nesse caso,
    considere reduzir `n_simulations` para prototipar e aumentar para a versao final.

---

## Ver Tambem

- [Previsao Condicional](conditional.md) — combinar cenarios com Monte Carlo
- [Scenario Builder](scenario-builder.md) — construir cenarios para simulacao
- [Fan Charts](fan-charts.md) — visualizar distribuicao preditiva como fan chart
