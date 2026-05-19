---
title: "Estabilidade de Pesos de Combinacao"
description: "Diagnostico de estabilidade temporal dos pesos de combinacao: rolling, recursive, bootstrap e testes de quebra estrutural com metricas e visualizacoes."
---

# Estabilidade de Pesos de Combinacao

!!! abstract "Key Takeaway"
    Pesos de combinacao estimados podem ser **instaveis ao longo do tempo**, tornando a combinacao pouco confiavel. Diagnosticar estabilidade antes de usar uma combinacao em producao e essencial — pesos que mudam drasticamente indicam que a relacao entre modelos e o alvo esta se alterando, e a combinacao pode nao generalizar para o futuro.

## Motivacao

Metodos de combinacao como [OLS](../user-guide/combination/ols.md), [BMA](../user-guide/combination/bma.md) e [Stacking](../user-guide/combination/stacking.md) estimam pesos otimos para um periodo amostral. Mas esses pesos sao estimados com **erro** e podem ser **nao-estacionarios**:

- **Mudancas estruturais** na economia alteram a relevancia relativa dos modelos
- **Overfitting amostral** gera pesos que nao se sustentam fora da amostra
- **Multicolinearidade** entre previsoes amplifica a instabilidade

!!! warning "Pesos instaveis = combinacao pouco confiavel"
    Se os pesos de combinacao flutuam drasticamente ao longo do tempo, o ganho da combinacao otima pode ser ilusorio. Neste caso, considere a [media simples](../user-guide/combination/simple.md), que nao sofre de instabilidade de estimacao e frequentemente supera metodos otimos fora da amostra.

## Metodos de Diagnostico

O forecastbox oferece quatro abordagens complementares para avaliar estabilidade:

### 1. Rolling Weights

Estima os pesos em **janela movel** de tamanho fixo $w$ e observa como evoluem ao longo do tempo.

$$
\hat{\mathbf{w}}_t = \arg\min_{\mathbf{w}} \sum_{s=t-w+1}^{t} L\left(y_s - \sum_{i=1}^{K} w_i \hat{y}_{i,s}\right), \quad t = w, w+1, \ldots, T
$$

=== "Codigo"

    ```python
    from forecastbox.diagnostics import weight_stability

    # actual: array (T,), forecasts: array (T, K) com K modelos
    stab = weight_stability(
        actual,
        forecasts,
        method="rolling",
        window=60,
        model_names=["ARIMA", "ETS", "VAR", "Theta", "Naive"],
    )

    print(stab.summary())
    ```

=== "Output"

    ```text
    ====================================================================
                   Weight Stability Diagnostic (Rolling)
    ====================================================================
    Window: 60 | Models: 5 | Periods: 120

    Model      Mean Weight   Std Weight   Min       Max       Range
    ─────────────────────────────────────────────────────────────────
    ARIMA      0.312         0.078        0.182     0.471     0.289
    ETS        0.241         0.042        0.158     0.334     0.176
    VAR        0.198         0.103        0.021     0.412     0.391
    Theta      0.157         0.038        0.089     0.243     0.154
    Naive      0.092         0.025        0.041     0.148     0.107

    Stability Metrics:
      Avg weight variance:      0.0042
      Max deviation from mean:  0.214  (VAR)
      Herfindahl index (mean):  0.228
      Herfindahl index (std):   0.031

    Verdict: MODERATE instability detected (VAR weight highly variable)
    ====================================================================
    ```

!!! tip "Escolha da janela"
    Use $w$ entre 30 e 120 observacoes. Janelas curtas capturam mudancas rapidas mas sao mais ruidosas; janelas longas sao mais suaves mas reagem devagar a quebras estruturais.

### 2. Recursive Weights

Estima os pesos com **amostra crescente** (expanding window), comecando com $w_0$ observacoes e adicionando uma a cada passo.

$$
\hat{\mathbf{w}}_t = \arg\min_{\mathbf{w}} \sum_{s=1}^{t} L\left(y_s - \sum_{i=1}^{K} w_i \hat{y}_{i,s}\right), \quad t = w_0, w_0+1, \ldots, T
$$

A diferenca para rolling e que os pesos recursivos **convergem** se o processo gerador e estavel. Se nao convergem, ha evidencia de instabilidade.

```python
stab_rec = weight_stability(
    actual,
    forecasts,
    method="recursive",
    min_window=40,
    model_names=["ARIMA", "ETS", "VAR", "Theta", "Naive"],
)

# Verificar convergencia
print(f"Convergencia: {stab_rec.convergence_test()}")
```

```text
Convergencia: PASS - pesos convergem (variancia dos ultimos 20% < 10% da total)
```

### 3. Bootstrap Stability

Avalia a **incerteza amostral** dos pesos via bootstrap nao-parametrico. Gera $B$ reamostras dos dados, estima pesos em cada uma, e calcula intervalos de confianca.

$$
\hat{w}_i^{(b)} = \text{peso do modelo } i \text{ na reamostra } b, \quad b = 1, \ldots, B
$$

$$
\text{IC}_{95\%}(w_i) = \left[\hat{w}_i^{(0.025)}, \hat{w}_i^{(0.975)}\right]
$$

```python
stab_boot = weight_stability(
    actual,
    forecasts,
    method="bootstrap",
    n_boot=1000,
    confidence=0.95,
    model_names=["ARIMA", "ETS", "VAR", "Theta", "Naive"],
)

# Intervalos de confianca dos pesos
for name, ci in zip(stab_boot.model_names, stab_boot.confidence_intervals):
    print(f"{name:8s}: peso = {ci['mean']:.3f}  IC95% = [{ci['lower']:.3f}, {ci['upper']:.3f}]")
```

```text
ARIMA   : peso = 0.310  IC95% = [0.195, 0.428]
ETS     : peso = 0.243  IC95% = [0.168, 0.321]
VAR     : peso = 0.195  IC95% = [0.042, 0.358]
Theta   : peso = 0.160  IC95% = [0.091, 0.234]
Naive   : peso = 0.092  IC95% = [0.038, 0.152]
```

!!! info "Interpretacao dos intervalos"
    - **IC estreito**: peso estimado com precisao — modelo tem contribuicao estavel
    - **IC largo**: peso incerto — modelo pode ou nao contribuir
    - **IC inclui zero**: modelo pode ser irrelevante na combinacao

### 4. Structural Break Test

Testa formalmente se os pesos sofreram **quebra estrutural** usando CUSUM e Bai-Perron.

**CUSUM**: monitora a soma acumulada dos residuos recursivos. Sob estabilidade, o CUSUM oscila dentro de bandas de significancia.

$$
\text{CUSUM}_t = \sum_{s=w_0+1}^{t} \frac{\hat{u}_s}{\hat{\sigma}_u}, \quad t = w_0+1, \ldots, T
$$

**Bai-Perron**: identifica datas de quebra otimas que minimizam a soma de residuos quadrados.

```python
stab_break = weight_stability(
    actual,
    forecasts,
    method="structural_break",
    test="both",  # CUSUM + Bai-Perron
    significance=0.05,
    model_names=["ARIMA", "ETS", "VAR", "Theta", "Naive"],
)

print(stab_break.summary())
```

```text
====================================================================
              Structural Break Test for Combination Weights
====================================================================
CUSUM Test:
  Statistic: 1.847
  Critical value (5%): 1.358
  Result: REJECT stability — evidence of parameter change

Bai-Perron Test:
  Number of breaks detected: 1
  Break date: 2023-06-15 (observation 84)
  SupF statistic: 14.23 (p-value: 0.0031)

Conclusion: Weights are NOT stable. Consider:
  1. Time-varying combination (DMA/DMS)
  2. Split sample at break point
  3. Simple average (robust to instability)
====================================================================
```

## Metricas de Estabilidade

### Variancia dos Pesos

A variancia media dos pesos ao longo do tempo mede a dispersao geral:

$$
\bar{V} = \frac{1}{K} \sum_{i=1}^{K} \text{Var}(\hat{w}_{i,t})
$$

Valores altos indicam que os pesos flutuam significativamente.

### Maximo Desvio da Media

Identifica o modelo cujo peso mais se afasta da media temporal:

$$
\Delta_{\max} = \max_{i,t} |\hat{w}_{i,t} - \bar{w}_i|
$$

### Indice de Herfindahl

O indice de Herfindahl mede a **concentracao** dos pesos:

$$
H_t = \sum_{i=1}^{K} w_{i,t}^2
$$

| $H_t$ | Interpretacao |
|--------|---------------|
| $1/K$ | Pesos perfeitamente iguais (diversificacao maxima) |
| $\approx 0.3$ | Concentracao moderada |
| $\approx 0.5$ | Alta concentracao — poucos modelos dominam |
| $\to 1$ | Um unico modelo domina (combinacao degenerada) |

!!! info "Herfindahl ao longo do tempo"
    Se $H_t$ aumenta ao longo do tempo, a combinacao esta se tornando mais concentrada — possivelmente um unico modelo esta dominando e a combinacao esta convergindo para selecao.

```python
# Calcular metricas de estabilidade
metrics = stab.stability_metrics()

print(f"Variancia media dos pesos:   {metrics['avg_variance']:.4f}")
print(f"Maximo desvio da media:      {metrics['max_deviation']:.4f}")
print(f"Herfindahl medio:            {metrics['herfindahl_mean']:.4f}")
print(f"Herfindahl desvio padrao:    {metrics['herfindahl_std']:.4f}")
```

```text
Variancia media dos pesos:   0.0042
Maximo desvio da media:      0.2140
Herfindahl medio:            0.2280
Herfindahl desvio padrao:    0.0310
```

## Visualizacoes

### Grafico de Area Empilhada

Mostra a evolucao dos pesos ao longo do tempo. Pesos estaveis geram bandas horizontais; pesos instaveis geram ondulacoes.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 6))
stab.plot_stacked_area(ax=ax)
ax.set_title("Evolucao dos Pesos de Combinacao (Rolling, w=60)")
ax.set_xlabel("Periodo")
ax.set_ylabel("Peso")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
```

### Boxplot de Distribuicao dos Pesos

Visualiza a dispersao de cada peso ao longo do tempo. Modelos com caixas largas sao instaveis.

```python
fig, ax = plt.subplots(figsize=(10, 6))
stab.plot_weight_boxplot(ax=ax)
ax.set_title("Distribuicao dos Pesos ao Longo do Tempo")
ax.set_ylabel("Peso")
plt.tight_layout()
plt.show()
```

### Heatmap de Correlacao entre Pesos

Mostra como os pesos co-movem. Correlacao negativa forte indica que os modelos competem pela mesma fatia — quando um ganha peso, outro perde.

```python
fig, ax = plt.subplots(figsize=(8, 7))
stab.plot_weight_correlation(ax=ax, cmap="RdBu_r", annot=True)
ax.set_title("Correlacao entre Pesos de Combinacao")
plt.tight_layout()
plt.show()
```

!!! tip "Interpretando a correlacao"
    - **Correlacao negativa forte** entre dois modelos: competem pela mesma informacao — considere remover um
    - **Correlacao positiva forte**: modelos se reforçam — a combinacao capta um padrao comum
    - **Correlacao proxima de zero**: modelos captam informacoes independentes — combinacao ideal

## Exemplo Completo: Estabilidade de BMA Weights

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecastbox.combination import BMA
from forecastbox.diagnostics import weight_stability

# Dados: 180 observacoes mensais, 5 modelos
np.random.seed(42)
T = 180
K = 5
model_names = ["ARIMA", "ETS", "VAR", "Theta", "Naive"]

# Simular previsoes e realizado
actual = np.cumsum(np.random.randn(T) * 0.3) + 100
forecasts = np.column_stack([
    actual + np.random.randn(T) * 0.5 - 0.02,   # ARIMA
    actual + np.random.randn(T) * 0.6 + 0.01,    # ETS
    actual + np.random.randn(T) * 0.8,            # VAR
    actual + np.random.randn(T) * 0.55 + 0.03,   # Theta
    actual + np.random.randn(T) * 1.0 + 0.1,     # Naive
])

# ===== DIAGNOSTICO COMPLETO =====

# 1. Rolling weights
stab_rolling = weight_stability(
    actual, forecasts,
    method="rolling",
    window=60,
    model_names=model_names,
)

# 2. Bootstrap stability
stab_boot = weight_stability(
    actual, forecasts,
    method="bootstrap",
    n_boot=1000,
    confidence=0.95,
    model_names=model_names,
)

# 3. Structural break test
stab_break = weight_stability(
    actual, forecasts,
    method="structural_break",
    test="both",
    model_names=model_names,
)

# ===== VISUALIZACAO COMPLETA =====

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Area empilhada
stab_rolling.plot_stacked_area(ax=axes[0, 0])
axes[0, 0].set_title("Evolucao dos Pesos (Rolling, w=60)")

# Boxplot
stab_rolling.plot_weight_boxplot(ax=axes[0, 1])
axes[0, 1].set_title("Distribuicao dos Pesos")

# Herfindahl ao longo do tempo
h_t = stab_rolling.herfindahl_series()
axes[1, 0].plot(h_t, color="teal", linewidth=1.5)
axes[1, 0].axhline(1/K, color="green", linestyle="--", label=f"Diversificacao maxima (1/K = {1/K:.2f})")
axes[1, 0].axhline(0.5, color="red", linestyle="--", label="Alta concentracao")
axes[1, 0].set_title("Indice de Herfindahl ao Longo do Tempo")
axes[1, 0].set_ylabel("$H_t = \\sum w_i^2$")
axes[1, 0].legend()

# Correlacao entre pesos
stab_rolling.plot_weight_correlation(ax=axes[1, 1], cmap="RdBu_r", annot=True)
axes[1, 1].set_title("Correlacao entre Pesos")

plt.suptitle("Diagnostico de Estabilidade: BMA com 5 Modelos", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ===== RESUMO =====
print(stab_rolling.summary())
print()
print(stab_break.summary())
```

### Interpretacao

!!! info "Leitura dos resultados"
    1. **Area empilhada**: se as bandas oscilam pouco, pesos sao estaveis. Oscilacoes grandes indicam instabilidade
    2. **Boxplot**: modelos com caixas estreitas tem pesos estaveis; caixas largas indicam instabilidade
    3. **Herfindahl**: valores proximos a $1/K$ indicam diversificacao; valores crescentes indicam concentracao progressiva
    4. **Correlacao**: correlacao negativa forte entre pesos sugere redundancia entre modelos

## Recomendacoes Praticas

!!! tip "Fluxo de decisao"
    Com base no diagnostico de estabilidade:

    1. **Pesos estaveis** ($\bar{V} < 0.01$, sem quebra): use a combinacao otima com confianca
    2. **Instabilidade moderada** ($0.01 \leq \bar{V} < 0.05$): considere [Time-Varying](../user-guide/combination/time-varying.md) weights
    3. **Alta instabilidade** ($\bar{V} \geq 0.05$ ou quebra detectada): use [media simples](../user-guide/combination/simple.md) ou [Stacking](../user-guide/combination/stacking.md) com regularizacao
    4. **Concentracao crescente** ($H_t \to 1$): a combinacao esta degenerando para selecao — reduza o numero de modelos

!!! warning "Armadilha: pesos instaveis nao invalidam a combinacao"
    Pesos instaveis significam que a combinacao **otima estimada** e pouco confiavel, nao que combinar e inutil. A media simples (pesos iguais) e uma combinacao que **nao sofre** de instabilidade de estimacao e frequentemente supera metodos otimos fora da amostra.

## Parametros de Referencia

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `actual` | `array` | — | Valores realizados |
| `forecasts` | `array` | — | Matriz de previsoes $(T \times K)$ |
| `method` | `str` | `"rolling"` | Metodo: `"rolling"`, `"recursive"`, `"bootstrap"`, `"structural_break"` |
| `window` | `int` | `60` | Tamanho da janela movel (rolling) |
| `min_window` | `int` | `40` | Janela minima (recursive) |
| `n_boot` | `int` | `1000` | Numero de reamostras bootstrap |
| `confidence` | `float` | `0.95` | Nivel de confianca para intervalos |
| `test` | `str` | `"both"` | Teste de quebra: `"cusum"`, `"bai_perron"`, `"both"` |
| `significance` | `float` | `0.05` | Nivel de significancia para testes |
| `model_names` | `list` | `None` | Nomes dos modelos (para labels) |

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [Combinacao de Previsoes](../theory/combination-theory.md) — fundamentos teoricos da combinacao
    - :material-notebook-edit: **User Guide**: [Escolhendo o Metodo](../user-guide/combination/choosing.md) — guia pratico para selecao de metodo de combinacao
    - :material-link-variant: **Relacionado**: [Encompassing Test](encompassing-test.md) — diagnosticar se modelos agregam informacao
    - :material-link-variant: **Relacionado**: [Combinacao — Media Simples](../user-guide/combination/simple.md) — alternativa robusta a instabilidade
    - :material-link-variant: **Relacionado**: [Combinacao — Time-Varying](../user-guide/combination/time-varying.md) — pesos adaptativos ao longo do tempo
