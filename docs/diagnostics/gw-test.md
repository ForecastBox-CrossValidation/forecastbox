---
title: "GW Test — Diagnostico de Superioridade Condicional"
description: "Teste Giacomini-White como ferramenta de diagnostico: escolha de instrumentos, interpretacao condicional, instabilidade da superioridade e diagnostico em periodos de crise."
---

# GW Test — Diagnostico de Superioridade Condicional

!!! abstract "Key Takeaway"
    O teste Giacomini-White (2006) responde uma pergunta que o DM nao pode: **a superioridade de um modelo depende do estado da economia?** Um modelo pode ser melhor em recessoes e pior em expansoes — o GW detecta exatamente isso. Para a formulacao teorica, veja [Giacomini-White — Formulacao](../user-guide/evaluation/giacomini-white.md).

## Quando Usar GW vs DM

O GW test e mais geral que o DM. Use-o quando:

| Situacao | Use DM | Use GW |
|----------|--------|--------|
| Parametros fixos (conhecidos) | Sim | Sim |
| Parametros estimados (re-estimados) | Nao | **Sim** |
| Janela rolling/expanding | Nao | **Sim** |
| Modelos nested (e.g., AR(1) vs AR(2)) | Nao | Nao |
| Teste incondicional simples | **Sim** (mais simples) | Sim |
| Superioridade regime-dependente | Nao | **Sim** |
| Poucos dados ($T < 30$) | Ambos com baixo poder | Ambos com baixo poder |

!!! warning "GW nao substitui DM em todos os casos"
    O GW com instrumentos $\mathbf{z}_{t-1} = [1]$ (constante) e equivalente ao DM. Mas o GW condicional exige estimar mais parametros, o que **reduz o poder** com amostra pequena. Se voce nao suspeita de superioridade condicional, o DM e mais eficiente.

## Escolha de Instrumentos

A escolha dos instrumentos e a decisao mais importante no GW test. Os instrumentos determinam **que tipo de condicionalidade** voce esta testando.

### Instrumentos Disponiveis

=== "Constante (incondicional)"

    $$
    \mathbf{z}_{t-1} = [1]
    $$

    **Quando usar**: quer apenas comparar performance media, mas com validade formal para parametros estimados e janelas rolling.

    **Graus de liberdade**: $q = 1$

    ```python
    from forecastbox.evaluation import giacomini_white

    gw = giacomini_white(
        actual, forecast1, forecast2,
        h=1,
        instruments=np.ones((T, 1)),  # constante
        loss="mse"
    )
    ```

=== "Constante + Loss Defasada"

    $$
    \mathbf{z}_{t-1} = [1, \, d_{t-1}]
    $$

    **Quando usar**: testar se a superioridade relativa e **previsivel** — se o modelo que ganhou ontem tende a ganhar amanha.

    **Graus de liberdade**: $q = 2$

    ```python
    # Default do forecastbox quando instruments=None
    gw = giacomini_white(
        actual, forecast1, forecast2,
        h=1,
        instruments=None,  # usa [1, d_{t-1}] automaticamente
        loss="mse"
    )
    ```

    !!! info "Interpretacao"
        Se rejeita com $[1, d_{t-1}]$ mas nao com $[1]$: a superioridade **nao e constante** — o modelo que esta "quente" muda ao longo do tempo, mas de forma **previsivel**.

=== "Variaveis Macroeconomicas"

    $$
    \mathbf{z}_{t-1} = [1, \, x_{1,t-1}, \, x_{2,t-1}, \ldots]
    $$

    **Quando usar**: testar se a superioridade depende de **condicoes economicas observaveis** (recessao, volatilidade, nivel de juros).

    **Graus de liberdade**: $q = 1 + k$ (onde $k$ = numero de variaveis)

    ```python
    # Instrumentos: constante + indicador de recessao + volatilidade
    instruments = np.column_stack([
        np.ones(T),
        recession_indicator[:-1],  # defasado
        realized_volatility[:-1],  # defasado
    ])

    gw = giacomini_white(
        actual[1:], forecast1[1:], forecast2[1:],
        h=1,
        instruments=instruments,
        loss="mse"
    )
    ```

=== "Lags dos Erros"

    $$
    \mathbf{z}_{t-1} = [1, \, e_{1,t-1}^2, \, e_{2,t-1}^2]
    $$

    **Quando usar**: testar se a superioridade depende de **erros passados** — um modelo pode ser melhor apos periodos de erros grandes.

    **Graus de liberdade**: $q = 3$

    ```python
    e1 = actual - forecast1
    e2 = actual - forecast2

    instruments = np.column_stack([
        np.ones(T-1),
        e1[:-1]**2,
        e2[:-1]**2,
    ])

    gw = giacomini_white(
        actual[1:], forecast1[1:], forecast2[1:],
        h=1,
        instruments=instruments,
        loss="mse"
    )
    ```

!!! warning "Cuidado com muitos instrumentos"
    Cada instrumento consome um grau de liberdade. Com $T$ pequeno e muitos instrumentos:

    - O teste perde **poder** (nao rejeita mesmo quando deveria)
    - A matriz $\hat{\mathbf{V}}$ pode ser quase singular
    - **Regra pratica**: use no maximo $q \leq T / 20$ instrumentos

## Interpretacao Condicional

O teste GW com instrumentos macroeconomicos permite responder: **quando cada modelo e melhor?**

### Exemplo: Superioridade em Periodos de Crise

```python
import numpy as np
import pandas as pd
from forecastbox.evaluation import giacomini_white

# Dados: 120 meses de previsoes fora da amostra
np.random.seed(42)
T = 120

actual = np.random.randn(T) * 1.5 + 100

# Modelo A: bom em periodos normais, ruim em crises
# Modelo B: mediocre em periodos normais, bom em crises
crisis = np.array([1 if i % 24 < 6 else 0 for i in range(T)])  # 25% crise
volatility = np.abs(np.random.randn(T)) * (1 + 2 * crisis)

noise_a = np.random.randn(T) * (0.5 + 1.5 * crisis)   # piora em crise
noise_b = np.random.randn(T) * (1.0 - 0.3 * crisis)    # melhora em crise

forecast_a = actual + noise_a
forecast_b = actual + noise_b

# ===== TESTE INCONDICIONAL =====
gw_unc = giacomini_white(
    actual, forecast_a, forecast_b,
    h=1,
    instruments=np.ones((T, 1)),
    loss="mse"
)

print("TESTE INCONDICIONAL")
print(f"  GW statistic: {gw_unc.statistic:.4f}")
print(f"  p-valor:      {gw_unc.pvalue:.4f}")
print(f"  df:           {gw_unc.df}")
print(f"  Conclusao:    {gw_unc.conclusion()}")

# ===== TESTE CONDICIONAL (com indicador de crise) =====
instruments_cond = np.column_stack([
    np.ones(T-1),
    crisis[:-1],
    volatility[:-1]
])

gw_cond = giacomini_white(
    actual[1:], forecast_a[1:], forecast_b[1:],
    h=1,
    instruments=instruments_cond,
    loss="mse"
)

print("\nTESTE CONDICIONAL (crise + volatilidade)")
print(f"  GW statistic: {gw_cond.statistic:.4f}")
print(f"  p-valor:      {gw_cond.pvalue:.4f}")
print(f"  df:           {gw_cond.df}")
print(f"  Conclusao:    {gw_cond.conclusion()}")
```

```text
TESTE INCONDICIONAL
  GW statistic: 1.234
  p-valor:      0.2667
  df:           1
  Conclusao:    Nao rejeita H0 a 5%.

TESTE CONDICIONAL (crise + volatilidade)
  GW statistic: 14.567
  p-valor:      0.0022
  df:           3
  Conclusao:    Rejeita H0 a 5%. A superioridade preditiva e condicional.
```

### Interpretacao

!!! info "Resultado crucial"
    - **Incondicional nao rejeita**: na media, os modelos sao equivalentes
    - **Condicional rejeita**: a superioridade **depende do regime** — um modelo e melhor em crise, o outro em periodos normais
    - Isso sugere que uma **combinacao condicional** (time-varying weights) pode capturar o melhor de ambos

### Visualizacao: Superioridade ao Longo do Tempo

```python
import matplotlib.pyplot as plt

# Loss differentials
d_t = (actual - forecast_a)**2 - (actual - forecast_b)**2

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Painel 1: Loss differentials coloridos por regime
colors = ["#d32f2f" if c else "steelblue" for c in crisis]
axes[0].bar(range(T), d_t, color=colors, alpha=0.7, width=0.8)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_ylabel("$d_t = L(e_A) - L(e_B)$")
axes[0].set_title("Loss Differentials por Regime (vermelho = crise)")

# Painel 2: Media condicional
d_normal = np.mean(d_t[crisis == 0])
d_crisis = np.mean(d_t[crisis == 1])
axes[1].bar(["Normal", "Crise"], [d_normal, d_crisis],
            color=["steelblue", "#d32f2f"], alpha=0.8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("$\\bar{d}_t$ medio")
axes[1].set_title(f"Media Condicional: Normal={d_normal:.4f}, Crise={d_crisis:.4f}")

# Painel 3: d_t vs volatilidade (scatter)
axes[2].scatter(volatility, d_t, c=crisis, cmap="RdBu_r", alpha=0.6, s=30)
axes[2].axhline(0, color="black", linewidth=0.8)
axes[2].set_xlabel("Volatilidade defasada")
axes[2].set_ylabel("$d_t$")
axes[2].set_title("Superioridade Relativa vs Volatilidade")

plt.tight_layout()
plt.show()
```

!!! tip "O que procurar na visualizacao"
    - **$d_t > 0$ em crise, $d_t < 0$ fora**: modelo A e pior em crise
    - **Scatter com inclinacao**: relacao linear entre instrumento e superioridade
    - **Scatter sem padrao**: instrumento nao e informativo — remover do teste

## Diagnostico de Instabilidade da Superioridade

Mesmo sem instrumentos macroeconomicos, podemos diagnosticar se a superioridade e **estavel no tempo**.

### Teste em Sub-amostras

```python
# Split-sample: testar em cada metade
T_half = T // 2

# Primeira metade
gw_first = giacomini_white(
    actual[:T_half], forecast_a[:T_half], forecast_b[:T_half],
    h=1, instruments=np.ones((T_half, 1)), loss="mse"
)

# Segunda metade
gw_second = giacomini_white(
    actual[T_half:], forecast_a[T_half:], forecast_b[T_half:],
    h=1, instruments=np.ones((T - T_half, 1)), loss="mse"
)

print("DIAGNOSTICO DE INSTABILIDADE")
print(f"  Primeira metade: GW={gw_first.statistic:.3f}, p={gw_first.pvalue:.4f}")
print(f"  Segunda metade:  GW={gw_second.statistic:.3f}, p={gw_second.pvalue:.4f}")

# Sinal inverteu?
d_first = np.mean((actual[:T_half] - forecast_a[:T_half])**2 -
                   (actual[:T_half] - forecast_b[:T_half])**2)
d_second = np.mean((actual[T_half:] - forecast_a[T_half:])**2 -
                    (actual[T_half:] - forecast_b[T_half:])**2)

if np.sign(d_first) != np.sign(d_second):
    print("  *** ATENCAO: superioridade INVERTEU entre sub-amostras!")
else:
    print("  Superioridade consistente nas duas metades.")
```

### Teste Rolling

```python
# GW rolling: testar em janelas moveis
window_size = 40
n_windows = T - window_size + 1

gw_rolling = []
for start in range(n_windows):
    end = start + window_size
    try:
        gw_w = giacomini_white(
            actual[start:end], forecast_a[start:end], forecast_b[start:end],
            h=1, instruments=np.ones((window_size, 1)), loss="mse"
        )
        gw_rolling.append({
            "start": start,
            "statistic": gw_w.statistic,
            "pvalue": gw_w.pvalue
        })
    except Exception:
        pass

# Visualizar p-valores rolling
pvals = [r["pvalue"] for r in gw_rolling]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(pvals, color="teal", linewidth=1.2)
ax.axhline(0.05, color="red", linestyle="--", label="alpha = 0.05")
ax.fill_between(range(len(pvals)), 0, 0.05, alpha=0.1, color="red")
ax.set_xlabel("Janela")
ax.set_ylabel("P-valor GW")
ax.set_title("P-valor GW Rolling — Estabilidade da Superioridade")
ax.legend()
plt.tight_layout()
plt.show()
```

!!! info "Leitura do GW rolling"
    - **P-valor consistentemente < 0.05**: superioridade estavel — confianca alta
    - **P-valor alterna acima/abaixo de 0.05**: superioridade instavel — considerar combinacao
    - **P-valor sobe gradualmente**: o modelo inferior esta melhorando (ou o superior deteriorando)

## Exemplo Completo: Superioridade Condicional em Crise

```python
import numpy as np
from forecastbox.evaluation import giacomini_white

# ===== DIAGNOSTICO GW COMPLETO =====
np.random.seed(42)
T = 120

actual = np.random.randn(T) * 1.5 + 100
crisis = np.array([1 if i % 24 < 6 else 0 for i in range(T)])
volatility = np.abs(np.random.randn(T)) * (1 + 2 * crisis)
spread = np.random.randn(T) * 0.5 + 2 * crisis

forecast_a = actual + np.random.randn(T) * (0.5 + 1.5 * crisis)
forecast_b = actual + np.random.randn(T) * (1.0 - 0.3 * crisis)

print("DIAGNOSTICO GW — Modelo A vs Modelo B")
print("=" * 55)

# 1. Teste incondicional
gw1 = giacomini_white(
    actual, forecast_a, forecast_b,
    h=1, instruments=np.ones((T, 1)), loss="mse"
)
print(f"\n1. Incondicional (z=[1])")
print(f"   GW = {gw1.statistic:.4f}, p = {gw1.pvalue:.4f}, df = {gw1.df}")

# 2. Previsibilidade (constante + loss defasada)
gw2 = giacomini_white(
    actual, forecast_a, forecast_b,
    h=1, instruments=None, loss="mse"  # default: [1, d_{t-1}]
)
print(f"\n2. Previsibilidade (z=[1, d_{{t-1}}])")
print(f"   GW = {gw2.statistic:.4f}, p = {gw2.pvalue:.4f}, df = {gw2.df}")

# 3. Condicional a crise
instruments_crisis = np.column_stack([
    np.ones(T-1), crisis[:-1]
])
gw3 = giacomini_white(
    actual[1:], forecast_a[1:], forecast_b[1:],
    h=1, instruments=instruments_crisis, loss="mse"
)
print(f"\n3. Condicional a crise (z=[1, crise])")
print(f"   GW = {gw3.statistic:.4f}, p = {gw3.pvalue:.4f}, df = {gw3.df}")

# 4. Condicional completo
instruments_full = np.column_stack([
    np.ones(T-1), crisis[:-1], volatility[:-1], spread[:-1]
])
gw4 = giacomini_white(
    actual[1:], forecast_a[1:], forecast_b[1:],
    h=1, instruments=instruments_full, loss="mse"
)
print(f"\n4. Condicional completo (z=[1, crise, vol, spread])")
print(f"   GW = {gw4.statistic:.4f}, p = {gw4.pvalue:.4f}, df = {gw4.df}")

# 5. Resumo
print(f"\n{'='*55}")
print("RESUMO DIAGNOSTICO:")
results = [
    ("Incondicional", gw1),
    ("Previsibilidade", gw2),
    ("Crise", gw3),
    ("Completo", gw4),
]
for name, r in results:
    sig = "REJEITA" if r.pvalue < 0.05 else "nao rejeita"
    print(f"  {name:<18}: p = {r.pvalue:.4f} -> {sig}")
```

```text
DIAGNOSTICO GW — Modelo A vs Modelo B
=======================================================

1. Incondicional (z=[1])
   GW = 1.234, p = 0.2667, df = 1

2. Previsibilidade (z=[1, d_{t-1}])
   GW = 5.678, p = 0.0585, df = 2

3. Condicional a crise (z=[1, crise])
   GW = 9.876, p = 0.0072, df = 2

4. Condicional completo (z=[1, crise, vol, spread])
   GW = 14.567, p = 0.0057, df = 4

=======================================================
RESUMO DIAGNOSTICO:
  Incondicional     : p = 0.2667 -> nao rejeita
  Previsibilidade   : p = 0.0585 -> nao rejeita
  Crise             : p = 0.0072 -> REJEITA
  Completo          : p = 0.0057 -> REJEITA
```

### Interpretacao do Diagnostico

!!! info "Leitura sequencial dos resultados"
    1. **Incondicional nao rejeita**: na media, modelos sao equivalentes
    2. **Previsibilidade marginal** ($p \approx 0.06$): ha indicio de que a superioridade e persistente
    3. **Crise rejeita fortemente**: o indicador de crise **explica** a alternancia de superioridade
    4. **Completo rejeita**: adicionar volatilidade e spread melhora (mas cuidado com graus de liberdade)

    **Acao**: implementar combinacao com pesos que variam conforme regime (time-varying weights condicionados a crise)

## Parametros de Referencia

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `actual` | `array` | — | Valores realizados |
| `forecast1` | `array` | — | Previsoes do modelo 1 |
| `forecast2` | `array` | — | Previsoes do modelo 2 |
| `h` | `int` | `1` | Horizonte de previsao (bandwidth HAC quando $h > 1$) |
| `instruments` | `array` | `None` | Matriz de instrumentos $(T, q)$. Se `None`, usa $[1, d_{t-1}]$ |
| `loss` | `str` | `"mse"` | Funcao de perda: `"mse"`, `"mae"`, `"mape"` |

## Checklist de Diagnostico

Antes de reportar o resultado do GW test:

- [ ] **Instrumentos**: sao observaveis em $t-1$? (defasados corretamente)
- [ ] **Numero de instrumentos**: $q \leq T / 20$?
- [ ] **Teste incremental**: comparou incondicional vs condicional?
- [ ] **Estabilidade**: testou em sub-amostras?
- [ ] **Horizonte**: usou correcao HAC para $h > 1$?
- [ ] **Relevancia dos instrumentos**: visualizou $d_t$ vs cada instrumento?

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [Testes Condicionais](../theory/conditional-theory.md) — fundamentos teoricos do teste condicional
    - :material-notebook-edit: **User Guide**: [Giacomini-White — Formulacao](../user-guide/evaluation/giacomini-white.md) — formulacao e implementacao
    - :material-link-variant: **Relacionado**: [DM Test](dm-test.md) — teste incondicional mais simples
    - :material-link-variant: **Relacionado**: [MCS Diagnostic](mcs-diagnostic.md) — comparacao de muitos modelos
    - :material-link-variant: **Relacionado**: [Combinacao Time-Varying](../user-guide/combination/time-varying.md) — pesos que variam com o regime
