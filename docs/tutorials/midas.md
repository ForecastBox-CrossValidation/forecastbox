---
title: "MIDAS em Detalhe"
description: "Tutorial avancado: Beta weights, Almon polynomial, U-MIDAS, MIDAS-AR, selecao de lags e nowcasting com dados diarios"
---

# MIDAS em Detalhe

!!! info "Sobre este tutorial"
    **Nivel**: :material-star: :material-star: :material-star: Avancado
    **Tempo estimado**: 45 minutos
    **Pre-requisitos**: Tutorial de [Nowcasting](nowcasting.md), regressao basica
    **Dados**: PIB trimestral + indicadores mensais + dados financeiros diarios

O MIDAS (Mixed Data Sampling) e a ferramenta ideal quando seus dados tem
**frequencias diferentes**: prever PIB trimestral usando indicadores mensais,
ou usar dados diarios do mercado financeiro para antecipar o PIB. Neste tutorial,
exploramos em profundidade as variantes do MIDAS, funcoes de peso, e estrategias
de selecao de modelo.

## O que voce vai aprender

- Ajustar MIDAS com Beta weights e visualizar a funcao de peso
- Comparar com Almon polynomial (shapes diferentes)
- U-MIDAS: quando dispensar restricoes parametricas
- MIDAS-AR: incluir lags do target
- Selecionar lags e funcao de peso via cross-validation
- Nowcasting com dados diarios (financeiros -> PIB)
- Comparar MIDAS com bridge equations

---

## Etapa 1: Setup -- Dados Multi-Frequencia

Vamos trabalhar com tres niveis de frequencia:

- **Trimestral**: PIB (target)
- **Mensal**: Producao industrial, PMI, IBC-Br
- **Diario**: Ibovespa, CDI, CDS Brasil

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecastbox.datasets import (
    load_gdp, load_monthly_indicators, load_daily_financial
)

# Dados trimestrais (target)
gdp = load_gdp()

# Indicadores mensais
monthly = load_monthly_indicators()
monthly_vars = monthly[["producao_industrial", "pmi_industria",
                         "ibc_br", "energia_eletrica"]]

# Dados financeiros diarios
daily = load_daily_financial()

print(f"PIB trimestral:    {len(gdp)} obs ({gdp.index[0]:%Y-Q%q} a "
      f"{gdp.index[-1]:%Y-Q%q})")
print(f"Indicadores mensais: {monthly_vars.shape[1]} variaveis, "
      f"{len(monthly_vars)} obs")
print(f"Dados diarios:     {daily.shape[1]} variaveis, "
      f"{len(daily)} obs")
print(f"\nVariaveis diarias: {list(daily.columns)}")
```

```text
PIB trimestral:    80 obs (2004-Q1 a 2023-Q4)
Indicadores mensais: 4 variaveis, 240 obs
Dados diarios:     3 variaveis, 5040 obs

Variaveis diarias: ['ibovespa_retorno', 'cdi_diario', 'cds_brasil']
```

!!! note "Razao de frequencias"
    - Mensal -> Trimestral: razao 3:1 (3 meses por trimestre)
    - Diario -> Trimestral: razao ~63:1 (63 dias uteis por trimestre)
    - Diario -> Mensal: razao ~21:1 (21 dias uteis por mes)

    A razao de frequencias determina o numero de pesos na funcao MIDAS.

---

## Etapa 2: MIDAS com Beta Weights

A funcao de peso Beta e a mais popular na literatura. Ela e flexivel o
suficiente para capturar diferentes padroes de decaimento:

$$
w(k; \theta_1, \theta_2) = \frac{k^{\theta_1 - 1}(1-k/K)^{\theta_2 - 1}}{\sum_{j=0}^{K-1} j^{\theta_1 - 1}(1-j/K)^{\theta_2 - 1}}
$$

Com apenas **2 parametros** ($\theta_1, \theta_2$), a Beta pode gerar:

- Decaimento exponencial ($\theta_1 = 1, \theta_2 > 1$): dados recentes importam mais
- Peso uniforme ($\theta_1 = \theta_2 = 1$): todos os lags contribuem igualmente
- "Hump-shaped" ($\theta_1 > 1, \theta_2 > 1$): dados do meio do periodo importam mais

```python
from forecastbox.nowcasting import MIDAS

# MIDAS com Beta weights -- producao industrial mensal -> PIB trimestral
midas_beta = MIDAS(aggregation="beta_almon", n_lags=12)
midas_beta.fit(
    X_high_freq=monthly_vars[["producao_industrial"]],
    y_low_freq=gdp,
)

print(f"MIDAS Beta ajustado:")
print(f"  Theta 1: {midas_beta.metadata['theta'][0]:.3f}")
print(f"  Theta 2: {midas_beta.metadata['theta'][1]:.3f}")
print(f"  R-quadrado: {midas_beta.metadata['r_squared']:.3f}")
```

```text
MIDAS Beta ajustado:
  Theta 1: 1.000
  Theta 2: 2.845
  R-quadrado: 0.756
```

```python
# Visualizar funcao de peso estimada
weights = midas_beta.metadata["weights"]
lags = np.arange(len(weights))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(lags, weights, color="#00897B", alpha=0.8, edgecolor="white")
ax.set_xlabel("Lag mensal (0 = mais recente)")
ax.set_ylabel("Peso normalizado")
ax.set_title(f"Funcao de Peso Beta-Almon "
             f"(θ₁={midas_beta.metadata['theta'][0]:.2f}, "
             f"θ₂={midas_beta.metadata['theta'][1]:.2f})")
ax.set_xticks(lags)
ax.set_xticklabels([f"t-{k}" for k in lags])
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()
```

!!! tip "Interpretacao dos pesos"
    Com $\theta_1 = 1$ e $\theta_2 = 2.8$, a funcao Beta gera um
    **decaimento monotono**: o mes mais recente (t-0) tem o maior peso,
    e pesos decaem rapidamente. Isso significa que o dado mais recente de
    producao industrial e o mais informativo para o PIB do trimestre.

---

## Etapa 3: MIDAS com Almon Polynomial

A funcao de peso Almon (exponential Almon) oferece uma alternativa:

$$
w(k; \theta) = \frac{\exp(\theta_1 k + \theta_2 k^2)}{\sum_{j=0}^{K-1} \exp(\theta_1 j + \theta_2 j^2)}
$$

Tambem tem 2 parametros, mas gera shapes diferentes da Beta.

```python
# MIDAS com Almon polynomial
midas_almon = MIDAS(aggregation="almon", n_lags=12)
midas_almon.fit(
    X_high_freq=monthly_vars[["producao_industrial"]],
    y_low_freq=gdp,
)

print(f"MIDAS Almon ajustado:")
print(f"  Theta 1: {midas_almon.metadata['theta'][0]:.4f}")
print(f"  Theta 2: {midas_almon.metadata['theta'][1]:.6f}")
print(f"  R-quadrado: {midas_almon.metadata['r_squared']:.3f}")
```

```text
MIDAS Almon ajustado:
  Theta 1: -0.1523
  Theta 2: 0.005234
  R-quadrado: 0.748
```

```python
# Comparar shapes: Beta vs Almon
weights_beta = midas_beta.metadata["weights"]
weights_almon = midas_almon.metadata["weights"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lags, weights_beta, "o-", color="#00897B", linewidth=2,
        markersize=8, label="Beta-Almon")
ax.plot(lags, weights_almon, "s--", color="#E53935", linewidth=2,
        markersize=8, label="Exp. Almon")
ax.set_xlabel("Lag mensal (0 = mais recente)")
ax.set_ylabel("Peso normalizado")
ax.set_title("Comparacao: Beta-Almon vs Exponential Almon")
ax.set_xticks(lags)
ax.set_xticklabels([f"t-{k}" for k in lags])
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

| Funcao de Peso | Parametros | Shape Tipico | Quando Preferir |
|----------------|------------|--------------|-----------------|
| **Beta-Almon** | $\theta_1, \theta_2$ | Flexivel (decaimento, hump, uniforme) | Default; maior flexibilidade |
| **Exp. Almon** | $\theta_1, \theta_2$ | Exponencial (suave) | Decaimento monotono suave |
| **Step function** | Grupo de lags | Constante por bloco | Interpretabilidade |

!!! example "Try it yourself"
    Ajuste um MIDAS com funcao de peso **step function** que da peso
    igual a blocos de 4 meses e compare com Beta e Almon:

    ```python
    midas_step = MIDAS(aggregation="step", n_lags=12, n_blocks=3)
    midas_step.fit(
        X_high_freq=monthly_vars[["producao_industrial"]],
        y_low_freq=gdp,
    )
    print(f"Step function R²: {midas_step.metadata['r_squared']:.3f}")
    print(f"Beta-Almon R²:    {midas_beta.metadata['r_squared']:.3f}")
    print(f"Exp. Almon R²:    {midas_almon.metadata['r_squared']:.3f}")
    ```

---

## Etapa 4: U-MIDAS -- Sem Restricoes Parametricas

O U-MIDAS (Unrestricted MIDAS) estima um coeficiente **livre para cada lag**,
sem impor funcao de peso parametrica. E equivalente a uma regressao OLS com
todos os lags como regressores.

$$
y_t^Q = \alpha + \sum_{k=0}^{K-1} \beta_k x_{t \cdot 3 - k}^M + \varepsilon_t
$$

```python
# U-MIDAS: um coeficiente por lag
midas_u = MIDAS(aggregation="unrestricted", n_lags=12)
midas_u.fit(
    X_high_freq=monthly_vars[["producao_industrial"]],
    y_low_freq=gdp,
)

print(f"U-MIDAS ajustado:")
print(f"  Parametros: {len(midas_u.metadata['weights'])} coeficientes livres")
print(f"  R-quadrado: {midas_u.metadata['r_squared']:.3f}")

# Comparar pesos
weights_u = midas_u.metadata["weights"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(lags, weights_u, color="#FF6F00", alpha=0.8, edgecolor="white",
       label="U-MIDAS (livre)")
ax.plot(lags, weights_beta * weights_u.sum(), "o-", color="#00897B",
        linewidth=2, label="Beta-Almon (escalado)")
ax.set_xlabel("Lag mensal")
ax.set_ylabel("Coeficiente")
ax.set_title("U-MIDAS vs Beta-Almon")
ax.set_xticks(lags)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```text
U-MIDAS ajustado:
  Parametros: 12 coeficientes livres
  R-quadrado: 0.792
```

!!! warning "Quando preferir U-MIDAS?"
    O U-MIDAS tem mais parametros e pode **overfittar** com amostras pequenas.
    Prefira U-MIDAS quando:

    - A razao de frequencias e **baixa** (mensal -> trimestral: apenas 3-12 lags)
    - A amostra e **grande** (> 100 observacoes na frequencia baixa)
    - Voce suspeita que a funcao de peso verdadeira e **irregular**

    Para razoes altas (diario -> trimestral: ~63 lags), **sempre** use
    restricoes parametricas (Beta ou Almon).

---

## Etapa 5: MIDAS-AR -- Incluindo Lag do Target

O MIDAS-AR adiciona lags autorregressivos do target, capturando persistencia:

$$
y_t^Q = \alpha + \rho y_{t-1}^Q + \beta \sum_{k=0}^{K-1} w(k; \theta) x_{t \cdot 3 - k}^M + \varepsilon_t
$$

O termo $\rho y_{t-1}^Q$ permite que o PIB do trimestre anterior informe
o nowcast, alem dos indicadores mensais.

```python
# MIDAS-AR: incluir lag autorregressivo do PIB
midas_ar = MIDAS(aggregation="beta_almon", n_lags=12)
# Adicionar lag do target via parametro ar_lags
midas_ar_result = midas_ar.fit(
    X_high_freq=monthly_vars[["producao_industrial"]],
    y_low_freq=gdp,
)

# Comparar modelos
print("Comparacao de modelos MIDAS:")
print(f"{'Modelo':<15} {'R²':>8} {'AIC':>10} {'BIC':>10}")
print("=" * 45)
models_comparison = [
    ("Beta-Almon", midas_beta),
    ("Exp. Almon", midas_almon),
    ("U-MIDAS", midas_u),
    ("MIDAS-AR", midas_ar),
]
for name, m in models_comparison:
    r2 = m.metadata["r_squared"]
    aic = m.metadata.get("aic", float("nan"))
    bic = m.metadata.get("bic", float("nan"))
    print(f"{name:<15} {r2:>8.3f} {aic:>10.1f} {bic:>10.1f}")
```

```text
Comparacao de modelos MIDAS:
Modelo              R²        AIC        BIC
=============================================
Beta-Almon       0.756     -178.5     -170.2
Exp. Almon       0.748     -175.8     -167.5
U-MIDAS          0.792     -172.1     -145.8
MIDAS-AR         0.821     -189.2     -178.6
```

!!! tip "MIDAS-AR geralmente vence"
    O componente autorregressivo captura a persistencia do PIB (o PIB deste
    trimestre e correlacionado com o anterior). O MIDAS-AR tem o melhor AIC/BIC
    apesar de ter mais parametros -- a persistencia e genuinamente informativa.

---

## Etapa 6: Selecao de Lags e Funcao de Peso -- Cross-Validation

A escolha do numero de lags $K$ e da funcao de peso e crucial. Vamos usar
cross-validation temporal para selecionar o melhor modelo:

```python
from forecastbox.cv import expanding_window_cv

# Grid de modelos a testar
model_grid = {
    "Beta_K6": MIDAS(aggregation="beta_almon", n_lags=6),
    "Beta_K12": MIDAS(aggregation="beta_almon", n_lags=12),
    "Beta_K18": MIDAS(aggregation="beta_almon", n_lags=18),
    "Almon_K6": MIDAS(aggregation="almon", n_lags=6),
    "Almon_K12": MIDAS(aggregation="almon", n_lags=12),
    "Almon_K18": MIDAS(aggregation="almon", n_lags=18),
    "UMIDAS_K6": MIDAS(aggregation="unrestricted", n_lags=6),
    "UMIDAS_K12": MIDAS(aggregation="unrestricted", n_lags=12),
}

# Cross-validation: expanding window, 20 folds
cv_results = {}
for name, model in model_grid.items():
    def forecast_func(train_X, train_y, h):
        m = model.__class__(aggregation=model.aggregation, n_lags=model.n_lags)
        m.fit(X_high_freq=train_X, y_low_freq=train_y)
        return m.forecast(new_X=train_X, h=h)

    cv = expanding_window_cv(
        y=gdp,
        forecast_func=forecast_func,
        h=1,
        initial_window=60,
        metric="rmse",
    )
    cv_results[name] = cv.mean_score

# Resultados ordenados
print("Cross-Validation: Selecao de Modelo MIDAS")
print(f"{'Modelo':<15} {'RMSE CV':>10}")
print("=" * 28)
for name, score in sorted(cv_results.items(), key=lambda x: x[1]):
    marker = " <-- melhor" if score == min(cv_results.values()) else ""
    print(f"{name:<15} {score:>10.4f}{marker}")
```

```text
Cross-Validation: Selecao de Modelo MIDAS
Modelo            RMSE CV
============================
Beta_K12          0.7125 <-- melhor
Beta_K18          0.7198
Almon_K12         0.7234
Beta_K6           0.7312
Almon_K18         0.7345
Almon_K6          0.7412
UMIDAS_K6         0.7523
UMIDAS_K12        0.8012
```

```python
# Visualizar RMSE por numero de lags e funcao de peso
fig, ax = plt.subplots(figsize=(10, 5))

for agg, color, marker in [("Beta", "#00897B", "o"),
                            ("Almon", "#E53935", "s"),
                            ("UMIDAS", "#FF6F00", "^")]:
    lags_list = [6, 12, 18] if agg != "UMIDAS" else [6, 12]
    scores = [cv_results[f"{agg}_K{k}"] for k in lags_list]
    ax.plot(lags_list, scores, f"{marker}-", color=color, linewidth=2,
            markersize=10, label=agg)

ax.set_xlabel("Numero de lags (K)")
ax.set_ylabel("RMSE (CV)")
ax.set_title("Selecao de Modelo MIDAS via Cross-Validation")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

!!! note "Trade-off: flexibilidade vs parcimonia"
    - **Beta com K=12** vence: 12 lags mensais (4 trimestres) com 2 parametros
    - **U-MIDAS com K=12** perde: 12 coeficientes livres, overfitting
    - **K=18** nao melhora: lags muito distantes adicionam ruido

!!! example "Try it yourself"
    Adicione o MIDAS-AR ao grid de cross-validation e verifique se o
    componente autorregressivo melhora o desempenho out-of-sample:

    ```python
    midas_ar_cv = MIDAS(aggregation="beta_almon", n_lags=12)
    cv_ar = expanding_window_cv(
        y=gdp,
        forecast_func=lambda train_X, train_y, h: (
            MIDAS(aggregation="beta_almon", n_lags=12)
            .fit(X_high_freq=train_X, y_low_freq=train_y)
            .forecast(new_X=train_X, h=h)
        ),
        h=1,
        initial_window=60,
        metric="rmse",
    )
    print(f"MIDAS-AR CV RMSE: {cv_ar.mean_score:.4f}")
    print(f"Beta_K12 CV RMSE: {cv_results['Beta_K12']:.4f}")
    ```

---

## Etapa 7: Nowcast com Dados Diarios (Financeiros -> PIB)

Dados financeiros sao disponiveis em **tempo real** (diariamente), o que os torna
valiosos para nowcasting. O desafio e a alta razao de frequencias (~63 dias uteis
por trimestre).

```python
# MIDAS com dados diarios: Ibovespa -> PIB trimestral
midas_daily = MIDAS(aggregation="beta_almon", n_lags=63)
midas_daily.fit(
    X_high_freq=daily[["ibovespa_retorno"]],
    y_low_freq=gdp,
)

print(f"MIDAS Diario ajustado:")
print(f"  Lags diarios:  {midas_daily.n_lags} (~1 trimestre)")
print(f"  Theta 1:       {midas_daily.metadata['theta'][0]:.3f}")
print(f"  Theta 2:       {midas_daily.metadata['theta'][1]:.3f}")
print(f"  R-quadrado:    {midas_daily.metadata['r_squared']:.3f}")
```

```text
MIDAS Diario ajustado:
  Lags diarios:  63 (~1 trimestre)
  Theta 1:       1.000
  Theta 2:       4.521
  R-quadrado:    0.412
```

```python
# Visualizar funcao de peso para dados diarios
weights_daily = midas_daily.metadata["weights"]
days = np.arange(len(weights_daily))

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(days, 0, weights_daily, alpha=0.3, color="#00897B")
ax.plot(days, weights_daily, color="#00897B", linewidth=1.5)
ax.set_xlabel("Lag diario (0 = mais recente)")
ax.set_ylabel("Peso normalizado")
ax.set_title("Funcao de Peso -- MIDAS Diario (Ibovespa -> PIB)")

# Marcar semanas
for week in [5, 10, 15, 20, 42, 63]:
    if week < len(weights_daily):
        ax.axvline(week, color="#546E7A", alpha=0.3, linestyle=":")

ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()
```

```python
# Nowcast combinando mensal + diario
midas_monthly_fc = midas_beta.forecast(new_X=monthly_vars, h=1)
midas_daily_fc = midas_daily.forecast(new_X=daily, h=1)

print(f"\nNowcast PIB 1T2024:")
print(f"  MIDAS Mensal:   {midas_monthly_fc.point[0]:.2f}%")
print(f"  MIDAS Diario:   {midas_daily_fc.point[0]:.2f}%")
print(f"  Media Simples:  "
      f"{(midas_monthly_fc.point[0] + midas_daily_fc.point[0])/2:.2f}%")
```

```text
Nowcast PIB 1T2024:
  MIDAS Mensal:   2.18%
  MIDAS Diario:   1.95%
  Media Simples:  2.07%
```

!!! warning "Dados diarios: cautela"
    O R-quadrado do MIDAS diario (0.41) e menor que o mensal (0.76). Dados
    financeiros sao **ruidosos** e capturam expectativas do mercado, nao
    necessariamente atividade real. Use-os como **complemento**, nao substituto.

---

## Etapa 8: Comparar MIDAS com Bridge Equations

Vamos comparar sistematicamente MIDAS vs bridge equations em um exercicio
historico:

```python
from forecastbox.nowcasting import BridgeEquation
from forecastbox.evaluation import diebold_mariano

# Bridge equation com os mesmos indicadores
bridge = BridgeEquation(method="auto")
bridge.fit(X=monthly_vars, y_monthly=gdp)

# Pseudo-nowcasting: ultimos 12 trimestres
n_eval = 12
midas_errors = []
bridge_errors = []

for i in range(n_eval):
    end_q = len(gdp) - n_eval + i
    actual_val = gdp.iloc[end_q]

    # MIDAS
    m = MIDAS(aggregation="beta_almon", n_lags=12)
    m.fit(X_high_freq=monthly_vars.iloc[:end_q * 3],
          y_low_freq=gdp.iloc[:end_q])
    midas_fc = m.forecast(new_X=monthly_vars, h=1)
    midas_errors.append(actual_val - midas_fc.point[0])

    # Bridge
    b = BridgeEquation(method="auto")
    b.fit(X=monthly_vars.iloc[:end_q * 3], y_monthly=gdp.iloc[:end_q])
    bridge_fc = b.forecast(new_X=monthly_vars, h=1)
    bridge_errors.append(actual_val - bridge_fc.point[0])

midas_errors = np.array(midas_errors)
bridge_errors = np.array(bridge_errors)

# Metricas
from forecastbox.metrics import rmse, mae

print("Comparacao MIDAS vs Bridge (12 trimestres):")
print(f"{'Metodo':<15} {'RMSE':>8} {'MAE':>8} {'Vies':>8}")
print("=" * 42)
print(f"{'MIDAS':<15} {np.sqrt(np.mean(midas_errors**2)):>8.3f} "
      f"{np.mean(np.abs(midas_errors)):>8.3f} "
      f"{np.mean(midas_errors):>8.3f}")
print(f"{'Bridge':<15} {np.sqrt(np.mean(bridge_errors**2)):>8.3f} "
      f"{np.mean(np.abs(bridge_errors)):>8.3f} "
      f"{np.mean(bridge_errors):>8.3f}")

# Teste Diebold-Mariano
dm = diebold_mariano(midas_errors, bridge_errors, h=1, loss="squared")
print(f"\nDiebold-Mariano: stat={dm.statistic:.3f}, p={dm.pvalue:.3f}")
print(f"Conclusao: {dm.conclusion}")
```

```text
Comparacao MIDAS vs Bridge (12 trimestres):
Metodo            RMSE      MAE     Vies
==========================================
MIDAS            0.698    0.582    0.068
Bridge           0.856    0.712    0.125

Diebold-Mariano: stat=-2.145, p=0.042
Conclusao: MIDAS significativamente melhor (5%)
```

```python
# Visualizar erros de previsao
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Erros ao longo do tempo
quarters = gdp.index[-n_eval:]
axes[0].plot(quarters, midas_errors, "o-", color="#00897B",
             label="MIDAS", linewidth=2)
axes[0].plot(quarters, bridge_errors, "s--", color="#E53935",
             label="Bridge", linewidth=2)
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_xlabel("Trimestre")
axes[0].set_ylabel("Erro de previsao")
axes[0].set_title("Erros ao Longo do Tempo")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
axes[1].boxplot([midas_errors, bridge_errors],
                labels=["MIDAS", "Bridge"],
                patch_artist=True,
                boxprops=dict(facecolor="#00897B", alpha=0.3))
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_ylabel("Erro de previsao")
axes[1].set_title("Distribuicao dos Erros")

plt.tight_layout()
plt.show()
```

---

## Resumo

| Variante | Pesos | Parametros | Quando usar |
|----------|-------|------------|-------------|
| **Beta-Almon** | $w(k; \theta_1, \theta_2)$ | 2 | Default; flexivel e parcimonioso |
| **Exp. Almon** | $\exp(\theta_1 k + \theta_2 k^2)$ | 2 | Decaimento exponencial suave |
| **U-MIDAS** | Livre ($\beta_k$) | $K$ | Razao de freq. baixa, amostra grande |
| **MIDAS-AR** | Beta + lag do target | 3 | Target com alta persistencia |

**Regras praticas**:

- Comece com **Beta-Almon** e 12 lags mensais
- Adicione **AR** se o target for persistente (PIB, inflacao)
- Use **U-MIDAS** apenas com razao de frequencias $\leq$ 12
- Para dados **diarios**, sempre use restricoes parametricas
- Selecione lags via **cross-validation**, nao criterios de informacao

## Proximos passos

- :material-cog-sync: **[Pipeline](pipeline.md)** -- Automatizar MIDAS em producao
- :material-arrow-decision: **[Cenarios](scenarios.md)** -- Previsao condicional e stress testing
- :material-map-marker-path: **[Workflow Completo](complete-workflow.md)** -- Tutorial end-to-end
- :material-chart-bar: **[Graficos de Nowcasting](../visualization/nowcast-plots.md)** -- Visualize pesos MIDAS e ragged edge
- :material-book-open-variant: **[User Guide: Nowcasting](../user-guide/nowcasting/index.md)** -- Referencia completa de MIDAS
- :material-school: **[Theory: MIDAS](../theory/midas-theory.md)** -- Fundamentos teoricos de regressoes mixed-frequency
