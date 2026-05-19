---
title: Benchmark Combinacao
description: Comparacao de 7 metodos de combinacao de previsoes no dataset M3 com analise de quando combinar vale a pena
---

# Benchmark: Combinacao de Previsoes

Comparacao sistematica dos 7 metodos de combinacao disponíveis no **forecastbox**.

---

## Setup Experimental

| Parametro | Valor |
|-----------|-------|
| **Dataset** | M3 mensal (1.000 series, seed=42) |
| **Modelos base** | AutoARIMA, AutoETS, Theta, SeasonalNaive, Naive |
| **Split** | 50% treino / 20% validacao / 30% teste |
| **Horizontes** | h = 1, 3, 6, 12 |
| **Metricas** | RMSE, MASE, melhoria sobre melhor individual (%) |
| **Metodos** | Simple, InverseMSE, OLS, BMA, Stacking, TimeVarying, Optimal |

### Protocolo

1. Ajustar 5 modelos base na janela de treino
2. Gerar previsoes out-of-sample na janela de validacao
3. Estimar pesos de combinacao usando previsoes da validacao
4. Avaliar combinacoes na janela de teste

```python
import forecastbox as fb

# Modelos base
base_models = {
    "AutoARIMA": fb.AutoARIMA(),
    "AutoETS": fb.AutoETS(),
    "Theta": fb.ThetaMethod(),
    "SeasonalNaive": fb.SeasonalNaive(seasonal_period=12),
    "Naive": fb.Naive(),
}

# Metodos de combinacao
combiners = {
    "Simple": fb.SimpleCombination(),
    "InverseMSE": fb.InverseMSE(),
    "OLS": fb.OLSCombination(),
    "BMA": fb.BMA(random_state=42),
    "Stacking": fb.StackingCombination(
        meta_learner="ridge", cv=5
    ),
    "TimeVarying": fb.TimeVaryingCombination(
        method="kalman"
    ),
    "Optimal": fb.OptimalCombination(
        method="shrinkage"
    ),
}
```

---

## Resultados: RMSE Medio (1.000 series)

| Metodo | h=1 | h=3 | h=6 | h=12 | Media | Tempo (s) |
|--------|-----|-----|-----|------|-------|-----------|
| **Melhor Individual** | 0.812 | 1.247 | 1.652 | 2.278 | 1.497 | — |
| **Simple** | 0.789 | 1.198 | 1.589 | 2.198 | 1.444 | 1.2 |
| **InverseMSE** | 0.782 | 1.187 | 1.572 | 2.178 | 1.430 | 2.4 |
| **OLS** | **0.768** | 1.179 | 1.561 | 2.201 | 1.427 | 8.7 |
| **BMA** | 0.775 | **1.171** | **1.548** | **2.156** | **1.413** | 45.3 |
| **Stacking** | 0.771 | 1.182 | 1.567 | 2.189 | 1.427 | 34.2 |
| **TimeVarying** | 0.778 | 1.184 | 1.559 | 2.167 | 1.422 | 67.8 |
| **Optimal** | 0.773 | 1.176 | 1.554 | 2.172 | 1.419 | 15.6 |

!!! info "Melhor Individual"
    O "Melhor Individual" e o melhor modelo base **selecionado ex-post** (oracle). Na pratica, nao se sabe qual e o melhor modelo antecipadamente, o que torna a combinacao ainda mais atrativa.

---

## Melhoria sobre o Melhor Individual (%)

| Metodo | h=1 | h=3 | h=6 | h=12 | Media |
|--------|-----|-----|-----|------|-------|
| **Simple** | 2.8% | 3.9% | 3.8% | 3.5% | 3.5% |
| **InverseMSE** | 3.7% | 4.8% | 4.8% | 4.4% | 4.4% |
| **OLS** | **5.4%** | 5.5% | 5.5% | 3.4% | 4.9% |
| **BMA** | 4.6% | **6.1%** | **6.3%** | **5.4%** | **5.6%** |
| **Stacking** | 5.1% | 5.2% | 5.1% | 3.9% | 4.8% |
| **TimeVarying** | 4.2% | 5.1% | 5.6% | 4.9% | 5.0% |
| **Optimal** | 4.8% | 5.7% | 5.9% | 4.7% | 5.3% |

---

## Resultados: MASE Medio

| Metodo | h=1 | h=3 | h=6 | h=12 | Media |
|--------|-----|-----|-----|------|-------|
| **Melhor Individual** | 0.781 | 0.843 | 0.897 | 0.941 | 0.866 |
| **Simple** | 0.756 | 0.812 | 0.862 | 0.908 | 0.835 |
| **InverseMSE** | 0.748 | 0.801 | 0.851 | 0.897 | 0.824 |
| **OLS** | **0.739** | 0.798 | 0.847 | 0.912 | 0.824 |
| **BMA** | 0.744 | **0.789** | **0.838** | **0.889** | **0.815** |
| **Stacking** | 0.741 | 0.796 | 0.845 | 0.903 | 0.821 |
| **TimeVarying** | 0.747 | 0.798 | 0.843 | 0.894 | 0.821 |
| **Optimal** | 0.742 | 0.793 | 0.841 | 0.896 | 0.818 |

---

## Distribuicao dos Pesos (Metodo BMA, h=12)

Pesos medios atribuidos pelo BMA aos modelos base (media sobre 1.000 series):

| Modelo Base | Peso Medio | Desvio-Padrao | Min | Max |
|-------------|-----------|---------------|-----|-----|
| **AutoETS** | 0.312 | 0.187 | 0.01 | 0.89 |
| **AutoARIMA** | 0.287 | 0.174 | 0.01 | 0.85 |
| **Theta** | 0.198 | 0.142 | 0.01 | 0.72 |
| **SeasonalNaive** | 0.134 | 0.112 | 0.01 | 0.64 |
| **Naive** | 0.069 | 0.078 | 0.01 | 0.45 |

!!! note "Heterogeneidade dos pesos"
    O alto desvio-padrao indica que os pesos variam substancialmente entre series. Nao existe um conjunto de pesos "universal" — a combinacao deve ser estimada para cada serie.

---

## Frequencia de Vitoria por Metodo

Percentual de series (de 1.000) onde cada metodo obteve o menor RMSE (h=12):

| Metodo | Vitorias (%) | Top-3 (%) |
|--------|-------------|-----------|
| **BMA** | 24.3% | 62.1% |
| **Optimal** | 18.7% | 58.4% |
| **TimeVarying** | 15.2% | 53.7% |
| **Stacking** | 13.1% | 48.9% |
| **OLS** | 11.4% | 45.2% |
| **InverseMSE** | 9.8% | 41.3% |
| **Simple** | 7.5% | 35.8% |

---

## Quando Combinacao Vale a Pena?

### Analise por Dispersao dos Modelos Base

A melhoria da combinacao depende de quao **diferentes** sao os modelos base:

| Dispersao RMSE entre modelos | Melhoria media da combinacao |
|------------------------------|------------------------------|
| Baixa (CV < 0.05) | 1.2% |
| Media (0.05 <= CV < 0.15) | 4.8% |
| Alta (CV >= 0.15) | 8.3% |

!!! tip "Regra pratica"
    A combinacao e mais valiosa quando os modelos base sao **diversos** (erros pouco correlacionados). Se todos os modelos dao previsoes similares, a media simples ja captura o beneficio.

### Analise por Tamanho da Amostra

| Observacoes (T) | Simple | OLS | BMA | Stacking |
|-----------------|--------|-----|-----|----------|
| T < 60 | **3.1%** | -0.4% | 1.8% | -1.2% |
| 60 <= T < 100 | 3.4% | 4.2% | **5.1%** | 3.7% |
| T >= 100 | 3.8% | **6.7%** | 5.9% | **6.3%** |

!!! warning "Amostras pequenas"
    Com $T < 60$, metodos com muitos parametros (OLS, Stacking) podem ter performance **pior** que a media simples devido a overfitting. Prefira Simple ou InverseMSE nestes casos.

---

## Trade-off Acuracia vs Complexidade

| Metodo | Parametros | Dados minimos | Acuracia | Robustez |
|--------|-----------|---------------|----------|----------|
| **Simple** | 0 | Qualquer | Boa | Excelente |
| **InverseMSE** | 0 | 20+ obs. | Boa | Excelente |
| **OLS** | K | 3K+ obs. | Muito boa | Moderada |
| **BMA** | K+priors | 30+ obs. | Muito boa | Boa |
| **Stacking** | K+meta | 50+ obs. | Muito boa | Moderada |
| **TimeVarying** | 2K | 60+ obs. | Excelente | Moderada |
| **Optimal** | K(K+1)/2 | 50+ obs. | Excelente | Baixa |

Onde $K$ = numero de modelos base.

---

## Reproducao

```python
import forecastbox as fb

bench = fb.benchmarks.CombinationBenchmark(
    dataset="m3_monthly",
    n_sample=1000,
    base_models=["arima", "ets", "theta", "seasonal_naive", "naive"],
    combiners=["simple", "inverse_mse", "ols", "bma",
               "stacking", "time_varying", "optimal"],
    horizons=[1, 3, 6, 12],
    seed=42
)

results = bench.run()
results.summary()
results.plot_improvement()    # Grafico de melhoria sobre individual
results.plot_weights("bma")   # Distribuicao de pesos BMA
```

---

## Conclusoes

1. **Combinacao sempre melhora**: todos os 7 metodos superam o melhor modelo individual na media
2. **BMA** e o metodo mais consistente, com maior melhoria media (5.6%) e maior taxa de vitoria (24.3%)
3. **Media simples** e surpreendentemente competitiva (melhoria de 3.5%) e nunca degrada significativamente
4. **OLS e Stacking** sao superiores com muitos dados ($T \geq 100$), mas arriscados com amostras pequenas
5. **TimeVarying** captura instabilidades e e recomendado quando a importancia relativa dos modelos muda ao longo do tempo
6. A melhoria da combinacao e proporcional a **diversidade** dos modelos base — escolha modelos complementares, nao redundantes
