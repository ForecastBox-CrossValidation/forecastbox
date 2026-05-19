---
title: Comparacao Geral
description: Tabela comparativa unificada de todas as abordagens e recomendacoes por cenario de uso
---

# Benchmark: Comparacao Geral

Visao unificada dos benchmarks e **recomendacoes praticas** por cenario de uso.

---

## Tabela Comparativa Unificada

### Auto-Forecast (M3 mensal, h=12, MASE)

| Metodo | MASE | Rank | Tempo/serie (s) | Complexidade |
|--------|------|------|-----------------|-------------|
| AutoETS | **0.941** | 1 | 0.187 | Baixa |
| AutoARIMA | 0.967 | 2 | 0.342 | Media |
| Theta | 0.958 | 3 | 0.012 | Baixa |
| SeasonalNaive | 0.974 | 4 | 0.001 | Nenhuma |
| Naive | 1.000 | 5 | <0.001 | Nenhuma |

### Combinacao (M3 mensal, h=12, MASE)

| Metodo | MASE | Melhoria vs melhor individual | Tempo (s) | Min. dados |
|--------|------|-------------------------------|-----------|-----------|
| BMA | **0.889** | **5.4%** | 45.3 | 30+ obs. |
| Optimal | 0.896 | 4.7% | 15.6 | 50+ obs. |
| TimeVarying | 0.894 | 4.9% | 67.8 | 60+ obs. |
| Stacking | 0.903 | 3.9% | 34.2 | 50+ obs. |
| OLS | 0.912 | 3.4% | 8.7 | 3K+ obs. |
| InverseMSE | 0.897 | 4.4% | 2.4 | 20+ obs. |
| Simple | 0.908 | 3.5% | 1.2 | Qualquer |

### Nowcasting (PIB BR trimestral, M2, RMSE p.p.)

| Metodo | RMSE (M2) | Reducao vs AR(1) | Tempo (s) | Requisitos |
|--------|-----------|-------------------|-----------|-----------|
| Bridge | **0.94** | **47.2%** | 0.34 | Indicadores agregados |
| U-MIDAS | 0.97 | 45.5% | 0.87 | Dados alta freq. |
| DFM | 0.98 | 44.9% | 2.45 | kalmanbox |
| MIDAS | 0.99 | 44.4% | 1.23 | Dados alta freq. |
| AR(1) | 1.78 | — | 0.02 | Nenhum |

---

## Recomendacoes por Cenario

### Cenario 1: Poucas observacoes (T < 50)

!!! tip "Recomendacao"
    **AutoETS** + **Media Simples**

| Aspecto | Recomendacao | Razao |
|---------|-------------|-------|
| Modelo | AutoETS ou Theta | Poucos parametros, convergem com poucos dados |
| Combinacao | Simple ou InverseMSE | Nao requer estimacao de pesos |
| Avaliacao | AIC/BIC (in-sample) | Poucos dados para out-of-sample confiavel |
| Evitar | OLS, Stacking, Optimal | Overfitting com amostras pequenas |

```python
import forecastbox as fb

# Setup para series curtas
model = fb.AutoETS(information_criterion="bic")
model.fit(y)  # T < 50

# Se combinar, usar media simples
combiner = fb.SimpleCombination()
```

---

### Cenario 2: Muitas observacoes (T > 200)

!!! tip "Recomendacao"
    **AutoARIMA** + **BMA ou Stacking**

| Aspecto | Recomendacao | Razao |
|---------|-------------|-------|
| Modelo | AutoARIMA, AutoETS, Theta | Aproveitar a grade de busca completa |
| Combinacao | BMA, Stacking, Optimal | Dados suficientes para estimar pesos |
| Avaliacao | MCS + DM test | Poder estatistico adequado |
| Extra | TimeVarying se ha mudanca estrutural | Captura instabilidade nos pesos |

---

### Cenario 3: Previsao em tempo real (nowcasting)

!!! tip "Recomendacao"
    **Bridge** para simplicidade, **DFM** para analise completa

| Aspecto | Recomendacao | Razao |
|---------|-------------|-------|
| Metodo | Bridge ou DFM | Melhor RMSE medio |
| Indicadores | 8-15 mensais | IBC-Br, PIM-PF, PMC como core |
| Avaliacao | Pseudo OOS com vintages | Simula disponibilidade real |
| Extra | News decomposition (DFM) | Identifica fontes de revisao |

---

### Cenario 4: Muitas series para prever (>100)

!!! tip "Recomendacao"
    **Theta** + **InverseMSE** com processamento em lote

| Aspecto | Recomendacao | Razao |
|---------|-------------|-------|
| Modelo | Theta (rapido) | 27x mais rapido que AutoARIMA |
| Combinacao | InverseMSE | Simples e robusto |
| Pipeline | Batch com `n_jobs=-1` | Paralelizacao |
| Fallback | Naive se modelo falhar | Garantir previsao para todas |

```python
import forecastbox as fb

pipeline = fb.Pipeline(
    models=["theta", "ets", "seasonal_naive"],
    combination="inverse_mse",
    batch_size=100,
    n_jobs=-1,
    timeout=30,
    fallback="naive"
)
results = pipeline.run_batch(series_dict)
```

---

### Cenario 5: Previsao para publicacao academica

!!! tip "Recomendacao"
    **AutoARIMA** + **BMA** + avaliacao rigorosa

| Aspecto | Recomendacao | Razao |
|---------|-------------|-------|
| Modelos | Multiplos (5+) | Comparar abordagens |
| Combinacao | BMA | Fundamentacao bayesiana |
| Avaliacao | MCS ($\alpha=0.05$) + DM + GW | Testes formais |
| Reproducao | Seed fixa, `n_jobs=1` | Resultados exatos |
| Exportacao | LaTeX via `to_latex()` | Tabelas formatadas |

---

### Cenario 6: Cenarios condicionais (stress test)

!!! tip "Recomendacao"
    **AutoVAR** + **ConditionalForecast**

| Aspecto | Recomendacao | Razao |
|---------|-------------|-------|
| Modelo | VAR ou BVAR | Multiplas variaveis endogenas |
| Cenarios | ConditionalForecast | Restricoes hard/soft |
| Incerteza | Monte Carlo (1000+ sim.) | Distribuicao completa |
| Visualizacao | Fan charts | Comunicacao de incerteza |

---

## Trade-off: Acuracia vs Complexidade vs Tempo

| | Baixa complexidade | Media complexidade | Alta complexidade |
|-|--------------------|--------------------|-------------------|
| **Rapido** | Naive, SeasonalNaive | Theta | — |
| **Moderado** | Simple avg | AutoETS, InverseMSE | Bridge, U-MIDAS |
| **Lento** | — | AutoARIMA, BMA | DFM, Stacking, TimeVarying, Optimal |

!!! info "Onde esta o sweet spot?"
    Para a maioria das aplicacoes praticas, o ponto ideal e **AutoETS + InverseMSE**: boa acuracia (MASE ~0.85), velocidade razoavel (~0.2s/serie), e robustez com poucos dados.

---

## Custo-Beneficio da Sofisticacao

Melhoria incremental ao adicionar camadas de sofisticacao:

| Etapa | Metodo | MASE (h=12) | Melhoria incremental |
|-------|--------|-------------|---------------------|
| 0 | Naive | 1.000 | — |
| 1 | Melhor modelo unico (AutoETS) | 0.941 | +5.9% |
| 2 | Combinacao simples (5 modelos) | 0.908 | +3.5% |
| 3 | Combinacao sofisticada (BMA) | 0.889 | +2.1% |
| 4 | + MCS pre-selecao | 0.881 | +0.9% |
| 5 | + TimeVarying pesos | 0.876 | +0.6% |

!!! note "Retornos decrescentes"
    Os maiores ganhos vêm das primeiras etapas: (1) escolher um bom modelo e (2) combinar. Sofisticacoes adicionais trazem ganhos marginais decrescentes. Avalie se o custo de implementacao e manutencao justifica o ganho.

---

## Melhores Praticas

### 1. Comece simples, sofistique se necessario

```
Naive -> AutoETS -> Combinacao simples -> BMA/Stacking
```

Cada etapa deve demonstrar melhoria out-of-sample antes de prosseguir.

### 2. Diversifique os modelos base

Escolha modelos que capturam diferentes aspectos da serie:

- **Tendencia**: AutoARIMA
- **Sazonalidade**: AutoETS, SeasonalNaive
- **Nivel**: Theta, Naive

### 3. Avalie corretamente

- Sempre out-of-sample (nunca in-sample)
- Use expanding window (nao fixed window)
- Reporte multiplos horizontes
- Teste significancia estatistica (DM test)

### 4. Monitore a performance

```python
import forecastbox as fb

monitor = fb.Monitor(
    pipeline=pipeline,
    alert_threshold=1.5,  # Alertar se RMSE > 1.5x historico
    frequency="monthly"
)
monitor.start()
```

---

## Resumo Executivo

| Pergunta | Resposta |
|----------|---------|
| Melhor modelo unico? | **AutoETS** (media geral) ou **AutoARIMA** (horizonte curto) |
| Melhor combinacao? | **BMA** (consistencia) ou **Simple** (robustez) |
| Melhor nowcasting? | **Bridge** (simplicidade) ou **DFM** (analise completa) |
| Quando combinar? | **Sempre** — melhoria minima de 3.5% na media |
| Quando usar nowcasting? | Quando ha indicadores de alta frequencia disponiveis |
| Maior ganho marginal? | Passar de modelo unico para combinacao (+3.5-5.6%) |
