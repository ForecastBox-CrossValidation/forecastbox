---
title: Benchmarks
description: Metodologia, datasets e reproducibilidade dos benchmarks de performance do forecastbox
---

# Benchmarks

Comparacoes sistematicas de performance entre os metodos disponiveis no **forecastbox**. Todos os resultados sao reproduziveis e baseados em datasets publicos.

---

## Metodologia

### Principios

Os benchmarks seguem principios de **avaliacao justa** de previsoes:

1. **Out-of-sample**: todas as metricas sao calculadas em dados nao vistos durante o ajuste
2. **Expanding window**: janela de treinamento expande a cada passo (simula uso real)
3. **Sem look-ahead bias**: nenhuma informacao futura e utilizada
4. **Multiplos horizontes**: avaliacao em horizontes de 1, 3, 6 e 12 passos
5. **Reproducibilidade**: seed fixa e codigo disponivel

### Protocolo de Avaliacao

Para cada serie temporal $y_1, \ldots, y_T$:

1. Definir ponto de corte inicial $t_0$ (70% dos dados)
2. Para $t = t_0, \ldots, T-h$:
    - Ajustar modelo com $y_1, \ldots, y_t$
    - Gerar previsao $\hat{y}_{t+1}, \ldots, \hat{y}_{t+h}$
    - Registrar erros $e_{t+j} = y_{t+j} - \hat{y}_{t+j}$ para $j = 1, \ldots, h$
3. Calcular metricas agregadas

---

## Datasets

### M3 Competition (mensal)

| Propriedade | Valor |
|-------------|-------|
| **Fonte** | Makridakis & Hibon (2000) |
| **Series** | 1.428 series mensais (amostra: 1.000) |
| **Comprimento** | 48-126 observacoes |
| **Categorias** | Micro, Industry, Macro, Finance, Demographic, Other |
| **Horizonte** | 18 meses |
| **Uso** | Benchmarks de auto-forecast e combinacao |

```python
import forecastbox as fb

# Carregar dataset M3
m3 = fb.datasets.load_m3(subset="monthly", n_sample=1000, seed=42)
print(f"Series: {len(m3)}")
print(f"Comprimento medio: {np.mean([len(s) for s in m3.values()]):.0f}")
```

### PIB Brasileiro Trimestral

| Propriedade | Valor |
|-------------|-------|
| **Fonte** | IBGE / BCB (via SGS) |
| **Periodo** | 1996Q1 - 2024Q4 |
| **Frequencia** | Trimestral |
| **Variaveis** | PIB real dessazonalizado + 15 indicadores mensais |
| **Uso** | Benchmarks de nowcasting |

Os 15 indicadores mensais incluem:

| Indicador | Fonte | Defasagem |
|-----------|-------|-----------|
| PIM-PF (Producao Industrial) | IBGE | ~45 dias |
| PMC (Comercio Varejista) | IBGE | ~60 dias |
| PMS (Servicos) | IBGE | ~60 dias |
| CAGED (Emprego Formal) | MTE | ~30 dias |
| Confianca do Consumidor | FGV | ~5 dias |
| Confianca da Industria | FGV | ~5 dias |
| IBC-Br (Indice de Atividade) | BCB | ~45 dias |
| Exportacoes (USD) | MDIC | ~15 dias |
| Importacoes (USD) | MDIC | ~15 dias |
| Taxa de Cambio (media) | BCB | Tempo real |
| IPCA (Inflacao) | IBGE | ~10 dias |
| Selic (meta) | BCB | Tempo real |
| M1 (Base Monetaria) | BCB | ~30 dias |
| Receita Tributaria Federal | RFB | ~30 dias |
| Energia Eletrica (consumo) | ONS | ~15 dias |

```python
# Carregar dataset macro BR
macro = fb.datasets.load_macro_br()
y_pib = macro["pib"]
X_indicators = macro["indicators"]
```

---

## Metricas

| Metrica | Formula | Interpretacao |
|---------|---------|---------------|
| **RMSE** | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}$ | Erro medio quadratico (mesma unidade da serie) |
| **MAE** | $\frac{1}{N}\sum_{i=1}^{N}\|e_i\|$ | Erro medio absoluto |
| **MASE** | $\frac{MAE}{MAE_{Naive}}$ | Erro relativo ao Naive (< 1 = melhor que Naive) |
| **RMSE Relativo** | $\frac{RMSE_{modelo}}{RMSE_{Naive}}$ | Performance relativa (< 1 = melhor que Naive) |
| **Tempo (s)** | Wall-clock time | Tempo total de ajuste + previsao |

!!! info "MASE como metrica primaria"
    O MASE (Mean Absolute Scaled Error) e a metrica recomendada para comparacoes entre series de diferentes escalas, pois normaliza pelo erro do Naive sazonal.

---

## Hardware

Todos os benchmarks foram executados no seguinte ambiente:

| Componente | Especificacao |
|------------|---------------|
| **CPU** | Intel Core i7-12700H (14 cores, 20 threads) |
| **RAM** | 32 GB DDR5 |
| **OS** | Ubuntu 22.04 LTS (WSL2) |
| **Python** | 3.11.7 |
| **forecastbox** | 0.1.0 |
| **NumPy** | 1.26.4 |
| **SciPy** | 1.12.0 |
| **statsmodels** | 0.14.1 |

!!! note "Reproducibilidade de tempo"
    Os tempos de execucao podem variar entre maquinas. Use os tempos relativos entre metodos como referencia, nao os valores absolutos.

---

## Como Reproduzir

Todos os benchmarks podem ser reproduzidos com o script incluso:

```bash
# Instalar dependencias
pip install forecastbox[kalman,bench]

# Rodar todos os benchmarks
python -m forecastbox.benchmarks.run_all --seed 42 --output results/

# Rodar benchmark especifico
python -m forecastbox.benchmarks.auto_forecast --seed 42
python -m forecastbox.benchmarks.combination --seed 42
python -m forecastbox.benchmarks.nowcasting --seed 42
```

Ou programaticamente:

```python
import forecastbox as fb

# Rodar benchmark de auto-forecast
bench = fb.benchmarks.AutoForecastBenchmark(
    dataset="m3_monthly",
    n_sample=1000,
    seed=42
)
results = bench.run()
results.summary()
results.to_excel("benchmark_auto_forecast.xlsx")
```

---

## Secoes

| Benchmark | Descricao |
|-----------|-----------|
| [Auto-Forecast](auto-forecast.md) | ARIMA vs ETS vs Theta vs Naive por horizonte |
| [Combinacao](combination.md) | 7 metodos de combinacao comparados |
| [Nowcasting](nowcasting.md) | DFM vs Bridge vs MIDAS para PIB brasileiro |
| [Comparacao Geral](comparison.md) | Tabela unificada e recomendacoes |
