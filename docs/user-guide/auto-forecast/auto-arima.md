---
title: AutoARIMA
description: Selecao automatica de modelos ARIMA sazonais com algoritmo stepwise, testes de raiz unitaria e diagnosticos.
---

# AutoARIMA

O `AutoARIMA` seleciona automaticamente as ordens $(p, d, q)$ e $(P, D, Q)[m]$ de um
modelo ARIMA sazonal. Suporta busca **stepwise** (rapida) e **grid search** (exaustiva),
com selecao via criterios de informacao.

---

## Modelo ARIMA Sazonal

Um modelo ARIMA$(p,d,q)(P,D,Q)[m]$ combina componentes autoregressivos, de integracao
e de media movel, tanto na parte regular quanto sazonal:

$$
\Phi_P(B^m)\phi_p(B)(1-B)^d(1-B^m)^D y_t = c + \Theta_Q(B^m)\theta_q(B)\varepsilon_t
$$

onde:

| Simbolo | Descricao |
|:--------|:----------|
| $\phi_p(B)$ | Polinomio autoregressivo de ordem $p$: $1 - \phi_1 B - \cdots - \phi_p B^p$ |
| $\theta_q(B)$ | Polinomio de media movel de ordem $q$: $1 + \theta_1 B + \cdots + \theta_q B^q$ |
| $\Phi_P(B^m)$ | Polinomio AR sazonal de ordem $P$ |
| $\Theta_Q(B^m)$ | Polinomio MA sazonal de ordem $Q$ |
| $(1-B)^d$ | Operador de diferenciacao de ordem $d$ |
| $(1-B^m)^D$ | Operador de diferenciacao sazonal de ordem $D$ |
| $c$ | Constante (drift) |
| $\varepsilon_t$ | Ruido branco $\sim N(0, \sigma^2)$ |
| $m$ | Periodo sazonal (ex: 12 para mensal, 4 para trimestral) |

---

## Algoritmo de Selecao

O `AutoARIMA` implementa dois algoritmos de busca:

### Stepwise (Hyndman-Khandakar)

O algoritmo stepwise, proposto por Hyndman & Khandakar (2008), e o metodo padrao.
Ele avalia um subconjunto inteligente de modelos candidatos:

1. **Determinar $d$** via testes de raiz unitaria (KPSS por padrao)
2. **Determinar $D$** via teste de sazonalidade (OCSB ou Canova-Hansen)
3. **Estimar modelos iniciais**: ARIMA$(0,d,0)$, ARIMA$(2,d,2)$, ARIMA$(1,d,0)$, ARIMA$(0,d,1)$
4. **Busca stepwise**: variar $p \pm 1$, $q \pm 1$, $P \pm 1$, $Q \pm 1$ a partir do melhor modelo
5. **Parar** quando nenhum vizinho melhora o criterio de informacao

!!! info "Velocidade vs Cobertura"

    O stepwise avalia tipicamente 20-40 modelos, enquanto o grid search pode
    avaliar centenas. Na maioria dos casos, o stepwise encontra o mesmo modelo
    otimo ou muito proximo.

### Grid Search

A busca exaustiva testa todas as combinacoes dentro dos limites especificados:

```python
arima = AutoARIMA(
    method="grid",       # busca exaustiva
    max_p=5, max_q=5,
    max_P=2, max_Q=2,
    seasonal=True, m=12,
)
```

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `method` | `str` | `"stepwise"` | Algoritmo de busca: `"stepwise"` ou `"grid"` |
| `max_p` | `int` | `5` | Ordem AR maxima |
| `max_d` | `int` | `2` | Ordem de diferenciacao maxima |
| `max_q` | `int` | `5` | Ordem MA maxima |
| `max_P` | `int` | `2` | Ordem AR sazonal maxima |
| `max_D` | `int` | `1` | Ordem de diferenciacao sazonal maxima |
| `max_Q` | `int` | `2` | Ordem MA sazonal maxima |
| `seasonal` | `bool` | `True` | Incluir componente sazonal |
| `m` | `int` | `1` | Periodo sazonal |
| `information_criterion` | `str` | `"aicc"` | Criterio: `"aic"`, `"bic"`, `"aicc"`, `"hqic"` |
| `test` | `str` | `"kpss"` | Teste de raiz unitaria: `"kpss"`, `"adf"`, `"pp"` |
| `seasonal_test` | `str` | `"ocsb"` | Teste sazonal: `"ocsb"`, `"ch"` |
| `with_intercept` | `bool \| str` | `"auto"` | Incluir constante: `True`, `False` ou `"auto"` |
| `n_jobs` | `int` | `1` | Paralelismo para grid search |

---

## Testes de Raiz Unitaria

O `AutoARIMA` determina automaticamente a ordem de diferenciacao $d$ usando testes
de raiz unitaria. O teste e aplicado iterativamente ate que a hipotese de estacionariedade
nao seja rejeitada.

### Testes Disponiveis

=== "KPSS (padrao)"

    Testa $H_0$: serie e estacionaria vs $H_1$: serie tem raiz unitaria.

    - Se $p$-valor $< \alpha$: rejeita estacionariedade, incrementa $d$
    - Conservador — tende a diferenciar menos
    - Recomendado para series economicas

    ```python
    arima = AutoARIMA(test="kpss", m=12)
    ```

=== "ADF"

    Testa $H_0$: serie tem raiz unitaria vs $H_1$: serie e estacionaria.

    - Se $p$-valor $> \alpha$: nao rejeita raiz unitaria, incrementa $d$
    - Pode sobre-diferenciar em series com quebra estrutural

    ```python
    arima = AutoARIMA(test="adf", m=12)
    ```

=== "PP"

    Phillips-Perron: similar ao ADF, mas robusto a autocorrelacao serial e
    heterocedasticidade.

    ```python
    arima = AutoARIMA(test="pp", m=12)
    ```

### Testes de Sazonalidade

Para determinar $D$ (diferenciacao sazonal):

| Teste | Hipotese Nula | Uso |
|:------|:-------------|:----|
| **OCSB** | Serie tem raiz unitaria sazonal | Padrao |
| **Canova-Hansen** | Serie e sazonalmente estacionaria | Alternativa |

!!! warning "Limites de Diferenciacao"

    O `max_d` e `max_D` limitam o numero maximo de diferenciacoes. Na pratica,
    $d > 2$ ou $D > 1$ raramente sao necessarios e podem indicar problemas nos
    dados (quebra estrutural, outliers).

---

## Exemplos

### Serie Mensal do PIB

```python
import pandas as pd
from forecastbox.auto import AutoARIMA

# Carregar serie mensal do PIB (proxy IBC-Br)
y = pd.read_csv("ibc_br.csv", index_col="date", parse_dates=True)["ibc"]

# Ajustar AutoARIMA com sazonalidade mensal
model = AutoARIMA(
    seasonal=True,
    m=12,
    information_criterion="aicc",
)
model.fit(y)

# Resumo do modelo selecionado
print(model.summary())
```

```text
AutoARIMA Summary
=================
Selected: ARIMA(1,1,1)(0,1,1)[12]
AICc: 1523.4  |  BIC: 1541.2

Coefficients:
         coef    se     z     p-value
ar1     0.412  0.087  4.74   <0.001
ma1    -0.628  0.072 -8.72   <0.001
sma1   -0.541  0.094 -5.76   <0.001

Unit Root Tests:
  KPSS: d=1 (p=0.013)
  OCSB: D=1 (p=0.002)

Residual Diagnostics:
  Ljung-Box(12): Q=14.3, p=0.281
  Jarque-Bera:   JB=2.1, p=0.350
```

```python
# Previsao 12 meses a frente
forecast = model.predict(horizon=12, level=[80, 95])
print(forecast)
```

```text
             point     lo80     hi80     lo95     hi95
2024-01    102.34    98.21   106.47    96.12   108.56
2024-02    103.12    97.89   108.35    95.23   111.01
2024-03    104.56    98.15   110.97    94.87   114.25
...
2024-12    108.23    97.42   119.04    91.89   124.57
```

### Serie com Sazonalidade Forte

```python
# Serie de producao industrial com sazonalidade marcante
y_ind = pd.read_csv("producao_industrial.csv", index_col="date", parse_dates=True)["prod"]

model = AutoARIMA(
    seasonal=True,
    m=12,
    max_P=2,
    max_Q=2,
    information_criterion="bic",  # BIC penaliza mais parametros
)
model.fit(y_ind)

print(f"Modelo: {model.order_} x {model.seasonal_order_}")
print(f"BIC: {model.bic_:.1f}")
```

```text
Modelo: (2, 1, 1) x (1, 1, 1, 12)
BIC: 2847.3
```

### Serie com Tendencia (sem Sazonalidade)

```python
# Serie trimestral de divida publica
y_div = pd.read_csv("divida_publica.csv", index_col="date", parse_dates=True)["divida"]

model = AutoARIMA(
    seasonal=False,   # sem sazonalidade
    max_p=3,
    max_q=3,
    test="adf",       # testar com ADF
)
model.fit(y_div)

print(f"Modelo: ARIMA{model.order_}")
print(f"AICc: {model.aicc_:.1f}")
```

```text
Modelo: ARIMA(1, 2, 0)
AICc: 412.7
```

---

## Diagnosticos Pos-Ajuste

Apos selecionar o modelo, e essencial verificar se os residuos se comportam como
ruido branco.

### Verificacao de Residuos

```python
# Diagnosticos completos
diag = model.diagnostics()
diag.plot()
```

O metodo `diagnostics()` retorna:

| Teste | Hipotese Nula | Interpretacao |
|:------|:-------------|:--------------|
| **Ljung-Box** | Residuos sao ruido branco | $p > 0.05$: residuos OK |
| **Jarque-Bera** | Residuos sao normais | $p > 0.05$: normalidade OK |
| **ARCH-LM** | Sem efeitos ARCH | $p > 0.05$: variancia constante |

### ACF/PACF dos Residuos

```python
# Grafico ACF/PACF dos residuos
model.plot_diagnostics(figsize=(12, 8))
```

!!! note "Residuos com Autocorrelacao"

    Se o teste Ljung-Box rejeita a hipotese nula ($p < 0.05$), os residuos
    apresentam autocorrelacao remanescente. Considere:

    - Aumentar `max_p` e `max_q`
    - Verificar se a sazonalidade esta correta (`m`)
    - Usar `method="grid"` para busca mais ampla

!!! tip "Criterio de Informacao"

    - **AICc** (padrao): melhor para previsao, penaliza menos parametros
    - **BIC**: melhor para identificacao do modelo verdadeiro, mais parcimonioso
    - Para series curtas ($n < 100$), prefira AICc sobre AIC

---

## Comparacao de Modelos

O `AutoARIMA` armazena todos os modelos avaliados, permitindo inspecao:

```python
# Top 5 modelos avaliados
results = model.results_table_
print(results.head())
```

```text
                    Model     AICc      BIC  Converged
0   ARIMA(1,1,1)(0,1,1)[12]  1523.4  1541.2       True
1   ARIMA(2,1,1)(0,1,1)[12]  1524.8  1546.5       True
2   ARIMA(1,1,2)(0,1,1)[12]  1525.1  1546.8       True
3   ARIMA(0,1,1)(0,1,1)[12]  1528.7  1542.6       True
4   ARIMA(1,1,1)(1,1,1)[12]  1525.2  1550.8       True
```

!!! warning "Convergencia"

    Modelos que nao convergiram sao descartados automaticamente. Se muitos modelos
    falham na convergencia, verifique se a serie tem observacoes ausentes, outliers
    extremos ou comprimento insuficiente.

---

## Proximos Passos

- **[AutoETS](auto-ets.md)** — suavizacao exponencial com selecao automatica de componentes
- **[Avaliacao](../evaluation/index.md)** — compare AutoARIMA com outros modelos
- **[Combinacao](../combination/index.md)** — combine AutoARIMA com AutoETS para ganhar robustez
