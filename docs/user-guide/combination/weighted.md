---
title: Pesos Fixos
description: Combinacao com pesos fixos definidos pelo usuario, inversamente proporcionais ao MSE ou por ranking de performance.
---

# Pesos Fixos (Weighted)

A combinacao com pesos fixos permite ao usuario **definir explicitamente** a importancia
de cada modelo, ou derivar pesos automaticamente a partir da **performance historica**.
Diferente da media simples, esse metodo incorpora informacao sobre a qualidade relativa
dos modelos.

---

## Metodos de Ponderacao

O forecastbox oferece tres estrategias para definir pesos fixos:

### Inverse MSE

Pesos inversamente proporcionais ao erro quadratico medio de cada modelo:

$$
w_i = \frac{1/\text{MSE}_i}{\sum_{j=1}^{N} 1/\text{MSE}_j}
$$

onde $\text{MSE}_i$ e o erro quadratico medio do modelo $i$ calculado em um periodo
de avaliacao (tipicamente out-of-sample).

- Modelos com menor erro recebem maior peso
- Normalizacao garante $\sum w_i = 1$
- Generalizavel para outros criterios: MAE, MAPE, RMSE

!!! info "Escolha do Criterio de Erro"

    O MSE penaliza mais erros grandes. Para series com outliers ocasionais, considere
    usar MAE ($w_i \propto 1/\text{MAE}_i$) que e mais robusto a valores extremos.

### Ranking

Pesos proporcionais ao ranking de performance de cada modelo:

$$
w_i = \frac{N - r_i + 1}{\sum_{j=1}^{N} (N - r_j + 1)}
$$

onde $r_i$ e o ranking do modelo $i$ ($r_i = 1$ para o melhor modelo).

- Menos sensivel a diferencas absolutas de performance
- O melhor modelo recebe peso proporcional a $N$, o pior recebe peso 1
- Util quando as metricas de erro tem escala dificil de interpretar

### Custom

Pesos definidos diretamente pelo usuario:

$$
w_i = \frac{u_i}{\sum_{j=1}^{N} u_j}
$$

onde $u_i$ sao os pesos fornecidos. Se `normalize=True` (padrao), os pesos sao
normalizados para somar 1.

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `method` | `str` | `"inverse_mse"` | Metodo de ponderacao: `"inverse_mse"`, `"rank"`, `"custom"` |
| `weights` | `list[float]` | `None` | Pesos customizados (obrigatorio para `method="custom"`) |
| `normalize` | `bool` | `True` | Normalizar pesos para somar 1 |
| `metric` | `str` | `"mse"` | Metrica de erro para `"inverse_mse"`: `"mse"`, `"mae"`, `"rmse"`, `"mape"` |
| `evaluation_data` | `pd.Series` | `None` | Serie de avaliacao para calcular metricas (obrigatorio para `"inverse_mse"` e `"rank"`) |

---

## Exemplos

### Pesos Baseados em Erro Out-of-Sample

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.models import Theta
from forecastbox.combine import combine
from forecastbox.evaluate import accuracy

# Dados: treino + validacao + teste
y = pd.read_csv("ipca.csv", index_col="date", parse_dates=True)["ipca"]
y_train = y[:"2022-12"]
y_val = y["2023-01":"2023-06"]
y_test = y["2023-07":"2023-12"]

# Ajustar modelos no treino
arima = AutoARIMA(seasonal=True, m=12).fit(y_train)
ets = AutoETS(seasonal_periods=12).fit(y_train)
theta = Theta().fit(y_train)

# Gerar previsoes para validacao
fc_arima = arima.predict(horizon=6)
fc_ets = ets.predict(horizon=6)
fc_theta = theta.predict(horizon=6)

# Combinar com pesos inverse MSE
fc_weighted = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="weighted",
    weighted_method="inverse_mse",
    evaluation_data=y_val,
)
print(fc_weighted.summary())
```

```text
Combination Summary
===================
Method: Inverse MSE Weights
Models: 3
Weights: [0.412, 0.358, 0.230]
Metric: MSE

Model Performance (validation):
  arima   MSE=0.1234  weight=0.412
  ets     MSE=0.1420  weight=0.358
  theta   MSE=0.2210  weight=0.230
```

### Pesos por Ranking

```python
# Combinar com pesos por ranking
fc_rank = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="weighted",
    weighted_method="rank",
    evaluation_data=y_val,
)
print(fc_rank.summary())
```

```text
Combination Summary
===================
Method: Rank Weights
Models: 3
Weights: [0.500, 0.333, 0.167]

Model Rankings (validation):
  arima   rank=1  weight=0.500
  ets     rank=2  weight=0.333
  theta   rank=3  weight=0.167
```

### Pesos Customizados (Expert Weights)

```python
# Pesos definidos pelo especialista
fc_custom = combine(
    forecasts=[fc_arima, fc_ets, fc_theta],
    method="weighted",
    weighted_method="custom",
    weights=[0.5, 0.3, 0.2],
)
print(fc_custom.weights_)
```

```text
[0.500, 0.300, 0.200]
```

### Comparacao dos Metodos de Ponderacao

```python
# Reajustar modelos no treino+validacao e prever teste
y_full = y[:"2023-06"]
arima = AutoARIMA(seasonal=True, m=12).fit(y_full)
ets = AutoETS(seasonal_periods=12).fit(y_full)
theta = Theta().fit(y_full)

fc_arima = arima.predict(horizon=6)
fc_ets = ets.predict(horizon=6)
fc_theta = theta.predict(horizon=6)

forecasts = [fc_arima, fc_ets, fc_theta]

# Comparar metodos
methods = {
    "Simple": combine(forecasts, method="simple"),
    "Inv. MSE": combine(forecasts, method="weighted", weighted_method="inverse_mse", evaluation_data=y_val),
    "Rank": combine(forecasts, method="weighted", weighted_method="rank", evaluation_data=y_val),
    "Custom": combine(forecasts, method="weighted", weighted_method="custom", weights=[0.5, 0.3, 0.2]),
}

for name, fc in methods.items():
    metrics = accuracy(fc, y_test)
    print(f"{name:10s}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
```

```text
Simple      RMSE=0.3421  MAE=0.2814
Inv. MSE    RMSE=0.3312  MAE=0.2723
Rank        RMSE=0.3345  MAE=0.2751
Custom      RMSE=0.3298  MAE=0.2698
```

!!! tip "Validacao Out-of-Sample"

    Sempre avalie os pesos em dados **fora da amostra** (out-of-sample).
    Pesos otimizados in-sample tendem a sobreajustar, especialmente com
    poucos modelos e series curtas.

---

## Inverse MSE vs Ranking: Quando Usar Cada

| Criterio | Inverse MSE | Ranking |
|:---------|:------------|:--------|
| Sensibilidade a outliers | Alta | Baixa |
| Diferenciacao entre modelos | Proporcional ao erro | Depende so da ordem |
| Estabilidade temporal | Menos estavel | Mais estavel |
| Quando usar | Diferenca de performance grande e estavel | Diferenca de performance pequena ou instavel |

!!! warning "Pesos Concentrados"

    Quando um modelo e muito superior aos demais, o Inverse MSE pode atribuir
    quase todo o peso a ele, efetivamente eliminando a diversificacao. Nesse caso,
    considere o metodo de ranking ou limitar os pesos minimos.

---

## Proximos Passos

- **[OLS](ols.md)** — estimar pesos otimos por regressao
- **[Stacking](stacking.md)** — pesos estimados com regularizacao
- **[Media Simples](simple.md)** — voltar ao baseline sem pesos

---

## Referencias

- **Bates, J.M. & Granger, C.W.J.** (1969). "The Combination of Forecasts." *Operational Research Quarterly*, 20(4), 451-468.
- **Stock, J.H. & Watson, M.W.** (2004). "Combination Forecasts of Output Growth in a Seven-Country Data Set." *Journal of Forecasting*, 23(6), 405-430.
- **Timmermann, A.** (2006). "Forecast Combinations." *Handbook of Economic Forecasting*, 1, 135-196.
