---
title: Media Simples
description: Combinacao por media simples, mediana e media aparada — o baseline surpreendentemente forte.
---

# Media Simples

A combinacao por media simples atribui pesos iguais a todos os modelos:
$w_i = 1/N$. Apesar de sua simplicidade, este metodo e um dos mais robustos
e frequentemente supera metodos mais sofisticados — o chamado
**forecast combination puzzle**.

---

## Por que Media Simples Funciona?

A media simples ignora informacao sobre a qualidade relativa dos modelos.
Intuitivamente, metodos que estimam pesos otimos deveriam ser superiores.
Na pratica, isso nem sempre ocorre por dois motivos:

1. **Erro de estimacao dos pesos**: estimar $w_i$ otimos requer estimar
   variancias e covariancias dos erros, o que introduce ruido amostral
2. **Instabilidade temporal**: a performance relativa dos modelos muda ao longo
   do tempo, tornando pesos estimados in-sample subotimos out-of-sample

!!! info "Resultado Teorico (Smith & Wallis, 2009)"

    A media simples e otima quando os erros de previsao dos modelos tem a
    mesma variancia ($\sigma_i^2 = \sigma^2$ para todo $i$) e mesma
    correlacao par-a-par ($\rho_{ij} = \rho$ para todo $i \neq j$). Na
    pratica, essa condicao e frequentemente uma boa aproximacao quando os
    modelos sao de qualidade similar.

---

## Variantes

O forecastbox implementa tres variantes de combinacao por pesos iguais:

### Media Aritmetica

A media simples classica:

$$
\hat{y}_{t+h|t}^c = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_{t+h|t}^{(i)}
$$

- Simples e intuitiva
- Sensivel a outliers (modelos com previsoes extremas)

### Mediana

A mediana das previsoes individuais:

$$
\hat{y}_{t+h|t}^c = \text{mediana}\left(\hat{y}_{t+h|t}^{(1)}, \ldots, \hat{y}_{t+h|t}^{(N)}\right)
$$

- Robusta a outliers e previsoes extremas
- Util quando o pool contem modelos de qualidade muito heterogenea
- Nao tem representacao como media ponderada com $\sum w_i = 1$

### Media Aparada (Trimmed Mean)

Remove uma fracao $\alpha$ das previsoes mais extremas antes de calcular a media:

$$
\hat{y}_{t+h|t}^c = \frac{1}{N - 2\lfloor \alpha N \rfloor} \sum_{i=\lfloor \alpha N \rfloor + 1}^{N - \lfloor \alpha N \rfloor} \hat{y}_{(i), t+h|t}
$$

onde $\hat{y}_{(i)}$ sao as previsoes ordenadas.

- Compromisso entre media e mediana
- $\alpha = 0$: media simples; $\alpha \to 0.5$: mediana
- Tipicamente $\alpha \in [0.05, 0.25]$

---

## Parametros

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `method` | `str` | `"mean"` | Variante: `"mean"`, `"median"`, `"trimmed"` |
| `trim_fraction` | `float` | `0.1` | Fracao a remover em cada cauda (so para `"trimmed"`) |

---

## Quando Usar

A media simples e recomendada quando:

- Ha **incerteza sobre a qualidade relativa** dos modelos
- O pool contem **muitos modelos** (diversificacao natural)
- O foco e **previsao out-of-sample** (robustez > otimalidade)
- Ha **poucos dados** para estimar pesos (evita overfitting)
- Como **benchmark** para comparar com metodos mais sofisticados

!!! tip "Regra Pratica"

    Comece sempre pela media simples. Metodos mais sofisticados so se justificam
    se demonstrarem ganho consistente em exercicios de previsao fora da amostra.

---

## Exemplos

### Combinando 5 Modelos

```python
import pandas as pd
from forecastbox.auto import AutoARIMA, AutoETS
from forecastbox.models import Theta, MSTL, Naive
from forecastbox.combine import combine

# Carregar serie temporal
y = pd.read_csv("ipca.csv", index_col="date", parse_dates=True)["ipca"]

# Ajustar 5 modelos
models = {
    "arima": AutoARIMA(seasonal=True, m=12),
    "ets": AutoETS(seasonal_periods=12),
    "theta": Theta(),
    "mstl": MSTL(seasonal_periods=12),
    "naive": Naive(seasonal=True, m=12),
}

forecasts = {}
for name, model in models.items():
    model.fit(y)
    forecasts[name] = model.predict(horizon=12)

# Combinar com media simples
fc_mean = combine(
    forecasts=list(forecasts.values()),
    method="simple",
    simple_method="mean",
)
print(fc_mean.summary())
```

```text
Combination Summary
===================
Method: Simple Average
Models: 5
Weights: [0.200, 0.200, 0.200, 0.200, 0.200]

             point     lo95     hi95
2024-01     4.52     3.78     5.26
2024-02     4.48     3.61     5.35
2024-03     4.55     3.52     5.58
...
2024-12     4.61     2.98     6.24
```

### Comparacao: Media vs Mediana vs Trimmed

```python
# Tres variantes
fc_mean = combine(forecasts=list(forecasts.values()), method="simple", simple_method="mean")
fc_median = combine(forecasts=list(forecasts.values()), method="simple", simple_method="median")
fc_trimmed = combine(forecasts=list(forecasts.values()), method="simple", simple_method="trimmed", trim_fraction=0.2)

# Comparar RMSE out-of-sample
from forecastbox.evaluate import accuracy

y_test = pd.read_csv("ipca_test.csv", index_col="date", parse_dates=True)["ipca"]

for name, fc in [("Mean", fc_mean), ("Median", fc_median), ("Trimmed", fc_trimmed)]:
    metrics = accuracy(fc, y_test)
    print(f"{name:8s}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}")
```

```text
Mean      RMSE=0.3421  MAE=0.2814
Median    RMSE=0.3387  MAE=0.2756
Trimmed   RMSE=0.3398  MAE=0.2781
```

### Comparacao: Combinacao vs Modelos Individuais

```python
# Comparar combinacao com cada modelo individual
print("Modelo Individual vs Combinacao (RMSE)")
print("=" * 45)

for name, fc in forecasts.items():
    metrics = accuracy(fc, y_test)
    print(f"{name:8s}  RMSE={metrics['rmse']:.4f}")

print("-" * 45)
metrics_comb = accuracy(fc_mean, y_test)
print(f"{'Comb':8s}  RMSE={metrics_comb['rmse']:.4f}  <-- combinacao")
```

```text
Modelo Individual vs Combinacao (RMSE)
=============================================
arima     RMSE=0.3612
ets       RMSE=0.3498
theta     RMSE=0.3789
mstl      RMSE=0.3567
naive     RMSE=0.4123
---------------------------------------------
Comb      RMSE=0.3421  <-- combinacao
```

!!! note "Combinacao Vence Modelos Individuais"

    Neste exemplo, a media simples (RMSE=0.3421) supera todos os modelos
    individuais, incluindo o melhor (ETS, RMSE=0.3498). Este resultado e
    tipico e ilustra o poder da diversificacao.

---

## Intervalos de Confianca

Os intervalos de confianca da combinacao sao calculados combinando as distribuicoes
preditivas dos modelos individuais. Para a media simples:

$$
\text{Var}(\hat{y}^c_{t+h|t}) = \frac{1}{N^2} \sum_{i=1}^{N} \sum_{j=1}^{N} \text{Cov}(\hat{y}^{(i)}_{t+h|t}, \hat{y}^{(j)}_{t+h|t})
$$

!!! warning "Correlacao entre Modelos"

    Se os modelos compartilham dados e metodologia similar, suas previsoes serao
    correlacionadas. Ignorar essa correlacao (assumindo independencia) subestima a
    variancia combinada e gera intervalos excessivamente estreitos.

---

## Proximos Passos

- **[Pesos Fixos](weighted.md)** — atribuir pesos baseados em performance historica
- **[OLS](ols.md)** — estimar pesos otimos por regressao
- **[Avaliacao](../evaluation/index.md)** — avaliar a combinacao com metricas out-of-sample

---

## Referencias

- **Smith, J. & Wallis, K.F.** (2009). "A Simple Explanation of the Forecast Combination Puzzle." *Oxford Bulletin of Economics and Statistics*, 71(3), 331-355.
- **Bates, J.M. & Granger, C.W.J.** (1969). "The Combination of Forecasts." *Operational Research Quarterly*, 20(4), 451-468.
- **Genre, V., Kenny, G., Meyler, A. & Timmermann, A.** (2013). "Combining Expert Forecasts: Can Anything Beat the Simple Average?" *International Journal of Forecasting*, 29(1), 108-121.
