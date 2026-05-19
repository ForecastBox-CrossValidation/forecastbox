---
title: "Metricas de Avaliacao"
description: "Metricas pontuais, percentuais, escaladas e probabilisticas para avaliacao de previsoes, com equacoes e guia de escolha."
---

# Metricas de Avaliacao

!!! abstract "Key Takeaway"
    Nenhuma metrica isolada captura toda a informacao sobre a qualidade de uma previsao. Use metricas **scale-dependent** para comparar modelos na mesma serie, **escaladas** (MASE) para comparar entre series, e **probabilisticas** (CRPS) para avaliar a distribuicao preditiva completa.

## Metricas Scale-Dependent

Metricas que dependem da escala da variavel — uteis para comparar modelos na mesma serie.

### MSE — Mean Squared Error

$$
\text{MSE} = \frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2
$$

Penaliza erros grandes desproporcionalmente. Base para muitos testes estatisticos.

### RMSE — Root Mean Squared Error

$$
\text{RMSE} = \sqrt{\frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2}
$$

Mesma unidade da variavel original. A metrica mais usada na pratica.

### MAE — Mean Absolute Error

$$
\text{MAE} = \frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|
$$

Robusta a outliers. Mediana da distribuicao de erros e otima sob MAE.

```python
from forecastbox.evaluation import forecast_metrics

metrics = forecast_metrics(
    actual=y_test,
    predicted=y_pred,
    metrics=["mse", "rmse", "mae"]
)
print(metrics)
```

```text
    MSE       RMSE      MAE
  0.00117   0.03421   0.02714
```

!!! note "RMSE vs MAE"
    Se $\text{RMSE} \gg \text{MAE}$, ha erros grandes esporadicos (outliers). A razao $\text{RMSE}/\text{MAE}$ varia de $1$ (erros uniformes) a $\sqrt{T}$ (erro concentrado em um ponto).

## Metricas Percentuais

Expressam o erro como proporcao do valor observado — uteis para comunicar resultados.

### MAPE — Mean Absolute Percentage Error

$$
\text{MAPE} = \frac{100}{T} \sum_{t=1}^{T} \left| \frac{y_t - \hat{y}_t}{y_t} \right|
$$

!!! warning "Limitacoes do MAPE"
    - Indefinido quando $y_t = 0$
    - Assimetrico: penaliza mais previsoes acima do observado
    - Nao deve ser usado com dados que cruzam zero (taxas de juros reais, inflacao)

### sMAPE — Symmetric MAPE

$$
\text{sMAPE} = \frac{200}{T} \sum_{t=1}^{T} \frac{|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}
$$

Corrige a assimetria do MAPE. Varia de 0% a 200%.

```python
metrics = forecast_metrics(
    actual=y_test,
    predicted=y_pred,
    metrics=["mape", "smape"]
)
print(metrics)
```

```text
    MAPE     sMAPE
   2.71%    2.68%
```

## Metricas Escaladas

### MASE — Mean Absolute Scaled Error

$$
\text{MASE} = \frac{\frac{1}{T} \sum_{t=1}^{T} |y_t - \hat{y}_t|}{\frac{1}{n-m} \sum_{i=m+1}^{n} |y_i - y_{i-m}|}
$$

onde $m$ e o periodo sazonal e o denominador e o MAE do naive sazonal no conjunto de treino (Hyndman & Koehler, 2006).

- $\text{MASE} < 1$: modelo supera o naive sazonal
- $\text{MASE} = 1$: equivalente ao naive sazonal
- $\text{MASE} > 1$: modelo perde para o naive sazonal

!!! tip "Por que usar MASE?"
    - Funciona com $y_t = 0$
    - Comparavel entre series de escalas diferentes
    - Simetrica e interpretavel
    - Recomendada por Hyndman & Koehler (2006) como metrica universal

```python
metrics = forecast_metrics(
    actual=y_test,
    predicted=y_pred,
    metrics=["mase"],
    seasonal_period=12  # para MASE sazonal
)
print(f"MASE: {metrics['mase']:.4f}")
```

## Metricas Probabilisticas

Avaliam a **distribuicao preditiva completa**, nao apenas a media.

### CRPS — Continuous Ranked Probability Score

$$
\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left( F(x) - \mathbb{1}(x \geq y) \right)^2 dx
$$

onde $F$ e a CDF preditiva e $y$ e o valor observado.

- Generaliza o MAE para distribuicoes
- Se $F$ e pontual (degenerada), $\text{CRPS} = \text{MAE}$
- Menor e melhor

Para distribuicao normal $N(\mu, \sigma^2)$:

$$
\text{CRPS}(N(\mu, \sigma^2), y) = \sigma \left[ \frac{y - \mu}{\sigma} \left( 2\Phi\left(\frac{y-\mu}{\sigma}\right) - 1 \right) + 2\phi\left(\frac{y-\mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}} \right]
$$

### Log Score

$$
\text{LogS} = -\frac{1}{T} \sum_{t=1}^{T} \log f_t(y_t)
$$

onde $f_t$ e a densidade preditiva. Fortemente sensivel a caudas — penaliza severamente eventos nos quais o modelo atribui probabilidade muito baixa.

### Winkler Score

Para um intervalo de previsao $[\ell_t, u_t]$ com nivel $(1-\alpha)$:

$$
W_t = \begin{cases}
(u_t - \ell_t) + \frac{2}{\alpha}(\ell_t - y_t) & \text{se } y_t < \ell_t \\
(u_t - \ell_t) & \text{se } \ell_t \leq y_t \leq u_t \\
(u_t - \ell_t) + \frac{2}{\alpha}(y_t - u_t) & \text{se } y_t > u_t
\end{cases}
$$

Penaliza intervalos largos e violacoes dos limites.

### PIT — Probability Integral Transform

$$
\text{PIT}_t = F_t(y_t)
$$

Se o modelo esta bem calibrado, $\text{PIT}_t \sim U(0,1)$. Use histogramas PIT para diagnosticar:

- **Uniforme**: calibracao perfeita
- **U-shaped**: distribuicao muito estreita (subdispersao)
- **Humped**: distribuicao muito larga (superdispersao)
- **Skewed**: vies direcional

```python
from forecastbox.evaluation import forecast_metrics

prob_metrics = forecast_metrics(
    actual=y_test,
    predicted_dist=dist,  # distribuicao preditiva
    metrics=["crps", "log_score", "winkler"],
    alpha=0.10  # para Winkler Score
)
print(prob_metrics)
```

```text
     CRPS    LogScore   Winkler
   0.0189    1.2341     0.1456
```

## Quando Usar Cada Metrica

| Metrica | Tipo | Vantagem | Limitacao | Quando Usar |
|---------|------|----------|-----------|-------------|
| **RMSE** | Scale-dep. | Interpretavel, mesma unidade | Sensivel a outliers | Comparar modelos na mesma serie |
| **MAE** | Scale-dep. | Robusta a outliers | Menos sensivel a erros grandes | Series com outliers |
| **MAPE** | Percentual | Facil interpretacao | Indefinida em $y=0$, assimetrica | Comunicacao com nao-tecnicos |
| **sMAPE** | Percentual | Simetrica | Escala 0-200% pouco intuitiva | Alternativa ao MAPE |
| **MASE** | Escalada | Comparavel entre series | Depende do benchmark naive | Comparacao entre series diferentes |
| **CRPS** | Probabilistica | Avalia toda a distribuicao | Requer distribuicao preditiva | Modelos probabilisticos |
| **Log Score** | Probabilistica | Sensivel a caudas | Muito punitiva | Eventos raros importam |
| **Winkler** | Probabilistica | Avalia intervalos diretamente | Depende do nivel $\alpha$ | Avaliar intervalos de confianca |
| **PIT** | Probabilistica | Diagnostico visual | Qualitativa | Checar calibracao |

!!! info "Referencia"
    Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.

## Ver Tambem

- [Diebold-Mariano](diebold-mariano.md) — testar se diferencas de metricas sao significativas
- [Cross-Validation](cross-validation.md) — calcular metricas de forma robusta
- :material-stethoscope: [Diagnostico de Eficiencia](../../diagnostics/efficiency.md) — diagnostico pratico de autocorrelacao, Ljung-Box e regressao auxiliar
