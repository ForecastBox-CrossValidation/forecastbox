---
title: "Teste de Vies"
description: "Diagnostico de vies em previsoes: vies incondicional, condicional, regressao de vies e tracking signal para monitoramento continuo."
---

# Teste de Vies

!!! abstract "Key Takeaway"
    Uma previsao viesada erra **sistematicamente** em uma direcao. O vies incondicional testa se a media dos erros e zero; o vies condicional verifica se o erro depende de informacao disponivel; o tracking signal monitora o vies ao longo do tempo.

## Conceito

O vies de previsao e a forma mais basica de patologia: se a previsao erra consistentemente para cima ou para baixo, e possivel melhorá-la com um simples ajuste de nivel — sem alterar o modelo.

Definimos o erro de previsao como:

$$
e_t = y_t - \hat{y}_t
$$

onde $y_t$ e o valor realizado e $\hat{y}_t$ a previsao.

## Vies Incondicional

O vies incondicional testa se a **media dos erros** e zero:

$$
H_0: E[e_t] = 0
$$

O teste e realizado via **t-test** para a media:

$$
t = \frac{\bar{e}}{s_e / \sqrt{T}}
$$

onde $\bar{e} = \frac{1}{T}\sum_{t=1}^{T} e_t$ e a media dos erros e $s_e$ o desvio-padrao amostral.

!!! info "Interpretacao"
    - $\bar{e} > 0$: previsao **subestima** o realizado (vies negativo da previsao)
    - $\bar{e} < 0$: previsao **superestima** o realizado (vies positivo da previsao)
    - p-valor $< 0.05$: evidencia estatistica de vies

### Exemplo: vies incondicional no IPCA

```python
import pandas as pd
from forecastbox.diagnostics import bias_test

# Previsao e realizado do IPCA mensal
actual = pd.Series([0.83, 0.84, 0.71, 0.44, 0.47, 0.67,
                    -0.02, 0.24, 0.26, 0.44, 0.51, 0.62],
                   name="IPCA")
predicted = pd.Series([0.75, 0.80, 0.65, 0.50, 0.42, 0.60,
                       0.05, 0.30, 0.20, 0.40, 0.55, 0.58],
                      name="Previsao")
errors = actual - predicted

# Teste de vies incondicional
bt = bias_test(errors, test="t_test")
print(f"Erro medio:  {bt.mean_error:.4f}")
print(f"Estatistica: {bt.statistic:.4f}")
print(f"p-valor:     {bt.pvalue:.4f}")
print(f"Viesado:     {bt.reject}")
```

```text
Erro medio:  0.0175
Estatistica: 0.5827
p-valor:     0.5723
Viesado:     False
```

!!! tip "Resultado"
    Com p-valor = 0.57, nao ha evidencia de vies incondicional: a previsao do IPCA nao erra sistematicamente em uma direcao.

## Vies Condicional

O vies incondicional pode ser zero na media, mas o erro pode depender de **informacao disponivel** no momento da previsao. O vies condicional testa se variaveis observaveis explicam o erro:

$$
e_t = \alpha + \beta z_t + u_t
$$

$$
H_0: \alpha = 0, \quad \beta = 0
$$

onde $z_t$ e um vetor de variaveis que estavam disponiveis ao forecaster (ex: nivel do cambio, taxa Selic, expectativas de mercado).

!!! warning "Por que testar vies condicional?"
    Uma previsao com media de erro zero pode ainda ter vies **regime-dependente**: subestimar em periodos de alta inflacao e superestimar em periodos de baixa. O teste condicional captura este padrao.

### Exemplo: vies condicional com Selic

```python
import pandas as pd
from forecastbox.diagnostics import bias_test

errors = actual - predicted
selic = pd.Series([13.75, 13.75, 13.75, 13.75, 13.25, 13.25,
                   12.75, 12.75, 12.25, 12.25, 11.75, 11.75],
                  name="Selic")

# Teste de vies condicional
bt_cond = bias_test(
    errors,
    test="regression",
    variables=pd.DataFrame({"selic": selic})
)
print(f"F-statistic: {bt_cond.statistic:.4f}")
print(f"p-valor:     {bt_cond.pvalue:.4f}")
print(f"Coef. selic: {bt_cond.coefficients['selic']:.4f}")
```

```text
F-statistic: 1.2341
p-valor:     0.3378
Coef. selic: 0.0052
```

## Vies por Horizonte

Previsoes frequentemente sao mais viesadas em horizontes longos. E util calcular o **erro medio por horizonte** $h$:

$$
\bar{e}_h = \frac{1}{T_h} \sum_{t=1}^{T_h} e_{t,h}
$$

```python
from forecastbox.diagnostics import bias_test

# Erros organizados por horizonte (DataFrame: linhas=tempo, colunas=horizonte)
for h in range(1, 13):
    bt_h = bias_test(errors_by_horizon[h])
    print(f"h={h:2d}: erro medio={bt_h.mean_error:+.4f}, p={bt_h.pvalue:.4f}")
```

```text
h= 1: erro medio=+0.0023, p=0.8812
h= 2: erro medio=+0.0089, p=0.6123
h= 3: erro medio=+0.0145, p=0.4521
h= 6: erro medio=+0.0312, p=0.1834
h=12: erro medio=+0.0587, p=0.0412 *
```

!!! info "Padrao tipico"
    O vies tende a **crescer com o horizonte**. Se o vies incondicional nao e significativo no agregado, pode ser significativo em horizontes especificos.

## Tracking Signal

O tracking signal monitora o **acumulo de vies** ao longo do tempo, permitindo detectar quando uma previsao antes nao-viesada comeca a divergir:

$$
TS_t = \frac{\sum_{s=1}^{t} e_s}{\text{MAD}_t}
$$

onde $\text{MAD}_t$ e o desvio absoluto medio:

$$
\text{MAD}_t = \frac{1}{t} \sum_{s=1}^{t} |e_s|
$$

!!! warning "Regra pratica"
    Se $|TS_t| > 4$, ha evidencia de vies sistematico acumulado. Valores persistentemente acima de 4 (ou abaixo de -4) indicam necessidade de recalibracao.

### Exemplo: tracking signal para monitoramento

```python
from forecastbox.diagnostics import tracking_signal

# Monitoramento continuo
ts = tracking_signal(errors)

print("Tracking Signal:")
for t, val in enumerate(ts.values, 1):
    flag = " *** ALERTA" if abs(val) > 4 else ""
    print(f"  t={t:2d}: TS={val:+.2f}{flag}")
```

```text
Tracking Signal:
  t= 1: TS=+1.00
  t= 2: TS=+1.33
  t= 3: TS=+1.56
  t= 4: TS=+0.82
  t= 5: TS=+1.15
  t= 6: TS=+1.43
  t= 7: TS=+0.27
  t= 8: TS=+0.12
  t= 9: TS=+0.35
  t=10: TS=+0.58
  t=11: TS=+0.73
  t=12: TS=+0.89
```

### Visualizacao

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Erro medio acumulado
axes[0].bar(range(1, len(errors)+1), errors, color="steelblue", alpha=0.7)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_ylabel("Erro de previsao")
axes[0].set_title("Erros de previsao do IPCA")

# Tracking signal
axes[1].plot(range(1, len(ts.values)+1), ts.values, "o-", color="teal")
axes[1].axhline(4, color="red", linestyle="--", label="Limite superior")
axes[1].axhline(-4, color="red", linestyle="--", label="Limite inferior")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Tracking Signal")
axes[1].set_xlabel("Periodo")
axes[1].legend()

plt.tight_layout()
plt.show()
```

## Parametros

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `errors` | `array-like` | — | Serie de erros de previsao ($e_t = y_t - \hat{y}_t$) |
| `test` | `str` | `"t_test"` | Tipo de teste: `"t_test"` (incondicional) ou `"regression"` (condicional) |
| `variables` | `DataFrame` | `None` | Variaveis condicionantes para teste de regressao |
| `horizon` | `int` | `None` | Se fornecido, calcula vies por horizonte |
| `alpha` | `float` | `0.05` | Nivel de significancia |

## Resumo dos Testes

| Teste | $H_0$ | Quando usar |
|-------|--------|-------------|
| t-test (incondicional) | $E[e_t] = 0$ | Primeiro diagnostico — verificar vies sistematico |
| Regressao (condicional) | $\alpha = 0, \beta = 0$ em $e_t = \alpha + \beta z_t + u_t$ | Quando ha suspeita de vies regime-dependente |
| Tracking signal | $\|TS_t\| \leq 4$ | Monitoramento continuo em producao |

## Proximos Passos

Uma previsao nao-viesada pode ainda ser **ineficiente** — os erros podem conter informacao previsivel. Veja o [teste de eficiencia](efficiency.md) para o proximo diagnostico na sequencia.

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [Avaliacao de Previsoes](../theory/evaluation-theory.md) — fundamentos teoricos dos testes de vies e eficiencia
    - :material-notebook-edit: **User Guide**: [Regressao Mincer-Zarnowitz](../user-guide/evaluation/mincer-zarnowitz.md) — teste conjunto de vies e eficiencia via regressao MZ
    - :material-arrow-right: **Proximo**: [Teste de Eficiencia](efficiency.md) — verificar se os erros contem informacao exploravel
