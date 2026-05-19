---
title: "Fundamentos de Previsao"
description: "Tutorial completo para iniciantes: AutoARIMA, AutoETS, Forecast container, metricas e visualizacao"
---

# Fundamentos de Previsao com forecastbox

!!! info "Sobre este tutorial"
    **Nivel**: :material-star: Iniciante
    **Tempo estimado**: 30 minutos
    **Pre-requisitos**: Python basico, pandas, estatistica descritiva
    **Dados**: PIB trimestral brasileiro (dataset incluso)

Neste tutorial voce vai aprender o workflow completo de previsao com o forecastbox:
desde carregar dados ate salvar previsoes. Ao final, voce estara confortavel com
os conceitos fundamentais e pronto para tutoriais mais avancados.

## O que voce vai aprender

- Carregar e explorar dados de series temporais
- Ajustar um modelo AutoARIMA
- Entender o Forecast container (values, intervals, metadata)
- Calcular metricas basicas (RMSE, MAE, MAPE)
- Visualizar previsoes com intervalos de confianca
- Comparar AutoARIMA vs AutoETS
- Salvar e carregar previsoes

---

## Etapa 1: Setup e Dados

Comece importando o forecastbox e carregando o dataset de PIB trimestral:

```python
import pandas as pd
import forecastbox as fb
from forecastbox.datasets import load_gdp

# Carregar serie de PIB trimestral
data = load_gdp()

print(f"Serie: {data.name}")
print(f"Frequencia: {data.index.freqstr}")
print(f"Periodos: {len(data)}")
print(f"Periodo: {data.index[0].strftime('%Y-Q%q')} a {data.index[-1].strftime('%Y-Q%q')}")
```

```text
Serie: gdp_growth
Frequencia: QS
Periodos: 80
Periodo: 2004-Q1 a 2023-Q4
```

!!! note "Datasets inclusos"
    O forecastbox inclui varios datasets para experimentacao:

    | Dataset | Funcao | Descricao |
    |---------|--------|-----------|
    | PIB | `load_gdp()` | Crescimento trimestral do PIB (%) |
    | Inflacao | `load_inflation()` | IPCA mensal (% a.m.) |
    | Cambio | `load_exchange_rate()` | Taxa R$/USD diaria |
    | Desemprego | `load_unemployment()` | Taxa de desemprego trimestral (%) |

---

## Etapa 2: Explorar os Dados

Antes de modelar, e fundamental entender os dados. Vamos analisar estatisticas
descritivas, tendencia e sazonalidade:

```python
# Estatisticas descritivas
print(data.describe())
```

```text
count    80.000000
mean      1.825000
std       2.341562
min      -5.000000
25%       0.900000
50%       2.050000
75%       2.800000
max       7.500000
Name: gdp_growth, dtype: float64
```

```python
# Visualizar a serie completa
from forecastbox.viz import plot_series

fig = plot_series(
    data,
    title="Crescimento do PIB Trimestral (%)",
    ylabel="Crescimento (%)",
)
fig.show()
```

O grafico mostra a evolucao do PIB trimestral ao longo do tempo.
Observe padroes sazonais e possiveis quebras estruturais.

```python
# Decomposicao sazonal
from forecastbox.viz import plot_decomposition

fig = plot_decomposition(data, period=4)
fig.show()
```

A decomposicao revela tres componentes:

- **Tendencia**: Direcao de longo prazo da serie
- **Sazonalidade**: Padrao que se repete a cada 4 trimestres
- **Residuo**: Variacao nao explicada pelos componentes anteriores

!!! example "Try it yourself"
    Carregue o dataset de inflacao com `load_inflation()` e repita a analise
    exploratoria. Quais diferencas voce observa em relacao ao PIB?

    ```python
    from forecastbox.datasets import load_inflation
    inflation = load_inflation()
    print(inflation.describe())
    ```

---

## Etapa 3: Primeiro Modelo -- AutoARIMA

O `AutoARIMA` seleciona automaticamente a melhor especificacao ARIMA$(p,d,q)$
via criterios de informacao (AIC/BIC). Vamos separar treino/teste e ajustar:

```python
from forecastbox import AutoARIMA

# Separar treino e teste (ultimos 8 trimestres para teste)
train, test = data[:-8], data[-8:]
print(f"Treino: {len(train)} obs ({train.index[0]} a {train.index[-1]})")
print(f"Teste:  {len(test)} obs ({test.index[0]} a {test.index[-1]})")
```

```text
Treino: 72 obs (2004-01-01 a 2021-12-01)
Teste:  8 obs (2022-01-01 a 2023-10-01)
```

```python
# Ajustar AutoARIMA
model_arima = AutoARIMA()
result_arima = model_arima.fit_predict(train, horizon=8)

print(f"Modelo selecionado: {result_arima.model_name}")
print(f"\nPrevisao (8 trimestres):")
print(result_arima.forecast)
```

```text
Modelo selecionado: ARIMA(1,1,1)

Previsao (8 trimestres):
2022Q1    2.3
2022Q2    2.1
2022Q3    1.9
2022Q4    1.8
2023Q1    1.7
2023Q2    1.7
2023Q3    1.6
2023Q4    1.6
Freq: QS, Name: gdp_growth, dtype: float64
```

O modelo ARIMA$(p,d,q)$ e definido como:

$$
\phi(B)(1-B)^d y_t = \theta(B)\varepsilon_t
$$

onde $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ e o polinomio autorregressivo
e $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ e o polinomio de medias moveis.

!!! note "Estrategias de busca"
    O `AutoARIMA` suporta duas estrategias:

    - `strategy="stepwise"` (padrao) -- busca rapida via algoritmo stepwise
    - `strategy="exhaustive"` -- busca exaustiva em todo o espaco de modelos

    ```python
    model = AutoARIMA(strategy="exhaustive", ic="bic")
    ```

---

## Etapa 4: O Forecast Container

O resultado de `fit_predict()` retorna um objeto `Forecast` que contem toda a
informacao da previsao. Vamos explorar seus atributos:

```python
# Valores previstos
print("=== Valores ===")
print(result_arima.forecast)

# Intervalos de confianca
print("\n=== Intervalos de Confianca (95%) ===")
print(result_arima.intervals)

# Metadata do modelo
print("\n=== Metadata ===")
print(f"Modelo: {result_arima.model_name}")
print(f"IC usado: {result_arima.metadata['ic']}")
print(f"AIC: {result_arima.metadata['aic']:.2f}")
print(f"BIC: {result_arima.metadata['bic']:.2f}")
print(f"Horizonte: {result_arima.horizon}")
```

```text
=== Valores ===
2022Q1    2.3
2022Q2    2.1
2022Q3    1.9
2022Q4    1.8
2023Q1    1.7
2023Q2    1.7
2023Q3    1.6
2023Q4    1.6
Freq: QS, Name: gdp_growth, dtype: float64

=== Intervalos de Confianca (95%) ===
         lower  upper
2022Q1   -0.5    5.1
2022Q2   -1.2    5.4
2022Q3   -1.7    5.5
2022Q4   -2.0    5.6
2023Q1   -2.2    5.6
2023Q2   -2.3    5.7
2023Q3   -2.4    5.6
2023Q4   -2.4    5.6

=== Metadata ===
Modelo: ARIMA(1,1,1)
IC usado: aic
AIC: 245.32
BIC: 252.18
Horizonte: 8
```

| Atributo | Tipo | Descricao |
|----------|------|-----------|
| `forecast` | `pd.Series` | Valores pontuais da previsao |
| `intervals` | `pd.DataFrame` | Intervalos de confianca (`lower`, `upper`) |
| `model_name` | `str` | Nome do modelo selecionado |
| `metadata` | `dict` | Informacoes adicionais (AIC, BIC, parametros) |
| `horizon` | `int` | Numero de passos a frente |
| `fitted_values` | `pd.Series` | Valores ajustados in-sample |
| `residuals` | `pd.Series` | Residuos do modelo |

!!! example "Try it yourself"
    Explore os residuos do modelo. Eles parecem ruido branco?

    ```python
    print(f"Media dos residuos: {result_arima.residuals.mean():.4f}")
    print(f"Std dos residuos: {result_arima.residuals.std():.4f}")

    from forecastbox.viz import plot_residuals
    fig = plot_residuals(result_arima.residuals)
    fig.show()
    ```

---

## Etapa 5: Metricas Basicas

Agora vamos avaliar a qualidade da previsao comparando com os dados de teste:

```python
from forecastbox.evaluate import metrics

# Calcular metricas
m = metrics.compute(test, result_arima.forecast)

print("=== Metricas de Avaliacao ===")
print(f"RMSE:  {m.rmse:.4f}")
print(f"MAE:   {m.mae:.4f}")
print(f"MAPE:  {m.mape:.2f}%")
```

```text
=== Metricas de Avaliacao ===
RMSE:  1.8523
MAE:   1.4217
MAPE:  12.34%
```

As tres metricas mais usadas em previsao:

| Metrica | Formula | Interpretacao |
|---------|---------|---------------|
| **RMSE** | $\sqrt{\frac{1}{h}\sum_{t=1}^{h}(y_t - \hat{y}_t)^2}$ | Penaliza erros grandes (mesma unidade de $y$) |
| **MAE** | $\frac{1}{h}\sum_{t=1}^{h}\lvert y_t - \hat{y}_t \rvert$ | Mais robusto a outliers (mesma unidade de $y$) |
| **MAPE** | $\frac{100}{h}\sum_{t=1}^{h}\left\lvert\frac{y_t - \hat{y}_t}{y_t}\right\rvert$ | Erro percentual -- facilita comparacao entre series |

!!! warning "Cuidado com MAPE"
    O MAPE e indefinido quando $y_t = 0$ e pode ser assimetrico. Para series
    que cruzam zero, prefira RMSE ou MAE. Para series com escala muito diferente,
    considere o MASE (Mean Absolute Scaled Error).

!!! example "Try it yourself"
    Calcule tambem o MASE e o Theil U:

    ```python
    m_full = metrics.compute(
        test, result_arima.forecast,
        metrics=["rmse", "mae", "mape", "mase", "theil_u"]
    )
    print(m_full)
    ```

---

## Etapa 6: Visualizar Previsao

Visualize a previsao junto com os dados historicos e intervalos de confianca:

```python
from forecastbox.viz import plot_forecast

fig = plot_forecast(
    actual=data,
    forecast=result_arima.forecast,
    intervals=result_arima.intervals,
    title="Previsao do PIB -- AutoARIMA",
    ylabel="Crescimento (%)",
    ci=True,
)
fig.show()
```

O grafico exibe:

- **Linha azul**: dados historicos observados
- **Linha vermelha**: previsao do modelo
- **Area sombreada**: intervalo de confianca de 95%
- **Linha vertical tracejada**: inicio do horizonte de previsao
- **Pontos pretos**: valores realizados (teste)

```python
# Adicionar os valores realizados ao grafico
fig = plot_forecast(
    actual=data,
    forecast=result_arima.forecast,
    intervals=result_arima.intervals,
    test=test,
    title="Previsao vs Realizado -- AutoARIMA",
    ylabel="Crescimento (%)",
    ci=True,
    show_test=True,
)
fig.show()
```

!!! example "Try it yourself"
    Experimente ajustar o nivel de confianca dos intervalos:

    ```python
    result_80 = model_arima.fit_predict(train, horizon=8, level=80)
    result_95 = model_arima.fit_predict(train, horizon=8, level=95)

    print("Intervalo 80%:", result_80.intervals.iloc[0].values)
    print("Intervalo 95%:", result_95.intervals.iloc[0].values)
    ```

---

## Etapa 7: Segundo Modelo -- AutoETS

Vamos ajustar um segundo modelo para comparacao. O `AutoETS` seleciona
automaticamente a melhor especificacao ETS (Error, Trend, Seasonal):

```python
from forecastbox import AutoETS

# Ajustar AutoETS
model_ets = AutoETS()
result_ets = model_ets.fit_predict(train, horizon=8)

print(f"Modelo selecionado: {result_ets.model_name}")
print(f"\nPrevisao (8 trimestres):")
print(result_ets.forecast)
```

```text
Modelo selecionado: ETS(M,A,M)

Previsao (8 trimestres):
2022Q1    2.5
2022Q2    2.2
2022Q3    2.0
2022Q4    1.9
2023Q1    2.4
2023Q2    2.1
2023Q3    1.9
2023Q4    1.8
Freq: QS, Name: gdp_growth, dtype: float64
```

O modelo ETS e definido pela tripla (Error, Trend, Seasonal), onde cada componente
pode ser:

| Componente | Opcoes | Descricao |
|------------|--------|-----------|
| **Error** | A (Aditivo), M (Multiplicativo) | Estrutura do erro |
| **Trend** | N (Nenhum), A (Aditivo), Ad (Amortecido) | Componente de tendencia |
| **Seasonal** | N (Nenhum), A (Aditivo), M (Multiplicativo) | Componente sazonal |

```python
# Comparar metricas
m_arima = metrics.compute(test, result_arima.forecast)
m_ets = metrics.compute(test, result_ets.forecast)

print("           RMSE     MAE    MAPE")
print(f"AutoARIMA  {m_arima.rmse:>6.4f}  {m_arima.mae:>6.4f}  {m_arima.mape:>5.2f}%")
print(f"AutoETS    {m_ets.rmse:>6.4f}  {m_ets.mae:>6.4f}  {m_ets.mape:>5.2f}%")
```

```text
           RMSE     MAE    MAPE
AutoARIMA  1.8523  1.4217  12.34%
AutoETS    1.7891  1.3645  11.87%
```

```python
# Visualizar ambos os modelos
from forecastbox.viz import plot_forecast_comparison

fig = plot_forecast_comparison(
    actual=data,
    forecasts={"AutoARIMA": result_arima, "AutoETS": result_ets},
    test=test,
    title="Comparacao: AutoARIMA vs AutoETS",
    ylabel="Crescimento (%)",
)
fig.show()
```

!!! note "Qual modelo escolher?"
    Neste exemplo, o AutoETS apresentou metricas ligeiramente melhores.
    Mas diferencas pequenas podem nao ser estatisticamente significativas.
    No tutorial de [Avaliacao](evaluation.md), voce aprendera a usar o teste
    de Diebold-Mariano para verificar se a diferenca e real.

!!! example "Try it yourself"
    Adicione um terceiro modelo -- o `Theta` -- e compare os tres:

    ```python
    from forecastbox import Theta

    model_theta = Theta()
    result_theta = model_theta.fit_predict(train, horizon=8)

    m_theta = metrics.compute(test, result_theta.forecast)
    print(f"Theta      {m_theta.rmse:>6.4f}  {m_theta.mae:>6.4f}  {m_theta.mape:>5.2f}%")
    ```

---

## Etapa 8: Salvar e Carregar Previsoes

O forecastbox permite salvar previsoes em diferentes formatos para
uso posterior ou compartilhamento:

```python
# Salvar previsao em CSV
result_arima.to_csv("previsao_pib_arima.csv")
print("Salvo: previsao_pib_arima.csv")

# Salvar previsao em formato nativo (inclui metadata)
result_arima.save("previsao_pib_arima.fcst")
print("Salvo: previsao_pib_arima.fcst")
```

```text
Salvo: previsao_pib_arima.csv
Salvo: previsao_pib_arima.fcst
```

```python
# Carregar previsao salva
from forecastbox import Forecast

loaded = Forecast.load("previsao_pib_arima.fcst")

print(f"Modelo: {loaded.model_name}")
print(f"Horizonte: {loaded.horizon}")
print(loaded.forecast)
```

```text
Modelo: ARIMA(1,1,1)
Horizonte: 8
2022Q1    2.3
2022Q2    2.1
2022Q3    1.9
2022Q4    1.8
2023Q1    1.7
2023Q2    1.7
2023Q3    1.6
2023Q4    1.6
Freq: QS, Name: gdp_growth, dtype: float64
```

| Formato | Metodo | Inclui Metadata | Uso |
|---------|--------|-----------------|-----|
| CSV | `to_csv()` | Nao | Compartilhamento, Excel |
| JSON | `to_json()` | Sim | APIs, integracao |
| Nativo | `save()` / `Forecast.load()` | Sim | Persistencia completa |

!!! example "Try it yourself"
    Salve a previsao em JSON e inspecione a estrutura:

    ```python
    result_arima.to_json("previsao_pib_arima.json")

    import json
    with open("previsao_pib_arima.json") as f:
        data_json = json.load(f)
    print(json.dumps(data_json, indent=2)[:500])
    ```

---

## Resumo

Neste tutorial voce aprendeu o workflow completo de previsao com o forecastbox:

| Etapa | O que voce fez | Funcao principal |
|-------|----------------|------------------|
| 1 | Carregou dados de exemplo | `load_gdp()` |
| 2 | Explorou a serie temporal | `plot_series()`, `plot_decomposition()` |
| 3 | Ajustou AutoARIMA | `AutoARIMA().fit_predict()` |
| 4 | Explorou o Forecast container | `.forecast`, `.intervals`, `.metadata` |
| 5 | Calculou metricas | `metrics.compute()` |
| 6 | Visualizou a previsao | `plot_forecast()` |
| 7 | Comparou com AutoETS | `AutoETS().fit_predict()` |
| 8 | Salvou e carregou previsoes | `.save()`, `Forecast.load()` |

---

## Proximos Passos

<div class="grid cards" markdown>

- :material-set-merge: **[Combinacao de Previsoes](combination.md)**

    Aprenda a combinar AutoARIMA + AutoETS + outros para obter previsoes mais robustas

- :material-test-tube: **[Avaliacao Rigorosa](evaluation.md)**

    Testes estatisticos para verificar se as diferencas entre modelos sao reais

- :material-chart-line: **[Graficos de Previsao](../visualization/forecast-plots.md)**

    Visualize previsoes com fan charts, residuos e decomposicao

- :material-book-open-variant: **[User Guide: Auto-Forecast](../user-guide/auto-forecast/index.md)**

    Referencia completa dos modelos automaticos

- :material-school: **[Theory: Fundamentos](../theory/nowcasting-theory.md)**

    Fundamentos teoricos dos modelos de previsao

</div>
