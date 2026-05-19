---
title: FAQ Geral
description: Perguntas frequentes sobre forecastbox, instalacao, dados e ecossistema NodesEcon
---

# FAQ Geral

Perguntas frequentes sobre o **forecastbox** e o ecossistema NodesEcon.

---

## O que e forecastbox e como se relaciona com o ecossistema NodesEcon?

O **forecastbox** e o motor de previsao econometrica do ecossistema **NodesEcon**. Ele fornece ferramentas para auto-forecast, combinacao de previsoes, avaliacao estatistica e nowcasting.

O ecossistema NodesEcon e composto por pacotes especializados:

| Pacote | Funcao |
|--------|--------|
| **chronobox** | Manipulacao de series temporais e calendarios |
| **kalmanbox** | Filtro de Kalman e modelos estado-espaco |
| **archbox** | Modelos GARCH e volatilidade |
| **panelbox** | Econometria de dados em painel |
| **forecastbox** | Previsao, combinacao e nowcasting |

O `forecastbox` utiliza `chronobox` como dependencia obrigatoria para manipulacao de datas e frequencias, e opcionalmente `kalmanbox` para modelos DFM no nowcasting.

---

## Qual a diferenca entre forecastbox e statsmodels/pmdarima?

O `forecastbox` nao reimplementa modelos estatisticos basicos — ele **orquestra** modelos existentes em workflows completos de previsao:

| Aspecto | statsmodels/pmdarima | forecastbox |
|---------|---------------------|-------------|
| **Foco** | Estimacao de modelos individuais | Workflow completo de previsao |
| **Auto-selecao** | `pmdarima.auto_arima` apenas | ARIMA, ETS, VAR, Theta, selecao automatica |
| **Combinacao** | Nao oferece | 7+ metodos (Simple, OLS, BMA, Stacking, ...) |
| **Avaliacao** | Metricas basicas | DM test, MCS, Giacomini-White, Mincer-Zarnowitz |
| **Nowcasting** | Nao oferece | DFM, Bridge, MIDAS |
| **Pipeline** | Manual | Automatizado com monitoramento |

O `forecastbox` usa `statsmodels` internamente para estimacao, adicionando camadas de automacao, combinacao e avaliacao.

---

## Preciso instalar kalmanbox para usar forecastbox?

**Nao para a maioria dos recursos.** O `kalmanbox` e uma dependencia **opcional**, necessaria apenas para:

- Modelos de Fatores Dinamicos (DFM) no nowcasting
- Pesos time-varying via filtro de Kalman na combinacao

Para instalar com suporte completo:

```bash
pip install forecastbox[kalman]
```

Ou instalar separadamente:

```bash
pip install kalmanbox
```

!!! tip "Verifique a instalacao"
    ```python
    import forecastbox as fb
    print(fb.check_dependencies())
    ```
    O comando lista quais dependencias opcionais estao disponiveis.

---

## forecastbox funciona com dados em painel?

O `forecastbox` e projetado para **series temporais univariadas e multivariadas**, nao para dados em painel. Para econometria de dados em painel, use o **panelbox**.

No entanto, voce pode usar `forecastbox` para prever **cada unidade do painel separadamente**:

```python
import pandas as pd
import forecastbox as fb

# Dados em painel: varias series
panel_data = pd.read_csv("dados_painel.csv")

results = {}
for entity in panel_data["entity"].unique():
    series = panel_data[panel_data["entity"] == entity]["y"]
    model = fb.AutoARIMA()
    model.fit(series)
    results[entity] = model.forecast(h=12)
```

!!! info "Pipeline para multiplas series"
    O `Pipeline` do forecastbox suporta processamento em lote de multiplas series via parametro `n_jobs` para paralelizacao.

---

## Posso usar modelos de ML (sklearn, lightgbm) no forecastbox?

Sim, atraves do **ModelZoo** voce pode registrar qualquer modelo que siga a interface `fit`/`predict`:

```python
import forecastbox as fb
from sklearn.ensemble import RandomForestRegressor

# Criar wrapper para sklearn
class RFForecaster(fb.BaseForecaster):
    def __init__(self, n_lags=12, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        self.n_lags = n_lags

    def fit(self, y, X=None):
        features, target = self._create_lags(y, self.n_lags)
        self.model.fit(features, target)
        return self

    def forecast(self, h=1, X=None):
        # Previsao iterativa
        return self._recursive_forecast(h)

# Registrar no ModelZoo
fb.ModelZoo.register("random_forest", RFForecaster)

# Usar como qualquer outro modelo
model = fb.AutoSelect(models=["arima", "ets", "random_forest"])
```

!!! warning "Cuidado com overfitting"
    Modelos de ML tendem a overfitting em series temporais curtas. Use cross-validation temporal (`TimeSeriesSplit`) e avalie com metricas out-of-sample.

---

## Como citar forecastbox em artigos academicos?

Use a seguinte referencia BibTeX:

```bibtex
@software{forecastbox2024,
  title     = {forecastbox: Motor de Previsao Econometrica para Python},
  author    = {NodesEcon},
  year      = {2024},
  url       = {https://github.com/nodesecon/forecastbox},
  version   = {0.1.0},
  note      = {Auto-forecast, combinacao de previsoes e nowcasting}
}
```

Ou em formato texto:

> NodesEcon (2024). *forecastbox: Motor de Previsao Econometrica para Python*. Disponivel em: https://github.com/nodesecon/forecastbox

---

## Quais formatos de dados sao suportados?

O `forecastbox` aceita dados nos seguintes formatos:

| Formato | Tipo Python | Exemplo |
|---------|-------------|---------|
| Series temporal | `pd.Series` com `DatetimeIndex` | Serie univariada |
| DataFrame | `pd.DataFrame` com `DatetimeIndex` | Multiplas series/exogenas |
| Array NumPy | `np.ndarray` | Sem indice temporal |
| Dicionario | `dict` de Series | Multiplas series nomeadas |

```python
import pandas as pd
import forecastbox as fb

# pd.Series com DatetimeIndex (recomendado)
y = pd.Series(
    [100, 102, 105, 103, 108],
    index=pd.date_range("2020-01", periods=5, freq="MS")
)

# DataFrame com multiplas colunas
df = pd.DataFrame({
    "pib": [100, 102, 105],
    "inflacao": [3.5, 3.2, 3.8]
}, index=pd.date_range("2020-01", periods=3, freq="QS"))
```

!!! note "Frequencias suportadas"
    O `forecastbox` via `chronobox` suporta todas as frequencias do pandas: diaria (`D`), semanal (`W`), mensal (`MS`), trimestral (`QS`), anual (`YS`) e frequencias customizadas.

---

## forecastbox suporta GPU?

**Nao nativamente.** O `forecastbox` utiliza NumPy e SciPy para computacao, que executam em CPU.

No entanto, se voce registrar modelos de ML no ModelZoo que usem GPU (como LightGBM com `device='gpu'` ou PyTorch), esses modelos individuais executarao em GPU:

```python
import forecastbox as fb
import lightgbm as lgb

class LGBMForecaster(fb.BaseForecaster):
    def __init__(self, n_lags=12):
        self.model = lgb.LGBMRegressor(device="gpu")
        self.n_lags = n_lags
    # ...
```

Para a maioria dos casos de uso em econometria, a CPU e suficiente. A paralelizacao via `n_jobs` e geralmente mais eficiente do que GPU para modelos estatisticos.

---

## Qual a licenca do forecastbox?

O `forecastbox` e distribuido sob a licenca **MIT**, que permite:

- Uso comercial e academico
- Modificacao e distribuicao
- Uso privado

A unica exigencia e manter o aviso de copyright original. Veja o arquivo `LICENSE` no repositorio.

---

## forecastbox suporta previsao probabilistica?

Sim. A maioria dos modelos no `forecastbox` suporta **intervalos de confianca** e **previsao probabilistica**:

```python
import forecastbox as fb

model = fb.AutoARIMA()
model.fit(y)

# Previsao pontual + intervalos
forecast = model.forecast(h=12, alpha=0.05)  # IC 95%
print(forecast.mean)       # Previsao pontual
print(forecast.lower)      # Limite inferior
print(forecast.upper)      # Limite superior

# Fan chart com multiplos quantis
fan = model.forecast(h=12, quantiles=[0.10, 0.25, 0.50, 0.75, 0.90])
```

Os metodos de combinacao tambem produzem intervalos combinados, e o `ScenarioBuilder` permite simulacoes de Monte Carlo para distribuicoes completas.

---

## Como atualizar o forecastbox?

=== "pip"

    ```bash
    pip install --upgrade forecastbox
    ```

=== "pip com extras"

    ```bash
    pip install --upgrade forecastbox[kalman]
    ```

=== "Verificar versao"

    ```python
    import forecastbox as fb
    print(fb.__version__)
    ```

!!! tip "Changelog"
    Consulte o [Changelog](../contributing/changelog.md) para ver as mudancas entre versoes.

---

## O forecastbox funciona com Python 3.13?

O `forecastbox` suporta **Python 3.10+**. A compatibilidade com versoes especificas depende das dependencias:

| Python | Status |
|--------|--------|
| 3.9 | Nao suportado |
| 3.10 | Suportado |
| 3.11 | Suportado |
| 3.12 | Suportado |
| 3.13 | Suportado |

!!! info "Recomendacao"
    Para melhor performance, use **Python 3.11+** que oferece melhorias significativas de velocidade no interpretador.
