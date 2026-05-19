---
title: ModelZoo
description: Registro, extensao e gerenciamento de modelos de previsao com interface ForecastModel e suporte a modelos customizados.
---

# ModelZoo

O `ModelZoo` e o **registro central** de modelos de previsao do forecastbox. Ele
permite listar modelos disponiveis, registrar modelos customizados e integrar
modelos de ML (scikit-learn, LightGBM, etc.) ao ecossistema de auto-selecao.

---

## Registry Pattern

O ModelZoo usa o padrao **registry** — um dicionario global que mapeia nomes a
classes de modelos:

```python
from forecastbox.auto import ModelZoo

# Listar modelos registrados
print(ModelZoo.list())
```

```text
['arima', 'ets', 'var', 'theta', 'naive', 'seasonal_naive', 'drift']
```

```python
# Obter classe de um modelo
ARIMAClass = ModelZoo.get("arima")
model = ARIMAClass(seasonal=True, m=12)
model.fit(y)

# Registrar modelo customizado
ModelZoo.register("prophet", ProphetWrapper)
```

### API do ModelZoo

| Metodo | Descricao |
|:-------|:----------|
| `ModelZoo.list()` | Lista nomes de todos os modelos registrados |
| `ModelZoo.get(name)` | Retorna a classe do modelo pelo nome |
| `ModelZoo.register(name, cls)` | Registra um novo modelo |
| `ModelZoo.unregister(name)` | Remove um modelo do registro |
| `ModelZoo.info(name)` | Retorna metadados do modelo (tipo, parametros, etc.) |

---

## Interface ForecastModel

Todo modelo no ModelZoo deve implementar a interface `ForecastModel`:

```python
from forecastbox.base import ForecastModel, Forecast, ModelSummary

class ForecastModel:
    """Interface base para modelos de previsao."""

    def fit(self, y, X=None):
        """
        Ajusta o modelo aos dados.

        Parameters
        ----------
        y : pd.Series
            Serie temporal (indice DatetimeIndex).
        X : pd.DataFrame, optional
            Variaveis exogenas.

        Returns
        -------
        self
        """
        ...

    def predict(self, horizon, X_future=None):
        """
        Gera previsoes h passos a frente.

        Parameters
        ----------
        horizon : int
            Numero de periodos a prever.
        X_future : pd.DataFrame, optional
            Variaveis exogenas futuras.

        Returns
        -------
        Forecast
            Objeto com point, lo, hi e metadados.
        """
        ...

    def summary(self):
        """
        Resumo do modelo ajustado.

        Returns
        -------
        ModelSummary
            Objeto com coeficientes, metricas e diagnosticos.
        """
        ...
```

### Objetos de Retorno

#### `Forecast`

```python
forecast = model.predict(horizon=12, level=[80, 95])

# Atributos
forecast.point      # pd.Series — previsao pontual
forecast.lo80       # pd.Series — limite inferior 80%
forecast.hi80       # pd.Series — limite superior 80%
forecast.lo95       # pd.Series — limite inferior 95%
forecast.hi95       # pd.Series — limite superior 95%
forecast.to_frame() # pd.DataFrame — todos os intervalos
```

#### `ModelSummary`

```python
summary = model.summary()

# Atributos
summary.model_name   # str — ex: "ARIMA(1,1,1)(0,1,1)[12]"
summary.coefficients # pd.DataFrame — coeficientes, se, p-valor
summary.aic          # float — criterio de informacao
summary.bic          # float
summary.n_params     # int — numero de parametros
summary.residuals    # pd.Series — residuos do ajuste
```

---

## Modelos Built-in

O forecastbox inclui os seguintes modelos pre-registrados:

| Modelo | Classe | Tipo | Descricao |
|:-------|:-------|:-----|:----------|
| `arima` | `AutoARIMA` | Univariado | ARIMA sazonal com selecao automatica de ordens |
| `ets` | `AutoETS` | Univariado | Suavizacao exponencial com selecao de componentes |
| `var` | `AutoVAR` | Multivariado | Vetores autoregressivos com selecao de ordem |
| `theta` | `ThetaModel` | Univariado | Metodo Theta (decomposicao em linhas theta) |
| `naive` | `NaiveModel` | Univariado | Ultimo valor observado como previsao |
| `seasonal_naive` | `SeasonalNaive` | Univariado | Ultimo ciclo sazonal como previsao |
| `drift` | `DriftModel` | Univariado | Tendencia linear entre primeiro e ultimo valor |

### Uso Rapido

```python
from forecastbox.auto import ModelZoo

# Ajustar qualquer modelo por nome
model = ModelZoo.get("theta")()
model.fit(y)
forecast = model.predict(horizon=12)

# Comparar todos os built-in
from forecastbox.auto import AutoSelect
selector = AutoSelect(models=ModelZoo.list())
selector.fit(y)
```

---

## Criando um Modelo Customizado

### Exemplo: Prophet Wrapper

Para integrar o [Prophet](https://facebook.github.io/prophet/) ao forecastbox, basta
criar uma classe que implemente `ForecastModel`:

```python
import pandas as pd
from prophet import Prophet
from forecastbox.base import ForecastModel, Forecast, ModelSummary


class ProphetWrapper(ForecastModel):
    """Wrapper do Prophet para uso no forecastbox."""

    def __init__(self, yearly_seasonality=True, weekly_seasonality=False,
                 changepoint_prior_scale=0.05):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self._model = None
        self._fitted = False

    def fit(self, y, X=None):
        # Prophet espera DataFrame com colunas 'ds' e 'y'
        df = pd.DataFrame({
            "ds": y.index,
            "y": y.values,
        })

        # Adicionar regressores exogenos
        if X is not None:
            for col in X.columns:
                df[col] = X[col].values

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        if X is not None:
            for col in X.columns:
                self._model.add_regressor(col)

        self._model.fit(df)
        self._fitted = True
        self._y = y
        return self

    def predict(self, horizon, X_future=None, level=None):
        future = self._model.make_future_dataframe(periods=horizon, freq=self._y.index.freq)

        if X_future is not None:
            for col in X_future.columns:
                future[col] = pd.concat([
                    pd.Series(self._y.index.map(lambda x: None)),
                    X_future[col],
                ]).values

        pred = self._model.predict(future).tail(horizon)

        return Forecast(
            point=pd.Series(pred["yhat"].values, index=pred["ds"]),
            lo95=pd.Series(pred["yhat_lower"].values, index=pred["ds"]),
            hi95=pd.Series(pred["yhat_upper"].values, index=pred["ds"]),
        )

    def summary(self):
        return ModelSummary(
            model_name="Prophet",
            n_params=len(self._model.params.get("delta", [])) + 4,
        )
```

### Registrar e Usar

```python
from forecastbox.auto import ModelZoo, AutoSelect

# Registrar
ModelZoo.register("prophet", ProphetWrapper)

# Verificar
print(ModelZoo.list())
# ['arima', 'ets', 'var', 'theta', 'naive', 'seasonal_naive', 'drift', 'prophet']

# Usar no AutoSelect
model = AutoSelect(
    models=["arima", "ets", "prophet"],
    strategy="best",
    cv_folds=5,
)
model.fit(y)
print(f"Melhor: {model.best_model_}")
```

---

## Registrando Modelos de ML

### Exemplo: LightGBM para Series Temporais

Modelos de ML precisam de feature engineering para transformar a serie temporal em
um problema supervisionado. O forecastbox fornece o mixin `MLForecastMixin` para
facilitar:

```python
import lightgbm as lgb
from forecastbox.base import ForecastModel, Forecast, MLForecastMixin


class LightGBMForecast(MLForecastMixin, ForecastModel):
    """LightGBM para previsao de series temporais."""

    def __init__(self, n_lags=12, n_estimators=100, learning_rate=0.1):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, y, X=None):
        # Criar features de lag
        features, target = self._create_lag_features(y, n_lags=self.n_lags)

        # Adicionar exogenas
        if X is not None:
            features = features.join(X)

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            verbose=-1,
        )
        self._model.fit(features, target)
        self._y = y
        self._fitted = True
        return self

    def predict(self, horizon, X_future=None):
        # Previsao recursiva (um passo de cada vez)
        predictions = self._recursive_predict(
            model=self._model,
            y=self._y,
            horizon=horizon,
            n_lags=self.n_lags,
            X_future=X_future,
        )
        return Forecast(point=predictions)

    def summary(self):
        return ModelSummary(
            model_name=f"LightGBM(lags={self.n_lags})",
            n_params=self._model.n_features_,
        )
```

```python
# Registrar e comparar com modelos estatisticos
ModelZoo.register("lightgbm", LightGBMForecast)

model = AutoSelect(
    models=["arima", "ets", "lightgbm"],
    metric="rmse",
    cv_folds=5,
)
model.fit(y)
```

```text
AutoSelect Summary
==================
Ranking:
  #   Model         RMSE    MAE     Params
  1   AutoETS       0.342   0.271     14
  2   LightGBM      0.358   0.285    100
  3   AutoARIMA     0.348   0.279      7
```

!!! warning "ML vs Estatisticos"

    Modelos de ML (LightGBM, XGBoost, Random Forest) frequentemente nao superam
    modelos estatisticos (ARIMA, ETS) em series temporais univariadas, especialmente
    com poucos dados. Eles brilham quando ha muitas features exogenas ou padroes
    nao lineares complexos.

### Exemplo: scikit-learn

```python
from sklearn.ensemble import RandomForestRegressor
from forecastbox.base import ForecastModel, Forecast, MLForecastMixin


class RandomForestForecast(MLForecastMixin, ForecastModel):
    """Random Forest para previsao de series temporais."""

    def __init__(self, n_lags=12, n_estimators=200, max_depth=10):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, y, X=None):
        features, target = self._create_lag_features(y, n_lags=self.n_lags)
        if X is not None:
            features = features.join(X)

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42,
        )
        self._model.fit(features, target)
        self._y = y
        self._fitted = True
        return self

    def predict(self, horizon, X_future=None):
        predictions = self._recursive_predict(
            model=self._model,
            y=self._y,
            horizon=horizon,
            n_lags=self.n_lags,
            X_future=X_future,
        )
        return Forecast(point=predictions)

    def summary(self):
        return ModelSummary(
            model_name=f"RandomForest(lags={self.n_lags})",
            n_params=self._model.n_features_in_,
        )


# Registrar
ModelZoo.register("random_forest", RandomForestForecast)
```

---

## Boas Praticas

!!! tip "Dicas para Modelos Customizados"

    1. **Sempre implemente `fit`, `predict` e `summary`** — o AutoSelect depende
       dos tres metodos
    2. **Retorne `self` em `fit`** — permite encadeamento (`model.fit(y).predict(12)`)
    3. **Use `Forecast` para retorno** — garante compatibilidade com combinacao e avaliacao
    4. **Trate `X=None`** — o modelo deve funcionar sem exogenas
    5. **Defina `n_params`** — usado pela estrategia `parsimonious` do AutoSelect

!!! info "Persistencia de Modelos"

    Modelos registrados no ModelZoo sao mantidos em memoria durante a sessao.
    Para registrar modelos permanentemente, adicione a chamada `ModelZoo.register()`
    em um modulo de inicializacao ou use o plugin de configuracao:

    ```python
    # forecastbox_config.py
    from forecastbox.auto import ModelZoo
    from mypackage.models import ProphetWrapper, LightGBMForecast

    ModelZoo.register("prophet", ProphetWrapper)
    ModelZoo.register("lightgbm", LightGBMForecast)
    ```

---

## Proximos Passos

- **[AutoSelect](auto-select.md)** — use o ModelZoo com selecao automatica
- **[AutoARIMA](auto-arima.md)** — detalhes do modelo ARIMA built-in
- **[AutoETS](auto-ets.md)** — detalhes do modelo ETS built-in
- **[AutoVAR](auto-var.md)** — detalhes do modelo VAR built-in
