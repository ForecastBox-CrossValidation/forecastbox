# forecastbox

### Motor de previsao econometrica para series temporais

[![CI](https://github.com/ForecastBox-CrossValidation/forecastbox/actions/workflows/ci.yml/badge.svg)](https://github.com/ForecastBox-CrossValidation/forecastbox/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ForecastBox-CrossValidation/forecastbox/branch/main/graph/badge.svg)](https://codecov.io/gh/ForecastBox-CrossValidation/forecastbox)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://badge.fury.io/py/forecastbox.svg)](https://badge.fury.io/py/forecastbox)
[![Python versions](https://img.shields.io/pypi/pyversions/forecastbox)](https://pypi.org/project/forecastbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Development Status](https://img.shields.io/badge/development%20status-alpha-orange)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/forecastbox?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/forecastbox)
[![Documentation](https://readthedocs.org/projects/forecastbox/badge/?version=latest)](https://forecastbox.readthedocs.io/)

**forecastbox** e um motor de previsao econometrica para series temporais. Oferece
containers de forecast, auto-forecasting com selecao automatica de modelos, combinacao
de previsoes por multiplos metodos, testes estatisticos de avaliacao, nowcasting em
tempo real, cenarios condicionais, pipeline de producao com monitoramento, geracao de
relatorios e uma CLI completa.

## Instalacao

```bash
pip install forecastbox
```

Requer **Python >= 3.11**. A instalacao base inclui `numpy`, `pandas`, `scipy`,
`matplotlib` e `click`, cobrindo o nucleo da biblioteca: containers, metricas, avaliacao
estatistica (Diebold-Mariano, MCS, etc.), combinacao, cenarios, nowcasting, visualizacao
e CLI.

Alguns recursos avancados dependem de bibliotecas adicionais, disponiveis como extras:

```bash
pip install "forecastbox[auto]"     # AutoARIMA/ETS/VAR e stacking (statsmodels, scikit-learn)
pip install "forecastbox[reports]"  # relatorios via templates (jinja2)
pip install "forecastbox[theta]"    # modelo Theta via backend chronobox
pip install "forecastbox[all]"      # todos os extras acima
```

## Quick Start

```python
from forecastbox.auto import AutoARIMA
from forecastbox.metrics import mae, rmse
from forecastbox.datasets import load_dataset

# Carregar uma serie e separar treino/teste
data = load_dataset("macro_brazil")
y = data["ipca"]
train, test = y[:-12], y[-12:]

# Auto-forecast com selecao automatica de modelo
result = AutoARIMA(seasonal=True, m=12).fit(train)
fc = result.forecast(h=12)  # retorna um container Forecast

# Avaliar
print(f"MAE:  {mae(test.values, fc.point):.4f}")
print(f"RMSE: {rmse(test.values, fc.point):.4f}")
```

> O auto-forecasting (`AutoARIMA`, `AutoETS`, `AutoVAR`) requer o extra `[auto]`:
> `pip install "forecastbox[auto]"`.

Para comparar varios modelos de uma vez, use o `ForecastExperiment`:

```python
from forecastbox import ForecastExperiment

exp = ForecastExperiment(
    data=data,
    target="ipca",
    models=["auto_arima", "auto_ets", "theta"],
    combination="bma",
    horizon=12,
)
results = exp.run()
results.report("report.html")
```

## Principais features

| Modulo | Descricao |
|:-------|:----------|
| **Core** | Containers `Forecast`, `ForecastResults`, `ForecastHorizon` e `DataVintage` |
| **Auto-Forecasting** | Selecao automatica com `AutoARIMA`, `AutoETS`, `Theta`, `AutoVAR` e `AutoSelect` |
| **Combinacao** | 7 metodos: mean, median, inverse_mse, ols, bma, stacking e optimal |
| **Avaliacao** | Diebold-Mariano, Model Confidence Set, Giacomini-White, Mincer-Zarnowitz e encompassing |
| **Metricas** | MAE, RMSE, MAPE, MASE, CRPS e coverage |
| **Cenarios** | Previsoes condicionais, stress testing e fan charts |
| **Nowcasting** | Dynamic Factor Models (DFM), bridge equations, MIDAS e news decomposition |
| **Pipeline** | `ForecastPipeline` e `ForecastMonitor` com alertas de degradacao |
| **Visualizacao** | Graficos de previsao, comparacao e fan charts |
| **Relatorios** | Geracao de relatorios em HTML, Markdown e JSON |
| **CLI** | 5 comandos: `forecast`, `evaluate`, `nowcast`, `monitor` e `combine` |
| **Datasets** | 20 datasets embutidos e cross-validation (expanding/sliding window) |

## Documentacao

A documentacao completa esta disponivel em
**[nodesecon.github.io/forecastbox](https://nodesecon.github.io/forecastbox/)**,
incluindo getting started, user guides, tutoriais e referencia de API.

Consulte o [CHANGELOG](CHANGELOG.md) para o historico de versoes.

## Contribuindo

Contribuicoes sao bem-vindas. Veja o guia de contribuicao em
[docs/contributing](docs/contributing/) para instrucoes de setup, padroes de codigo e
processo de pull request.

## Licenca

Distribuido sob a licenca MIT. Veja o arquivo
[LICENSE](https://github.com/nodesecon/forecastbox/blob/main/LICENSE) para detalhes.
