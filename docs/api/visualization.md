---
title: "Visualization API"
description: "API reference for forecastbox.plot — forecast plots, comparison charts, weight evolution, diagnostics, and theming"
---

# Visualization API Reference

!!! info "Module"
    **Import**: `from forecastbox.plot import plot_forecast, plot_comparison, plot_weights, plot_weights_evolution, plot_dm_test, plot_mcs_inclusion, plot_nowcast_evolution, plot_news_waterfall, plot_fan_chart, plot_residuals, plot_calibration`
    **Import**: `from forecastbox.plot import Theme, set_theme, get_theme, reset_theme`
    **Source**: `forecastbox/plot/`

## Overview

The visualization module provides publication-ready plots for all forecastbox workflows:

- **Forecast plots** — Point forecasts with prediction intervals and fan charts
- **Comparison plots** — Side-by-side model comparison by metric
- **Combination plots** — Weight distributions and evolution over time
- **Diagnostic plots** — Residuals, calibration, and statistical test results
- **Nowcasting plots** — Nowcast evolution and news decomposition
- **Theming** — Consistent styling via `Theme`, `set_theme()`, `get_theme()`, `reset_theme()`

| Function | Category | Description |
|----------|----------|-------------|
| [`plot_forecast()`](#plot_forecast) | Forecast | Point forecast with intervals |
| [`plot_comparison()`](#plot_comparison) | Comparison | Multi-model metric comparison |
| [`plot_weights()`](#plot_weights) | Combination | Weight bar chart |
| [`plot_weights_evolution()`](#plot_weights_evolution) | Combination | Weights over time |
| [`plot_dm_test()`](#plot_dm_test) | Diagnostic | Diebold-Mariano test visualization |
| [`plot_mcs_inclusion()`](#plot_mcs_inclusion) | Diagnostic | Model Confidence Set inclusion |
| [`plot_nowcast_evolution()`](#plot_nowcast_evolution) | Nowcasting | Nowcast revision trajectory |
| [`plot_news_waterfall()`](#plot_news_waterfall) | Nowcasting | News decomposition waterfall |
| [`plot_fan_chart()`](#plot_fan_chart) | Forecast | Density fan chart |
| [`plot_residuals()`](#plot_residuals) | Diagnostic | Residual analysis |
| [`plot_calibration()`](#plot_calibration) | Diagnostic | PIT calibration |

### Common Parameters

All plotting functions accept the following keyword arguments:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | `tuple[float, float]` | `(10, 6)` | Figure size in inches `(width, height)` |
| `backend` | `str` | `"matplotlib"` | Plotting backend: `"matplotlib"` or `"plotly"` |
| `style` | `str \| None` | `None` | Plot style override (uses current theme if `None`) |
| `save_path` | `str \| Path \| None` | `None` | If provided, save figure to this path (format inferred from extension) |
| `ax` | `matplotlib.axes.Axes \| None` | `None` | Existing axes to plot on (matplotlib only) |
| `title` | `str \| None` | `None` | Custom title (auto-generated if `None`) |

---

## plot_forecast()

Plot a single forecast with point predictions, prediction intervals, and optional actuals.

```python
plot_forecast(
    forecast: Forecast,
    actual: NDArray[np.float64] | pd.Series | None = None,
    show_intervals: bool = True,
    show_point: bool = True,
    interval_alpha: float = 0.3,
    actual_label: str = "Actual",
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | [`Forecast`](core.md#forecast) | *required* | Forecast object to plot |
| `actual` | `NDArray[np.float64] \| pd.Series \| None` | `None` | Realized values for comparison |
| `show_intervals` | `bool` | `True` | Show 80% and 95% prediction intervals |
| `show_point` | `bool` | `True` | Show point forecast line |
| `interval_alpha` | `float` | `0.3` | Transparency of interval shading |
| `actual_label` | `str` | `"Actual"` | Label for the actuals series in the legend |

**Returns**: `matplotlib.figure.Figure` or `plotly.graph_objects.Figure` depending on backend.

### Example

```python
from forecastbox.plot import plot_forecast

fig = plot_forecast(
    forecast=gdp_forecast,
    actual=y_test,
    show_intervals=True,
    figsize=(12, 6),
    title="GDP Forecast — ARIMA(1,1,1)",
    save_path="gdp_forecast.png",
)
```

---

## plot_comparison()

Bar or radar chart comparing multiple forecasts on a selected metric.

```python
plot_comparison(
    forecasts: dict[str, Forecast] | ForecastResults,
    metric: str = "rmse",
    actual: NDArray[np.float64] | pd.Series | None = None,
    chart_type: str = "bar",
    sort: bool = True,
    highlight_best: bool = True,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecasts` | `dict[str, Forecast] \| ForecastResults` | *required* | Named forecasts to compare |
| `metric` | `str` | `"rmse"` | Metric for comparison (`"rmse"`, `"mae"`, `"mape"`, `"mase"`) |
| `actual` | `NDArray[np.float64] \| pd.Series \| None` | `None` | Realized values (required if `forecasts` is a raw dict) |
| `chart_type` | `str` | `"bar"` | Chart type: `"bar"`, `"radar"`, or `"heatmap"` |
| `sort` | `bool` | `True` | Sort models by metric value |
| `highlight_best` | `bool` | `True` | Highlight the best-performing model |

**Returns**: `Figure`

### Example

```python
from forecastbox.plot import plot_comparison

fig = plot_comparison(
    forecasts={"ARIMA": f_arima, "ETS": f_ets, "VAR": f_var},
    metric="rmse",
    actual=y_test,
    chart_type="bar",
    title="Model Comparison — RMSE",
)
```

---

## plot_weights()

Bar chart of combination weights for each model in a forecast combination.

```python
plot_weights(
    combination: CombinationResult,
    top_n: int | None = None,
    show_values: bool = True,
    colormap: str = "Blues",
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `combination` | [`CombinationResult`](combination.md) | *required* | Fitted combination result containing model weights |
| `top_n` | `int \| None` | `None` | Show only the top N models by weight (all if `None`) |
| `show_values` | `bool` | `True` | Display numeric weight values on bars |
| `colormap` | `str` | `"Blues"` | Matplotlib colormap name |

**Returns**: `Figure`

### Example

```python
from forecastbox.plot import plot_weights

fig = plot_weights(
    combination=bma_result,
    top_n=10,
    title="BMA Weights — GDP Models",
)
```

---

## plot_weights_evolution()

Line or area chart showing how combination weights evolve over rolling windows or time.

```python
plot_weights_evolution(
    combination: CombinationResult,
    top_n: int | None = None,
    chart_type: str = "area",
    normalize: bool = True,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `combination` | [`CombinationResult`](combination.md) | *required* | Combination result with time-varying weights |
| `top_n` | `int \| None` | `None` | Show only the top N models (rest grouped as "Other") |
| `chart_type` | `str` | `"area"` | Chart type: `"area"` (stacked) or `"line"` |
| `normalize` | `bool` | `True` | Normalize weights to sum to 1 at each period |

**Returns**: `Figure`

### Example

```python
from forecastbox.plot import plot_weights_evolution

fig = plot_weights_evolution(
    combination=bma_result,
    top_n=5,
    chart_type="area",
    title="Weight Evolution — BMA Rolling",
    figsize=(14, 6),
)
```

---

## plot_dm_test()

Visualize the result of a Diebold-Mariano test showing the test statistic, critical values, and rejection region.

```python
plot_dm_test(
    result: DMTestResult,
    alpha: float = 0.05,
    show_distribution: bool = True,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | [`DMTestResult`](evaluation.md#diebold_mariano) | *required* | Result from `diebold_mariano()` |
| `alpha` | `float` | `0.05` | Significance level for critical values |
| `show_distribution` | `bool` | `True` | Show the null distribution curve |

**Returns**: `Figure`

### Example

```python
from forecastbox.evaluation import diebold_mariano
from forecastbox.plot import plot_dm_test

dm_result = diebold_mariano(actual, forecast_a, forecast_b)
fig = plot_dm_test(dm_result, alpha=0.05, title="DM Test: ARIMA vs ETS")
```

---

## plot_mcs_inclusion()

Horizontal bar chart showing Model Confidence Set inclusion with p-values for each model.

```python
plot_mcs_inclusion(
    mcs_result: MCSResult,
    alpha: float = 0.05,
    show_pvalues: bool = True,
    highlight_included: bool = True,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mcs_result` | [`MCSResult`](evaluation.md#model_confidence_set) | *required* | Result from `model_confidence_set()` |
| `alpha` | `float` | `0.05` | Significance level for inclusion threshold |
| `show_pvalues` | `bool` | `True` | Display p-values on bars |
| `highlight_included` | `bool` | `True` | Color-code included vs. excluded models |

**Returns**: `Figure`

### Example

```python
from forecastbox.evaluation import model_confidence_set
from forecastbox.plot import plot_mcs_inclusion

mcs = model_confidence_set(actual, forecasts, metric="mse")
fig = plot_mcs_inclusion(mcs, alpha=0.10, title="MCS Inclusion — 10% Level")
```

---

## plot_nowcast_evolution()

Track how a nowcast evolves as new data releases arrive within a quarter.

```python
plot_nowcast_evolution(
    nowcasts: list[Forecast] | dict[str, Forecast],
    target_actual: float | None = None,
    date_labels: list[str] | None = None,
    show_intervals: bool = True,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nowcasts` | `list[Forecast] \| dict[str, Forecast]` | *required* | Sequence of nowcasts ordered by information set |
| `target_actual` | `float \| None` | `None` | Realized value (shown as horizontal line) |
| `date_labels` | `list[str] \| None` | `None` | Labels for each nowcast vintage (auto-generated if `None`) |
| `show_intervals` | `bool` | `True` | Show prediction intervals around each nowcast |

**Returns**: `Figure`

### Example

```python
from forecastbox.plot import plot_nowcast_evolution

fig = plot_nowcast_evolution(
    nowcasts=[nc_jan, nc_feb, nc_mar],
    target_actual=0.8,
    date_labels=["Jan release", "Feb release", "Mar release"],
    title="GDP Nowcast Evolution — 2024Q1",
)
```

---

## plot_news_waterfall()

Waterfall chart decomposing the nowcast revision into contributions from each data release ("news").

```python
plot_news_waterfall(
    news: NewsDecomposition,
    top_n: int | None = None,
    show_values: bool = True,
    sort_by: str = "absolute",
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `news` | `NewsDecomposition` | *required* | News decomposition from `DFM.news()` or `nowcast_news()` |
| `top_n` | `int \| None` | `None` | Show only the top N contributors (rest grouped as "Other") |
| `show_values` | `bool` | `True` | Display numeric values on bars |
| `sort_by` | `str` | `"absolute"` | Sort order: `"absolute"` (largest impact) or `"contribution"` (positive first) |

**Returns**: `Figure`

### Example

```python
from forecastbox.plot import plot_news_waterfall

fig = plot_news_waterfall(
    news=dfm.news(vintage_old, vintage_new),
    top_n=10,
    title="Nowcast Revision Decomposition — 2024Q1",
)
```

---

## plot_fan_chart()

Fan chart showing the density forecast as layered quantile bands.

```python
plot_fan_chart(
    forecast: Forecast,
    quantiles: list[float] = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95],
    actual: NDArray[np.float64] | pd.Series | None = None,
    colormap: str = "Blues",
    central: str = "median",
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | [`Forecast`](core.md#forecast) | *required* | Forecast with density draws (`forecast.density` must not be `None`) |
| `quantiles` | `list[float]` | `[0.05, 0.10, 0.25, 0.75, 0.90, 0.95]` | Quantile levels for the bands (symmetric pairs) |
| `actual` | `NDArray[np.float64] \| pd.Series \| None` | `None` | Realized values |
| `colormap` | `str` | `"Blues"` | Colormap for the quantile bands |
| `central` | `str` | `"median"` | Central tendency line: `"median"` or `"mean"` |

**Returns**: `Figure`

!!! note
    The `forecast` object must contain density draws (shape `(H, N_draws)`). Use models with `method="sampling"` or density-capable combiners (BMA, BPS) to generate density forecasts.

### Example

```python
from forecastbox.plot import plot_fan_chart

fig = plot_fan_chart(
    forecast=bma_density_forecast,
    quantiles=[0.05, 0.25, 0.75, 0.95],
    actual=y_test,
    title="GDP Fan Chart — BMA Density",
    figsize=(12, 6),
)
```

---

## plot_residuals()

Multi-panel residual diagnostic plot: residuals over time, histogram, ACF, and Q-Q plot.

```python
plot_residuals(
    forecast: Forecast,
    actual: NDArray[np.float64] | pd.Series | None = None,
    lags: int = 20,
    panels: tuple[str, ...] = ("time", "histogram", "acf", "qq"),
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | [`Forecast`](core.md#forecast) | *required* | Forecast object (uses in-sample residuals if `actual` is `None`) |
| `actual` | `NDArray[np.float64] \| pd.Series \| None` | `None` | Realized values for computing residuals |
| `lags` | `int` | `20` | Number of lags for the ACF panel |
| `panels` | `tuple[str, ...]` | `("time", "histogram", "acf", "qq")` | Which panels to include |

**Returns**: `Figure` with a 2x2 subplot grid (or fewer if `panels` is customized).

### Example

```python
from forecastbox.plot import plot_residuals

fig = plot_residuals(
    forecast=arima_forecast,
    actual=y_test,
    lags=24,
    title="Residual Diagnostics — ARIMA(2,1,1)",
    save_path="residuals.png",
)
```

---

## plot_calibration()

PIT (Probability Integral Transform) histogram for assessing density calibration.

```python
plot_calibration(
    forecast: Forecast,
    actual: NDArray[np.float64] | pd.Series | None = None,
    n_bins: int = 10,
    show_confidence: bool = True,
    **kwargs,
) -> Figure
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | [`Forecast`](core.md#forecast) | *required* | Forecast with density draws |
| `actual` | `NDArray[np.float64] \| pd.Series \| None` | `None` | Realized values |
| `n_bins` | `int` | `10` | Number of histogram bins |
| `show_confidence` | `bool` | `True` | Show 95% confidence band for uniformity |

**Returns**: `Figure`

!!! tip
    A well-calibrated density forecast produces a uniform PIT histogram. Systematic deviations indicate biased location (U-shaped = underdispersed) or poor tail calibration (humped = overdispersed).

### Example

```python
from forecastbox.plot import plot_calibration

fig = plot_calibration(
    forecast=bma_forecast,
    actual=y_test,
    n_bins=20,
    title="PIT Calibration — BMA Density",
)
```

---

## Theming

### Theme

Class representing a plot theme with colors, fonts, and style parameters.

```python
Theme(
    name: str = "forecastbox",
    colors: dict[str, str] | None = None,
    font_family: str = "sans-serif",
    font_size: int = 12,
    grid: bool = True,
    grid_alpha: float = 0.3,
    spine_visible: bool = False,
    dpi: int = 150,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"forecastbox"` | Theme name |
| `colors` | `dict[str, str] \| None` | `None` | Color palette (`{"primary": "#1f77b4", "secondary": "#ff7f0e", ...}`) |
| `font_family` | `str` | `"sans-serif"` | Font family for labels and titles |
| `font_size` | `int` | `12` | Base font size in points |
| `grid` | `bool` | `True` | Show grid lines |
| `grid_alpha` | `float` | `0.3` | Grid transparency |
| `spine_visible` | `bool` | `False` | Show axis spines |
| `dpi` | `int` | `150` | Resolution for saved figures |

#### Built-in Themes

| Theme Name | Description |
|------------|-------------|
| `"forecastbox"` | Default clean theme with NodesEcon colors |
| `"minimal"` | Minimalist — no grid, no spines |
| `"publication"` | Serif fonts, high DPI, suitable for journals |
| `"presentation"` | Large fonts, bold colors, suitable for slides |
| `"dark"` | Dark background theme |

### set_theme()

Set the global plotting theme.

```python
set_theme(
    theme: str | Theme = "forecastbox",
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `theme` | `str \| Theme` | `"forecastbox"` | Theme name (built-in) or `Theme` instance |

### get_theme()

Return the current active theme.

```python
get_theme() -> Theme
```

**Returns**: `Theme` — the currently active theme.

### reset_theme()

Reset the global theme to the default (`"forecastbox"`).

```python
reset_theme() -> None
```

### Example — Theming

```python
from forecastbox.plot import set_theme, get_theme, reset_theme, Theme

# Use a built-in theme
set_theme("publication")

# Create a custom theme
custom = Theme(
    name="custom",
    colors={"primary": "#2c3e50", "secondary": "#e74c3c"},
    font_family="serif",
    font_size=14,
    dpi=300,
)
set_theme(custom)

# Check current theme
print(get_theme().name)
# custom

# Reset to default
reset_theme()
```

---

## See Also

- [Core API](core.md) — `Forecast` and `ForecastResults` objects used as inputs
- [Evaluation API](evaluation.md) — Statistical test results (`DMTestResult`, `MCSResult`)
- [Combination API](combination.md) — `CombinationResult` with weight data
- [Nowcasting API](nowcasting.md) — Nowcast objects and news decomposition
- [Reports API](reports.md) — Embed plots in automated reports
