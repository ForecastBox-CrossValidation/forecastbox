---
title: "Reports API"
description: "API reference for forecastbox.reports — ForecastReport, templates, and multi-format report generation (HTML, PDF, Excel, Markdown)"
---

# Reports API Reference

!!! info "Module"
    **Import**: `from forecastbox.reports import ForecastReport`
    **Source**: `forecastbox/reports/`

## Overview

The reports module generates publication-quality forecast reports in multiple formats:

- **`ForecastReport`** — Builder for composing structured reports with forecasts, evaluation results, scenarios, and custom sections
- **Templates** — Built-in templates: `summary`, `detailed`, `executive`, `technical`
- **Formats** — HTML, PDF (via weasyprint), Excel, Markdown

| Class / Function | Description |
|------------------|-------------|
| [`ForecastReport`](#forecastreport) | Main report builder |
| [Templates](#built-in-templates) | Pre-defined report layouts |

---

## ForecastReport

Builder class for composing and generating forecast reports. Sections are added sequentially and rendered into the chosen template and output format.

### Constructor

```python
ForecastReport(
    title: str = "Forecast Report",
    author: str = "",
    template: str = "summary",
    date: str | datetime | None = None,
    logo_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | `"Forecast Report"` | Report title (appears on cover page and header) |
| `author` | `str` | `""` | Author name |
| `template` | `str` | `"summary"` | Template name: `"summary"`, `"detailed"`, `"executive"`, `"technical"` |
| `date` | `str \| datetime \| None` | `None` | Report date (defaults to current date) |
| `logo_path` | `str \| Path \| None` | `None` | Path to logo image for the header |
| `metadata` | `dict[str, Any] \| None` | `None` | Additional metadata included in the report footer |

### Methods

##### `.add_section(title, content)`

Add a custom text section to the report.

```python
report.add_section(
    title: str,
    content: str,
    position: int | None = None,
) -> ForecastReport
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | *required* | Section heading |
| `content` | `str` | *required* | Section body (supports Markdown formatting) |
| `position` | `int \| None` | `None` | Insert position (appended at end if `None`) |

**Returns**: `ForecastReport` (self, for method chaining).

##### `.add_forecast(forecast, description)`

Add a forecast section with automatic plots and summary statistics.

```python
report.add_forecast(
    forecast: Forecast,
    description: str = "",
    show_intervals: bool = True,
    show_table: bool = True,
) -> ForecastReport
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `forecast` | [`Forecast`](core.md#forecast) | *required* | Forecast object to include |
| `description` | `str` | `""` | Narrative description of the forecast |
| `show_intervals` | `bool` | `True` | Include prediction interval chart |
| `show_table` | `bool` | `True` | Include point forecast table |

**Returns**: `ForecastReport`

##### `.add_evaluation(results, description)`

Add an evaluation section with metrics tables and comparison charts.

```python
report.add_evaluation(
    results: pd.DataFrame | EvaluationResult,
    description: str = "",
    highlight_best: bool = True,
) -> ForecastReport
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `results` | `pd.DataFrame \| EvaluationResult` | *required* | Evaluation results (from `evaluate()` or a metric DataFrame) |
| `description` | `str` | `""` | Narrative description of the evaluation |
| `highlight_best` | `bool` | `True` | Highlight best-performing model in the table |

**Returns**: `ForecastReport`

##### `.add_scenarios(scenarios, description)`

Add a scenario analysis section with conditional forecast paths.

```python
report.add_scenarios(
    scenarios: dict[str, Forecast] | ScenarioResult,
    description: str = "",
    show_comparison: bool = True,
) -> ForecastReport
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scenarios` | `dict[str, Forecast] \| ScenarioResult` | *required* | Named scenario forecasts |
| `description` | `str` | `""` | Narrative description of the scenario analysis |
| `show_comparison` | `bool` | `True` | Include a comparison chart across scenarios |

**Returns**: `ForecastReport`

##### `.generate(format, output_path)`

Render the report to the specified format and write to disk.

```python
report.generate(
    format: str = "html",
    output_path: str | Path | None = None,
) -> Path
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str` | `"html"` | Output format: `"html"`, `"pdf"`, `"excel"`, `"markdown"` |
| `output_path` | `str \| Path \| None` | `None` | Output file path (auto-generated from title if `None`) |

**Returns**: `Path` — path to the generated report file.

!!! warning
    PDF generation requires the `weasyprint` package. Install with `pip install forecastbox[reports]` or `pip install weasyprint`.

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `title` | `str` | Report title |
| `author` | `str` | Author name |
| `template` | `str` | Template name |
| `sections` | `list[ReportSection]` | Ordered list of report sections |
| `created_at` | `datetime` | Report creation timestamp |

---

## Built-in Templates

| Template | Description | Use Case |
|----------|-------------|----------|
| `"summary"` | One-page overview with key forecasts and headline metrics | Quick stakeholder updates |
| `"detailed"` | Full report with all sections, tables, and charts | Internal analysis documentation |
| `"executive"` | Non-technical summary with business-focused language | Board presentations, management reports |
| `"technical"` | Full model details, diagnostics, residuals, and statistical tests | Peer review, model validation |

!!! tip
    The `"executive"` template automatically simplifies chart labels, omits statistical test details, and uses plain-language summaries. Use `"technical"` when the audience is familiar with econometrics.

---

## Example — Executive GDP Forecast Report

```python
from forecastbox.reports import ForecastReport
from forecastbox.core import Forecast
from forecastbox.evaluation import evaluate

# Build the report
report = ForecastReport(
    title="GDP Forecast Report — 2024Q4",
    author="Economics Team",
    template="executive",
)

# Add sections
report.add_section(
    title="Summary",
    content="This report presents GDP growth forecasts for 2024Q4 "
            "using three competing models evaluated over the last 8 quarters.",
)

report.add_forecast(
    forecast=gdp_forecast,
    description="Baseline GDP growth forecast using BMA combination.",
)

report.add_evaluation(
    results=evaluate(y_test, forecasts, metrics=("rmse", "mae", "mape")),
    description="Out-of-sample evaluation over 2022Q4–2024Q3.",
)

report.add_scenarios(
    scenarios={"Baseline": f_base, "Stress": f_stress, "Optimistic": f_opt},
    description="Scenario analysis under alternative interest rate paths.",
)

# Generate
path = report.generate(format="pdf", output_path="gdp_report_2024q4.pdf")
print(f"Report saved to: {path}")
# Report saved to: gdp_report_2024q4.pdf
```

---

## Example — Markdown Report for Version Control

```python
report = ForecastReport(
    title="Monthly Inflation Forecast",
    template="detailed",
)

report.add_forecast(forecast=ipca_forecast)
report.add_evaluation(results=eval_df)

path = report.generate(format="markdown", output_path="reports/inflation.md")
```

!!! note
    Markdown reports embed plots as base64 images by default. Set `embed_images=False` in `.generate()` to save images as separate files and use relative links.

---

## See Also

- [Core API](core.md) — `Forecast` objects used as report inputs
- [Evaluation API](evaluation.md) — Evaluation results for `.add_evaluation()`
- [Scenarios API](scenarios.md) — Scenario results for `.add_scenarios()`
- [Visualization API](visualization.md) — Plot functions used internally by reports
