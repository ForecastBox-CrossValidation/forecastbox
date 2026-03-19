"""Report sections for forecastbox reports."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastbox.pipeline.pipeline import PipelineResults


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _fig_to_path(fig: plt.Figure, path: str) -> str:
    """Save figure to file and return path."""
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def section_summary(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate executive summary section.

    Parameters
    ----------
    results : PipelineResults
        Pipeline results.

    Returns
    -------
    dict[str, Any]
        Section content with title, text, key_metrics.
    """
    best = results.best_model()
    n_models = len(results.forecasts)

    text_parts: list[str] = []
    text_parts.append(f"This report summarizes forecast results from {n_models} model(s).")

    if best:
        text_parts.append(f"The best performing model is **{best}**.")

    if results.combination is not None:
        text_parts.append(
            f"A combined forecast was generated using {results.combination.model_name}."
        )

    key_metrics: dict[str, float] = {}
    if not results.evaluation.empty and best:
        for col in results.evaluation.columns:
            val = results.evaluation.loc[best, col]
            if not pd.isna(val):
                key_metrics[col] = float(val)

    total_time = sum(results.execution_time.values()) if results.execution_time else 0.0

    return {
        "type": "summary",
        "title": kwargs.get("title", "Executive Summary"),
        "text": " ".join(text_parts),
        "best_model": best,
        "n_models": n_models,
        "key_metrics": key_metrics,
        "total_time": total_time,
    }


def section_data(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate data description section.

    Returns
    -------
    dict[str, Any]
        Section with data statistics.
    """
    data_length = results.metadata.get("data_length", 0)
    horizon = results.metadata.get("horizon", 0)

    return {
        "type": "data",
        "title": kwargs.get("title", "Data Description"),
        "text": (
            f"The analysis uses {data_length} observations"
            f" to forecast {horizon} periods ahead."
        ),
        "data_length": data_length,
        "horizon": horizon,
        "preprocess": results.metadata.get("preprocess", []),
    }


def section_models(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate models section.

    Returns
    -------
    dict[str, Any]
        Section with model specifications.
    """
    model_info: list[dict[str, Any]] = []
    for name, fc in results.forecasts.items():
        info: dict[str, Any] = {
            "name": name,
            "horizon": fc.horizon,
            "has_intervals": fc.lower_80 is not None,
            "metadata": fc.metadata,
        }
        model_info.append(info)

    return {
        "type": "models",
        "title": kwargs.get("title", "Models"),
        "text": f"{len(model_info)} model(s) were fitted and evaluated.",
        "models": model_info,
    }


def section_forecasts(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate forecasts section with tables.

    Returns
    -------
    dict[str, Any]
        Section with forecast tables.
    """
    tables: dict[str, pd.DataFrame] = {}
    for name, fc in results.forecasts.items():
        tables[name] = fc.to_dataframe()

    if results.combination is not None:
        tables["Combined"] = results.combination.to_dataframe()

    # Generate plot
    plot_base64 = None
    try:
        from forecastbox.viz.forecast_plots import comparison_plot
        fig, ax = plt.subplots(figsize=(12, 6))
        comparison_plot(results.forecasts, ax=ax)
        plot_base64 = _fig_to_base64(fig)
    except Exception:
        pass

    return {
        "type": "forecasts",
        "title": kwargs.get("title", "Forecasts"),
        "text": "Point forecasts and prediction intervals by model.",
        "tables": {name: df.to_dict() for name, df in tables.items()},
        "plot_base64": plot_base64,
    }


def section_evaluation(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate evaluation section.

    Returns
    -------
    dict[str, Any]
        Section with evaluation metrics.
    """
    eval_dict: dict[str, Any] = {}
    if not results.evaluation.empty:
        eval_dict = results.evaluation.to_dict()

    best = results.best_model()

    return {
        "type": "evaluation",
        "title": kwargs.get("title", "Evaluation"),
        "text": (
            f"Model evaluation results. Best model: {best}."
            if best
            else "Model evaluation results."
        ),
        "evaluation": eval_dict,
        "best_model": best,
        "cv_results": {
            name: {k: v for k, v in info.items() if not isinstance(v, (np.ndarray, pd.Series))}
            for name, info in results.cv_results.items()
        } if results.cv_results else {},
    }


def section_combination(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate combination section.

    Returns
    -------
    dict[str, Any]
        Section with combination details.
    """
    has_combination = results.combination is not None
    method = results.metadata.get("combination", None)

    weights: dict[str, float] = {}
    if has_combination and results.forecasts:
        n = len(results.forecasts)
        weights = {name: 1.0 / n for name in results.forecasts}

    return {
        "type": "combination",
        "title": kwargs.get("title", "Forecast Combination"),
        "text": f"Combination method: {method}." if method else "No combination applied.",
        "method": method,
        "weights": weights,
        "has_combination": has_combination,
    }


def section_scenarios(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate scenarios section.

    Returns
    -------
    dict[str, Any]
        Section with scenario details.
    """
    return {
        "type": "scenarios",
        "title": kwargs.get("title", "Scenarios"),
        "text": "Scenario analysis results.",
        "scenarios": {},
    }


def section_nowcast(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate nowcasting section.

    Returns
    -------
    dict[str, Any]
        Section with nowcast details.
    """
    return {
        "type": "nowcast",
        "title": kwargs.get("title", "Nowcasting"),
        "text": "Current period nowcast results.",
        "nowcast": {},
    }


def section_diagnostics(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate diagnostics section.

    Returns
    -------
    dict[str, Any]
        Section with diagnostic details.
    """
    diagnostics: dict[str, Any] = {}

    for name, fc in results.forecasts.items():
        residuals = fc.point - np.mean(fc.point)
        diagnostics[name] = {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "skewness": float(
                np.mean(((residuals - np.mean(residuals)) / max(np.std(residuals), 1e-10)) ** 3)
            ),
        }

    return {
        "type": "diagnostics",
        "title": kwargs.get("title", "Diagnostics"),
        "text": "Residual diagnostics and model checks.",
        "diagnostics": diagnostics,
    }


def section_appendix(results: PipelineResults, **kwargs: Any) -> dict[str, Any]:
    """Generate appendix section.

    Returns
    -------
    dict[str, Any]
        Section with detailed tables and parameters.
    """
    return {
        "type": "appendix",
        "title": kwargs.get("title", "Appendix"),
        "text": "Detailed tables, parameters, and technical notes.",
        "metadata": results.metadata,
        "execution_time": results.execution_time,
    }


# Registry of section generators
SECTION_REGISTRY: dict[str, Any] = {
    "summary": section_summary,
    "data": section_data,
    "models": section_models,
    "forecasts": section_forecasts,
    "evaluation": section_evaluation,
    "combination": section_combination,
    "scenarios": section_scenarios,
    "nowcast": section_nowcast,
    "diagnostics": section_diagnostics,
    "appendix": section_appendix,
}
