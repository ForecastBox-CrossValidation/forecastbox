"""Pipeline dashboard and combination plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastbox.pipeline.pipeline import PipelineResults
from forecastbox.viz._style import format_axis, get_color_palette


def combination_weights_plot(
    weights: dict[str, float] | pd.Series | pd.DataFrame,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot combination weights as bar chart or stacked bars.

    Parameters
    ----------
    weights : dict, Series, or DataFrame
        Model weights. If DataFrame, rows are time periods (stacked bars).
    ax : plt.Axes or None
        Axes to plot on.
    title : str or None
        Plot title.

    Returns
    -------
    plt.Axes
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if isinstance(weights, dict):
        weights = pd.Series(weights)

    if isinstance(weights, pd.Series):
        colors = get_color_palette(len(weights))
        bars = ax.bar(range(len(weights)), weights.values, color=colors)
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels([str(k) for k in weights.index], rotation=45, ha="right")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9,
            )

    elif isinstance(weights, pd.DataFrame):
        # Time-varying weights - stacked bar
        colors = get_color_palette(len(weights.columns))
        weights.plot(kind="bar", stacked=True, ax=ax, color=colors)
        ax.set_xticklabels(
            [str(idx) for idx in weights.index], rotation=45, ha="right",
        )

    title = title or "Combination Weights"
    return format_axis(ax, title=title, xlabel="Model", ylabel="Weight")


def pipeline_dashboard(
    results: PipelineResults,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Create dashboard summary of pipeline results (2x2 grid).

    Parameters
    ----------
    results : PipelineResults
        Pipeline results to visualize.
    fig : plt.Figure or None
        Figure to use.

    Returns
    -------
    plt.Figure
        The matplotlib Figure.
    """
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        axes = fig.subplots(2, 2)

    # Panel 1: Forecasts
    ax1 = axes[0, 0]
    colors = get_color_palette(len(results.forecasts))
    for (name, fc), color in zip(results.forecasts.items(), colors, strict=False):
        x = fc.index if fc.index is not None else np.arange(fc.horizon)
        ax1.plot(x, fc.point, color=color, linewidth=1.5, label=name)
    if results.combination is not None:
        x = (
            results.combination.index
            if results.combination.index is not None
            else np.arange(results.combination.horizon)
        )
        ax1.plot(
            x, results.combination.point, color="black",
            linewidth=2.5, linestyle="--", label="Combined",
        )
    format_axis(ax1, title="Forecasts", ylabel="Value")

    # Panel 2: Accuracy metrics
    ax2 = axes[0, 1]
    if not results.evaluation.empty:
        metric_col = "rmse" if "rmse" in results.evaluation.columns else (
            results.evaluation.columns[0] if len(results.evaluation.columns) > 0 else None
        )
        if metric_col:
            ax2.bar(
                range(len(results.evaluation)),
                results.evaluation[metric_col].values,
                color=get_color_palette(len(results.evaluation)),
            )
            ax2.set_xticks(range(len(results.evaluation)))
            ax2.set_xticklabels(results.evaluation.index, rotation=45, ha="right")
            format_axis(
                ax2, title=f"Accuracy ({metric_col.upper()})",
                ylabel=metric_col.upper(), legend=False,
            )
    else:
        ax2.text(
            0.5, 0.5, "No evaluation data", ha="center", va="center",
            transform=ax2.transAxes,
        )

    # Panel 3: Combination weights (if available)
    ax3 = axes[1, 0]
    if results.combination is not None and len(results.forecasts) > 1:
        n_models = len(results.forecasts)
        equal_weights = {name: 1.0 / n_models for name in results.forecasts}
        colors_w = get_color_palette(n_models)
        ax3.bar(range(n_models), list(equal_weights.values()), color=colors_w)
        ax3.set_xticks(range(n_models))
        ax3.set_xticklabels(list(equal_weights.keys()), rotation=45, ha="right")
        format_axis(ax3, title="Combination Weights", ylabel="Weight", legend=False)
    else:
        ax3.text(
            0.5, 0.5, "No combination", ha="center", va="center",
            transform=ax3.transAxes,
        )

    # Panel 4: Execution time
    ax4 = axes[1, 1]
    if results.execution_time:
        steps = list(results.execution_time.keys())
        times = list(results.execution_time.values())
        colors_t = get_color_palette(len(steps))
        ax4.barh(range(len(steps)), times, color=colors_t)
        ax4.set_yticks(range(len(steps)))
        ax4.set_yticklabels(steps)
        format_axis(ax4, title="Execution Time", xlabel="Seconds", legend=False)
    else:
        ax4.text(
            0.5, 0.5, "No timing data", ha="center", va="center",
            transform=ax4.transAxes,
        )

    fig.suptitle("Pipeline Dashboard", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()

    return fig
