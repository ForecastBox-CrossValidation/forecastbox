"""Evaluation and diagnostic plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure

from forecastbox.viz._style import NODESECON_COLORS, format_axis, get_color_palette


def accuracy_plot(
    evaluation: pd.DataFrame,
    metric: str = "rmse",
    by: str = "model",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot accuracy metrics by model or horizon.

    Parameters
    ----------
    evaluation : pd.DataFrame
        Evaluation DataFrame with models as index and metrics as columns.
    metric : str
        Metric to plot: 'rmse', 'mae', 'mape'.
    by : str
        Grouping: 'model' for bar chart, 'horizon' for line chart.
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
        _, ax = plt.subplots(figsize=(10, 6))  # type: ignore[reportUnknownMemberType]

    if metric not in evaluation.columns:
        available = list(evaluation.columns)
        metric = available[0] if available else metric

    if by == "model" and metric in evaluation.columns:
        colors = get_color_palette(len(evaluation))
        bars: BarContainer = ax.bar(range(len(evaluation)), evaluation[metric].to_numpy(), color=colors)  # type: ignore[reportUnknownMemberType]
        ax.set_xticks(range(len(evaluation)))  # type: ignore[reportUnknownMemberType]
        ax.set_xticklabels(evaluation.index, rotation=45, ha="right")  # type: ignore[reportUnknownMemberType]

        # Add value labels on bars
        for bar in bars.patches:
            height: float = bar.get_height()
            x_pos: float = bar.get_x() + bar.get_width() / 2.0
            ax.text(  # type: ignore[reportUnknownMemberType]
                x_pos, height,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9,
            )

    elif by == "horizon":
        # Plot metric by horizon (if data structured that way)
        ax.plot(  # type: ignore[reportUnknownMemberType]
            evaluation.index, evaluation[metric].to_numpy(),
            color=NODESECON_COLORS["primary"], marker="o", linewidth=2,
        )

    title = title or f"Accuracy: {metric.upper()} by {by.title()}"
    return format_axis(ax, title=title, xlabel=by.title(), ylabel=metric.upper())


def cv_plot(
    cv_results: dict[str, object],
    metric: str = "rmse",
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot cross-validation results.

    Parameters
    ----------
    cv_results : dict[str, Any]
        CV results per model.
    metric : str
        Metric to display.
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
        _, ax = plt.subplots(figsize=(10, 6))  # type: ignore[reportUnknownMemberType]

    colors = get_color_palette(len(cv_results))
    model_names: list[str] = []
    metric_values: list[float] = []

    for name, result in cv_results.items():
        model_names.append(name)
        if isinstance(result, dict):
            result_dict: dict[str, object] = result  # type: ignore[reportUnknownVariableType]
            value = result_dict.get(f"mean_{metric}", 0.0)
            metric_values.append(float(value) if isinstance(value, (int, float)) else 0.0)
        else:
            metric_values.append(0.0)

    if model_names:
        bars: BarContainer = ax.bar(  # type: ignore[reportUnknownMemberType]
            range(len(model_names)), metric_values,
            color=colors[:len(model_names)],
        )
        ax.set_xticks(range(len(model_names)))  # type: ignore[reportUnknownMemberType]
        ax.set_xticklabels(model_names, rotation=45, ha="right")  # type: ignore[reportUnknownMemberType]

        for bar in bars.patches:
            height: float = bar.get_height()
            x_pos: float = bar.get_x() + bar.get_width() / 2.0
            ax.text(  # type: ignore[reportUnknownMemberType]
                x_pos, height,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9,
            )

    title = title or f"Cross-Validation: {metric.upper()}"
    return format_axis(ax, title=title, xlabel="Model", ylabel=metric.upper())


def residual_plot(
    residuals: np.ndarray | pd.Series,
    model_name: str = "",
    fig: Figure | None = None,
) -> Figure:
    """Create 4-panel residual diagnostic plot.

    Panels: (1) Residuals vs time, (2) Histogram, (3) ACF, (4) QQ-plot.

    Parameters
    ----------
    residuals : array-like
        Residual values.
    model_name : str
        Model name for title.
    fig : plt.Figure or None
        Figure to use. Creates new 2x2 figure if None.

    Returns
    -------
    plt.Figure
        The matplotlib Figure.
    """
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # type: ignore[reportUnknownMemberType]
    else:
        axes = fig.subplots(2, 2)

    residuals_arr: npt.NDArray[np.float64] = np.asarray(residuals, dtype=np.float64)
    n = len(residuals_arr)

    # Panel 1: Residuals vs time
    ax1 = axes[0, 0]
    ax1.plot(
        range(n), residuals_arr, color=NODESECON_COLORS["primary"],
        linewidth=1, marker=".", markersize=3,
    )
    ax1.axhline(y=0, color=NODESECON_COLORS["danger"], linestyle="--", linewidth=1)
    format_axis(ax1, title="Residuals vs Time", xlabel="Time", ylabel="Residual", legend=False)

    # Panel 2: Histogram
    ax2 = axes[0, 1]
    n_bins = min(30, max(10, n // 5))
    ax2.hist(
        residuals_arr, bins=n_bins,
        color=NODESECON_COLORS["secondary"], alpha=0.7, edgecolor="white",
    )
    # Overlay normal curve
    x_range: npt.NDArray[np.float64] = np.linspace(
        float(residuals_arr.min()), float(residuals_arr.max()), 100,
    )
    std = np.std(residuals_arr)
    mean = np.mean(residuals_arr)
    if std > 0:
        normal = (
            (1 / (std * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
        )
        # Scale to match histogram
        bin_width = (residuals_arr.max() - residuals_arr.min()) / n_bins
        ax2.plot(x_range, normal * n * bin_width, color=NODESECON_COLORS["danger"], linewidth=2)
    format_axis(ax2, title="Distribution", xlabel="Residual", ylabel="Count", legend=False)

    # Panel 3: ACF
    ax3 = axes[1, 0]
    max_lags = min(40, n // 2 - 1)
    if max_lags > 0:
        acf_values: list[float] = []
        for lag in range(max_lags + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                r = np.corrcoef(residuals_arr[lag:], residuals_arr[:-lag])[0, 1]
                acf_values.append(float(r) if not np.isnan(r) else 0.0)

        ax3.bar(
            range(len(acf_values)), acf_values,
            color=NODESECON_COLORS["secondary"], width=0.3,
        )
        # Confidence bands
        ci = 1.96 / np.sqrt(n)
        ax3.axhline(y=ci, color=NODESECON_COLORS["danger"], linestyle="--", linewidth=0.8)
        ax3.axhline(y=-ci, color=NODESECON_COLORS["danger"], linestyle="--", linewidth=0.8)
        ax3.axhline(y=0, color="black", linewidth=0.5)
    format_axis(ax3, title="Autocorrelation (ACF)", xlabel="Lag", ylabel="ACF", legend=False)

    # Panel 4: QQ-plot
    ax4 = axes[1, 1]
    sorted_res = np.sort(residuals_arr)
    theoretical_q = (
        np.sort(np.random.default_rng(42).normal(mean, std, n))
        if std > 0
        else np.zeros(n)
    )
    ax4.scatter(
        theoretical_q, sorted_res, s=15,
        color=NODESECON_COLORS["secondary"], alpha=0.6,
    )
    # Reference line
    min_val = min(theoretical_q.min(), sorted_res.min())
    max_val = max(theoretical_q.max(), sorted_res.max())
    ax4.plot(
        [min_val, max_val], [min_val, max_val],
        color=NODESECON_COLORS["danger"], linewidth=1.5, linestyle="--",
    )
    format_axis(
        ax4, title="QQ-Plot", xlabel="Theoretical Quantiles",
        ylabel="Sample Quantiles", legend=False,
    )

    fig.suptitle(f"Residual Diagnostics: {model_name}", fontsize=16, fontweight="bold", y=1.02)  # type: ignore[reportUnknownMemberType]
    fig.tight_layout()

    return fig
