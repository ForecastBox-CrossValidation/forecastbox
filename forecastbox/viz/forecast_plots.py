"""Forecast visualization plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from forecastbox.core.forecast import Forecast
from forecastbox.viz._style import NODESECON_COLORS, format_axis, get_color_palette


def forecast_plot(
    forecast: Forecast,
    actual: np.ndarray | pd.Series | None = None,
    history: pd.Series | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    show_intervals: bool = True,
) -> Axes:
    """Plot forecast with history and confidence intervals.

    Parameters
    ----------
    forecast : Forecast
        Forecast to plot.
    actual : array-like or None
        Actual realized values.
    history : pd.Series or None
        Historical data to show before the forecast.
    ax : Axes or None
        Axes to plot on. Creates new figure if None.
    title : str or None
        Plot title.
    show_intervals : bool
        Whether to show prediction intervals.

    Returns
    -------
    Axes
        The matplotlib Axes.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 6))  # type: ignore[reportUnknownMemberType]

    x = forecast.index if forecast.index is not None else np.arange(forecast.horizon)

    # Plot history
    if history is not None:
        ax.plot(  # type: ignore[reportUnknownMemberType]
            history.index, history.values,  # type: ignore[reportArgumentType]
            color="#888888",
            linewidth=1.5, label="History",
        )

    # Plot point forecast
    ax.plot(  # type: ignore[reportUnknownMemberType]
        x, forecast.point, color=NODESECON_COLORS["primary"],
        linewidth=2.5, label=f"Forecast ({forecast.model_name})",
    )

    # Plot intervals
    if show_intervals:
        if forecast.lower_95 is not None and forecast.upper_95 is not None:
            ax.fill_between(  # type: ignore[reportUnknownMemberType]
                x, forecast.lower_95, forecast.upper_95,
                alpha=0.12, color=NODESECON_COLORS["secondary"], label="95% CI",
            )
        if forecast.lower_80 is not None and forecast.upper_80 is not None:
            ax.fill_between(  # type: ignore[reportUnknownMemberType]
                x, forecast.lower_80, forecast.upper_80,
                alpha=0.25, color=NODESECON_COLORS["secondary"], label="80% CI",
            )

    # Plot actuals
    if actual is not None:
        actual_arr = np.asarray(actual)
        actual_x = x[:len(actual_arr)]
        ax.plot(  # type: ignore[reportUnknownMemberType]
            actual_x, actual_arr, "o", color=NODESECON_COLORS["danger"],
            markersize=6, label="Actual", zorder=5,
        )

    title = title or f"Forecast: {forecast.model_name}"
    return format_axis(ax, title=title, ylabel="Value")


def fan_chart(
    forecast: Forecast,
    history: pd.Series | None = None,
    history_periods: int = 36,
    n_levels: int = 5,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Create fan chart (Bank of England style) with graduated shading.

    Parameters
    ----------
    forecast : Forecast
        Forecast to plot.
    history : pd.Series or None
        Historical data.
    history_periods : int
        Number of history periods to show.
    n_levels : int
        Number of fan levels (shading gradations).
    ax : Axes or None
        Axes to plot on.
    title : str or None
        Plot title.

    Returns
    -------
    Axes
        The matplotlib Axes.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(14, 7))  # type: ignore[reportUnknownMemberType]

    x = forecast.index if forecast.index is not None else np.arange(forecast.horizon)

    # Plot history
    if history is not None:
        hist = history.tail(history_periods)
        ax.plot(  # type: ignore[reportUnknownMemberType]
            hist.index, hist.values,  # type: ignore[reportArgumentType]
            color="#555555", linewidth=1.5,
        )

    # Create fan levels
    if forecast.lower_95 is not None and forecast.upper_95 is not None:
        point = forecast.point
        lower = forecast.lower_95
        upper = forecast.upper_95

        for i in range(n_levels, 0, -1):
            frac = i / n_levels
            fan_lower = point - frac * (point - lower)
            fan_upper = point + frac * (upper - point)
            alpha = 0.08 + (n_levels - i) * 0.06
            ax.fill_between(  # type: ignore[reportUnknownMemberType]
                x, fan_lower, fan_upper,
                alpha=alpha, color=NODESECON_COLORS["secondary"],
            )

    # Median line
    ax.plot(  # type: ignore[reportUnknownMemberType]
        x, forecast.point, color=NODESECON_COLORS["primary"],
        linewidth=2.5, label="Median",
    )

    title = title or f"Fan Chart: {forecast.model_name}"
    return format_axis(ax, title=title, ylabel="Value")


def comparison_plot(
    forecasts: dict[str, Forecast],
    actual: np.ndarray | pd.Series | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot multiple model forecasts overlaid.

    Parameters
    ----------
    forecasts : dict[str, Forecast]
        Dictionary of model_name -> Forecast.
    actual : array-like or None
        Actual realized values.
    ax : Axes or None
        Axes to plot on.
    title : str or None
        Plot title.

    Returns
    -------
    Axes
        The matplotlib Axes.
    """
    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 6))  # type: ignore[reportUnknownMemberType]

    colors = get_color_palette(len(forecasts))

    for (name, fc), color in zip(forecasts.items(), colors, strict=False):
        x = fc.index if fc.index is not None else np.arange(fc.horizon)
        ax.plot(x, fc.point, color=color, linewidth=2, label=name)  # type: ignore[reportUnknownMemberType]

    # Plot actuals
    if actual is not None:
        actual_arr = np.asarray(actual)
        first_fc = next(iter(forecasts.values()))
        x = first_fc.index if first_fc.index is not None else np.arange(first_fc.horizon)
        actual_x = x[:len(actual_arr)]
        ax.plot(actual_x, actual_arr, "ko", markersize=6, label="Actual", zorder=5)  # type: ignore[reportUnknownMemberType]

    title = title or "Model Comparison"
    return format_axis(ax, title=title, ylabel="Value")
