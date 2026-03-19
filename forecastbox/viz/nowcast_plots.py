"""Nowcasting visualization plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from forecastbox.viz._style import NODESECON_COLORS, format_axis


def nowcast_plot(
    nowcasts: dict[str, float] | pd.Series,
    actual: float | None = None,
    vintages: list[str] | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot nowcast evolution by vintage.

    Parameters
    ----------
    nowcasts : dict or pd.Series
        Nowcast values keyed by vintage date/label.
    actual : float or None
        Final actual value to show as reference.
    vintages : list[str] or None
        Specific vintages to highlight.
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
        _, ax = plt.subplots(figsize=(12, 6))

    nowcast_series = pd.Series(nowcasts) if isinstance(nowcasts, dict) else nowcasts

    # Plot nowcast evolution
    ax.plot(
        range(len(nowcast_series)), nowcast_series.values,
        color=NODESECON_COLORS["primary"], marker="o",
        linewidth=2, markersize=6, label="Nowcast",
    )

    # Set x labels
    ax.set_xticks(range(len(nowcast_series)))
    ax.set_xticklabels(
        [str(v) for v in nowcast_series.index],
        rotation=45, ha="right",
    )

    # Highlight specific vintages
    if vintages:
        for v in vintages:
            if v in nowcast_series.index:
                idx = list(nowcast_series.index).index(v)
                ax.plot(
                    idx, nowcast_series[v], "s",
                    color=NODESECON_COLORS["accent"], markersize=10, zorder=5,
                )

    # Plot actual
    if actual is not None:
        ax.axhline(
            y=actual, color=NODESECON_COLORS["danger"],
            linestyle="--", linewidth=2, label=f"Actual: {actual:.2f}",
        )

    title = title or "Nowcast Evolution by Vintage"
    return format_axis(ax, title=title, xlabel="Vintage", ylabel="Nowcast Value")
