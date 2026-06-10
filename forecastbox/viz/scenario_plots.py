"""Scenario visualization plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from forecastbox.core.forecast import Forecast
from forecastbox.viz._style import format_axis, get_color_palette


def scenario_plot(
    scenarios: dict[str, Forecast | np.ndarray],
    base_scenario: str | None = None,
    history: pd.Series | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot multiple scenarios overlaid.

    Parameters
    ----------
    scenarios : dict[str, Forecast or array-like]
        Dictionary of scenario_name -> Forecast or values.
    base_scenario : str or None
        Name of the base scenario to highlight.
    history : pd.Series or None
        Historical data to show.
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
        _, ax = plt.subplots(figsize=(12, 6))  # type: ignore[reportUnknownMemberType]

    # Plot history
    if history is not None:
        ax.plot(  # type: ignore[reportUnknownMemberType]
            history.index,
            history.values,  # type: ignore[reportArgumentType]
            color="#888888",
            linewidth=1.5,
            label="History",
        )

    colors = get_color_palette(len(scenarios))

    for (name, scenario), color in zip(scenarios.items(), colors, strict=False):
        if isinstance(scenario, Forecast):
            x = scenario.index if scenario.index is not None else np.arange(scenario.horizon)
            values = scenario.point
            linewidth = 3.0 if name == base_scenario else 1.8
            linestyle = "-" if name == base_scenario else "--"

            ax.plot(  # type: ignore[reportUnknownMemberType]
                x, values, color=color, linewidth=linewidth,
                linestyle=linestyle, label=name,
            )

            # Show interval for base scenario
            if (
                name == base_scenario
                and scenario.lower_95 is not None
                and scenario.upper_95 is not None
            ):
                ax.fill_between(  # type: ignore[reportUnknownMemberType]
                    x, scenario.lower_95, scenario.upper_95, alpha=0.1, color=color,
                )
        else:
            values = np.asarray(scenario)
            x_arr = np.arange(len(values))
            linewidth = 3.0 if name == base_scenario else 1.8
            ax.plot(x_arr, values, color=color, linewidth=linewidth, label=name)  # type: ignore[reportUnknownMemberType]

    title = title or "Scenario Comparison"
    return format_axis(ax, title=title, ylabel="Value")
