"""NodesEcon visual style for forecastbox plots."""

from __future__ import annotations

import matplotlib.pyplot as plt

# NodesEcon color palette
NODESECON_COLORS: dict[str, str] = {
    "primary": "#1B3A5C",      # dark blue
    "secondary": "#2E86AB",    # medium blue
    "accent": "#A23B72",       # magenta
    "success": "#2CA58D",      # teal
    "warning": "#F18F01",      # orange
    "danger": "#C73E1D",       # red
    "info": "#5C8A97",         # slate blue
    "light": "#E8E8E8",        # light gray
    "dark": "#2D2D2D",         # dark gray
    "background": "#FAFAFA",   # off-white
}

NODESECON_PALETTE: list[str] = [
    "#1B3A5C",  # dark blue
    "#2E86AB",  # medium blue
    "#A23B72",  # magenta
    "#2CA58D",  # teal
    "#F18F01",  # orange
    "#C73E1D",  # red
    "#5C8A97",  # slate blue
    "#7B68EE",  # medium slate blue
    "#DAA520",  # goldenrod
    "#556B2F",  # dark olive green
]


def set_nodesecon_style() -> None:
    """Apply NodesEcon style to matplotlib globally."""
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6),
            "figure.dpi": 100,
            "figure.facecolor": NODESECON_COLORS["background"],
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.prop_cycle": plt.cycler("color", NODESECON_PALETTE),
            "grid.color": "#E0E0E0",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#CCCCCC",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def get_color_palette(n: int) -> list[str]:
    """Get n colors from the NodesEcon palette.

    Parameters
    ----------
    n : int
        Number of colors needed.

    Returns
    -------
    list[str]
        List of hex color strings.
    """
    if n <= len(NODESECON_PALETTE):
        return NODESECON_PALETTE[:n]

    # Cycle through palette for more colors
    colors: list[str] = []
    for i in range(n):
        colors.append(NODESECON_PALETTE[i % len(NODESECON_PALETTE)])
    return colors


def format_axis(
    ax: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
) -> plt.Axes:
    """Apply standard formatting to an axis.

    Parameters
    ----------
    ax : plt.Axes
        Axes to format.
    title : str or None
        Title for the axis.
    xlabel : str or None
        X-axis label.
    ylabel : str or None
        Y-axis label.
    legend : bool
        Whether to show legend.

    Returns
    -------
    plt.Axes
        Formatted axes.
    """
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if legend and ax.get_legend_handles_labels()[1]:
        ax.legend(framealpha=0.9, edgecolor="#CCCCCC")

    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax
