"""Helper functions for scenario notebooks."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data"


def load_us_macro_quarterly() -> pd.DataFrame:
    """Load quarterly US macroeconomic dataset."""
    df = pd.read_csv(DATA_DIR / "us_macro_quarterly.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def load_macro_brazil() -> pd.DataFrame:
    """Load Brazilian macroeconomic dataset."""
    df = pd.read_csv(DATA_DIR / "macro_brazil.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def plot_fan_chart(
    forecast_mean: np.ndarray,
    forecast_quantiles: dict[float, np.ndarray],
    dates: pd.DatetimeIndex | None = None,
    title: str = "Fan Chart",
    ylabel: str = "",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plot a fan chart with multiple confidence bands."""
    fig, ax = plt.subplots(figsize=figsize)
    x = dates if dates is not None else np.arange(len(forecast_mean))

    ax.plot(x, forecast_mean, color="navy", linewidth=2, label="Point forecast")

    colors = ["#b3cde3", "#8c96c6", "#88419d"]
    sorted_qs = sorted(forecast_quantiles.keys())
    for i, (q_low, q_high) in enumerate(zip(sorted_qs[:len(sorted_qs)//2], reversed(sorted_qs[len(sorted_qs)//2:]))):
        ax.fill_between(x, forecast_quantiles[q_low], forecast_quantiles[q_high],
                        alpha=0.5, color=colors[i % len(colors)],
                        label=f"{int((q_high - q_low) * 100)}% CI")

    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
