"""Helper functions for basic forecasting notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"


def load_macro_brazil() -> pd.DataFrame:
    """Load Brazilian macroeconomic dataset."""
    df = pd.read_csv(DATA_DIR / "macro_brazil.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def load_macro_us() -> pd.DataFrame:
    """Load US macroeconomic dataset."""
    df = pd.read_csv(DATA_DIR / "macro_us.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def plot_series(
    series: pd.Series,
    title: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot a single time series."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values, linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_forecast(
    actual: pd.Series,
    forecast: pd.Series,
    title: str = "",
    ylabel: str = "",
    ci_lower: pd.Series | None = None,
    ci_upper: pd.Series | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot actual vs forecast with optional confidence intervals."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(actual.index, actual.values, label="Actual", linewidth=1.5)
    ax.plot(forecast.index, forecast.values, label="Forecast", linewidth=1.5, linestyle="--")
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(ci_lower.index, ci_lower.values, ci_upper.values, alpha=0.2)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
