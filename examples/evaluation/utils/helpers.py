"""Helper functions for evaluation notebooks."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


def load_inflation_forecasts() -> pd.DataFrame:
    """Load inflation forecasts dataset."""
    df = pd.read_csv(DATA_DIR / "inflation_forecasts.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def load_m4_sample() -> pd.DataFrame:
    """Load M4 sample dataset."""
    return pd.read_csv(DATA_DIR / "m4_sample.csv")


def compute_loss_differential(
    actual: np.ndarray,
    fc1: np.ndarray,
    fc2: np.ndarray,
    loss: str = "squared",
) -> np.ndarray:
    """Compute loss differential series for DM test.

    Parameters
    ----------
    actual : array of actual values
    fc1, fc2 : arrays of forecasts
    loss : 'squared' or 'absolute'

    Returns
    -------
    d : loss differential d_t = L(e1_t) - L(e2_t)
    """
    e1 = actual - fc1
    e2 = actual - fc2
    if loss == "squared":
        return e1**2 - e2**2
    elif loss == "absolute":
        return np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: {loss}")
