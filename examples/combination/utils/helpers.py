"""Helper functions for combination notebooks."""

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


def get_forecast_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract actual values and forecast matrix from inflation_forecasts DataFrame.

    Returns (actual, forecasts) where forecasts is shape (n, n_models).
    """
    actual = df["actual"].values
    fc_cols = [c for c in df.columns if c.startswith("fc_")]
    forecasts = df[fc_cols].values
    return actual, forecasts
