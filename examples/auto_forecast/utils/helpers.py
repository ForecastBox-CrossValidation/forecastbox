"""Helper functions for auto-forecast notebooks."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_m3_sample() -> pd.DataFrame:
    """Load M3 sample dataset."""
    return pd.read_csv(DATA_DIR / "m3_sample.csv")


def load_airline() -> pd.DataFrame:
    """Load airline passengers dataset."""
    df = pd.read_csv(DATA_DIR / "airline.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def get_series(df: pd.DataFrame, series_id: str) -> pd.Series:
    """Extract a single series from M3 sample by series_id."""
    subset = df[df["series_id"] == series_id].sort_values("period")
    return pd.Series(subset["value"].values, name=series_id)
