"""Helper functions for nowcasting notebooks."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


def load_gdp_vintages() -> pd.DataFrame:
    """Load GDP vintages dataset."""
    df = pd.read_csv(DATA_DIR / "gdp_vintages.csv", parse_dates=["reference_date"])
    return df


def load_mixed_freq() -> pd.DataFrame:
    """Load mixed frequency dataset."""
    df = pd.read_csv(DATA_DIR / "mixed_freq.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def load_macro_brazil() -> pd.DataFrame:
    """Load Brazilian macroeconomic dataset."""
    df = pd.read_csv(DATA_DIR / "macro_brazil.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def get_vintage(df: pd.DataFrame, vintage: int) -> pd.DataFrame:
    """Extract a specific vintage from GDP vintages dataset."""
    return df[df["vintage"] == vintage].set_index("reference_date")


def simulate_ragged_edge(df: pd.DataFrame, missing_months: dict[str, int]) -> pd.DataFrame:
    """Simulate ragged-edge missing data pattern for nowcasting.

    Parameters
    ----------
    df : DataFrame with mixed frequency data
    missing_months : dict mapping column -> number of trailing months to set as NaN
    """
    result = df.copy()
    n = len(result)
    for col, n_missing in missing_months.items():
        if col in result.columns:
            result.iloc[n - n_missing:, result.columns.get_loc(col)] = np.nan
    return result
