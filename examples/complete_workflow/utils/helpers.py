"""Helper functions for complete workflow notebooks."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """Load all available datasets as a dictionary."""
    datasets = {}
    for f in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(f)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            datasets[f.stem] = df
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    return datasets


def generate_workflow_report(
    forecasts: dict[str, np.ndarray],
    actual: np.ndarray,
    model_names: list[str],
) -> pd.DataFrame:
    """Generate a summary report for a forecasting workflow.

    Parameters
    ----------
    forecasts : dict mapping model name to forecast array
    actual : array of actual values
    model_names : list of model names to include

    Returns
    -------
    DataFrame with MAE, RMSE, MASE for each model
    """
    records = []
    naive_mae = np.mean(np.abs(np.diff(actual)))  # for MASE denominator

    for name in model_names:
        fc = forecasts[name]
        errors = actual - fc
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mase = mae / naive_mae if naive_mae > 0 else np.nan

        records.append({
            "model": name,
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "mase": round(mase, 4),
        })

    return pd.DataFrame(records).sort_values("mae")
