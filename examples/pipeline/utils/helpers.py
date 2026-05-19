"""Helper functions for pipeline notebooks."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import json
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"


def load_macro_brazil() -> pd.DataFrame:
    """Load Brazilian macroeconomic dataset."""
    df = pd.read_csv(DATA_DIR / "macro_brazil.csv", parse_dates=["date"])
    df = df.set_index("date")
    return df


def create_pipeline_config(
    target: str = "gdp_growth",
    models: list[str] | None = None,
    horizon: int = 12,
    retrain_frequency: str = "monthly",
) -> dict:
    """Create a pipeline configuration dictionary."""
    if models is None:
        models = ["auto_arima", "auto_ets", "naive"]
    return {
        "target": target,
        "models": models,
        "horizon": horizon,
        "retrain_frequency": retrain_frequency,
        "combination_method": "inverse_mse",
        "evaluation_metrics": ["mae", "rmse", "mase"],
        "created_at": datetime.now().isoformat(),
    }


def save_pipeline_config(config: dict, path: Path) -> None:
    """Save pipeline configuration to JSON."""
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_pipeline_config(path: Path) -> dict:
    """Load pipeline configuration from JSON."""
    with open(path) as f:
        return json.load(f)
