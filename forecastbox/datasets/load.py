"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"

_DATASETS: dict[str, dict[str, Any]] = {
    "macro_brazil": {
        "path": "macro_brazil.csv",
        "description": (
            "Brazilian macroeconomic series (monthly): "
            "ipca, selic, cambio, pib_mensal, producao_industrial"
        ),
    },
    "macro_us": {
        "path": "macro_us.csv",
        "description": (
            "US macroeconomic series (monthly): "
            "cpi, fed_funds, unemployment, gdp_monthly, industrial_production"
        ),
    },
    "macro_europe": {
        "path": "macro_europe.csv",
        "description": (
            "European macroeconomic series (monthly): "
            "hicp, ecb_rate, unemployment, gdp_index, industrial_production"
        ),
    },
    "airline": {
        "path": "airline.csv",
        "description": "International airline passengers (monthly, 144 obs) - Box-Jenkins classic",
    },
    "sunspot": {
        "path": "sunspot.csv",
        "description": "Monthly sunspot numbers (~300 obs) - classic AR/cycle series",
    },
    "m3_monthly": {
        "path": "m3_monthly.csv",
        "description": "M3 competition subset - 5 monthly series (~120 obs each)",
    },
    "m3_quarterly": {
        "path": "m3_quarterly.csv",
        "description": "M3 competition subset - 5 quarterly series (~80 obs each)",
    },
    "m4_monthly": {
        "path": "m4_monthly.csv",
        "description": "M4 competition subset - 5 monthly series (~150 obs each)",
    },
    "m4_quarterly": {
        "path": "m4_quarterly.csv",
        "description": "M4 competition subset - 5 quarterly series (~80 obs each)",
    },
    "tourism": {
        "path": "tourism.csv",
        "description": "Australian tourism data (quarterly, 4 regions) - hierarchical series",
    },
    "electricity": {
        "path": "electricity.csv",
        "description": "Electricity production (monthly, ~200 obs) - strong seasonality",
    },
    "gas": {
        "path": "gas.csv",
        "description": "Gas production (monthly, ~200 obs) - trend + seasonality",
    },
    "retail": {
        "path": "retail.csv",
        "description": "Retail sales (monthly, 5 categories, ~300 obs) - December spikes",
    },
    "exchange_rates": {
        "path": "exchange_rates.csv",
        "description": "Exchange rates (monthly, 6 currency pairs, ~250 obs) - financial series",
    },
    "interest_rates": {
        "path": "interest_rates.csv",
        "description": (
            "Interest rate term structure (monthly, 7 maturities, ~300 obs)"
            " - factor model"
        ),
    },
    "commodity_prices": {
        "path": "commodity_prices.csv",
        "description": "Commodity prices (monthly, 6 commodities, ~300 obs) - cycles + volatility",
    },
    "us_gdp_vintages": {
        "path": "us_gdp_vintages.csv",
        "description": "US GDP vintages (quarterly, 4 vintages, ~80 obs) - real-time data",
    },
    "brazil_gdp_vintages": {
        "path": "brazil_gdp_vintages.csv",
        "description": "Brazil GDP vintages (quarterly, 4 vintages, ~60 obs) - real-time data",
    },
    "simulated_var": {
        "path": "simulated_var.csv",
        "description": "Simulated VAR(2) process (monthly, 3 variables, 500 obs)",
    },
    "simulated_dfm": {
        "path": "simulated_dfm.csv",
        "description": "Simulated DFM (monthly, 2 factors, 10 series, 300 obs)",
    },
}


def load_dataset(name: str) -> dict[str, pd.Series]:
    """Load a built-in dataset by name.

    Parameters
    ----------
    name : str
        Dataset name. Use list_datasets() to see available names.

    Returns
    -------
    dict[str, pd.Series]
        Dictionary mapping column names to pandas Series with DatetimeIndex.
    """
    if name not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise ValueError(msg)

    info = _DATASETS[name]
    path = _DATA_DIR / info["path"]
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")

    result: dict[str, pd.Series] = {}
    for col in df.columns:
        result[col] = df[col].dropna()
    return result


def list_datasets() -> list[str]:
    """List available dataset names.

    Returns
    -------
    list[str]
        Sorted list of dataset names.
    """
    return sorted(_DATASETS.keys())


def describe_dataset(name: str) -> str:
    """Return description of a dataset.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    str
        Human-readable description.
    """
    if name not in _DATASETS:
        available = ", ".join(sorted(_DATASETS.keys()))
        msg = f"Unknown dataset '{name}'. Available: {available}"
        raise ValueError(msg)
    return _DATASETS[name]["description"]
