"""Generate synthetic datasets for scenario analysis examples.

Datasets:
- us_macro_quarterly.csv: Quarterly US macro data (GDP, inflation, fed funds, unemployment)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_us_macro_quarterly(seed: int = 42) -> pd.DataFrame:
    """Generate quarterly US macro data with realistic VAR dynamics.

    Variables: gdp_growth, inflation, fed_funds, unemployment
    Period: 1990Q1-2024Q4 (140 quarters)
    """
    rng = np.random.default_rng(seed)
    n = 140

    dates = pd.date_range("1990-01-01", periods=n, freq="QS")

    # VAR(1) with realistic coefficients
    # State: [gdp, inflation, fed_funds, unemployment]
    A = np.array([
        [0.60, -0.10, -0.05,  0.00],  # gdp
        [0.05,  0.70,  0.02, -0.03],  # inflation
        [0.10,  0.15,  0.85,  0.00],  # fed_funds
        [-0.15, 0.00,  0.05,  0.90],  # unemployment
    ])
    mu = np.array([0.8, 0.5, 1.0, 1.0])  # long-run means (approx)
    Sigma = np.array([
        [0.40, 0.05, 0.02, -0.10],
        [0.05, 0.15, 0.03, -0.02],
        [0.02, 0.03, 0.10,  0.01],
        [-0.10, -0.02, 0.01, 0.08],
    ])
    L = np.linalg.cholesky(Sigma)

    y = np.zeros((n, 4))
    y[0] = [2.5, 2.0, 4.0, 5.5]

    for t in range(1, n):
        shock = L @ rng.standard_normal(4)
        y[t] = mu + A @ (y[t-1] - mu) + shock

    # Ensure reasonable ranges
    y[:, 2] = np.maximum(y[:, 2], 0.0)     # fed funds >= 0
    y[:, 3] = np.maximum(y[:, 3], 2.0)     # unemployment >= 2%

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "gdp_growth": np.round(y[:, 0], 4),
        "inflation": np.round(y[:, 1], 4),
        "fed_funds": np.round(y[:, 2], 4),
        "unemployment": np.round(y[:, 3], 4),
    })
    return df


if __name__ == "__main__":
    import os
    import shutil
    from pathlib import Path

    data_dir = os.path.dirname(os.path.abspath(__file__))

    df_us = generate_us_macro_quarterly()
    df_us.to_csv(os.path.join(data_dir, "us_macro_quarterly.csv"), index=False)
    print(f"us_macro_quarterly.csv: {len(df_us)} rows, columns: {list(df_us.columns)}")

    # Copy macro_brazil from basic_forecasting
    src = Path(data_dir).parent.parent / "basic_forecasting" / "data" / "macro_brazil.csv"
    dst = Path(data_dir) / "macro_brazil.csv"
    if src.exists():
        shutil.copy2(src, dst)
        print("macro_brazil.csv copied from basic_forecasting")
    else:
        print("WARNING: macro_brazil.csv not found, run FASE1 setup first")
