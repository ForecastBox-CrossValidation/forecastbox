"""Generate synthetic macroeconomic datasets for basic forecasting examples.

All datasets use seed=42 for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_macro_brazil(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic Brazilian macroeconomic data (monthly, 2010-2024).

    Variables: gdp_growth, inflation, interest_rate, unemployment, exchange_rate
    """
    rng = np.random.default_rng(seed)
    n = 180  # 15 years monthly

    dates = pd.date_range("2010-01-01", periods=n, freq="MS")

    # GDP growth: AR(1) with drift ~0.2% monthly
    gdp = np.empty(n)
    gdp[0] = 0.3
    for t in range(1, n):
        gdp[t] = 0.05 + 0.7 * gdp[t - 1] + rng.normal(0, 0.3)

    # Inflation: AR(1) with level ~0.5% monthly
    inflation = np.empty(n)
    inflation[0] = 0.5
    for t in range(1, n):
        inflation[t] = 0.15 + 0.65 * inflation[t - 1] + rng.normal(0, 0.15)

    # Interest rate: slow-moving, correlated with inflation
    interest = np.empty(n)
    interest[0] = 10.0
    for t in range(1, n):
        interest[t] = 0.5 + 0.95 * interest[t - 1] + 0.3 * inflation[t] + rng.normal(0, 0.2)

    # Unemployment: counter-cyclical
    unemployment = np.empty(n)
    unemployment[0] = 7.0
    for t in range(1, n):
        unemployment[t] = 0.3 + 0.96 * unemployment[t - 1] - 0.1 * gdp[t] + rng.normal(0, 0.15)

    # Exchange rate: random walk with drift
    exchange = np.empty(n)
    exchange[0] = 1.75
    for t in range(1, n):
        exchange[t] = exchange[t - 1] * np.exp(0.002 + rng.normal(0, 0.03))

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "gdp_growth": np.round(gdp, 4),
        "inflation": np.round(inflation, 4),
        "interest_rate": np.round(interest, 4),
        "unemployment": np.round(unemployment, 4),
        "exchange_rate": np.round(exchange, 4),
    })
    return df


def generate_macro_us(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic US macroeconomic data (monthly, 2010-2024).

    Variables: gdp_growth, cpi_inflation, fed_funds_rate, unemployment
    """
    rng = np.random.default_rng(seed)
    n = 180

    dates = pd.date_range("2010-01-01", periods=n, freq="MS")

    gdp = np.empty(n)
    gdp[0] = 0.2
    for t in range(1, n):
        gdp[t] = 0.04 + 0.75 * gdp[t - 1] + rng.normal(0, 0.2)

    cpi = np.empty(n)
    cpi[0] = 0.2
    for t in range(1, n):
        cpi[t] = 0.05 + 0.7 * cpi[t - 1] + rng.normal(0, 0.1)

    fed_funds = np.empty(n)
    fed_funds[0] = 0.25
    for t in range(1, n):
        fed_funds[t] = max(0.0, 0.02 + 0.98 * fed_funds[t - 1] + 0.1 * cpi[t] + rng.normal(0, 0.05))

    unemployment = np.empty(n)
    unemployment[0] = 9.0
    for t in range(1, n):
        unemployment[t] = max(2.0, 0.2 + 0.97 * unemployment[t - 1] - 0.15 * gdp[t] + rng.normal(0, 0.1))

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "gdp_growth": np.round(gdp, 4),
        "cpi_inflation": np.round(cpi, 4),
        "fed_funds_rate": np.round(fed_funds, 4),
        "unemployment": np.round(unemployment, 4),
    })
    return df


if __name__ == "__main__":
    import os

    data_dir = os.path.dirname(os.path.abspath(__file__))

    df_br = generate_macro_brazil()
    df_br.to_csv(os.path.join(data_dir, "macro_brazil.csv"), index=False)
    print(f"macro_brazil.csv: {len(df_br)} rows, columns: {list(df_br.columns)}")

    df_us = generate_macro_us()
    df_us.to_csv(os.path.join(data_dir, "macro_us.csv"), index=False)
    print(f"macro_us.csv: {len(df_us)} rows, columns: {list(df_us.columns)}")
