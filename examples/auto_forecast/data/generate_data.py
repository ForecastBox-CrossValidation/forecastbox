"""Generate synthetic datasets for auto-forecast examples.

Datasets:
- m3_sample.csv: Sample of M3-like time series (monthly, quarterly)
- airline.csv: Classic airline passengers-like dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_m3_sample(seed: int = 42) -> pd.DataFrame:
    """Generate M3-like sample with multiple time series.

    Creates 10 series with different characteristics:
    - 5 monthly series (trending, seasonal, stationary, random walk, damped)
    - 5 quarterly series (same patterns)
    """
    rng = np.random.default_rng(seed)
    records = []

    for freq_label, n, period in [("monthly", 144, 12), ("quarterly", 60, 4)]:
        # Trending series
        trend = np.linspace(100, 200, n) + rng.normal(0, 5, n)
        for t in range(n):
            records.append({"series_id": f"{freq_label}_trend", "period": t + 1, "value": round(trend[t], 2), "frequency": freq_label})

        # Seasonal series
        seasonal = 100 + 20 * np.sin(2 * np.pi * np.arange(n) / period) + rng.normal(0, 3, n)
        for t in range(n):
            records.append({"series_id": f"{freq_label}_seasonal", "period": t + 1, "value": round(seasonal[t], 2), "frequency": freq_label})

        # Trend + seasonal
        trend_seas = np.linspace(50, 150, n) + 15 * np.sin(2 * np.pi * np.arange(n) / period) + rng.normal(0, 4, n)
        for t in range(n):
            records.append({"series_id": f"{freq_label}_trend_seasonal", "period": t + 1, "value": round(trend_seas[t], 2), "frequency": freq_label})

        # Stationary (AR(1))
        ar1 = np.empty(n)
        ar1[0] = 100
        for t in range(1, n):
            ar1[t] = 50 + 0.5 * ar1[t - 1] + rng.normal(0, 5)
        for t in range(n):
            records.append({"series_id": f"{freq_label}_stationary", "period": t + 1, "value": round(ar1[t], 2), "frequency": freq_label})

        # Random walk
        rw = np.cumsum(rng.normal(0.1, 2, n)) + 100
        for t in range(n):
            records.append({"series_id": f"{freq_label}_random_walk", "period": t + 1, "value": round(rw[t], 2), "frequency": freq_label})

    return pd.DataFrame(records)


def generate_airline(seed: int = 42) -> pd.DataFrame:
    """Generate airline passengers-like dataset (monthly, 1949-1960)."""
    rng = np.random.default_rng(seed)
    n = 144  # 12 years monthly

    dates = pd.date_range("1949-01-01", periods=n, freq="MS")
    trend = np.linspace(100, 500, n)
    seasonal = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 12) + 0.1 * np.cos(4 * np.pi * np.arange(n) / 12)
    noise = rng.normal(1, 0.03, n)

    passengers = trend * seasonal * noise

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "passengers": np.round(passengers, 0).astype(int),
    })
    return df


if __name__ == "__main__":
    import os
    data_dir = os.path.dirname(os.path.abspath(__file__))

    df_m3 = generate_m3_sample()
    df_m3.to_csv(os.path.join(data_dir, "m3_sample.csv"), index=False)
    print(f"m3_sample.csv: {len(df_m3)} rows, {df_m3['series_id'].nunique()} series")

    df_airline = generate_airline()
    df_airline.to_csv(os.path.join(data_dir, "airline.csv"), index=False)
    print(f"airline.csv: {len(df_airline)} rows")
