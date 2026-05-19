"""Generate synthetic datasets for forecast combination examples.

Datasets:
- inflation_forecasts.csv: Actual inflation + forecasts from 5 models
- m4_sample.csv: M4-like series with individual model forecasts
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_inflation_forecasts(seed: int = 42) -> pd.DataFrame:
    """Generate actual inflation and forecasts from 5 different models.

    Models: ARIMA, ETS, VAR, naive, drift
    Period: 120 months (10 years), out-of-sample from month 61
    """
    rng = np.random.default_rng(seed)
    n = 120

    dates = pd.date_range("2015-01-01", periods=n, freq="MS")

    # Actual inflation (AR(1) + seasonal)
    actual = np.empty(n)
    actual[0] = 0.4
    for t in range(1, n):
        seasonal = 0.05 * np.sin(2 * np.pi * t / 12)
        actual[t] = 0.1 + 0.65 * actual[t - 1] + seasonal + rng.normal(0, 0.1)

    # Model forecasts (actual + model-specific bias and noise)
    fc_arima = actual + rng.normal(0.02, 0.08, n)    # slight positive bias
    fc_ets = actual + rng.normal(-0.01, 0.10, n)     # slight negative bias
    fc_var = actual + rng.normal(0.00, 0.12, n)      # unbiased but noisy
    fc_naive = np.roll(actual, 1)                      # lagged actual
    fc_naive[0] = actual[0]
    fc_drift = actual + np.linspace(-0.05, 0.05, n) + rng.normal(0, 0.09, n)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "actual": np.round(actual, 4),
        "fc_arima": np.round(fc_arima, 4),
        "fc_ets": np.round(fc_ets, 4),
        "fc_var": np.round(fc_var, 4),
        "fc_naive": np.round(fc_naive, 4),
        "fc_drift": np.round(fc_drift, 4),
    })
    return df


def generate_m4_sample(seed: int = 42) -> pd.DataFrame:
    """Generate M4-like sample with 6 series and pre-computed forecasts."""
    rng = np.random.default_rng(seed)
    records = []

    for i in range(6):
        n = 120
        # Different DGPs
        if i < 2:   # trending
            y = np.cumsum(rng.normal(0.5, 1, n)) + 100
        elif i < 4:  # seasonal
            y = 100 + 15 * np.sin(2 * np.pi * np.arange(n) / 12) + rng.normal(0, 3, n)
        else:        # stationary
            y = np.empty(n)
            y[0] = 100
            for t in range(1, n):
                y[t] = 30 + 0.7 * y[t - 1] + rng.normal(0, 5)

        # Generate 3 model forecasts
        fc1 = y + rng.normal(0, 2, n)
        fc2 = y + rng.normal(0.5, 3, n)
        fc3 = y + rng.normal(-0.3, 2.5, n)

        for t in range(n):
            records.append({
                "series_id": f"M4_{i+1:03d}",
                "period": t + 1,
                "actual": round(y[t], 2),
                "fc_model1": round(fc1[t], 2),
                "fc_model2": round(fc2[t], 2),
                "fc_model3": round(fc3[t], 2),
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    import os
    data_dir = os.path.dirname(os.path.abspath(__file__))

    df_inf = generate_inflation_forecasts()
    df_inf.to_csv(os.path.join(data_dir, "inflation_forecasts.csv"), index=False)
    print(f"inflation_forecasts.csv: {len(df_inf)} rows")

    df_m4 = generate_m4_sample()
    df_m4.to_csv(os.path.join(data_dir, "m4_sample.csv"), index=False)
    print(f"m4_sample.csv: {len(df_m4)} rows, {df_m4['series_id'].nunique()} series")
