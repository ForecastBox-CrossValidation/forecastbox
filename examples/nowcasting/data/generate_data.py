"""Generate synthetic datasets for nowcasting examples.

Datasets:
- gdp_vintages.csv: GDP data vintages with revisions
- mixed_freq.csv: Mixed frequency data (monthly indicators + quarterly GDP)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_gdp_vintages(seed: int = 42) -> pd.DataFrame:
    """Generate GDP vintages with realistic revision patterns.

    Each vintage is the GDP estimate available at that publication date.
    Earlier vintages are noisier (subject to revision).
    Period: 2015Q1-2024Q4 (40 quarters), 5 vintages per quarter.
    """
    rng = np.random.default_rng(seed)
    n_quarters = 40
    n_vintages = 5  # first release, 1st revision, ..., final

    dates = pd.date_range("2015-01-01", periods=n_quarters, freq="QS")

    # True GDP growth
    true_gdp = np.empty(n_quarters)
    true_gdp[0] = 2.5
    for t in range(1, n_quarters):
        true_gdp[t] = 0.5 + 0.7 * true_gdp[t - 1] + rng.normal(0, 0.8)

    records = []
    for t in range(n_quarters):
        for v in range(n_vintages):
            # Earlier vintages have more noise (revision)
            noise_scale = 0.5 * (1 - v / (n_vintages - 1)) if n_vintages > 1 else 0
            revision = rng.normal(0, noise_scale)
            value = true_gdp[t] + revision

            pub_offset = v + 1  # months after quarter end
            records.append({
                "reference_date": dates[t].strftime("%Y-%m-%d"),
                "vintage": v + 1,
                "pub_lag_months": pub_offset,
                "gdp_growth": round(value, 4),
            })

    return pd.DataFrame(records)


def generate_mixed_freq(seed: int = 42) -> pd.DataFrame:
    """Generate mixed-frequency dataset.

    Monthly indicators: industrial_production, retail_sales, confidence_index
    Quarterly target: gdp_growth (only available at quarterly dates)
    Period: 2015-01 to 2024-12 (120 months = 40 quarters)
    """
    rng = np.random.default_rng(seed)
    n_months = 120

    dates = pd.date_range("2015-01-01", periods=n_months, freq="MS")

    # Monthly indicators
    ip = np.empty(n_months)
    ip[0] = 100
    for t in range(1, n_months):
        ip[t] = 2 + 0.95 * ip[t - 1] + rng.normal(0, 2)

    retail = np.empty(n_months)
    retail[0] = 100
    for t in range(1, n_months):
        retail[t] = 3 + 0.90 * retail[t - 1] + rng.normal(0, 3)

    confidence = np.empty(n_months)
    confidence[0] = 50
    for t in range(1, n_months):
        confidence[t] = 5 + 0.85 * confidence[t - 1] + rng.normal(0, 4)

    # Quarterly GDP (average of monthly latent factor + noise)
    gdp = np.full(n_months, np.nan)
    for q in range(0, n_months, 3):
        monthly_avg = (ip[q] + ip[min(q+1, n_months-1)] + ip[min(q+2, n_months-1)]) / 3
        gdp[q + 2] = 0.02 * monthly_avg + 0.01 * retail[q + 2] + rng.normal(0, 0.3)

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "industrial_production": np.round(ip, 4),
        "retail_sales": np.round(retail, 4),
        "confidence_index": np.round(confidence, 4),
        "gdp_growth": np.round(gdp, 4),
    })
    return df


if __name__ == "__main__":
    import os
    import shutil
    from pathlib import Path

    data_dir = os.path.dirname(os.path.abspath(__file__))

    df_vintages = generate_gdp_vintages()
    df_vintages.to_csv(os.path.join(data_dir, "gdp_vintages.csv"), index=False)
    print(f"gdp_vintages.csv: {len(df_vintages)} rows")

    df_mixed = generate_mixed_freq()
    df_mixed.to_csv(os.path.join(data_dir, "mixed_freq.csv"), index=False)
    print(f"mixed_freq.csv: {len(df_mixed)} rows")

    # Copy macro_brazil
    src = Path(data_dir).parent.parent / "basic_forecasting" / "data" / "macro_brazil.csv"
    dst = Path(data_dir) / "macro_brazil.csv"
    if src.exists():
        shutil.copy2(src, dst)
        print("macro_brazil.csv copied")
