"""Generate synthetic datasets for forecastbox.

Run this script once to generate all CSV files.
After generation, this script can be removed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent


def generate_macro_europe() -> None:
    """Generate European macroeconomic data."""
    rng = np.random.default_rng(200)
    dates = pd.date_range("2000-01-01", periods=250, freq="MS")
    n = len(dates)

    hicp = 0.15 + rng.normal(0, 0.1, n)
    hicp = np.clip(hicp, -0.5, 1.5)

    ecb_rate = np.zeros(n)
    ecb_rate[0] = 3.0
    for i in range(1, n):
        ecb_rate[i] = ecb_rate[i - 1] + rng.normal(0, 0.05)
    ecb_rate = np.clip(ecb_rate, -0.5, 5.0)

    unemployment = np.zeros(n)
    unemployment[0] = 8.0
    for i in range(1, n):
        unemployment[i] = unemployment[i - 1] + rng.normal(0, 0.1)
    unemployment = np.clip(unemployment, 3.0, 13.0)

    gdp_index = 100 + np.cumsum(rng.normal(0.1, 0.5, n))
    industrial_prod = 100 + np.cumsum(rng.normal(0.05, 0.8, n))

    df = pd.DataFrame({
        "date": dates,
        "hicp": np.round(hicp, 4),
        "ecb_rate": np.round(ecb_rate, 2),
        "unemployment": np.round(unemployment, 1),
        "gdp_index": np.round(gdp_index, 2),
        "industrial_production": np.round(industrial_prod, 2),
    })
    df.to_csv(DATA_DIR / "macro_europe.csv", index=False)


def generate_airline() -> None:
    """Generate airline passengers data (Box-Jenkins style)."""
    rng = np.random.default_rng(300)
    dates = pd.date_range("1949-01-01", periods=144, freq="MS")
    n = len(dates)

    trend = np.linspace(100, 500, n)
    seasonal = 30 * np.sin(2 * np.pi * np.arange(n) / 12) + 15 * np.cos(
        4 * np.pi * np.arange(n) / 12
    )
    # Multiplicative effect: seasonal amplitude grows with trend
    amplitude = trend / 200
    passengers = trend + seasonal * amplitude + rng.normal(0, 10, n)
    passengers = np.maximum(passengers, 50)

    df = pd.DataFrame({
        "date": dates,
        "passengers": np.round(passengers, 0).astype(int),
    })
    df.to_csv(DATA_DIR / "airline.csv", index=False)


def generate_sunspot() -> None:
    """Generate sunspot data."""
    rng = np.random.default_rng(400)
    dates = pd.date_range("2000-01-01", periods=300, freq="MS")
    n = len(dates)

    # Solar cycle ~11 years = 132 months
    cycle = 80 * np.sin(2 * np.pi * np.arange(n) / 132) + 80
    noise = rng.normal(0, 15, n)
    sunspots = cycle + noise
    sunspots = np.maximum(sunspots, 0)

    df = pd.DataFrame({
        "date": dates,
        "sunspots": np.round(sunspots, 1),
    })
    df.to_csv(DATA_DIR / "sunspot.csv", index=False)


def generate_m3_monthly() -> None:
    """Generate M3-style monthly competition data (5 series)."""
    rng = np.random.default_rng(500)
    dates = pd.date_range("1990-01-01", periods=120, freq="MS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    for i in range(5):
        trend = np.linspace(100, 100 + rng.uniform(20, 80), n)
        seasonal = rng.uniform(5, 20) * np.sin(
            2 * np.pi * np.arange(n) / 12 + rng.uniform(0, 2 * np.pi)
        )
        noise = rng.normal(0, rng.uniform(2, 8), n)
        data[f"series_{i + 1}"] = np.round(trend + seasonal + noise, 2)

    pd.DataFrame(data).to_csv(DATA_DIR / "m3_monthly.csv", index=False)


def generate_m3_quarterly() -> None:
    """Generate M3-style quarterly competition data (5 series)."""
    rng = np.random.default_rng(501)
    dates = pd.date_range("1990-01-01", periods=80, freq="QS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    for i in range(5):
        trend = np.linspace(100, 100 + rng.uniform(30, 100), n)
        seasonal = rng.uniform(5, 15) * np.sin(
            2 * np.pi * np.arange(n) / 4 + rng.uniform(0, 2 * np.pi)
        )
        noise = rng.normal(0, rng.uniform(3, 10), n)
        data[f"series_{i + 1}"] = np.round(trend + seasonal + noise, 2)

    pd.DataFrame(data).to_csv(DATA_DIR / "m3_quarterly.csv", index=False)


def generate_m4_monthly() -> None:
    """Generate M4-style monthly competition data (5 series)."""
    rng = np.random.default_rng(600)
    dates = pd.date_range("1995-01-01", periods=150, freq="MS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    for i in range(5):
        trend = np.linspace(100, 100 + rng.uniform(40, 120), n)
        seasonal = rng.uniform(8, 25) * np.sin(
            2 * np.pi * np.arange(n) / 12 + rng.uniform(0, 2 * np.pi)
        )
        noise = rng.normal(0, rng.uniform(3, 10), n)
        data[f"series_{i + 1}"] = np.round(trend + seasonal + noise, 2)

    pd.DataFrame(data).to_csv(DATA_DIR / "m4_monthly.csv", index=False)


def generate_m4_quarterly() -> None:
    """Generate M4-style quarterly competition data (5 series)."""
    rng = np.random.default_rng(601)
    dates = pd.date_range("1995-01-01", periods=80, freq="QS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    for i in range(5):
        trend = np.linspace(100, 100 + rng.uniform(30, 90), n)
        seasonal = rng.uniform(5, 15) * np.sin(
            2 * np.pi * np.arange(n) / 4 + rng.uniform(0, 2 * np.pi)
        )
        noise = rng.normal(0, rng.uniform(3, 8), n)
        data[f"series_{i + 1}"] = np.round(trend + seasonal + noise, 2)

    pd.DataFrame(data).to_csv(DATA_DIR / "m4_quarterly.csv", index=False)


def generate_tourism() -> None:
    """Generate Australian tourism data (quarterly)."""
    rng = np.random.default_rng(700)
    dates = pd.date_range("2000-01-01", periods=80, freq="QS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    regions = ["sydney", "melbourne", "brisbane", "perth"]
    for region in regions:
        trend = np.linspace(1000, 1000 + rng.uniform(200, 800), n)
        seasonal = rng.uniform(50, 200) * np.sin(
            2 * np.pi * np.arange(n) / 4 + rng.uniform(0, 2 * np.pi)
        )
        noise = rng.normal(0, rng.uniform(30, 80), n)
        data[region] = np.round(trend + seasonal + noise, 0).astype(int)

    pd.DataFrame(data).to_csv(DATA_DIR / "tourism.csv", index=False)


def generate_electricity() -> None:
    """Generate electricity production data (monthly)."""
    rng = np.random.default_rng(800)
    dates = pd.date_range("2000-01-01", periods=200, freq="MS")
    n = len(dates)

    trend = np.linspace(100, 150, n)
    # Strong seasonality (winter/summer)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 12) + 10 * np.cos(
        4 * np.pi * np.arange(n) / 12
    )
    noise = rng.normal(0, 3, n)
    production = trend + seasonal + noise

    df = pd.DataFrame({
        "date": dates,
        "production": np.round(production, 2),
    })
    df.to_csv(DATA_DIR / "electricity.csv", index=False)


def generate_gas() -> None:
    """Generate gas production data (monthly)."""
    rng = np.random.default_rng(801)
    dates = pd.date_range("2000-01-01", periods=200, freq="MS")
    n = len(dates)

    trend = np.linspace(50, 120, n)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 12 + np.pi)  # Peak in winter
    noise = rng.normal(0, 4, n)
    production = trend + seasonal + noise

    df = pd.DataFrame({
        "date": dates,
        "production": np.round(production, 2),
    })
    df.to_csv(DATA_DIR / "gas.csv", index=False)


def generate_retail() -> None:
    """Generate retail sales data (monthly, multiple series)."""
    rng = np.random.default_rng(900)
    dates = pd.date_range("2000-01-01", periods=300, freq="MS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    categories = ["food", "clothing", "electronics", "furniture", "automotive"]
    for cat in categories:
        trend = np.linspace(100, 100 + rng.uniform(50, 200), n)
        seasonal = rng.uniform(10, 30) * np.sin(
            2 * np.pi * np.arange(n) / 12 + rng.uniform(0, 2 * np.pi)
        )
        # December spike
        december_boost = np.zeros(n)
        december_idx = np.arange(n)[pd.Series(dates).dt.month.values == 12]
        december_boost[december_idx] = rng.uniform(20, 60)
        noise = rng.normal(0, rng.uniform(5, 15), n)
        data[cat] = np.round(trend + seasonal + december_boost + noise, 2)

    pd.DataFrame(data).to_csv(DATA_DIR / "retail.csv", index=False)


def generate_exchange_rates() -> None:
    """Generate exchange rate data (monthly, 6 currencies)."""
    rng = np.random.default_rng(1000)
    dates = pd.date_range("2000-01-01", periods=250, freq="MS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    currencies = {
        "eur_usd": (1.10, 0.01),
        "gbp_usd": (1.40, 0.015),
        "usd_jpy": (110.0, 1.5),
        "usd_brl": (3.50, 0.05),
        "usd_cny": (6.80, 0.02),
        "usd_inr": (65.0, 0.5),
    }

    for currency, (initial, vol) in currencies.items():
        rate = np.zeros(n)
        rate[0] = initial
        for i in range(1, n):
            rate[i] = rate[i - 1] * np.exp(rng.normal(0, vol))
        data[currency] = np.round(rate, 4)

    pd.DataFrame(data).to_csv(DATA_DIR / "exchange_rates.csv", index=False)


def generate_interest_rates() -> None:
    """Generate interest rate term structure data (monthly)."""
    rng = np.random.default_rng(1100)
    dates = pd.date_range("2000-01-01", periods=300, freq="MS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    maturities = ["3m", "6m", "1y", "2y", "5y", "10y", "30y"]
    base_levels = [2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 4.5]

    # Common factor (level)
    level = np.zeros(n)
    level[0] = 3.0
    for i in range(1, n):
        level[i] = level[i - 1] + rng.normal(0, 0.05)
    level = np.clip(level, 0, 8)

    for mat, base in zip(maturities, base_levels, strict=True):
        spread = base - 3.0
        noise = rng.normal(0, 0.1, n)
        data[mat] = np.round(level + spread + noise, 3)
        data[mat] = np.clip(data[mat], 0, 10).round(3)

    pd.DataFrame(data).to_csv(DATA_DIR / "interest_rates.csv", index=False)


def generate_commodity_prices() -> None:
    """Generate commodity price data (monthly)."""
    rng = np.random.default_rng(1200)
    dates = pd.date_range("2000-01-01", periods=300, freq="MS")
    n = len(dates)

    data: dict[str, object] = {"date": dates}
    commodities = {
        "oil_brent": (60.0, 0.06),
        "gold": (1200.0, 0.03),
        "copper": (6000.0, 0.05),
        "iron_ore": (80.0, 0.07),
        "soybeans": (1000.0, 0.04),
        "corn": (400.0, 0.05),
    }

    for commodity, (initial, vol) in commodities.items():
        price = np.zeros(n)
        price[0] = initial
        for i in range(1, n):
            price[i] = price[i - 1] * np.exp(rng.normal(0.001, vol))
        data[commodity] = np.round(price, 2)

    pd.DataFrame(data).to_csv(DATA_DIR / "commodity_prices.csv", index=False)


def generate_us_gdp_vintages() -> None:
    """Generate US GDP vintages (quarterly with revisions)."""
    rng = np.random.default_rng(1300)
    dates = pd.date_range("2005-01-01", periods=80, freq="QS")
    n = len(dates)

    # True GDP growth
    true_growth = 2.0 + rng.normal(0, 1.5, n)

    data: dict[str, object] = {"date": dates}
    # Create 4 vintage columns (first release, 1st revision, 2nd revision, final)
    vintages = ["first_release", "revision_1", "revision_2", "final"]
    revision_stds = [0.8, 0.4, 0.2, 0.0]

    for vintage, rev_std in zip(vintages, revision_stds, strict=True):
        revision = rng.normal(0, rev_std, n)
        data[vintage] = np.round(true_growth + revision, 2)

    pd.DataFrame(data).to_csv(DATA_DIR / "us_gdp_vintages.csv", index=False)


def generate_brazil_gdp_vintages() -> None:
    """Generate Brazil GDP vintages (quarterly with revisions)."""
    rng = np.random.default_rng(1400)
    dates = pd.date_range("2008-01-01", periods=60, freq="QS")
    n = len(dates)

    true_growth = 1.5 + rng.normal(0, 2.0, n)

    data: dict[str, object] = {"date": dates}
    vintages = ["first_release", "revision_1", "revision_2", "final"]
    revision_stds = [1.0, 0.5, 0.25, 0.0]

    for vintage, rev_std in zip(vintages, revision_stds, strict=True):
        revision = rng.normal(0, rev_std, n)
        data[vintage] = np.round(true_growth + revision, 2)

    pd.DataFrame(data).to_csv(DATA_DIR / "brazil_gdp_vintages.csv", index=False)


def generate_simulated_var() -> None:
    """Generate simulated VAR(2) data (3 variables, 500 obs)."""
    rng = np.random.default_rng(1500)
    n = 500
    k = 3  # 3 variables
    dates = pd.date_range("1980-01-01", periods=n, freq="MS")

    # VAR(2) coefficients
    a1 = np.array([
        [0.5, 0.1, 0.0],
        [0.2, 0.4, 0.1],
        [0.0, 0.1, 0.6],
    ])
    a2 = np.array([
        [0.2, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ])

    # Covariance matrix
    sigma = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ])

    y = np.zeros((n, k))
    y[0] = rng.multivariate_normal(np.zeros(k), sigma)
    y[1] = rng.multivariate_normal(np.zeros(k), sigma)

    for t in range(2, n):
        eps = rng.multivariate_normal(np.zeros(k), sigma)
        y[t] = a1 @ y[t - 1] + a2 @ y[t - 2] + eps

    df = pd.DataFrame({
        "date": dates,
        "var_1": np.round(y[:, 0], 4),
        "var_2": np.round(y[:, 1], 4),
        "var_3": np.round(y[:, 2], 4),
    })
    df.to_csv(DATA_DIR / "simulated_var.csv", index=False)


def generate_simulated_dfm() -> None:
    """Generate simulated DFM data (2 factors, 10 series, 300 obs)."""
    rng = np.random.default_rng(1600)
    n = 300
    k_factors = 2
    k_series = 10
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")

    # Generate factors as AR(1) processes
    factors = np.zeros((n, k_factors))
    phi = [0.8, 0.6]
    for f in range(k_factors):
        for t in range(1, n):
            factors[t, f] = phi[f] * factors[t - 1, f] + rng.normal(0, 1)

    # Loading matrix
    loadings = rng.normal(0.5, 0.3, (k_series, k_factors))

    # Generate series
    idiosyncratic = rng.normal(0, 0.5, (n, k_series))
    series = factors @ loadings.T + idiosyncratic

    data: dict[str, object] = {"date": dates}
    for i in range(k_series):
        data[f"series_{i + 1}"] = np.round(series[:, i], 4)

    pd.DataFrame(data).to_csv(DATA_DIR / "simulated_dfm.csv", index=False)


def main() -> None:
    """Generate all datasets."""
    print("Generating datasets...")

    generate_macro_europe()
    print("  macro_europe.csv")

    generate_airline()
    print("  airline.csv")

    generate_sunspot()
    print("  sunspot.csv")

    generate_m3_monthly()
    print("  m3_monthly.csv")

    generate_m3_quarterly()
    print("  m3_quarterly.csv")

    generate_m4_monthly()
    print("  m4_monthly.csv")

    generate_m4_quarterly()
    print("  m4_quarterly.csv")

    generate_tourism()
    print("  tourism.csv")

    generate_electricity()
    print("  electricity.csv")

    generate_gas()
    print("  gas.csv")

    generate_retail()
    print("  retail.csv")

    generate_exchange_rates()
    print("  exchange_rates.csv")

    generate_interest_rates()
    print("  interest_rates.csv")

    generate_commodity_prices()
    print("  commodity_prices.csv")

    generate_us_gdp_vintages()
    print("  us_gdp_vintages.csv")

    generate_brazil_gdp_vintages()
    print("  brazil_gdp_vintages.csv")

    generate_simulated_var()
    print("  simulated_var.csv")

    generate_simulated_dfm()
    print("  simulated_dfm.csv")

    print("Done! All datasets generated.")


if __name__ == "__main__":
    main()
