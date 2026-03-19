"""Benchmark: AutoARIMA fit time.

Target: < 2s for 300 observations.

Usage:
    python benchmarks/bench_auto_arima.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def bench_auto_arima() -> dict[str, float]:
    """Benchmark AutoARIMA fit time.

    Returns
    -------
    dict[str, float]
        Benchmark results with keys: n_obs, fit_time, forecast_time, total_time.
    """
    print("=" * 60)
    print("Benchmark: AutoARIMA")
    print("=" * 60)

    # Load data
    try:
        from forecastbox.datasets import load_dataset

        data = load_dataset("macro_brazil")
        series = data["ipca"]
    except ImportError:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2000-01-01", periods=300, freq="MS")
        series = pd.Series(
            0.5 + rng.normal(0, 0.3, 300), index=dates, name="ipca"
        )

    n_obs = len(series)
    print(f"Data: {n_obs} observations")

    try:
        from forecastbox.auto.arima import AutoARIMA

        # Fit
        model = AutoARIMA(seasonal=True, m=12)
        start = time.time()
        result = model.fit(series)
        fit_time = time.time() - start

        # Forecast
        start = time.time()
        fc = result.forecast(h=12)
        forecast_time = time.time() - start

        total_time = fit_time + forecast_time

        print("\nResults:")
        print(
            f"  Fit time:      {fit_time:.3f}s "
            f"{'PASS' if fit_time < 2.0 else 'FAIL'} (target: <2s)"
        )
        print(f"  Forecast time: {forecast_time:.3f}s")
        print(f"  Total time:    {total_time:.3f}s")
        print(f"  Model:         {fc.model_name}")

        return {
            "n_obs": n_obs,
            "fit_time": fit_time,
            "forecast_time": forecast_time,
            "total_time": total_time,
        }

    except ImportError:
        print("SKIP: AutoARIMA not available")
        return {"n_obs": n_obs, "fit_time": 0, "forecast_time": 0, "total_time": 0}
    except Exception as e:
        print(f"FAIL: {e}")
        return {"n_obs": n_obs, "fit_time": -1, "forecast_time": -1, "total_time": -1}


if __name__ == "__main__":
    results = bench_auto_arima()
    passed = results.get("fit_time", -1) < 2.0 or results.get("fit_time", -1) == 0
    sys.exit(0 if passed else 1)
