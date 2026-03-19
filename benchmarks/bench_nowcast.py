"""Benchmark: DFM nowcast.

Target: < 5s for 10 series.

Usage:
    python benchmarks/bench_nowcast.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def bench_nowcast() -> dict[str, float]:
    """Benchmark DFM nowcast.

    Returns
    -------
    dict[str, float]
        Benchmark results.
    """
    print("=" * 60)
    print("Benchmark: DFM Nowcast")
    print("=" * 60)

    # Load simulated DFM data
    try:
        from forecastbox.datasets import load_dataset

        data = load_dataset("simulated_dfm")
        df = pd.DataFrame(data)
    except (ImportError, KeyError):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2000-01-01", periods=300, freq="MS")
        df = pd.DataFrame(
            {f"series_{i + 1}": rng.normal(0, 1, 300) for i in range(10)},
            index=dates,
        )

    n_series = len(df.columns)
    n_obs = len(df)
    print(f"Data: {n_series} series, {n_obs} observations")

    try:
        from forecastbox.nowcasting.dfm import DFMNowcaster

        # Build frequency map (all monthly for synthetic data)
        freq_map = {col: "M" for col in df.columns}

        # Fit
        start = time.time()
        nowcaster = DFMNowcaster(n_factors=2, frequency_map=freq_map)
        nowcaster.fit(df)
        fit_time = time.time() - start

        # Nowcast
        start = time.time()
        nowcaster.nowcast(target=df.columns[0])
        nowcast_time = time.time() - start

        total_time = fit_time + nowcast_time

        print("\nResults:")
        print(f"  Fit time:     {fit_time:.3f}s")
        print(f"  Nowcast time: {nowcast_time:.3f}s")
        print(
            f"  Total time:   {total_time:.3f}s "
            f"{'PASS' if total_time < 5.0 else 'FAIL'} (target: <5s)"
        )

        return {
            "n_series": float(n_series),
            "n_obs": float(n_obs),
            "fit_time": fit_time,
            "nowcast_time": nowcast_time,
            "total_time": total_time,
        }

    except ImportError:
        print("SKIP: DFM module not available")
        return {"n_series": float(n_series), "n_obs": float(n_obs), "total_time": 0}
    except Exception as e:
        print(f"FAIL: {e}")
        return {"n_series": float(n_series), "n_obs": float(n_obs), "total_time": -1}


if __name__ == "__main__":
    results = bench_nowcast()
    sys.exit(0 if results.get("total_time", -1) < 5.0 else 1)
