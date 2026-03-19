"""Benchmark: Full pipeline.

Target: < 60s for 3 models, 300 observations.

Usage:
    python benchmarks/bench_pipeline.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def bench_pipeline() -> dict[str, float]:
    """Benchmark full forecast pipeline.

    Returns
    -------
    dict[str, float]
        Benchmark results.
    """
    print("=" * 60)
    print("Benchmark: Full Pipeline")
    print("=" * 60)

    # Load data
    try:
        from forecastbox.datasets import load_dataset

        data = load_dataset("macro_brazil")
        df = pd.DataFrame(data)
    except ImportError:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2000-01-01", periods=300, freq="MS")
        df = pd.DataFrame(
            {
                "ipca": 0.5 + rng.normal(0, 0.3, 300),
                "selic": 10 + np.cumsum(rng.normal(0, 0.5, 300)),
                "cambio": 3 + np.cumsum(rng.normal(0, 0.1, 300)),
            },
            index=dates,
        )

    n_obs = len(df)
    print(f"Data: {n_obs} observations, {len(df.columns)} variables")

    try:
        from forecastbox.experiment import ForecastExperiment

        models = ["auto_arima", "auto_ets", "theta"]
        print(f"Models: {models}")

        start = time.time()
        exp = ForecastExperiment(
            data=df,
            target="ipca",
            models=models,
            combination="mean",
            horizon=12,
            cv_type="expanding",
        )
        results = exp.run()
        total_time = time.time() - start

        n_forecasts = len(results.forecasts)

        print("\nResults:")
        print(f"  Models fitted: {n_forecasts}/{len(models)}")
        print(f"  Combination: {'Yes' if results.combination else 'No'}")
        print(
            f"  Total time:  {total_time:.2f}s "
            f"{'PASS' if total_time < 60.0 else 'FAIL'} (target: <60s)"
        )

        return {
            "n_obs": float(n_obs),
            "n_models": float(n_forecasts),
            "total_time": total_time,
        }

    except ImportError:
        print("SKIP: Required modules not available")
        return {"n_obs": float(n_obs), "n_models": 0, "total_time": 0}
    except Exception as e:
        print(f"FAIL: {e}")
        return {"n_obs": float(n_obs), "n_models": 0, "total_time": -1}


if __name__ == "__main__":
    results = bench_pipeline()
    t = results.get("total_time", -1)
    sys.exit(0 if t < 60.0 or t == 0 else 1)
