"""Benchmark: Forecast combination methods.

Target: < 1s for 7 methods, 12-step ahead.

Usage:
    python benchmarks/bench_combination.py
"""

from __future__ import annotations

import sys
import time

import numpy as np


def bench_combination() -> dict[str, float]:
    """Benchmark combination methods.

    Returns
    -------
    dict[str, float]
        Time per method.
    """
    print("=" * 60)
    print("Benchmark: Forecast Combination")
    print("=" * 60)

    from forecastbox.core.forecast import Forecast

    # Create 5 synthetic forecasts
    rng = np.random.default_rng(42)
    forecasts: list[Forecast] = []
    for i in range(5):
        fc = Forecast(
            point=100 + rng.normal(0, 5, 12),
            lower_80=100 + rng.normal(0, 5, 12) - 5,
            upper_80=100 + rng.normal(0, 5, 12) + 5,
            model_name=f"Model_{i + 1}",
        )
        forecasts.append(fc)

    actual = 100 + rng.normal(0, 3, 12)

    results: dict[str, float] = {}

    # Simple methods (always available via Forecast.combine)
    for method in ["mean", "median"]:
        start = time.time()
        Forecast.combine(forecasts, method=method)
        elapsed = time.time() - start
        results[method] = elapsed
        print(f"  {method}: {elapsed:.4f}s")

    # Advanced methods via combination module
    combiner_map = {
        "inverse_mse": "forecastbox.combination.weighted",
        "ols": "forecastbox.combination.ols",
        "bma": "forecastbox.combination.bma",
        "stacking": "forecastbox.combination.stacking",
        "optimal": "forecastbox.combination.optimal",
    }

    # Extract point arrays for training
    fc_arrays = [fc.point for fc in forecasts]

    for method, module_path in combiner_map.items():
        try:
            import importlib

            mod = importlib.import_module(module_path)
            # Get the combiner class (first class that ends with Combiner)
            combiner_cls = None
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name.endswith("Combiner")
                    and attr_name != "BaseCombiner"
                ):
                    combiner_cls = attr
                    break

            if combiner_cls is None:
                print(f"  {method}: SKIP (no combiner class found)")
                continue

            start = time.time()
            combiner = combiner_cls()
            combiner.fit(fc_arrays, actual)
            combiner.combine(forecasts)
            elapsed = time.time() - start
            results[method] = elapsed
            print(f"  {method}: {elapsed:.4f}s")
        except ImportError:
            print(f"  {method}: SKIP (module not available)")
        except Exception as e:
            print(f"  {method}: FAIL ({e})")
            results[method] = -1

    total = sum(v for v in results.values() if v > 0)
    print(
        f"\nTotal time: {total:.4f}s "
        f"{'PASS' if total < 1.0 else 'FAIL'} (target: <1s)"
    )

    return results


if __name__ == "__main__":
    results = bench_combination()
    total = sum(v for v in results.values() if v > 0)
    sys.exit(0 if total < 1.0 else 1)
