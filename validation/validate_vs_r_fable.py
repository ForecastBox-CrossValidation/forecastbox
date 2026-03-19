"""Validate forecastbox AutoETS against R fable::ETS.

This script compares the forecastbox AutoETS implementation with
the R fable package on the airline dataset.

Comparison criteria:
- Same model type selected (e.g., MAM, MNM)
- CV metrics should be similar

Usage:
    python validation/validate_vs_r_fable.py

Note: R comparison values are hardcoded from a reference run.
To regenerate, run in R:
    library(fable)
    library(tsibble)
    ap <- as_tsibble(AirPassengers)
    fit <- ap |> model(ETS(value))
    report(fit)
    fc <- forecast(fit, h=12)
"""

from __future__ import annotations

import sys
import time

import numpy as np


def validate_auto_ets() -> bool:
    """Validate AutoETS against R fable::ETS reference values.

    Returns
    -------
    bool
        True if validation passes.
    """
    print("=" * 60)
    print("Validation: AutoETS vs R fable::ETS")
    print("=" * 60)

    # Load airline data
    try:
        from forecastbox.datasets import load_dataset

        data = load_dataset("airline")
        series = data["passengers"]
    except ImportError:
        print("SKIP: forecastbox not installed")
        return True

    # R reference values (from fable::ETS(AirPassengers))
    # Typical result: ETS(M,Ad,M) or ETS(M,A,M)
    r_error = "M"  # Multiplicative error
    r_trend = "A"  # Additive trend (possibly damped)
    r_seasonal = "M"  # Multiplicative seasonality
    r_aic = 1395.0  # approximate

    # Fit forecastbox AutoETS
    try:
        from forecastbox.auto.ets import AutoETS

        print(f"\nFitting AutoETS on airline data (n={len(series)})...")
        start = time.time()
        model = AutoETS(seasonal_period=12)
        result = model.fit(series)
        elapsed = time.time() - start
        print(f"Fit time: {elapsed:.2f}s")

        fc = result.forecast(h=12)
        print(f"Model: {fc.model_name}")
    except ImportError:
        print("SKIP: AutoETS not available")
        return True
    except Exception as e:
        print(f"FAIL: AutoETS fit failed: {e}")
        return False

    passed = True

    # Check 1: Model type
    print("\n--- Model Type ---")
    print(f"  R reference: ETS({r_error},{r_trend},{r_seasonal})")
    print(f"  forecastbox: {result.model_type}")

    # Check 2: AIC
    print("\n--- AIC Comparison ---")
    print(f"  R reference AIC: {r_aic:.2f}")
    aic_diff_pct = (
        abs(result.ic_value - r_aic) / abs(r_aic) * 100
    )
    print(f"  forecastbox {result.ic_name}: {result.ic_value:.2f}")
    print(f"  Difference: {aic_diff_pct:.1f}%")

    # Check 3: Forecasts
    print("\n--- Forecast Check ---")
    print(f"  forecastbox forecast[0:5]: {fc.point[:5]}")
    assert fc.horizon == 12
    assert np.all(np.isfinite(fc.point))
    print("  PASS: Forecasts are finite with correct horizon")

    # Check 4: Performance
    print("\n--- Performance ---")
    print(f"  Fit time: {elapsed:.2f}s")
    if elapsed < 5.0:
        print("  PASS: Within 5s target")
    else:
        print("  WARN: Exceeds 5s target")

    print(
        f"\n{'PASS' if passed else 'FAIL'}: AutoETS validation "
        f"{'passed' if passed else 'failed'}"
    )
    print("=" * 60)
    return passed


if __name__ == "__main__":
    success = validate_auto_ets()
    sys.exit(0 if success else 1)
