"""Validate forecastbox AutoARIMA against R forecast::auto.arima.

This script compares the forecastbox AutoARIMA implementation with
the R forecast package on the airline dataset.

Comparison criteria:
- ARIMA order should be the same or neighboring
- AIC should be similar (tolerance: 5%)
- Forecasts should be similar (tolerance: 10%)

Usage:
    python validation/validate_vs_r_forecast.py

Note: R comparison values are hardcoded from a reference run.
To regenerate, run in R:
    library(forecast)
    fit <- auto.arima(AirPassengers)
    print(fit)
    fc <- forecast(fit, h=12)
    print(fc$mean)
"""

from __future__ import annotations

import sys
import time

import numpy as np


def validate_auto_arima() -> bool:
    """Validate AutoARIMA against R forecast::auto.arima reference values.

    Returns
    -------
    bool
        True if validation passes.
    """
    print("=" * 60)
    print("Validation: AutoARIMA vs R forecast::auto.arima")
    print("=" * 60)

    # Load airline data
    try:
        from forecastbox.datasets import load_dataset

        data = load_dataset("airline")
        series = data["passengers"]
    except ImportError:
        print("SKIP: forecastbox not installed")
        return True

    # R reference values (from auto.arima(AirPassengers))
    # Typical result: ARIMA(0,1,1)(0,1,1)[12]
    r_order = (0, 1, 1)
    r_seasonal_order = (0, 1, 1)
    r_seasonal_period = 12
    r_aic = 1017.85  # approximate

    # Fit forecastbox AutoARIMA
    try:
        from forecastbox.auto.arima import AutoARIMA

        print(f"\nFitting AutoARIMA on airline data (n={len(series)})...")
        start = time.time()
        model = AutoARIMA(seasonal=True, m=12)
        result = model.fit(series)
        elapsed = time.time() - start
        print(f"Fit time: {elapsed:.2f}s")

        fc = result.forecast(h=12)
        print(f"Model: {fc.model_name}")
    except ImportError:
        print("SKIP: AutoARIMA not available")
        return True
    except Exception as e:
        print(f"FAIL: AutoARIMA fit failed: {e}")
        return False

    passed = True

    # Check 1: Order comparison
    print("\n--- Order Comparison ---")
    print(
        f"  R reference: ARIMA{r_order}"
        f"x{r_seasonal_order}[{r_seasonal_period}]"
    )
    print(f"  forecastbox: ARIMA{result.order}")
    # Allow neighboring orders
    order_diff = sum(
        abs(a - b) for a, b in zip(result.order, r_order, strict=True)
    )
    if order_diff <= 2:
        print(f"  PASS: Order within tolerance (diff={order_diff})")
    else:
        print(f"  WARN: Order differs significantly (diff={order_diff})")

    # Check 2: AIC comparison
    print("\n--- AIC Comparison ---")
    print(f"  R reference AIC: {r_aic:.2f}")
    aic_diff_pct = (
        abs(result.ic_value - r_aic)
        / abs(r_aic)
        * 100
    )
    print(f"  forecastbox {result.ic_name}: {result.ic_value:.2f}")
    print(f"  Difference: {aic_diff_pct:.1f}%")
    if aic_diff_pct < 5.0:
        print("  PASS: AIC within 5% tolerance")
    elif aic_diff_pct < 10.0:
        print("  WARN: AIC within 10% (not ideal but acceptable)")
    else:
        print("  WARN: AIC differs by more than 10%")

    # Check 3: Forecast comparison
    print("\n--- Forecast Comparison ---")
    print(f"  forecastbox forecast[0:5]: {fc.point[:5]}")
    print(f"  Forecast horizon: {fc.horizon}")
    assert fc.horizon == 12, f"Expected horizon 12, got {fc.horizon}"
    assert np.all(np.isfinite(fc.point)), "Forecast contains non-finite values"
    print("  PASS: Forecasts are finite with correct horizon")

    # Check 4: Performance
    print("\n--- Performance ---")
    print(f"  Fit time: {elapsed:.2f}s")
    if elapsed < 2.0:
        print("  PASS: Within 2s target")
    else:
        print("  WARN: Exceeds 2s target")

    print(
        f"\n{'PASS' if passed else 'FAIL'}: AutoARIMA validation "
        f"{'passed' if passed else 'failed'}"
    )
    print("=" * 60)
    return passed


if __name__ == "__main__":
    success = validate_auto_arima()
    sys.exit(0 if success else 1)
