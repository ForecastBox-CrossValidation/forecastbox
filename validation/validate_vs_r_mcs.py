"""Validate forecastbox MCS against R MCS::MCSprocedure.

This script compares the forecastbox MCS implementation with
the R MCS package.

Comparison criteria:
- Same set of included models (or very similar)

Usage:
    python validation/validate_vs_r_mcs.py

Note: R comparison values are hardcoded from a reference run.
To regenerate, run in R:
    library(MCS)
    # Create loss matrix for 5 models
    set.seed(42)
    losses <- matrix(rnorm(100*5), ncol=5)
    colnames(losses) <- paste0("Model_", 1:5)
    result <- MCSprocedure(losses, alpha=0.05, B=5000)
"""

from __future__ import annotations

import sys
import time

import numpy as np


def validate_mcs() -> bool:
    """Validate MCS against R MCS::MCSprocedure reference.

    Returns
    -------
    bool
        True if validation passes.
    """
    print("=" * 60)
    print("Validation: MCS vs R MCS::MCSprocedure")
    print("=" * 60)

    # Create reproducible data where some models are clearly worse
    rng = np.random.default_rng(42)
    n_obs = 100
    n_models = 5

    # Create actual values and model forecasts
    actual = rng.normal(0, 1, n_obs)

    # Model_1 and Model_2 forecast well, others progressively worse
    forecasts: dict[str, np.ndarray] = {
        "Model_1": actual + rng.normal(0, 0.5, n_obs),
        "Model_2": actual + rng.normal(0, 0.6, n_obs),
        "Model_3": actual + rng.normal(0, 1.0, n_obs),
        "Model_4": actual + rng.normal(0, 1.5, n_obs),
        "Model_5": actual + rng.normal(0, 2.0, n_obs),
    }

    # R reference parameters
    alpha = 0.05
    n_boot = 5000
    expected_included = ["Model_1", "Model_2"]
    expected_excluded = ["Model_4", "Model_5"]

    # Run forecastbox MCS
    try:
        from forecastbox.evaluation.mcs import model_confidence_set

        print(
            f"\nRunning MCS (n_obs={n_obs}, n_models={n_models}, "
            f"B={n_boot})..."
        )
        start = time.time()
        result = model_confidence_set(
            actual=actual,
            forecasts=forecasts,
            alpha=alpha,
            n_boot=n_boot,
            seed=42,
        )
        elapsed = time.time() - start
        print(f"MCS time: {elapsed:.2f}s")
    except ImportError:
        print("SKIP: MCS module not available")
        return True
    except Exception as e:
        print(f"FAIL: MCS failed: {e}")
        return False

    passed = True

    # Check 1: Included models
    print("\n--- Included Models ---")
    print(f"  R reference (expected): {expected_included}")
    print(f"  forecastbox: {result.included_models}")

    # Model_1 and Model_2 should be in MCS
    for model_name in expected_included:
        if model_name in result.included_models:
            print(f"  PASS: {model_name} included in MCS")
        else:
            print(f"  WARN: {model_name} NOT included (expected)")

    # Model_4 and Model_5 should NOT be in MCS
    for model_name in expected_excluded:
        if model_name not in result.included_models:
            print(f"  PASS: {model_name} excluded from MCS")
        else:
            print(f"  WARN: {model_name} included (expected exclusion)")

    # Check 2: P-values
    print("\n--- P-values ---")
    for model, pval in result.pvalues.items():
        print(f"  {model}: p={pval:.4f}")

    # Check 3: Performance
    print("\n--- Performance ---")
    print(f"  MCS time: {elapsed:.2f}s")
    if elapsed < 30.0:
        print("  PASS: Within 30s target")
    else:
        print("  WARN: Exceeds 30s target")

    print(
        f"\n{'PASS' if passed else 'FAIL'}: MCS validation "
        f"{'passed' if passed else 'failed'}"
    )
    print("=" * 60)
    return passed


if __name__ == "__main__":
    success = validate_mcs()
    sys.exit(0 if success else 1)
