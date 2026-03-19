"""Shared fixtures for combination tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.core.forecast import Forecast


@pytest.fixture
def three_forecasts() -> list[Forecast]:
    """Three forecasts with known values for testing."""
    index = pd.date_range("2024-01", periods=4, freq="MS")
    fc1 = Forecast(
        point=np.array([100.0, 110.0, 120.0, 130.0]),
        lower_80=np.array([95.0, 105.0, 115.0, 125.0]),
        upper_80=np.array([105.0, 115.0, 125.0, 135.0]),
        lower_95=np.array([90.0, 100.0, 110.0, 120.0]),
        upper_95=np.array([110.0, 120.0, 130.0, 140.0]),
        index=index,
        model_name="Model_A",
    )
    fc2 = Forecast(
        point=np.array([102.0, 108.0, 122.0, 128.0]),
        lower_80=np.array([97.0, 103.0, 117.0, 123.0]),
        upper_80=np.array([107.0, 113.0, 127.0, 133.0]),
        lower_95=np.array([92.0, 98.0, 112.0, 118.0]),
        upper_95=np.array([112.0, 118.0, 132.0, 138.0]),
        index=index,
        model_name="Model_B",
    )
    fc3 = Forecast(
        point=np.array([98.0, 112.0, 118.0, 132.0]),
        lower_80=np.array([93.0, 107.0, 113.0, 127.0]),
        upper_80=np.array([103.0, 117.0, 123.0, 137.0]),
        lower_95=np.array([88.0, 102.0, 108.0, 122.0]),
        upper_95=np.array([108.0, 122.0, 128.0, 142.0]),
        index=index,
        model_name="Model_C",
    )
    return [fc1, fc2, fc3]


@pytest.fixture
def actual_train() -> np.ndarray:
    """Actual training values for fitting combiners."""
    rng = np.random.default_rng(42)
    return 100.0 + np.cumsum(rng.normal(0, 1, size=60))


@pytest.fixture
def forecasts_train_3(actual_train: np.ndarray) -> list[np.ndarray]:
    """Three training forecast arrays (with varying accuracy)."""
    rng = np.random.default_rng(42)
    # Model A: best (low noise)
    fc_a = actual_train + rng.normal(0, 1.0, size=len(actual_train))
    # Model B: medium noise
    fc_b = actual_train + rng.normal(0, 2.0, size=len(actual_train))
    # Model C: worst (high noise)
    fc_c = actual_train + rng.normal(0, 4.0, size=len(actual_train))
    return [fc_a, fc_b, fc_c]


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)
