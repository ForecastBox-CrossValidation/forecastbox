"""Shared test fixtures for forecastbox."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.datasets import load_dataset


@pytest.fixture
def macro_brazil() -> dict[str, pd.Series]:
    """Load the macro_brazil dataset."""
    return load_dataset("macro_brazil")


@pytest.fixture
def macro_us() -> dict[str, pd.Series]:
    """Load the macro_us dataset."""
    return load_dataset("macro_us")


@pytest.fixture
def sample_forecast_data() -> dict[str, np.ndarray]:
    """Sample data for creating test forecasts."""
    return {
        "point": np.array([100.5, 101.2, 102.0]),
        "lower_80": np.array([98.0, 97.5, 96.8]),
        "upper_80": np.array([103.0, 104.9, 107.2]),
        "lower_95": np.array([96.0, 94.5, 92.8]),
        "upper_95": np.array([105.0, 107.9, 111.2]),
        "actual": np.array([100.8, 100.9, 103.1]),
    }


@pytest.fixture
def sample_index() -> pd.DatetimeIndex:
    """Sample DatetimeIndex for tests."""
    return pd.date_range("2024-01", periods=3, freq="MS")


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)
