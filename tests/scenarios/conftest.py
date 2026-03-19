"""Shared fixtures for scenarios tests."""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.scenarios._protocols import SimpleVAR


@pytest.fixture
def simple_var_model() -> SimpleVAR:
    """Create a simple 3-variable VAR(2) model for testing.

    Variables: ipca, selic, cambio
    Data is synthetic with known relationships:
    - selic affects ipca negatively (Taylor rule)
    - cambio affects ipca positively (pass-through)
    """
    rng = np.random.default_rng(42)
    T = 200
    k = 3  # ipca, selic, cambio

    # Generate synthetic VAR(2) data
    A1 = np.array([
        [0.5, -0.1, 0.05],   # ipca depends on past ipca, selic(-), cambio(+)
        [0.1, 0.8, 0.0],     # selic is persistent
        [0.0, -0.05, 0.7],   # cambio is persistent, affected by selic(-)
    ])
    A2 = np.array([
        [0.2, -0.05, 0.02],
        [0.05, 0.1, 0.0],
        [0.0, -0.02, 0.15],
    ])
    c = np.array([0.3, 0.5, 0.2])
    Sigma = np.array([
        [0.04, 0.005, 0.002],
        [0.005, 0.09, -0.003],
        [0.002, -0.003, 0.06],
    ])

    # Simulate data
    data = np.zeros((T, k))
    data[0] = c
    data[1] = c + rng.multivariate_normal(np.zeros(k), Sigma)

    for t in range(2, T):
        data[t] = (
            c
            + A1 @ data[t - 1]
            + A2 @ data[t - 2]
            + rng.multivariate_normal(np.zeros(k), Sigma)
        )

    return SimpleVAR(data, p_order=2, var_names=["ipca", "selic", "cambio"])


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)
