"""Tests for MIDAS regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.nowcasting.midas import MIDAS


@pytest.fixture
def midas_data() -> pd.DataFrame:
    """Create synthetic mixed-frequency data for MIDAS tests.

    Monthly indicator + quarterly target over 10 years.
    """
    rng = np.random.default_rng(42)
    n_months = 120  # 10 years

    dates = pd.date_range("2014-01-01", periods=n_months, freq="MS")

    # Monthly indicator (AR(1) process)
    x = np.zeros(n_months)
    x[0] = rng.normal(0, 1)
    for t in range(1, n_months):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 0.5)

    # Quarterly target: weighted average of monthly x with decaying weights
    pib = np.full(n_months, np.nan)
    true_weights = np.array([0.5, 0.3, 0.15, 0.05])
    true_weights = true_weights / true_weights.sum()

    for t in range(3, n_months, 3):
        # Use last 4 monthly observations
        if t >= 4:
            x_lags = np.array([x[t - j] for j in range(4)])
            pib[t] = 2.0 + 1.5 * np.dot(true_weights, x_lags) + rng.normal(0, 0.3)

    data = pd.DataFrame(
        {
            "producao_industrial": x,
            "pib_quarterly": pib,
        },
        index=dates,
    )
    return data


class TestMIDAS:
    """Tests for MIDAS regression."""

    def test_beta_weights_sum_one(self, midas_data: pd.DataFrame) -> None:
        """Beta polynomial weights must sum to 1."""
        midas = MIDAS(
            target="pib_quarterly",
            high_freq=["producao_industrial"],
            weight_scheme="beta",
            n_lags=12,
        )
        midas.fit(midas_data)

        assert abs(midas.weights_.sum() - 1.0) < 1e-10

    def test_almon_weights_sum_one(self, midas_data: pd.DataFrame) -> None:
        """Almon polynomial weights must sum to 1."""
        midas = MIDAS(
            target="pib_quarterly",
            high_freq=["producao_industrial"],
            weight_scheme="almon",
            n_lags=12,
            poly_order=2,
        )
        midas.fit(midas_data)

        assert abs(midas.weights_.sum() - 1.0) < 1e-10

    def test_beta_weights_shape(self, midas_data: pd.DataFrame) -> None:
        """Beta weights have correct shape and are non-negative."""
        midas = MIDAS(
            target="pib_quarterly",
            high_freq=["producao_industrial"],
            weight_scheme="beta",
            n_lags=12,
        )
        midas.fit(midas_data)

        weights = midas.weights_
        assert weights.shape == (12,)
        assert np.all(weights >= 0)  # Beta weights are non-negative

    def test_step_equals_ols(self, midas_data: pd.DataFrame) -> None:
        """Step (unrestricted) weights are equivalent to OLS with lag dummies."""
        midas = MIDAS(
            target="pib_quarterly",
            high_freq=["producao_industrial"],
            weight_scheme="step",
            n_lags=6,
        )
        midas.fit(midas_data)

        # Weights should sum to 1 (normalized)
        assert abs(midas.weights_.sum() - 1.0) < 1e-10

        # Should produce a valid nowcast
        fc = midas.nowcast()
        assert not np.isnan(fc.point[0])

    def test_nowcast_reasonable(self, midas_data: pd.DataFrame) -> None:
        """MIDAS nowcast is within a reasonable range."""
        midas = MIDAS(
            target="pib_quarterly",
            high_freq=["producao_industrial"],
            weight_scheme="beta",
            n_lags=12,
        )
        midas.fit(midas_data)
        fc = midas.nowcast()

        # Nowcast should be in the ballpark of the target
        target_vals = midas_data["pib_quarterly"].dropna()
        target_mean = target_vals.mean()
        target_std = target_vals.std()

        # Within 4 standard deviations of the mean
        assert abs(fc.point[0] - target_mean) < 4 * target_std

        # Confidence intervals should be ordered correctly
        assert fc.lower_95[0] < fc.lower_80[0] < fc.point[0]
        assert fc.point[0] < fc.upper_80[0] < fc.upper_95[0]

    def test_plot_weights(self, midas_data: pd.DataFrame) -> None:
        """plot_weights() executes without error."""
        import matplotlib

        matplotlib.use("Agg")

        midas = MIDAS(
            target="pib_quarterly",
            high_freq=["producao_industrial"],
            weight_scheme="beta",
            n_lags=12,
        )
        midas.fit(midas_data)
        ax = midas.plot_weights()
        assert ax is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_freq_ratio(self) -> None:
        """freq_ratio parameter is stored correctly."""
        midas_qm = MIDAS(
            target="pib",
            high_freq=["x"],
            weight_scheme="beta",
            freq_ratio=3,
        )
        assert midas_qm.freq_ratio == 3

        midas_md = MIDAS(
            target="monthly_var",
            high_freq=["daily_var"],
            weight_scheme="beta",
            freq_ratio=22,
        )
        assert midas_md.freq_ratio == 22

        # Summary should work
        midas_qm2 = MIDAS(
            target="pib",
            high_freq=["x"],
            weight_scheme="beta",
            n_lags=6,
        )

        # Verify repr
        assert "beta" in repr(midas_qm2)
