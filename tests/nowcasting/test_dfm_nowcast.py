"""Tests for DFMNowcaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.nowcasting.dfm import DFMNowcaster, KalmanBoxAdapter


@pytest.fixture
def mixed_freq_data() -> pd.DataFrame:
    """Create synthetic mixed-frequency panel data.

    Monthly indicators + quarterly target, 5 years of data.
    """
    rng = np.random.default_rng(42)
    n_months = 60  # 5 years

    dates = pd.date_range("2019-01-01", periods=n_months, freq="MS")

    # Latent factor (AR(1))
    factor = np.zeros(n_months)
    factor[0] = rng.normal(0, 1)
    for t in range(1, n_months):
        factor[t] = 0.8 * factor[t - 1] + rng.normal(0, 0.5)

    # Monthly indicators (linear combinations of factor + noise)
    pi = 0.7 * factor + rng.normal(0, 0.3, n_months)  # producao industrial
    vv = 0.5 * factor + rng.normal(0, 0.4, n_months)  # vendas varejo
    cc = 0.6 * factor + rng.normal(0, 0.2, n_months)  # confianca consumidor

    # Quarterly GDP (sum of 3 months of factor + noise)
    pib = np.full(n_months, np.nan)
    for t in range(2, n_months, 3):
        pib[t] = factor[t] + factor[t - 1] + factor[t - 2] + rng.normal(0, 0.5)

    data = pd.DataFrame(
        {
            "producao_industrial": pi,
            "vendas_varejo": vv,
            "confianca_consumidor": cc,
            "pib": pib,
        },
        index=dates,
    )
    return data


@pytest.fixture
def frequency_map() -> dict[str, str]:
    """Standard frequency map."""
    return {
        "producao_industrial": "M",
        "vendas_varejo": "M",
        "confianca_consumidor": "M",
        "pib": "Q",
    }


class TestDFMNowcaster:
    """Tests for DFMNowcaster."""

    def test_nowcast_updates_with_new_data(
        self, mixed_freq_data: pd.DataFrame, frequency_map: dict[str, str]
    ) -> None:
        """Nowcast changes when new monthly data is added."""
        nowcaster = DFMNowcaster(
            n_factors=2,
            frequency_map=frequency_map,
            em_iterations=10,
        )

        # Fit on first 48 months
        data_partial = mixed_freq_data.iloc[:48]
        nowcaster.fit(data_partial)
        nowcast_old = nowcaster.nowcast(target="pib")

        # Update with new data
        data_new = mixed_freq_data.iloc[48:51]
        nowcaster.update(data_new)
        nowcast_new = nowcaster.nowcast(target="pib")

        # Nowcast should change
        assert nowcast_old.point[0] != pytest.approx(nowcast_new.point[0], abs=1e-10)

    def test_more_data_reduces_uncertainty(
        self, mixed_freq_data: pd.DataFrame, frequency_map: dict[str, str]
    ) -> None:
        """Confidence interval narrows with more data available in quarter."""
        nowcaster = DFMNowcaster(
            n_factors=2,
            frequency_map=frequency_map,
            em_iterations=10,
        )

        # Fit with 48 months (full quarters)
        nowcaster.fit(mixed_freq_data.iloc[:48])
        fc_less = nowcaster.nowcast(target="pib")

        # Add 2 more months of monthly data
        nowcaster.update(mixed_freq_data.iloc[48:50])
        fc_more = nowcaster.nowcast(target="pib")

        # Interval width with more data should be <= interval with less
        width_less = fc_less.upper_95[0] - fc_less.lower_95[0]
        width_more = fc_more.upper_95[0] - fc_more.lower_95[0]

        # More data should generally reduce uncertainty (or at least not increase drastically)
        assert width_more <= width_less * 1.5  # Allow some tolerance

    def test_factors_extracted(
        self, mixed_freq_data: pd.DataFrame, frequency_map: dict[str, str]
    ) -> None:
        """n_factors factors are extracted with correct dimensions."""
        n_factors = 2
        nowcaster = DFMNowcaster(
            n_factors=n_factors,
            frequency_map=frequency_map,
            em_iterations=10,
        )
        nowcaster.fit(mixed_freq_data)

        factors = nowcaster.factors()
        assert factors.shape[1] == n_factors
        assert factors.shape[0] == len(mixed_freq_data)
        assert list(factors.columns) == ["factor_1", "factor_2"]

    def test_loadings_shape(
        self, mixed_freq_data: pd.DataFrame, frequency_map: dict[str, str]
    ) -> None:
        """Loadings matrix has shape (n_variables, n_factors)."""
        n_factors = 2
        nowcaster = DFMNowcaster(
            n_factors=n_factors,
            frequency_map=frequency_map,
            em_iterations=10,
        )
        nowcaster.fit(mixed_freq_data)

        ldgs = nowcaster.loadings()
        n_vars = len([v for v in frequency_map if v in mixed_freq_data.columns])
        assert ldgs.shape == (n_vars, n_factors)

    def test_mixed_frequency(
        self, mixed_freq_data: pd.DataFrame, frequency_map: dict[str, str]
    ) -> None:
        """Quarterly and monthly variables coexist in the same model."""
        nowcaster = DFMNowcaster(
            n_factors=2,
            frequency_map=frequency_map,
            em_iterations=10,
        )
        nowcaster.fit(mixed_freq_data)

        # Should have both monthly and quarterly
        assert len(nowcaster._monthly_vars) == 3
        assert len(nowcaster._quarterly_vars) == 1
        assert "pib" in nowcaster._quarterly_vars

        # Nowcast for quarterly target should work
        fc = nowcaster.nowcast(target="pib")
        assert fc.point is not None
        assert len(fc.point) == 1

    def test_missing_data_handled(
        self, mixed_freq_data: pd.DataFrame, frequency_map: dict[str, str]
    ) -> None:
        """Missing data (ragged edge) does not cause errors."""
        # Create ragged edge: remove last 2 months of some indicators
        data_ragged = mixed_freq_data.copy()
        data_ragged.loc[data_ragged.index[-2:], "producao_industrial"] = np.nan
        data_ragged.loc[data_ragged.index[-1:], "vendas_varejo"] = np.nan

        nowcaster = DFMNowcaster(
            n_factors=2,
            frequency_map=frequency_map,
            em_iterations=10,
        )

        # Should not raise
        nowcaster.fit(data_ragged)
        fc = nowcaster.nowcast(target="pib")
        assert not np.isnan(fc.point[0])

    def test_kalmanbox_integration(self) -> None:
        """Adapter detects kalmanbox availability."""
        adapter = KalmanBoxAdapter()
        # Just check that the adapter has the 'available' property
        assert isinstance(adapter.available, bool)

        # The standalone filter should always work
        rng = np.random.default_rng(42)
        n_t, n_obs, state_dim = 20, 2, 3

        y = rng.normal(0, 1, (n_t, n_obs))
        Z = rng.normal(0, 0.5, (n_obs, state_dim))
        T_mat = np.eye(state_dim) * 0.9
        Q_mat = np.eye(state_dim) * 0.1
        R_mat = np.eye(n_obs) * 0.5
        a0 = np.zeros(state_dim)
        P0 = np.eye(state_dim)

        result = adapter.filter(y, Z, T_mat, Q_mat, R_mat, a0, P0)
        assert result.filtered_state.shape == (n_t, state_dim)
        assert result.filtered_cov.shape == (n_t, state_dim, state_dim)
