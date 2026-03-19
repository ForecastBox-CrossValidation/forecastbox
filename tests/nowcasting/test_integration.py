"""End-to-end integration test for the nowcasting module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.nowcasting import (
    MIDAS,
    BridgeEquation,
    DFMNowcaster,
    NewsDecomposition,
    NewsResult,
    RealTimeDataManager,
    SeriesInfo,  # noqa: F401
)
from forecastbox.nowcasting import (
    __all__ as _nowcasting_all,
)


@pytest.fixture
def full_dataset() -> pd.DataFrame:
    """Create a comprehensive synthetic dataset for integration testing.

    Contains monthly indicators and quarterly target over 8 years.
    """
    rng = np.random.default_rng(42)
    n_months = 96  # 8 years

    dates = pd.date_range("2016-01-01", periods=n_months, freq="MS")

    # Latent factor (AR(1))
    factor = np.zeros(n_months)
    factor[0] = rng.normal(0, 1)
    for t in range(1, n_months):
        factor[t] = 0.8 * factor[t - 1] + rng.normal(0, 0.5)

    # Monthly indicators
    producao_industrial = 100 + 0.7 * factor + rng.normal(0, 0.3, n_months)
    vendas_varejo = 100 + 0.5 * factor + rng.normal(0, 0.4, n_months)
    confianca_consumidor = 50 + 0.6 * factor + rng.normal(0, 0.2, n_months)
    pmi = 50 + 0.4 * factor + rng.normal(0, 0.5, n_months)

    # Quarterly GDP (sum of 3 monthly factors + noise)
    pib = np.full(n_months, np.nan)
    for t in range(2, n_months, 3):
        pib[t] = (
            2.0
            + 0.5 * np.mean(producao_industrial[max(0, t - 2) : t + 1])
            + 0.3 * np.mean(vendas_varejo[max(0, t - 2) : t + 1])
            + rng.normal(0, 1)
        )

    data = pd.DataFrame(
        {
            "producao_industrial": producao_industrial,
            "vendas_varejo": vendas_varejo,
            "confianca_consumidor": confianca_consumidor,
            "pmi": pmi,
            "pib": pib,
        },
        index=dates,
    )
    return data


class TestNowcastingIntegration:
    """End-to-end integration tests for the nowcasting pipeline."""

    def test_full_pipeline(self, full_dataset: pd.DataFrame) -> None:
        """Complete nowcasting pipeline: data -> models -> nowcast -> news.

        This test exercises the entire module in a realistic scenario.
        """
        import matplotlib
        matplotlib.use("Agg")

        data = full_dataset

        # --- Step 1: RealTimeDataManager ---
        rtdm = RealTimeDataManager()
        rtdm.add_series(
            "producao_industrial", frequency="M", release_calendar="monthly",
            lag_days=45, data=data["producao_industrial"],
        )
        rtdm.add_series(
            "vendas_varejo", frequency="M", release_calendar="monthly",
            lag_days=40, data=data["vendas_varejo"],
        )
        rtdm.add_series(
            "confianca_consumidor", frequency="M", release_calendar="monthly",
            lag_days=10, data=data["confianca_consumidor"],
        )
        rtdm.add_series(
            "pib", frequency="Q", release_calendar="quarterly",
            lag_days=60, data=data["pib"].dropna(),
        )

        # Verify ragged edge
        ragged = rtdm.get_ragged_edge("2023-06-15")
        assert isinstance(ragged, dict)

        # --- Step 2: DFMNowcaster ---
        frequency_map = {
            "producao_industrial": "M",
            "vendas_varejo": "M",
            "confianca_consumidor": "M",
            "pib": "Q",
        }

        # Use first 84 months for training
        train_data = data.iloc[:84]

        nowcaster = DFMNowcaster(
            n_factors=2,
            frequency_map=frequency_map,
            em_iterations=10,
        )
        nowcaster.fit(train_data)

        # Verify factors
        factors = nowcaster.factors()
        assert factors.shape == (84, 2)

        # Verify loadings
        ldgs = nowcaster.loadings()
        assert ldgs.shape == (4, 2)

        # Generate nowcast
        fc_dfm = nowcaster.nowcast(target="pib")
        assert fc_dfm.point is not None
        assert len(fc_dfm.point) == 1
        assert fc_dfm.lower_95[0] < fc_dfm.point[0] < fc_dfm.upper_95[0]

        # --- Step 3: BridgeEquation ---
        # Bridge uses resample("QS") which produces quarter-start dates.
        # We need pib values aligned to quarter-start months.
        bridge_data = train_data.copy()
        pib_q = bridge_data["pib"].dropna()
        pib_bridge = pd.Series(np.nan, index=bridge_data.index)
        for dt, val in pib_q.items():
            # Shift quarter-end to quarter-start (e.g. Mar -> Jan)
            qs = pd.Timestamp(dt) - pd.DateOffset(months=2)
            if qs in pib_bridge.index:
                pib_bridge.loc[qs] = val
        bridge_data["pib"] = pib_bridge

        bridge = BridgeEquation(
            target="pib",
            indicators=["producao_industrial", "vendas_varejo", "confianca_consumidor"],
            aggregation="mean",
        )
        bridge.fit(bridge_data)

        fc_bridge = bridge.nowcast()
        assert fc_bridge.point is not None
        assert not np.isnan(fc_bridge.point[0])

        r2 = bridge.r_squared()
        assert r2 > 0.0

        # --- Step 4: MIDAS ---
        midas = MIDAS(
            target="pib",
            high_freq=["producao_industrial"],
            weight_scheme="beta",
            n_lags=12,
        )
        midas.fit(train_data)

        fc_midas = midas.nowcast()
        assert fc_midas.point is not None
        assert abs(midas.weights_.sum() - 1.0) < 1e-10

        # --- Step 5: NewsDecomposition ---
        old_data = data.iloc[:84]
        new_data = data.iloc[:87]  # 3 new monthly observations

        news = NewsDecomposition(nowcaster)
        result = news.decompose(old_data, new_data, target="pib")

        assert isinstance(result, NewsResult)
        assert abs(sum(result.contributions.values()) - result.total_revision) < 1e-10

        # Verify to_dataframe
        df = result.to_dataframe()
        assert "indicator" in df.columns
        assert "contribution" in df.columns

        # --- Step 6: Verify all exports are present ---
        expected = {
            "BridgeEquation", "DFMNowcaster", "MIDAS",
            "NewsDecomposition", "NewsResult",
            "RealTimeDataManager", "SeriesInfo",
        }
        assert expected <= set(_nowcasting_all)

    def test_multiple_models_comparison(self, full_dataset: pd.DataFrame) -> None:
        """Compare nowcasts from DFM, Bridge, and MIDAS."""
        data = full_dataset.iloc[:84]

        frequency_map = {
            "producao_industrial": "M",
            "vendas_varejo": "M",
            "confianca_consumidor": "M",
            "pib": "Q",
        }

        # DFM
        dfm = DFMNowcaster(n_factors=2, frequency_map=frequency_map, em_iterations=10)
        dfm.fit(data)
        fc_dfm = dfm.nowcast(target="pib")

        # Bridge (needs quarter-start aligned pib)
        bridge_data = data.copy()
        pib_q = bridge_data["pib"].dropna()
        pib_bridge = pd.Series(np.nan, index=bridge_data.index)
        for dt, val in pib_q.items():
            qs = pd.Timestamp(dt) - pd.DateOffset(months=2)
            if qs in pib_bridge.index:
                pib_bridge.loc[qs] = val
        bridge_data["pib"] = pib_bridge

        bridge = BridgeEquation(
            target="pib",
            indicators=["producao_industrial", "vendas_varejo"],
            aggregation="mean",
        )
        bridge.fit(bridge_data)
        fc_bridge = bridge.nowcast()

        # MIDAS
        midas = MIDAS(
            target="pib",
            high_freq=["producao_industrial"],
            weight_scheme="beta",
            n_lags=9,
        )
        midas.fit(data)
        fc_midas = midas.nowcast()

        # All should produce non-NaN predictions
        assert not np.isnan(fc_dfm.point[0])
        assert not np.isnan(fc_bridge.point[0])
        assert not np.isnan(fc_midas.point[0])

        # At least two methods should give different results
        values = [fc_dfm.point[0], fc_bridge.point[0], fc_midas.point[0]]
        assert not (
            np.allclose(values[0], values[1])
            and np.allclose(values[1], values[2])
        )

    def test_update_and_news_flow(self, full_dataset: pd.DataFrame) -> None:
        """Simulate the real-time flow: fit -> nowcast -> update -> news."""
        data = full_dataset

        frequency_map = {
            "producao_industrial": "M",
            "vendas_varejo": "M",
            "pib": "Q",
        }

        cols = ["producao_industrial", "vendas_varejo", "pib"]

        # Initial fit with more EM iterations for stability
        nowcaster = DFMNowcaster(
            n_factors=1,
            frequency_map=frequency_map,
            em_iterations=20,
        )
        initial_data = data[cols].iloc[:78]
        nowcaster.fit(initial_data)
        fc_initial = nowcaster.nowcast(target="pib")

        # Update with new data
        extended_data = data[cols].iloc[78:81]
        nowcaster.update(extended_data)
        fc_updated = nowcaster.nowcast(target="pib")

        # Nowcast should have changed
        assert fc_initial.point[0] != pytest.approx(fc_updated.point[0], abs=1e-10)

        # News decomposition
        old = data[cols].iloc[:78]
        new = data[cols].iloc[:81]

        news = NewsDecomposition(nowcaster)
        result = news.decompose(old, new, target="pib")

        # Revision should match
        assert abs(sum(result.contributions.values()) - result.total_revision) < 1e-10
