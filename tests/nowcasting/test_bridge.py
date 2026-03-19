"""Tests for BridgeEquation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.nowcasting.bridge import BridgeEquation


@pytest.fixture
def bridge_data() -> pd.DataFrame:
    """Create synthetic monthly + quarterly data for bridge equation tests.

    Monthly indicators: producao_industrial, vendas_varejo, pmi
    Quarterly target: pib_quarterly (available at quarter-start months)
    """
    rng = np.random.default_rng(42)
    n_months = 60  # 5 years

    dates = pd.date_range("2019-01-01", periods=n_months, freq="MS")

    # Monthly indicators with some correlation to quarterly target
    pi = 100 + np.cumsum(rng.normal(0.1, 1.0, n_months))
    vv = 100 + np.cumsum(rng.normal(0.05, 0.8, n_months))
    pmi = 50 + rng.normal(0, 3, n_months)

    # Quarterly target: related to monthly indicators
    pib = np.full(n_months, np.nan)
    for t in range(0, n_months, 3):
        # PIB ~ weighted average of indicators over quarter
        end = min(t + 3, n_months)
        pi_q = np.mean(pi[t:end])
        vv_q = np.mean(vv[t:end])
        pmi_q = np.mean(pmi[t:end])
        pib[t] = 0.5 * pi_q + 0.3 * vv_q + 0.1 * pmi_q + rng.normal(0, 2)

    data = pd.DataFrame(
        {
            "producao_industrial": pi,
            "vendas_varejo": vv,
            "pmi": pmi,
            "pib_quarterly": pib,
        },
        index=dates,
    )
    return data


class TestBridgeEquation:
    """Tests for BridgeEquation."""

    def test_bridge_quarterly_prediction(self, bridge_data: pd.DataFrame) -> None:
        """Bridge equation predicts quarterly variable."""
        bridge = BridgeEquation(
            target="pib_quarterly",
            indicators=["producao_industrial", "vendas_varejo", "pmi"],
            aggregation="mean",
        )
        bridge.fit(bridge_data)
        nowcast = bridge.nowcast()

        assert nowcast.point is not None
        assert len(nowcast.point) == 1
        assert not np.isnan(nowcast.point[0])
        assert nowcast.lower_95[0] < nowcast.point[0] < nowcast.upper_95[0]

    def test_aggregation_methods(self, bridge_data: pd.DataFrame) -> None:
        """Different aggregation methods produce different results."""
        results = {}
        for agg in ["mean", "sum", "last"]:
            bridge = BridgeEquation(
                target="pib_quarterly",
                indicators=["producao_industrial", "vendas_varejo", "pmi"],
                aggregation=agg,
            )
            bridge.fit(bridge_data)
            fc = bridge.nowcast()
            results[agg] = fc.point[0]

        # At least two should be different
        values = list(results.values())
        assert not (
            np.allclose(values[0], values[1])
            and np.allclose(values[1], values[2])
        ), "All aggregation methods produced the same result"

    def test_partial_quarter(self, bridge_data: pd.DataFrame) -> None:
        """Nowcast works with 1 or 2 months of the current quarter."""
        # Remove last 2 months of indicators (partial quarter)
        partial_data = bridge_data.copy()
        partial_data.loc[partial_data.index[-2:], "producao_industrial"] = np.nan
        partial_data.loc[partial_data.index[-2:], "vendas_varejo"] = np.nan
        partial_data.loc[partial_data.index[-2:], "pmi"] = np.nan

        bridge = BridgeEquation(
            target="pib_quarterly",
            indicators=["producao_industrial", "vendas_varejo", "pmi"],
            aggregation="mean",
            fill_method="ar1",
        )
        bridge.fit(bridge_data)  # Fit on full data
        nowcast = bridge.nowcast(data=partial_data)  # Nowcast with partial data

        assert nowcast.point is not None
        assert not np.isnan(nowcast.point[0])

    def test_fill_ar1(self, bridge_data: pd.DataFrame) -> None:
        """fill_method='ar1' projects missing months via AR(1)."""
        bridge = BridgeEquation(
            target="pib_quarterly",
            indicators=["producao_industrial"],
            aggregation="mean",
            fill_method="ar1",
        )

        # Create data with missing last month
        monthly = bridge_data[["producao_industrial"]].copy()
        monthly.iloc[-1] = np.nan

        filled = bridge._fill_missing_months(monthly, "ar1")

        # Last value should be filled
        assert not pd.isna(filled.iloc[-1]["producao_industrial"])

        # AR(1) fill should be close to previous values
        last_filled = filled.iloc[-1]["producao_industrial"]
        prev_val = monthly.iloc[-2]["producao_industrial"]
        # Should be in reasonable range
        assert abs(last_filled - prev_val) < 20  # AR(1) shouldn't deviate wildly

    def test_coefficients(self, bridge_data: pd.DataFrame) -> None:
        """Estimated coefficients are returned correctly."""
        bridge = BridgeEquation(
            target="pib_quarterly",
            indicators=["producao_industrial", "vendas_varejo", "pmi"],
            aggregation="mean",
        )
        bridge.fit(bridge_data)

        coefs = bridge.coefficients()
        assert "variable" in coefs.columns
        assert "coefficient" in coefs.columns
        assert len(coefs) == 4  # intercept + 3 indicators
        assert coefs.iloc[0]["variable"] == "intercept"

        # Producao industrial should have a meaningful coefficient
        pi_coef = coefs[coefs["variable"] == "producao_industrial"][
            "coefficient"
        ].values[0]
        assert pi_coef != 0.0

    def test_r_squared(self, bridge_data: pd.DataFrame) -> None:
        """R-squared is positive for a fitted bridge equation."""
        bridge = BridgeEquation(
            target="pib_quarterly",
            indicators=["producao_industrial", "vendas_varejo", "pmi"],
            aggregation="mean",
        )
        bridge.fit(bridge_data)

        r2 = bridge.r_squared()
        assert r2 > 0.0
        assert r2 <= 1.0

        # Summary should contain R-squared
        summary_text = bridge.summary()
        assert "R-squared" in summary_text
        assert "Bridge Equation" in summary_text
