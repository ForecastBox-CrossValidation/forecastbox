"""Tests for NewsDecomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.nowcasting.dfm import DFMNowcaster
from forecastbox.nowcasting.news import NewsDecomposition


@pytest.fixture
def fitted_nowcaster() -> DFMNowcaster:
    """Create and fit a DFMNowcaster on synthetic data."""
    rng = np.random.default_rng(42)
    n_months = 60

    dates = pd.date_range("2019-01-01", periods=n_months, freq="MS")

    # Latent factor
    factor = np.zeros(n_months)
    factor[0] = rng.normal(0, 1)
    for t in range(1, n_months):
        factor[t] = 0.8 * factor[t - 1] + rng.normal(0, 0.5)

    pi = 0.7 * factor + rng.normal(0, 0.3, n_months)
    vv = 0.5 * factor + rng.normal(0, 0.4, n_months)
    cc = 0.6 * factor + rng.normal(0, 0.2, n_months)

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

    frequency_map = {
        "producao_industrial": "M",
        "vendas_varejo": "M",
        "confianca_consumidor": "M",
        "pib": "Q",
    }

    nowcaster = DFMNowcaster(
        n_factors=1,
        frequency_map=frequency_map,
        em_iterations=5,
    )
    nowcaster.fit(data)
    return nowcaster


@pytest.fixture
def old_and_new_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create old and new vintage data (new vintage has extra observations)."""
    rng = np.random.default_rng(42)
    n_months = 60

    dates = pd.date_range("2019-01-01", periods=n_months, freq="MS")

    factor = np.zeros(n_months)
    factor[0] = rng.normal(0, 1)
    for t in range(1, n_months):
        factor[t] = 0.8 * factor[t - 1] + rng.normal(0, 0.5)

    pi = 0.7 * factor + rng.normal(0, 0.3, n_months)
    vv = 0.5 * factor + rng.normal(0, 0.4, n_months)
    cc = 0.6 * factor + rng.normal(0, 0.2, n_months)

    pib = np.full(n_months, np.nan)
    for t in range(2, n_months, 3):
        pib[t] = factor[t] + factor[t - 1] + factor[t - 2] + rng.normal(0, 0.5)

    # Old data: first 54 months
    old_data = pd.DataFrame(
        {
            "producao_industrial": pi[:54],
            "vendas_varejo": vv[:54],
            "confianca_consumidor": cc[:54],
            "pib": pib[:54],
        },
        index=dates[:54],
    )

    # New data: first 57 months (3 new monthly observations)
    new_data = pd.DataFrame(
        {
            "producao_industrial": pi[:57],
            "vendas_varejo": vv[:57],
            "confianca_consumidor": cc[:57],
            "pib": pib[:57],
        },
        index=dates[:57],
    )

    return old_data, new_data


class TestNewsDecomposition:
    """Tests for NewsDecomposition."""

    def test_decomposition_exact(
        self,
        fitted_nowcaster: DFMNowcaster,
        old_and_new_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Sum of contributions equals total revision (exact decomposition)."""
        old_data, new_data = old_and_new_data

        news_decomp = NewsDecomposition(fitted_nowcaster)
        result = news_decomp.decompose(old_data, new_data, target="pib")

        # Exact decomposition
        contrib_sum = sum(result.contributions.values())
        assert abs(contrib_sum - result.total_revision) < 1e-10

    def test_no_news_no_revision(
        self,
        fitted_nowcaster: DFMNowcaster,
        old_and_new_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """If old_data == new_data, total revision should be 0."""
        old_data, _ = old_and_new_data

        news_decomp = NewsDecomposition(fitted_nowcaster)
        result = news_decomp.decompose(old_data, old_data, target="pib")

        assert abs(result.total_revision) < 1e-10

    def test_positive_surprise_positive_contribution(
        self,
        fitted_nowcaster: DFMNowcaster,
        old_and_new_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Positive news with positive loading leads to positive contribution."""
        old_data, new_data = old_and_new_data

        news_decomp = NewsDecomposition(fitted_nowcaster)
        result = news_decomp.decompose(old_data, new_data, target="pib")

        # For indicators with positive news AND positive weight,
        # contribution should be positive
        for indicator in result.contributions:
            news_val = result.news.get(indicator, 0.0)
            weight_val = result.weights.get(indicator, 0.0)
            contrib_val = result.contributions[indicator]

            # contribution = weight * news (approximately, up to scaling)
            if abs(news_val) > 1e-10 and abs(weight_val) > 1e-10:
                expected_sign = np.sign(news_val * weight_val)
                actual_sign = np.sign(contrib_val)
                # Signs should match
                assert expected_sign == actual_sign or abs(contrib_val) < 1e-8

    def test_contributions_sum(
        self,
        fitted_nowcaster: DFMNowcaster,
        old_and_new_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Individual contributions sum to total revision."""
        old_data, new_data = old_and_new_data

        news_decomp = NewsDecomposition(fitted_nowcaster)
        result = news_decomp.decompose(old_data, new_data, target="pib")

        contrib_sum = sum(result.contributions.values())
        assert abs(contrib_sum - result.total_revision) < 1e-10
        assert result.new_nowcast == pytest.approx(
            result.old_nowcast + result.total_revision, abs=1e-10
        )

    def test_plot_contributions(
        self,
        fitted_nowcaster: DFMNowcaster,
        old_and_new_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """plot_contributions() and plot_waterfall() execute without error."""
        import matplotlib

        matplotlib.use("Agg")

        old_data, new_data = old_and_new_data

        news_decomp = NewsDecomposition(fitted_nowcaster)
        result = news_decomp.decompose(old_data, new_data, target="pib")

        ax1 = result.plot_contributions()
        assert ax1 is not None

        ax2 = result.plot_waterfall()
        assert ax2 is not None

        import matplotlib.pyplot as plt

        plt.close("all")

    def test_dataframe_columns(
        self,
        fitted_nowcaster: DFMNowcaster,
        old_and_new_data: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """to_dataframe() has columns [indicator, news, weight, contribution]."""
        old_data, new_data = old_and_new_data

        news_decomp = NewsDecomposition(fitted_nowcaster)
        result = news_decomp.decompose(old_data, new_data, target="pib")

        df = result.to_dataframe()
        assert "indicator" in df.columns
        assert "news" in df.columns
        assert "weight" in df.columns
        assert "contribution" in df.columns

        # Should have at least one row
        assert len(df) > 0

        # Summary should work
        summary_text = result.summary()
        assert "News Decomposition" in summary_text
        assert "Total revision" in summary_text
