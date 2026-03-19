"""Tests for RealTimeDataManager."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from forecastbox.nowcasting.realtime import RealTimeDataManager, SeriesInfo


@pytest.fixture
def rtdm_with_data() -> RealTimeDataManager:
    """Create a RealTimeDataManager with sample series and data."""
    rtdm = RealTimeDataManager()

    # Monthly series: producao industrial, lag 45 days
    monthly_idx = pd.date_range("2024-01-01", periods=3, freq="MS")
    monthly_data = pd.Series([100.0, 101.5, 102.3], index=monthly_idx)
    rtdm.add_series(
        "producao_industrial",
        frequency="M",
        release_calendar="monthly",
        lag_days=45,
        source="IBGE",
        data=monthly_data,
    )

    # Monthly series: confianca consumidor, lag 10 days
    conf_idx = pd.date_range("2024-01-01", periods=4, freq="MS")
    conf_data = pd.Series([95.0, 96.2, 97.1, 95.8], index=conf_idx)
    rtdm.add_series(
        "confianca_consumidor",
        frequency="M",
        release_calendar="monthly",
        lag_days=10,
        source="FGV",
        data=conf_data,
    )

    # Quarterly series: PIB, lag 60 days
    q_idx = pd.to_datetime(["2023-10-01", "2024-01-01"])
    q_data = pd.Series([1.2, 0.8], index=q_idx)
    rtdm.add_series(
        "pib",
        frequency="Q",
        release_calendar="quarterly",
        lag_days=60,
        source="IBGE",
        data=q_data,
    )

    return rtdm


class TestRealTimeDataManager:
    """Tests for RealTimeDataManager."""

    def test_add_series(self) -> None:
        """Test adding series with different frequencies and lags."""
        rtdm = RealTimeDataManager()
        rtdm.add_series("ipca", frequency="M", release_calendar="monthly", lag_days=15)
        rtdm.add_series("pib", frequency="Q", release_calendar="quarterly", lag_days=60)
        rtdm.add_series("selic", frequency="D", release_calendar="daily", lag_days=0)

        assert "ipca" in rtdm.series
        assert "pib" in rtdm.series
        assert "selic" in rtdm.series
        assert rtdm.series["ipca"].frequency == "M"
        assert rtdm.series["pib"].frequency == "Q"
        assert rtdm.series["selic"].lag_days == 0
        assert len(rtdm.series_names) == 3

    def test_ragged_edge(self, rtdm_with_data: RealTimeDataManager) -> None:
        """Test ragged edge returns correct last observation dates."""
        # Reference: 2024-03-15
        # producao_industrial: jan (pub 2024-02-15 ok),
        #   feb (pub 2024-03-17 NOT ok), mar (pub 2024-04-16 NOT)
        # confianca_consumidor: jan (pub 2024-02-10 ok),
        #   feb (pub 2024-03-10 ok), mar (pub 2024-04-10 NOT),
        #   apr (pub 2024-05-10 NOT)
        # pib: Q4/23 (pub 2024-03-01 ok), Q1/24 (pub 2024-04-01 NOT)
        ragged = rtdm_with_data.get_ragged_edge("2024-03-15")

        assert ragged["producao_industrial"] == date(2024, 1, 1)
        assert ragged["confianca_consumidor"] == date(2024, 2, 1)
        assert ragged["pib"] == date(2023, 10, 1)

    def test_available_data(self, rtdm_with_data: RealTimeDataManager) -> None:
        """Test get_available_data returns only data published before reference date."""
        available = rtdm_with_data.get_available_data("2024-03-15")

        # producao_industrial: only jan available
        assert "producao_industrial" in available.columns
        pi_valid = available["producao_industrial"].dropna()
        assert len(pi_valid) == 1
        assert pi_valid.iloc[0] == pytest.approx(100.0)

        # confianca_consumidor: jan and feb available
        cc_valid = available["confianca_consumidor"].dropna()
        assert len(cc_valid) == 2

    def test_missing_pattern(self, rtdm_with_data: RealTimeDataManager) -> None:
        """Test missing pattern panel is correct."""
        pattern = rtdm_with_data.get_missing_pattern("2024-03-15")

        assert not pattern.empty
        # Should be a boolean DataFrame
        assert pattern.dtypes.apply(lambda x: x is bool).all()

    def test_simulate_publication(self, rtdm_with_data: RealTimeDataManager) -> None:
        """Test simulation respects calendar and lags."""
        events = rtdm_with_data.simulate_publication("2024-02-01", "2024-04-30")

        assert len(events) > 0
        # Events should be sorted by date
        dates = [e["date"] for e in events]
        assert dates == sorted(dates)

        # Each event should have required keys
        for event in events:
            assert "date" in event
            assert "series" in event
            assert "period" in event
            assert "lag_days" in event

    def test_update_vintage(self, rtdm_with_data: RealTimeDataManager) -> None:
        """Test update creates new vintage in the data store."""
        # Get initial vintage count
        initial_vintages = len(rtdm_with_data._vintages["producao_industrial"])

        # Update with new data
        new_idx = pd.date_range("2024-04-01", periods=1, freq="MS")
        new_data = pd.Series([103.0], index=new_idx)
        rtdm_with_data.update({"producao_industrial": new_data})

        # Check vintage was created
        assert len(rtdm_with_data._vintages["producao_industrial"]) == initial_vintages + 1

        # Check data was updated
        pi_data = rtdm_with_data._data["producao_industrial"]
        assert len(pi_data) == 4  # 3 original + 1 new
        assert pi_data.iloc[-1] == pytest.approx(103.0)


class TestSeriesInfo:
    """Tests for SeriesInfo dataclass."""

    def test_valid_series_info(self) -> None:
        """Test creating a valid SeriesInfo."""
        info = SeriesInfo(
            name="ipca",
            frequency="M",
            release_calendar="monthly",
            lag_days=15,
            source="IBGE",
            transform="log_diff",
        )
        assert info.name == "ipca"
        assert info.frequency == "M"

    def test_invalid_frequency(self) -> None:
        """Test that invalid frequency raises ValueError."""
        with pytest.raises(ValueError, match="frequency"):
            SeriesInfo(name="x", frequency="X", release_calendar="monthly", lag_days=0)

    def test_invalid_release_calendar(self) -> None:
        """Test that invalid release_calendar raises ValueError."""
        with pytest.raises(ValueError, match="release_calendar"):
            SeriesInfo(name="x", frequency="M", release_calendar="biweekly", lag_days=0)

    def test_negative_lag_days(self) -> None:
        """Test that negative lag_days raises ValueError."""
        with pytest.raises(ValueError, match="lag_days"):
            SeriesInfo(name="x", frequency="M", release_calendar="monthly", lag_days=-5)
