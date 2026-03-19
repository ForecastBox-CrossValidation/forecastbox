"""Tests for ForecastHorizon and MultiHorizon."""

from __future__ import annotations

import pytest

from forecastbox.core.horizon import (
    ForecastHorizon,
    MultiHorizon,
    h_step_ahead,
    quarterly_from_monthly,
)


class TestForecastHorizon:
    """Tests for ForecastHorizon."""

    def test_horizon_monthly(self) -> None:
        """ForecastHorizon(12, 'MS', '2024-01-01') generates 12 months."""
        fh = ForecastHorizon(12, "MS", "2024-01-01")
        idx = fh.to_index()
        assert len(idx) == 12
        assert str(idx[0].date()) == "2024-02-01"
        assert str(idx[-1].date()) == "2025-01-01"

    def test_horizon_quarterly(self) -> None:
        """ForecastHorizon(4, 'QS', '2024-01-01') generates 4 quarters."""
        fh = ForecastHorizon(4, "QS", "2024-01-01")
        idx = fh.to_index()
        assert len(idx) == 4

    def test_horizon_len(self) -> None:
        """len() returns h."""
        fh = ForecastHorizon(6)
        assert len(fh) == 6

    def test_horizon_iter(self) -> None:
        """Iteration produces 1..h."""
        fh = ForecastHorizon(3)
        assert list(fh) == [1, 2, 3]

    def test_horizon_invalid(self) -> None:
        """h < 1 raises ValueError."""
        with pytest.raises(ValueError):
            ForecastHorizon(0)

    def test_h_step_ahead(self) -> None:
        """h_step_ahead generates correct index."""
        idx = h_step_ahead(6, "MS", "2024-01-01")
        assert len(idx) == 6

    def test_quarterly_from_monthly(self) -> None:
        """quarterly_from_monthly generates quarterly dates."""
        idx = quarterly_from_monthly("2024-01-01", 4)
        assert len(idx) == 4


class TestMultiHorizon:
    """Tests for MultiHorizon."""

    def test_multi_horizon(self) -> None:
        """MultiHorizon([1, 3, 6, 12]) has 4 elements."""
        mh = MultiHorizon([1, 3, 6, 12])
        assert len(mh) == 4
        assert list(mh) == [1, 3, 6, 12]

    def test_multi_horizon_contains(self) -> None:
        """Contains check works."""
        mh = MultiHorizon([1, 3, 6, 12])
        assert 3 in mh
        assert 5 not in mh

    def test_multi_horizon_empty(self) -> None:
        """Empty list raises ValueError."""
        with pytest.raises(ValueError):
            MultiHorizon([])
