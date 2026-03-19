"""Tests for DataVintage."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from forecastbox.core.vintage import DataVintage


class TestDataVintage:
    """Tests for DataVintage."""

    def _make_vintage(self) -> DataVintage:
        """Helper to create a DataVintage with 3 releases."""
        dv = DataVintage("test_series")
        idx = pd.date_range("2024-01-01", periods=6, freq="MS")

        # First release: 4 months available
        dv.add_vintage(date(2024, 5, 1), pd.Series([1.0, 1.1, 1.2, 1.3], index=idx[:4]))
        # Second release: 5 months, some revisions
        dv.add_vintage(date(2024, 6, 1), pd.Series([1.0, 1.15, 1.18, 1.3, 1.4], index=idx[:5]))
        # Third release: 6 months
        dv.add_vintage(date(2024, 7, 1), pd.Series([1.0, 1.15, 1.2, 1.32, 1.4, 1.5], index=idx[:6]))

        return dv

    def test_add_vintage(self) -> None:
        """Add 3 vintages, verify len(release_dates) == 3."""
        dv = self._make_vintage()
        assert len(dv.release_dates) == 3

    def test_get_vintage(self) -> None:
        """Returns data available at release_date."""
        dv = self._make_vintage()
        v1 = dv.get_vintage(date(2024, 5, 1))
        assert len(v1) == 4

    def test_get_latest(self) -> None:
        """Returns latest vintage."""
        dv = self._make_vintage()
        latest = dv.get_latest()
        assert len(latest) == 6

    def test_revision(self) -> None:
        """Revision between vintages calculated correctly."""
        dv = self._make_vintage()
        rev = dv.get_revision("2024-02-01", date(2024, 5, 1), date(2024, 6, 1))
        assert rev == pytest.approx(0.05)  # 1.15 - 1.10

    def test_revision_history(self) -> None:
        """Revision history for a period."""
        dv = self._make_vintage()
        history = dv.revision_history("2024-02-01")
        assert len(history) == 3

    def test_to_dataframe(self) -> None:
        """Vintage matrix has correct format."""
        dv = self._make_vintage()
        df = dv.to_dataframe()
        assert df.shape[1] == 3  # 3 releases

    def test_triangle(self) -> None:
        """Triangle has correct format."""
        dv = self._make_vintage()
        tri = dv.triangle()
        assert tri.shape[1] == 3
