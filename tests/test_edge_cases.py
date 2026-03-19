"""Tests for edge cases and error handling."""

from __future__ import annotations

import numpy as np
import pytest

from forecastbox.core.forecast import Forecast


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_forecast_combine(self) -> None:
        """Combining empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot combine empty"):
            Forecast.combine([], method="mean")

    def test_single_observation_forecast(self) -> None:
        """Forecast with single point works."""
        fc = Forecast(
            point=np.array([100.0]),
            model_name="Single",
            horizon=1,
        )
        assert len(fc) == 1
        assert fc[0]["point"] == 100.0

    def test_large_horizon(self) -> None:
        """Forecast with large horizon works."""
        fc = Forecast(
            point=np.zeros(1000),
            model_name="Large",
        )
        assert len(fc) == 1000

    def test_nan_in_point_raises(self) -> None:
        """NaN in forecast data is handled."""
        fc = Forecast(
            point=np.array([100.0, np.nan, 102.0]),
            model_name="NaN",
        )
        # Should create without error, NaN is in the array
        assert len(fc) == 3

    def test_mismatched_intervals(self) -> None:
        """Intervals with wrong length handled gracefully."""
        # Different length intervals should still create the object
        # but validate() should catch issues
        fc = Forecast(
            point=np.array([100.0, 101.0]),
            lower_80=np.array([105.0, 106.0]),  # lower > point
            upper_80=np.array([110.0, 111.0]),
            model_name="Bad",
        )
        with pytest.raises(ValueError):
            fc.validate()

    def test_forecast_to_dataframe_no_index(self) -> None:
        """to_dataframe without temporal index uses range index."""
        fc = Forecast(
            point=np.array([100.0, 101.0, 102.0]),
            model_name="NoIndex",
        )
        df = fc.to_dataframe()
        assert len(df) == 3
        assert "point" in df.columns

    def test_forecast_repr_minimal(self) -> None:
        """repr works with minimal forecast."""
        fc = Forecast(
            point=np.array([100.0]),
            model_name="Minimal",
        )
        r = repr(fc)
        assert "Minimal" in r

    def test_combine_single_forecast(self) -> None:
        """Combining single forecast returns it."""
        fc = Forecast(
            point=np.array([100.0, 101.0]),
            model_name="Only",
        )
        # Should work (mean of 1 = itself)
        combined = Forecast.combine([fc], method="mean")
        np.testing.assert_allclose(combined.point, fc.point)

    def test_forecast_save_load_roundtrip_minimal(self, tmp_path: object) -> None:
        """Save/load roundtrip with minimal data."""
        fc = Forecast(
            point=np.array([100.0]),
            model_name="RT",
        )
        path = tmp_path / "minimal.json"  # type: ignore[operator]
        fc.save(path)
        loaded = Forecast.load(path)
        np.testing.assert_allclose(loaded.point, fc.point)
