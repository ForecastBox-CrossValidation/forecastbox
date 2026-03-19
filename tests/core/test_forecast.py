"""Tests for Forecast container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.core.forecast import Forecast


class TestForecast:
    """Tests for Forecast container."""

    def test_create_forecast(
        self, sample_forecast_data: dict, sample_index: pd.DatetimeIndex
    ) -> None:
        """Create Forecast with all fields, verify attributes."""
        fc = Forecast(
            point=sample_forecast_data["point"],
            lower_80=sample_forecast_data["lower_80"],
            upper_80=sample_forecast_data["upper_80"],
            lower_95=sample_forecast_data["lower_95"],
            upper_95=sample_forecast_data["upper_95"],
            index=sample_index,
            model_name="TestModel",
            horizon=3,
        )
        assert fc.model_name == "TestModel"
        assert fc.horizon == 3
        assert len(fc.point) == 3
        assert fc.index is not None

    def test_forecast_validate(
        self, sample_forecast_data: dict, sample_index: pd.DatetimeIndex
    ) -> None:
        """Validate lower < point < upper for all horizons."""
        fc = Forecast(
            point=sample_forecast_data["point"],
            lower_80=sample_forecast_data["lower_80"],
            upper_80=sample_forecast_data["upper_80"],
            lower_95=sample_forecast_data["lower_95"],
            upper_95=sample_forecast_data["upper_95"],
            index=sample_index,
            model_name="TestModel",
        )
        fc.validate()  # should not raise

    def test_forecast_validate_fails(self) -> None:
        """Reject when lower > upper."""
        fc = Forecast(
            point=np.array([100.0]),
            lower_80=np.array([105.0]),  # lower > point!
            upper_80=np.array([110.0]),
            model_name="Bad",
        )
        with pytest.raises(ValueError):
            fc.validate()

    def test_forecast_len(self, sample_forecast_data: dict) -> None:
        """len(fc) == horizon."""
        fc = Forecast(point=sample_forecast_data["point"], horizon=3)
        assert len(fc) == 3

    def test_forecast_getitem(self, sample_forecast_data: dict) -> None:
        """fc[0] returns dict with point and intervals for h=1."""
        fc = Forecast(
            point=sample_forecast_data["point"],
            lower_80=sample_forecast_data["lower_80"],
            upper_80=sample_forecast_data["upper_80"],
        )
        item = fc[0]
        assert "point" in item
        assert "lower_80" in item
        assert "upper_80" in item
        assert item["point"] == pytest.approx(100.5)

    def test_forecast_to_dataframe(
        self, sample_forecast_data: dict, sample_index: pd.DatetimeIndex
    ) -> None:
        """DataFrame with correct columns and temporal index."""
        fc = Forecast(
            point=sample_forecast_data["point"],
            lower_80=sample_forecast_data["lower_80"],
            upper_80=sample_forecast_data["upper_80"],
            lower_95=sample_forecast_data["lower_95"],
            upper_95=sample_forecast_data["upper_95"],
            index=sample_index,
        )
        df = fc.to_dataframe()
        assert "point" in df.columns
        assert "lower_80" in df.columns
        assert "upper_95" in df.columns
        assert len(df) == 3

    def test_forecast_from_distribution(self, rng: np.random.Generator) -> None:
        """Create from 1000 draws, verify point = median."""
        draws = rng.normal(100, 10, size=(3, 1000))
        fc = Forecast.from_distribution(draws, model_name="Draws")
        expected_median = np.median(draws, axis=1)
        np.testing.assert_allclose(fc.point, expected_median)
        assert fc.density is not None
        assert fc.lower_80 is not None
        assert fc.upper_95 is not None

    def test_forecast_plot(self, sample_forecast_data: dict) -> None:
        """plot() executes without error."""
        import matplotlib

        matplotlib.use("Agg")

        fc = Forecast(
            point=sample_forecast_data["point"],
            lower_80=sample_forecast_data["lower_80"],
            upper_80=sample_forecast_data["upper_80"],
            model_name="PlotTest",
        )
        ax = fc.plot(actual=sample_forecast_data["actual"])
        assert ax is not None
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_forecast_save_load(
        self, sample_forecast_data: dict, sample_index: pd.DatetimeIndex, tmp_path
    ) -> None:
        """save -> load -> verify equality."""
        fc = Forecast(
            point=sample_forecast_data["point"],
            lower_80=sample_forecast_data["lower_80"],
            upper_80=sample_forecast_data["upper_80"],
            lower_95=sample_forecast_data["lower_95"],
            upper_95=sample_forecast_data["upper_95"],
            index=sample_index,
            model_name="SaveTest",
            metadata={"param": 42},
        )
        path = tmp_path / "forecast.json"
        fc.save(path)

        loaded = Forecast.load(path)
        np.testing.assert_allclose(loaded.point, fc.point)
        np.testing.assert_allclose(loaded.lower_80, fc.lower_80)
        np.testing.assert_allclose(loaded.upper_95, fc.upper_95)
        assert loaded.model_name == "SaveTest"
        assert loaded.metadata["param"] == 42

    def test_forecast_combine_mean(self) -> None:
        """Combine forecasts with mean method."""
        fc1 = Forecast(point=np.array([100.0, 110.0]), model_name="M1")
        fc2 = Forecast(point=np.array([102.0, 108.0]), model_name="M2")
        combined = Forecast.combine([fc1, fc2], method="mean")
        np.testing.assert_allclose(combined.point, [101.0, 109.0])

    def test_forecast_repr(self, sample_forecast_data: dict) -> None:
        """Test string representation."""
        fc = Forecast(
            point=sample_forecast_data["point"],
            lower_80=sample_forecast_data["lower_80"],
            upper_80=sample_forecast_data["upper_80"],
            model_name="ReprTest",
        )
        assert "ReprTest" in repr(fc)
        assert "80%CI" in repr(fc)
