"""Tests for SimpleCombiner."""

from __future__ import annotations

import numpy as np

from forecastbox.combination.simple import SimpleCombiner
from forecastbox.core.forecast import Forecast


class TestSimpleCombiner:
    """Tests for SimpleCombiner (mean, median, trimmed)."""

    def test_mean_equal_forecasts(self) -> None:
        """K identical forecasts -> combined == individual."""
        point = np.array([100.0, 110.0, 120.0])
        fc1 = Forecast(point=point.copy(), model_name="M1")
        fc2 = Forecast(point=point.copy(), model_name="M2")
        fc3 = Forecast(point=point.copy(), model_name="M3")

        combiner = SimpleCombiner(method="mean")
        result = combiner.combine([fc1, fc2, fc3])

        np.testing.assert_allclose(result.point, point)

    def test_mean_formula(self) -> None:
        """3 forecasts with known values -> mean is correct."""
        fc1 = Forecast(point=np.array([100.0, 200.0]), model_name="M1")
        fc2 = Forecast(point=np.array([110.0, 220.0]), model_name="M2")
        fc3 = Forecast(point=np.array([105.0, 210.0]), model_name="M3")

        combiner = SimpleCombiner(method="mean")
        result = combiner.combine([fc1, fc2, fc3])

        expected = np.array([105.0, 210.0])
        np.testing.assert_allclose(result.point, expected)

    def test_median_odd(self) -> None:
        """3 forecasts -> median = middle value at each horizon."""
        fc1 = Forecast(point=np.array([100.0, 200.0]), model_name="M1")
        fc2 = Forecast(point=np.array([110.0, 180.0]), model_name="M2")
        fc3 = Forecast(point=np.array([105.0, 220.0]), model_name="M3")

        combiner = SimpleCombiner(method="median")
        result = combiner.combine([fc1, fc2, fc3])

        # h=0: sorted [100, 105, 110] -> median = 105
        # h=1: sorted [180, 200, 220] -> median = 200
        expected = np.array([105.0, 200.0])
        np.testing.assert_allclose(result.point, expected)

    def test_median_even(self) -> None:
        """4 forecasts -> median = average of two middle values."""
        fc1 = Forecast(point=np.array([100.0]), model_name="M1")
        fc2 = Forecast(point=np.array([110.0]), model_name="M2")
        fc3 = Forecast(point=np.array([120.0]), model_name="M3")
        fc4 = Forecast(point=np.array([130.0]), model_name="M4")

        combiner = SimpleCombiner(method="median")
        result = combiner.combine([fc1, fc2, fc3, fc4])

        # sorted [100, 110, 120, 130] -> median = (110 + 120) / 2 = 115
        expected = np.array([115.0])
        np.testing.assert_allclose(result.point, expected)

    def test_trimmed_excludes(self) -> None:
        """Trimmed mean with 5 forecasts, trim=0.2 -> excludes min and max."""
        fc1 = Forecast(point=np.array([90.0]), model_name="M1")   # min -> trimmed
        fc2 = Forecast(point=np.array([100.0]), model_name="M2")
        fc3 = Forecast(point=np.array([105.0]), model_name="M3")
        fc4 = Forecast(point=np.array([110.0]), model_name="M4")
        fc5 = Forecast(point=np.array([150.0]), model_name="M5")  # max -> trimmed

        combiner = SimpleCombiner(method="trimmed", trim_fraction=0.2)
        result = combiner.combine([fc1, fc2, fc3, fc4, fc5])

        # trim=0.2 with K=5: floor(0.2*5) = 1 from each end
        # remaining: [100, 105, 110] -> mean = 105.0
        expected = np.array([105.0])
        np.testing.assert_allclose(result.point, expected)

    def test_no_fit_required(self) -> None:
        """SimpleCombiner does not require fit() before combine()."""
        fc1 = Forecast(point=np.array([100.0, 110.0]), model_name="M1")
        fc2 = Forecast(point=np.array([102.0, 108.0]), model_name="M2")

        combiner = SimpleCombiner(method="mean")
        # combine without fit -> should work
        result = combiner.combine([fc1, fc2])

        expected = np.array([101.0, 109.0])
        np.testing.assert_allclose(result.point, expected)

        # Verify fit() is a no-op that returns self
        ret = combiner.fit([], np.array([]))
        assert ret is combiner
