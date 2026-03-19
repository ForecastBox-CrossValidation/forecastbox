"""Baseline forecasting models (always available, no external dependencies).

These models serve as reference benchmarks and are registered in ModelZoo
by default. They implement the ForecastModel protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from forecastbox.core.forecast import Forecast


class NaiveBaseline:
    """Naive forecast: repeats the last observed value.

    Parameters
    ----------
    None
    """

    def __init__(self, **kwargs: Any) -> None:
        self._last_value: float | None = None
        self._fitted: bool = False

    def fit(self, y: pd.Series | NDArray[np.float64], **kwargs: Any) -> NaiveBaseline:
        """Fit by storing the last value.

        Parameters
        ----------
        y : pd.Series or NDArray
            Time series data.

        Returns
        -------
        NaiveBaseline
            Fitted model (self).
        """
        arr = np.asarray(y, dtype=np.float64)
        self._last_value = float(arr[-1])
        self._fitted = True
        return self

    def forecast(self, h: int, level: tuple[int, ...] = (80, 95), **kwargs: Any) -> Forecast:
        """Generate naive forecast.

        Parameters
        ----------
        h : int
            Forecast horizon.
        level : tuple[int, ...]
            Confidence levels for prediction intervals.

        Returns
        -------
        Forecast
            Forecast with constant point predictions.
        """
        if not self._fitted or self._last_value is None:
            msg = "Model must be fit before forecasting"
            raise RuntimeError(msg)

        point = np.full(h, self._last_value)

        return Forecast(
            point=point,
            model_name="Naive",
            horizon=h,
            metadata={"method": "naive", "last_value": self._last_value},
        )


class SeasonalNaiveBaseline:
    """Seasonal naive forecast: repeats the last seasonal cycle.

    Parameters
    ----------
    seasonal_period : int
        Length of the seasonal cycle (e.g., 12 for monthly data).
    """

    def __init__(self, seasonal_period: int = 12, **kwargs: Any) -> None:
        self.seasonal_period = seasonal_period
        self._last_season: NDArray[np.float64] | None = None
        self._fitted: bool = False

    def fit(
        self, y: pd.Series | NDArray[np.float64], **kwargs: Any
    ) -> SeasonalNaiveBaseline:
        """Fit by storing the last seasonal cycle.

        Parameters
        ----------
        y : pd.Series or NDArray
            Time series data (must have at least seasonal_period observations).

        Returns
        -------
        SeasonalNaiveBaseline
            Fitted model (self).
        """
        arr = np.asarray(y, dtype=np.float64)
        if len(arr) < self.seasonal_period:
            msg = (
                f"Series length ({len(arr)}) must be >= seasonal_period "
                f"({self.seasonal_period})"
            )
            raise ValueError(msg)

        self._last_season = arr[-self.seasonal_period :]
        self._fitted = True
        return self

    def forecast(self, h: int, level: tuple[int, ...] = (80, 95), **kwargs: Any) -> Forecast:
        """Generate seasonal naive forecast.

        Parameters
        ----------
        h : int
            Forecast horizon.
        level : tuple[int, ...]
            Confidence levels for prediction intervals.

        Returns
        -------
        Forecast
            Forecast repeating the seasonal pattern.
        """
        if not self._fitted or self._last_season is None:
            msg = "Model must be fit before forecasting"
            raise RuntimeError(msg)

        # Tile the seasonal pattern to cover the horizon
        n_repeats = (h // self.seasonal_period) + 1
        tiled = np.tile(self._last_season, n_repeats)
        point = tiled[:h]

        return Forecast(
            point=point,
            model_name=f"SeasonalNaive(m={self.seasonal_period})",
            horizon=h,
            metadata={
                "method": "seasonal_naive",
                "seasonal_period": self.seasonal_period,
            },
        )


class DriftBaseline:
    """Random walk with drift forecast.

    The drift is estimated as the average change between first and last observations:
        drift = (y_T - y_1) / (T - 1)

    Forecast: y_{T+h} = y_T + h * drift

    Parameters
    ----------
    None
    """

    def __init__(self, **kwargs: Any) -> None:
        self._last_value: float | None = None
        self._drift: float | None = None
        self._fitted: bool = False

    def fit(self, y: pd.Series | NDArray[np.float64], **kwargs: Any) -> DriftBaseline:
        """Fit by computing the drift.

        Parameters
        ----------
        y : pd.Series or NDArray
            Time series data (must have at least 2 observations).

        Returns
        -------
        DriftBaseline
            Fitted model (self).
        """
        arr = np.asarray(y, dtype=np.float64)
        if len(arr) < 2:
            msg = "Series must have at least 2 observations for drift estimation"
            raise ValueError(msg)

        self._last_value = float(arr[-1])
        self._drift = float((arr[-1] - arr[0]) / (len(arr) - 1))
        self._fitted = True
        return self

    def forecast(self, h: int, level: tuple[int, ...] = (80, 95), **kwargs: Any) -> Forecast:
        """Generate drift forecast.

        Parameters
        ----------
        h : int
            Forecast horizon.
        level : tuple[int, ...]
            Confidence levels for prediction intervals.

        Returns
        -------
        Forecast
            Forecast with linear drift.
        """
        if not self._fitted or self._last_value is None or self._drift is None:
            msg = "Model must be fit before forecasting"
            raise RuntimeError(msg)

        steps = np.arange(1, h + 1, dtype=np.float64)
        point = self._last_value + steps * self._drift

        return Forecast(
            point=point,
            model_name="Drift",
            horizon=h,
            metadata={
                "method": "drift",
                "last_value": self._last_value,
                "drift": self._drift,
            },
        )
