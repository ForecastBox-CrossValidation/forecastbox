"""Adapter classes for chronobox models.

These adapters wrap chronobox model classes to conform to the ForecastModel
protocol used by forecastbox. If chronobox is not installed, importing this
module will raise ImportError (caught by ModelZoo._register_builtins).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from forecastbox.core.forecast import Forecast

try:
    import chronobox
    HAS_CHRONOBOX = True
except ImportError:
    HAS_CHRONOBOX = False


class ARIMAAdapter:
    """Adapter for chronobox ARIMA model.

    Parameters
    ----------
    order : tuple[int, int, int]
        ARIMA (p, d, q) order.
    seasonal_order : tuple[int, int, int, int] or None
        Seasonal (P, D, Q, m) order.
    **kwargs : Any
        Additional arguments passed to chronobox.ARIMA.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        if not HAS_CHRONOBOX:
            msg = "chronobox is required for ARIMAAdapter. Install with: pip install chronobox"
            raise ImportError(msg)
        self.order = order
        self.seasonal_order = seasonal_order
        self._kwargs = kwargs
        self._model: Any = None
        self._fitted: bool = False

    def fit(self, y: pd.Series | NDArray[np.float64], **kwargs: Any) -> ARIMAAdapter:
        """Fit the ARIMA model via chronobox.

        Parameters
        ----------
        y : pd.Series or NDArray
            Time series data.

        Returns
        -------
        ARIMAAdapter
            Fitted adapter (self).
        """
        fit_kwargs = {**self._kwargs, **kwargs}
        self._model = chronobox.ARIMA(
            order=self.order,
            seasonal_order=self.seasonal_order,
            **fit_kwargs,
        )
        self._model.fit(y)
        self._fitted = True
        return self

    def forecast(self, h: int, level: tuple[int, ...] = (80, 95), **kwargs: Any) -> Forecast:
        """Generate forecast via chronobox ARIMA.

        Parameters
        ----------
        h : int
            Forecast horizon.
        level : tuple[int, ...]
            Confidence levels.

        Returns
        -------
        Forecast
            Forecast object.
        """
        if not self._fitted or self._model is None:
            msg = "Model must be fit before forecasting"
            raise RuntimeError(msg)
        result = self._model.forecast(h)
        return Forecast(
            point=np.asarray(result.point, dtype=np.float64),
            model_name=f"ARIMA{self.order}",
            horizon=h,
        )


class ETSAdapter:
    """Adapter for chronobox ETS model.

    Parameters
    ----------
    error : str
        Error type: 'A' (additive) or 'M' (multiplicative).
    trend : str
        Trend type: 'N', 'A', 'Ad', 'M', 'Md'.
    seasonal : str
        Seasonal type: 'N', 'A', 'M'.
    seasonal_period : int
        Seasonal period.
    **kwargs : Any
        Additional arguments passed to chronobox.ETS.
    """

    def __init__(
        self,
        error: str = "A",
        trend: str = "N",
        seasonal: str = "N",
        seasonal_period: int = 1,
        **kwargs: Any,
    ) -> None:
        if not HAS_CHRONOBOX:
            msg = "chronobox is required for ETSAdapter. Install with: pip install chronobox"
            raise ImportError(msg)
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self._kwargs = kwargs
        self._model: Any = None
        self._fitted: bool = False

    def fit(self, y: pd.Series | NDArray[np.float64], **kwargs: Any) -> ETSAdapter:
        """Fit the ETS model via chronobox."""
        fit_kwargs = {**self._kwargs, **kwargs}
        self._model = chronobox.ETS(
            error=self.error,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_period=self.seasonal_period,
            **fit_kwargs,
        )
        self._model.fit(y)
        self._fitted = True
        return self

    def forecast(self, h: int, level: tuple[int, ...] = (80, 95), **kwargs: Any) -> Forecast:
        """Generate forecast via chronobox ETS."""
        if not self._fitted or self._model is None:
            msg = "Model must be fit before forecasting"
            raise RuntimeError(msg)
        result = self._model.forecast(h)
        return Forecast(
            point=np.asarray(result.point, dtype=np.float64),
            model_name=f"ETS({self.error},{self.trend},{self.seasonal})",
            horizon=h,
        )


class VARAdapter:
    """Adapter for chronobox VAR model.

    Parameters
    ----------
    maxlags : int
        Maximum number of lags.
    **kwargs : Any
        Additional arguments passed to chronobox.VAR.
    """

    def __init__(self, maxlags: int = 12, **kwargs: Any) -> None:
        if not HAS_CHRONOBOX:
            msg = "chronobox is required for VARAdapter. Install with: pip install chronobox"
            raise ImportError(msg)
        self.maxlags = maxlags
        self._kwargs = kwargs
        self._model: Any = None
        self._fitted: bool = False

    def fit(self, y: pd.DataFrame, **kwargs: Any) -> VARAdapter:
        """Fit the VAR model via chronobox."""
        fit_kwargs = {**self._kwargs, **kwargs}
        self._model = chronobox.VAR(maxlags=self.maxlags, **fit_kwargs)
        self._model.fit(y)
        self._fitted = True
        return self

    def forecast(self, h: int, level: tuple[int, ...] = (80, 95), **kwargs: Any) -> Forecast:
        """Generate forecast via chronobox VAR."""
        if not self._fitted or self._model is None:
            msg = "Model must be fit before forecasting"
            raise RuntimeError(msg)
        result = self._model.forecast(h)
        return Forecast(
            point=np.asarray(result.point, dtype=np.float64),
            model_name=f"VAR({self.maxlags})",
            horizon=h,
        )


class ThetaAdapter:
    """Adapter for chronobox Theta method.

    Parameters
    ----------
    **kwargs : Any
        Additional arguments passed to chronobox.Theta.
    """

    def __init__(self, **kwargs: Any) -> None:
        if not HAS_CHRONOBOX:
            msg = "chronobox is required for ThetaAdapter. Install with: pip install chronobox"
            raise ImportError(msg)
        self._kwargs = kwargs
        self._model: Any = None
        self._fitted: bool = False

    def fit(self, y: pd.Series | NDArray[np.float64], **kwargs: Any) -> ThetaAdapter:
        """Fit the Theta model via chronobox."""
        fit_kwargs = {**self._kwargs, **kwargs}
        self._model = chronobox.Theta(**fit_kwargs)
        self._model.fit(y)
        self._fitted = True
        return self

    def forecast(self, h: int, level: tuple[int, ...] = (80, 95), **kwargs: Any) -> Forecast:
        """Generate forecast via chronobox Theta."""
        if not self._fitted or self._model is None:
            msg = "Model must be fit before forecasting"
            raise RuntimeError(msg)
        result = self._model.forecast(h)
        return Forecast(
            point=np.asarray(result.point, dtype=np.float64),
            model_name="Theta",
            horizon=h,
        )
