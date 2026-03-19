"""Base class for forecast combination methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from forecastbox.core.forecast import Forecast


class BaseCombiner(ABC):
    """Abstract base class for forecast combination methods.

    All combiners follow a fit/combine pattern:
    1. ``fit(forecasts_train, actual)`` estimates combination weights
       from historical forecast-actual pairs.
    2. ``combine(forecasts)`` applies the estimated weights to produce
       a single combined Forecast.

    Attributes
    ----------
    weights_ : NDArray[np.float64] | None
        Estimated combination weights after fitting. Shape (n_models,).
    is_fitted_ : bool
        Whether the combiner has been fitted.
    n_models_ : int
        Number of models (set after fit or first combine).
    """

    def __init__(self) -> None:
        self.weights_: NDArray[np.float64] | None = None
        self.is_fitted_: bool = False
        self.n_models_: int = 0

    @abstractmethod
    def fit(
        self,
        forecasts_train: list[NDArray[np.float64]],
        actual: NDArray[np.float64],
    ) -> BaseCombiner:
        """Estimate combination weights from training data.

        Parameters
        ----------
        forecasts_train : list[NDArray[np.float64]]
            List of K arrays, each of shape (T,), containing historical
            forecasts from each model.
        actual : NDArray[np.float64]
            Array of shape (T,) with realized values.

        Returns
        -------
        BaseCombiner
            self, for method chaining.
        """

    def combine(self, forecasts: list[Forecast]) -> Forecast:
        """Combine forecasts using estimated weights.

        Parameters
        ----------
        forecasts : list[Forecast]
            List of K Forecast objects to combine.

        Returns
        -------
        Forecast
            Combined forecast.

        Raises
        ------
        ValueError
            If forecasts list is empty or weights have not been estimated.
        """
        self._validate_forecasts(forecasts)

        if self.weights_ is None:
            msg = (
                "Combiner has no weights. Call fit() first or use a "
                "combiner that does not require fitting (e.g. SimpleCombiner)."
            )
            raise ValueError(msg)

        return self._weighted_combine(forecasts, self.weights_)

    def _validate_forecasts(self, forecasts: list[Any]) -> None:
        """Validate a list of forecasts for combination.

        Parameters
        ----------
        forecasts : list
            List of Forecast objects.

        Raises
        ------
        ValueError
            If the list is empty or forecasts have inconsistent horizons.
        TypeError
            If elements are not Forecast instances.
        """
        if not forecasts:
            msg = "Cannot combine empty list of forecasts."
            raise ValueError(msg)

        if not all(isinstance(fc, Forecast) for fc in forecasts):
            msg = "All elements must be Forecast instances."
            raise TypeError(msg)

        horizon = len(forecasts[0].point)
        for i, fc in enumerate(forecasts):
            if len(fc.point) != horizon:
                msg = (
                    f"All forecasts must have the same horizon. "
                    f"Forecast 0 has {horizon}, forecast {i} has {len(fc.point)}."
                )
                raise ValueError(msg)

        if self.n_models_ > 0 and len(forecasts) != self.n_models_:
            msg = (
                f"Expected {self.n_models_} forecasts (from fit), "
                f"got {len(forecasts)}."
            )
            raise ValueError(msg)

    def _weighted_combine(
        self,
        forecasts: list[Forecast],
        weights: NDArray[np.float64],
    ) -> Forecast:
        """Generic weighted combination of forecasts.

        Parameters
        ----------
        forecasts : list[Forecast]
            List of K Forecast objects.
        weights : NDArray[np.float64]
            Array of shape (K,) with combination weights.

        Returns
        -------
        Forecast
            Weighted combined forecast with point, intervals, and metadata.
        """
        weights = np.asarray(weights, dtype=np.float64)

        # Combine point forecasts
        points = np.array([fc.point for fc in forecasts])  # (K, H)
        combined_point = weights @ points  # (H,)

        # Combine intervals if available
        lower_80 = None
        upper_80 = None
        lower_95 = None
        upper_95 = None

        if all(fc.lower_80 is not None for fc in forecasts):
            lower_80_arr = np.array([fc.lower_80 for fc in forecasts])
            lower_80 = weights @ lower_80_arr
        if all(fc.upper_80 is not None for fc in forecasts):
            upper_80_arr = np.array([fc.upper_80 for fc in forecasts])
            upper_80 = weights @ upper_80_arr
        if all(fc.lower_95 is not None for fc in forecasts):
            lower_95_arr = np.array([fc.lower_95 for fc in forecasts])
            lower_95 = weights @ lower_95_arr
        if all(fc.upper_95 is not None for fc in forecasts):
            upper_95_arr = np.array([fc.upper_95 for fc in forecasts])
            upper_95 = weights @ upper_95_arr

        model_names = [fc.model_name for fc in forecasts]
        combiner_name = type(self).__name__

        return Forecast(
            point=combined_point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            index=forecasts[0].index,
            model_name=f"Combined({combiner_name})",
            metadata={
                "combiner": combiner_name,
                "weights": weights.tolist(),
                "models": model_names,
            },
        )
