"""Simple forecast combination methods: mean, median, trimmed mean."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from forecastbox.combination.base import BaseCombiner
from forecastbox.core.forecast import Forecast


class SimpleCombiner(BaseCombiner):
    """Simple forecast combination without weight estimation.

    Supports three methods:
    - ``mean``: Equal-weighted average. f_c = (1/K) * sum(f_k)
    - ``median``: Pointwise median across models.
    - ``trimmed``: Trimmed mean, excluding a fraction of extreme values.

    The ``fit()`` method is a no-op: SimpleCombiner does not need
    historical data to compute weights.

    Parameters
    ----------
    method : str
        Combination method: 'mean', 'median', or 'trimmed'.
    trim_fraction : float
        Fraction of values to trim from each tail (only used when
        method='trimmed'). Must be in [0, 0.5). Default is 0.1.

    Examples
    --------
    >>> combiner = SimpleCombiner(method='mean')
    >>> fc_combined = combiner.combine([fc1, fc2, fc3])
    """

    def __init__(
        self,
        method: str = "mean",
        trim_fraction: float = 0.1,
    ) -> None:
        super().__init__()
        valid_methods = ("mean", "median", "trimmed")
        if method not in valid_methods:
            msg = f"method must be one of {valid_methods}, got '{method}'"
            raise ValueError(msg)
        if not 0 <= trim_fraction < 0.5:
            msg = f"trim_fraction must be in [0, 0.5), got {trim_fraction}"
            raise ValueError(msg)

        self.method = method
        self.trim_fraction = trim_fraction

    def fit(
        self,
        forecasts_train: list[NDArray[np.float64]],
        actual: NDArray[np.float64],
    ) -> SimpleCombiner:
        """No-op fit. SimpleCombiner does not require training data.

        Parameters
        ----------
        forecasts_train : list[NDArray[np.float64]]
            Ignored.
        actual : NDArray[np.float64]
            Ignored.

        Returns
        -------
        SimpleCombiner
            self, unchanged.
        """
        self.is_fitted_ = True
        return self

    def combine(self, forecasts: list[Forecast]) -> Forecast:
        """Combine forecasts using the specified simple method.

        Parameters
        ----------
        forecasts : list[Forecast]
            List of K Forecast objects to combine.

        Returns
        -------
        Forecast
            Combined forecast.
        """
        self._validate_forecasts(forecasts)
        k = len(forecasts)
        self.n_models_ = k

        # Stack point forecasts: shape (K, H)
        points = np.array([fc.point for fc in forecasts])

        if self.method == "mean":
            self.weights_ = np.full(k, 1.0 / k)
            combined_point = np.mean(points, axis=0)
        elif self.method == "median":
            self.weights_ = np.full(k, 1.0 / k)  # nominal weights
            combined_point = np.median(points, axis=0)
        elif self.method == "trimmed":
            self.weights_ = np.full(k, 1.0 / k)  # nominal weights
            combined_point = stats.trim_mean(points, self.trim_fraction, axis=0)
        else:
            msg = f"Unknown method: {self.method}"
            raise ValueError(msg)

        # Combine intervals if available
        lower_80 = self._combine_interval(forecasts, "lower_80")
        upper_80 = self._combine_interval(forecasts, "upper_80")
        lower_95 = self._combine_interval(forecasts, "lower_95")
        upper_95 = self._combine_interval(forecasts, "upper_95")

        model_names = [fc.model_name for fc in forecasts]

        return Forecast(
            point=combined_point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            index=forecasts[0].index,
            model_name=f"Combined(Simple-{self.method})",
            metadata={
                "combiner": "SimpleCombiner",
                "method": self.method,
                "models": model_names,
            },
        )

    def _combine_interval(
        self,
        forecasts: list[Forecast],
        attr: str,
    ) -> NDArray[np.float64] | None:
        """Combine an interval attribute across forecasts.

        Parameters
        ----------
        forecasts : list[Forecast]
            List of Forecast objects.
        attr : str
            Attribute name: 'lower_80', 'upper_80', 'lower_95', 'upper_95'.

        Returns
        -------
        NDArray[np.float64] | None
            Combined interval or None if not all forecasts have the attribute.
        """
        values = [getattr(fc, attr) for fc in forecasts]
        if any(v is None for v in values):
            return None

        arr = np.array(values)  # (K, H)

        if self.method == "mean":
            return np.mean(arr, axis=0)
        elif self.method == "median":
            return np.median(arr, axis=0)
        elif self.method == "trimmed":
            return stats.trim_mean(arr, self.trim_fraction, axis=0)
        return None
