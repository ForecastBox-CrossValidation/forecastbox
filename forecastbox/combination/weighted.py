"""Weighted forecast combination based on historical performance."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forecastbox.combination.base import BaseCombiner


class WeightedCombiner(BaseCombiner):
    """Forecast combination with performance-based weights.

    Assigns higher weights to models with better historical performance.
    Three weighting schemes are available:

    - ``inverse_mse``: Weights inversely proportional to MSE.
      w_k = (1/MSE_k) / sum(1/MSE_j)
    - ``aic_weights``: Akaike weights based on AIC differences.
      w_k = exp(-0.5 * delta_AIC_k) / sum(exp(-0.5 * delta_AIC_j))
    - ``bic_weights``: Schwarz weights based on BIC differences.
      w_k = exp(-0.5 * delta_BIC_k) / sum(exp(-0.5 * delta_BIC_j))

    Parameters
    ----------
    method : str
        Weighting method: 'inverse_mse', 'aic_weights', or 'bic_weights'.
    n_params : list[int] | None
        Number of parameters for each model (used for AIC/BIC).
        Default is [1, 1, ...] if not provided.

    Attributes
    ----------
    weights_ : NDArray[np.float64]
        Estimated combination weights after fitting.
    mse_ : NDArray[np.float64]
        MSE of each model on training data.
    ic_values_ : NDArray[np.float64] | None
        Information criterion values (AIC or BIC) if applicable.

    Examples
    --------
    >>> combiner = WeightedCombiner(method='inverse_mse')
    >>> combiner.fit(forecasts_train, actual)
    >>> fc_combined = combiner.combine(forecasts)
    """

    def __init__(
        self,
        method: str = "inverse_mse",
        n_params: list[int] | None = None,
    ) -> None:
        super().__init__()
        valid_methods = ("inverse_mse", "aic_weights", "bic_weights")
        if method not in valid_methods:
            msg = f"method must be one of {valid_methods}, got '{method}'"
            raise ValueError(msg)

        self.method = method
        self.n_params = n_params
        self.mse_: NDArray[np.float64] | None = None
        self.ic_values_: NDArray[np.float64] | None = None

    def fit(
        self,
        forecasts_train: list[NDArray[np.float64]],
        actual: NDArray[np.float64],
    ) -> WeightedCombiner:
        """Estimate weights from historical forecast performance.

        Parameters
        ----------
        forecasts_train : list[NDArray[np.float64]]
            List of K arrays, each of shape (T,), containing historical
            forecasts from each model.
        actual : NDArray[np.float64]
            Array of shape (T,) with realized values.

        Returns
        -------
        WeightedCombiner
            self, for method chaining.
        """
        actual = np.asarray(actual, dtype=np.float64)
        k = len(forecasts_train)
        t = len(actual)
        self.n_models_ = k

        # Compute MSE for each model
        mse = np.zeros(k)
        for i, fc in enumerate(forecasts_train):
            fc_arr = np.asarray(fc, dtype=np.float64)
            errors = actual - fc_arr
            mse[i] = np.mean(errors**2)

        self.mse_ = mse

        if self.method == "inverse_mse":
            # w_k = (1/MSE_k) / sum(1/MSE_j)
            inv_mse = 1.0 / mse
            self.weights_ = inv_mse / np.sum(inv_mse)

        elif self.method == "aic_weights":
            n_params = self.n_params if self.n_params is not None else [1] * k
            aic = np.zeros(k)
            for i in range(k):
                aic[i] = t * np.log(mse[i]) + 2 * n_params[i]
            delta_aic = aic - np.min(aic)
            exp_weights = np.exp(-0.5 * delta_aic)
            self.weights_ = exp_weights / np.sum(exp_weights)
            self.ic_values_ = aic

        elif self.method == "bic_weights":
            n_params = self.n_params if self.n_params is not None else [1] * k
            bic = np.zeros(k)
            for i in range(k):
                bic[i] = t * np.log(mse[i]) + n_params[i] * np.log(t)
            delta_bic = bic - np.min(bic)
            exp_weights = np.exp(-0.5 * delta_bic)
            self.weights_ = exp_weights / np.sum(exp_weights)
            self.ic_values_ = bic

        self.is_fitted_ = True
        return self
