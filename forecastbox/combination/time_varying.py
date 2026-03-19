"""Time-varying forecast combination with exponential decay."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from forecastbox.combination.base import BaseCombiner


class TimeVaryingCombiner(BaseCombiner):
    """Forecast combination with exponentially weighted time-varying weights.

    Updates model weights over time using exponential smoothing of
    squared forecast errors. Models that perform better in recent
    periods receive higher weights.

    The recursive MSE update is:

        MSE_{k,t} = lambda * MSE_{k,t-1} + (1 - lambda) * (y_t - f_{k,t})^2

    And the weights at time t are:

        w_{k,t} = (1 / MSE_{k,t}) / sum_j (1 / MSE_{j,t})

    Parameters
    ----------
    decay : float
        Exponential decay parameter lambda in (0, 1). Higher values
        give more weight to historical performance (longer memory).
        Default is 0.95.
    initial_mse : float | None
        Initial MSE for all models. If None, uses the variance of
        the actual values. Default is None.

    Attributes
    ----------
    weights_ : NDArray[np.float64]
        Final weights (at time T) after fitting.
    weights_history_ : NDArray[np.float64]
        Full history of weights, shape (T, K).
    mse_history_ : NDArray[np.float64]
        Full history of exponentially smoothed MSE, shape (T, K).

    Examples
    --------
    >>> combiner = TimeVaryingCombiner(decay=0.95)
    >>> combiner.fit(forecasts_train, actual)
    >>> combiner.plot_weights()
    >>> fc_combined = combiner.combine(forecasts)
    """

    def __init__(
        self,
        decay: float = 0.95,
        initial_mse: float | None = None,
    ) -> None:
        super().__init__()
        if not 0 < decay < 1:
            msg = f"decay must be in (0, 1), got {decay}"
            raise ValueError(msg)

        self.decay = decay
        self.initial_mse = initial_mse
        self.weights_history_: NDArray[np.float64] | None = None
        self.mse_history_: NDArray[np.float64] | None = None

    def fit(
        self,
        forecasts_train: list[NDArray[np.float64]],
        actual: NDArray[np.float64],
    ) -> TimeVaryingCombiner:
        """Estimate time-varying weights from historical data.

        Parameters
        ----------
        forecasts_train : list[NDArray[np.float64]]
            List of K arrays, each of shape (T,), with historical forecasts.
        actual : NDArray[np.float64]
            Array of shape (T,) with realized values.

        Returns
        -------
        TimeVaryingCombiner
            self, for method chaining.
        """
        actual = np.asarray(actual, dtype=np.float64)
        k = len(forecasts_train)
        t = len(actual)
        self.n_models_ = k

        # Build forecast matrix: shape (T, K)
        f_matrix = np.column_stack(
            [np.asarray(fc, dtype=np.float64) for fc in forecasts_train]
        )

        # Initial MSE
        if self.initial_mse is not None:
            mse_init = self.initial_mse
        else:
            mse_init = float(np.var(actual)) if np.var(actual) > 0 else 1.0

        # Initialize arrays
        mse_history = np.zeros((t, k))
        weights_history = np.zeros((t, k))

        # Current MSE for each model
        current_mse = np.full(k, mse_init)

        lam = self.decay

        for step in range(t):
            # Squared errors at this time step
            errors_sq = (actual[step] - f_matrix[step, :]) ** 2

            # Update MSE with exponential smoothing
            current_mse = lam * current_mse + (1.0 - lam) * errors_sq

            # Store MSE history
            mse_history[step, :] = current_mse

            # Compute weights: inverse MSE, normalized
            inv_mse = 1.0 / np.maximum(current_mse, 1e-15)
            weights = inv_mse / np.sum(inv_mse)
            weights_history[step, :] = weights

        self.mse_history_ = mse_history
        self.weights_history_ = weights_history
        self.weights_ = weights_history[-1, :]
        self.is_fitted_ = True
        return self

    def plot_weights(
        self,
        model_names: list[str] | None = None,
        ax: plt.Axes | None = None,
        title: str = "Time-Varying Combination Weights",
    ) -> plt.Axes:
        """Plot the evolution of combination weights over time.

        Parameters
        ----------
        model_names : list[str] | None
            Names for each model. Default: 'Model 0', 'Model 1', ...
        ax : plt.Axes | None
            Matplotlib axes. Creates new figure if None.
        title : str
            Plot title.

        Returns
        -------
        plt.Axes
            The matplotlib axes object.
        """
        if self.weights_history_ is None:
            msg = "TimeVaryingCombiner must be fitted before plotting."
            raise ValueError(msg)

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        t, k = self.weights_history_.shape

        if model_names is None:
            model_names = [f"Model {i}" for i in range(k)]

        for i in range(k):
            ax.plot(
                range(t),
                self.weights_history_[:, i],
                label=model_names[i],
                linewidth=1.5,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Weight")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        return ax
