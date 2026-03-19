"""Cross-validation framework for time series."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class CVResults:
    """Container for cross-validation results.

    Parameters
    ----------
    errors : NDArray[np.float64]
        Error matrix of shape (n_folds, horizon).
    forecasts : list[NDArray[np.float64]]
        Forecasts for each fold.
    actuals : list[NDArray[np.float64]]
        Actual values for each fold.
    horizon : int
        Forecast horizon.
    """

    def __init__(
        self,
        errors: NDArray[np.float64],
        forecasts: list[NDArray[np.float64]],
        actuals: list[NDArray[np.float64]],
        horizon: int,
    ) -> None:
        self.errors = errors
        self.forecasts = forecasts
        self.actuals = actuals
        self.n_folds = len(forecasts)
        self.horizon = horizon
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        """Compute metrics by horizon and overall."""
        # Metrics by horizon
        rows = []
        for h in range(self.horizon):
            h_errors = self.errors[:, h]
            valid = ~np.isnan(h_errors)
            if valid.any():
                h_err = h_errors[valid]
                rows.append({
                    "horizon": h + 1,
                    "mae": float(np.mean(np.abs(h_err))),
                    "rmse": float(np.sqrt(np.mean(h_err**2))),
                    "me": float(np.mean(h_err)),
                })
        self.metrics_by_horizon = pd.DataFrame(rows)

        # Overall metrics
        all_errors = self.errors.flatten()
        valid = ~np.isnan(all_errors)
        if valid.any():
            err = all_errors[valid]
            self.metrics_overall = {
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "me": float(np.mean(err)),
                "n_folds": self.n_folds,
            }
        else:
            self.metrics_overall = {}

    def summary(self) -> str:
        """Generate formatted summary table.

        Returns
        -------
        str
            Summary string with metrics by horizon and overall.
        """
        lines = [
            "=" * 50,
            "Cross-Validation Results",
            "=" * 50,
            f"Folds: {self.n_folds}",
            f"Horizon: {self.horizon}",
            "-" * 50,
            "Metrics by Horizon:",
            self.metrics_by_horizon.to_string(index=False),
            "-" * 50,
            "Overall Metrics:",
        ]
        for k, v in self.metrics_overall.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def mean_metric(self, metric: str = "rmse") -> float:
        """Get mean metric across all folds and horizons.

        Parameters
        ----------
        metric : str
            Metric name ('mae', 'rmse', 'me').

        Returns
        -------
        float
            Mean metric value.
        """
        if metric not in self.metrics_overall:
            msg = f"Unknown metric '{metric}'. Available: {list(self.metrics_overall.keys())}"
            raise ValueError(msg)
        return self.metrics_overall[metric]

    def plot_errors(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot metrics by horizon.

        Parameters
        ----------
        ax : plt.Axes or None
            Axes to plot on.

        Returns
        -------
        plt.Axes
            Matplotlib axes.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        h = self.metrics_by_horizon["horizon"]
        ax.plot(h, self.metrics_by_horizon["mae"], "o-", label="MAE")
        ax.plot(h, self.metrics_by_horizon["rmse"], "s-", label="RMSE")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Error")
        ax.set_title("Forecast Error by Horizon")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def plot_forecast_vs_actual(
        self, fold: int | None = None, ax: plt.Axes | None = None
    ) -> plt.Axes:
        """Plot forecast vs actual for a specific fold or all folds.

        Parameters
        ----------
        fold : int or None
            Fold index to plot. If None, plots the last fold.
        ax : plt.Axes or None
            Axes to plot on.

        Returns
        -------
        plt.Axes
            Matplotlib axes.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        if fold is None:
            fold = self.n_folds - 1

        fc = self.forecasts[fold]
        act = self.actuals[fold]
        h_range = np.arange(1, len(fc) + 1)

        ax.plot(h_range, act, "ko-", label="Actual", linewidth=2)
        ax.plot(h_range, fc, "b--", label="Forecast", linewidth=1.5)
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Value")
        ax.set_title(f"Fold {fold + 1}: Forecast vs Actual")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax


def expanding_window_cv(
    data: pd.Series,
    model_fn: Callable[[pd.Series], Any],
    initial_window: int,
    horizon: int,
    step: int = 1,
    metrics: tuple[str, ...] = ("mae", "rmse"),
    verbose: bool = False,
) -> CVResults:
    """Expanding window cross-validation for time series.

    Parameters
    ----------
    data : pd.Series
        Full time series data.
    model_fn : Callable[[pd.Series], Any]
        Function that takes training data (pd.Series) and returns either:
        - An object with a .forecast(h) method returning ndarray
        - Directly an ndarray of h predictions
    initial_window : int
        Size of the initial training window.
    horizon : int
        Forecast horizon.
    step : int
        Step size between folds.
    metrics : tuple[str, ...]
        Metrics to compute (used in CVResults).
    verbose : bool
        Print progress information.

    Returns
    -------
    CVResults
        Cross-validation results.
    """
    n = len(data)
    if initial_window < 1:
        msg = f"initial_window must be >= 1, got {initial_window}"
        raise ValueError(msg)
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)
    if initial_window + horizon > n:
        msg = (
            f"Not enough data: need at least {initial_window + horizon} "
            f"observations, got {n}"
        )
        raise ValueError(msg)

    all_forecasts: list[NDArray[np.float64]] = []
    all_actuals: list[NDArray[np.float64]] = []
    all_errors: list[NDArray[np.float64]] = []

    i = 0
    while True:
        train_end = initial_window + i * step
        test_start = train_end
        test_end = test_start + horizon

        if test_end > n:
            break

        train = data.iloc[:train_end]
        test = data.iloc[test_start:test_end].values.astype(np.float64)

        # Get forecast
        result = model_fn(train)
        if isinstance(result, np.ndarray):
            forecast = result.astype(np.float64)
        elif hasattr(result, "forecast"):
            fc_result = result.forecast(horizon)
            forecast = np.asarray(fc_result, dtype=np.float64)
        else:
            msg = "model_fn must return ndarray or object with .forecast(h) method"
            raise TypeError(msg)

        # Ensure forecast length matches horizon
        forecast = forecast[:horizon]
        errors = test - forecast

        all_forecasts.append(forecast)
        all_actuals.append(test)
        all_errors.append(errors)

        if verbose:
            fold_mae = float(np.mean(np.abs(errors)))
            print(f"  Fold {i + 1}: train={len(train)}, MAE={fold_mae:.4f}")

        i += 1

    if not all_errors:
        msg = "No folds could be created. Check data size, initial_window, and horizon."
        raise ValueError(msg)

    errors_matrix = np.array(all_errors)

    return CVResults(
        errors=errors_matrix,
        forecasts=all_forecasts,
        actuals=all_actuals,
        horizon=horizon,
    )
