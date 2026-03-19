"""Rolling window and blocked cross-validation for time series.

Implements fixed-window rolling CV and blocked leave-one-out CV
with optional gap for temporal dependence.

References
----------
Bergmeir, C. & Benitez, J.M. (2012). "On the use of cross-validation
    for time series predictor evaluation." Information Sciences, 191, 192-213.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class CVFold:
    """A single cross-validation fold result.

    Attributes
    ----------
    fold_id : int
        Fold number (0-based).
    train_start : int
        Start index of training window.
    train_end : int
        End index of training window (exclusive).
    test_start : int
        Start index of test window.
    test_end : int
        End index of test window (exclusive).
    actual : NDArray[np.float64]
        Actual test values.
    predicted : NDArray[np.float64]
        Predicted test values.
    metrics : dict[str, float]
        Computed metrics for this fold.
    """

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    actual: NDArray[np.float64]
    predicted: NDArray[np.float64]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class CVResults:
    """Aggregated cross-validation results.

    Attributes
    ----------
    folds : list[CVFold]
        List of individual fold results.
    method : str
        CV method used ('rolling_window' or 'blocked').
    """

    folds: list[CVFold]
    method: str

    @property
    def n_folds(self) -> int:
        """Number of folds."""
        return len(self.folds)

    def mean_metrics(self) -> dict[str, float]:
        """Compute mean of each metric across folds.

        Returns
        -------
        dict[str, float]
            Mean metric values.
        """
        if not self.folds:
            return {}

        all_metrics: dict[str, list[float]] = {}
        for fold in self.folds:
            for name, value in fold.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        return {name: float(np.mean(values)) for name, values in all_metrics.items()}

    def std_metrics(self) -> dict[str, float]:
        """Compute std of each metric across folds.

        Returns
        -------
        dict[str, float]
            Standard deviation of metric values.
        """
        if not self.folds:
            return {}

        all_metrics: dict[str, list[float]] = {}
        for fold in self.folds:
            for name, value in fold.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        return {name: float(np.std(values, ddof=1)) for name, values in all_metrics.items()}

    def summary(self) -> str:
        """Return formatted summary of CV results.

        Returns
        -------
        str
            Formatted summary string.
        """
        means = self.mean_metrics()
        stds = self.std_metrics()

        lines = [
            f"Cross-Validation Results ({self.method})",
            "=" * 50,
            f"Number of folds: {self.n_folds}",
            "-" * 50,
            f"{'Metric':<15} {'Mean':>10} {'Std':>10}",
            "-" * 50,
        ]
        for name in sorted(means.keys()):
            lines.append(f"{name:<15} {means[name]:>10.4f} {stds.get(name, 0.0):>10.4f}")

        return "\n".join(lines)


def _compute_metric(
    actual: NDArray[np.float64], predicted: NDArray[np.float64], name: str
) -> float:
    """Compute a named metric.

    Parameters
    ----------
    actual : NDArray[np.float64]
        Actual values.
    predicted : NDArray[np.float64]
        Predicted values.
    name : str
        Metric name: 'mae', 'rmse', 'mse', 'mape'.

    Returns
    -------
    float
        Metric value.
    """
    errors = actual - predicted
    if name == "mae":
        return float(np.mean(np.abs(errors)))
    elif name == "rmse":
        return float(np.sqrt(np.mean(errors**2)))
    elif name == "mse":
        return float(np.mean(errors**2))
    elif name == "mape":
        mask = actual != 0
        if not np.any(mask):
            return 0.0
        return float(100.0 * np.mean(np.abs(errors[mask] / actual[mask])))
    else:
        msg = f"Unknown metric: '{name}'"
        raise ValueError(msg)


def rolling_window_cv(
    data: pd.Series | NDArray[np.float64],
    model_fn: Callable[[pd.Series], Any],
    window: int,
    horizon: int,
    step: int = 1,
    metrics: tuple[str, ...] = ("mae", "rmse"),
    verbose: bool = False,
) -> CVResults:
    """Rolling (fixed) window cross-validation for time series.

    Slides a fixed-size training window through the data, generating
    out-of-sample forecasts at each step. The training window does NOT
    grow (unlike expanding window CV).

    Parameters
    ----------
    data : pd.Series or array-like
        Time series data.
    model_fn : callable
        Function that takes a pd.Series (training data) and returns
        a forecast object or array-like with 'horizon' predictions.
        Must return an object with a .point attribute (NDArray) or
        directly return an array-like of predictions.
    window : int
        Fixed training window size.
    horizon : int
        Forecast horizon (number of steps ahead).
    step : int
        Step size between consecutive windows. Default: 1.
    metrics : tuple[str, ...]
        Metric names to compute. Default: ('mae', 'rmse').
    verbose : bool
        If True, print progress. Default: False.

    Returns
    -------
    CVResults
        Cross-validation results with all folds.

    Raises
    ------
    ValueError
        If window + horizon > len(data) or invalid parameters.
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    n = len(data)

    if window < 1:
        msg = f"window must be >= 1, got {window}"
        raise ValueError(msg)
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)
    if window + horizon > n:
        msg = f"window ({window}) + horizon ({horizon}) > data length ({n})"
        raise ValueError(msg)

    folds: list[CVFold] = []
    fold_id = 0

    i = 0
    while True:
        train_start = i * step
        train_end = train_start + window
        test_start = train_end
        test_end = test_start + horizon

        if test_end > n:
            break

        train = data.iloc[train_start:train_end]
        test_actual = np.asarray(data.iloc[test_start:test_end], dtype=np.float64)

        if verbose:
            print(
                f"Fold {fold_id}: train[{train_start}:{train_end}] "
                f"test[{test_start}:{test_end}]"
            )

        # Fit model and get predictions
        result = model_fn(train)
        if hasattr(result, "point"):
            predicted = np.asarray(result.point, dtype=np.float64)[:horizon]
        else:
            predicted = np.asarray(result, dtype=np.float64)[:horizon]

        # Compute metrics
        fold_metrics = {}
        for metric_name in metrics:
            fold_metrics[metric_name] = _compute_metric(
                test_actual, predicted, metric_name
            )

        folds.append(
            CVFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                actual=test_actual,
                predicted=predicted,
                metrics=fold_metrics,
            )
        )

        fold_id += 1
        i += 1

    return CVResults(folds=folds, method="rolling_window")


def blocked_cv(
    data: pd.Series | NDArray[np.float64],
    model_fn: Callable[[pd.Series], Any],
    n_blocks: int = 5,
    horizon: int = 12,
    gap: int = 0,
    metrics: tuple[str, ...] = ("mae", "rmse"),
    verbose: bool = False,
) -> CVResults:
    """Blocked cross-validation for time series.

    Divides data into contiguous blocks and uses leave-one-block-out
    with optional gap between training and test sets.

    Parameters
    ----------
    data : pd.Series or array-like
        Time series data.
    model_fn : callable
        Function that takes a pd.Series (training data) and returns
        a forecast object or array-like with 'horizon' predictions.
    n_blocks : int
        Number of blocks to divide data into. Default: 5.
    horizon : int
        Forecast horizon. Default: 12.
    gap : int
        Number of observations to exclude around the test block
        to avoid temporal dependence. Default: 0.
    metrics : tuple[str, ...]
        Metric names to compute. Default: ('mae', 'rmse').
    verbose : bool
        If True, print progress. Default: False.

    Returns
    -------
    CVResults
        Cross-validation results with n_blocks folds.

    Raises
    ------
    ValueError
        If n_blocks < 2 or data too short.
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    n_obs = len(data)

    if n_blocks < 2:
        msg = f"n_blocks must be >= 2, got {n_blocks}"
        raise ValueError(msg)

    block_size = n_obs // n_blocks

    if block_size < horizon:
        msg = (
            f"Block size ({block_size}) must be >= horizon ({horizon}). "
            f"Reduce n_blocks or horizon."
        )
        raise ValueError(msg)

    folds: list[CVFold] = []

    for k in range(n_blocks):
        test_start = k * block_size
        test_end = min((k + 1) * block_size, n_obs) if k < n_blocks - 1 else n_obs

        # Gap boundaries
        gap_start = max(0, test_start - gap)
        gap_end = min(n_obs, test_end + gap)

        # Training indices: everything except test block + gap
        train_indices_before = list(range(0, gap_start))
        train_indices_after = list(range(gap_end, n_obs))
        train_indices = train_indices_before + train_indices_after

        if len(train_indices) < 1:
            continue

        train = data.iloc[train_indices]

        # Test: first 'horizon' points of the test block
        test_actual_end = min(test_start + horizon, test_end)
        test_actual = np.asarray(
            data.iloc[test_start:test_actual_end], dtype=np.float64
        )
        actual_horizon = len(test_actual)

        if actual_horizon == 0:
            continue

        if verbose:
            print(
                f"Fold {k}: test[{test_start}:{test_end}], "
                f"gap=[{gap_start},{gap_end}], "
                f"train_size={len(train_indices)}"
            )

        # Fit model and get predictions
        result = model_fn(train)
        if hasattr(result, "point"):
            predicted = np.asarray(result.point, dtype=np.float64)[:actual_horizon]
        else:
            predicted = np.asarray(result, dtype=np.float64)[:actual_horizon]

        # Compute metrics
        fold_metrics = {}
        for metric_name in metrics:
            fold_metrics[metric_name] = _compute_metric(
                test_actual, predicted, metric_name
            )

        folds.append(
            CVFold(
                fold_id=k,
                train_start=train_indices[0] if train_indices else 0,
                train_end=train_indices[-1] + 1 if train_indices else 0,
                test_start=test_start,
                test_end=test_end,
                actual=test_actual,
                predicted=predicted,
                metrics=fold_metrics,
            )
        )

    return CVResults(folds=folds, method="blocked")
