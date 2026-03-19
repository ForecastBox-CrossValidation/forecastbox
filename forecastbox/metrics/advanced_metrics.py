"""Advanced forecast evaluation metrics.

Includes MFE, Theil U1/U2, sMAPE, Log Score, and CRPS.

References
----------
Gneiting, T. & Raftery, A.E. (2007). "Strictly Proper Scoring Rules,
    Prediction, and Estimation." JASA, 102(477), 359-378.
Gneiting, T., Balabdaoui, F. & Raftery, A.E. (2005). "Probabilistic
    forecasts, calibration and sharpness." JRSS-B, 69(2), 243-268.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats as scipy_stats


def mfe(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float:
    """Mean Forecast Error (bias measure).

    MFE = (1/T) * sum(y_t - f_t)

    Positive MFE indicates under-forecasting (actual > predicted on average).
    Negative MFE indicates over-forecasting (actual < predicted on average).

    Parameters
    ----------
    actual : array-like
        Actual observed values.
    predicted : array-like
        Predicted (forecast) values.

    Returns
    -------
    float
        Mean forecast error.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.mean(actual - predicted))


def theil_u1(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float:
    """Theil U1 inequality coefficient.

    U1 = sqrt(sum((y-f)^2)) / (sqrt(sum(y^2)) + sqrt(sum(f^2)))

    Always between 0 (perfect) and 1 (worst).

    Parameters
    ----------
    actual : array-like
        Actual observed values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        Theil U1 coefficient (0 = perfect, 1 = worst).
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    numerator = math.sqrt(float(np.sum((actual - predicted) ** 2)))
    denominator = math.sqrt(float(np.sum(actual**2))) + math.sqrt(
        float(np.sum(predicted**2))
    )

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def theil_u2(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
    naive: NDArray[np.float64] | list[float] | None = None,
) -> float:
    """Theil U2 coefficient (relative to naive forecast).

    If naive is None, uses random walk (naive_t = y_{t-1}):
        U2 = sqrt(sum(((y_t - f_t)/y_{t-1})^2) / sum(((y_t - y_{t-1})/y_{t-1})^2))

    If naive is provided:
        U2 = sqrt(sum((y - f)^2) / sum((y - naive)^2))

    U2 < 1: better than naive; U2 = 0: perfect; U2 > 1: worse than naive.

    Parameters
    ----------
    actual : array-like
        Actual observed values.
    predicted : array-like
        Predicted values.
    naive : array-like or None
        Naive forecast. If None, uses random walk (y_{t-1}).

    Returns
    -------
    float
        Theil U2 coefficient.

    Raises
    ------
    ValueError
        If arrays have incompatible lengths.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if naive is not None:
        naive = np.asarray(naive, dtype=np.float64)
        num = float(np.sum((actual - predicted) ** 2))
        den = float(np.sum((actual - naive) ** 2))
    else:
        # Random walk naive: y_{t-1}
        if len(actual) < 2:
            msg = "Need at least 2 observations for Theil U2 with random walk"
            raise ValueError(msg)
        y_prev = actual[:-1]
        y_curr = actual[1:]
        f_curr = predicted[1:]

        # Avoid division by zero
        mask = y_prev != 0.0
        if not np.any(mask):
            return 0.0

        num = float(np.sum(((y_curr[mask] - f_curr[mask]) / y_prev[mask]) ** 2))
        den = float(
            np.sum(((y_curr[mask] - y_prev[mask]) / y_prev[mask]) ** 2)
        )

    if den == 0.0:
        return 0.0

    return math.sqrt(num / den)


def smape(
    actual: NDArray[np.float64] | list[float],
    predicted: NDArray[np.float64] | list[float],
) -> float:
    """Symmetric Mean Absolute Percentage Error.

    sMAPE = (200/T) * sum(|y - f| / (|y| + |f|))

    Symmetric: sMAPE(y, f) = sMAPE(f, y). Range: [0, 200].

    Parameters
    ----------
    actual : array-like
        Actual observed values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        sMAPE value (0 = perfect, 200 = worst).
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    n = len(actual)
    denominator = np.abs(actual) + np.abs(predicted)

    # Avoid division by zero
    mask = denominator > 0
    if not np.any(mask):
        return 0.0

    ratios = np.zeros(n)
    ratios[mask] = np.abs(actual[mask] - predicted[mask]) / denominator[mask]

    return float(200.0 / n * np.sum(ratios))


def log_score(
    actual: NDArray[np.float64] | list[float],
    density_fn: Callable[[float], float],
) -> float:
    """Log score for probabilistic forecasts.

    LogScore = (1/T) * sum(log(p(y_t)))

    Higher is better (less negative).

    Parameters
    ----------
    actual : array-like
        Actual observed values.
    density_fn : callable
        Function that takes a scalar y and returns the predictive
        density p(y). Must return positive values.

    Returns
    -------
    float
        Mean log score.
    """
    actual = np.asarray(actual, dtype=np.float64)
    n = len(actual)

    total = 0.0
    for t in range(n):
        p = density_fn(float(actual[t]))
        if p <= 0:
            total += -1e10  # penalty for zero density
        else:
            total += math.log(p)

    return total / n


def crps(
    actual: NDArray[np.float64] | list[float],
    ensemble: NDArray[np.float64] | list[list[float]],
) -> float:
    """Continuous Ranked Probability Score via ensemble method.

    CRPS = (1/T) * sum_t [ (1/M) sum_m |x_{t,m} - y_t|
                           - 1/(2M^2) sum_m sum_n |x_{t,m} - x_{t,n}| ]

    Uses the efficient sorted computation from Gneiting & Raftery (2007).

    Lower is better. CRPS = 0 for a perfect deterministic forecast.

    Parameters
    ----------
    actual : array-like
        Actual observed values of length T.
    ensemble : array-like
        Ensemble draws of shape (T, M) where M is the number of draws.
        Each row contains M draws from the predictive distribution for
        time t.

    Returns
    -------
    float
        Mean CRPS across all time points.
    """
    actual = np.asarray(actual, dtype=np.float64)
    ensemble = np.asarray(ensemble, dtype=np.float64)

    if ensemble.ndim == 1:
        # Single observation: reshape to (1, M)
        ensemble = ensemble.reshape(1, -1)
        actual = actual.reshape(1)

    n_obs, n_draws = ensemble.shape

    if len(actual) != n_obs:
        msg = f"ensemble has {n_obs} rows but actual has {len(actual)} elements"
        raise ValueError(msg)

    total_crps = 0.0
    for t in range(n_obs):
        y = actual[t]
        x = np.sort(ensemble[t])  # sorted draws

        # Term 1: (1/M) * sum|x_m - y|
        term1 = float(np.mean(np.abs(x - y)))

        # Term 2: (1/M^2) * sum_m x_m * (2m - M - 1)  (efficient formula)
        m_indices = np.arange(1, n_draws + 1)  # 1-based
        term2 = float(np.sum(x * (2.0 * m_indices - n_draws - 1.0))) / (n_draws * n_draws)

        total_crps += term1 - term2

    return total_crps / n_obs


def crps_gaussian(
    actual: NDArray[np.float64] | list[float],
    mean: NDArray[np.float64] | list[float],
    std: NDArray[np.float64] | list[float],
) -> float:
    """Analytical CRPS for Gaussian predictive distributions.

    CRPS_Normal(mu, sigma, y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]

    where z = (y - mu) / sigma, Phi = standard normal CDF, phi = standard normal PDF.

    Parameters
    ----------
    actual : array-like
        Actual observed values of length T.
    mean : array-like
        Predictive means of length T.
    std : array-like
        Predictive standard deviations of length T (must be positive).

    Returns
    -------
    float
        Mean CRPS across all time points.

    Raises
    ------
    ValueError
        If any std is non-positive.
    """
    actual = np.asarray(actual, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)

    if np.any(std <= 0):
        msg = "All standard deviations must be positive"
        raise ValueError(msg)

    z = (actual - mean) / std

    # CRPS = sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    phi_z = scipy_stats.norm.pdf(z)
    cdf_z = scipy_stats.norm.cdf(z)

    crps_values = std * (
        z * (2.0 * cdf_z - 1.0) + 2.0 * phi_z - 1.0 / math.sqrt(math.pi)
    )

    return float(np.mean(crps_values))
