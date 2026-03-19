"""HAC (Heteroskedasticity and Autocorrelation Consistent) variance estimator.

Implements the Newey-West (1987) estimator with Bartlett kernel for robust
long-run variance estimation under heteroskedasticity and serial correlation.

References
----------
Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-Definite,
    Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
    Econometrica, 55(3), 703-708.
Newey, W.K. & West, K.D. (1994). "Automatic Lag Selection in Covariance
    Matrix Estimation." Review of Economic Studies, 61(4), 631-653.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def auto_bandwidth(n_obs: int) -> int:
    """Compute automatic bandwidth via Andrews rule.

    Uses the data-driven rule from Newey-West (1994):
        L = floor(4 * (T / 100)^{2/9})

    Parameters
    ----------
    n_obs : int
        Sample size (T).

    Returns
    -------
    int
        Optimal truncation lag (bandwidth).
    """
    if n_obs <= 0:
        msg = f"Sample size T must be positive, got {n_obs}"
        raise ValueError(msg)
    return int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def hac_variance(x: NDArray[np.float64], max_lag: int | None = None) -> float:
    """Compute HAC variance for a univariate series using Newey-West estimator.

    Estimates the long-run variance of x using the Bartlett kernel:

        hat_gamma = gamma_0 + 2 * sum_{j=1}^{L} (1 - j/(L+1)) * gamma_j

    where gamma_j is the j-th autocovariance of x.

    Parameters
    ----------
    x : NDArray[np.float64]
        Univariate time series of length T.
    max_lag : int or None
        Maximum lag (bandwidth). If None, uses auto_bandwidth(T).

    Returns
    -------
    float
        HAC variance estimate (long-run variance).

    Raises
    ------
    ValueError
        If x has fewer than 2 elements.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if n < 2:
        msg = f"Series must have at least 2 observations, got {n}"
        raise ValueError(msg)

    if max_lag is None:
        max_lag = auto_bandwidth(n)

    # Ensure max_lag is valid
    max_lag = min(max_lag, n - 1)
    max_lag = max(max_lag, 0)

    x_demeaned = x - np.mean(x)

    # Gamma_0: variance
    gamma_0 = float(np.dot(x_demeaned, x_demeaned) / n)

    # Sum weighted autocovariances with Bartlett kernel
    weighted_sum = 0.0
    for j in range(1, max_lag + 1):
        # Autocovariance at lag j
        gamma_j = float(np.dot(x_demeaned[j:], x_demeaned[:-j]) / n)
        # Bartlett kernel weight
        weight = 1.0 - j / (max_lag + 1.0)
        weighted_sum += weight * gamma_j

    return gamma_0 + 2.0 * weighted_sum


def newey_west(
    z: NDArray[np.float64], max_lag: int | None = None
) -> NDArray[np.float64]:
    """Compute Newey-West long-run covariance matrix.

    For a multivariate series z of shape (T, q), estimates:

        hat_Omega = Gamma_0 + sum_{j=1}^{L} k(j,L) * (Gamma_j + Gamma_j')

    where:
        Gamma_j = (1/T) * sum_{t=j+1}^{T} (z_t - z_bar)(z_{t-j} - z_bar)'
        k(j, L) = 1 - j/(L+1)   (Bartlett kernel)

    Parameters
    ----------
    z : NDArray[np.float64]
        Multivariate time series of shape (T, q). If 1-dimensional,
        treated as (T, 1).
    max_lag : int or None
        Maximum lag (bandwidth). If None, uses auto_bandwidth(T).

    Returns
    -------
    NDArray[np.float64]
        Long-run covariance matrix of shape (q, q).

    Raises
    ------
    ValueError
        If z has fewer than 2 observations.
    """
    z = np.asarray(z, dtype=np.float64)

    if z.ndim == 1:
        z = z.reshape(-1, 1)

    n, _q = z.shape

    if n < 2:
        msg = f"Series must have at least 2 observations, got {n}"
        raise ValueError(msg)

    if max_lag is None:
        max_lag = auto_bandwidth(n)

    max_lag = min(max_lag, n - 1)
    max_lag = max(max_lag, 0)

    z_demeaned = z - np.mean(z, axis=0)

    # Gamma_0
    omega: NDArray[np.float64] = (z_demeaned.T @ z_demeaned) / n

    # Add weighted cross-covariances
    for j in range(1, max_lag + 1):
        gamma_j = (z_demeaned[j:].T @ z_demeaned[:-j]) / n
        weight = 1.0 - j / (max_lag + 1.0)
        omega = omega + weight * (gamma_j + gamma_j.T)

    return omega
