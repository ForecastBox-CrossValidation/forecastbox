"""Internal stepwise algorithm utilities for AutoARIMA.

This module provides the core functions used by the Hyndman-Khandakar (2008)
stepwise search algorithm:

- _generate_neighbors: generates candidate models by varying p, q, P, Q by +/-1
- _is_valid_order: validates order constraints (bounds, max_order)
- _kpss_test: KPSS test for determining differencing order d
- _ocsb_test: OCSB test for determining seasonal differencing order D

References
----------
Hyndman, R.J. & Khandakar, Y. (2008). "Automatic time series forecasting:
the forecast package for R." Journal of Statistical Software, 27(3), 1-22.

Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992).
"Testing the null hypothesis of stationarity against the alternative
of a unit root." Journal of Econometrics, 54(1-3), 159-178.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _generate_neighbors(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    include_constant: bool,
    max_p: int = 5,
    max_q: int = 5,
    max_seasonal_p: int = 2,
    max_seasonal_q: int = 2,
    max_order: int = 5,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int], bool]]:
    """Generate neighboring ARIMA models by varying p, q, P, Q by +/-1.

    The neighborhood includes:
    - p +/- 1 (holding q, P, Q fixed)
    - q +/- 1 (holding p, P, Q fixed)
    - P +/- 1 (holding p, q, Q fixed) if seasonal
    - Q +/- 1 (holding p, q, P fixed) if seasonal
    - p and q both +/- 1 simultaneously
    - Toggle include_constant

    Parameters
    ----------
    order : tuple[int, int, int]
        Current (p, d, q) order.
    seasonal_order : tuple[int, int, int, int]
        Current (P, D, Q, m) seasonal order.
    include_constant : bool
        Whether current model includes a constant.
    max_p : int
        Maximum p.
    max_q : int
        Maximum q.
    max_seasonal_p : int
        Maximum P.
    max_seasonal_q : int
        Maximum Q.
    max_order : int
        Maximum p + q + P + Q.

    Returns
    -------
    list[tuple[tuple[int, int, int], tuple[int, int, int, int], bool]]
        List of (order, seasonal_order, include_constant) tuples for valid neighbors.
    """
    p, d, q = order
    big_p, big_d, big_q, m = seasonal_order
    is_seasonal = m > 1

    neighbors: list[tuple[tuple[int, int, int], tuple[int, int, int, int], bool]] = []
    seen: set[tuple[int, int, int, int, int, int, bool]] = set()

    def _add_if_valid(
        np_: int, nq: int, n_big_p: int, n_big_q: int, nc: bool
    ) -> None:
        """Add neighbor if it passes validation and hasn't been seen."""
        key = (np_, d, nq, n_big_p, big_d, n_big_q, nc)
        if key in seen:
            return
        seen.add(key)

        if not _is_valid_order(
            np_, d, nq, n_big_p, big_d, n_big_q,
            max_p, max_q, max_seasonal_p, max_seasonal_q, max_order,
        ):
            return

        new_order = (np_, d, nq)
        new_seasonal = (n_big_p, big_d, n_big_q, m)
        neighbors.append((new_order, new_seasonal, nc))

    # Vary p by +/-1
    for dp in [-1, 1]:
        _add_if_valid(p + dp, q, big_p, big_q, include_constant)

    # Vary q by +/-1
    for dq in [-1, 1]:
        _add_if_valid(p, q + dq, big_p, big_q, include_constant)

    # Vary P by +/-1 (if seasonal)
    if is_seasonal:
        for d_big_p in [-1, 1]:
            _add_if_valid(p, q, big_p + d_big_p, big_q, include_constant)

    # Vary Q by +/-1 (if seasonal)
    if is_seasonal:
        for d_big_q in [-1, 1]:
            _add_if_valid(p, q, big_p, big_q + d_big_q, include_constant)

    # Vary p and q simultaneously by +/-1
    for dp in [-1, 1]:
        for dq in [-1, 1]:
            _add_if_valid(p + dp, q + dq, big_p, big_q, include_constant)

    # Toggle constant
    _add_if_valid(p, q, big_p, big_q, not include_constant)

    return neighbors


def _is_valid_order(
    p: int,
    d: int,
    q: int,
    big_p: int,
    big_d: int,
    big_q: int,
    max_p: int = 5,
    max_q: int = 5,
    max_seasonal_p: int = 2,
    max_seasonal_q: int = 2,
    max_order: int = 5,
) -> bool:
    """Validate ARIMA order constraints.

    Parameters
    ----------
    p : int
        AR order.
    d : int
        Differencing order.
    q : int
        MA order.
    big_p : int
        Seasonal AR order.
    big_d : int
        Seasonal differencing order.
    big_q : int
        Seasonal MA order.
    max_p : int
        Maximum p allowed.
    max_q : int
        Maximum q allowed.
    max_seasonal_p : int
        Maximum P allowed.
    max_seasonal_q : int
        Maximum Q allowed.
    max_order : int
        Maximum p + q + P + Q allowed.

    Returns
    -------
    bool
        True if the order is valid.
    """
    # Non-negative check
    if p < 0 or q < 0 or big_p < 0 or big_q < 0:
        return False

    # Upper bound checks
    if p > max_p or q > max_q:
        return False
    if big_p > max_seasonal_p or big_q > max_seasonal_q:
        return False

    # Total order constraint
    return (p + q + big_p + big_q) <= max_order


def _kpss_test(
    y: NDArray[np.float64],
    alpha: float = 0.05,
    regression: str = "c",
    nlags: str | int = "auto",
) -> bool:
    """Perform KPSS test for stationarity.

    The KPSS test has the null hypothesis that the series is stationary.
    If the test rejects (p-value < alpha), the series is non-stationary
    and needs differencing.

    Parameters
    ----------
    y : NDArray[np.float64]
        Time series data.
    alpha : float
        Significance level (default 0.05).
    regression : str
        Type of regression: 'c' (constant) or 'ct' (constant + trend).
    nlags : str or int
        Number of lags for HAC estimator. 'auto' uses Schwert (1989) rule.

    Returns
    -------
    bool
        True if the series IS stationary (fail to reject H0).
        False if the series is NOT stationary (reject H0, needs differencing).
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    if n < 6:
        # Too few observations; assume stationary
        return True

    # Determine number of lags for HAC estimator (Schwert 1989 rule for "auto")
    n_lags = int(4.0 * (n / 100.0) ** 0.25) if nlags == "auto" else int(nlags)

    n_lags = max(1, min(n_lags, n // 2))

    # Regression: demean or detrend
    if regression == "ct":
        # Detrend: y = a + b*t + residuals
        t = np.arange(1, n + 1, dtype=np.float64)
        x_mat = np.column_stack([np.ones(n), t])
        beta = np.linalg.lstsq(x_mat, y, rcond=None)[0]
        resid = y - x_mat @ beta
    else:
        # Demean
        resid = y - np.mean(y)

    # Partial sums of residuals
    s_t = np.cumsum(resid)

    # Estimate long-run variance using Bartlett kernel
    gamma_0 = float(np.sum(resid**2) / n)
    gamma_sum = 0.0
    for lag in range(1, n_lags + 1):
        weight = 1.0 - lag / (n_lags + 1)  # Bartlett kernel
        gamma_j = float(np.sum(resid[lag:] * resid[:-lag]) / n)
        gamma_sum += 2.0 * weight * gamma_j

    long_run_var = gamma_0 + gamma_sum

    if long_run_var <= 0:
        # Degenerate case
        return True

    # KPSS statistic
    kpss_stat = float(np.sum(s_t**2)) / (n**2 * long_run_var)

    # Critical values (asymptotic, from Kwiatkowski et al. 1992, Table 1)
    if regression == "ct":
        # Critical values for level + trend
        critical_values = {0.10: 0.119, 0.05: 0.146, 0.025: 0.176, 0.01: 0.216}
    else:
        # Critical values for level only
        critical_values = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}

    # Find the critical value for our alpha
    # Use the closest available alpha
    available_alphas = sorted(critical_values.keys())
    chosen_alpha = min(available_alphas, key=lambda a: abs(a - alpha))
    critical_value = critical_values[chosen_alpha]

    # Reject H0 (stationarity) if test statistic > critical value
    is_stationary = kpss_stat <= critical_value

    return is_stationary


def _ocsb_test(
    y: NDArray[np.float64],
    m: int,
    alpha: float = 0.05,
) -> bool:
    """Perform OCSB test for seasonal unit root.

    Simplified OCSB (Osborn, Chui, Smith, Birchenhall) test to determine
    whether seasonal differencing is needed. Tests whether the seasonal
    pattern is deterministic or stochastic.

    The approach uses a regression-based test: if the seasonal lag
    coefficient is significant, seasonal differencing is needed.

    Parameters
    ----------
    y : NDArray[np.float64]
        Time series data.
    m : int
        Seasonal period (e.g., 12 for monthly, 4 for quarterly).
    alpha : float
        Significance level.

    Returns
    -------
    bool
        True if seasonal differencing is NOT needed (series is seasonally stationary).
        False if seasonal differencing IS needed.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    if m <= 1:
        # No seasonality possible
        return True

    if n < 2 * m + 1:
        # Not enough data for seasonal test
        return True

    # Compute seasonal strength using ACF at seasonal lag
    # If the autocorrelation at lag m is strong, seasonal differencing may be needed
    mean_y = np.mean(y)
    y_centered = y - mean_y

    # Variance
    var_y = float(np.sum(y_centered**2) / n)
    if var_y == 0:
        return True

    # ACF at lag m
    acf_m = float(np.sum(y_centered[m:] * y_centered[:-m]) / n) / var_y

    # Compute seasonal differences
    y_sdiff = y[m:] - y[:-m]

    # Compute variance ratio: Var(seasonal_diff) / Var(original)
    var_sdiff = float(np.var(y_sdiff))
    if var_y == 0:
        return True

    variance_ratio = var_sdiff / var_y

    # Heuristic decision rule (inspired by Canova-Hansen and OCSB approaches):
    # If the variance ratio is close to 2 (what we'd expect for white noise
    # seasonal differences) AND the seasonal ACF is strong,
    # then seasonal differencing is needed.
    #
    # More specifically:
    # - High seasonal ACF (> critical value) suggests seasonal pattern
    # - If variance reduces after seasonal differencing, it was needed

    # Critical value for ACF significance at lag m
    acf_critical = 1.96 / np.sqrt(n)

    # Seasonal differencing is needed if:
    # 1. Seasonal ACF is significantly positive, AND
    # 2. Seasonal differencing reduces variance
    needs_seasonal_diff = (acf_m > acf_critical) and (variance_ratio < 1.5)

    return not needs_seasonal_diff


def _determine_d(
    y: NDArray[np.float64],
    max_d: int = 2,
    alpha: float = 0.05,
) -> int:
    """Determine the order of differencing d via sequential KPSS tests.

    Algorithm:
    1. Test y: if KPSS says stationary, d = 0
    2. Test diff(y): if KPSS says stationary, d = 1
    3. Otherwise d = 2

    Parameters
    ----------
    y : NDArray[np.float64]
        Time series data.
    max_d : int
        Maximum differencing order (default 2).
    alpha : float
        Significance level for KPSS test.

    Returns
    -------
    int
        Recommended differencing order.
    """
    y = np.asarray(y, dtype=np.float64)

    for d in range(max_d + 1):
        test_series = y if d == 0 else np.diff(y, n=d)

        if len(test_series) < 6:
            return d

        if _kpss_test(test_series, alpha=alpha):
            return d

    return max_d


def _determine_seasonal_d(
    y: NDArray[np.float64],
    m: int,
    max_d_seasonal: int = 1,
    alpha: float = 0.05,
) -> int:
    """Determine the order of seasonal differencing D.

    Algorithm:
    1. If m <= 1: D = 0
    2. Test y with OCSB: if seasonally stationary, D = 0
    3. Otherwise D = 1

    Parameters
    ----------
    y : NDArray[np.float64]
        Time series data.
    m : int
        Seasonal period.
    max_d_seasonal : int
        Maximum seasonal differencing order (default 1).
    alpha : float
        Significance level.

    Returns
    -------
    int
        Recommended seasonal differencing order.
    """
    if m <= 1:
        return 0

    y = np.asarray(y, dtype=np.float64)

    if len(y) < 2 * m:
        return 0

    if _ocsb_test(y, m=m, alpha=alpha):
        return 0

    return min(1, max_d_seasonal)
