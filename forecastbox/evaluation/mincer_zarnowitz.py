"""Mincer-Zarnowitz regression for forecast efficiency testing.

Tests whether a forecast is efficient (unbiased and optimally calibrated)
by regressing actual values on forecasts and testing H0: alpha=0, beta=1.

References
----------
Mincer, J.A. & Zarnowitz, V. (1969). "The Evaluation of Economic Forecasts."
    Economic Forecasts and Expectations, NBER, 1-46.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from forecastbox.evaluation._hac import newey_west


@dataclass
class MZResult:
    """Result of a Mincer-Zarnowitz regression.

    Attributes
    ----------
    alpha : float
        Estimated intercept.
    beta : float
        Estimated slope coefficient.
    alpha_se : float
        Standard error of alpha.
    beta_se : float
        Standard error of beta.
    alpha_tstat : float
        t-statistic for H0: alpha = 0.
    beta_tstat : float
        t-statistic for H0: beta = 1.
    f_statistic : float
        F-statistic for the joint test H0: alpha=0, beta=1.
    pvalue : float
        p-value of the joint F-test.
    r_squared : float
        R-squared of the regression.
    T : int
        Number of observations.
    """

    alpha: float
    beta: float
    alpha_se: float
    beta_se: float
    alpha_tstat: float
    beta_tstat: float
    f_statistic: float
    pvalue: float
    r_squared: float
    T: int

    def is_efficient(self, alpha: float = 0.05) -> bool:
        """Test whether the forecast is efficient at the given significance level.

        Parameters
        ----------
        alpha : float
            Significance level. Default: 0.05.

        Returns
        -------
        bool
            True if we fail to reject H0 (forecast is efficient).
        """
        return self.pvalue >= alpha

    def summary(self) -> str:
        """Return formatted summary of the regression results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            "Mincer-Zarnowitz Regression: y_t = alpha + beta * f_t + eps_t",
            "=" * 60,
            f"{'Parameter':<15} {'Estimate':>10} {'Std.Error':>10} {'t-stat':>10} {'p-value':>10}",
            "-" * 60,
            f"{'alpha':<15} {self.alpha:>10.4f} {self.alpha_se:>10.4f} "
            f"{self.alpha_tstat:>10.3f}"
            f" {float(2 * stats.t.sf(abs(self.alpha_tstat), df=self.T - 2)):>10.4f}",
            f"{'beta':<15} {self.beta:>10.4f} {self.beta_se:>10.4f} "
            f"{self.beta_tstat:>10.3f}"
            f" {float(2 * stats.t.sf(abs(self.beta_tstat), df=self.T - 2)):>10.4f}",
            "-" * 60,
            f"R-squared: {self.r_squared:.4f}",
            f"Joint test (alpha=0, beta=1): F={self.f_statistic:.3f}, p={self.pvalue:.4f}",
            f"Efficient: {'Yes' if self.is_efficient() else 'No'} (at 5% level)",
        ]
        return "\n".join(lines)


def mincer_zarnowitz(
    actual: NDArray[np.float64] | list[float],
    forecast: NDArray[np.float64] | list[float],
    hac: bool = False,
    h: int = 1,
) -> MZResult:
    """Perform the Mincer-Zarnowitz efficiency regression.

    Regresses actual values on forecasts:
        y_t = alpha + beta * f_t + epsilon_t

    Tests the joint hypothesis H0: alpha = 0, beta = 1
    (forecast is efficient / unbiased / well-calibrated).

    Parameters
    ----------
    actual : array-like
        Actual observed values of length T.
    forecast : array-like
        Forecast values of length T.
    hac : bool
        If True, use HAC (Newey-West) standard errors.
        Default: False (OLS standard errors).
    h : int
        Forecast horizon (used for HAC bandwidth when hac=True).
        Default: 1.

    Returns
    -------
    MZResult
        Regression results with coefficients, test statistics, and p-values.

    Raises
    ------
    ValueError
        If arrays have different lengths or too few observations.
    """
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)

    if len(actual) != len(forecast):
        msg = f"actual and forecast must have same length, got {len(actual)} and {len(forecast)}"
        raise ValueError(msg)

    n = len(actual)

    if n < 3:
        msg = f"Need at least 3 observations, got {n}"
        raise ValueError(msg)

    # OLS regression: y = alpha + beta * f + eps
    y = actual
    f = forecast

    # Design matrix
    x_mat = np.column_stack([np.ones(n), f])  # (n, 2)

    # OLS estimates: b = (X'X)^{-1} X'y
    xtx = x_mat.T @ x_mat
    xtx_inv = np.linalg.inv(xtx)
    b = xtx_inv @ (x_mat.T @ y)

    alpha_hat = float(b[0])
    beta_hat = float(b[1])

    # Residuals
    residuals = y - x_mat @ b

    # R-squared
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors
    if hac:
        # HAC standard errors via Newey-West
        # Score: X_t * epsilon_t
        scores = x_mat * residuals.reshape(-1, 1)  # (n, 2)
        max_lag = max(h - 1, 0)
        omega_hat = newey_west(scores, max_lag=max_lag)
        var_b = xtx_inv @ (n * omega_hat) @ xtx_inv
    else:
        # Classical OLS standard errors
        s2 = ss_res / (n - 2)
        var_b = s2 * xtx_inv

    alpha_se = float(np.sqrt(max(var_b[0, 0], 0.0)))
    beta_se = float(np.sqrt(max(var_b[1, 1], 0.0)))

    # Individual t-statistics
    alpha_tstat = alpha_hat / alpha_se if alpha_se > 0 else 0.0
    beta_tstat = (beta_hat - 1.0) / beta_se if beta_se > 0 else 0.0

    # Joint F-test: H0: alpha = 0 and beta = 1
    # R*b - r where R = I(2), r = [0, 1]
    r_vec = np.array([0.0, 1.0])
    diff = b - r_vec  # [alpha - 0, beta - 1]

    # F = (R*b - r)' * [R * Var(b) * R']^{-1} * (R*b - r) / q
    # Since R = I: F = diff' * Var(b)^{-1} * diff / 2
    try:
        var_b_inv = np.linalg.inv(var_b)
    except np.linalg.LinAlgError:
        var_b_inv = np.linalg.pinv(var_b)

    f_statistic = float(diff @ var_b_inv @ diff) / 2.0
    f_statistic = max(f_statistic, 0.0)

    # p-value from F(2, n-2) distribution
    pvalue = float(1.0 - stats.f.cdf(f_statistic, dfn=2, dfd=n - 2))

    return MZResult(
        alpha=alpha_hat,
        beta=beta_hat,
        alpha_se=alpha_se,
        beta_se=beta_se,
        alpha_tstat=alpha_tstat,
        beta_tstat=beta_tstat,
        f_statistic=f_statistic,
        pvalue=pvalue,
        r_squared=r_squared,
        T=n,
    )
