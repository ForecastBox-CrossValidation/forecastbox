"""Forecast encompassing test (Harvey, Leybourne & Newbold 1998).

Tests whether one forecast encompasses another, i.e., contains all
the useful predictive information of the other forecast.

References
----------
Harvey, D., Leybourne, S. & Newbold, P. (1998). "Tests for Forecast
    Encompassing." Journal of Business & Economic Statistics, 16(2), 254-259.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from forecastbox.evaluation._hac import hac_variance


@dataclass
class EncompassingResult:
    """Result of a forecast encompassing test.

    Attributes
    ----------
    lambda_hat : float
        Estimated weight of forecast 1 in the encompassing regression.
    lambda_se : float
        HAC standard error of lambda.
    statistic : float
        t-statistic for H0: lambda = 0.
    pvalue : float
        p-value (two-sided) from t(T-1) distribution.
    f1_encompasses_f2 : bool
        True if lambda ~ 1 (f1 contains all info; f2 is redundant).
    f2_encompasses_f1 : bool
        True if lambda ~ 0 (f2 contains all info; f1 is redundant).
    neither_encompasses : bool
        True if 0 < lambda < 1 (both have unique information).
    T : int
        Number of observations.
    """

    lambda_hat: float
    lambda_se: float
    statistic: float
    pvalue: float
    f1_encompasses_f2: bool
    f2_encompasses_f1: bool
    neither_encompasses: bool
    T: int

    def summary(self) -> str:
        """Return formatted summary of the encompassing test.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            "Encompassing Test (HLN 1998)",
            "y_t - f2_t = lambda * (f1_t - f2_t) + eps_t",
            "=" * 50,
            f"lambda_hat = {self.lambda_hat:.4f}  (SE = {self.lambda_se:.4f})",
            f"t-statistic = {self.statistic:.3f}",
            f"p-value = {self.pvalue:.4f}",
            "-" * 50,
            f"H0: lambda = 0 (f2 encompasses f1): "
            f"{'Not rejected' if self.f2_encompasses_f1 else 'Rejected'}",
            "",
        ]

        if self.f1_encompasses_f2:
            lines.append("Conclusion: f1 encompasses f2 (lambda ~ 1)")
        elif self.f2_encompasses_f1:
            lines.append("Conclusion: f2 encompasses f1 (lambda ~ 0)")
        elif self.neither_encompasses:
            lines.append("Conclusion: Neither encompasses the other (0 < lambda < 1)")
        else:
            lines.append("Conclusion: Inconclusive")

        return "\n".join(lines)


def encompassing_test(
    actual: NDArray[np.float64] | list[float],
    forecast1: NDArray[np.float64] | list[float],
    forecast2: NDArray[np.float64] | list[float],
    h: int = 1,
    alpha: float = 0.05,
) -> EncompassingResult:
    """Perform the HLN forecast encompassing test.

    Tests whether forecast 1 adds information beyond forecast 2
    by estimating the encompassing regression:

        y_t - f2_t = lambda * (f1_t - f2_t) + epsilon_t

    H0: lambda = 0 (f2 encompasses f1, f1 adds no information)
    H1: lambda != 0 (f1 has useful information not in f2)

    Parameters
    ----------
    actual : array-like
        Actual observed values of length T.
    forecast1 : array-like
        First forecast of length T.
    forecast2 : array-like
        Second forecast of length T.
    h : int
        Forecast horizon (for HAC bandwidth). Default: 1.
    alpha : float
        Significance level for determining encompassing conclusions.
        Default: 0.05.

    Returns
    -------
    EncompassingResult
        Test result with lambda estimate, t-statistic, and conclusions.

    Raises
    ------
    ValueError
        If arrays have different lengths or too few observations.
    """
    actual = np.asarray(actual, dtype=np.float64)
    forecast1 = np.asarray(forecast1, dtype=np.float64)
    forecast2 = np.asarray(forecast2, dtype=np.float64)

    if len(actual) != len(forecast1) or len(actual) != len(forecast2):
        msg = (
            f"All arrays must have same length. "
            f"Got {len(actual)}, {len(forecast1)}, {len(forecast2)}."
        )
        raise ValueError(msg)

    n = len(actual)

    if n < 3:
        msg = f"Need at least 3 observations, got {n}"
        raise ValueError(msg)

    # Encompassing regression (no intercept):
    # dep = y - f2
    # reg = f1 - f2
    dep = actual - forecast2
    reg = forecast1 - forecast2

    # OLS without intercept: lambda_hat = sum(dep * reg) / sum(reg^2)
    sum_reg2 = float(np.sum(reg**2))

    if sum_reg2 < 1e-15:
        # Forecasts are identical: inconclusive
        return EncompassingResult(
            lambda_hat=0.5,
            lambda_se=np.inf,
            statistic=0.0,
            pvalue=1.0,
            f1_encompasses_f2=False,
            f2_encompasses_f1=False,
            neither_encompasses=False,
            T=n,
        )

    lambda_hat = float(np.sum(dep * reg)) / sum_reg2

    # Residuals
    residuals = dep - lambda_hat * reg

    # HAC standard error
    # Score function: reg_t * residual_t
    scores = reg * residuals

    max_lag = max(h - 1, 0)
    v_scores = hac_variance(scores, max_lag=max_lag)

    # Variance of lambda_hat
    # Simplified: v_scores / (mean(reg^2))^2 / n
    mean_reg2 = sum_reg2 / n
    var_lambda = v_scores / (mean_reg2**2 * n)
    lambda_se = float(np.sqrt(max(var_lambda, 0.0)))

    # t-statistic for H0: lambda = 0
    t_stat = lambda_hat / lambda_se if lambda_se > 0 else 0.0

    # p-value from t(n-1)
    pvalue = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))

    # Determine encompassing conclusions
    # Test lambda = 0 (f2 encompasses f1)
    f2_encompasses_f1 = pvalue >= alpha  # cannot reject lambda=0

    # Test lambda = 1 (f1 encompasses f2)
    if lambda_se > 0:
        t_stat_1 = (lambda_hat - 1.0) / lambda_se
        pvalue_1 = float(2.0 * stats.t.sf(abs(t_stat_1), df=n - 1))
        f1_encompasses_f2 = pvalue_1 >= alpha  # cannot reject lambda=1
    else:
        f1_encompasses_f2 = False

    # Neither encompasses if lambda significantly different from both 0 and 1
    neither_encompasses = not f2_encompasses_f1 and not f1_encompasses_f2

    return EncompassingResult(
        lambda_hat=lambda_hat,
        lambda_se=lambda_se,
        statistic=t_stat,
        pvalue=pvalue,
        f1_encompasses_f2=f1_encompasses_f2,
        f2_encompasses_f1=f2_encompasses_f1,
        neither_encompasses=neither_encompasses,
        T=n,
    )
