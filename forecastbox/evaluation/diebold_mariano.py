"""Diebold-Mariano test for comparing predictive accuracy.

Implements the Diebold-Mariano (1995) test with the Harvey, Leybourne
and Newbold (1997) small-sample correction. Uses HAC variance estimation
via Newey-West for multi-step-ahead forecasts.

References
----------
Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy."
    Journal of Business & Economic Statistics, 13(3), 253-263.
Harvey, D., Leybourne, S. & Newbold, P. (1997). "Testing the equality of
    prediction mean squared errors." International Journal of Forecasting,
    13(2), 281-291.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from forecastbox.evaluation._hac import hac_variance


@dataclass
class DMResult:
    """Result of a Diebold-Mariano test.

    Attributes
    ----------
    statistic : float
        DM test statistic (with HLN correction if applied).
    pvalue : float
        p-value (two-sided or one-sided depending on test).
    loss_differential : NDArray[np.float64]
        Series d_t = L(e1_t) - L(e2_t).
    mean_loss_diff : float
        Mean of the loss differential (d_bar).
    h : int
        Forecast horizon used.
    loss : str
        Loss function used ('mse', 'mae', or 'mape').
    hln_corrected : bool
        Whether the HLN small-sample correction was applied.
    one_sided : bool
        Whether a one-sided test was performed.
    """

    statistic: float
    pvalue: float
    loss_differential: NDArray[np.float64]
    mean_loss_diff: float
    h: int
    loss: str
    hln_corrected: bool
    one_sided: bool

    def conclusion(self, alpha: float = 0.05) -> str:
        """Return textual interpretation of the test result.

        Parameters
        ----------
        alpha : float
            Significance level. Default: 0.05.

        Returns
        -------
        str
            Human-readable conclusion.
        """
        side = "one-sided" if self.one_sided else "two-sided"
        if self.pvalue < alpha:
            better = "forecast 1" if self.mean_loss_diff < 0 else "forecast 2"
            return (
                f"Reject H0 at {alpha:.0%} level ({side} p={self.pvalue:.4f}). "
                f"{better} is significantly more accurate (loss='{self.loss}')."
            )
        return (
            f"Fail to reject H0 at {alpha:.0%} level ({side} p={self.pvalue:.4f}). "
            f"No significant difference in predictive accuracy (loss='{self.loss}')."
        )


def _compute_loss_differential(
    actual: NDArray[np.float64],
    forecast1: NDArray[np.float64],
    forecast2: NDArray[np.float64],
    loss: str,
) -> NDArray[np.float64]:
    """Compute the loss differential series d_t = L(e1_t) - L(e2_t).

    Parameters
    ----------
    actual : NDArray[np.float64]
        Actual values.
    forecast1 : NDArray[np.float64]
        First forecast.
    forecast2 : NDArray[np.float64]
        Second forecast.
    loss : str
        Loss function: 'mse', 'mae', or 'mape'.

    Returns
    -------
    NDArray[np.float64]
        Loss differential series.
    """
    e1 = actual - forecast1
    e2 = actual - forecast2

    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "mape":
        d = np.abs(e1 / actual) - np.abs(e2 / actual)
    else:
        msg = f"Unknown loss function: '{loss}'. Use 'mse', 'mae', or 'mape'."
        raise ValueError(msg)

    return d


def diebold_mariano(
    actual: NDArray[np.float64] | list[float],
    forecast1: NDArray[np.float64] | list[float],
    forecast2: NDArray[np.float64] | list[float],
    h: int = 1,
    loss: str = "mse",
    one_sided: bool = False,
    hln_correction: bool = True,
) -> DMResult:
    """Perform the Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t).

    If d_bar < 0, forecast 1 has lower average loss (is better).
    If d_bar > 0, forecast 2 has lower average loss (is better).

    Parameters
    ----------
    actual : array-like
        Actual observed values of length T.
    forecast1 : array-like
        First forecast of length T.
    forecast2 : array-like
        Second forecast of length T.
    h : int
        Forecast horizon. Determines HAC truncation lag (h-1).
        Default: 1 (no autocorrelation correction needed).
    loss : str
        Loss function: 'mse', 'mae', or 'mape'. Default: 'mse'.
    one_sided : bool
        If True, perform one-sided test (H1: forecast 1 is better).
        Default: False (two-sided test).
    hln_correction : bool
        If True, apply the Harvey-Leybourne-Newbold (1997) small-sample
        correction. Default: True.

    Returns
    -------
    DMResult
        Test result with statistic, p-value, and diagnostics.

    Raises
    ------
    ValueError
        If arrays have different lengths or invalid loss function.
    """
    actual = np.asarray(actual, dtype=np.float64)
    forecast1 = np.asarray(forecast1, dtype=np.float64)
    forecast2 = np.asarray(forecast2, dtype=np.float64)

    if len(actual) != len(forecast1) or len(actual) != len(forecast2):
        msg = (
            f"All arrays must have the same length. "
            f"Got actual={len(actual)}, forecast1={len(forecast1)}, "
            f"forecast2={len(forecast2)}."
        )
        raise ValueError(msg)

    n = len(actual)

    if n < 3:
        msg = f"Need at least 3 observations, got {n}"
        raise ValueError(msg)

    if h < 1:
        msg = f"Horizon h must be >= 1, got {h}"
        raise ValueError(msg)

    # Compute loss differential
    d = _compute_loss_differential(actual, forecast1, forecast2, loss)
    d_bar = float(np.mean(d))

    # HAC variance with truncation lag = h-1
    max_lag = max(h - 1, 0)
    v_hat = hac_variance(d, max_lag=max_lag)

    # Avoid division by zero
    if v_hat <= 0:
        v_hat = float(np.var(d, ddof=1))
    if v_hat <= 0:
        # All loss differentials are identical
        return DMResult(
            statistic=0.0,
            pvalue=1.0,
            loss_differential=d,
            mean_loss_diff=d_bar,
            h=h,
            loss=loss,
            hln_corrected=hln_correction,
            one_sided=one_sided,
        )

    # DM statistic
    dm_stat = d_bar / math.sqrt(v_hat / n)

    # HLN correction
    if hln_correction:
        hln_factor = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_stat = dm_stat * hln_factor

    # p-value using t(T-1) distribution
    df = n - 1
    if one_sided:
        # One-sided: H1 is that forecast 1 is better (d_bar < 0)
        pvalue = float(stats.t.cdf(dm_stat, df=df))
    else:
        # Two-sided
        pvalue = float(2.0 * stats.t.sf(abs(dm_stat), df=df))

    return DMResult(
        statistic=dm_stat,
        pvalue=pvalue,
        loss_differential=d,
        mean_loss_diff=d_bar,
        h=h,
        loss=loss,
        hln_corrected=hln_correction,
        one_sided=one_sided,
    )
