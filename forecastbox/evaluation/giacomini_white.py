"""Giacomini-White test for conditional predictive ability.

Tests whether the relative predictive ability of two forecasts depends
on the state of the economy, using instrumental variables.

References
----------
Giacomini, R. & White, H. (2006). "Tests of Conditional Predictive Ability."
    Econometrica, 74(6), 1545-1578.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from forecastbox.evaluation._hac import newey_west


@dataclass
class GWResult:
    """Result of a Giacomini-White test.

    Attributes
    ----------
    statistic : float
        GW test statistic.
    pvalue : float
        p-value from chi-squared distribution.
    df : int
        Degrees of freedom (number of instruments).
    instruments_used : str
        Description of instruments used.
    """

    statistic: float
    pvalue: float
    df: int
    instruments_used: str

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
        if self.pvalue < alpha:
            return (
                f"Reject H0 at {alpha:.0%} level (p={self.pvalue:.4f}, "
                f"chi2({self.df})={self.statistic:.3f}). "
                f"Conditional predictive ability differs between forecasts. "
                f"Instruments: {self.instruments_used}."
            )
        return (
            f"Fail to reject H0 at {alpha:.0%} level (p={self.pvalue:.4f}, "
            f"chi2({self.df})={self.statistic:.3f}). "
            f"No significant difference in conditional predictive ability. "
            f"Instruments: {self.instruments_used}."
        )


def giacomini_white(
    actual: NDArray[np.float64] | list[float],
    forecast1: NDArray[np.float64] | list[float],
    forecast2: NDArray[np.float64] | list[float],
    h: int = 1,
    instruments: NDArray[np.float64] | None = None,
    loss: str = "mse",
) -> GWResult:
    """Perform the Giacomini-White test for conditional predictive ability.

    Tests H0: E[h_t * d_t] = 0, where h_t are instruments and
    d_t = L(e1_t) - L(e2_t) is the loss differential.

    Parameters
    ----------
    actual : array-like
        Actual observed values of length T.
    forecast1 : array-like
        First forecast of length T.
    forecast2 : array-like
        Second forecast of length T.
    h : int
        Forecast horizon. Used for HAC bandwidth when h > 1.
        Default: 1.
    instruments : NDArray or None
        Instrument matrix of shape (T, q). If None, uses
        h_t = [1, d_{t-1}] (constant + lagged loss differential),
        which means the first observation is dropped.
    loss : str
        Loss function: 'mse', 'mae', or 'mape'. Default: 'mse'.

    Returns
    -------
    GWResult
        Test result with statistic, p-value, and degrees of freedom.

    Raises
    ------
    ValueError
        If arrays have different lengths or invalid parameters.
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

    # Compute loss differential
    e1 = actual - forecast1
    e2 = actual - forecast2

    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "mape":
        d = np.abs(e1 / actual) - np.abs(e2 / actual)
    else:
        msg = f"Unknown loss: '{loss}'. Use 'mse', 'mae', or 'mape'."
        raise ValueError(msg)

    if instruments is None:
        # Default instruments: [1, d_{t-1}]
        # Drop first observation for the lag
        d_lag = d[:-1]
        d_current = d[1:]
        n = len(d_current)

        # Instrument matrix: (n, 2) = [constant, lagged loss diff]
        h_mat = np.column_stack([np.ones(n), d_lag])
        instruments_desc = "[1, d_{t-1}]"
    else:
        instruments = np.asarray(instruments, dtype=np.float64)
        if instruments.ndim == 1:
            instruments = instruments.reshape(-1, 1)
        if len(instruments) != len(d):
            msg = (
                f"Instruments must have same length as data. "
                f"Got {len(instruments)}, expected {len(d)}."
            )
            raise ValueError(msg)
        h_mat = instruments
        d_current = d
        n = len(d_current)
        instruments_desc = f"custom ({h_mat.shape[1]} instruments)"

    q = h_mat.shape[1]  # number of instruments

    # z_t = h_t * d_t (element-wise multiply each instrument by d_t)
    z = h_mat * d_current.reshape(-1, 1)  # (n, q)

    # z_bar
    z_bar = np.mean(z, axis=0).reshape(-1, 1)  # (q, 1)

    # Estimate Sigma using Newey-West if h > 1, simple variance otherwise
    sigma_hat = (
        newey_west(z, max_lag=h - 1)
        if h > 1
        else (z.T @ z) / n - z_bar @ z_bar.T
    )

    # GW statistic = n * z_bar' * Sigma^{-1} * z_bar
    try:
        sigma_inv = np.linalg.inv(sigma_hat)
    except np.linalg.LinAlgError:
        sigma_inv = np.linalg.pinv(sigma_hat)

    gw_stat = float((n * z_bar.T @ sigma_inv @ z_bar).item())
    gw_stat = max(gw_stat, 0.0)  # ensure non-negative

    # p-value from chi-squared(q)
    pvalue = float(1.0 - stats.chi2.cdf(gw_stat, df=q))

    return GWResult(
        statistic=gw_stat,
        pvalue=pvalue,
        df=q,
        instruments_used=instruments_desc,
    )
