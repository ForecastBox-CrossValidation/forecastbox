"""Protocols for model interfaces used by scenarios module."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class VARModelProtocol(Protocol):
    """Protocol for VAR model interface.

    Any VAR model (from chronobox, statsmodels, or custom) can be used
    with the scenarios module as long as it implements this protocol.
    """

    @property
    def coef(self) -> list[NDArray[np.float64]]:
        """VAR coefficient matrices [A_1, ..., A_p], each (k x k)."""
        ...

    @property
    def intercept(self) -> NDArray[np.float64]:
        """Intercept vector (k,)."""
        ...

    @property
    def sigma_u(self) -> NDArray[np.float64]:
        """Residual covariance matrix (k x k)."""
        ...

    @property
    def k_vars(self) -> int:
        """Number of endogenous variables."""
        ...

    @property
    def p_order(self) -> int:
        """VAR lag order."""
        ...

    @property
    def var_names(self) -> list[str]:
        """Names of endogenous variables."""
        ...

    @property
    def endog(self) -> NDArray[np.float64]:
        """Endogenous data array (T x k)."""
        ...

    @property
    def residuals(self) -> NDArray[np.float64]:
        """Residual array (T-p x k)."""
        ...


@runtime_checkable
class UnivariateForecastModelProtocol(Protocol):
    """Protocol for univariate forecast model interface.

    Used by MonteCarlo for univariate models (ARIMA, ETS, etc.).
    """

    @property
    def sigma2(self) -> float:
        """Estimated error variance."""
        ...

    @property
    def residuals(self) -> NDArray[np.float64]:
        """Model residuals."""
        ...

    def forecast(self, steps: int) -> NDArray[np.float64]:
        """Generate point forecasts for given number of steps."""
        ...

    def simulate(
        self,
        steps: int,
        n_paths: int,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Simulate future paths. Returns (n_paths, steps)."""
        ...


class SimpleVAR:
    """Simple VAR implementation for testing and standalone use.

    This is a minimal VAR(p) that implements VARModelProtocol.
    It estimates coefficients via OLS equation-by-equation.
    """

    def __init__(
        self,
        endog: NDArray[np.float64],
        p_order: int,
        var_names: list[str] | None = None,
    ) -> None:
        self._endog = np.asarray(endog, dtype=np.float64)
        self._p_order = p_order
        self._k_vars = self._endog.shape[1]
        self._var_names = var_names or [f"y{i}" for i in range(self._k_vars)]

        if len(self._var_names) != self._k_vars:
            msg = f"var_names length {len(self._var_names)} != k_vars {self._k_vars}"
            raise ValueError(msg)

        self._coef: list[NDArray[np.float64]] = []
        self._intercept: NDArray[np.float64] = np.zeros(self._k_vars)
        self._sigma_u: NDArray[np.float64] = np.eye(self._k_vars)
        self._residuals: NDArray[np.float64] = np.zeros((0, self._k_vars))

        self._fit()

    def _fit(self) -> None:
        """Estimate VAR coefficients via OLS."""
        n_total, k = self._endog.shape
        p = self._p_order

        if p + 1 >= n_total:
            msg = f"Not enough observations ({n_total}) for VAR({p})"
            raise ValueError(msg)

        # Build design matrix: y_mat = x_mat * beta + U
        # y_mat: (T-p, k), x_mat: (T-p, kp+1), beta: (kp+1, k)
        y_mat = self._endog[p:]  # (T-p, k)
        n_obs = n_total - p

        # x_mat = [y_{t-1}, y_{t-2}, ..., y_{t-p}, 1]
        x_mat = np.zeros((n_obs, k * p + 1))
        for lag in range(p):
            x_mat[:, lag * k : (lag + 1) * k] = self._endog[
                p - lag - 1 : n_total - lag - 1
            ]
        x_mat[:, -1] = 1.0  # intercept

        # OLS: beta = (X'X)^{-1} X'Y
        xtx = x_mat.T @ x_mat
        xty = x_mat.T @ y_mat
        beta = np.linalg.solve(xtx, xty)  # (kp+1, k)

        # Extract coefficients
        self._coef = []
        for lag in range(p):
            a_lag = beta[lag * k : (lag + 1) * k, :].T  # (k, k)
            self._coef.append(a_lag)

        self._intercept = beta[-1, :]  # (k,)

        # Residuals
        self._residuals = y_mat - x_mat @ beta  # (T-p, k)

        # Covariance of residuals
        self._sigma_u = (self._residuals.T @ self._residuals) / (n_obs - k * p - 1)

    @property
    def coef(self) -> list[NDArray[np.float64]]:
        return self._coef

    @property
    def intercept(self) -> NDArray[np.float64]:
        return self._intercept

    @property
    def sigma_u(self) -> NDArray[np.float64]:
        return self._sigma_u

    @property
    def k_vars(self) -> int:
        return self._k_vars

    @property
    def p_order(self) -> int:
        return self._p_order

    @property
    def var_names(self) -> list[str]:
        return self._var_names

    @property
    def endog(self) -> NDArray[np.float64]:
        return self._endog

    @property
    def residuals(self) -> NDArray[np.float64]:
        return self._residuals
