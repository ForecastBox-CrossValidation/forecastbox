"""OLS forecast combination (Granger-Ramanathan 1984)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from forecastbox.combination.base import BaseCombiner


class OLSCombiner(BaseCombiner):
    """Forecast combination via OLS regression (Granger-Ramanathan 1984).

    Fits a regression of actual values on individual forecasts:

        y_t = beta_0 + sum_k(w_k * f_{k,t}) + epsilon_t

    Three variants:

    1. **Unconstrained** (intercept=True, constrained=False):
       Standard OLS. Weights can be negative and do not sum to 1.
       w = (X'X)^{-1} X'y

    2. **No intercept** (intercept=False, constrained=False):
       OLS without intercept. w = (F'F)^{-1} F'y

    3. **Constrained** (constrained=True):
       Solves: min ||y - F*w||^2  s.t. sum(w)=1, w >= 0
       via scipy.optimize.minimize with SLSQP.

    Parameters
    ----------
    intercept : bool
        Whether to include an intercept term. Default False.
    constrained : bool
        Whether to constrain weights: sum(w)=1, w>=0. Default True.
    regularization : str | None
        Regularization type: None, 'ridge', or 'lasso'. Default None.
    alpha : float
        Regularization strength. Default 0.01.

    Attributes
    ----------
    weights_ : NDArray[np.float64]
        Estimated combination weights.
    intercept_ : float
        Estimated intercept (0.0 if intercept=False).
    residuals_ : NDArray[np.float64]
        Residuals from the fitted regression.

    References
    ----------
    Granger, C.W.J. & Ramanathan, R. (1984). "Improved Methods of Combining
    Forecasts." *Journal of Forecasting*, 3(2), 197-204.
    """

    def __init__(
        self,
        intercept: bool = False,
        constrained: bool = True,
        regularization: str | None = None,
        alpha: float = 0.01,
    ) -> None:
        super().__init__()
        self.intercept = intercept
        self.constrained = constrained
        self.regularization = regularization
        self.alpha = alpha
        self.intercept_: float = 0.0
        self.residuals_: NDArray[np.float64] | None = None

    def fit(
        self,
        forecasts_train: list[NDArray[np.float64]],
        actual: NDArray[np.float64],
    ) -> OLSCombiner:
        """Estimate combination weights via OLS or constrained optimization.

        Parameters
        ----------
        forecasts_train : list[NDArray[np.float64]]
            List of K arrays, each of shape (T,), with historical forecasts.
        actual : NDArray[np.float64]
            Array of shape (T,) with realized values.

        Returns
        -------
        OLSCombiner
            self, for method chaining.
        """
        actual = np.asarray(actual, dtype=np.float64)
        k = len(forecasts_train)
        self.n_models_ = k

        # Build forecast matrix F: shape (T, K)
        f_matrix = np.column_stack(
            [np.asarray(fc, dtype=np.float64) for fc in forecasts_train]
        )

        if self.constrained:
            weights = self._fit_constrained(f_matrix, actual)
            self.intercept_ = 0.0
        else:
            weights = self._fit_unconstrained(f_matrix, actual)

        self.weights_ = weights

        # Compute residuals
        if self.intercept and not self.constrained:
            self.residuals_ = actual - (self.intercept_ + f_matrix @ weights)
        else:
            self.residuals_ = actual - f_matrix @ weights

        self.is_fitted_ = True
        return self

    def _fit_unconstrained(
        self,
        f_matrix: NDArray[np.float64],
        actual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Unconstrained OLS estimation.

        Parameters
        ----------
        f_matrix : NDArray[np.float64]
            Forecast matrix of shape (T, K).
        actual : NDArray[np.float64]
            Realized values of shape (T,).

        Returns
        -------
        NDArray[np.float64]
            Estimated weights of shape (K,).
        """
        x = np.column_stack([np.ones(len(actual)), f_matrix]) if self.intercept else f_matrix

        if self.regularization == "ridge":
            # Ridge: (X'X + alpha*I)^{-1} X'y
            n_cols = x.shape[1]
            reg_matrix = self.alpha * np.eye(n_cols)
            if self.intercept:
                reg_matrix[0, 0] = 0.0  # Don't regularize intercept
            beta = np.linalg.solve(x.T @ x + reg_matrix, x.T @ actual)
        else:
            # Standard OLS: (X'X)^{-1} X'y
            beta, _, _, _ = np.linalg.lstsq(x, actual, rcond=None)

        if self.intercept:
            self.intercept_ = float(beta[0])
            return beta[1:]
        return beta

    def _fit_constrained(
        self,
        f_matrix: NDArray[np.float64],
        actual: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Constrained estimation: sum(w)=1, w>=0 via SLSQP.

        Solves:
            min_w  ||y - F*w||^2
            s.t.   sum(w) = 1
                   w_k >= 0  for all k

        Parameters
        ----------
        f_matrix : NDArray[np.float64]
            Forecast matrix of shape (T, K).
        actual : NDArray[np.float64]
            Realized values of shape (T,).

        Returns
        -------
        NDArray[np.float64]
            Estimated weights of shape (K,), non-negative and summing to 1.
        """
        k = f_matrix.shape[1]

        def objective(w: NDArray[np.float64]) -> float:
            residuals = actual - f_matrix @ w
            return float(np.sum(residuals**2))

        def objective_jac(w: NDArray[np.float64]) -> NDArray[np.float64]:
            residuals = actual - f_matrix @ w
            return -2.0 * f_matrix.T @ residuals

        # Initial weights: equal
        w0 = np.full(k, 1.0 / k)

        # Constraints: sum(w) = 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Bounds: w_k >= 0
        bounds = [(0.0, None) for _ in range(k)]

        result = minimize(
            objective,
            w0,
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            msg = f"Constrained optimization failed: {result.message}"
            raise RuntimeError(msg)

        return np.asarray(result.x, dtype=np.float64)
