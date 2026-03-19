"""Optimal forecast combination (Bates-Granger 1969)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forecastbox.combination.base import BaseCombiner


class OptimalCombiner(BaseCombiner):
    """Optimal forecast combination based on error covariance (Bates-Granger 1969).

    Derives combination weights that minimize the variance of the combined
    forecast, using the covariance matrix of forecast errors.

    For K=2 models with errors e1, e2:

        sigma1^2 = Var(e1), sigma2^2 = Var(e2), rho = Corr(e1, e2)

        w1* = (sigma2^2 - rho*sigma1*sigma2)
              / (sigma1^2 + sigma2^2 - 2*rho*sigma1*sigma2)
        w2* = 1 - w1*

    For general K models, with error covariance matrix Sigma and iota = (1,...,1)':

        w* = (Sigma^{-1} * iota) / (iota' * Sigma^{-1} * iota)

    These weights minimize Var(f_combined) = w' * Sigma * w subject to sum(w) = 1.

    The fundamental property is:

        Var(f_combined) = 1 / (iota' * Sigma^{-1} * iota) <= min_k Var(e_k)

    Parameters
    ----------
    shrinkage : float
        Shrinkage parameter in [0, 1] for the covariance matrix.
        Sigma_shrunk = (1-alpha)*Sigma_sample + alpha*diag(Sigma_sample).
        Default is 0.0 (no shrinkage).
    min_obs : int
        Minimum number of observations to estimate covariance. Default 20.

    Attributes
    ----------
    weights_ : NDArray[np.float64]
        Optimal combination weights w*.
    cov_matrix_ : NDArray[np.float64]
        Estimated (possibly shrunken) error covariance matrix.
    optimal_variance_ : float
        Minimum variance of the combined forecast:
        1 / (iota' * Sigma^{-1} * iota).
    individual_variances_ : NDArray[np.float64]
        Variance of each individual model's errors.

    References
    ----------
    Bates, J.M. & Granger, C.W.J. (1969). "The Combination of Forecasts."
    *Operational Research Quarterly*, 20(4), 451-468.
    """

    def __init__(
        self,
        shrinkage: float = 0.0,
        min_obs: int = 20,
    ) -> None:
        super().__init__()
        if not 0 <= shrinkage <= 1:
            msg = f"shrinkage must be in [0, 1], got {shrinkage}"
            raise ValueError(msg)
        if min_obs < 2:
            msg = f"min_obs must be >= 2, got {min_obs}"
            raise ValueError(msg)

        self.shrinkage = shrinkage
        self.min_obs = min_obs
        self.cov_matrix_: NDArray[np.float64] | None = None
        self.optimal_variance_: float | None = None
        self.individual_variances_: NDArray[np.float64] | None = None

    def fit(
        self,
        forecasts_train: list[NDArray[np.float64]],
        actual: NDArray[np.float64],
    ) -> OptimalCombiner:
        """Estimate optimal weights from the error covariance matrix.

        Parameters
        ----------
        forecasts_train : list[NDArray[np.float64]]
            List of K arrays, each of shape (T,), with historical forecasts.
        actual : NDArray[np.float64]
            Array of shape (T,) with realized values.

        Returns
        -------
        OptimalCombiner
            self, for method chaining.

        Raises
        ------
        ValueError
            If T < min_obs.
        """
        actual = np.asarray(actual, dtype=np.float64)
        k = len(forecasts_train)
        t = len(actual)
        self.n_models_ = k

        if t < self.min_obs:
            msg = (
                f"Need at least {self.min_obs} observations to estimate "
                f"covariance, got {t}."
            )
            raise ValueError(msg)

        # Compute forecast errors: shape (T, K)
        errors = np.column_stack(
            [actual - np.asarray(fc, dtype=np.float64) for fc in forecasts_train]
        )

        # Estimate covariance matrix with optional shrinkage
        cov_matrix = self._estimate_cov(errors)
        self.cov_matrix_ = cov_matrix
        self.individual_variances_ = np.diag(cov_matrix)

        # Compute optimal weights: w* = (Sigma^{-1} * iota) / (iota' * Sigma^{-1} * iota)
        iota = np.ones(k)

        try:
            sigma_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # If singular, use pseudoinverse
            sigma_inv = np.linalg.pinv(cov_matrix)

        sigma_inv_iota = sigma_inv @ iota
        denom = iota @ sigma_inv_iota

        if abs(denom) < 1e-15:
            # Degenerate case: equal weights
            self.weights_ = np.full(k, 1.0 / k)
            self.optimal_variance_ = float(np.min(self.individual_variances_))
        else:
            self.weights_ = sigma_inv_iota / denom
            # Optimal variance: 1 / (iota' * Sigma^{-1} * iota)
            self.optimal_variance_ = float(1.0 / denom)

        self.is_fitted_ = True
        return self

    def _estimate_cov(
        self,
        errors: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Estimate covariance matrix with optional shrinkage.

        Applies linear shrinkage toward the diagonal:
            Sigma_shrunk = (1-alpha)*Sigma_sample + alpha*diag(Sigma_sample)

        Parameters
        ----------
        errors : NDArray[np.float64]
            Error matrix of shape (T, K).

        Returns
        -------
        NDArray[np.float64]
            Covariance matrix of shape (K, K).
        """
        # Sample covariance (unbiased, ddof=1)
        sigma_sample = np.cov(errors, rowvar=False, ddof=1)

        # Ensure 2D (when K=1, np.cov returns scalar)
        if sigma_sample.ndim == 0:
            sigma_sample = np.array([[float(sigma_sample)]])

        if self.shrinkage == 0.0:
            return sigma_sample

        # Shrinkage toward diagonal
        diag_sigma = np.diag(np.diag(sigma_sample))
        sigma_shrunk = (
            (1.0 - self.shrinkage) * sigma_sample + self.shrinkage * diag_sigma
        )

        return sigma_shrunk
