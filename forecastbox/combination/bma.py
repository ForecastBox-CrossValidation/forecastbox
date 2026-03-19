"""Bayesian Model Averaging (BMA) forecast combination."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from forecastbox.combination.base import BaseCombiner
from forecastbox.core.forecast import Forecast


class BMACombiner(BaseCombiner):
    """Bayesian Model Averaging forecast combination.

    Computes posterior model probabilities using an approximation to the
    marginal likelihood, then combines forecasts with these posterior
    weights.

    The posterior probability of model k given data y is:

        P(M_k | y) = P(y | M_k) * P(M_k) / sum_j P(y | M_j) * P(M_j)

    The marginal likelihood is approximated via BIC:

        log P(y | M_k) ≈ -0.5 * BIC_k

    where BIC_k = T * ln(MSE_k) + p_k * ln(T).

    The BMA combined forecast is:

        E[y_{T+h} | y] = sum_k w_k * f_{k,T+h}

    The BMA variance includes both within-model and between-model uncertainty:

        Var[y_{T+h} | y] = sum_k w_k * (sigma^2_k + f^2_{k,T+h})
                           - (sum_k w_k * f_{k,T+h})^2

    Parameters
    ----------
    prior_weights : NDArray[np.float64] | None
        Prior model probabilities. If None, uniform prior (1/K).
        Must sum to 1.
    approximation : str
        Method for marginal likelihood approximation: 'bic', 'aic',
        or 'loglike'. Default 'bic'.
    n_params : list[int] | None
        Number of parameters for each model. Default [1, 1, ...].

    Attributes
    ----------
    weights_ : NDArray[np.float64]
        Posterior model weights (same as posterior_weights_).
    posterior_weights_ : NDArray[np.float64]
        Posterior model probabilities P(M_k | y).
    bma_variance_ : float | None
        BMA variance of the combined forecast (set after combine).
    model_mse_ : NDArray[np.float64]
        MSE of each model on training data.
    ic_values_ : NDArray[np.float64]
        Information criterion values used for weight computation.

    References
    ----------
    Hoeting, J.A., Madigan, D., Raftery, A.E. & Volinsky, C.T. (1999).
    "Bayesian Model Averaging: A Tutorial." *Statistical Science*, 14(4), 382-417.
    """

    def __init__(
        self,
        prior_weights: NDArray[np.float64] | None = None,
        approximation: str = "bic",
        n_params: list[int] | None = None,
    ) -> None:
        super().__init__()
        valid_approx = ("bic", "aic", "loglike")
        if approximation not in valid_approx:
            msg = (
                f"approximation must be one of {valid_approx}, "
                f"got '{approximation}'"
            )
            raise ValueError(msg)

        self.prior_weights_input = prior_weights
        self.approximation = approximation
        self.n_params = n_params
        self.posterior_weights_: NDArray[np.float64] | None = None
        self.bma_variance_: float | None = None
        self.model_mse_: NDArray[np.float64] | None = None
        self.ic_values_: NDArray[np.float64] | None = None

    def fit(
        self,
        forecasts_train: list[NDArray[np.float64]],
        actual: NDArray[np.float64],
        ic_values: NDArray[np.float64] | None = None,
    ) -> BMACombiner:
        """Compute posterior model weights from training data.

        Parameters
        ----------
        forecasts_train : list[NDArray[np.float64]]
            List of K arrays, each of shape (T,), with historical forecasts.
        actual : NDArray[np.float64]
            Array of shape (T,) with realized values.
        ic_values : NDArray[np.float64] | None
            Pre-computed information criterion values. If None, computed
            from forecasts_train and actual.

        Returns
        -------
        BMACombiner
            self, for method chaining.
        """
        actual = np.asarray(actual, dtype=np.float64)
        k = len(forecasts_train)
        t = len(actual)
        self.n_models_ = k

        # Prior weights
        if self.prior_weights_input is not None:
            prior = np.asarray(self.prior_weights_input, dtype=np.float64)
            if len(prior) != k:
                msg = f"prior_weights must have length {k}, got {len(prior)}"
                raise ValueError(msg)
            if abs(np.sum(prior) - 1.0) > 1e-8:
                msg = f"prior_weights must sum to 1, got {np.sum(prior)}"
                raise ValueError(msg)
        else:
            prior = np.full(k, 1.0 / k)

        # Compute MSE for each model
        mse = np.zeros(k)
        for i, fc in enumerate(forecasts_train):
            fc_arr = np.asarray(fc, dtype=np.float64)
            errors = actual - fc_arr
            mse[i] = np.mean(errors**2)

        self.model_mse_ = mse
        n_params = self.n_params if self.n_params is not None else [1] * k

        if ic_values is not None:
            ic = np.asarray(ic_values, dtype=np.float64)
        elif self.approximation == "bic":
            ic = np.zeros(k)
            for i in range(k):
                ic[i] = t * np.log(mse[i]) + n_params[i] * np.log(t)
        elif self.approximation == "aic":
            ic = np.zeros(k)
            for i in range(k):
                ic[i] = t * np.log(mse[i]) + 2 * n_params[i]
        elif self.approximation == "loglike":
            ic = np.zeros(k)
            for i in range(k):
                ic[i] = t * np.log(mse[i])
        else:
            msg = f"Unknown approximation: {self.approximation}"
            raise ValueError(msg)

        self.ic_values_ = ic

        # Compute posterior weights:
        # w_k = prior_k * exp(-0.5 * IC_k) / sum_j prior_j * exp(-0.5 * IC_j)
        # Use log-sum-exp trick for numerical stability
        log_unnorm = np.log(prior) - 0.5 * ic
        log_unnorm -= np.max(log_unnorm)  # log-sum-exp trick
        unnorm = np.exp(log_unnorm)
        posterior = unnorm / np.sum(unnorm)

        self.posterior_weights_ = posterior
        self.weights_ = posterior
        self.is_fitted_ = True
        return self

    def combine(self, forecasts: list[Forecast]) -> Forecast:
        """Combine forecasts using BMA posterior weights.

        Produces a combined forecast with BMA variance that accounts
        for both within-model and between-model uncertainty.

        Parameters
        ----------
        forecasts : list[Forecast]
            List of K Forecast objects to combine.

        Returns
        -------
        Forecast
            Combined forecast with BMA point and variance.
        """
        self._validate_forecasts(forecasts)

        if self.posterior_weights_ is None:
            msg = "BMACombiner must be fitted before calling combine()."
            raise ValueError(msg)

        weights = self.posterior_weights_
        points = np.array([fc.point for fc in forecasts])  # (K, H)

        # BMA point forecast: E[y] = sum_k w_k * f_k
        combined_point = weights @ points  # (H,)

        # BMA variance: Var[y] = sum_k w_k * (sigma^2_k + f^2_k)
        #                        - (sum_k w_k * f_k)^2
        # sigma^2_k estimated from training MSE
        if self.model_mse_ is not None:
            sigma2 = self.model_mse_  # (K,)
            bma_var = np.zeros(len(combined_point))
            for h in range(len(combined_point)):
                bma_var[h] = np.sum(
                    weights * (sigma2 + points[:, h] ** 2)
                ) - combined_point[h] ** 2

            self.bma_variance_ = float(np.mean(bma_var))

            # Use BMA variance for approximate prediction intervals
            bma_std = np.sqrt(np.maximum(bma_var, 0.0))
            lower_80 = combined_point - 1.28 * bma_std
            upper_80 = combined_point + 1.28 * bma_std
            lower_95 = combined_point - 1.96 * bma_std
            upper_95 = combined_point + 1.96 * bma_std
        else:
            lower_80 = None
            upper_80 = None
            lower_95 = None
            upper_95 = None

        model_names = [fc.model_name for fc in forecasts]

        return Forecast(
            point=combined_point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            index=forecasts[0].index,
            model_name="Combined(BMA)",
            metadata={
                "combiner": "BMACombiner",
                "posterior_weights": weights.tolist(),
                "bma_variance": self.bma_variance_,
                "models": model_names,
            },
        )

    def inclusion_probability(self, model_idx: int) -> float:
        """Return the posterior inclusion probability for a model.

        Parameters
        ----------
        model_idx : int
            Index of the model (0-based).

        Returns
        -------
        float
            Posterior probability P(M_k | y).
        """
        if self.posterior_weights_ is None:
            msg = "BMACombiner must be fitted first."
            raise ValueError(msg)
        if model_idx < 0 or model_idx >= len(self.posterior_weights_):
            msg = (
                f"model_idx must be in "
                f"[0, {len(self.posterior_weights_) - 1}], "
                f"got {model_idx}"
            )
            raise ValueError(msg)
        return float(self.posterior_weights_[model_idx])
