"""MIDAS (Mixed Data Sampling) regression for nowcasting (Ghysels et al 2004)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize


class MIDAS:
    """MIDAS regression for nowcasting with mixed-frequency data.

    Estimates a regression of a low-frequency variable on high-frequency
    regressors using parameterized weight functions (Beta, Almon, or Step).

    Parameters
    ----------
    target : str
        Name of the low-frequency target variable.
    high_freq : list[str]
        Names of high-frequency regressor variables.
    weight_scheme : str
        Weight parameterization: 'beta', 'almon', or 'step'.
    n_lags : int
        Number of high-frequency lags to include.
    poly_order : int
        Polynomial order for Almon weights.
    freq_ratio : int
        Frequency ratio (3 for quarterly/monthly, 22 for monthly/daily).

    Examples
    --------
    >>> midas = MIDAS(
    ...     target='pib_quarterly',
    ...     high_freq=['producao_industrial'],
    ...     weight_scheme='beta',
    ...     n_lags=12
    ... )
    >>> midas.fit(data)
    >>> nowcast = midas.nowcast()
    >>> print(f"Weights sum to: {midas.weights_.sum():.10f}")
    """

    def __init__(
        self,
        target: str,
        high_freq: list[str],
        weight_scheme: str = "beta",
        n_lags: int = 12,
        poly_order: int = 2,
        freq_ratio: int = 3,
    ) -> None:
        valid_schemes = {"beta", "almon", "step"}
        if weight_scheme not in valid_schemes:
            msg = f"weight_scheme must be one of {valid_schemes}, got '{weight_scheme}'"
            raise ValueError(msg)

        self.target = target
        self.high_freq = list(high_freq)
        self.weight_scheme = weight_scheme
        self.n_lags = n_lags
        self.poly_order = poly_order
        self.freq_ratio = freq_ratio

        # Fitted attributes
        self._fitted = False
        self._alpha: float = 0.0  # Intercept
        self._beta_coef: float = 1.0  # Scale coefficient
        self._theta: NDArray[np.float64] | None = None  # Weight parameters
        self._weights: NDArray[np.float64] | None = None  # Computed weights
        self._sigma2: float = 0.0
        self._n_obs: int = 0
        self._y: NDArray[np.float64] | None = None
        self._X_hf: NDArray[np.float64] | None = None
        self._data: pd.DataFrame | None = None

    @property
    def weights_(self) -> NDArray[np.float64]:
        """Return the estimated (normalized) weights.

        Returns
        -------
        NDArray[np.float64]
            Weight vector of shape (n_lags,) that sums to 1.
        """
        if not self._fitted or self._weights is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)
        return self._weights.copy()

    @staticmethod
    def _beta_weights(
        theta1: float, theta2: float, k: int
    ) -> NDArray[np.float64]:
        """Compute Beta polynomial weights (Ghysels et al 2006).

        Parameters
        ----------
        theta1 : float
            First shape parameter (must be > 0).
        theta2 : float
            Second shape parameter (must be > 0).
        k : int
            Number of lags.

        Returns
        -------
        NDArray[np.float64]
            Normalized weight vector of shape (k,).
        """
        # Ensure positive parameters
        theta1 = max(theta1, 1e-4)
        theta2 = max(theta2, 1e-4)

        # Compute unnormalized weights using Beta density
        eps = 1e-6
        x = np.linspace(eps, 1 - eps, k)

        # Beta density: f(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a,b)
        log_weights = (theta1 - 1) * np.log(x) + (theta2 - 1) * np.log(1 - x)

        # Normalize in log space for numerical stability
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

        # Normalize to sum to 1
        total = np.sum(weights)
        weights = weights / total if total > 0 else np.ones(k) / k

        return weights.astype(np.float64)

    @staticmethod
    def _almon_weights(
        theta: NDArray[np.float64], k: int
    ) -> NDArray[np.float64]:
        """Compute Almon polynomial weights.

        Parameters
        ----------
        theta : NDArray[np.float64]
            Polynomial coefficients of shape (poly_order,).
        k : int
            Number of lags.

        Returns
        -------
        NDArray[np.float64]
            Normalized weight vector of shape (k,).
        """
        j = np.arange(k, dtype=np.float64)

        # Compute polynomial: theta_1 * j + theta_2 * j^2 + ...
        log_weights = np.zeros(k)
        for p, theta_p in enumerate(theta):
            log_weights += theta_p * j ** (p + 1)

        # Normalize via softmax for numerical stability
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)

        total = np.sum(weights)
        weights = weights / total if total > 0 else np.ones(k) / k

        return weights.astype(np.float64)

    def _compute_weights(
        self, theta: NDArray[np.float64], scheme: str | None = None
    ) -> NDArray[np.float64]:
        """Compute weights given parameters and scheme.

        Parameters
        ----------
        theta : NDArray[np.float64]
            Weight parameters.
        scheme : str or None
            Weight scheme. If None, uses self.weight_scheme.

        Returns
        -------
        NDArray[np.float64]
            Normalized weights of shape (n_lags,).
        """
        if scheme is None:
            scheme = self.weight_scheme

        if scheme == "beta":
            return self._beta_weights(theta[0], theta[1], self.n_lags)
        elif scheme == "almon":
            return self._almon_weights(theta, self.n_lags)
        elif scheme == "step":
            # For step, theta ARE the weights (normalized)
            weights = np.abs(theta)
            total = np.sum(weights)
            if total > 0:
                return (weights / total).astype(np.float64)
            return (np.ones(self.n_lags) / self.n_lags).astype(np.float64)
        else:
            msg = f"Unknown weight scheme: {scheme}"
            raise ValueError(msg)

    def _build_hf_matrix(
        self, hf_data: pd.DataFrame, target_dates: pd.DatetimeIndex
    ) -> NDArray[np.float64]:
        """Build the high-frequency regressor matrix.

        For each low-frequency observation, collect n_lags high-frequency values.

        Parameters
        ----------
        hf_data : pd.DataFrame
            High-frequency data with DatetimeIndex.
        target_dates : pd.DatetimeIndex
            Dates of the low-frequency target observations.

        Returns
        -------
        NDArray[np.float64]
            Matrix of shape (n_target, n_lags * n_hf_vars).
        """
        n_target = len(target_dates)
        n_hf_vars = len(self.high_freq)
        x_mat = np.zeros((n_target, self.n_lags * n_hf_vars))

        for i, t_date in enumerate(target_dates):
            for v, var_name in enumerate(self.high_freq):
                series = hf_data[var_name]
                # Find the latest HF observation before/at the target date
                available = series[series.index <= t_date].dropna()
                if len(available) >= self.n_lags:
                    vals = available.values[-self.n_lags:]
                    x_mat[i, v * self.n_lags : (v + 1) * self.n_lags] = vals[::-1]

        return x_mat.astype(np.float64)

    def _objective(
        self,
        params: NDArray[np.float64],
        x_hf: NDArray[np.float64],
        y_lf: NDArray[np.float64],
    ) -> float:
        """NLS objective function for MIDAS estimation.

        Parameters
        ----------
        params : NDArray[np.float64]
            Parameter vector: [alpha, beta, theta_1, theta_2, ...]
        x_hf : NDArray[np.float64]
            High-frequency regressor matrix, shape (n_obs, n_lags * n_hf_vars).
        y_lf : NDArray[np.float64]
            Low-frequency target vector, shape (n_obs,).

        Returns
        -------
        float
            Sum of squared residuals.
        """
        alpha = params[0]
        beta_coef = params[1]
        theta = params[2:]

        n_hf_vars = len(self.high_freq)
        y_hat = np.full(len(y_lf), alpha)

        for v in range(n_hf_vars):
            weights = self._compute_weights(theta)
            x_v = x_hf[:, v * self.n_lags : (v + 1) * self.n_lags]
            y_hat += beta_coef * x_v @ weights

        residuals = y_lf - y_hat
        return float(np.sum(residuals**2))

    def fit(self, data: pd.DataFrame | dict[str, pd.Series]) -> MIDAS:
        """Estimate MIDAS regression.

        For 'beta' and 'almon': NLS estimation via scipy.optimize.
        For 'step': OLS estimation.

        Parameters
        ----------
        data : pd.DataFrame or dict[str, pd.Series]
            Panel data with target and high-frequency variables.

        Returns
        -------
        MIDAS
            Self, for method chaining.
        """
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        self._data = data.copy()

        if self.target not in data.columns:
            msg = f"Target '{self.target}' not found in data columns"
            raise ValueError(msg)

        missing_hf = [v for v in self.high_freq if v not in data.columns]
        if missing_hf:
            msg = f"High-frequency variables not found: {missing_hf}"
            raise ValueError(msg)

        # Get target (low frequency, non-NaN)
        target_series = data[self.target].dropna()
        target_dates = target_series.index

        # Build high-frequency regressor matrix
        hf_data = data[self.high_freq]
        x_hf = self._build_hf_matrix(hf_data, target_dates)
        y_lf = target_series.values.astype(np.float64)

        # Remove rows with all zeros in X (insufficient data)
        valid = np.any(x_hf != 0, axis=1)
        x_hf = x_hf[valid]
        y_lf = y_lf[valid]

        if len(y_lf) < 3:
            msg = f"Need at least 3 valid observations, got {len(y_lf)}"
            raise ValueError(msg)

        self._y = y_lf
        self._X_hf = x_hf

        if self.weight_scheme == "step":
            # OLS: unrestricted weights
            self._fit_step(x_hf, y_lf)
        else:
            # NLS: parametric weights
            self._fit_nls(x_hf, y_lf)

        self._n_obs = len(y_lf)
        self._fitted = True
        return self

    def _fit_step(
        self, x_hf: NDArray[np.float64], y_lf: NDArray[np.float64]
    ) -> None:
        """Fit unrestricted (step) weights via OLS.

        Parameters
        ----------
        x_hf : NDArray[np.float64]
            High-frequency regressors, shape (n_obs, n_lags * n_hf_vars).
        y_lf : NDArray[np.float64]
            Low-frequency target, shape (n_obs,).
        """
        n = len(y_lf)
        # Add intercept
        x_with_const = np.column_stack([np.ones(n), x_hf])

        # OLS
        try:
            params, _, _, _ = np.linalg.lstsq(x_with_const, y_lf, rcond=None)
        except np.linalg.LinAlgError:
            params = np.linalg.pinv(x_with_const) @ y_lf

        self._alpha = float(params[0])
        raw_weights = params[1:]

        # For multiple HF variables, extract per-variable weights
        n_hf_vars = len(self.high_freq)
        all_weights = []
        for v in range(n_hf_vars):
            w = raw_weights[v * self.n_lags : (v + 1) * self.n_lags]
            total = np.sum(np.abs(w))
            w_norm = np.abs(w) / total if total > 0 else np.ones(self.n_lags) / self.n_lags
            all_weights.append(w_norm)

        self._weights = all_weights[0].astype(np.float64)  # Primary HF variable
        self._beta_coef = float(np.sum(raw_weights[: self.n_lags]))
        self._theta = raw_weights.astype(np.float64)

        # Residual variance
        y_hat = x_with_const @ params
        residuals_arr = y_lf - y_hat
        self._sigma2 = float(np.sum(residuals_arr**2) / max(1, n - len(params)))

    def _fit_nls(
        self, x_hf: NDArray[np.float64], y_lf: NDArray[np.float64]
    ) -> None:
        """Fit parametric weights via NLS optimization.

        Parameters
        ----------
        x_hf : NDArray[np.float64]
            High-frequency regressors.
        y_lf : NDArray[np.float64]
            Low-frequency target.
        """
        # Initial parameter values
        alpha_init = float(np.mean(y_lf))
        beta_init = 1.0

        if self.weight_scheme == "beta":
            theta_init = np.array([1.0, 3.0])  # Decaying weights
            bounds: list[tuple[float | None, float | None]] = [
                (None, None),
                (None, None),
                (0.01, 50.0),
                (0.01, 50.0),
            ]
        elif self.weight_scheme == "almon":
            theta_init = np.zeros(self.poly_order)
            theta_init[0] = -0.01  # Slight decay
            bounds = [(None, None), (None, None)] + [(None, None)] * self.poly_order
        else:
            msg = f"NLS not supported for scheme: {self.weight_scheme}"
            raise ValueError(msg)

        params_init = np.concatenate([[alpha_init, beta_init], theta_init])

        # Optimize
        result = minimize(
            self._objective,
            params_init,
            args=(x_hf, y_lf),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success:
            # Try Nelder-Mead as fallback (no bounds)
            result = minimize(
                self._objective,
                params_init,
                args=(x_hf, y_lf),
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-10},
            )

        self._alpha = float(result.x[0])
        self._beta_coef = float(result.x[1])
        self._theta = result.x[2:].astype(np.float64)
        self._weights = self._compute_weights(self._theta)

        # Residual variance
        y_hat = self._predict(x_hf)
        residuals_arr = y_lf - y_hat
        n = len(y_lf)
        n_params = len(result.x)
        self._sigma2 = float(np.sum(residuals_arr**2) / max(1, n - n_params))

    def _predict(self, x_hf: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict using current parameters.

        Parameters
        ----------
        x_hf : NDArray[np.float64]
            High-frequency regressor matrix.

        Returns
        -------
        NDArray[np.float64]
            Predictions.
        """
        n_hf_vars = len(self.high_freq)
        y_hat = np.full(len(x_hf), self._alpha)

        for v in range(n_hf_vars):
            weights = (
                self._weights
                if self._weights is not None
                else np.ones(self.n_lags) / self.n_lags
            )
            x_v = x_hf[:, v * self.n_lags : (v + 1) * self.n_lags]
            y_hat += self._beta_coef * x_v @ weights

        return y_hat.astype(np.float64)

    def nowcast(self, data: pd.DataFrame | None = None) -> Any:
        """Generate nowcast using the estimated MIDAS model.

        Parameters
        ----------
        data : pd.DataFrame or None
            Updated data. If None, uses the data from fit().

        Returns
        -------
        Forecast
            Nowcast as a Forecast object.
        """
        if not self._fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        from forecastbox.core.forecast import Forecast

        if data is not None:
            if isinstance(data, dict):
                data = pd.DataFrame(data)
        else:
            data = self._data

        if data is None:
            msg = "No data available"
            raise RuntimeError(msg)

        # Get latest HF data for prediction
        hf_data = data[self.high_freq]

        # Collect the last n_lags observations per HF variable
        x_new = np.zeros(self.n_lags * len(self.high_freq))
        for v, var_name in enumerate(self.high_freq):
            series = hf_data[var_name].dropna()
            if len(series) >= self.n_lags:
                vals = series.values[-self.n_lags:]
                x_new[v * self.n_lags : (v + 1) * self.n_lags] = vals[::-1]

        # Predict
        x_pred = x_new.reshape(1, -1)
        point_est = float(self._predict(x_pred)[0])

        std_est = np.sqrt(max(self._sigma2, 1e-10))

        point = np.array([point_est])
        lower_80 = np.array([point_est - 1.28 * std_est])
        upper_80 = np.array([point_est + 1.28 * std_est])
        lower_95 = np.array([point_est - 1.96 * std_est])
        upper_95 = np.array([point_est + 1.96 * std_est])

        return Forecast(
            point=point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            model_name=f"MIDAS({self.weight_scheme},{self.n_lags})",
            horizon=1,
            metadata={
                "target": self.target,
                "high_freq": self.high_freq,
                "weight_scheme": self.weight_scheme,
                "n_lags": self.n_lags,
                "weights_sum": float(self.weights_.sum()),
            },
        )

    def plot_weights(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot the estimated MIDAS weight function.

        Parameters
        ----------
        ax : matplotlib Axes or None
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if not self._fitted or self._weights is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        lags = np.arange(self.n_lags)
        ax.bar(lags, self._weights, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Lag (high-frequency periods)")
        ax.set_ylabel("Weight")
        ax.set_title(
            f"MIDAS Weights ({self.weight_scheme.capitalize()}, "
            f"sum={self._weights.sum():.6f})"
        )
        ax.grid(True, alpha=0.3, axis="y")

        return ax

    def summary(self) -> str:
        """Return a text summary of the MIDAS estimation.

        Returns
        -------
        str
            Formatted summary string.
        """
        if not self._fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        lines = [
            "=" * 60,
            f"MIDAS Regression: {self.target}",
            "=" * 60,
            f"Weight scheme: {self.weight_scheme}",
            f"Number of lags: {self.n_lags}",
            f"Frequency ratio: {self.freq_ratio}",
            f"Observations: {self._n_obs}",
            f"Residual std: {np.sqrt(self._sigma2):.4f}",
            "-" * 60,
            "Parameters:",
            f"  alpha (intercept):  {self._alpha:.4f}",
            f"  beta (scale):       {self._beta_coef:.4f}",
        ]

        if self._theta is not None:
            lines.append(f"  theta (weights):    {self._theta}")

        if self._weights is not None:
            lines.append("-" * 60)
            lines.append("Weight statistics:")
            lines.append(f"  Sum:    {self._weights.sum():.10f}")
            lines.append(
                f"  Max:    {self._weights.max():.6f} (lag {np.argmax(self._weights)})"
            )
            lines.append(
                f"  Min:    {self._weights.min():.6f} (lag {np.argmin(self._weights)})"
            )
            lines.append(f"  Mean:   {self._weights.mean():.6f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"MIDAS(target='{self.target}', "
            f"scheme='{self.weight_scheme}', "
            f"n_lags={self.n_lags}, {status})"
        )
