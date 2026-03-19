"""DFM Nowcaster with mixed frequency support via state-space models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class KalmanResult:
    """Result from Kalman filter/smoother operations.

    Attributes
    ----------
    filtered_state : NDArray
        Filtered state estimates, shape (T, state_dim).
    filtered_cov : NDArray
        Filtered state covariance, shape (T, state_dim, state_dim).
    smoothed_state : NDArray or None
        Smoothed state estimates, shape (T, state_dim).
    smoothed_cov : NDArray or None
        Smoothed state covariance, shape (T, state_dim, state_dim).
    log_likelihood : float
        Log-likelihood of the data.
    """

    filtered_state: NDArray[np.float64]
    filtered_cov: NDArray[np.float64]
    smoothed_state: NDArray[np.float64] | None = None
    smoothed_cov: NDArray[np.float64] | None = None
    log_likelihood: float = 0.0


class KalmanBoxAdapter:
    """Adapter for kalmanbox integration.

    Provides a unified interface for Kalman filtering and smoothing.
    Falls back to a standalone implementation if kalmanbox is not installed.
    """

    def __init__(self) -> None:
        self._kf: Any = None
        self._smoother: Any = None
        self._dfm: Any = None
        try:
            from kalmanbox import DFM, KalmanFilter, RTSSmoother  # type: ignore[import-untyped]

            self._kf_cls = KalmanFilter
            self._smoother_cls = RTSSmoother
            self._dfm_cls = DFM
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        """Whether kalmanbox is installed and available."""
        return self._available

    def filter(
        self,
        y: NDArray[np.float64],
        Z: NDArray[np.float64],
        T: NDArray[np.float64],
        Q: NDArray[np.float64],
        R: NDArray[np.float64],
        a0: NDArray[np.float64],
        P0: NDArray[np.float64],
        missing_mask: NDArray[np.bool_] | None = None,
    ) -> KalmanResult:
        """Run Kalman filter on the state-space model.

        Parameters
        ----------
        y : NDArray, shape (T_obs, n_obs)
            Observation matrix.
        Z : NDArray, shape (n_obs, state_dim) or (T_obs, n_obs, state_dim)
            Observation/measurement matrix. Time-varying if 3D.
        T : NDArray, shape (state_dim, state_dim)
            Transition matrix.
        Q : NDArray, shape (state_dim, state_dim)
            State noise covariance.
        R : NDArray, shape (n_obs, n_obs)
            Observation noise covariance (diagonal).
        a0 : NDArray, shape (state_dim,)
            Initial state mean.
        P0 : NDArray, shape (state_dim, state_dim)
            Initial state covariance.
        missing_mask : NDArray[bool] or None, shape (T_obs, n_obs)
            True where data is missing.

        Returns
        -------
        KalmanResult
            Filtered state estimates and covariances.
        """
        n_t, n_obs = y.shape
        state_dim = T.shape[0]

        filtered_state = np.zeros((n_t, state_dim))
        filtered_cov = np.zeros((n_t, state_dim, state_dim))
        log_lik = 0.0

        a = a0.copy()
        P = P0.copy()

        for t in range(n_t):
            # Prediction step
            a_pred = T @ a
            P_pred = T @ P @ T.T + Q

            # Get Z for this time step
            Z_t = Z[t] if Z.ndim == 3 else Z

            # Handle missing data
            obs_mask = ~missing_mask[t] if missing_mask is not None else ~np.isnan(y[t])

            if np.any(obs_mask):
                # Select observed variables
                y_obs = y[t][obs_mask]
                Z_obs = Z_t[obs_mask]
                R_obs = R[np.ix_(obs_mask, obs_mask)]

                # Innovation
                v = y_obs - Z_obs @ a_pred
                F = Z_obs @ P_pred @ Z_obs.T + R_obs

                # Kalman gain
                try:
                    F_inv = np.linalg.inv(F)
                except np.linalg.LinAlgError:
                    F_inv = np.linalg.pinv(F)

                K = P_pred @ Z_obs.T @ F_inv

                # Update
                a = a_pred + K @ v
                P = P_pred - K @ Z_obs @ P_pred

                # Log-likelihood contribution
                sign, logdet = np.linalg.slogdet(F)
                if sign > 0:
                    log_lik += -0.5 * (
                        len(y_obs) * np.log(2 * np.pi) + logdet + v @ F_inv @ v
                    )
            else:
                a = a_pred
                P = P_pred

            filtered_state[t] = a
            filtered_cov[t] = P

        return KalmanResult(
            filtered_state=filtered_state,
            filtered_cov=filtered_cov,
            log_likelihood=log_lik,
        )

    def smooth(
        self,
        y: NDArray[np.float64],
        Z: NDArray[np.float64],
        T: NDArray[np.float64],
        Q: NDArray[np.float64],
        R: NDArray[np.float64],
        a0: NDArray[np.float64],
        P0: NDArray[np.float64],
        missing_mask: NDArray[np.bool_] | None = None,
    ) -> KalmanResult:
        """Run Kalman filter + RTS smoother.

        Parameters are the same as filter(). Returns KalmanResult with
        both filtered and smoothed estimates.
        """
        # First run filter
        result = self.filter(y, Z, T, Q, R, a0, P0, missing_mask)

        n_t = y.shape[0]
        state_dim = T.shape[0]

        smoothed_state = np.zeros((n_t, state_dim))
        smoothed_cov = np.zeros((n_t, state_dim, state_dim))

        # Initialize with last filtered values
        smoothed_state[-1] = result.filtered_state[-1]
        smoothed_cov[-1] = result.filtered_cov[-1]

        # Backward pass (RTS smoother)
        for t in range(n_t - 2, -1, -1):
            a_filt = result.filtered_state[t]
            P_filt = result.filtered_cov[t]

            a_pred = T @ a_filt
            P_pred = T @ P_filt @ T.T + Q

            try:
                P_pred_inv = np.linalg.inv(P_pred)
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.pinv(P_pred)

            J = P_filt @ T.T @ P_pred_inv

            smoothed_state[t] = a_filt + J @ (smoothed_state[t + 1] - a_pred)
            smoothed_cov[t] = P_filt + J @ (smoothed_cov[t + 1] - P_pred) @ J.T

        return KalmanResult(
            filtered_state=result.filtered_state,
            filtered_cov=result.filtered_cov,
            smoothed_state=smoothed_state,
            smoothed_cov=smoothed_cov,
            log_likelihood=result.log_likelihood,
        )


class DFMNowcaster:
    """Nowcaster based on Dynamic Factor Model with mixed frequencies.

    Extracts common latent factors from a panel of indicators with potentially
    different frequencies (monthly, quarterly). Uses the state-space representation
    with Mariano-Murasawa (2003) triangular matrix for flow variables.

    Parameters
    ----------
    n_factors : int
        Number of latent factors to extract.
    factor_lags : int
        Number of lags in the factor VAR dynamics.
    frequency_map : dict[str, str]
        Mapping from variable name to frequency ('M' for monthly, 'Q' for quarterly).
    aggregation : str
        Aggregation method for quarterly variables: 'sum' (flow) or 'last' (stock).
    em_iterations : int
        Maximum EM algorithm iterations.
    em_tol : float
        Convergence tolerance for EM algorithm.

    Examples
    --------
    >>> nowcaster = DFMNowcaster(n_factors=2, frequency_map={
    ...     'producao_industrial': 'M',
    ...     'vendas_varejo': 'M',
    ...     'pib': 'Q'
    ... })
    >>> nowcaster.fit(data)
    >>> nowcast = nowcaster.nowcast(target='pib')
    """

    def __init__(
        self,
        n_factors: int = 2,
        factor_lags: int = 2,
        frequency_map: dict[str, str] | None = None,
        aggregation: str = "sum",
        em_iterations: int = 100,
        em_tol: float = 1e-6,
    ) -> None:
        self.n_factors = n_factors
        self.factor_lags = factor_lags
        self.frequency_map = frequency_map or {}
        self.aggregation = aggregation
        self.em_iterations = em_iterations
        self.em_tol = em_tol

        # Fitted attributes
        self._fitted = False
        self._Lambda: NDArray[np.float64] | None = None  # Loadings (n_vars, n_factors)
        self._A: NDArray[np.float64] | None = None  # Transition matrices
        self._Q: NDArray[np.float64] | None = None  # State noise covariance
        self._R: NDArray[np.float64] | None = None  # Observation noise covariance
        self._factors_df: pd.DataFrame | None = None  # Estimated factors
        self._data: pd.DataFrame | None = None  # Stored data
        self._var_names: list[str] = []
        self._monthly_vars: list[str] = []
        self._quarterly_vars: list[str] = []
        self._adapter = KalmanBoxAdapter()
        self._kalman_result: KalmanResult | None = None

    def _classify_variables(self) -> None:
        """Classify variables into monthly and quarterly."""
        self._monthly_vars = [v for v, f in self.frequency_map.items() if f == "M"]
        self._quarterly_vars = [v for v, f in self.frequency_map.items() if f == "Q"]

    @property
    def _state_dim(self) -> int:
        """Dimension of the state vector."""
        # factors * factor_lags + accumulated flow states for quarterly vars
        base = self.n_factors * self.factor_lags
        if self.aggregation == "sum":
            # Need 2 extra lags per quarterly variable for flow accumulation
            base += self.n_factors * 2
        return base

    def _build_state_space(
        self,
        data: pd.DataFrame,
    ) -> tuple[
        NDArray[np.float64],  # y (T, n_obs)
        NDArray[np.float64],  # Z (T, n_obs, state_dim) time-varying
        NDArray[np.float64],  # T_mat (state_dim, state_dim)
        NDArray[np.float64],  # Q_mat (state_dim, state_dim)
        NDArray[np.float64],  # R_mat (n_obs, n_obs)
        NDArray[np.bool_],  # missing_mask (T, n_obs)
    ]:
        """Build the complete state-space representation.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with DatetimeIndex. Columns are variable names.

        Returns
        -------
        tuple
            (y, Z, T_mat, Q_mat, R_mat, missing_mask)
        """
        n_t = len(data)
        n_obs = len(self._var_names)
        state_dim = self._state_dim
        nf = self.n_factors

        # Observation matrix
        y = data[self._var_names].values.astype(np.float64)

        # Missing mask
        missing_mask = np.isnan(y)

        # Quarterly variables: only observed in quarter-end months (3, 6, 9, 12)
        for i, var in enumerate(self._var_names):
            if var in self._quarterly_vars:
                for t in range(n_t):
                    month = data.index[t].month
                    if month not in (3, 6, 9, 12):
                        missing_mask[t, i] = True

        # Fill NaN with 0 for computation (masked out anyway)
        y_filled = np.where(missing_mask, 0.0, y)

        # Transition matrix T
        T_mat = np.zeros((state_dim, state_dim))

        # Factor VAR block: [f_t, f_{t-1}, ..., f_{t-p+1}]
        if self._A is not None:
            for lag in range(self.factor_lags):
                T_mat[:nf, lag * nf : (lag + 1) * nf] = self._A[
                    :, lag * nf : (lag + 1) * nf
                ]
        else:
            T_mat[:nf, :nf] = 0.5 * np.eye(nf)

        # Shift block for lags: f_{t-k} = f_{t-k}
        for lag in range(1, self.factor_lags):
            row_start = lag * nf
            col_start = (lag - 1) * nf
            T_mat[row_start : row_start + nf, col_start : col_start + nf] = np.eye(nf)

        # Flow accumulation block for quarterly variables (Mariano-Murasawa)
        if self.aggregation == "sum" and nf > 0:
            acc_start = nf * self.factor_lags
            # f^1_t = f_{t-1} (first lag of accumulated)
            T_mat[acc_start : acc_start + nf, :nf] = np.eye(nf)
            # f^2_t = f^1_{t-1} (second lag of accumulated)
            T_mat[acc_start + nf : acc_start + 2 * nf, acc_start : acc_start + nf] = (
                np.eye(nf)
            )

        # State noise Q
        Q_mat = np.zeros((state_dim, state_dim))
        if self._Q is not None:
            Q_mat[:nf, :nf] = self._Q
        else:
            Q_mat[:nf, :nf] = np.eye(nf) * 0.1

        # Observation noise R (diagonal)
        R_mat = self._R if self._R is not None else np.eye(n_obs) * 1.0

        # Time-varying measurement matrix Z (for mixed frequency)
        Z = np.zeros((n_t, n_obs, state_dim))

        for i, var in enumerate(self._var_names):
            if var in self._monthly_vars:
                # Monthly: Z_t = [lambda_i, 0, 0, ...]
                if self._Lambda is not None:
                    Z[:, i, :nf] = self._Lambda[i]
                else:
                    Z[:, i, :nf] = 1.0 / nf  # Initial guess
            elif var in self._quarterly_vars:
                # Quarterly: Mariano-Murasawa triangular accumulation
                for t in range(n_t):
                    month_in_quarter = (data.index[t].month - 1) % 3  # 0, 1, 2

                    lam = self._Lambda[i] if self._Lambda is not None else np.ones(nf) / nf

                    if self.aggregation == "sum":
                        # Flow: accumulate
                        acc_start = nf * self.factor_lags
                        Z[t, i, :nf] = lam  # Current factor

                        if month_in_quarter >= 1:
                            Z[t, i, acc_start : acc_start + nf] = lam  # f_{t-1}
                        if month_in_quarter >= 2:
                            Z[t, i, acc_start + nf : acc_start + 2 * nf] = (
                                lam  # f_{t-2}
                            )
                    else:
                        # Stock: just current factor
                        Z[t, i, :nf] = lam

        return y_filled, Z, T_mat, Q_mat, R_mat, missing_mask

    def _handle_missing(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
        """Handle missing data in the panel.

        Parameters
        ----------
        data : pd.DataFrame
            Input panel data.

        Returns
        -------
        tuple
            (data_standardized, missing_mask)
        """
        # Standardize column selection
        available_vars = [v for v in self._var_names if v in data.columns]
        df = data[available_vars].copy()

        # Ensure float
        df = df.astype(np.float64)

        # Create missing mask
        mask = df.isna().values

        return df, mask

    def _mixed_frequency_ssm(
        self,
    ) -> tuple[int, list[str], list[str]]:
        """Determine the SSM structure for mixed frequency data.

        Returns
        -------
        tuple
            (state_dim, monthly_vars, quarterly_vars)
        """
        self._classify_variables()
        return self._state_dim, self._monthly_vars, self._quarterly_vars

    def fit(self, data: pd.DataFrame | dict[str, pd.Series]) -> DFMNowcaster:
        """Estimate the DFM via EM algorithm with Kalman filter/smoother.

        Parameters
        ----------
        data : pd.DataFrame or dict[str, pd.Series]
            Panel data. If dict, keys are variable names and values are Series
            with DatetimeIndex. If DataFrame, columns are variable names.

        Returns
        -------
        DFMNowcaster
            Self, for method chaining.
        """
        # Convert dict to DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        self._data = data.copy()
        self._var_names = [v for v in self.frequency_map if v in data.columns]

        if not self._var_names:
            msg = "No variables from frequency_map found in data columns"
            raise ValueError(msg)

        self._classify_variables()

        n_obs = len(self._var_names)
        nf = self.n_factors
        state_dim = self._state_dim

        # Initialize parameters
        # Lambda via PCA on standardized data
        df_std = data[self._var_names].copy()
        means = df_std.mean()
        stds = df_std.std()
        stds = stds.replace(0, 1)
        df_norm = (df_std - means) / stds

        # Fill NaN with 0 for PCA initialization
        df_filled = df_norm.fillna(0)
        cov_mat = df_filled.T @ df_filled / len(df_filled)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat.values)

        # Take top n_factors eigenvectors (sorted descending)
        idx = np.argsort(eigenvalues)[::-1][:nf]
        self._Lambda = eigenvectors[:, idx].astype(np.float64)

        # Initialize A as small diagonal
        self._A = np.zeros((nf, nf * self.factor_lags))
        self._A[:, :nf] = 0.5 * np.eye(nf)

        # Initialize Q and R
        self._Q = np.eye(nf) * 0.1
        self._R = np.eye(n_obs) * 1.0

        # EM iterations
        prev_ll = -np.inf
        result: KalmanResult | None = None

        for iteration in range(self.em_iterations):
            # Build state-space
            y, Z, T_mat, Q_mat, R_mat, miss_mask = self._build_state_space(data)

            # Initial state
            a0 = np.zeros(state_dim)
            P0 = np.eye(state_dim) * 10.0

            # E-step: Kalman smoother
            result = self._adapter.smooth(
                y, Z, T_mat, Q_mat, R_mat, a0, P0, miss_mask
            )
            self._kalman_result = result

            # Check convergence
            ll = result.log_likelihood
            if abs(ll - prev_ll) < self.em_tol and iteration > 0:
                break
            prev_ll = ll

            # M-step: Update parameters
            if result.smoothed_state is None:
                break

            smoothed = result.smoothed_state
            smoothed_cov_all = result.smoothed_cov

            n_t = len(y)

            # Update Lambda (per-variable sums)
            sum_xf = np.zeros((n_obs, nf))
            sum_ff = np.zeros((n_obs, nf, nf))

            for t in range(n_t):
                f_t = smoothed[t, :nf]
                P_t = smoothed_cov_all[t, :nf, :nf]  # type: ignore[index]
                ff = np.outer(f_t, f_t) + P_t

                for i in range(n_obs):
                    if not miss_mask[t, i]:
                        sum_xf[i] += y[t, i] * f_t
                        sum_ff[i] += ff

            for i in range(n_obs):
                try:
                    sum_ff_inv = np.linalg.inv(sum_ff[i])
                except np.linalg.LinAlgError:
                    sum_ff_inv = np.linalg.pinv(sum_ff[i])
                self._Lambda[i] = sum_xf[i] @ sum_ff_inv  # type: ignore[index]

            self._Lambda = self._Lambda.astype(np.float64)  # type: ignore[union-attr]

            # Update R (diagonal)
            diag_r = np.zeros(n_obs)
            count_obs = np.zeros(n_obs)
            for t in range(n_t):
                f_t = smoothed[t, :nf]
                for i in range(n_obs):
                    if not miss_mask[t, i]:
                        resid = y[t, i] - self._Lambda[i] @ f_t
                        diag_r[i] += resid**2
                        count_obs[i] += 1

            for i in range(n_obs):
                if count_obs[i] > 0:
                    diag_r[i] /= count_obs[i]
                else:
                    diag_r[i] = 1.0
            diag_r = np.maximum(diag_r, 1e-6)
            self._R = np.diag(diag_r).astype(np.float64)

            # Update A (factor dynamics)
            sum_ff_lag = np.zeros((nf, nf * self.factor_lags))
            sum_ff_lag_lag = np.zeros(
                (nf * self.factor_lags, nf * self.factor_lags)
            )

            for t in range(self.factor_lags, n_t):
                f_t = smoothed[t, :nf]
                f_lags = np.concatenate(
                    [smoothed[t - k, :nf] for k in range(1, self.factor_lags + 1)]
                )
                sum_ff_lag += np.outer(f_t, f_lags)
                sum_ff_lag_lag += np.outer(f_lags, f_lags)

            try:
                self._A = (sum_ff_lag @ np.linalg.inv(sum_ff_lag_lag)).astype(
                    np.float64
                )
            except np.linalg.LinAlgError:
                self._A = (sum_ff_lag @ np.linalg.pinv(sum_ff_lag_lag)).astype(
                    np.float64
                )

            # Update Q
            sum_uu = np.zeros((nf, nf))
            for t in range(self.factor_lags, n_t):
                f_t = smoothed[t, :nf]
                f_lags = np.concatenate(
                    [smoothed[t - k, :nf] for k in range(1, self.factor_lags + 1)]
                )
                u = f_t - self._A @ f_lags
                sum_uu += np.outer(u, u)

            self._Q = (sum_uu / max(1, n_t - self.factor_lags)).astype(np.float64)
            self._Q = (self._Q + self._Q.T) / 2  # Symmetrize
            self._Q = np.maximum(self._Q, np.eye(nf) * 1e-6)

        # Store factors
        if result is not None and result.smoothed_state is not None:
            factor_cols = [f"factor_{i+1}" for i in range(nf)]
            self._factors_df = pd.DataFrame(
                result.smoothed_state[:, :nf],
                index=data.index,
                columns=factor_cols,
            )

        self._fitted = True
        return self

    def nowcast(
        self,
        target: str | None = None,
        reference_date: str | pd.Timestamp | None = None,
    ) -> Any:
        """Generate nowcast for the target variable.

        Parameters
        ----------
        target : str or None
            Target variable name. If None, uses the first quarterly variable.
        reference_date : str, Timestamp, or None
            Reference date for the nowcast. If None, uses latest available.

        Returns
        -------
        Forecast
            Nowcast as a Forecast object with point estimate and intervals.
        """
        if not self._fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        from forecastbox.core.forecast import Forecast

        if target is None:
            target = self._quarterly_vars[0] if self._quarterly_vars else self._var_names[0]

        if target not in self._var_names:
            msg = f"Target '{target}' not in model variables: {self._var_names}"
            raise ValueError(msg)

        # Get target index
        target_idx = self._var_names.index(target)

        # Use the last smoothed/filtered state to produce nowcast
        if self._kalman_result is None:
            msg = "No Kalman result available"
            raise RuntimeError(msg)

        if self._kalman_result.smoothed_state is not None:
            state = self._kalman_result.smoothed_state
            state_cov = self._kalman_result.smoothed_cov
        else:
            state = self._kalman_result.filtered_state
            state_cov = self._kalman_result.filtered_cov

        # Point nowcast from the loading and last state
        nf = self.n_factors
        last_state = state[-1, :nf]
        assert self._Lambda is not None
        point_est = float(self._Lambda[target_idx] @ last_state)

        # Uncertainty from state covariance
        lam = self._Lambda[target_idx]
        assert state_cov is not None
        assert self._R is not None
        var_est = float(
            lam @ state_cov[-1, :nf, :nf] @ lam + self._R[target_idx, target_idx]
        )
        std_est = np.sqrt(max(var_est, 1e-10))

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
            model_name=f"DFM({self.n_factors}f)-{target}",
            horizon=1,
            metadata={
                "target": target,
                "n_factors": self.n_factors,
                "n_variables": len(self._var_names),
            },
        )

    def update(self, new_data: pd.DataFrame | dict[str, pd.Series]) -> DFMNowcaster:
        """Update nowcast with new data via incremental Kalman filter.

        Parameters
        ----------
        new_data : pd.DataFrame or dict[str, pd.Series]
            New observations to incorporate.

        Returns
        -------
        DFMNowcaster
            Self, for method chaining.
        """
        if not self._fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        if isinstance(new_data, dict):
            new_data = pd.DataFrame(new_data)

        # Append new data
        if self._data is not None:
            combined = pd.concat([self._data, new_data])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            self._data = combined

        # Re-run Kalman filter with updated data
        assert self._data is not None
        y, Z, T_mat, Q_mat, R_mat, miss_mask = self._build_state_space(self._data)
        a0 = np.zeros(self._state_dim)
        P0 = np.eye(self._state_dim) * 10.0

        self._kalman_result = self._adapter.smooth(
            y, Z, T_mat, Q_mat, R_mat, a0, P0, miss_mask
        )

        # Update factors
        nf = self.n_factors
        if self._kalman_result.smoothed_state is not None:
            factor_cols = [f"factor_{i+1}" for i in range(nf)]
            self._factors_df = pd.DataFrame(
                self._kalman_result.smoothed_state[:, :nf],
                index=self._data.index,
                columns=factor_cols,
            )

        return self

    def factors(self) -> pd.DataFrame:
        """Return estimated latent factors.

        Returns
        -------
        pd.DataFrame
            DataFrame with factor columns (factor_1, factor_2, ...) and
            DatetimeIndex.
        """
        if not self._fitted or self._factors_df is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)
        return self._factors_df.copy()

    def loadings(self) -> pd.DataFrame:
        """Return estimated factor loadings.

        Returns
        -------
        pd.DataFrame
            DataFrame with shape (n_variables, n_factors). Row index is
            variable names, columns are factor_1, factor_2, etc.
        """
        if not self._fitted or self._Lambda is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        factor_cols = [f"factor_{i+1}" for i in range(self.n_factors)]
        return pd.DataFrame(
            self._Lambda,
            index=self._var_names,
            columns=factor_cols,
        )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"DFMNowcaster(n_factors={self.n_factors}, "
            f"factor_lags={self.factor_lags}, "
            f"n_vars={len(self.frequency_map)}, {status})"
        )
