"""Conditional forecasting via Waggoner-Zha (1999) algorithm.

References
----------
Waggoner, D.F. & Zha, T. (1999). "Conditional Forecasts in Dynamic
Multivariate Models." Review of Economics and Statistics, 81(4), 639-651.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from forecastbox.scenarios._protocols import VARModelProtocol


class ConditionalForecast:
    """Conditional forecast for VAR models using Waggoner-Zha (1999).

    Given a VAR(p) model and conditions on some variables' future paths,
    computes the optimal conditional forecast for all variables.

    The algorithm finds the minimum-variance forecast that satisfies
    the imposed conditions exactly, using the formula:

        y_cond = y_unc + sig_f @ sel.T @ inv(sel @ sig_f @ sel.T) @ (vals - sel @ y_unc)

    Parameters
    ----------
    model : VARModelProtocol
        Fitted VAR model with attributes: coef, intercept, sigma_u,
        k_vars, p_order, var_names, endog.
    method : str
        Forecasting method: 'analytic' (closed-form) or 'gibbs' (sampling).
        Default is 'analytic'.

    Examples
    --------
    >>> cf = ConditionalForecast(var_model)
    >>> results = cf.forecast(
    ...     steps=12,
    ...     conditions={'selic': [13.75, 13.25, 12.75, 12.25]},
    ...     n_draws=1000
    ... )
    >>> print(results['ipca'].point)
    """

    def __init__(
        self,
        model: VARModelProtocol,
        method: str = "analytic",
    ) -> None:
        if method not in ("analytic", "gibbs"):
            msg = f"method must be 'analytic' or 'gibbs', got '{method}'"
            raise ValueError(msg)

        self.model = model
        self.method = method
        self._k = model.k_vars
        self._p = model.p_order
        self._var_names = model.var_names

    def forecast(
        self,
        steps: int,
        conditions: dict[str, list[float] | NDArray[np.float64]] | None = None,
        n_draws: int = 1000,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate conditional forecast.

        Parameters
        ----------
        steps : int
            Number of forecast steps.
        conditions : dict[str, array-like] or None
            Dictionary mapping variable names to their imposed paths.
            Each path can be shorter than `steps` (conditions apply to
            the first len(path) periods only).
            If None, returns unconditional forecast.
        n_draws : int
            Number of Monte Carlo draws (used for intervals and gibbs method).
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        dict[str, Forecast]
            Dictionary mapping variable names to Forecast objects.
            Each Forecast contains point, lower_80, upper_80, lower_95, upper_95.
        """
        from forecastbox.core.forecast import Forecast

        rng = np.random.default_rng(seed)

        if conditions is None or len(conditions) == 0:
            # No conditions: return unconditional forecast
            y_unc, sig_f = self._unconditional_forecast(steps)

            results: dict[str, Any] = {}
            for i, name in enumerate(self._var_names):
                point = y_unc[i :: self._k]  # every k-th element starting from i

                # Generate draws from unconditional distribution
                draws = rng.multivariate_normal(
                    y_unc, sig_f, size=n_draws
                )  # (n_draws, k*steps)

                var_draws = draws[:, i :: self._k]  # (n_draws, steps)

                results[name] = Forecast(
                    point=point,
                    lower_80=np.quantile(var_draws, 0.10, axis=0),
                    upper_80=np.quantile(var_draws, 0.90, axis=0),
                    lower_95=np.quantile(var_draws, 0.025, axis=0),
                    upper_95=np.quantile(var_draws, 0.975, axis=0),
                    model_name=f"ConditionalForecast({self.method})",
                    horizon=steps,
                )

            return results

        # Validate conditions
        for var_name in conditions:
            if var_name not in self._var_names:
                msg = f"Unknown variable '{var_name}'. Available: {self._var_names}"
                raise ValueError(msg)

        # Convert conditions to arrays
        cond_arrays: dict[str, NDArray[np.float64]] = {}
        for var_name, path in conditions.items():
            cond_arrays[var_name] = np.asarray(path, dtype=np.float64)
            if len(cond_arrays[var_name]) > steps:
                msg = (
                    f"Condition path for '{var_name}' has length "
                    f"{len(cond_arrays[var_name])}, but steps={steps}"
                )
                raise ValueError(msg)

        if self.method == "gibbs":
            return self._gibbs_conditional(steps, cond_arrays, n_draws, rng)

        # Analytic method
        y_unc, sig_f = self._unconditional_forecast(steps)
        sel, vals = self._build_constraint_matrix(cond_arrays, steps)
        y_cond = self._apply_conditions(y_unc, sig_f, sel, vals)

        # Conditional covariance for intervals
        # sig_cond = sig_f - sig_f @ sel.T @ inv(sel @ sig_f @ sel.T) @ sel @ sig_f
        rsr = sel @ sig_f @ sel.T
        rsr_inv = np.linalg.solve(rsr, np.eye(rsr.shape[0]))
        sig_cond = sig_f - sig_f @ sel.T @ rsr_inv @ sel @ sig_f

        # Ensure positive semi-definite
        sig_cond = (sig_cond + sig_cond.T) / 2
        eigvals = np.linalg.eigvalsh(sig_cond)
        if np.any(eigvals < -1e-10):
            # Clip small negative eigenvalues
            eigvals_clip = np.maximum(eigvals, 0)
            eigvecs = np.linalg.eigh(sig_cond)[1]
            sig_cond = eigvecs @ np.diag(eigvals_clip) @ eigvecs.T

        # Generate draws from conditional distribution
        draws = rng.multivariate_normal(y_cond, sig_cond, size=n_draws)

        results = {}
        for i, name in enumerate(self._var_names):
            point = y_cond[i :: self._k]
            var_draws = draws[:, i :: self._k]  # (n_draws, steps)

            results[name] = Forecast(
                point=point,
                lower_80=np.quantile(var_draws, 0.10, axis=0),
                upper_80=np.quantile(var_draws, 0.90, axis=0),
                lower_95=np.quantile(var_draws, 0.025, axis=0),
                upper_95=np.quantile(var_draws, 0.975, axis=0),
                model_name="ConditionalForecast(analytic)",
                horizon=steps,
            )

        return results

    def _build_constraint_matrix(
        self,
        conditions: dict[str, NDArray[np.float64]],
        steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Build the constraint matrix sel and vector vals.

        The stacked forecast vector y_f has dimension (k*H, 1) ordered as:
        [y1_{T+1}, y2_{T+1}, ..., yk_{T+1}, y1_{T+2}, ..., yk_{T+H}]

        For each condition on variable i at horizon h, we add a row to sel
        that selects element (h*k + i) and the corresponding value to vals.

        Parameters
        ----------
        conditions : dict[str, ndarray]
            Variable name -> imposed path values.
        steps : int
            Total forecast horizon.

        Returns
        -------
        sel : ndarray (m, k*H)
            Selection matrix.
        vals : ndarray (m,)
            Imposed values vector.
        """
        k = self._k
        total_dim = k * steps

        rows: list[NDArray[np.float64]] = []
        values: list[float] = []

        for var_name, path in conditions.items():
            var_idx = self._var_names.index(var_name)
            for h, val in enumerate(path):
                row = np.zeros(total_dim)
                row[h * k + var_idx] = 1.0
                rows.append(row)
                values.append(float(val))

        sel = np.array(rows, dtype=np.float64)  # (m, k*H)
        vals = np.array(values, dtype=np.float64)  # (m,)

        return sel, vals

    def _unconditional_forecast(
        self,
        steps: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute unconditional forecast and forecast covariance.

        Parameters
        ----------
        steps : int
            Number of forecast steps.

        Returns
        -------
        y_unc : ndarray (k*H,)
            Stacked unconditional forecast vector.
        sig_f : ndarray (k*H, k*H)
            Forecast error covariance matrix.
        """
        k = self._k
        p = self._p
        coef = self.model.coef  # [A_1, ..., A_p], each (k, k)
        intercept = self.model.intercept  # (k,)
        sigma_u = self.model.sigma_u  # (k, k)
        endog = self.model.endog  # (T, k)

        # Last p observations for recursive forecasting
        last_obs = endog[-p:]  # (p, k)

        # Recursive unconditional forecast
        forecasts = np.zeros((steps, k))
        # Buffer of past values (most recent first)
        history = list(reversed([last_obs[i] for i in range(p)]))

        for h in range(steps):
            y_h = intercept.copy()
            for lag in range(min(p, h + p)):
                if lag < len(history):
                    y_h = y_h + coef[lag] @ history[lag]
            forecasts[h] = y_h
            # Prepend new forecast to history
            history.insert(0, y_h)

        # Stack into vector: [y_{T+1}, y_{T+2}, ..., y_{T+H}]
        y_unc = forecasts.flatten()  # (k*H,)

        # Compute impulse response matrices phi_j
        # phi_0 = I_k, phi_j = sum_{i=1}^{min(j,p)} A_i @ phi_{j-i}
        phi_mats: list[NDArray[np.float64]] = [np.eye(k)]  # phi_0
        for j in range(1, steps):
            phi_j = np.zeros((k, k))
            for i in range(1, min(j, p) + 1):
                phi_j = phi_j + coef[i - 1] @ phi_mats[j - i]
            phi_mats.append(phi_j)

        # Build forecast covariance
        # sig_f[h1, h2] = sum_{j=0}^{min(h1,h2)} phi_{h1-j} @ sigma @ phi_{h2-j}.T
        sig_f = np.zeros((k * steps, k * steps))
        for h1 in range(steps):
            for h2 in range(h1, steps):
                block = np.zeros((k, k))
                for j in range(h1 + 1):
                    block = block + phi_mats[h1 - j] @ sigma_u @ phi_mats[h2 - j].T
                sig_f[h1 * k : (h1 + 1) * k, h2 * k : (h2 + 1) * k] = block
                if h1 != h2:
                    sig_f[h2 * k : (h2 + 1) * k, h1 * k : (h1 + 1) * k] = block.T

        return y_unc, sig_f

    def _apply_conditions(
        self,
        y_unc: NDArray[np.float64],
        sig_f: NDArray[np.float64],
        sel: NDArray[np.float64],
        vals: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply Waggoner-Zha conditional projection.

        y_cond = y_unc + sig_f @ sel.T @ inv(sel @ sig_f @ sel.T) @ (vals - sel @ y_unc)

        Parameters
        ----------
        y_unc : ndarray (k*H,)
            Unconditional forecast vector.
        sig_f : ndarray (k*H, k*H)
            Forecast covariance matrix.
        sel : ndarray (m, k*H)
            Constraint selection matrix.
        vals : ndarray (m,)
            Imposed values.

        Returns
        -------
        y_cond : ndarray (k*H,)
            Conditional forecast vector satisfying sel @ y_cond = vals.
        """
        # sel @ sig_f @ sel.T  -> (m, m)
        rsr = sel @ sig_f @ sel.T

        # Solve for adjustment
        # adj = sig_f @ sel.T @ inv(rsr) @ (vals - sel @ y_unc)
        deviation = vals - sel @ y_unc  # (m,)
        rsr_inv_dev = np.linalg.solve(rsr, deviation)  # (m,)
        adjustment = sig_f @ sel.T @ rsr_inv_dev  # (k*H,)

        return y_unc + adjustment

    def _gibbs_conditional(
        self,
        steps: int,
        conditions: dict[str, NDArray[np.float64]],
        n_draws: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Conditional forecast via Gibbs sampling.

        For each draw:
        1. Perturb VAR parameters from approximate posterior
        2. Compute unconditional forecast with perturbed parameters
        3. Apply Waggoner-Zha conditions
        4. Add stochastic shock from conditional covariance

        Parameters
        ----------
        steps : int
            Number of forecast steps.
        conditions : dict[str, ndarray]
            Variable name -> imposed path.
        n_draws : int
            Number of Gibbs draws.
        rng : numpy Generator
            Random number generator.

        Returns
        -------
        dict[str, Forecast]
            Variable name -> Forecast with point and intervals from draws.
        """
        from forecastbox.core.forecast import Forecast

        k = self._k
        p = self._p
        coef = self.model.coef
        intercept = self.model.intercept
        sigma_u = self.model.sigma_u
        endog = self.model.endog
        residuals = self.model.residuals

        t_eff = residuals.shape[0]  # effective sample size

        # Pre-compute constraint matrix
        sel, vals = self._build_constraint_matrix(conditions, steps)

        # Storage for draws
        all_draws = np.zeros((n_draws, k * steps))

        for s in range(n_draws):
            # Step 1: Draw Sigma from inverse-Wishart posterior
            scale = rng.gamma(t_eff / 2, 2.0 / t_eff)
            sig_draw = sigma_u * scale

            # Ensure positive definite
            sig_draw = (sig_draw + sig_draw.T) / 2
            eigvals = np.linalg.eigvalsh(sig_draw)
            if np.any(eigvals <= 0):
                sig_draw = sigma_u  # fallback

            # Step 2: Draw coefficients from posterior
            coef_draw: list[NDArray[np.float64]] = []
            for lag_idx in range(p):
                noise = rng.normal(0, 0.01, size=(k, k)) * np.sqrt(
                    np.diag(sigma_u)
                ).reshape(-1, 1)
                a_draw = coef[lag_idx] + noise / np.sqrt(t_eff)
                coef_draw.append(a_draw)
            intercept_draw = intercept + rng.normal(0, 0.01, size=k) / np.sqrt(t_eff)

            # Step 3: Compute unconditional forecast with drawn parameters
            last_obs = endog[-p:]
            forecasts = np.zeros((steps, k))
            history = list(reversed([last_obs[i] for i in range(p)]))

            for h in range(steps):
                y_h = intercept_draw.copy()
                for lag in range(min(p, h + p)):
                    if lag < len(history):
                        y_h = y_h + coef_draw[lag] @ history[lag]
                forecasts[h] = y_h
                history.insert(0, y_h)

            y_unc = forecasts.flatten()

            # Compute forecast covariance with drawn Sigma
            phi_mats: list[NDArray[np.float64]] = [np.eye(k)]
            for j in range(1, steps):
                phi_j = np.zeros((k, k))
                for i in range(1, min(j, p) + 1):
                    phi_j = phi_j + coef_draw[i - 1] @ phi_mats[j - i]
                phi_mats.append(phi_j)

            sig_f = np.zeros((k * steps, k * steps))
            for h1 in range(steps):
                for h2 in range(h1, steps):
                    block = np.zeros((k, k))
                    for j in range(h1 + 1):
                        block = block + phi_mats[h1 - j] @ sig_draw @ phi_mats[h2 - j].T
                    sig_f[h1 * k : (h1 + 1) * k, h2 * k : (h2 + 1) * k] = block
                    if h1 != h2:
                        sig_f[h2 * k : (h2 + 1) * k, h1 * k : (h1 + 1) * k] = block.T

            # Step 4: Apply conditions
            y_cond = self._apply_conditions(y_unc, sig_f, sel, vals)

            # Step 5: Add stochastic perturbation from conditional covariance
            rsr = sel @ sig_f @ sel.T
            rsr_inv = np.linalg.solve(rsr, np.eye(rsr.shape[0]))
            sig_cond = sig_f - sig_f @ sel.T @ rsr_inv @ sel @ sig_f
            sig_cond = (sig_cond + sig_cond.T) / 2

            # Clip negative eigenvalues for numerical stability
            eigvals_c, eigvecs_c = np.linalg.eigh(sig_cond)
            eigvals_c = np.maximum(eigvals_c, 0)
            sig_cond = eigvecs_c @ np.diag(eigvals_c) @ eigvecs_c.T

            try:
                perturbation = rng.multivariate_normal(
                    np.zeros(k * steps), sig_cond
                )
                y_draw = y_cond + perturbation
            except np.linalg.LinAlgError:
                y_draw = y_cond

            # Enforce hard constraints on conditioned variables
            for var_name, path in conditions.items():
                var_idx = self._var_names.index(var_name)
                for h_idx, val in enumerate(path):
                    y_draw[h_idx * k + var_idx] = val

            all_draws[s] = y_draw

        # Build Forecast objects from draws
        results: dict[str, Any] = {}
        for i, name in enumerate(self._var_names):
            var_draws = all_draws[:, i::k]  # (n_draws, steps)
            point = np.mean(var_draws, axis=0)

            results[name] = Forecast(
                point=point,
                lower_80=np.quantile(var_draws, 0.10, axis=0),
                upper_80=np.quantile(var_draws, 0.90, axis=0),
                lower_95=np.quantile(var_draws, 0.025, axis=0),
                upper_95=np.quantile(var_draws, 0.975, axis=0),
                model_name="ConditionalForecast(gibbs)",
                horizon=steps,
            )

        return results
