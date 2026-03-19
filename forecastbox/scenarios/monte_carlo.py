"""Monte Carlo simulation for forecast uncertainty quantification.

Generates N future trajectories from a fitted model by sampling
from the estimated error distribution (parametric or bootstrap).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable


class MonteCarlo:
    """Monte Carlo simulation engine for forecast models.

    Generates N stochastic future trajectories by sampling from the
    estimated error distribution, either parametrically (Gaussian) or
    via bootstrap (residual resampling).

    Parameters
    ----------
    model : VARModelProtocol or UnivariateForecastModelProtocol
        Fitted forecast model. Must provide either VAR interface
        (coef, intercept, sigma_u, k_vars, p_order, endog, residuals)
        or univariate interface (sigma2, residuals, forecast, simulate).
    n_paths : int
        Number of simulation trajectories. Default 5000.
    seed : int or None
        Random seed for reproducibility.
    parametric : bool
        If True, draw errors from N(0, Sigma). If False, bootstrap
        residuals with replacement. Default True.

    Examples
    --------
    >>> mc = MonteCarlo(var_model, n_paths=5000, seed=42)
    >>> paths = mc.simulate(steps=24)
    >>> print(paths.shape)  # (5000, 24, 3) for 3-variable VAR
    >>> fan = mc.fan_chart()
    >>> fan.plot()
    """

    def __init__(
        self,
        model: Any,
        n_paths: int = 5000,
        seed: int | None = None,
        parametric: bool = True,
    ) -> None:
        self.model = model
        self.n_paths = n_paths
        self.seed = seed
        self.parametric = parametric
        self._rng = np.random.default_rng(seed)
        self._paths: NDArray[np.float64] | None = None
        self._is_var = hasattr(model, "coef") and hasattr(model, "sigma_u")

    def simulate(self, steps: int) -> NDArray[np.float64]:
        """Generate N stochastic future trajectories.

        Parameters
        ----------
        steps : int
            Number of forecast steps ahead.

        Returns
        -------
        paths : ndarray
            For VAR: shape (n_paths, steps, k_vars).
            For univariate: shape (n_paths, steps).
        """
        if self._is_var:
            self._paths = self._simulate_var(steps)
        else:
            self._paths = self._simulate_univariate(steps)

        return self._paths

    def _simulate_var(self, steps: int) -> NDArray[np.float64]:
        """Simulate VAR trajectories.

        Algorithm:
        For each path s = 1..N, for each step h = 1..H:
            u_{T+h}^(s) ~ N(0, Sigma_u) or bootstrap from residuals
            y_{T+h}^(s) = c + A_1 @ y_{T+h-1}^(s) + ... + A_p @ y_{T+h-p}^(s) + u_{T+h}^(s)

        Parameters
        ----------
        steps : int
            Number of forecast steps.

        Returns
        -------
        paths : ndarray (n_paths, steps, k_vars)
        """
        model = self.model
        k = model.k_vars
        p = model.p_order
        coef = model.coef  # [A_1, ..., A_p]
        intercept = model.intercept
        sigma_u = model.sigma_u
        endog = model.endog
        residuals = model.residuals

        n_paths = self.n_paths
        rng = self._rng

        # Last p observations for initialization
        last_obs = endog[-p:]  # (p, k)

        paths = np.zeros((n_paths, steps, k))

        for s in range(n_paths):
            # Initialize history buffer (most recent first)
            history = [last_obs[-(i + 1)].copy() for i in range(p)]

            for h in range(steps):
                # Generate error
                if self.parametric:
                    u = rng.multivariate_normal(np.zeros(k), sigma_u)
                else:
                    # Bootstrap: sample a random residual
                    idx = rng.integers(0, len(residuals))
                    u = residuals[idx].copy()

                # Compute forecast
                y_h = intercept.copy() + u
                for lag in range(min(p, len(history))):
                    y_h = y_h + coef[lag] @ history[lag]

                paths[s, h] = y_h

                # Update history
                history.insert(0, y_h.copy())
                if len(history) > p:
                    history.pop()

        return paths

    def _simulate_univariate(self, steps: int) -> NDArray[np.float64]:
        """Simulate univariate model trajectories.

        Uses the model's simulate method if available, otherwise
        generates errors and propagates through forecast recursion.

        Parameters
        ----------
        steps : int
            Number of forecast steps.

        Returns
        -------
        paths : ndarray (n_paths, steps)
        """
        model = self.model

        # If model has its own simulate, use it
        if hasattr(model, "simulate") and callable(model.simulate):
            paths = model.simulate(
                steps=steps,
                n_paths=self.n_paths,
                seed=self.seed,
            )
            return np.asarray(paths, dtype=np.float64)

        # Fallback: generate point forecast + errors
        point = model.forecast(steps)
        sigma2 = model.sigma2
        residuals = model.residuals
        rng = self._rng

        n_paths = self.n_paths
        paths = np.zeros((n_paths, steps))

        for s in range(n_paths):
            if self.parametric:
                errors = rng.normal(0, np.sqrt(sigma2), size=steps)
            else:
                indices = rng.integers(0, len(residuals), size=steps)
                errors = residuals[indices]

            # Cumulative error propagation (random walk around point forecast)
            paths[s] = point + np.cumsum(errors) / np.sqrt(np.arange(1, steps + 1))

        return paths

    def fan_chart(
        self,
        quantiles: list[float] | None = None,
        variable: int | str | None = None,
    ) -> Any:
        """Create FanChart from simulated paths.

        Parameters
        ----------
        quantiles : list[float] or None
            Quantile levels. Default: [0.10, ..., 0.90].
        variable : int, str, or None
            For VAR models, which variable to use (index or name).
            None means first variable.

        Returns
        -------
        FanChart
            FanChart object from the simulated paths.

        Raises
        ------
        RuntimeError
            If simulate() has not been called.
        """
        if self._paths is None:
            msg = "Must call simulate() before fan_chart()"
            raise RuntimeError(msg)

        from forecastbox.scenarios.fan_chart import FanChart

        if self._is_var:
            var_idx = self._resolve_variable_index(variable)
            draws = self._paths[:, :, var_idx]  # (n_paths, steps)
        else:
            draws = self._paths  # (n_paths, steps)

        # draws is (n_paths, steps), FanChart.from_ensemble expects (n_draws, H)
        return FanChart.from_ensemble(draws)

    def statistics(
        self,
        variable: int | str | None = None,
    ) -> pd.DataFrame:
        """Compute summary statistics by horizon.

        Parameters
        ----------
        variable : int, str, or None
            For VAR models, which variable (index or name).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: mean, median, std, q5, q25, q75, q95
            indexed by horizon (1..H).

        Raises
        ------
        RuntimeError
            If simulate() has not been called.
        """
        if self._paths is None:
            msg = "Must call simulate() before statistics()"
            raise RuntimeError(msg)

        if self._is_var:
            var_idx = self._resolve_variable_index(variable)
            data = self._paths[:, :, var_idx]  # (n_paths, steps)
        else:
            data = self._paths  # (n_paths, steps)

        steps = data.shape[1]
        stats: dict[str, NDArray[np.float64]] = {
            "mean": np.mean(data, axis=0),
            "median": np.median(data, axis=0),
            "std": np.std(data, axis=0, ddof=1),
            "q5": np.quantile(data, 0.05, axis=0),
            "q25": np.quantile(data, 0.25, axis=0),
            "q75": np.quantile(data, 0.75, axis=0),
            "q95": np.quantile(data, 0.95, axis=0),
        }

        return pd.DataFrame(stats, index=range(1, steps + 1))

    def probability(
        self,
        condition_fn: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
        variable: int | str | None = None,
    ) -> NDArray[np.float64]:
        """Compute probability of a condition by horizon.

        P(condition | h) = fraction of paths satisfying condition at horizon h.

        Parameters
        ----------
        condition_fn : callable
            Function that takes array of values and returns boolean array.
            Example: lambda y: y > 5.0
        variable : int, str, or None
            For VAR models, which variable.

        Returns
        -------
        probs : ndarray (steps,)
            Probability at each horizon. 0 <= probs[h] <= 1.

        Raises
        ------
        RuntimeError
            If simulate() has not been called.

        Examples
        --------
        >>> prob_above_5 = mc.probability(lambda y: y > 5.0)
        >>> print(prob_above_5)  # [0.02, 0.05, 0.08, ...]
        """
        if self._paths is None:
            msg = "Must call simulate() before probability()"
            raise RuntimeError(msg)

        if self._is_var:
            var_idx = self._resolve_variable_index(variable)
            data = self._paths[:, :, var_idx]
        else:
            data = self._paths

        # Apply condition to all paths at each horizon
        mask = condition_fn(data)  # (n_paths, steps) boolean
        probs = np.mean(mask.astype(np.float64), axis=0)  # (steps,)

        return probs

    def expected_shortfall(
        self,
        threshold: float,
        variable: int | str | None = None,
        alpha: float = 0.05,
    ) -> NDArray[np.float64]:
        """Compute Expected Shortfall (CVaR) by horizon.

        ES_alpha(h) = E[y_{T+h} | y_{T+h} < q_alpha(h)]

        This is the expected value of the variable given that it falls
        below the alpha-quantile. Also known as Conditional Value at Risk.

        Parameters
        ----------
        threshold : float
            Threshold value (not used if alpha is specified).
            If provided, computes E[y | y < threshold] instead.
        variable : int, str, or None
            For VAR models, which variable.
        alpha : float
            Quantile level for ES. Default 0.05 (5th percentile).

        Returns
        -------
        es : ndarray (steps,)
            Expected shortfall at each horizon.

        Raises
        ------
        RuntimeError
            If simulate() has not been called.
        """
        if self._paths is None:
            msg = "Must call simulate() before expected_shortfall()"
            raise RuntimeError(msg)

        if self._is_var:
            var_idx = self._resolve_variable_index(variable)
            data = self._paths[:, :, var_idx]
        else:
            data = self._paths

        steps = data.shape[1]
        es = np.zeros(steps)

        for h in range(steps):
            values = data[:, h]
            q = np.quantile(values, alpha)
            mask = values <= q
            if np.any(mask):
                es[h] = np.mean(values[mask])
            else:
                es[h] = q

        return es

    def _resolve_variable_index(self, variable: int | str | None) -> int:
        """Resolve variable specification to integer index.

        Parameters
        ----------
        variable : int, str, or None
            Variable index, name, or None (first variable).

        Returns
        -------
        int
            Variable index.
        """
        if variable is None:
            return 0

        if isinstance(variable, int):
            return variable

        if isinstance(variable, str) and hasattr(self.model, "var_names"):
            var_names = self.model.var_names
            if variable in var_names:
                return var_names.index(variable)
            msg = f"Unknown variable '{variable}'. Available: {var_names}"
            raise ValueError(msg)

        msg = f"Cannot resolve variable: {variable}"
        raise ValueError(msg)
