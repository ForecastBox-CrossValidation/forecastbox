"""Counterfactual analysis: 'what if X had been different?'

Compares the observed trajectory with an alternative trajectory
where one or more variables followed a different path.

References
----------
Kilian, L. & Lutkepohl, H. (2017). Structural Vector Autoregressive Analysis.
Cambridge University Press. Chapter 4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from forecastbox.scenarios._protocols import VARModelProtocol


class CounterfactualResult:
    """Container for counterfactual analysis results.

    Attributes
    ----------
    actual : ndarray
        Observed trajectory of the target variable.
    counterfactual : ndarray
        Counterfactual trajectory of the target variable.
    diff : ndarray
        Difference: actual - counterfactual.
    cumulative_diff : ndarray
        Cumulative difference.
    periods : ndarray or DatetimeIndex
        Time periods analyzed.
    target : str
        Target variable name.
    altered_variables : list[str]
        Variables whose paths were altered.
    """

    def __init__(
        self,
        actual: NDArray[np.float64],
        counterfactual: NDArray[np.float64],
        periods: Any,
        target: str,
        altered_variables: list[str],
    ) -> None:
        self.actual = np.asarray(actual, dtype=np.float64)
        self.counterfactual = np.asarray(counterfactual, dtype=np.float64)
        self.diff = self.actual - self.counterfactual
        self.cumulative_diff = np.cumsum(self.diff)
        self.periods = periods
        self.target = target
        self.altered_variables = altered_variables

    def plot(
        self,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot actual vs counterfactual trajectories.

        Parameters
        ----------
        ax : matplotlib Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax_main = axes[0]
            ax_diff = axes[1]
        else:
            ax_main = ax
            ax_diff = None

        x = self.periods if self.periods is not None else np.arange(len(self.actual))

        # Main plot: actual vs counterfactual
        ax_main.plot(x, self.actual, "b-o", label="Actual", linewidth=2, markersize=4)
        ax_main.plot(
            x,
            self.counterfactual,
            "r--s",
            label="Counterfactual",
            linewidth=2,
            markersize=4,
        )
        ax_main.set_ylabel(self.target)
        ax_main.set_title(title or f"Counterfactual Analysis: {self.target}")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # Difference plot
        if ax_diff is not None:
            ax_diff.bar(x, self.diff, color="gray", alpha=0.6, label="Difference")
            ax_diff.axhline(y=0, color="black", linewidth=0.8)
            ax_diff.set_ylabel("Difference")
            ax_diff.set_xlabel("Period")
            ax_diff.legend()
            ax_diff.grid(True, alpha=0.3)

        return ax_main

    def summary(self) -> str:
        """Generate summary of counterfactual analysis.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("COUNTERFACTUAL ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"Target variable: {self.target}")
        lines.append(f"Altered variables: {', '.join(self.altered_variables)}")
        lines.append(f"Periods analyzed: {len(self.actual)}")
        lines.append(f"Mean difference: {np.mean(self.diff):+.4f}")
        lines.append(f"Max difference: {np.max(np.abs(self.diff)):+.4f}")
        lines.append(f"Cumulative difference: {self.cumulative_diff[-1]:+.4f}")
        return "\n".join(lines)


class Counterfactual:
    """Counterfactual analysis engine.

    Compares the observed outcome with a hypothetical scenario where
    one or more variables followed a different path, while keeping
    all other structural shocks constant (ceteris paribus).

    Parameters
    ----------
    model : VARModelProtocol
        Fitted VAR model on historical data.
    history : DataFrame or ndarray
        Historical data (T x k). If DataFrame, columns are variable names.

    Examples
    --------
    >>> cf = Counterfactual(model, history=data)
    >>> result = cf.run(
    ...     actual_path={'selic': selic_actual},
    ...     counter_path={'selic': selic_counter},
    ...     target='ipca',
    ...     periods=slice('2023-01', '2023-12')
    ... )
    >>> print(f"Difference: {result.diff.mean():.2f} p.p.")
    """

    def __init__(
        self,
        model: VARModelProtocol,
        history: pd.DataFrame | NDArray[np.float64],
    ) -> None:
        self.model = model

        if isinstance(history, pd.DataFrame):
            self._history_df = history
            self._history = history.values.astype(np.float64)
            self._history_index = history.index
        else:
            self._history_df = None
            self._history = np.asarray(history, dtype=np.float64)
            self._history_index = None

    def run(
        self,
        actual_path: dict[str, NDArray[np.float64] | list[float]],
        counter_path: dict[str, NDArray[np.float64] | list[float]],
        target: str,
        periods: slice | None = None,
    ) -> CounterfactualResult:
        """Run counterfactual analysis.

        Parameters
        ----------
        actual_path : dict[str, array-like]
            Actual observed paths for altered variables.
        counter_path : dict[str, array-like]
            Counterfactual (hypothetical) paths for altered variables.
        target : str
            Target variable to analyze.
        periods : slice or None
            Time period slice (for DataFrame with DatetimeIndex).
            If None, uses full history.

        Returns
        -------
        CounterfactualResult
            Results with actual, counterfactual, and differences.

        Raises
        ------
        ValueError
            If variable names are invalid.
        """
        var_names = self.model.var_names

        # Validate target
        if target not in var_names:
            msg = f"Unknown target '{target}'. Available: {var_names}"
            raise ValueError(msg)
        target_idx = var_names.index(target)

        # Validate altered variables
        altered_vars: list[str] = list(counter_path.keys())
        for var_name in altered_vars:
            if var_name not in var_names:
                msg = f"Unknown variable '{var_name}'. Available: {var_names}"
                raise ValueError(msg)

        # Resolve periods
        if periods is not None and self._history_df is not None:
            selected = self._history_df.loc[periods]
            analysis_data = selected.values.astype(np.float64)
            analysis_index = selected.index

            # Find starting position in full history
            full_index = self._history_df.index
            start_pos = full_index.get_loc(selected.index[0])
            if isinstance(start_pos, slice):
                start_pos = start_pos.start
        else:
            # Use the length of counter_path
            n_periods = len(next(iter(counter_path.values())))
            analysis_data = self._history[-n_periods:]
            analysis_index = np.arange(n_periods)
            start_pos = len(self._history) - n_periods

        n_periods = len(analysis_data)

        # Step 1: Recover historical residuals
        residuals = self._decompose_shocks()

        # Step 2: Simulate counterfactual trajectory
        counterfactual_target = self._simulate_counterfactual(
            counter_path=counter_path,
            residuals=residuals,
            start_pos=start_pos,
            n_periods=n_periods,
            target_idx=target_idx,
        )

        # Step 3: Extract actual target values
        actual_target = analysis_data[:, target_idx]

        return CounterfactualResult(
            actual=actual_target,
            counterfactual=counterfactual_target,
            periods=analysis_index,
            target=target,
            altered_variables=altered_vars,
        )

    def _decompose_shocks(self) -> NDArray[np.float64]:
        """Recover structural shocks (residuals) from history.

        u_t = y_t - c - A_1 @ y_{t-1} - ... - A_p @ y_{t-p}

        Returns
        -------
        residuals : ndarray (T-p, k)
            Recovered residuals.
        """
        k = self.model.k_vars
        p = self.model.p_order
        coef = self.model.coef
        intercept = self.model.intercept
        history = self._history

        t_total = len(history)
        n_residuals = t_total - p

        residuals = np.zeros((n_residuals, k))

        for t in range(p, t_total):
            y_predicted = intercept.copy()
            for lag in range(p):
                y_predicted = y_predicted + coef[lag] @ history[t - lag - 1]
            residuals[t - p] = history[t] - y_predicted

        return residuals

    def _simulate_counterfactual(
        self,
        counter_path: dict[str, NDArray[np.float64] | list[float]],
        residuals: NDArray[np.float64],
        start_pos: int,
        n_periods: int,
        target_idx: int,
    ) -> NDArray[np.float64]:
        """Simulate counterfactual trajectory.

        For each period:
        1. Use counterfactual path for altered variables.
        2. Keep same structural shocks (residuals).
        3. Propagate through the VAR model.

        Parameters
        ----------
        counter_path : dict[str, array-like]
            Counterfactual paths for altered variables.
        residuals : ndarray (T-p, k)
            Historical residuals.
        start_pos : int
            Starting position in full history.
        n_periods : int
            Number of periods to simulate.
        target_idx : int
            Index of the target variable.

        Returns
        -------
        ndarray (n_periods,)
            Counterfactual trajectory for the target variable.
        """
        var_names = self.model.var_names
        k = self.model.k_vars
        p = self.model.p_order
        coef = self.model.coef
        intercept = self.model.intercept
        history = self._history

        # Convert counter paths to arrays
        counter_arrays: dict[int, NDArray[np.float64]] = {}
        for var_name, path in counter_path.items():
            var_idx = var_names.index(var_name)
            counter_arrays[var_idx] = np.asarray(path, dtype=np.float64)

        # Initialize with actual history before the analysis period
        # We need p periods before start_pos for lag initialization
        cf_data = history[: start_pos + n_periods].copy()

        # Simulate counterfactual
        for t_rel in range(n_periods):
            t_abs = start_pos + t_rel
            residual_idx = t_abs - p

            if residual_idx < 0 or residual_idx >= len(residuals):
                u_t = np.zeros(k)
            else:
                u_t = residuals[residual_idx]

            # Compute prediction using counterfactual history
            y_cf = intercept.copy() + u_t
            for lag in range(p):
                t_lag = t_abs - lag - 1
                if t_lag >= 0:
                    y_cf = y_cf + coef[lag] @ cf_data[t_lag]

            # Override altered variables with counterfactual path
            for var_idx, path in counter_arrays.items():
                if t_rel < len(path):
                    y_cf[var_idx] = path[t_rel]

            cf_data[t_abs] = y_cf

        # Extract target variable from counterfactual
        return cf_data[start_pos : start_pos + n_periods, target_idx]
