"""Stress testing: analyze impact of extreme shocks on forecasts.

Simulates the propagation of exogenous shocks through a VAR model
using impulse response functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from forecastbox.scenarios._protocols import VARModelProtocol


@dataclass
class Shock:
    """Specification of an exogenous shock.

    Attributes
    ----------
    variable : str
        Variable to shock.
    magnitude : float
        Shock magnitude (interpretation depends on shock_type).
    shock_type : str
        Type of shock: 'absolute', 'std_dev', or 'percent'.
    period : int
        Period when shock hits (1-based, 1 = first forecast step).
    duration : int
        Duration of the shock in periods. 1 = temporary (single period).
    decay : float
        Exponential decay rate per period. 0 = no decay (constant within duration).
        0.5 = 50% decay per period.
    """

    variable: str
    magnitude: float
    shock_type: str = "std_dev"
    period: int = 1
    duration: int = 1
    decay: float = 0.0

    def __post_init__(self) -> None:
        if self.shock_type not in ("absolute", "std_dev", "percent"):
            msg = f"shock_type must be 'absolute', 'std_dev', or 'percent', got '{self.shock_type}'"
            raise ValueError(msg)
        if self.period < 1:
            msg = f"period must be >= 1, got {self.period}"
            raise ValueError(msg)
        if self.duration < 1:
            msg = f"duration must be >= 1, got {self.duration}"
            raise ValueError(msg)
        if not 0 <= self.decay <= 1:
            msg = f"decay must be in [0, 1], got {self.decay}"
            raise ValueError(msg)


class StressResult:
    """Container for stress test results.

    Attributes
    ----------
    baseline : dict[str, Forecast]
        Baseline forecasts (no shocks).
    stressed : dict[str, Forecast]
        Stressed forecasts (with shocks applied).
    impact : dict[str, ndarray]
        Impact by variable: stressed - baseline point forecasts.
    shocks : list[Shock]
        Applied shocks.
    var_names : list[str]
        Variable names.
    steps : int
        Forecast horizon.
    """

    def __init__(
        self,
        baseline: dict[str, Any],
        stressed: dict[str, Any],
        shocks: list[Shock],
        var_names: list[str],
        steps: int,
    ) -> None:
        self.baseline = baseline
        self.stressed = stressed
        self.shocks = shocks
        self.var_names = var_names
        self.steps = steps

        # Compute impact
        self.impact: dict[str, NDArray[np.float64]] = {}
        for name in var_names:
            self.impact[name] = stressed[name].point - baseline[name].point

    def max_impact(self, variable: str) -> tuple[float, int]:
        """Find maximum absolute impact and its horizon.

        Parameters
        ----------
        variable : str
            Variable name.

        Returns
        -------
        tuple[float, int]
            (max_impact_value, horizon_index) where horizon is 0-based.
        """
        if variable not in self.impact:
            msg = f"Unknown variable '{variable}'. Available: {list(self.impact.keys())}"
            raise KeyError(msg)

        impacts = self.impact[variable]
        max_idx = int(np.argmax(np.abs(impacts)))
        return float(impacts[max_idx]), max_idx

    def plot_impact(
        self,
        variable: str,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot the impact (stressed - baseline) for a variable.

        Parameters
        ----------
        variable : str
            Variable to plot.
        ax : matplotlib Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        if variable not in self.impact:
            msg = f"Unknown variable '{variable}'. Available: {list(self.impact.keys())}"
            raise KeyError(msg)

        x = np.arange(1, self.steps + 1)
        impacts = self.impact[variable]

        ax.bar(x, impacts, color="red", alpha=0.6, label="Impact")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xlabel("Horizon")
        ax.set_ylabel(f"Impact on {variable}")
        ax.set_title(title or f"Stress Test Impact: {variable}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_comparison(
        self,
        variable: str,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot baseline vs stressed forecasts.

        Parameters
        ----------
        variable : str
            Variable to plot.
        ax : matplotlib Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(1, self.steps + 1)
        base = self.baseline[variable].point
        stress = self.stressed[variable].point

        ax.plot(x, base, "b-o", label="Baseline", linewidth=2, markersize=4)
        ax.plot(x, stress, "r-s", label="Stressed", linewidth=2, markersize=4)

        # Show intervals for stressed if available
        stressed_fc = self.stressed[variable]
        if stressed_fc.lower_95 is not None and stressed_fc.upper_95 is not None:
            ax.fill_between(
                x,
                stressed_fc.lower_95,
                stressed_fc.upper_95,
                alpha=0.1,
                color="red",
            )

        ax.set_xlabel("Horizon")
        ax.set_ylabel(variable)
        ax.set_title(title or f"Stress Test: {variable}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def summary(self) -> str:
        """Generate summary of stress test results.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("STRESS TEST RESULTS")
        lines.append("=" * 60)

        lines.append("\nApplied Shocks:")
        for shock in self.shocks:
            lines.append(
                f"  - {shock.variable}: {shock.magnitude:+.2f} "
                f"({shock.shock_type}) at period {shock.period}, "
                f"duration={shock.duration}, decay={shock.decay}"
            )

        lines.append("\nMax Impact by Variable:")
        for name in self.var_names:
            max_val, max_h = self.max_impact(name)
            lines.append(f"  {name}: {max_val:+.4f} at horizon {max_h + 1}")

        return "\n".join(lines)


class StressTest:
    """Stress test engine for VAR models.

    Analyzes the impact of exogenous shocks by comparing baseline
    forecasts with shocked forecasts.

    Parameters
    ----------
    model : VARModelProtocol
        Fitted VAR model.

    Examples
    --------
    >>> stress = StressTest(model)
    >>> stress.add_shock('cambio', magnitude=2.0, shock_type='std_dev', period=1)
    >>> stress.add_shock('selic', magnitude=3.0, shock_type='absolute', period=1)
    >>> result = stress.run(steps=12)
    >>> result.plot_impact('ipca')
    """

    def __init__(self, model: VARModelProtocol) -> None:
        self.model = model
        self._shocks: list[Shock] = []
        self._baseline: dict[str, Any] | None = None

    def add_shock(
        self,
        variable: str,
        magnitude: float,
        shock_type: str = "std_dev",
        period: int = 1,
        duration: int = 1,
        decay: float = 0.0,
    ) -> None:
        """Add a shock to the stress test.

        Parameters
        ----------
        variable : str
            Variable to shock.
        magnitude : float
            Shock magnitude.
        shock_type : str
            'absolute', 'std_dev', or 'percent'.
        period : int
            Period when shock hits (1-based).
        duration : int
            Duration in periods.
        decay : float
            Exponential decay rate.

        Raises
        ------
        ValueError
            If variable name not recognized.
        """
        if variable not in self.model.var_names:
            msg = f"Unknown variable '{variable}'. Available: {self.model.var_names}"
            raise ValueError(msg)

        self._shocks.append(
            Shock(
                variable=variable,
                magnitude=magnitude,
                shock_type=shock_type,
                period=period,
                duration=duration,
                decay=decay,
            )
        )

    def clear_shocks(self) -> None:
        """Remove all registered shocks."""
        self._shocks.clear()

    def run(
        self,
        steps: int,
        n_draws: int = 1000,
        seed: int | None = None,
    ) -> StressResult:
        """Execute stress test.

        1. Compute baseline forecast (no shocks).
        2. Convert shocks to absolute values.
        3. Compute impulse response functions.
        4. Propagate shocks through the model.
        5. Compute stressed forecast = baseline + accumulated impact.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        n_draws : int
            Number of draws for intervals.
        seed : int or None
            Random seed.

        Returns
        -------
        StressResult
            Results with baseline, stressed, and impact.
        """
        from forecastbox.core.forecast import Forecast
        from forecastbox.scenarios.conditional import ConditionalForecast

        k = self.model.k_vars
        sigma_u = self.model.sigma_u
        var_names = self.model.var_names

        # Step 1: Compute baseline forecast
        cf = ConditionalForecast(self.model, method="analytic")
        baseline = cf.forecast(steps=steps, conditions=None, n_draws=n_draws, seed=seed)

        if not self._shocks:
            # No shocks: stressed = baseline
            return StressResult(
                baseline=baseline,
                stressed=baseline,
                shocks=[],
                var_names=var_names,
                steps=steps,
            )

        # Step 2: Compute impulse response functions
        irf_matrices = self._compute_irf(steps)

        # Step 3: Build shock magnitudes (absolute values)
        shock_magnitudes = self._build_shock_matrix(steps, baseline, sigma_u)

        # Step 4: Compute total impact via IRF
        total_impact = np.zeros((steps, k))

        for shock in self._shocks:
            var_idx = var_names.index(shock.variable)
            abs_magnitude = shock_magnitudes[shock.variable]

            for h in range(steps):
                for d in range(shock.duration):
                    shock_period = shock.period - 1 + d  # 0-based
                    if shock_period > h:
                        continue

                    lag = h - shock_period
                    if lag >= len(irf_matrices):
                        continue

                    decay_factor = (1 - shock.decay) ** d
                    effective_magnitude = abs_magnitude * decay_factor

                    # Impulse response: Phi[lag][:, var_idx] * effective_magnitude
                    total_impact[h, :] += (
                        irf_matrices[lag][:, var_idx] * effective_magnitude
                    )

        # Step 5: Build stressed forecasts
        stressed: dict[str, Any] = {}

        for i, name in enumerate(var_names):
            base_point = baseline[name].point
            stressed_point = base_point + total_impact[:, i]

            # Generate intervals around stressed point
            base_fc = baseline[name]
            if base_fc.upper_95 is not None and base_fc.lower_95 is not None:
                base_width_95 = base_fc.upper_95 - base_fc.lower_95
                base_width_80 = (
                    (base_fc.upper_80 - base_fc.lower_80)
                    if base_fc.upper_80 is not None and base_fc.lower_80 is not None
                    else base_width_95 * 0.6
                )

                stressed[name] = Forecast(
                    point=stressed_point,
                    lower_80=stressed_point - base_width_80 / 2,
                    upper_80=stressed_point + base_width_80 / 2,
                    lower_95=stressed_point - base_width_95 / 2,
                    upper_95=stressed_point + base_width_95 / 2,
                    model_name="StressTest",
                    horizon=steps,
                )
            else:
                stressed[name] = Forecast(
                    point=stressed_point,
                    model_name="StressTest",
                    horizon=steps,
                )

        return StressResult(
            baseline=baseline,
            stressed=stressed,
            shocks=list(self._shocks),
            var_names=var_names,
            steps=steps,
        )

    def _compute_irf(self, steps: int) -> list[NDArray[np.float64]]:
        """Compute impulse response function matrices.

        Phi_0 = I_k
        Phi_h = sum_{i=1}^{min(h,p)} A_i @ Phi_{h-i}

        Parameters
        ----------
        steps : int
            Number of horizons.

        Returns
        -------
        list[ndarray]
            [Phi_0, Phi_1, ..., Phi_{steps-1}], each (k, k).
        """
        k = self.model.k_vars
        p = self.model.p_order
        coef = self.model.coef

        irf: list[NDArray[np.float64]] = [np.eye(k)]
        for h in range(1, steps):
            phi_h = np.zeros((k, k))
            for i in range(1, min(h, p) + 1):
                phi_h = phi_h + coef[i - 1] @ irf[h - i]
            irf.append(phi_h)

        return irf

    def _build_shock_matrix(
        self,
        steps: int,
        baseline: dict[str, Any],
        sigma_u: NDArray[np.float64],
    ) -> dict[str, float]:
        """Convert all shocks to absolute magnitudes.

        Parameters
        ----------
        steps : int
            Forecast horizon.
        baseline : dict[str, Forecast]
            Baseline forecasts.
        sigma_u : ndarray (k, k)
            Residual covariance.

        Returns
        -------
        dict[str, float]
            Variable name -> absolute shock magnitude.
        """
        var_names = self.model.var_names
        result: dict[str, float] = {}

        for shock in self._shocks:
            var_idx = var_names.index(shock.variable)

            if shock.shock_type == "absolute":
                result[shock.variable] = shock.magnitude

            elif shock.shock_type == "std_dev":
                std = float(np.sqrt(sigma_u[var_idx, var_idx]))
                result[shock.variable] = shock.magnitude * std

            elif shock.shock_type == "percent":
                # Use baseline value at shock period
                period_idx = min(shock.period - 1, steps - 1)
                base_val = float(baseline[shock.variable].point[period_idx])
                result[shock.variable] = shock.magnitude / 100.0 * base_val

        return result

    def run_reverse(
        self,
        target_variable: str,
        target_value: float,
        shock_variable: str,
        steps: int,
    ) -> StressResult:
        """Reverse stress test: find shock that produces target outcome.

        "What shock to shock_variable causes target_variable to reach target_value?"

        Uses IRF to estimate the required shock magnitude.

        Parameters
        ----------
        target_variable : str
            Variable whose forecast we want to match.
        target_value : float
            Desired forecast value at horizon 1.
        shock_variable : str
            Variable to shock.
        steps : int
            Forecast horizon.

        Returns
        -------
        StressResult
            Result with the found shock.
        """
        from forecastbox.scenarios.conditional import ConditionalForecast

        cf = ConditionalForecast(self.model, method="analytic")
        baseline = cf.forecast(steps=steps, conditions=None, n_draws=100, seed=42)

        base_val = baseline[target_variable].point[0]
        target_diff = target_value - base_val

        # Use IRF to estimate required shock
        irf_matrices = self._compute_irf(steps)
        var_names = self.model.var_names
        target_idx = var_names.index(target_variable)
        shock_idx = var_names.index(shock_variable)

        # Phi[0][target_idx, shock_idx] gives the instantaneous response
        response = irf_matrices[0][target_idx, shock_idx]
        if abs(response) < 1e-12:
            msg = (
                f"Variable '{shock_variable}' has no contemporaneous "
                f"effect on '{target_variable}'"
            )
            raise ValueError(msg)

        required_magnitude = target_diff / response

        # Run stress test with found magnitude
        self.clear_shocks()
        self.add_shock(
            shock_variable,
            magnitude=required_magnitude,
            shock_type="absolute",
            period=1,
        )
        return self.run(steps=steps)
