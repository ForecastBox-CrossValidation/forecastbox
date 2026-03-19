"""AutoETS - Automatic ETS model selection.

Implements exhaustive search over 30 ETS (Error, Trend, Seasonal) combinations
to find the best model by information criterion. Supports restrictions for
multiplicative components and allows fixing individual components.

The ETS framework covers:
- Error:    Additive (A), Multiplicative (M)
- Trend:    None (N), Additive (A), Additive Damped (Ad), Multiplicative (M), Mult Damped (Md)
- Seasonal: None (N), Additive (A), Multiplicative (M)

Total: 2 * 5 * 3 = 30 combinations

References
----------
Hyndman, R.J., Koehler, A.B., Snyder, R.D. & Grose, S. (2002). "A state space
framework for automatic forecasting using exponential smoothing methods."
International Journal of Forecasting, 18(3), 439-454.

Usage
-----
>>> from forecastbox.auto import AutoETS
>>> auto_ets = AutoETS(seasonal_period=12)
>>> result = auto_ets.fit(data)
>>> print(result.model_type)
>>> forecast = result.forecast(12)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from forecastbox._logging import get_logger
from forecastbox.core.forecast import Forecast

logger = get_logger("auto.ets")

# All possible ETS components
ERROR_TYPES = ("A", "M")
TREND_TYPES = ("N", "A", "Ad", "M", "Md")
SEASONAL_TYPES = ("N", "A", "M")


def _try_import_ets() -> Any:
    """Try to import statsmodels ExponentialSmoothing."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        return ExponentialSmoothing
    except ImportError:
        return None


def _compute_ets_ic(
    sse: float,
    n_params: int,
    n_obs: int,
    ic: str = "aicc",
) -> float:
    """Compute information criterion for ETS model.

    Uses the approximation: AIC = n * log(SSE/n) + 2*k

    Parameters
    ----------
    sse : float
        Sum of squared errors (residuals).
    n_params : int
        Number of estimated parameters.
    n_obs : int
        Number of observations.
    ic : str
        Criterion: 'aic', 'aicc', or 'bic'.

    Returns
    -------
    float
        Information criterion value (lower is better).
    """
    if sse <= 0 or n_obs <= 0:
        return np.inf

    k = n_params
    log_lik_approx = -0.5 * n_obs * (np.log(2 * np.pi * sse / n_obs) + 1)

    if ic == "aic":
        return float(-2.0 * log_lik_approx + 2.0 * k)
    if ic == "aicc":
        aic = -2.0 * log_lik_approx + 2.0 * k
        correction = (2.0 * k * (k + 1)) / (n_obs - k - 1) if n_obs - k - 1 > 0 else np.inf
        return float(aic + correction)
    if ic == "bic":
        return float(-2.0 * log_lik_approx + k * np.log(n_obs))

    msg = f"Unknown IC: '{ic}'. Use 'aic', 'aicc', or 'bic'."
    raise ValueError(msg)


@dataclass
class _ETSCandidateResult:
    """Result of fitting a single ETS candidate."""

    error: str
    trend: str
    seasonal: str
    damped: bool
    ic_value: float
    sse: float
    n_params: int
    converged: bool
    model: Any = None


@dataclass
class AutoETSResult:
    """Result of AutoETS model selection.

    Attributes
    ----------
    model_type : str
        Selected model type string, e.g., 'ETS(M,Ad,M)'.
    error : str
        Selected error type: 'A' or 'M'.
    trend : str
        Selected trend type: 'N', 'A', 'Ad', 'M', 'Md'.
    seasonal : str
        Selected seasonal type: 'N', 'A', 'M'.
    damped : bool
        Whether the trend is damped.
    ic_value : float
        Information criterion value of the best model.
    ic_name : str
        Name of the information criterion used.
    model : Any
        The fitted model object (statsmodels HoltWintersResults or similar).
    n_fits : int
        Number of models fitted during search.
    all_models : pd.DataFrame
        DataFrame with all candidates and their IC values.
    """

    model_type: str
    error: str
    trend: str
    seasonal: str
    damped: bool
    ic_value: float
    ic_name: str
    model: Any
    n_fits: int
    all_models: pd.DataFrame
    _y: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.array([]))

    def forecast(
        self,
        h: int,
        level: tuple[int, ...] = (80, 95),
    ) -> Forecast:
        """Generate forecast from the selected model.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps ahead).
        level : tuple[int, ...]
            Confidence levels for prediction intervals.

        Returns
        -------
        Forecast
            Forecast object with point predictions and intervals.
        """
        if self.model is None:
            msg = "No fitted model available for forecasting"
            raise RuntimeError(msg)

        try:
            # statsmodels forecast
            forecast_result = self.model.forecast(h)
            point = np.asarray(forecast_result, dtype=np.float64)

            # Compute simple prediction intervals based on residual variance
            if hasattr(self.model, "resid"):
                resid_std = float(np.std(self.model.resid))
            else:
                resid_std = float(np.std(self._y[-20:]))  # fallback

            horizons = np.arange(1, h + 1, dtype=np.float64)
            # Intervals widen with horizon (sqrt of horizon for PI)
            width_multiplier = np.sqrt(horizons)

            lower_80 = point - 1.28 * resid_std * width_multiplier
            upper_80 = point + 1.28 * resid_std * width_multiplier
            lower_95 = point - 1.96 * resid_std * width_multiplier
            upper_95 = point + 1.96 * resid_std * width_multiplier

        except Exception:
            point = np.full(h, np.nan)
            lower_80 = None
            upper_80 = None
            lower_95 = None
            upper_95 = None

        return Forecast(
            point=point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            model_name=self.model_type,
            horizon=h,
            metadata={
                "error": self.error,
                "trend": self.trend,
                "seasonal": self.seasonal,
                "damped": self.damped,
                "ic_name": self.ic_name,
                "ic_value": self.ic_value,
                "n_fits": self.n_fits,
            },
        )

    def summary(self) -> str:
        """Return a text summary of the AutoETS result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "AutoETS Results",
            "=" * 50,
            f"  Selected model:    {self.model_type}",
            f"  Error:             {self.error}",
            f"  Trend:             {self.trend} (damped={self.damped})",
            f"  Seasonal:          {self.seasonal}",
            f"  {self.ic_name}:              {self.ic_value:.4f}",
            f"  Models fitted:     {self.n_fits}",
            "",
            "Top 5 models:",
        ]

        top5 = self.all_models.nsmallest(5, "ic_value")
        lines.append(top5.to_string(index=False))

        return "\n".join(lines)


class AutoETS:
    """Automatic ETS model selection via exhaustive search.

    Tests all valid ETS(Error, Trend, Seasonal) combinations and selects
    the best model by information criterion.

    Parameters
    ----------
    seasonal_period : int
        Seasonal period (1 = no seasonality). Default 1.
    error : str or None
        Fix error type ('A' or 'M'). None = test both.
    trend : str or None
        Fix trend type ('N', 'A', 'Ad', 'M', 'Md'). None = test all.
    seasonal : str or None
        Fix seasonal type ('N', 'A', 'M'). None = test all.
    damped : bool or None
        Fix damped flag. None = test both where applicable.
    ic : str
        Information criterion: 'aic', 'aicc', 'bic'. Default 'aicc'.
    restrict : bool
        Apply admissibility restrictions. Default True.

    Examples
    --------
    >>> auto_ets = AutoETS(seasonal_period=12)
    >>> result = auto_ets.fit(data)
    >>> print(result.model_type)
    ETS(M,Ad,M)
    >>> forecast = result.forecast(12)
    """

    def __init__(
        self,
        seasonal_period: int = 1,
        error: str | None = None,
        trend: str | None = None,
        seasonal: str | None = None,
        damped: bool | None = None,
        ic: str = "aicc",
        restrict: bool = True,
    ) -> None:
        self.seasonal_period = seasonal_period
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.damped = damped
        self.ic = ic
        self.restrict = restrict

        if ic not in ("aic", "aicc", "bic"):
            msg = f"ic must be 'aic', 'aicc', or 'bic', got '{ic}'"
            raise ValueError(msg)

        # Validate fixed components
        if error is not None and error not in ERROR_TYPES:
            msg = f"error must be 'A' or 'M', got '{error}'"
            raise ValueError(msg)
        if trend is not None and trend not in TREND_TYPES:
            msg = f"trend must be one of {TREND_TYPES}, got '{trend}'"
            raise ValueError(msg)
        if seasonal is not None and seasonal not in SEASONAL_TYPES:
            msg = f"seasonal must be one of {SEASONAL_TYPES}, got '{seasonal}'"
            raise ValueError(msg)

    def _enumerate_models(
        self, y: NDArray[np.float64]
    ) -> list[tuple[str, str, str, bool]]:
        """Enumerate valid ETS combinations given the data.

        Parameters
        ----------
        y : NDArray
            Time series data.

        Returns
        -------
        list[tuple[str, str, str, bool]]
            List of (error, trend, seasonal, damped) tuples.
        """
        has_non_positive = bool(np.any(y <= 0))

        # Determine which components to test
        error_types = (self.error,) if self.error is not None else ERROR_TYPES
        trend_types = (self.trend,) if self.trend is not None else TREND_TYPES
        seasonal_types = (
            (self.seasonal,) if self.seasonal is not None else SEASONAL_TYPES
        )

        # If seasonal_period < 2, only allow N for seasonal
        if self.seasonal_period < 2 and self.seasonal is None:
            seasonal_types = ("N",)

        models: list[tuple[str, str, str, bool]] = []

        for e in error_types:
            for t in trend_types:
                for s in seasonal_types:
                    # Determine damped flag
                    if t in ("Ad", "Md"):
                        is_damped = True
                        base_trend = t[0]  # A or M
                    else:
                        is_damped = False
                        base_trend = t

                    # Handle damped filter
                    if self.damped is not None:
                        if self.damped and not is_damped:
                            continue
                        if not self.damped and is_damped:
                            continue

                    # Apply restrictions for multiplicative components
                    if self.restrict:
                        if e == "M" and has_non_positive:
                            continue
                        if base_trend == "M" and has_non_positive:
                            continue
                        if s == "M" and has_non_positive:
                            continue
                        if s == "M" and self.seasonal_period < 2:
                            continue
                        if s != "N" and self.seasonal_period < 2:
                            continue

                    models.append((e, t, s, is_damped))

        return models

    def _apply_restrictions(
        self, models: list[tuple[str, str, str, bool]]
    ) -> list[tuple[str, str, str, bool]]:
        """Apply additional admissibility restrictions.

        Parameters
        ----------
        models : list
            List of (error, trend, seasonal, damped) tuples.

        Returns
        -------
        list
            Filtered list of admissible models.
        """
        # Additional restrictions could be added here
        # For now, the main restrictions are applied in _enumerate_models
        return models

    def _fit_candidate(
        self,
        y: NDArray[np.float64],
        error: str,
        trend: str,
        seasonal: str,
        damped: bool,
    ) -> _ETSCandidateResult:
        """Fit a single ETS candidate model.

        Parameters
        ----------
        y : NDArray
            Time series data.
        error : str
            Error type.
        trend : str
            Trend type (including damped variants like 'Ad').
        seasonal : str
            Seasonal type.
        damped : bool
            Whether trend is damped.

        Returns
        -------
        _ETSCandidateResult
            Fit result with IC value and model object.
        """
        exponential_smoothing = _try_import_ets()

        if exponential_smoothing is None:
            return _ETSCandidateResult(
                error=error,
                trend=trend,
                seasonal=seasonal,
                damped=damped,
                ic_value=np.inf,
                sse=np.inf,
                n_params=0,
                converged=False,
                model=None,
            )

        # Map ETS notation to statsmodels parameters
        # Trend: N -> None, A/Ad -> 'add', M/Md -> 'mul'
        base_trend = trend[0] if trend != "N" else trend
        sm_trend: str | None
        if base_trend == "N":
            sm_trend = None
        elif base_trend == "A":
            sm_trend = "add"
        elif base_trend == "M":
            sm_trend = "mul"
        else:
            sm_trend = None

        # Seasonal: N -> None, A -> 'add', M -> 'mul'
        sm_seasonal: str | None
        if seasonal == "N":
            sm_seasonal = None
        elif seasonal == "A":
            sm_seasonal = "add"
        elif seasonal == "M":
            sm_seasonal = "mul"
        else:
            sm_seasonal = None

        sp = self.seasonal_period if sm_seasonal is not None else None

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = exponential_smoothing(
                    y,
                    trend=sm_trend,
                    damped_trend=damped,
                    seasonal=sm_seasonal,
                    seasonal_periods=sp,
                    initialization_method="estimated",
                )
                result = model.fit(optimized=True, remove_bias=False)

            # Compute SSE and IC
            sse = float(np.sum(result.resid**2))
            n_obs = len(y)

            # Count parameters
            n_params = 1  # alpha (smoothing for level)
            if sm_trend is not None:
                n_params += 1  # beta (smoothing for trend)
            if damped:
                n_params += 1  # phi (damping parameter)
            if sm_seasonal is not None:
                n_params += 1  # gamma (smoothing for seasonal)
            # Initial states
            n_params += 1  # initial level
            if sm_trend is not None:
                n_params += 1  # initial trend
            if sm_seasonal is not None and sp is not None:
                n_params += sp - 1  # initial seasonal states
            n_params += 1  # sigma^2

            ic_value = _compute_ets_ic(sse, n_params, n_obs, self.ic)

            return _ETSCandidateResult(
                error=error,
                trend=trend,
                seasonal=seasonal,
                damped=damped,
                ic_value=ic_value,
                sse=sse,
                n_params=n_params,
                converged=True,
                model=result,
            )

        except Exception as e:
            logger.debug(
                "Failed to fit ETS(%s,%s,%s): %s", error, trend, seasonal, e
            )
            return _ETSCandidateResult(
                error=error,
                trend=trend,
                seasonal=seasonal,
                damped=damped,
                ic_value=np.inf,
                sse=np.inf,
                n_params=0,
                converged=False,
                model=None,
            )

    def fit(self, y: pd.Series | NDArray[np.float64]) -> AutoETSResult:
        """Fit the AutoETS model to data.

        Tests all valid ETS combinations and selects the best by IC.

        Parameters
        ----------
        y : pd.Series or NDArray
            Time series data.

        Returns
        -------
        AutoETSResult
            Result with selected model, components, and diagnostics.
        """
        y_arr = np.asarray(y, dtype=np.float64)

        if len(y_arr) < 4:
            msg = (
                f"Series too short for AutoETS: "
                f"{len(y_arr)} observations (need >= 4)"
            )
            raise ValueError(msg)

        # Enumerate valid models
        model_specs = self._enumerate_models(y_arr)

        if self.restrict:
            model_specs = self._apply_restrictions(model_specs)

        if not model_specs:
            msg = "No valid ETS models to test given the data and restrictions"
            raise RuntimeError(msg)

        # Fit all candidates
        all_candidates: list[_ETSCandidateResult] = []
        for error, trend, seasonal, damped in model_specs:
            candidate = self._fit_candidate(y_arr, error, trend, seasonal, damped)
            all_candidates.append(candidate)

        # Find best
        best = min(all_candidates, key=lambda c: c.ic_value)

        # Build all_models DataFrame
        all_models_data = []
        for c in all_candidates:
            model_type = f"ETS({c.error},{c.trend},{c.seasonal})"
            all_models_data.append(
                {
                    "model_type": model_type,
                    "error": c.error,
                    "trend": c.trend,
                    "seasonal": c.seasonal,
                    "damped": c.damped,
                    "ic_value": c.ic_value,
                    "converged": c.converged,
                }
            )

        all_models_df = pd.DataFrame(all_models_data)
        all_models_df = all_models_df.sort_values("ic_value").reset_index(drop=True)

        best_model_type = f"ETS({best.error},{best.trend},{best.seasonal})"

        return AutoETSResult(
            model_type=best_model_type,
            error=best.error,
            trend=best.trend,
            seasonal=best.seasonal,
            damped=best.damped,
            ic_value=best.ic_value,
            ic_name=self.ic,
            model=best.model,
            n_fits=len(all_candidates),
            all_models=all_models_df,
            _y=y_arr,
        )
