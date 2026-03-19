"""AutoARIMA - Automatic ARIMA model selection.

Implements the Hyndman-Khandakar (2008) stepwise algorithm for automatic
ARIMA order selection. Supports seasonal ARIMA models, information criteria
selection (AIC, AICc, BIC), and both stepwise and grid search strategies.

References
----------
Hyndman, R.J. & Khandakar, Y. (2008). "Automatic time series forecasting:
the forecast package for R." Journal of Statistical Software, 27(3), 1-22.

Usage
-----
>>> from forecastbox.auto import AutoARIMA
>>> auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
>>> result = auto.fit(data)
>>> print(result.order)
>>> forecast = result.forecast(12)
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from forecastbox._logging import get_logger
from forecastbox.auto._stepwise import (
    _determine_d,
    _determine_seasonal_d,
    _generate_neighbors,
    _is_valid_order,
)
from forecastbox.core.forecast import Forecast

logger = get_logger("auto.arima")


def _try_import_statsmodels() -> Any:
    """Try to import statsmodels SARIMAX."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX
    except ImportError:
        return None


def _compute_ic(
    log_likelihood: float,
    n_params: int,
    n_obs: int,
    ic: str = "aicc",
) -> float:
    """Compute information criterion.

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the fitted model.
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
    k = n_params

    if ic == "aic":
        return -2.0 * log_likelihood + 2.0 * k
    elif ic == "aicc":
        aic = -2.0 * log_likelihood + 2.0 * k
        correction = (2.0 * k * (k + 1)) / (n_obs - k - 1) if n_obs - k - 1 > 0 else np.inf
        return aic + correction
    elif ic == "bic":
        return -2.0 * log_likelihood + k * np.log(n_obs)
    else:
        msg = f"Unknown information criterion: '{ic}'. Use 'aic', 'aicc', or 'bic'."
        raise ValueError(msg)


@dataclass
class _CandidateResult:
    """Result of fitting a single ARIMA candidate."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    include_constant: bool
    ic_value: float
    log_likelihood: float
    n_params: int
    converged: bool
    model: Any = None


@dataclass
class AutoARIMAResult:
    """Result of AutoARIMA model selection.

    Attributes
    ----------
    order : tuple[int, int, int]
        Selected (p, d, q) order.
    seasonal_order : tuple[int, int, int, int]
        Selected (P, D, Q, m) seasonal order.
    ic_value : float
        Information criterion value of the best model.
    ic_name : str
        Name of the information criterion used.
    include_constant : bool
        Whether the selected model includes a constant.
    model : Any
        The fitted model object (statsmodels SARIMAX result or similar).
    n_fits : int
        Number of models fitted during search.
    all_models : pd.DataFrame
        DataFrame with all candidates and their IC values.
    search_method : str
        'stepwise' or 'grid'.
    """

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    ic_value: float
    ic_name: str
    include_constant: bool
    model: Any
    n_fits: int
    all_models: pd.DataFrame
    search_method: str
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
            # statsmodels SARIMAX forecast
            forecast_result = self.model.get_forecast(steps=h)
            point = forecast_result.predicted_mean.values
            ci_80 = forecast_result.conf_int(alpha=0.20)
            ci_95 = forecast_result.conf_int(alpha=0.05)

            lower_80 = ci_80.iloc[:, 0].values
            upper_80 = ci_80.iloc[:, 1].values
            lower_95 = ci_95.iloc[:, 0].values
            upper_95 = ci_95.iloc[:, 1].values
        except (AttributeError, Exception):
            # Fallback: generate point forecast only
            try:
                forecast_result = self.model.forecast(steps=h)
                if hasattr(forecast_result, "values"):
                    point = forecast_result.values
                else:
                    point = np.asarray(forecast_result)
            except Exception:
                point = np.full(h, np.nan)
            lower_80 = None
            upper_80 = None
            lower_95 = None
            upper_95 = None

        p, d, q = self.order
        big_p, big_d, big_q, m = self.seasonal_order
        model_name = f"ARIMA({p},{d},{q})"
        if m > 1:
            model_name += f"({big_p},{big_d},{big_q})[{m}]"

        return Forecast(
            point=np.asarray(point, dtype=np.float64),
            lower_80=np.asarray(lower_80, dtype=np.float64) if lower_80 is not None else None,
            upper_80=np.asarray(upper_80, dtype=np.float64) if upper_80 is not None else None,
            lower_95=np.asarray(lower_95, dtype=np.float64) if lower_95 is not None else None,
            upper_95=np.asarray(upper_95, dtype=np.float64) if upper_95 is not None else None,
            model_name=model_name,
            horizon=h,
            metadata={
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "ic_name": self.ic_name,
                "ic_value": self.ic_value,
                "include_constant": self.include_constant,
                "n_fits": self.n_fits,
                "search_method": self.search_method,
            },
        )

    def summary(self) -> str:
        """Return a text summary of the AutoARIMA result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        p, d, q = self.order
        big_p, big_d, big_q, m = self.seasonal_order

        lines = [
            "AutoARIMA Results",
            "=" * 50,
            f"  Selected order:    ARIMA({p},{d},{q})",
        ]
        if m > 1:
            lines.append(f"  Seasonal order:    ({big_p},{big_d},{big_q})[{m}]")
        lines.extend([
            f"  Include constant:  {self.include_constant}",
            f"  {self.ic_name}:              {self.ic_value:.4f}",
            f"  Search method:     {self.search_method}",
            f"  Models fitted:     {self.n_fits}",
            "",
            "Top 5 models:",
        ])

        top5 = self.all_models.nsmallest(5, "ic_value")
        lines.append(top5.to_string(index=False))

        return "\n".join(lines)


class AutoARIMA:
    """Automatic ARIMA model selection via stepwise search.

    Implements the Hyndman-Khandakar (2008) algorithm for automatic
    selection of ARIMA(p,d,q)(P,D,Q)[m] models.

    Parameters
    ----------
    d : int or None
        Order of first differencing. None = auto via KPSS test.
    D : int or None
        Order of seasonal differencing. None = auto via OCSB test.
    max_p : int
        Maximum p (AR order). Default 5.
    max_q : int
        Maximum q (MA order). Default 5.
    max_P : int
        Maximum P (seasonal AR order). Default 2.
    max_Q : int
        Maximum Q (seasonal MA order). Default 2.
    max_order : int
        Maximum p + q + P + Q. Default 5.
    seasonal : bool
        Whether to include seasonal component. Default True.
    m : int
        Seasonal period (1 = non-seasonal). Default 1.
    stepwise : bool
        Use stepwise search (True) or grid search (False). Default True.
    ic : str
        Information criterion: 'aic', 'aicc', 'bic'. Default 'aicc'.
    trace : bool
        Print progress during search. Default False.

    Examples
    --------
    >>> auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
    >>> result = auto.fit(data)
    >>> print(result.order)
    (1, 1, 1)
    >>> forecast = result.forecast(12)
    """

    def __init__(
        self,
        d: int | None = None,
        D: int | None = None,  # noqa: N803
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,  # noqa: N803
        max_Q: int = 2,  # noqa: N803
        max_order: int = 5,
        seasonal: bool = True,
        m: int = 1,
        stepwise: bool = True,
        ic: str = "aicc",
        trace: bool = False,
    ) -> None:
        self.d = d
        self.D = D
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_order = max_order
        self.seasonal = seasonal
        self.m = m if seasonal else 1
        self.stepwise = stepwise
        self.ic = ic
        self.trace = trace

        if ic not in ("aic", "aicc", "bic"):
            msg = f"ic must be 'aic', 'aicc', or 'bic', got '{ic}'"
            raise ValueError(msg)

    def fit(self, y: pd.Series | NDArray[np.float64]) -> AutoARIMAResult:
        """Fit the AutoARIMA model to data.

        Parameters
        ----------
        y : pd.Series or NDArray
            Time series data.

        Returns
        -------
        AutoARIMAResult
            Result with selected model, order, and diagnostics.
        """
        y_arr = np.asarray(y, dtype=np.float64)

        if len(y_arr) < 10:
            msg = f"Series too short for AutoARIMA: {len(y_arr)} observations (need >= 10)"
            raise ValueError(msg)

        # Step 1: Determine d
        d = self.d if self.d is not None else _determine_d(y_arr)
        if self.trace:
            print(f"  Determined d = {d}")

        # Step 2: Determine D
        if self.seasonal and self.m > 1:
            big_d = self.D if self.D is not None else _determine_seasonal_d(y_arr, self.m)
        else:
            big_d = 0
        if self.trace:
            print(f"  Determined D = {big_d}")

        # Step 3: Search
        if self.stepwise:
            return self._stepwise_search(y_arr, d, big_d)
        return self._grid_search(y_arr, d, big_d)

    def _fit_candidate(
        self,
        y: NDArray[np.float64],
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        include_constant: bool = True,
    ) -> _CandidateResult:
        """Fit a single ARIMA candidate model.

        Parameters
        ----------
        y : NDArray
            Time series data.
        order : tuple[int, int, int]
            (p, d, q) order.
        seasonal_order : tuple[int, int, int, int]
            (P, D, Q, m) seasonal order.
        include_constant : bool
            Whether to include a constant/drift term.

        Returns
        -------
        _CandidateResult
            Fit result with IC value and model object.
        """
        sarimax_cls = _try_import_statsmodels()

        if sarimax_cls is None:
            return _CandidateResult(
                order=order,
                seasonal_order=seasonal_order,
                include_constant=include_constant,
                ic_value=np.inf,
                log_likelihood=-np.inf,
                n_params=0,
                converged=False,
                model=None,
            )

        p, d, q = order
        big_p, big_d, big_q, m = seasonal_order

        # Determine trend parameter
        if include_constant:
            if d + big_d == 0:
                trend = "c"
            elif d + big_d == 1:
                trend = "t"
            else:
                trend = "n"
        else:
            trend = "n"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sarimax_cls(
                    y,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=200)

            log_likelihood = float(result.llf)
            n_params = int(result.df_model) + 1  # +1 for variance
            n_obs = len(y) - d - big_d * m
            ic_value = _compute_ic(log_likelihood, n_params, n_obs, self.ic)

            return _CandidateResult(
                order=order,
                seasonal_order=seasonal_order,
                include_constant=include_constant,
                ic_value=ic_value,
                log_likelihood=log_likelihood,
                n_params=n_params,
                converged=True,
                model=result,
            )

        except Exception as e:
            logger.debug("Failed to fit ARIMA%s%s: %s", order, seasonal_order, e)
            return _CandidateResult(
                order=order,
                seasonal_order=seasonal_order,
                include_constant=include_constant,
                ic_value=np.inf,
                log_likelihood=-np.inf,
                n_params=0,
                converged=False,
                model=None,
            )

    def _stepwise_search(
        self,
        y: NDArray[np.float64],
        d: int,
        big_d: int,
    ) -> AutoARIMAResult:
        """Perform stepwise search for best ARIMA order.

        Implements the Hyndman-Khandakar (2008) stepwise algorithm:
        1. Fit 4 initial models
        2. Iterate: try all neighbors of current best
        3. Stop when no neighbor improves IC

        Parameters
        ----------
        y : NDArray
            Time series data.
        d : int
            Regular differencing order.
        big_d : int
            Seasonal differencing order.

        Returns
        -------
        AutoARIMAResult
            Result with best model and all candidates.
        """
        m = self.m
        all_candidates: list[_CandidateResult] = []
        seen: set[tuple[int, int, int, int, int, int, bool]] = set()

        def _fit_and_record(
            order: tuple[int, int, int],
            seasonal_order: tuple[int, int, int, int],
            include_constant: bool,
        ) -> _CandidateResult | None:
            p_val, d_val, q_val = order
            big_p_val, big_d_val, big_q_val, m_val = seasonal_order
            key = (p_val, d_val, q_val, big_p_val, big_d_val, big_q_val, include_constant)
            if key in seen:
                return None
            seen.add(key)

            candidate = self._fit_candidate(y, order, seasonal_order, include_constant)
            all_candidates.append(candidate)

            if self.trace:
                status = "OK" if candidate.converged else "FAIL"
                print(
                    f"  ARIMA({p_val},{d_val},{q_val})"
                    f"({big_p_val},{big_d_val},{big_q_val})[{m_val}]"
                    f" const={include_constant}"
                    f" {self.ic}={candidate.ic_value:.4f}"
                    f" [{status}]"
                )

            return candidate

        # Step 4: Fit initial models
        if self.trace:
            print("Step 1: Fitting initial models...")

        if self.seasonal and m > 1:
            initial_models: list[tuple[tuple[int, int, int], tuple[int, int, int, int], bool]] = [
                ((2, d, 2), (1, big_d, 1, m), True),
                ((0, d, 0), (0, big_d, 0, m), True),
                ((1, d, 0), (1, big_d, 0, m), True),
                ((0, d, 1), (0, big_d, 1, m), True),
            ]
        else:
            initial_models = [
                ((2, d, 2), (0, 0, 0, m), True),
                ((0, d, 0), (0, 0, 0, m), True),
                ((1, d, 0), (0, 0, 0, m), True),
                ((0, d, 1), (0, 0, 0, m), True),
            ]

        for order, seas_order, const in initial_models:
            p_val, d_val, q_val = order
            big_p_val, big_d_val, big_q_val, m_val = seas_order
            if _is_valid_order(
                p_val, d_val, q_val, big_p_val, big_d_val, big_q_val,
                self.max_p, self.max_q, self.max_P, self.max_Q, self.max_order,
            ):
                _fit_and_record(order, seas_order, const)

        # Find best so far
        best = min(all_candidates, key=lambda c: c.ic_value)

        # Step 5-6: Stepwise iteration
        if self.trace:
            print(f"\nStep 2: Stepwise search from ARIMA{best.order}{best.seasonal_order}...")

        improved = True
        while improved:
            improved = False

            neighbors = _generate_neighbors(
                order=best.order,
                seasonal_order=best.seasonal_order,
                include_constant=best.include_constant,
                max_p=self.max_p,
                max_q=self.max_q,
                max_seasonal_p=self.max_P,
                max_seasonal_q=self.max_Q,
                max_order=self.max_order,
            )

            for nb_order, nb_seasonal, nb_const in neighbors:
                result = _fit_and_record(nb_order, nb_seasonal, nb_const)
                if result is not None and result.ic_value < best.ic_value:
                    best = result
                    improved = True

        # Build all_models DataFrame
        all_models_df = self._build_all_models_df(all_candidates)

        if self.trace:
            print(f"\nBest model: ARIMA{best.order}{best.seasonal_order}")
            print(f"  {self.ic} = {best.ic_value:.4f}")
            print(f"  Models fitted: {len(all_candidates)}")

        return AutoARIMAResult(
            order=best.order,
            seasonal_order=best.seasonal_order,
            ic_value=best.ic_value,
            ic_name=self.ic,
            include_constant=best.include_constant,
            model=best.model,
            n_fits=len(all_candidates),
            all_models=all_models_df,
            search_method="stepwise",
            _y=y,
        )

    def _grid_search(
        self,
        y: NDArray[np.float64],
        d: int,
        big_d: int,
    ) -> AutoARIMAResult:
        """Perform exhaustive grid search for best ARIMA order.

        Tests all valid combinations of (p, q, P, Q) within bounds.

        Parameters
        ----------
        y : NDArray
            Time series data.
        d : int
            Regular differencing order.
        big_d : int
            Seasonal differencing order.

        Returns
        -------
        AutoARIMAResult
            Result with best model and all candidates.
        """
        m = self.m
        all_candidates: list[_CandidateResult] = []

        p_range = range(0, self.max_p + 1)
        q_range = range(0, self.max_q + 1)

        if self.seasonal and m > 1:
            big_p_range = range(0, self.max_P + 1)
            big_q_range = range(0, self.max_Q + 1)
        else:
            big_p_range = range(0, 1)
            big_q_range = range(0, 1)

        if self.trace:
            print("Grid search: testing all valid combinations...")

        for p, q, big_p, big_q in itertools.product(p_range, q_range, big_p_range, big_q_range):
            if not _is_valid_order(
                p, d, q, big_p, big_d, big_q,
                self.max_p, self.max_q, self.max_P, self.max_Q, self.max_order,
            ):
                continue

            order = (p, d, q)
            seasonal_order = (big_p, big_d, big_q, m)

            for include_constant in [True, False]:
                candidate = self._fit_candidate(y, order, seasonal_order, include_constant)
                all_candidates.append(candidate)

                if self.trace:
                    status = "OK" if candidate.converged else "FAIL"
                    print(
                        f"  ARIMA({p},{d},{q})({big_p},{big_d},{big_q})[{m}]"
                        f" const={include_constant}"
                        f" {self.ic}={candidate.ic_value:.4f}"
                        f" [{status}]"
                    )

        if not all_candidates:
            msg = "No valid ARIMA models found in grid search"
            raise RuntimeError(msg)

        best = min(all_candidates, key=lambda c: c.ic_value)
        all_models_df = self._build_all_models_df(all_candidates)

        if self.trace:
            print(f"\nBest model: ARIMA{best.order}{best.seasonal_order}")
            print(f"  {self.ic} = {best.ic_value:.4f}")
            print(f"  Models fitted: {len(all_candidates)}")

        return AutoARIMAResult(
            order=best.order,
            seasonal_order=best.seasonal_order,
            ic_value=best.ic_value,
            ic_name=self.ic,
            include_constant=best.include_constant,
            model=best.model,
            n_fits=len(all_candidates),
            all_models=all_models_df,
            search_method="grid",
            _y=y,
        )

    @staticmethod
    def _build_all_models_df(candidates: list[_CandidateResult]) -> pd.DataFrame:
        """Build a sorted DataFrame of all candidate results."""
        all_models_data = []
        for c in candidates:
            p_val, d_val, q_val = c.order
            big_p_val, big_d_val, big_q_val, m_val = c.seasonal_order
            all_models_data.append({
                "order": f"({p_val},{d_val},{q_val})",
                "seasonal": f"({big_p_val},{big_d_val},{big_q_val})[{m_val}]",
                "constant": c.include_constant,
                "ic_value": c.ic_value,
                "converged": c.converged,
            })

        df = pd.DataFrame(all_models_data)
        return df.sort_values("ic_value").reset_index(drop=True)
