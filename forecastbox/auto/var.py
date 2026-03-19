"""AutoVAR - Automatic VAR model selection.

Implements automatic lag selection and optional variable selection for
Vector Autoregression (VAR) models using information criteria.

Lag Selection:
    Tests VAR(1) through VAR(max_lags) and selects the lag with the
    lowest information criterion (AIC, BIC, HQC, or FPE).

Variable Selection:
    Forward stepwise: starts with the most informative variable and
    adds variables one at a time while IC improves.

IC Formulas for VAR(p) with k variables:
    AIC  = log|Sigma| + (2/T) * k^2 * p
    BIC  = log|Sigma| + (log(T)/T) * k^2 * p
    HQC  = log|Sigma| + (2*log(log(T))/T) * k^2 * p
    FPE  = |Sigma| * ((T + k*p + 1) / (T - k*p - 1))^k

References
----------
Lutkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
Springer. Chapter 4.

Usage
-----
>>> from forecastbox.auto import AutoVAR
>>> auto_var = AutoVAR(max_lags=12, ic='bic')
>>> result = auto_var.fit(df)
>>> print(result.selected_lag)
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

logger = get_logger("auto.var")


def _try_import_var() -> Any:
    """Try to import statsmodels VAR."""
    try:
        from statsmodels.tsa.api import VAR

        return VAR
    except ImportError:
        return None


def _compute_var_ic(
    sigma: NDArray[np.float64],
    n_obs: int,
    n_vars: int,
    n_lags: int,
    ic: str = "bic",
) -> float:
    """Compute information criterion for VAR(p) model.

    Parameters
    ----------
    sigma : NDArray[np.float64]
        Residual covariance matrix (k x k).
    n_obs : int
        Number of observations (T).
    n_vars : int
        Number of variables (k).
    n_lags : int
        Number of lags (p).
    ic : str
        Criterion: 'aic', 'bic', 'hqc', 'fpe'.

    Returns
    -------
    float
        Information criterion value (lower is better).
    """
    t = n_obs
    k = n_vars
    p = n_lags

    # Determinant of covariance matrix
    det_sigma = np.linalg.det(sigma)
    if det_sigma <= 0:
        return np.inf

    log_det = np.log(det_sigma)

    if ic == "aic":
        return float(log_det + (2.0 / t) * k**2 * p)
    elif ic == "bic":
        return float(log_det + (np.log(t) / t) * k**2 * p)
    elif ic == "hqc":
        if np.log(t) <= 0:
            return np.inf
        return float(log_det + (2.0 * np.log(np.log(t)) / t) * k**2 * p)
    elif ic == "fpe":
        denom = t - k * p - 1
        if denom <= 0:
            return np.inf
        return float(det_sigma * ((t + k * p + 1) / denom) ** k)
    else:
        msg = f"Unknown IC: '{ic}'. Use 'aic', 'bic', 'hqc', or 'fpe'."
        raise ValueError(msg)


def _fit_var_model(
    data: pd.DataFrame,
    n_lags: int,
    trend: str = "c",
) -> tuple[Any, NDArray[np.float64], int]:
    """Fit a VAR model and return result, residual covariance, and n_obs.

    Parameters
    ----------
    data : pd.DataFrame
        Multivariate time series data (columns = variables).
    n_lags : int
        Number of lags.
    trend : str
        Trend specification: 'n', 'c', 'ct', 'ctt'.

    Returns
    -------
    tuple[Any, NDArray, int]
        (fitted_model, residual_covariance, effective_n_obs)
    """
    var_cls = _try_import_var()

    if var_cls is None:
        msg = (
            "statsmodels is required for AutoVAR. "
            "Install with: pip install statsmodels"
        )
        raise ImportError(msg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = var_cls(data)
        result = model.fit(maxlags=n_lags, trend=trend)

    sigma = np.asarray(result.sigma_u, dtype=np.float64)
    n_obs = result.nobs

    return result, sigma, n_obs


@dataclass
class AutoVARResult:
    """Result of AutoVAR model selection.

    Attributes
    ----------
    selected_lag : int
        Selected number of lags.
    selected_variables : list[str]
        Selected variable names.
    ic_table : pd.DataFrame
        DataFrame with IC values for each lag tested.
    ic_name : str
        Name of the information criterion used.
    model : Any
        The fitted VAR model (statsmodels VARResults or similar).
    n_vars : int
        Number of variables in the model.
    """

    selected_lag: int
    selected_variables: list[str]
    ic_table: pd.DataFrame
    ic_name: str
    model: Any
    n_vars: int
    _data: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def forecast(
        self,
        h: int,
        level: tuple[int, ...] = (80, 95),
    ) -> Forecast:
        """Generate multivariate forecast from the selected VAR model.

        Parameters
        ----------
        h : int
            Forecast horizon (number of steps ahead).
        level : tuple[int, ...]
            Confidence levels for prediction intervals.

        Returns
        -------
        Forecast
            Forecast object. For multivariate models, point contains the
            forecast for the first variable. Metadata contains all variables.
        """
        if self.model is None:
            msg = "No fitted model available for forecasting"
            raise RuntimeError(msg)

        try:
            # statsmodels VAR forecast
            fc_result = self.model.forecast(
                self.model.endog[-self.selected_lag :],
                steps=h,
            )
            fc_array = np.asarray(fc_result, dtype=np.float64)

            # Point forecast for first variable
            point = fc_array[:, 0]

            # Compute simple prediction intervals using residual std
            resid_std = float(np.std(self.model.resid.iloc[:, 0]))
            horizons = np.arange(1, h + 1, dtype=np.float64)
            width = np.sqrt(horizons) * resid_std

            lower_80 = point - 1.28 * width
            upper_80 = point + 1.28 * width
            lower_95 = point - 1.96 * width
            upper_95 = point + 1.96 * width

            # Store all variables in metadata
            all_forecasts: dict[str, list[float]] = {}
            for i, var_name in enumerate(self.selected_variables):
                all_forecasts[var_name] = fc_array[:, i].tolist()

        except Exception:
            point = np.full(h, np.nan)
            lower_80 = None
            upper_80 = None
            lower_95 = None
            upper_95 = None
            all_forecasts = {}

        return Forecast(
            point=point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            model_name=f"VAR({self.selected_lag})",
            horizon=h,
            metadata={
                "selected_lag": self.selected_lag,
                "selected_variables": self.selected_variables,
                "ic_name": self.ic_name,
                "n_vars": self.n_vars,
                "all_variable_forecasts": all_forecasts,
            },
        )

    def summary(self) -> str:
        """Return a text summary of the AutoVAR result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "AutoVAR Results",
            "=" * 50,
            f"  Selected lag:      {self.selected_lag}",
            f"  Variables:         {self.selected_variables}",
            f"  Number of vars:    {self.n_vars}",
            f"  IC criterion:      {self.ic_name}",
            "",
            "IC Table (by lag):",
        ]

        lines.append(self.ic_table.to_string(index=False))

        return "\n".join(lines)

    def irf(self, steps: int = 20) -> Any:
        """Compute impulse response functions.

        Parameters
        ----------
        steps : int
            Number of steps for IRF computation.

        Returns
        -------
        Any
            IRF result object from statsmodels.
        """
        if self.model is None:
            msg = "No fitted model available for IRF"
            raise RuntimeError(msg)

        try:
            return self.model.irf(steps)
        except Exception as e:
            msg = f"Failed to compute IRF: {e}"
            raise RuntimeError(msg) from e


class AutoVAR:
    """Automatic VAR model selection via lag and variable selection.

    Tests VAR(1) through VAR(max_lags) and selects the lag with the
    lowest information criterion. Optionally performs forward stepwise
    variable selection.

    Parameters
    ----------
    max_lags : int
        Maximum number of lags to test. Default 12.
    ic : str
        Information criterion: 'aic', 'bic', 'hqc', 'fpe'. Default 'bic'.
    trend : str
        Trend specification: 'n' (none), 'c' (constant), 'ct' (constant+trend),
        'ctt' (constant+trend+trend^2). Default 'c'.
    select_variables : bool
        Whether to perform forward stepwise variable selection. Default False.
    max_vars : int or None
        Maximum number of variables (if select_variables=True).
        Default None (no limit).

    Examples
    --------
    >>> auto_var = AutoVAR(max_lags=12, ic='bic')
    >>> result = auto_var.fit(df)
    >>> print(result.selected_lag)
    >>> print(result.ic_table)
    """

    def __init__(
        self,
        max_lags: int = 12,
        ic: str = "bic",
        trend: str = "c",
        select_variables: bool = False,
        max_vars: int | None = None,
    ) -> None:
        self.max_lags = max_lags
        self.ic = ic
        self.trend = trend
        self.select_variables = select_variables
        self.max_vars = max_vars

        if ic not in ("aic", "bic", "hqc", "fpe"):
            msg = f"ic must be 'aic', 'bic', 'hqc', or 'fpe', got '{ic}'"
            raise ValueError(msg)

        if trend not in ("n", "c", "ct", "ctt"):
            msg = f"trend must be 'n', 'c', 'ct', or 'ctt', got '{trend}'"
            raise ValueError(msg)

    def _lag_selection(
        self, data: pd.DataFrame
    ) -> tuple[int, pd.DataFrame]:
        """Select optimal lag order by information criterion.

        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series data.

        Returns
        -------
        tuple[int, pd.DataFrame]
            (selected_lag, ic_table DataFrame)
        """
        n_obs_total = len(data)
        n_vars = data.shape[1]

        # Ensure max_lags doesn't exceed available observations
        effective_max_lags = min(
            self.max_lags,
            (n_obs_total - 1) // (n_vars + 1),
        )
        effective_max_lags = max(1, effective_max_lags)

        ic_records: list[dict[str, Any]] = []
        best_lag = 1
        best_ic = np.inf

        for lag in range(1, effective_max_lags + 1):
            try:
                _result, sigma, n_obs = _fit_var_model(data, lag, self.trend)
                ic_value = _compute_var_ic(sigma, n_obs, n_vars, lag, self.ic)

                # Also compute all IC variants for the table
                aic = _compute_var_ic(sigma, n_obs, n_vars, lag, "aic")
                bic = _compute_var_ic(sigma, n_obs, n_vars, lag, "bic")
                hqc = _compute_var_ic(sigma, n_obs, n_vars, lag, "hqc")

                ic_records.append(
                    {
                        "lag": lag,
                        "aic": aic,
                        "bic": bic,
                        "hqc": hqc,
                        "selected_ic": ic_value,
                    }
                )

                if ic_value < best_ic:
                    best_ic = ic_value
                    best_lag = lag

            except Exception as e:
                logger.debug("Failed to fit VAR(%d): %s", lag, e)
                ic_records.append(
                    {
                        "lag": lag,
                        "aic": np.inf,
                        "bic": np.inf,
                        "hqc": np.inf,
                        "selected_ic": np.inf,
                    }
                )

        ic_table = pd.DataFrame(ic_records)

        return best_lag, ic_table

    def _variable_selection(
        self,
        data: pd.DataFrame,
        lag: int,
    ) -> list[str]:
        """Perform forward stepwise variable selection.

        Starts with the pair of variables with lowest IC, then adds
        one variable at a time while IC improves.

        Parameters
        ----------
        data : pd.DataFrame
            Full multivariate time series data.
        lag : int
            Number of lags (already selected).

        Returns
        -------
        list[str]
            Selected variable names in order of selection.
        """
        all_vars = list(data.columns)

        if len(all_vars) <= 2:
            return all_vars

        max_vars = self.max_vars if self.max_vars is not None else len(all_vars)

        # Start: find the best pair
        best_pair_ic = np.inf
        best_pair: list[str] = all_vars[:2]

        for i in range(len(all_vars)):
            for j in range(i + 1, len(all_vars)):
                pair = [all_vars[i], all_vars[j]]
                try:
                    _, sigma, n_obs = _fit_var_model(
                        data[pair], lag, self.trend
                    )
                    ic_val = _compute_var_ic(sigma, n_obs, 2, lag, self.ic)
                    if ic_val < best_pair_ic:
                        best_pair_ic = ic_val
                        best_pair = pair
                except Exception:
                    continue

        selected = list(best_pair)
        remaining = [v for v in all_vars if v not in selected]
        current_ic = best_pair_ic

        # Forward stepwise: add one variable at a time
        while remaining and len(selected) < max_vars:
            best_addition_ic = current_ic
            best_addition: str | None = None

            for var in remaining:
                candidate = selected + [var]
                try:
                    _, sigma, n_obs = _fit_var_model(
                        data[candidate], lag, self.trend
                    )
                    ic_val = _compute_var_ic(
                        sigma, n_obs, len(candidate), lag, self.ic
                    )
                    if ic_val < best_addition_ic:
                        best_addition_ic = ic_val
                        best_addition = var
                except Exception:
                    continue

            if best_addition is not None:
                selected.append(best_addition)
                remaining.remove(best_addition)
                current_ic = best_addition_ic
            else:
                break  # No improvement

        return selected

    def fit(self, data: pd.DataFrame) -> AutoVARResult:
        """Fit the AutoVAR model to multivariate data.

        Parameters
        ----------
        data : pd.DataFrame
            Multivariate time series data (columns = variables, index = time).

        Returns
        -------
        AutoVARResult
            Result with selected lag, variables, and fitted model.
        """
        if not isinstance(data, pd.DataFrame):
            msg = (
                f"data must be a pandas DataFrame, got {type(data).__name__}"
            )
            raise TypeError(msg)

        if data.shape[1] < 2:
            msg = f"VAR requires at least 2 variables, got {data.shape[1]}"
            raise ValueError(msg)

        if len(data) < 10:
            msg = (
                f"Series too short for AutoVAR: "
                f"{len(data)} observations (need >= 10)"
            )
            raise ValueError(msg)

        # Variable selection (if requested)
        if self.select_variables:
            # First do a rough lag selection with all variables
            rough_lag, _ = self._lag_selection(data)
            selected_vars = self._variable_selection(data, rough_lag)
            data_selected = data[selected_vars]
        else:
            selected_vars = list(data.columns)
            data_selected = data

        # Lag selection on (possibly reduced) dataset
        selected_lag, ic_table = self._lag_selection(data_selected)

        # Fit final model
        try:
            final_model, _sigma, _n_obs = _fit_var_model(
                data_selected, selected_lag, self.trend
            )
        except Exception as e:
            msg = f"Failed to fit final VAR({selected_lag}) model: {e}"
            raise RuntimeError(msg) from e

        return AutoVARResult(
            selected_lag=selected_lag,
            selected_variables=selected_vars,
            ic_table=ic_table,
            ic_name=self.ic,
            model=final_model,
            n_vars=len(selected_vars),
            _data=data_selected,
        )
