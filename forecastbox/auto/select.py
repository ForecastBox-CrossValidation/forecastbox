"""AutoSelect - Cross-family model selection via cross-validation.

Compares the best model from each family (ARIMA, ETS, etc.) using
out-of-sample cross-validation to select the overall best model.

Algorithm:
1. For each family: select best model within family (via AutoARIMA, AutoETS, etc.)
2. For each best model: run expanding/rolling window CV
3. Rank by aggregate forecast metric (RMSE, MAE, etc.)
4. Return ranking and best model

Usage
-----
>>> from forecastbox.auto import AutoSelect
>>> selector = AutoSelect(families=['arima', 'ets'])
>>> result = selector.fit(data, cv_horizon=12)
>>> print(result.ranking)
>>> print(result.best_model)
>>> forecast = result.forecast(12)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from forecastbox._logging import get_logger
from forecastbox.core.forecast import Forecast

logger = get_logger("auto.select")


def _compute_metric(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    metric: str = "rmse",
) -> float:
    """Compute forecast evaluation metric.

    Parameters
    ----------
    actual : NDArray
        Actual values.
    predicted : NDArray
        Predicted values.
    metric : str
        Metric name: 'rmse', 'mae', 'mape', 'mase'.

    Returns
    -------
    float
        Metric value.
    """
    errors = actual - predicted

    if metric == "rmse":
        return float(np.sqrt(np.mean(errors**2)))
    elif metric == "mae":
        return float(np.mean(np.abs(errors)))
    elif metric == "mape":
        nonzero = actual != 0
        if not np.any(nonzero):
            return np.inf
        return float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100)
    elif metric == "mase":
        # Simplified MASE using naive forecast (last value)
        naive_errors = np.abs(np.diff(actual))
        if len(naive_errors) == 0 or np.mean(naive_errors) == 0:
            return np.inf
        return float(np.mean(np.abs(errors)) / np.mean(naive_errors))
    else:
        msg = f"Unknown metric: '{metric}'. Use 'rmse', 'mae', 'mape', or 'mase'."
        raise ValueError(msg)


@dataclass
class _FamilyResult:
    """Result of fitting and evaluating a single family."""

    family: str
    model_name: str
    model: Any
    cv_scores: list[float]
    cv_mean: float
    cv_by_horizon: list[float]
    forecast_fn: Any  # callable to generate forecasts


@dataclass
class AutoSelectResult:
    """Result of AutoSelect cross-family model selection.

    Attributes
    ----------
    ranking : pd.DataFrame
        DataFrame with model ranking by cross-validation metric.
        Columns: family, model_name, cv_mean, cv_std.
    best_model : Any
        The best fitted model object.
    best_family : str
        Family of the best model.
    best_model_name : str
        Name/description of the best model.
    all_cv_results : dict[str, list[float]]
        CV scores per family.
    metric_name : str
        Name of the metric used for ranking.
    """

    ranking: pd.DataFrame
    best_model: Any
    best_family: str
    best_model_name: str
    all_cv_results: dict[str, list[float]]
    metric_name: str
    _y: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.array([]))
    _family_results: list[_FamilyResult] = field(repr=False, default_factory=list)

    def forecast(
        self,
        h: int,
        level: tuple[int, ...] = (80, 95),
    ) -> Forecast:
        """Generate forecast using the best model.

        Parameters
        ----------
        h : int
            Forecast horizon.
        level : tuple[int, ...]
            Confidence levels for prediction intervals.

        Returns
        -------
        Forecast
            Forecast from the best model.
        """
        # Find the best family result
        for fr in self._family_results:
            if fr.family == self.best_family:
                try:
                    fc = fr.forecast_fn(h)
                    if isinstance(fc, Forecast):
                        return fc
                    # If forecast_fn returns something else, wrap it
                    point = np.asarray(fc, dtype=np.float64)
                    return Forecast(
                        point=point,
                        model_name=self.best_model_name,
                        horizon=h,
                    )
                except Exception:
                    pass

        # Fallback: try direct forecast on best_model
        if hasattr(self.best_model, "forecast"):
            result = self.best_model.forecast(h)
            if isinstance(result, Forecast):
                return result

        msg = "Unable to generate forecast from the best model"
        raise RuntimeError(msg)

    def summary(self) -> str:
        """Return a text summary of the AutoSelect result.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "AutoSelect Results",
            "=" * 60,
            f"  Best family:       {self.best_family}",
            f"  Best model:        {self.best_model_name}",
            f"  Metric:            {self.metric_name}",
            "",
            "Ranking:",
        ]
        lines.append(self.ranking.to_string(index=False))

        return "\n".join(lines)

    def plot_comparison(
        self,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot comparison of CV metrics across families.

        Parameters
        ----------
        ax : matplotlib Axes or None
            Axes to plot on. Creates new figure if None.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        families = self.ranking["family"].tolist()
        cv_means = self.ranking["cv_mean"].tolist()
        cv_stds = self.ranking.get("cv_std", pd.Series([0] * len(families))).tolist()

        x_pos = range(len(families))
        bars = ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)

        # Highlight best model
        if len(bars) > 0:
            bars[0].set_color("green")
            bars[0].set_alpha(1.0)

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(families, rotation=45, ha="right")
        ax.set_ylabel(self.metric_name.upper())
        ax.set_title(title or f"Cross-Family Comparison ({self.metric_name.upper()})")
        ax.grid(True, alpha=0.3, axis="y")

        return ax


class AutoSelect:
    """Cross-family model selection via cross-validation.

    Compares the best model from each family using out-of-sample
    cross-validation to select the overall best model.

    Parameters
    ----------
    families : list[str]
        Families to test. Default ['arima', 'ets'].
        Supported: 'arima', 'ets', 'naive', 'snaive', 'drift'.
        Custom families can be added via ModelZoo.
    cv_type : str
        Type of CV: 'expanding' or 'rolling'. Default 'expanding'.
    cv_initial : int or None
        Initial window size. Default None (2/3 of data).
    cv_horizon : int
        Forecast horizon for CV. Default 12.
    cv_step : int
        Step between CV folds. Default 1.
    metric : str
        Metric for ranking: 'rmse', 'mae', 'mape', 'mase'. Default 'rmse'.

    Examples
    --------
    >>> selector = AutoSelect(families=['arima', 'ets'])
    >>> result = selector.fit(data, cv_horizon=12)
    >>> print(result.ranking)
    >>> print(result.best_model)
    """

    def __init__(
        self,
        families: list[str] | None = None,
        cv_type: str = "expanding",
        cv_initial: int | None = None,
        cv_horizon: int = 12,
        cv_step: int = 1,
        metric: str = "rmse",
    ) -> None:
        self.families = families or ["arima", "ets"]
        self.cv_type = cv_type
        self.cv_initial = cv_initial
        self.cv_horizon = cv_horizon
        self.cv_step = cv_step
        self.metric = metric

        if cv_type not in ("expanding", "rolling"):
            msg = f"cv_type must be 'expanding' or 'rolling', got '{cv_type}'"
            raise ValueError(msg)

        if metric not in ("rmse", "mae", "mape", "mase"):
            msg = f"metric must be 'rmse', 'mae', 'mape', or 'mase', got '{metric}'"
            raise ValueError(msg)

    def _get_best_in_family(
        self,
        y: NDArray[np.float64],
        family: str,
        m: int = 1,
    ) -> tuple[Any, str, Any]:
        """Get the best model within a family.

        Parameters
        ----------
        y : NDArray
            Time series data.
        family : str
            Model family name.
        m : int
            Seasonal period.

        Returns
        -------
        tuple[Any, str, Any]
            (auto_result, model_name, forecast_callable)
        """
        if family == "arima":
            from forecastbox.auto.arima import AutoARIMA

            auto = AutoARIMA(seasonal=m > 1, m=m, stepwise=True, ic="aicc")
            result = auto.fit(y)
            p, d, q = result.order
            sp, sd, sq, m_val = result.seasonal_order
            name = f"ARIMA({p},{d},{q})"
            if m_val > 1:
                name += f"({sp},{sd},{sq})[{m_val}]"
            return result, name, result.forecast

        elif family == "ets":
            from forecastbox.auto.ets import AutoETS

            auto = AutoETS(seasonal_period=m, ic="aicc")
            result = auto.fit(y)
            return result, result.model_type, result.forecast

        elif family in ("naive", "snaive", "drift"):
            from forecastbox.auto._baselines import (
                DriftBaseline,
                NaiveBaseline,
                SeasonalNaiveBaseline,
            )

            if family == "naive":
                model = NaiveBaseline()
            elif family == "snaive":
                model = SeasonalNaiveBaseline(seasonal_period=max(m, 2))
            else:
                model = DriftBaseline()

            model.fit(y)
            return model, family.capitalize(), model.forecast

        else:
            # Try ModelZoo
            try:
                from forecastbox.auto.zoo import ModelZoo

                zoo = ModelZoo()
                model = zoo.create(family)
                model.fit(y)
                return model, family, model.forecast
            except (KeyError, Exception) as e:
                msg = f"Unknown family '{family}' and not found in ModelZoo: {e}"
                raise ValueError(msg) from e

    def _cross_validate(
        self,
        y: NDArray[np.float64],
        family: str,
        m: int = 1,
    ) -> tuple[list[float], list[float]]:
        """Run cross-validation for a model family.

        Parameters
        ----------
        y : NDArray
            Full time series data.
        family : str
            Model family.
        m : int
            Seasonal period.

        Returns
        -------
        tuple[list[float], list[float]]
            (list of per-fold metric values, list of per-horizon mean metrics)
        """
        n = len(y)
        initial = self.cv_initial if self.cv_initial is not None else int(2 * n / 3)
        h = self.cv_horizon
        step = self.cv_step

        fold_scores: list[float] = []
        horizon_scores: list[list[float]] = [[] for _ in range(h)]

        fold_start = initial
        while fold_start + h <= n:
            train = y[:fold_start]
            test = y[fold_start : fold_start + h]

            if self.cv_type == "rolling":
                # Rolling window: fixed size
                window_size = initial
                if fold_start > window_size:
                    train = y[fold_start - window_size : fold_start]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, _, forecast_fn = self._get_best_in_family(train, family, m)
                    fc = forecast_fn(h)

                    if isinstance(fc, Forecast):
                        predicted = fc.point
                    else:
                        predicted = np.asarray(fc, dtype=np.float64)

                    if len(predicted) != len(test):
                        predicted = predicted[: len(test)]

                    score = _compute_metric(test, predicted, self.metric)
                    fold_scores.append(score)

                    # Per-horizon scores
                    for i in range(min(h, len(test), len(predicted))):
                        horizon_scores[i].append(
                            abs(test[i] - predicted[i])
                        )

            except Exception as e:
                logger.debug("CV fold failed for family '%s': %s", family, e)
                fold_scores.append(np.inf)

            fold_start += step

        # Mean per-horizon
        per_horizon = [
            float(np.mean(hs)) if hs else np.inf for hs in horizon_scores
        ]

        return fold_scores, per_horizon

    def fit(
        self,
        y: pd.Series | NDArray[np.float64],
        cv_horizon: int | None = None,
        m: int = 1,
    ) -> AutoSelectResult:
        """Run cross-family model selection.

        Parameters
        ----------
        y : pd.Series or NDArray
            Time series data.
        cv_horizon : int or None
            Override the cv_horizon parameter. Default None (use constructor value).
        m : int
            Seasonal period (used for AutoARIMA/AutoETS). Default 1.

        Returns
        -------
        AutoSelectResult
            Result with ranking, best model, and CV results.
        """
        y_arr = np.asarray(y, dtype=np.float64)

        if cv_horizon is not None:
            self.cv_horizon = cv_horizon

        if len(y_arr) < self.cv_horizon + 20:
            msg = (
                f"Series too short for AutoSelect CV: {len(y_arr)} observations "
                f"(need at least cv_horizon + 20 = {self.cv_horizon + 20})"
            )
            raise ValueError(msg)

        family_results: list[_FamilyResult] = []
        all_cv_results: dict[str, list[float]] = {}

        for family in self.families:
            logger.info("Evaluating family '%s'...", family)

            try:
                # Cross-validate
                cv_scores, cv_by_horizon = self._cross_validate(y_arr, family, m)
                finite_cv = [s for s in cv_scores if np.isfinite(s)]
                cv_mean = float(np.mean(finite_cv)) if finite_cv else np.inf

                # Fit on full data
                model, model_name, forecast_fn = self._get_best_in_family(
                    y_arr, family, m
                )

                family_results.append(
                    _FamilyResult(
                        family=family,
                        model_name=model_name,
                        model=model,
                        cv_scores=cv_scores,
                        cv_mean=cv_mean,
                        cv_by_horizon=cv_by_horizon,
                        forecast_fn=forecast_fn,
                    )
                )
                all_cv_results[family] = cv_scores

            except Exception as e:
                logger.warning("Failed to evaluate family '%s': %s", family, e)
                family_results.append(
                    _FamilyResult(
                        family=family,
                        model_name=f"{family} (failed)",
                        model=None,
                        cv_scores=[np.inf],
                        cv_mean=np.inf,
                        cv_by_horizon=[np.inf] * self.cv_horizon,
                        forecast_fn=lambda h: None,
                    )
                )
                all_cv_results[family] = [np.inf]

        # Sort by CV mean (ascending)
        family_results.sort(key=lambda fr: fr.cv_mean)

        # Build ranking DataFrame
        ranking_data = []
        for fr in family_results:
            finite_scores = [s for s in fr.cv_scores if np.isfinite(s)]
            ranking_data.append(
                {
                    "family": fr.family,
                    "model_name": fr.model_name,
                    "cv_mean": fr.cv_mean,
                    "cv_std": float(np.std(finite_scores))
                    if finite_scores
                    else np.inf,
                    "n_folds": len(finite_scores),
                }
            )

        ranking = pd.DataFrame(ranking_data)

        best = family_results[0]

        return AutoSelectResult(
            ranking=ranking,
            best_model=best.model,
            best_family=best.family,
            best_model_name=best.model_name,
            all_cv_results=all_cv_results,
            metric_name=self.metric,
            _y=y_arr,
            _family_results=family_results,
        )
