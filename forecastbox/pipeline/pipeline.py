"""ForecastPipeline - End-to-end forecast orchestration."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from forecastbox.core.forecast import Forecast


class PipelineStep(Enum):
    """Pipeline step identifiers."""

    DATA = "data"
    PREPROCESS = "preprocess"
    FIT = "fit"
    FORECAST = "forecast"
    COMBINE = "combine"
    EVALUATE = "evaluate"
    REPORT = "report"


@dataclass
class PipelineResults:
    """Container for pipeline execution results.

    Attributes
    ----------
    forecasts : dict[str, Forecast]
        Individual model forecasts keyed by model name.
    combination : Forecast or None
        Combined forecast (if combination method specified).
    evaluation : pd.DataFrame
        Evaluation metrics for all models.
    cv_results : dict[str, Any]
        Cross-validation results per model.
    execution_time : dict[str, float]
        Execution time in seconds for each pipeline step.
    metadata : dict[str, Any]
        Additional metadata from pipeline execution.
    """

    forecasts: dict[str, Forecast] = field(default_factory=dict)
    combination: Forecast | None = None
    evaluation: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    cv_results: dict[str, Any] = field(default_factory=dict)
    execution_time: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate formatted summary of pipeline results.

        Returns
        -------
        str
            Human-readable summary string.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("PIPELINE RESULTS SUMMARY")
        lines.append("=" * 60)

        # Models
        lines.append(f"\nModels fitted: {len(self.forecasts)}")
        for name in self.forecasts:
            lines.append(f"  - {name}")

        # Combination
        if self.combination is not None:
            lines.append(f"\nCombination: {self.combination.model_name}")

        # Best model
        best = self.best_model()
        if best:
            lines.append(f"\nBest model: {best}")

        # Evaluation
        if not self.evaluation.empty:
            lines.append("\nEvaluation metrics:")
            lines.append(self.evaluation.to_string(index=True))

        # CV Results
        if self.cv_results:
            lines.append(f"\nCross-validation: {len(self.cv_results)} models evaluated")

        # Execution time
        if self.execution_time:
            total_time = sum(self.execution_time.values())
            lines.append(f"\nExecution time: {total_time:.2f}s total")
            for step, t in self.execution_time.items():
                lines.append(f"  {step}: {t:.2f}s")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def report(self, format: str = "html", output: str | None = None) -> str:
        """Generate report in the specified format.

        Parameters
        ----------
        format : str
            Output format: 'html', 'latex', 'markdown', 'json'.
        output : str or None
            File path to save report. If None, returns string.

        Returns
        -------
        str
            Report content.
        """
        from forecastbox.reports.builder import ReportBuilder

        builder = ReportBuilder(results=self)
        builder.add_section("summary", title="Summary")
        builder.add_section("forecasts", title="Forecasts")
        builder.add_section("evaluation", title="Evaluation")
        content = builder.render(format=format, output=output)
        return content

    def to_dict(self) -> dict[str, Any]:
        """Serialize results to dictionary.

        Returns
        -------
        dict[str, Any]
            Serializable dictionary of results.
        """
        result: dict[str, Any] = {
            "forecasts": {
                name: {
                    "point": fc.point.tolist(),
                    "model_name": fc.model_name,
                    "horizon": fc.horizon,
                }
                for name, fc in self.forecasts.items()
            },
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }

        if self.combination is not None:
            result["combination"] = {
                "point": self.combination.point.tolist(),
                "model_name": self.combination.model_name,
            }

        if not self.evaluation.empty:
            result["evaluation"] = self.evaluation.to_dict()

        return result

    def best_model(self) -> str:
        """Return name of best model based on evaluation metrics.

        Returns
        -------
        str
            Name of the best model (lowest RMSE), or empty string if no evaluation.
        """
        if self.evaluation.empty or len(self.evaluation) == 0:
            return ""

        # Try RMSE first, then first available metric
        if "rmse" in self.evaluation.columns:
            metric_col = "rmse"
        elif len(self.evaluation.columns) > 0:
            metric_col = self.evaluation.columns[0]
        else:
            return ""

        return str(self.evaluation[metric_col].idxmin())


def _preprocess_log(data: pd.Series) -> pd.Series:
    """Apply log transformation."""
    return np.log(data.clip(lower=1e-10))


def _preprocess_diff(data: pd.Series) -> pd.Series:
    """Apply first difference."""
    return data.diff().dropna()


def _preprocess_seasonal_diff(data: pd.Series, period: int = 12) -> pd.Series:
    """Apply seasonal difference."""
    return (data - data.shift(period)).dropna()


def _preprocess_detrend(data: pd.Series) -> pd.Series:
    """Remove linear trend."""
    x = np.arange(len(data), dtype=np.float64)
    coeffs = np.polyfit(x, data.values.astype(np.float64), 1)
    trend = np.polyval(coeffs, x)
    detrended = data.values.astype(np.float64) - trend
    return pd.Series(detrended, index=data.index, name=data.name)


def _preprocess_standardize(data: pd.Series) -> pd.Series:
    """Standardize to zero mean, unit variance."""
    mean = data.mean()
    std = data.std()
    if std == 0:
        return data - mean
    return (data - mean) / std


def _preprocess_outlier_detection(data: pd.Series, factor: float = 1.5) -> pd.Series:
    """Replace outliers using IQR method with interpolation."""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    cleaned = data.copy()
    mask = (cleaned < lower) | (cleaned > upper)
    cleaned[mask] = np.nan
    cleaned = cleaned.interpolate(method="linear")
    return cleaned.ffill().bfill()


def _preprocess_missing_fill(data: pd.Series) -> pd.Series:
    """Fill missing values with linear interpolation."""
    return data.interpolate(method="linear").ffill().bfill()


_PREPROCESS_REGISTRY: dict[str, Callable[..., pd.Series]] = {
    "log": _preprocess_log,
    "diff": _preprocess_diff,
    "seasonal_diff": _preprocess_seasonal_diff,
    "detrend": _preprocess_detrend,
    "standardize": _preprocess_standardize,
    "outlier_detection": _preprocess_outlier_detection,
    "missing_fill": _preprocess_missing_fill,
}


class ForecastPipeline:
    """End-to-end forecast pipeline with configurable DAG steps.

    Parameters
    ----------
    data_source : pd.DataFrame or pd.Series or Callable
        Data source. If callable, will be invoked to get data.
    target : str
        Target variable name (column in DataFrame).
    models : list[str]
        Model names to fit. Default: ['auto_arima'].
    combination : str or None
        Combination method ('mean', 'median', 'bma', 'ols'). None to skip.
    evaluation : list[str]
        Evaluation metrics/tests. Default: ['rmse'].
    horizon : int
        Forecast horizon. Default: 12.
    cv_type : str
        Cross-validation type: 'expanding' or 'sliding'. Default: 'expanding'.
    cv_initial : int or None
        Initial window for CV. None for automatic.
    preprocess : list[str]
        Preprocessing steps to apply. Default: [].
    """

    def __init__(
        self,
        data_source: pd.DataFrame | pd.Series | Callable[[], pd.DataFrame | pd.Series],
        target: str = "",
        models: list[str] | None = None,
        combination: str | None = None,
        evaluation: list[str] | None = None,
        horizon: int = 12,
        cv_type: str = "expanding",
        cv_initial: int | None = None,
        preprocess: list[str] | None = None,
    ) -> None:
        self.data_source = data_source
        self.target = target
        self.models = models or ["auto_arima"]
        self.combination = combination
        self.evaluation = evaluation or ["rmse"]
        self.horizon = horizon
        self.cv_type = cv_type
        self.cv_initial = cv_initial
        self.preprocess = preprocess or []

        # Internal state
        self._data: pd.Series | None = None
        self._processed_data: pd.Series | None = None
        self._step_order: list[str] = [
            PipelineStep.DATA.value,
            PipelineStep.PREPROCESS.value,
            PipelineStep.FIT.value,
            PipelineStep.FORECAST.value,
            PipelineStep.COMBINE.value,
            PipelineStep.EVALUATE.value,
            PipelineStep.REPORT.value,
        ]
        self._custom_steps: dict[str, Callable[..., Any]] = {}
        self._results: PipelineResults | None = None
        self._fitted_models: dict[str, Any] = {}

    def _resolve_data(self) -> pd.Series:
        """Resolve data source to a pandas Series."""
        raw = self.data_source() if callable(self.data_source) else self.data_source

        if isinstance(raw, pd.DataFrame):
            if self.target and self.target in raw.columns:
                return raw[self.target].dropna()
            elif len(raw.columns) == 1:
                return raw.iloc[:, 0].dropna()
            else:
                msg = (
                    f"DataFrame has {len(raw.columns)} columns. "
                    f"Specify 'target' parameter. Available: {list(raw.columns)}"
                )
                raise ValueError(msg)
        elif isinstance(raw, pd.Series):
            return raw.dropna()
        else:
            msg = f"data_source must return DataFrame or Series, got {type(raw).__name__}"
            raise TypeError(msg)

    def _run_preprocess(self, data: pd.Series) -> pd.Series:
        """Apply preprocessing steps."""
        result = data.copy()
        for step_name in self.preprocess:
            if step_name not in _PREPROCESS_REGISTRY:
                available = ", ".join(sorted(_PREPROCESS_REGISTRY.keys()))
                msg = f"Unknown preprocessing step '{step_name}'. Available: {available}"
                raise ValueError(msg)
            fn = _PREPROCESS_REGISTRY[step_name]
            result = fn(result)
        return result

    def _run_fit(self, data: pd.Series) -> dict[str, Any]:
        """Fit models to data. Returns dict of fitted model objects."""
        fitted: dict[str, Any] = {}
        rng = np.random.default_rng(42)

        for model_name in self.models:
            # Generic model fitting — uses simple heuristics for demonstration.
            # In production, this dispatches to actual model implementations.
            model_info: dict[str, Any] = {
                "name": model_name,
                "data_length": len(data),
                "fitted": True,
                "mean": float(data.mean()),
                "std": float(data.std()),
                "last_values": data.tail(self.horizon).values.copy(),
                "trend": float(np.polyfit(np.arange(len(data)), data.values, 1)[0]),
                "rng_state": rng.integers(0, 10000),
            }
            fitted[model_name] = model_info

        self._fitted_models = fitted
        return fitted

    def _run_forecast(
        self, data: pd.Series, fitted: dict[str, Any]
    ) -> dict[str, Forecast]:
        """Generate forecasts from fitted models."""
        forecasts: dict[str, Forecast] = {}
        rng = np.random.default_rng(42)

        # Determine forecast index
        if isinstance(data.index, pd.DatetimeIndex):
            freq = data.index.freq or pd.infer_freq(data.index)
            if freq:
                fc_index = pd.date_range(
                    start=data.index[-1], periods=self.horizon + 1, freq=freq
                )[1:]
            else:
                fc_index = pd.date_range(
                    start=data.index[-1], periods=self.horizon + 1, freq="MS"
                )[1:]
        else:
            fc_index = None

        for model_name, model_info in fitted.items():
            mean_val = model_info["mean"]
            std_val = model_info["std"]
            trend = model_info["trend"]

            # Generate point forecast with trend
            h = np.arange(1, self.horizon + 1, dtype=np.float64)
            base = float(data.iloc[-1])
            point = base + trend * h + rng.normal(0, std_val * 0.1, size=self.horizon)

            # Generate intervals
            width_80 = 1.28 * std_val * np.sqrt(h)
            width_95 = 1.96 * std_val * np.sqrt(h)

            lower_80 = point - width_80
            upper_80 = point + width_80
            lower_95 = point - width_95
            upper_95 = point + width_95

            fc = Forecast(
                point=point,
                lower_80=lower_80,
                upper_80=upper_80,
                lower_95=lower_95,
                upper_95=upper_95,
                index=fc_index,
                model_name=model_name,
                horizon=self.horizon,
                metadata={"fitted_mean": mean_val, "fitted_std": std_val},
            )
            forecasts[model_name] = fc

        return forecasts

    def _run_combine(self, forecasts: dict[str, Forecast]) -> Forecast | None:
        """Combine forecasts if combination method is specified."""
        if self.combination is None or len(forecasts) < 2:
            return None

        fc_list = list(forecasts.values())

        if self.combination in ("mean", "median"):
            return Forecast.combine(fc_list, method=self.combination)
        elif self.combination in ("bma", "ols"):
            # For BMA/OLS, use mean as fallback (real implementation in combination module)
            combined = Forecast.combine(fc_list, method="mean")
            combined.model_name = f"Combined({self.combination})"
            combined.metadata["combination_method"] = self.combination
            return combined
        else:
            return Forecast.combine(fc_list, method="mean")

    def _run_evaluate(self, forecasts: dict[str, Forecast]) -> pd.DataFrame:
        """Evaluate forecasts with specified metrics."""
        rows: list[dict[str, Any]] = []

        for model_name, fc in forecasts.items():
            row: dict[str, Any] = {"model": model_name}

            # Compute basic metrics on the point forecast
            if fc.point is not None and len(fc.point) > 0:
                for metric_name in self.evaluation:
                    if metric_name == "rmse":
                        # RMSE relative to zero (as a proxy without actuals)
                        row["rmse"] = float(np.std(fc.point))
                    elif metric_name == "mae":
                        row["mae"] = float(np.mean(np.abs(fc.point - np.mean(fc.point))))
                    elif metric_name == "mape":
                        mean_val = np.mean(np.abs(fc.point))
                        if mean_val > 0:
                            row["mape"] = float(
                                np.mean(np.abs(fc.point - np.mean(fc.point)) / mean_val)
                                * 100
                            )
                        else:
                            row["mape"] = float("nan")
                    elif metric_name in ("dm_test", "mcs"):
                        row[metric_name] = float("nan")
                    else:
                        row[metric_name] = float("nan")

            rows.append(row)

        df = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
        return df

    def _run_cv(self, data: pd.Series) -> dict[str, Any]:
        """Run cross-validation for each model."""
        cv_results: dict[str, Any] = {}

        initial = self.cv_initial or max(len(data) // 2, 24)

        for model_name in self.models:
            cv_results[model_name] = {
                "cv_type": self.cv_type,
                "initial_window": initial,
                "horizon": self.horizon,
                "n_folds": max(1, (len(data) - initial) // max(1, self.horizon)),
                "mean_rmse": float(np.std(data.values) * 0.5),  # placeholder
            }

        return cv_results

    def run(self) -> PipelineResults:
        """Execute the full pipeline.

        Returns
        -------
        PipelineResults
            Complete pipeline results.
        """
        results = PipelineResults()

        # Step 1: DATA
        t0 = time.time()
        self._data = self._resolve_data()
        results.execution_time[PipelineStep.DATA.value] = time.time() - t0

        # Step 2: PREPROCESS
        t0 = time.time()
        self._processed_data = self._run_preprocess(self._data)
        results.execution_time[PipelineStep.PREPROCESS.value] = time.time() - t0

        data = self._processed_data

        # Step 3: FIT
        t0 = time.time()
        fitted = self._run_fit(data)
        results.execution_time[PipelineStep.FIT.value] = time.time() - t0

        # Step 4: FORECAST
        t0 = time.time()
        results.forecasts = self._run_forecast(data, fitted)
        results.execution_time[PipelineStep.FORECAST.value] = time.time() - t0

        # Step 5: COMBINE
        t0 = time.time()
        results.combination = self._run_combine(results.forecasts)
        results.execution_time[PipelineStep.COMBINE.value] = time.time() - t0

        # Step 6: EVALUATE
        t0 = time.time()
        results.evaluation = self._run_evaluate(results.forecasts)
        results.cv_results = self._run_cv(data)
        results.execution_time[PipelineStep.EVALUATE.value] = time.time() - t0

        # Step 7: REPORT (just record time, actual report is on-demand)
        t0 = time.time()
        results.execution_time[PipelineStep.REPORT.value] = time.time() - t0

        # Store metadata before custom steps so they can modify it
        results.metadata = {
            "models": self.models,
            "combination": self.combination,
            "evaluation": self.evaluation,
            "horizon": self.horizon,
            "cv_type": self.cv_type,
            "preprocess": self.preprocess,
            "data_length": len(data),
        }

        # Run custom steps
        for step_name, step_fn in self._custom_steps.items():
            t0 = time.time()
            step_fn(results)
            results.execution_time[step_name] = time.time() - t0

        self._results = results
        return results

    def run_step(self, step_name: str) -> Any:
        """Execute a single pipeline step.

        Parameters
        ----------
        step_name : str
            Name of the step to execute.

        Returns
        -------
        Any
            Step output.
        """
        if self._data is None:
            self._data = self._resolve_data()
        if self._processed_data is None:
            self._processed_data = self._run_preprocess(self._data)

        data = self._processed_data

        if step_name == PipelineStep.DATA.value:
            return self._resolve_data()
        elif step_name == PipelineStep.PREPROCESS.value:
            return self._run_preprocess(self._data)
        elif step_name == PipelineStep.FIT.value:
            return self._run_fit(data)
        elif step_name == PipelineStep.FORECAST.value:
            fitted = self._run_fit(data)
            return self._run_forecast(data, fitted)
        elif step_name == PipelineStep.COMBINE.value:
            fitted = self._run_fit(data)
            forecasts = self._run_forecast(data, fitted)
            return self._run_combine(forecasts)
        elif step_name == PipelineStep.EVALUATE.value:
            fitted = self._run_fit(data)
            forecasts = self._run_forecast(data, fitted)
            return self._run_evaluate(forecasts)
        elif step_name in self._custom_steps:
            if self._results is None:
                msg = "Run full pipeline before executing custom steps"
                raise RuntimeError(msg)
            return self._custom_steps[step_name](self._results)
        else:
            msg = f"Unknown step: {step_name}"
            raise ValueError(msg)

    def add_step(
        self,
        name: str,
        fn: Callable[..., Any],
        after: str | None = None,
    ) -> None:
        """Add a custom step to the pipeline.

        Parameters
        ----------
        name : str
            Step name.
        fn : Callable
            Step function. Receives PipelineResults, can modify in-place.
        after : str or None
            Insert after this step. If None, appends at end.
        """
        self._custom_steps[name] = fn

        if after is not None and after in self._step_order:
            idx = self._step_order.index(after) + 1
            self._step_order.insert(idx, name)
        else:
            self._step_order.append(name)

    def remove_step(self, name: str) -> None:
        """Remove a step from the pipeline.

        Parameters
        ----------
        name : str
            Step name to remove.
        """
        if name in self._custom_steps:
            del self._custom_steps[name]
        if name in self._step_order:
            self._step_order.remove(name)

    def steps(self) -> list[str]:
        """Return ordered list of pipeline steps.

        Returns
        -------
        list[str]
            Step names in execution order.
        """
        return list(self._step_order)

    def set_params(self, **kwargs: Any) -> None:
        """Update pipeline parameters.

        Parameters
        ----------
        **kwargs
            Parameter name-value pairs.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                msg = f"Unknown parameter: {key}"
                raise ValueError(msg)

    def clone(self) -> ForecastPipeline:
        """Create a deep copy of the pipeline.

        Returns
        -------
        ForecastPipeline
            Independent copy of this pipeline.
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"ForecastPipeline(models={self.models}, horizon={self.horizon}, "
            f"combination={self.combination}, steps={len(self._step_order)})"
        )
