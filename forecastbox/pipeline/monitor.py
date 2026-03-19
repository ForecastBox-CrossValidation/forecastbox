"""ForecastMonitor - Continuous accuracy monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastbox.pipeline.pipeline import ForecastPipeline


@dataclass
class MonitorReport:
    """Report from ForecastMonitor accuracy assessment.

    Attributes
    ----------
    overall_metrics : dict[str, float]
        Overall metrics (RMSE, MAE, MAPE, bias).
    rolling_rmse : pd.Series
        Rolling RMSE over time.
    rolling_mae : pd.Series
        Rolling MAE over time.
    bias : float
        Mean forecast error (positive = over-prediction).
    hit_rate : float
        Fraction of actuals within 95% prediction interval.
    """

    overall_metrics: dict[str, float] = field(default_factory=dict)
    rolling_rmse: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rolling_mae: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    bias: float = 0.0
    hit_rate: float = 0.0

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines: list[str] = []
        lines.append("=" * 50)
        lines.append("FORECAST MONITOR REPORT")
        lines.append("=" * 50)

        if self.overall_metrics:
            lines.append("\nOverall Metrics:")
            for name, value in self.overall_metrics.items():
                lines.append(f"  {name}: {value:.4f}")

        lines.append(f"\nBias (MFE): {self.bias:.4f}")
        lines.append(f"Hit Rate (95% CI): {self.hit_rate:.1%}")

        if not self.rolling_rmse.empty:
            lines.append(f"\nRolling RMSE (latest): {self.rolling_rmse.iloc[-1]:.4f}")
        if not self.rolling_mae.empty:
            lines.append(f"Rolling MAE (latest): {self.rolling_mae.iloc[-1]:.4f}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)


class ForecastMonitor:
    """Continuous monitoring of forecast accuracy.

    Parameters
    ----------
    pipeline : ForecastPipeline
        Pipeline being monitored.
    """

    def __init__(self, pipeline: ForecastPipeline) -> None:
        self.pipeline = pipeline
        self.actuals: list[tuple[datetime | pd.Timestamp, float]] = []
        self.forecasted: list[
            tuple[datetime | pd.Timestamp, float, float | None, float | None]
        ] = []
        self._history: list[dict[str, Any]] = []

    def add_actual(self, date: datetime | pd.Timestamp, value: float) -> None:
        """Add a single realized value.

        Parameters
        ----------
        date : datetime or pd.Timestamp
            Date of the realization.
        value : float
            Actual realized value.
        """
        self.actuals.append((pd.Timestamp(date), float(value)))
        self.actuals.sort(key=lambda x: x[0])

    def add_actuals(self, data: pd.Series) -> None:
        """Add multiple realized values from a Series.

        Parameters
        ----------
        data : pd.Series
            Series with DatetimeIndex and actual values.
        """
        for date, value in data.items():
            self.add_actual(date, float(value))

    def add_forecast(
        self,
        date: datetime | pd.Timestamp,
        point: float,
        lower_95: float | None = None,
        upper_95: float | None = None,
    ) -> None:
        """Add a forecasted value for comparison.

        Parameters
        ----------
        date : datetime or pd.Timestamp
            Forecast target date.
        point : float
            Point forecast.
        lower_95 : float or None
            Lower bound of 95% CI.
        upper_95 : float or None
            Upper bound of 95% CI.
        """
        self.forecasted.append((pd.Timestamp(date), float(point), lower_95, upper_95))
        self.forecasted.sort(key=lambda x: x[0])

    def _get_matched_pairs(self) -> pd.DataFrame:
        """Match forecasted and actual values by date."""
        if not self.actuals or not self.forecasted:
            return pd.DataFrame()

        actual_dict = {ts: val for ts, val in self.actuals}
        rows: list[dict[str, Any]] = []
        for ts, point, lower, upper in self.forecasted:
            if ts in actual_dict:
                rows.append(
                    {
                        "date": ts,
                        "forecast": point,
                        "actual": actual_dict[ts],
                        "error": actual_dict[ts] - point,
                        "lower_95": lower,
                        "upper_95": upper,
                    }
                )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("date").sort_index()

    def accuracy_report(self) -> MonitorReport:
        """Generate comprehensive accuracy report.

        Returns
        -------
        MonitorReport
            Report with overall metrics, rolling metrics, bias, and hit rate.
        """
        pairs = self._get_matched_pairs()

        if pairs.empty:
            return MonitorReport()

        errors = pairs["error"].values.astype(np.float64)
        actuals_arr = pairs["actual"].values.astype(np.float64)

        # Overall metrics
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        mfe = float(np.mean(errors))

        # MAPE (avoid division by zero)
        nonzero = np.abs(actuals_arr) > 1e-10
        if np.any(nonzero):
            mape = float(np.mean(np.abs(errors[nonzero] / actuals_arr[nonzero])) * 100)
        else:
            mape = float("nan")

        overall: dict[str, float] = {
            "rmse": rmse,
            "mae": mae,
            "mfe": mfe,
            "mape": mape,
        }

        # Rolling metrics (window=min(12, len))
        window = min(12, len(errors))
        error_series = pd.Series(errors, index=pairs.index)

        rolling_rmse = error_series.rolling(window).apply(
            lambda x: float(np.sqrt(np.mean(x**2))), raw=True
        ).dropna()

        rolling_mae = error_series.abs().rolling(window).mean().dropna()

        # Hit rate
        hit_rate = 0.0
        if "lower_95" in pairs.columns and "upper_95" in pairs.columns:
            valid = pairs.dropna(subset=["lower_95", "upper_95"])
            if len(valid) > 0:
                inside = (valid["actual"] >= valid["lower_95"]) & (
                    valid["actual"] <= valid["upper_95"]
                )
                hit_rate = float(inside.mean())

        report = MonitorReport(
            overall_metrics=overall,
            rolling_rmse=rolling_rmse,
            rolling_mae=rolling_mae,
            bias=mfe,
            hit_rate=hit_rate,
        )

        return report

    def rolling_accuracy(self, window: int = 12, metric: str = "rmse") -> pd.Series:
        """Calculate rolling accuracy metric.

        Parameters
        ----------
        window : int
            Rolling window size.
        metric : str
            Metric to compute: 'rmse', 'mae', 'mape'.

        Returns
        -------
        pd.Series
            Rolling metric values.
        """
        pairs = self._get_matched_pairs()
        if pairs.empty:
            return pd.Series(dtype=float)

        errors = pd.Series(
            pairs["actual"].values - pairs["forecast"].values,
            index=pairs.index,
        )

        if metric == "rmse":
            return errors.rolling(window).apply(
                lambda x: float(np.sqrt(np.mean(x**2))), raw=True
            ).dropna()
        elif metric == "mae":
            return errors.abs().rolling(window).mean().dropna()
        elif metric == "mape":
            actuals_s = pairs["actual"]
            pct_errors = (errors / actuals_s.where(actuals_s.abs() > 1e-10)).abs() * 100
            return pct_errors.rolling(window).mean().dropna()
        else:
            msg = f"Unknown metric: {metric}. Use 'rmse', 'mae', or 'mape'."
            raise ValueError(msg)

    def cumulative_accuracy(self, metric: str = "rmse") -> pd.Series:
        """Calculate cumulative accuracy metric.

        Parameters
        ----------
        metric : str
            Metric to compute: 'rmse', 'mae'.

        Returns
        -------
        pd.Series
            Cumulative metric values.
        """
        pairs = self._get_matched_pairs()
        if pairs.empty:
            return pd.Series(dtype=float)

        errors = pd.Series(
            pairs["actual"].values - pairs["forecast"].values,
            index=pairs.index,
        )

        if metric == "rmse":
            return errors.expanding().apply(
                lambda x: float(np.sqrt(np.mean(x**2))), raw=True
            ).dropna()
        elif metric == "mae":
            return errors.abs().expanding().mean().dropna()
        else:
            msg = f"Unknown metric: {metric}. Use 'rmse' or 'mae'."
            raise ValueError(msg)

    def bias_tracker(self) -> pd.Series:
        """Track bias evolution over time.

        Returns
        -------
        pd.Series
            Cumulative mean forecast error.
        """
        pairs = self._get_matched_pairs()
        if pairs.empty:
            return pd.Series(dtype=float)

        errors = pd.Series(
            pairs["actual"].values - pairs["forecast"].values,
            index=pairs.index,
        )
        return errors.expanding().mean().dropna()

    def degradation_test(self, window: int = 12, threshold: float | None = None) -> bool:
        """Test whether forecast accuracy has degraded.

        Compares recent window RMSE to historical RMSE. If recent RMSE
        exceeds historical by more than threshold, returns True.

        Parameters
        ----------
        window : int
            Window size for recent period.
        threshold : float or None
            Multiplier threshold. Default: 1.5 (50% degradation).

        Returns
        -------
        bool
            True if degradation detected.
        """
        if threshold is None:
            threshold = 1.5

        pairs = self._get_matched_pairs()
        if len(pairs) < window * 2:
            return False

        errors = pairs["actual"].values - pairs["forecast"].values

        recent_rmse = float(np.sqrt(np.mean(errors[-window:] ** 2)))
        historical_rmse = float(np.sqrt(np.mean(errors[:-window] ** 2)))

        if historical_rmse < 1e-10:
            return False

        return bool(recent_rmse > threshold * historical_rmse)

    def plot_accuracy_evolution(self, metric: str = "rmse", ax: plt.Axes | None = None) -> plt.Axes:
        """Plot accuracy metric evolution over time.

        Parameters
        ----------
        metric : str
            Metric to plot.
        ax : matplotlib Axes or None
            Axes to plot on.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        rolling = self.rolling_accuracy(metric=metric)
        cumulative = self.cumulative_accuracy(metric=metric)

        if not rolling.empty:
            ax.plot(rolling.index, rolling.values, label=f"Rolling {metric.upper()}", linewidth=2)
        if not cumulative.empty:
            ax.plot(
                cumulative.index, cumulative.values,
                label=f"Cumulative {metric.upper()}", linestyle="--", linewidth=1.5,
            )

        ax.set_title(f"Accuracy Evolution ({metric.upper()})")
        ax.set_xlabel("Date")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_forecast_vs_actual(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot forecasts vs actual values.

        Parameters
        ----------
        ax : matplotlib Axes or None
            Axes to plot on.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        pairs = self._get_matched_pairs()
        if pairs.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return ax

        ax.plot(pairs.index, pairs["actual"], "ko-", label="Actual", linewidth=1.5)
        ax.plot(pairs.index, pairs["forecast"], "b-", label="Forecast", linewidth=2)

        if "lower_95" in pairs.columns and "upper_95" in pairs.columns:
            valid = pairs.dropna(subset=["lower_95", "upper_95"])
            if len(valid) > 0:
                ax.fill_between(
                    valid.index, valid["lower_95"], valid["upper_95"],
                    alpha=0.15, color="blue", label="95% CI",
                )

        ax.set_title("Forecast vs Actual")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def __repr__(self) -> str:
        return (
            f"ForecastMonitor(actuals={len(self.actuals)}, "
            f"forecasts={len(self.forecasted)})"
        )
