"""ForecastResults - collection of forecasts with comparison tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from forecastbox.core.forecast import Forecast

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class ForecastResults:
    """Collection of forecasts with evaluation and comparison tools.

    Parameters
    ----------
    forecasts : dict[str, Forecast] or None
        Initial map of model_name -> Forecast.
    actual : NDArray[np.float64] or None
        Actual (realized) values.
    """

    def __init__(
        self,
        forecasts: dict[str, Forecast] | None = None,
        actual: np.ndarray | None = None,
    ) -> None:
        self.forecasts: dict[str, Forecast] = forecasts or {}
        self.actual: np.ndarray | None = (
            np.asarray(actual, dtype=np.float64) if actual is not None else None
        )
        self.metrics: dict[str, dict[str, float]] = {}
        self.cv_results: dict | None = None
        self.combination: Forecast | None = None

    def add_forecast(self, name: str, forecast: Forecast) -> None:
        """Add a forecast to the collection.

        Parameters
        ----------
        name : str
            Model name.
        forecast : Forecast
            Forecast object.
        """
        self.forecasts[name] = forecast

    def set_actual(self, actual: np.ndarray) -> None:
        """Set actual (realized) values.

        Parameters
        ----------
        actual : array-like
            Realized values.
        """
        self.actual = np.asarray(actual, dtype=np.float64)

    def evaluate(
        self, metrics: tuple[str, ...] = ("mae", "rmse", "mape", "mase")
    ) -> pd.DataFrame:
        """Calculate metrics for all models.

        Parameters
        ----------
        metrics : tuple[str, ...]
            Metric names to calculate.

        Returns
        -------
        pd.DataFrame
            DataFrame with models as rows and metrics as columns.
        """
        if self.actual is None:
            msg = "Cannot evaluate without actual values. Call set_actual() first."
            raise ValueError(msg)

        from forecastbox.metrics.point_metrics import mae, mape, rmse

        metric_fns = {
            "mae": lambda a, p: mae(a, p),
            "rmse": lambda a, p: rmse(a, p),
            "mape": lambda a, p: mape(a, p),
            "me": lambda a, p: float(np.mean(a - p)),
        }

        results: dict[str, dict[str, float]] = {}
        for name, fc in self.forecasts.items():
            results[name] = {}
            for metric in metrics:
                if metric == "mase":
                    # MASE requires training series -- skip if not available
                    continue
                if metric in metric_fns:
                    results[name][metric] = metric_fns[metric](self.actual, fc.point)

        self.metrics = results
        return pd.DataFrame(results).T

    def rank(self, metric: str = "rmse") -> list[str]:
        """Rank models by a metric (ascending = best first).

        Parameters
        ----------
        metric : str
            Metric name to rank by.

        Returns
        -------
        list[str]
            Model names sorted by metric.
        """
        if not self.metrics:
            self.evaluate()

        scores = {name: m.get(metric, float("inf")) for name, m in self.metrics.items()}
        return sorted(scores, key=lambda n: scores[n])

    def best(self, metric: str = "rmse") -> str:
        """Return the best model name by a metric.

        Parameters
        ----------
        metric : str
            Metric name.

        Returns
        -------
        str
            Name of the best model.
        """
        ranking = self.rank(metric)
        return ranking[0]

    def summary(self) -> str:
        """Generate a formatted summary table.

        Returns
        -------
        str
            Formatted string with model comparison.
        """
        if not self.metrics:
            if self.actual is not None:
                self.evaluate()
            else:
                return "No metrics available. Set actual values and call evaluate()."

        df = pd.DataFrame(self.metrics).T
        lines = [
            "=" * 60,
            "ForecastResults Summary",
            "=" * 60,
            f"Models: {len(self.forecasts)}",
            f"Horizon: {next(iter(self.forecasts.values())).horizon if self.forecasts else 'N/A'}",
            "-" * 60,
            df.to_string(),
            "-" * 60,
        ]

        if self.metrics:
            for metric in df.columns:
                best_name = self.best(metric)
                lines.append(f"Best ({metric}): {best_name}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """All forecasts in long format.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: model, horizon, point, lower_80, upper_80, etc.
        """
        rows = []
        for name, fc in self.forecasts.items():
            for h in range(fc.horizon):
                row: dict[str, object] = {
                    "model": name,
                    "horizon": h + 1,
                    "point": fc.point[h],
                }
                if fc.lower_80 is not None:
                    row["lower_80"] = fc.lower_80[h]
                if fc.upper_80 is not None:
                    row["upper_80"] = fc.upper_80[h]
                if fc.lower_95 is not None:
                    row["lower_95"] = fc.lower_95[h]
                if fc.upper_95 is not None:
                    row["upper_95"] = fc.upper_95[h]
                if self.actual is not None and h < len(self.actual):
                    row["actual"] = self.actual[h]
                rows.append(row)
        return pd.DataFrame(rows)

    def plot_comparison(self, metric: str = "rmse") -> plt.Axes:
        """Bar chart comparing models by a metric.

        Parameters
        ----------
        metric : str
            Metric name to compare.

        Returns
        -------
        plt.Axes
            Matplotlib axes.
        """
        import matplotlib.pyplot as plt

        if not self.metrics:
            self.evaluate()

        names = list(self.metrics.keys())
        values = [self.metrics[n].get(metric, 0.0) for n in names]

        _, ax = plt.subplots(figsize=(10, 5))
        ax.barh(names, values, color="steelblue")
        ax.set_xlabel(metric.upper())
        ax.set_title(f"Model Comparison: {metric.upper()}")
        ax.grid(True, alpha=0.3, axis="x")
        return ax

    def plot_forecasts(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot all forecasts overlaid.

        Parameters
        ----------
        ax : plt.Axes or None
            Axes to plot on.

        Returns
        -------
        plt.Axes
            Matplotlib axes.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        for name, fc in self.forecasts.items():
            x = fc.index if fc.index is not None else np.arange(fc.horizon)
            ax.plot(x, fc.point, label=name, linewidth=1.5)

        if self.actual is not None:
            x_actual = next(iter(self.forecasts.values())).index
            if x_actual is None:
                x_actual = np.arange(len(self.actual))
            ax.plot(x_actual, self.actual, "ko-", label="Actual", linewidth=2)

        ax.set_title("Forecast Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
