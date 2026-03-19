"""RecurringForecast - Scheduled forecast execution with history."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastbox.pipeline.pipeline import ForecastPipeline, PipelineResults


class RecurringForecast:
    """Recurring forecast execution with history tracking.

    Parameters
    ----------
    pipeline : ForecastPipeline
        Pipeline to execute repeatedly.
    frequency : str
        Execution frequency: 'daily', 'weekly', 'monthly', 'quarterly'.
    data_updater : Callable or None
        Function that returns updated data. Called before each execution.
    max_history : int
        Maximum number of executions to keep in history.
    """

    def __init__(
        self,
        pipeline: ForecastPipeline,
        frequency: str = "monthly",
        data_updater: Callable[[], pd.DataFrame | pd.Series] | None = None,
        max_history: int = 100,
    ) -> None:
        self.pipeline = pipeline
        self.frequency = frequency
        self.data_updater = data_updater
        self.max_history = max_history
        self._history: list[dict[str, Any]] = []

    def run_once(self) -> PipelineResults:
        """Execute the pipeline once with current/updated data.

        If data_updater is provided, it is called first to refresh data.

        Returns
        -------
        PipelineResults
            Results from this execution.
        """
        # Update data if updater is provided
        if self.data_updater is not None:
            new_data = self.data_updater()
            self.pipeline.data_source = new_data

        # Run pipeline
        results = self.pipeline.run()

        # Store in history
        entry: dict[str, Any] = {
            "timestamp": datetime.now(),
            "results": results,
            "execution_number": len(self._history) + 1,
        }
        self._history.append(entry)

        # Trim history if needed
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

        return results

    def history(self) -> list[PipelineResults]:
        """Return history of pipeline results.

        Returns
        -------
        list[PipelineResults]
            List of results from past executions, oldest first.
        """
        return [entry["results"] for entry in self._history]

    def last_result(self) -> PipelineResults:
        """Return the most recent execution result.

        Returns
        -------
        PipelineResults
            Last execution result.

        Raises
        ------
        RuntimeError
            If no executions have been performed.
        """
        if not self._history:
            msg = "No executions performed yet. Call run_once() first."
            raise RuntimeError(msg)
        return self._history[-1]["results"]

    def forecast_evolution(self, variable: str | None = None) -> pd.DataFrame:
        """Show how forecasts evolved across executions.

        Parameters
        ----------
        variable : str or None
            Model name to track. If None, uses first model.

        Returns
        -------
        pd.DataFrame
            DataFrame with execution timestamps as index and forecast
            horizons as columns.
        """
        if not self._history:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for entry in self._history:
            results: PipelineResults = entry["results"]
            timestamp = entry["timestamp"]

            # Select which forecast to track
            if variable and variable in results.forecasts:
                fc = results.forecasts[variable]
            elif results.combination is not None:
                fc = results.combination
            elif results.forecasts:
                fc = next(iter(results.forecasts.values()))
            else:
                continue

            row: dict[str, Any] = {"timestamp": timestamp}
            for h in range(fc.horizon):
                row[f"h{h + 1}"] = float(fc.point[h])
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp")
        return df

    def revision_analysis(self) -> pd.DataFrame:
        """Analyze revisions between consecutive executions.

        Returns
        -------
        pd.DataFrame
            DataFrame with revision statistics (mean revision, std, max).
        """
        if len(self._history) < 2:
            return pd.DataFrame()

        revisions: list[dict[str, Any]] = []

        for i in range(1, len(self._history)):
            prev_results: PipelineResults = self._history[i - 1]["results"]
            curr_results: PipelineResults = self._history[i]["results"]

            prev_ts = self._history[i - 1]["timestamp"]
            curr_ts = self._history[i]["timestamp"]

            # Compare first available model
            prev_fc = None
            curr_fc = None

            if prev_results.combination is not None and curr_results.combination is not None:
                prev_fc = prev_results.combination
                curr_fc = curr_results.combination
            else:
                # Use first common model
                common_models = set(prev_results.forecasts.keys()) & set(
                    curr_results.forecasts.keys()
                )
                if common_models:
                    model = sorted(common_models)[0]
                    prev_fc = prev_results.forecasts[model]
                    curr_fc = curr_results.forecasts[model]

            if prev_fc is not None and curr_fc is not None:
                min_len = min(len(prev_fc.point), len(curr_fc.point))
                diff = curr_fc.point[:min_len] - prev_fc.point[:min_len]
                revisions.append(
                    {
                        "from": prev_ts,
                        "to": curr_ts,
                        "mean_revision": float(np.mean(diff)),
                        "std_revision": float(np.std(diff)),
                        "max_abs_revision": float(np.max(np.abs(diff))),
                    }
                )

        return pd.DataFrame(revisions)

    def plot_evolution(self, variable: str | None = None, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot forecast evolution across executions.

        Parameters
        ----------
        variable : str or None
            Model name to track. If None, uses first model.
        ax : matplotlib Axes or None
            Axes to plot on.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        evolution = self.forecast_evolution(variable)
        if evolution.empty:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
            return ax

        for col in evolution.columns:
            ax.plot(evolution.index, evolution[col], marker="o", label=col, markersize=4)

        ax.set_title("Forecast Evolution Over Time")
        ax.set_xlabel("Execution Date")
        ax.set_ylabel("Forecast Value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        return ax

    def __repr__(self) -> str:
        return (
            f"RecurringForecast(frequency='{self.frequency}', "
            f"executions={len(self._history)})"
        )
