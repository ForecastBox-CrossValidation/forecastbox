"""ForecastPlotter - Main visualization class."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastbox.core.forecast import Forecast
from forecastbox.pipeline.pipeline import PipelineResults
from forecastbox.viz._style import set_nodesecon_style
from forecastbox.viz.eval_plots import accuracy_plot, cv_plot, residual_plot
from forecastbox.viz.forecast_plots import comparison_plot, fan_chart, forecast_plot
from forecastbox.viz.nowcast_plots import nowcast_plot
from forecastbox.viz.pipeline_plots import combination_weights_plot, pipeline_dashboard
from forecastbox.viz.scenario_plots import scenario_plot


class ForecastPlotter:
    """Main visualization class for forecastbox results.

    Parameters
    ----------
    results : PipelineResults
        Pipeline results to visualize.
    style : str
        Style preset: 'nodesecon' (default) or 'default'.
    """

    def __init__(
        self,
        results: PipelineResults,
        style: str = "nodesecon",
    ) -> None:
        self.results = results
        self.style = style

        if style == "nodesecon":
            set_nodesecon_style()

    def forecast_plot(
        self,
        model: str | None = None,
        actual: np.ndarray | pd.Series | None = None,
        history: pd.Series | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot forecast for a specific model.

        Parameters
        ----------
        model : str or None
            Model name. If None, uses first model or combined.
        actual : array-like or None
            Actual values.
        history : pd.Series or None
            Historical data.
        ax : plt.Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        fc = self._get_forecast(model)
        return forecast_plot(fc, actual=actual, history=history, ax=ax, title=title)

    def fan_chart(
        self,
        model: str | None = None,
        history: pd.Series | None = None,
        history_periods: int = 36,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Create fan chart for a model.

        Parameters
        ----------
        model : str or None
            Model name.
        history : pd.Series or None
            Historical data.
        history_periods : int
            Periods of history to show.
        ax : plt.Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        fc = self._get_forecast(model)
        return fan_chart(
            fc, history=history, history_periods=history_periods,
            ax=ax, title=title,
        )

    def comparison_plot(
        self,
        models: list[str] | None = None,
        actual: np.ndarray | pd.Series | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot multiple models overlaid.

        Parameters
        ----------
        models : list[str] or None
            Model names. If None, uses all.
        actual : array-like or None
            Actual values.
        ax : plt.Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        if models is None:
            forecasts = self.results.forecasts
        else:
            forecasts = {
                m: self.results.forecasts[m] for m in models
                if m in self.results.forecasts
            }
        return comparison_plot(forecasts, actual=actual, ax=ax, title=title)

    def accuracy_plot(
        self,
        metric: str = "rmse",
        by: str = "model",
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot accuracy metrics.

        Parameters
        ----------
        metric : str
            Metric name.
        by : str
            Grouping: 'model' or 'horizon'.
        ax : plt.Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        return accuracy_plot(
            self.results.evaluation, metric=metric,
            by=by, ax=ax, title=title,
        )

    def scenario_plot(
        self,
        variable: str | None = None,
        scenarios: dict[str, Forecast | np.ndarray] | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot scenario comparison.

        Parameters
        ----------
        variable : str or None
            Variable name (unused, for API compatibility).
        scenarios : dict or None
            Scenarios to plot. If None, uses forecasts as scenarios.
        ax : plt.Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        if scenarios is None:
            scenarios = dict(self.results.forecasts)
        return scenario_plot(scenarios, ax=ax, title=title)

    def nowcast_plot(
        self,
        target: str | None = None,
        nowcasts: dict[str, float] | pd.Series | None = None,
        vintages: list[str] | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot nowcast evolution.

        Parameters
        ----------
        target : str or None
            Target variable name.
        nowcasts : dict or Series or None
            Nowcast values by vintage. If None, uses forecast point values.
        vintages : list[str] or None
            Vintages to highlight.
        ax : plt.Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        if nowcasts is None:
            # Use first model point forecasts as proxy
            fc = next(iter(self.results.forecasts.values()))
            nowcasts = {f"h{i + 1}": float(fc.point[i]) for i in range(fc.horizon)}
        return nowcast_plot(nowcasts, vintages=vintages, ax=ax, title=title)

    def combination_weights_plot(
        self,
        weights: dict[str, float] | pd.Series | pd.DataFrame | None = None,
        method: str | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot combination weights.

        Parameters
        ----------
        weights : dict, Series, DataFrame, or None
            Weights to plot. If None, uses equal weights.
        method : str or None
            Combination method name (for title).
        ax : plt.Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        if weights is None:
            n = len(self.results.forecasts)
            weights = {name: 1.0 / n for name in self.results.forecasts} if n > 0 else {}
        return combination_weights_plot(weights, ax=ax, title=title)

    def cv_plot(
        self,
        model: str | None = None,
        metric: str = "rmse",
        ax: plt.Axes | None = None,
        title: str | None = None,
    ) -> plt.Axes:
        """Plot cross-validation results.

        Parameters
        ----------
        model : str or None
            Model name.
        metric : str
            Metric to display.
        ax : plt.Axes or None
            Axes to plot on.
        title : str or None
            Plot title.

        Returns
        -------
        plt.Axes
        """
        cv_data = self.results.cv_results
        if model and model in cv_data:
            cv_data = {model: cv_data[model]}
        return cv_plot(cv_data, metric=metric, ax=ax, title=title)

    def residual_plot(
        self,
        model: str | None = None,
        fig: plt.Figure | None = None,
    ) -> plt.Figure:
        """Create residual diagnostic plot.

        Parameters
        ----------
        model : str or None
            Model name.
        fig : plt.Figure or None
            Figure to use.

        Returns
        -------
        plt.Figure
        """
        fc = self._get_forecast(model)
        # Use point forecast deviations from mean as residuals (proxy)
        residuals = fc.point - np.mean(fc.point)
        return residual_plot(residuals, model_name=fc.model_name, fig=fig)

    def pipeline_dashboard(self, fig: plt.Figure | None = None) -> plt.Figure:
        """Create pipeline summary dashboard.

        Parameters
        ----------
        fig : plt.Figure or None
            Figure to use.

        Returns
        -------
        plt.Figure
        """
        return pipeline_dashboard(self.results, fig=fig)

    def _get_forecast(self, model: str | None = None) -> Forecast:
        """Get forecast by model name.

        Parameters
        ----------
        model : str or None
            Model name. If None, returns combined or first.

        Returns
        -------
        Forecast
        """
        if model and model in self.results.forecasts:
            return self.results.forecasts[model]
        if self.results.combination is not None:
            return self.results.combination
        if self.results.forecasts:
            return next(iter(self.results.forecasts.values()))
        msg = "No forecasts available"
        raise ValueError(msg)

    def __repr__(self) -> str:
        return (
            f"ForecastPlotter(models={len(self.results.forecasts)}, "
            f"style='{self.style}')"
        )
