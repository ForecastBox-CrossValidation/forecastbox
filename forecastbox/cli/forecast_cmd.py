"""CLI command: forecastbox forecast."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd

from forecastbox._logging import get_logger

logger = get_logger("cli.forecast")


@click.command("forecast")
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to CSV with data.")
@click.option("--target", required=True, type=str, help="Target column name.")
@click.option(
    "--model",
    type=click.Choice(["auto_arima", "auto_ets", "auto_select", "theta"]),
    default="auto_arima",
    help="Model to use (default: auto_arima).",
)
@click.option("--horizon", type=int, default=12, help="Forecast horizon (default: 12).")
@click.option("--seasonal-period", type=int, default=None, help="Seasonal period (default: auto).")
@click.option("--cv/--no-cv", default=True, help="Run cross-validation (default: True).")
@click.option("--output", type=click.Path(), default=None, help="Path to save forecast (JSON/CSV).")
@click.option("--plot/--no-plot", default=True, help="Show plot (default: True).")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv"]),
    default="json",
    help="Output format (default: json).",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
def forecast(
    data: str,
    target: str,
    model: str,
    horizon: int,
    seasonal_period: int | None,
    cv: bool,
    output: str | None,
    plot: bool,
    output_format: str,
    verbose: bool,
) -> None:
    """Generate forecasts from time series data.

    Reads a CSV file, fits the specified model, and produces
    point forecasts with prediction intervals.

    Example:
        forecastbox forecast --data ipca.csv --target ipca --model auto_arima --horizon 12
    """
    import logging

    if verbose:
        logging.getLogger("forecastbox").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.debug("Loading data from %s", data)

    try:
        df = pd.read_csv(data, parse_dates=True, index_col=0)
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    if target not in df.columns:
        click.echo(
            f"Error: column '{target}' not found in data. Available: {list(df.columns)}",
            err=True,
        )
        sys.exit(1)

    series = df[target].dropna()
    logger.debug("Series length: %d", len(series))

    # Import model classes
    try:
        if model == "auto_arima":
            from forecastbox.auto.auto_arima import AutoARIMA

            estimator = AutoARIMA(seasonal_period=seasonal_period)
        elif model == "auto_ets":
            from forecastbox.auto.auto_ets import AutoETS

            estimator = AutoETS(seasonal_period=seasonal_period)
        elif model == "theta":
            from forecastbox.auto.theta import Theta

            estimator = Theta(seasonal_period=seasonal_period)
        elif model == "auto_select":
            from forecastbox.auto.auto_select import AutoSelect

            estimator = AutoSelect(seasonal_period=seasonal_period)
        else:
            click.echo(f"Error: unknown model '{model}'", err=True)
            sys.exit(1)
    except ImportError as e:
        click.echo(f"Error importing model: {e}", err=True)
        sys.exit(1)

    # Fit and forecast
    click.echo(f"Fitting {model} on '{target}' (n={len(series)})...")
    try:
        estimator.fit(series)
        fc = estimator.forecast(horizon=horizon)
    except Exception as e:
        click.echo(f"Error during fitting/forecasting: {e}", err=True)
        sys.exit(1)

    click.echo(f"Model: {fc.model_name}")
    click.echo(f"Horizon: {fc.horizon}")

    # Cross-validation
    cv_metrics: dict[str, float] = {}
    if cv:
        click.echo("Running cross-validation...")
        try:
            from forecastbox.cv import expanding_window_cv

            cv_results = expanding_window_cv(
                data=series,
                model_fn=lambda s: estimator.__class__(seasonal_period=seasonal_period)
                .fit(s)
                .forecast(horizon=horizon),
                initial_window=max(60, len(series) // 2),
                horizon=horizon,
                step=1,
            )
            cv_metrics = cv_results.summary()
            click.echo(f"CV Metrics: {cv_metrics}")
        except Exception as e:
            logger.warning("Cross-validation failed: %s", e)
            click.echo(f"Warning: CV failed: {e}")

    # Build output
    result: dict[str, Any] = {
        "model": fc.model_name,
        "horizon": fc.horizon,
        "point": fc.point.tolist(),
    }
    if fc.lower_80 is not None:
        result["lower_80"] = fc.lower_80.tolist()
    if fc.upper_80 is not None:
        result["upper_80"] = fc.upper_80.tolist()
    if fc.lower_95 is not None:
        result["lower_95"] = fc.lower_95.tolist()
    if fc.upper_95 is not None:
        result["upper_95"] = fc.upper_95.tolist()
    if cv_metrics:
        result["metrics"] = cv_metrics
    result["metadata"] = fc.metadata

    # Save or print
    if output:
        output_path = Path(output)
        if output_format == "csv":
            fc.to_dataframe().to_csv(output_path)
            click.echo(f"Forecast saved to {output_path} (CSV)")
        else:
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            click.echo(f"Forecast saved to {output_path} (JSON)")
    else:
        click.echo(json.dumps(result, indent=2))

    # Plot
    if plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fc.plot()
            plt.tight_layout()
            plot_path = Path(output).with_suffix(".png") if output else Path("forecast_plot.png")
            plt.savefig(plot_path, dpi=150)
            click.echo(f"Plot saved to {plot_path}")
            plt.close("all")
        except Exception as e:
            logger.warning("Plot failed: %s", e)

    if verbose:
        click.echo(f"Metadata: {fc.metadata}")
