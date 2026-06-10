"""CLI command: forecastbox monitor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import click
import pandas as pd

from forecastbox._logging import get_logger

logger = get_logger("cli.monitor")


def _load_pipeline_config(path: str) -> dict[str, Any]:
    """Load a pipeline config from YAML (preferred) or JSON."""
    try:
        import yaml

        with open(path) as f:
            config = yaml.safe_load(f)
    except ImportError:
        with open(path) as f:
            config = json.load(f)
    if not isinstance(config, dict):
        msg = f"Pipeline config must be a mapping, got {type(config).__name__}."
        raise ValueError(msg)
    return cast("dict[str, Any]", config)


def _build_pipeline(config: dict[str, Any], data: pd.Series) -> Any:
    """Build a ForecastPipeline from a config dict and a data Series.

    Only keys understood by ForecastPipeline.__init__ are forwarded; unknown
    keys (e.g. ``name`` or ``thresholds``) are ignored here.
    """
    from forecastbox.pipeline.pipeline import ForecastPipeline

    kwargs: dict[str, Any] = {}
    if "target" in config:
        kwargs["target"] = config["target"]
    if "horizon" in config:
        kwargs["horizon"] = int(config["horizon"])
    # ``model`` (singular) or ``models`` (list) -> models list
    if "models" in config:
        kwargs["models"] = list(config["models"])
    elif "model" in config:
        kwargs["models"] = [config["model"]]
    if "combination" in config:
        kwargs["combination"] = config["combination"]
    if "evaluation" in config:
        kwargs["evaluation"] = list(config["evaluation"])
    if "preprocess" in config:
        kwargs["preprocess"] = list(config["preprocess"])

    return ForecastPipeline(data, **kwargs)


@click.command("monitor")
@click.option(
    "--pipeline", required=True, type=click.Path(exists=True), help="YAML/JSON pipeline config.",
)
@click.option(
    "--actual",
    required=True,
    type=click.Path(exists=True),
    help="CSV with a date index plus 'actual' and 'forecast' columns "
    "(optional 'lower_95'/'upper_95').",
)
@click.option("--alerts/--no-alerts", default=True, help="Check and report alerts (default: True).")
@click.option("--window", type=int, default=12, help="Rolling/degradation window (default: 12).")
@click.option(
    "--metric",
    type=click.Choice(["rmse", "mae", "mape"]),
    default="rmse",
    help="Rolling accuracy metric (default: rmse).",
)
@click.option("--output", type=click.Path(), default=None, help="Path to save monitor report JSON.")
@click.option("--plot/--no-plot", default=False, help="Save a forecast-vs-actual plot (PNG).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
def monitor(
    pipeline: str,
    actual: str,
    alerts: bool,
    window: int,
    metric: str,
    output: str | None,
    plot: bool,
    verbose: bool,
) -> None:
    """Monitor forecast pipeline performance.

    Matches forecasted values against realized actuals, reports rolling and
    overall accuracy metrics, hit rate, and detects forecast degradation.

    The --actual CSV must have a date index in its first column and at least
    'actual' and 'forecast' columns. Optional 'lower_95'/'upper_95' columns
    enable hit-rate computation.

    Example:
        forecastbox monitor --pipeline pipeline.yaml --actual actual.csv --alerts
    """
    import logging

    if verbose:
        logging.getLogger("forecastbox").setLevel(logging.DEBUG)

    # Load pipeline config
    try:
        pipeline_config = _load_pipeline_config(pipeline)
    except Exception as e:
        click.echo(f"Error loading pipeline config: {e}", err=True)
        sys.exit(1)

    # Load actual / forecast values
    try:
        df = pd.read_csv(actual, parse_dates=True, index_col=0)
    except Exception as e:
        click.echo(f"Error loading actual values: {e}", err=True)
        sys.exit(1)

    if "actual" not in df.columns or "forecast" not in df.columns:
        click.echo(
            "Error: --actual CSV must contain 'actual' and 'forecast' columns. "
            f"Found: {list(df.columns)}",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Monitoring pipeline (window={window})...")

    result: dict[str, Any] = {
        "pipeline": pipeline_config,
        "window": window,
        "metric": metric,
        "alerts_enabled": alerts,
        "overall_metrics": {},
        "rolling_metric": {},
        "bias": None,
        "hit_rate": None,
        "alerts": [],
    }

    try:
        from forecastbox.pipeline.monitor import ForecastMonitor

        data_series = pd.Series(df["actual"].to_numpy(dtype=float), index=df.index)
        fc_pipeline = _build_pipeline(pipeline_config, data_series)

        mon = ForecastMonitor(fc_pipeline)
        mon.add_actuals(pd.Series(df["actual"].to_numpy(dtype=float), index=df.index))

        has_intervals = "lower_95" in df.columns and "upper_95" in df.columns
        for date, row in df.iterrows():
            lower = float(row["lower_95"]) if has_intervals else None
            upper = float(row["upper_95"]) if has_intervals else None
            ts = cast(pd.Timestamp, date)
            mon.add_forecast(ts, float(row["forecast"]), lower, upper)

        report = mon.accuracy_report()
        result["overall_metrics"] = {
            k: round(float(v), 6) for k, v in report.overall_metrics.items()
        }
        result["bias"] = round(float(report.bias), 6)
        result["hit_rate"] = round(float(report.hit_rate), 6)

        rolling = mon.rolling_accuracy(window=window, metric=metric)
        result["rolling_metric"] = {
            str(idx): round(float(val), 6) for idx, val in rolling.items()
        }

        click.echo(report.summary())

        if not rolling.empty:
            click.echo(f"\nRolling {metric.upper()} (window={window}):")
            click.echo(f"  latest: {float(rolling.iloc[-1]):.4f}")

        if alerts:
            triggered: list[str] = []
            thresholds = pipeline_config.get("thresholds", {})
            for name, value in report.overall_metrics.items():
                limit = thresholds.get(name)
                if limit is not None and float(value) > float(limit):
                    triggered.append(
                        f"{name.upper()} {float(value):.4f} exceeds threshold {float(limit):.4f}"
                    )

            if mon.degradation_test(window=window):
                triggered.append(
                    f"Forecast degradation detected (window={window})"
                )

            result["alerts"] = triggered
            if triggered:
                click.echo("\n=== ALERTS ===")
                for alert in triggered:
                    click.echo(f"  [!] {alert}")
            else:
                click.echo("\nNo alerts triggered.")

        if plot:
            import matplotlib
            from matplotlib.figure import Figure

            matplotlib.use("Agg")
            ax = mon.plot_forecast_vs_actual()
            plot_path = (
                Path(output).with_suffix(".png")
                if output
                else Path("monitor_plot.png")
            )
            fig = ax.figure
            if isinstance(fig, Figure):
                fig.savefig(plot_path)  # type: ignore[reportUnknownMemberType]
            click.echo(f"Plot saved to {plot_path}")

    except ImportError as e:
        click.echo(f"Error: pipeline module not available: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during monitoring: {e}", err=True)
        sys.exit(1)

    # Save results
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"Monitor report saved to {output_path}")
