"""CLI command: forecastbox monitor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd

from forecastbox._logging import get_logger

logger = get_logger("cli.monitor")


@click.command("monitor")
@click.option(
    "--pipeline", required=True, type=click.Path(exists=True), help="YAML pipeline config.",
)
@click.option(
    "--actual", required=True, type=click.Path(exists=True), help="CSV with actual values.",
)
@click.option("--alerts/--no-alerts", default=True, help="Check and report alerts (default: True).")
@click.option("--window", type=int, default=12, help="Rolling metrics window (default: 12).")
@click.option("--output", type=click.Path(), default=None, help="Path to save monitor report.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
def monitor(
    pipeline: str,
    actual: str,
    alerts: bool,
    window: int,
    output: str | None,
    verbose: bool,
) -> None:
    """Monitor forecast pipeline performance.

    Tracks rolling metrics, detects forecast degradation,
    and triggers alerts when thresholds are exceeded.

    Example:
        forecastbox monitor --pipeline pipeline.yaml --actual actual.csv --alerts
    """
    import logging

    if verbose:
        logging.getLogger("forecastbox").setLevel(logging.DEBUG)

    # Load pipeline config
    try:
        import yaml

        with open(pipeline) as f:
            pipeline_config = yaml.safe_load(f)
    except ImportError:
        # Fallback: try JSON
        try:
            with open(pipeline) as f:
                pipeline_config = json.load(f)
        except Exception as e:
            click.echo(f"Error loading pipeline config: {e}", err=True)
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading pipeline config: {e}", err=True)
        sys.exit(1)

    # Load actual values
    try:
        actual_df = pd.read_csv(actual, parse_dates=True, index_col=0)
    except Exception as e:
        click.echo(f"Error loading actual values: {e}", err=True)
        sys.exit(1)

    click.echo(f"Monitoring pipeline (window={window})...")

    result: dict[str, Any] = {
        "pipeline": pipeline_config,
        "window": window,
        "alerts_enabled": alerts,
        "metrics": {},
        "alerts": [],
    }

    try:
        from forecastbox.pipeline.monitor import ForecastMonitor

        mon = ForecastMonitor(config=pipeline_config, window=window)
        mon.update(actual_df)

        rolling_metrics = mon.rolling_metrics()
        result["metrics"] = {k: round(float(v), 6) for k, v in rolling_metrics.items()}

        click.echo(f"Rolling metrics (window={window}):")
        for k, v in result["metrics"].items():
            click.echo(f"  {k}: {v}")

        if alerts:
            triggered_alerts = mon.check_alerts()
            result["alerts"] = triggered_alerts
            if triggered_alerts:
                click.echo("\n=== ALERTS ===")
                for alert in triggered_alerts:
                    click.echo(f"  [!] {alert}")
            else:
                click.echo("\nNo alerts triggered.")

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
