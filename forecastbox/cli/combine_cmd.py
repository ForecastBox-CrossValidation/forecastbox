"""CLI command: forecastbox combine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from forecastbox._logging import get_logger

logger = get_logger("cli.combine")


@click.command("combine")
@click.option(
    "--forecasts",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Forecast JSON files (multiple).",
)
@click.option(
    "--actual",
    type=click.Path(exists=True),
    default=None,
    help="CSV with actuals for weight estimation.",
)
@click.option(
    "--method",
    type=click.Choice(["mean", "median", "inverse_mse", "ols", "bma", "stacking", "optimal"]),
    default="mean",
    help="Combination method (default: mean).",
)
@click.option("--output", type=click.Path(), default=None, help="Path to save combined forecast.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
def combine(
    forecasts: tuple[str, ...],
    actual: str | None,
    method: str,
    output: str | None,
    verbose: bool,
) -> None:
    """Combine multiple forecasts.

    Supports simple methods (mean, median) and advanced methods
    (BMA, stacking, optimal) that use actual values to estimate weights.

    Example:
        forecastbox combine --forecasts fc1.json fc2.json fc3.json \
            --method bma --output combined.json
    """
    import logging

    if verbose:
        logging.getLogger("forecastbox").setLevel(logging.DEBUG)

    # Load forecasts
    from forecastbox.core.forecast import Forecast

    loaded_forecasts: list[Forecast] = []
    for fc_path in forecasts:
        try:
            fc = Forecast.load(fc_path)
            loaded_forecasts.append(fc)
            logger.debug("Loaded forecast: %s", fc.model_name)
        except Exception as e:
            click.echo(f"Error loading forecast {fc_path}: {e}", err=True)
            sys.exit(1)

    if len(loaded_forecasts) < 2:
        click.echo("Error: at least 2 forecasts required for combination.", err=True)
        sys.exit(1)

    click.echo(f"Combining {len(loaded_forecasts)} forecasts using '{method}'...")

    # Load actual values if needed
    actual_values: np.ndarray | None = None
    if actual:
        try:
            actual_df = pd.read_csv(actual, parse_dates=True, index_col=0)
            actual_values = actual_df.iloc[:, 0].values
        except Exception as e:
            click.echo(f"Error loading actual values: {e}", err=True)
            sys.exit(1)

    # Combine
    try:
        if method in ("mean", "median"):
            combined = Forecast.combine(loaded_forecasts, method=method)
        else:
            from forecastbox.combination.methods import combine_forecasts

            combined = combine_forecasts(
                forecasts=loaded_forecasts,
                actual=actual_values,
                method=method,
            )
    except ImportError as e:
        click.echo(f"Error: combination module not available: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during combination: {e}", err=True)
        sys.exit(1)

    click.echo(f"Combined model: {combined.model_name}")
    click.echo(f"Point forecasts: {combined.point.tolist()}")

    # Save or print
    if output:
        output_path = Path(output)
        combined.save(output_path)
        click.echo(f"Combined forecast saved to {output_path}")
    else:
        result: dict[str, Any] = {
            "model": combined.model_name,
            "method": method,
            "n_models": len(loaded_forecasts),
            "models": [fc.model_name for fc in loaded_forecasts],
            "point": combined.point.tolist(),
        }
        if combined.lower_80 is not None:
            result["lower_80"] = combined.lower_80.tolist()
        if combined.upper_80 is not None:
            result["upper_80"] = combined.upper_80.tolist()
        if combined.lower_95 is not None:
            result["lower_95"] = combined.lower_95.tolist()
        if combined.upper_95 is not None:
            result["upper_95"] = combined.upper_95.tolist()
        click.echo(json.dumps(result, indent=2))
