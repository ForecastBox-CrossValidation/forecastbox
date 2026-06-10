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

# Methods that estimate weights from realized values and therefore require
# the --actual option. Simple methods (mean/median) do not.
_FIT_METHODS = frozenset(
    {"inverse_mse", "ols", "bma", "stacking", "optimal"}
)


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
    help="CSV with actuals for weight estimation (required for fit-based methods).",
)
@click.option(
    "--method",
    type=click.Choice(
        ["mean", "median", "inverse_mse", "ols", "bma", "stacking", "optimal"]
    ),
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

    Supports simple methods (mean, median) and weight-estimating methods
    (inverse_mse, ols, bma, stacking, optimal) that use actual values to
    estimate combination weights.

    Example:
        forecastbox combine --forecasts fc1.json fc2.json fc3.json \
            --method bma --actual actuals.csv --output combined.json
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

    # Load actual values if provided.
    actual_values: np.ndarray | None = None
    if actual:
        try:
            actual_df = pd.read_csv(actual, parse_dates=True, index_col=0)
            actual_values = np.asarray(actual_df.iloc[:, 0].to_numpy(), dtype=float)
        except Exception as e:
            click.echo(f"Error loading actual values: {e}", err=True)
            sys.exit(1)

    if method in _FIT_METHODS and actual_values is None:
        click.echo(
            f"Error: method '{method}' requires --actual values to estimate weights.",
            err=True,
        )
        sys.exit(1)

    # Combine
    try:
        combined = _combine(method, loaded_forecasts, actual_values)
    except ImportError as e:
        click.echo(f"Error: combination backend not available: {e}", err=True)
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


def _combine(
    method: str,
    forecasts: list[Any],
    actual_values: np.ndarray | None,
) -> Any:
    """Build a combiner for ``method`` and produce the combined forecast.

    Simple methods (mean/median) use :class:`SimpleCombiner` and need no
    fitting. Weight-estimating methods fit on the loaded forecast point
    arrays (as historical forecasts) against ``actual_values``.
    """
    from forecastbox.combination import (
        BMACombiner,
        OLSCombiner,
        OptimalCombiner,
        SimpleCombiner,
        StackingCombiner,
        WeightedCombiner,
    )

    if method in ("mean", "median"):
        return SimpleCombiner(method=method).combine(forecasts)

    # Fit-based methods: use the forecast point arrays as the historical
    # forecasts and align them with the realized actuals.
    assert actual_values is not None  # guaranteed by caller
    forecasts_train = [np.asarray(fc.point, dtype=float) for fc in forecasts]
    train_len = min(min(len(arr) for arr in forecasts_train), len(actual_values))
    forecasts_train = [arr[:train_len] for arr in forecasts_train]
    actual_train = actual_values[:train_len]

    combiner: Any
    if method == "inverse_mse":
        combiner = WeightedCombiner(method="inverse_mse")
    elif method == "ols":
        combiner = OLSCombiner()
    elif method == "bma":
        combiner = BMACombiner()
    elif method == "stacking":
        combiner = StackingCombiner()
    elif method == "optimal":
        combiner = OptimalCombiner()
    else:  # pragma: no cover - guarded by click.Choice
        msg = f"Unknown combination method: {method}"
        raise ValueError(msg)

    combiner.fit(forecasts_train, actual_train)
    return combiner.combine(forecasts)
