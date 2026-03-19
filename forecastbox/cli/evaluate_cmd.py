"""CLI command: forecastbox evaluate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from forecastbox._logging import get_logger

logger = get_logger("cli.evaluate")


@click.command("evaluate")
@click.option(
    "--forecasts",
    required=True,
    multiple=True,
    type=click.Path(exists=True),
    help="Forecast JSON files (multiple).",
)
@click.option(
    "--actual", required=True, type=click.Path(exists=True), help="CSV with actual values.",
)
@click.option(
    "--tests",
    multiple=True,
    type=click.Choice(["dm", "mcs", "gw", "mz", "encompassing"]),
    help="Statistical tests to run.",
)
@click.option(
    "--metrics",
    "metric_names",
    multiple=True,
    type=click.Choice(["mae", "rmse", "mape", "mase", "crps"]),
    default=("mae", "rmse", "mape"),
    help="Metrics to compute.",
)
@click.option("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
@click.option("--output", type=click.Path(), default=None, help="Path to save results.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
def evaluate(
    forecasts: tuple[str, ...],
    actual: str,
    tests: tuple[str, ...],
    metric_names: tuple[str, ...],
    alpha: float,
    output: str | None,
    verbose: bool,
) -> None:
    """Evaluate and compare forecast models.

    Computes metrics and runs statistical tests (Diebold-Mariano, MCS, etc.)
    across multiple forecast files.

    Example:
        forecastbox evaluate --forecasts fc1.json fc2.json --actual actual.csv --tests dm mcs
    """
    import logging

    if verbose:
        logging.getLogger("forecastbox").setLevel(logging.DEBUG)

    # Load actual values
    try:
        actual_df = pd.read_csv(actual, parse_dates=True, index_col=0)
        actual_values = actual_df.iloc[:, 0].values
    except Exception as e:
        click.echo(f"Error loading actual values: {e}", err=True)
        sys.exit(1)

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

    # Compute metrics
    from forecastbox.metrics import point_metrics

    metric_fns: dict[str, Any] = {
        "mae": point_metrics.mae,
        "rmse": point_metrics.rmse,
        "mape": point_metrics.mape,
    }

    if hasattr(point_metrics, "mase"):
        metric_fns["mase"] = point_metrics.mase

    results: dict[str, Any] = {"models": [], "metrics": {}, "tests": {}}

    for fc in loaded_forecasts:
        model_name = fc.model_name or "Unknown"
        results["models"].append(model_name)

        n = min(len(actual_values), len(fc.point))
        act = actual_values[:n]
        pred = fc.point[:n]

        model_metrics: dict[str, float] = {}
        for m_name in metric_names:
            if m_name in metric_fns:
                try:
                    val = metric_fns[m_name](act, pred)
                    model_metrics[m_name] = round(float(val), 6)
                except Exception as e:
                    logger.warning("Metric %s failed for %s: %s", m_name, model_name, e)

        results["metrics"][model_name] = model_metrics

    click.echo("=== Metrics ===")
    for model_name, model_metrics in results["metrics"].items():
        click.echo(f"\n{model_name}:")
        for m_name, m_val in model_metrics.items():
            click.echo(f"  {m_name}: {m_val}")

    # Run statistical tests
    if tests and len(loaded_forecasts) >= 2:
        errors_dict: dict[str, np.ndarray] = {}
        for fc in loaded_forecasts:
            n = min(len(actual_values), len(fc.point))
            errors_dict[fc.model_name or "Unknown"] = actual_values[:n] - fc.point[:n]

        for test_name in tests:
            click.echo(f"\n=== {test_name.upper()} Test ===")
            try:
                if test_name == "dm":
                    from forecastbox.evaluation.dm_test import dm_test

                    e1 = list(errors_dict.values())[0]
                    e2 = list(errors_dict.values())[1]
                    result_dm = dm_test(e1, e2, alpha=alpha)
                    results["tests"]["dm"] = {
                        "statistic": float(result_dm.statistic),
                        "p_value": float(result_dm.p_value),
                        "significant": bool(result_dm.p_value < alpha),
                    }
                    click.echo(f"  Statistic: {result_dm.statistic:.4f}")
                    click.echo(f"  P-value: {result_dm.p_value:.4f}")
                    click.echo(f"  Significant at {alpha}: {result_dm.p_value < alpha}")

                elif test_name == "mcs":
                    from forecastbox.evaluation.mcs import mcs_test

                    errors_matrix = np.column_stack(list(errors_dict.values()))
                    model_names = list(errors_dict.keys())
                    result_mcs = mcs_test(errors_matrix, model_names=model_names, alpha=alpha)
                    results["tests"]["mcs"] = {
                        "included_models": result_mcs.included_models,
                        "p_values": {k: float(v) for k, v in result_mcs.p_values.items()},
                    }
                    click.echo(f"  Included models: {result_mcs.included_models}")

                elif test_name == "gw":
                    from forecastbox.evaluation.gw_test import gw_test

                    e1 = list(errors_dict.values())[0]
                    e2 = list(errors_dict.values())[1]
                    result_gw = gw_test(e1, e2, alpha=alpha)
                    results["tests"]["gw"] = {
                        "statistic": float(result_gw.statistic),
                        "p_value": float(result_gw.p_value),
                    }
                    click.echo(f"  Statistic: {result_gw.statistic:.4f}")
                    click.echo(f"  P-value: {result_gw.p_value:.4f}")

                elif test_name == "mz":
                    from forecastbox.evaluation.mz_test import mz_test

                    pred = loaded_forecasts[0].point
                    n = min(len(actual_values), len(pred))
                    result_mz = mz_test(actual_values[:n], pred[:n])
                    results["tests"]["mz"] = {
                        "alpha": float(result_mz.alpha),
                        "beta": float(result_mz.beta),
                        "p_value": float(result_mz.p_value),
                    }
                    click.echo(f"  Alpha: {result_mz.alpha:.4f}")
                    click.echo(f"  Beta: {result_mz.beta:.4f}")

                elif test_name == "encompassing":
                    from forecastbox.evaluation.encompassing import encompassing_test

                    f1 = loaded_forecasts[0].point
                    f2 = loaded_forecasts[1].point
                    n = min(len(actual_values), len(f1), len(f2))
                    result_enc = encompassing_test(actual_values[:n], f1[:n], f2[:n])
                    results["tests"]["encompassing"] = {
                        "lambda": float(result_enc.lambda_),
                        "p_value": float(result_enc.p_value),
                    }
                    click.echo(f"  Lambda: {result_enc.lambda_:.4f}")
                    click.echo(f"  P-value: {result_enc.p_value:.4f}")

            except ImportError as e:
                click.echo(f"  Test module not available: {e}")
            except Exception as e:
                click.echo(f"  Test failed: {e}")

    # Save results
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"\nResults saved to {output_path}")
