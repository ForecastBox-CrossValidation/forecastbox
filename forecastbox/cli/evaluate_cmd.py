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
    type=click.Choice(["mae", "rmse", "mape"]),
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
        forecastbox evaluate --forecasts fc1.json --forecasts fc2.json \
            --actual actual.csv --tests dm --tests mcs
    """
    import logging

    if verbose:
        logging.getLogger("forecastbox").setLevel(logging.DEBUG)

    # Load actual values
    try:
        actual_df = pd.read_csv(actual, parse_dates=True, index_col=0)
        actual_values = np.asarray(actual_df.iloc[:, 0].values, dtype=np.float64)
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
    from forecastbox.metrics import mae, mape, rmse

    metric_fns: dict[str, Any] = {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }

    results: dict[str, Any] = {"models": [], "metrics": {}, "tests": {}}

    for fc in loaded_forecasts:
        model_name = fc.model_name or "Unknown"
        results["models"].append(model_name)

        n = min(len(actual_values), len(fc.point))
        act = actual_values[:n]
        pred = np.asarray(fc.point[:n], dtype=np.float64)

        model_metrics: dict[str, float] = {}
        for m_name in metric_names:
            if m_name in metric_fns:
                try:
                    # point metrics take (actual, predicted)
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
        # Aligned point forecasts per model and a common actual window.
        common_n = min(len(actual_values), *(len(fc.point) for fc in loaded_forecasts))
        act = actual_values[:common_n]
        preds: dict[str, np.ndarray] = {}
        for fc in loaded_forecasts:
            name = fc.model_name or "Unknown"
            preds[name] = np.asarray(fc.point[:common_n], dtype=np.float64)

        pred_list = list(preds.values())
        names_list = list(preds.keys())

        for test_name in tests:
            click.echo(f"\n=== {test_name.upper()} Test ===")
            try:
                if test_name == "dm":
                    from forecastbox.evaluation import diebold_mariano

                    res = diebold_mariano(act, pred_list[0], pred_list[1])
                    significant = bool(res.pvalue < alpha)
                    results["tests"]["dm"] = {
                        "statistic": float(res.statistic),
                        "pvalue": float(res.pvalue),
                        "significant": significant,
                    }
                    click.echo(f"  Statistic: {res.statistic:.4f}")
                    click.echo(f"  P-value: {res.pvalue:.4f}")
                    click.echo(f"  Significant at {alpha}: {significant}")
                    click.echo(f"  {res.conclusion(alpha=alpha)}")

                elif test_name == "mcs":
                    from forecastbox.evaluation import model_confidence_set

                    res = model_confidence_set(act, preds, alpha=alpha, seed=0)
                    results["tests"]["mcs"] = {
                        "included_models": res.included_models,
                        "excluded_models": res.excluded_models,
                        "pvalues": {k: float(v) for k, v in res.pvalues.items()},
                    }
                    click.echo(f"  Included models: {res.included_models}")
                    click.echo(f"  Excluded models: {res.excluded_models}")

                elif test_name == "gw":
                    from forecastbox.evaluation import giacomini_white

                    res = giacomini_white(act, pred_list[0], pred_list[1])
                    results["tests"]["gw"] = {
                        "statistic": float(res.statistic),
                        "pvalue": float(res.pvalue),
                        "df": int(res.df),
                    }
                    click.echo(f"  Statistic: {res.statistic:.4f}")
                    click.echo(f"  P-value: {res.pvalue:.4f}")
                    click.echo(f"  {res.conclusion(alpha=alpha)}")

                elif test_name == "mz":
                    from forecastbox.evaluation import mincer_zarnowitz

                    res = mincer_zarnowitz(act, pred_list[0])
                    results["tests"]["mz"] = {
                        "alpha": float(res.alpha),
                        "beta": float(res.beta),
                        "f_statistic": float(res.f_statistic),
                        "pvalue": float(res.pvalue),
                        "efficient": bool(res.is_efficient(alpha=alpha)),
                    }
                    click.echo(f"  Intercept (alpha): {res.alpha:.4f}")
                    click.echo(f"  Slope (beta): {res.beta:.4f}")
                    click.echo(f"  F-statistic: {res.f_statistic:.4f}")
                    click.echo(f"  P-value: {res.pvalue:.4f}")
                    click.echo(f"  Efficient at {alpha}: {res.is_efficient(alpha=alpha)}")

                elif test_name == "encompassing":
                    from forecastbox.evaluation import encompassing_test

                    res = encompassing_test(act, pred_list[0], pred_list[1], alpha=alpha)
                    results["tests"]["encompassing"] = {
                        "lambda_hat": float(res.lambda_hat),
                        "statistic": float(res.statistic),
                        "pvalue": float(res.pvalue),
                        "f1_encompasses_f2": bool(res.f1_encompasses_f2),
                        "f2_encompasses_f1": bool(res.f2_encompasses_f1),
                    }
                    click.echo(f"  Lambda: {res.lambda_hat:.4f}")
                    click.echo(f"  Statistic: {res.statistic:.4f}")
                    click.echo(f"  P-value: {res.pvalue:.4f}")
                    click.echo(f"  f1 encompasses f2: {res.f1_encompasses_f2}")
                    click.echo(f"  f2 encompasses f1: {res.f2_encompasses_f1}")

            except ImportError as e:
                click.echo(f"  Test module not available: {e}")
            except Exception as e:
                click.echo(f"  Test failed: {e}")
        # names_list referenced for potential future labeling; kept aligned with pred_list.
        logger.debug("Tested models: %s", names_list)

    # Save results
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        click.echo(f"\nResults saved to {output_path}")
