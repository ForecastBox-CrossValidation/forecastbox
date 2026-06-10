"""CLI command: forecastbox nowcast."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd

from forecastbox._logging import get_logger

logger = get_logger("cli.nowcast")


@click.command("nowcast")
@click.option(
    "--data", required=True, type=click.Path(exists=True), help="Path to CSV with panel data.",
)
@click.option("--target", required=True, type=str, help="Target variable (quarterly).")
@click.option(
    "--method",
    type=click.Choice(["dfm", "bridge", "midas"]),
    default="dfm",
    help="Nowcasting method (default: dfm).",
)
@click.option("--factors", type=int, default=2, help="Number of factors for DFM (default: 2).")
@click.option("--reference-date", type=str, default=None, help="Reference date (default: today).")
@click.option("--output", type=click.Path(), default=None, help="Path to save nowcast.")
@click.option(
    "--news/--no-news", default=False, help="Compute news decomposition (default: False).",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
def nowcast(
    data: str,
    target: str,
    method: str,
    factors: int,
    reference_date: str | None,
    output: str | None,
    news: bool,
    verbose: bool,
) -> None:
    """Generate nowcasts from panel data.

    Uses Dynamic Factor Models (DFM), bridge equations, or MIDAS
    to nowcast a quarterly target variable from higher-frequency indicators.

    Example:
        forecastbox nowcast --data panel.csv --target pib --method dfm --factors 2
    """
    import logging

    if verbose:
        logging.getLogger("forecastbox").setLevel(logging.DEBUG)

    # Load data
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

    indicators = [c for c in df.columns if c != target]
    if not indicators:
        click.echo(
            f"Error: no indicator columns found besides target '{target}'.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Nowcasting '{target}' using {method} (factors={factors})...")

    result: dict[str, Any] = {
        "target": target,
        "method": method,
        "factors": factors,
        "reference_date": reference_date,
    }

    try:
        if method == "dfm":
            from forecastbox.nowcasting.dfm import DFMNowcaster

            # The target is treated as quarterly; all other columns as monthly.
            frequency_map = {target: "Q"}
            for col in indicators:
                frequency_map[col] = "M"

            nowcaster = DFMNowcaster(n_factors=factors, frequency_map=frequency_map)
            nowcaster.fit(df)
            fc = nowcaster.nowcast(target=target, reference_date=reference_date)
            result["nowcast"] = float(fc.point[0])
            result["model_name"] = fc.model_name
            result["model_info"] = dict(fc.metadata)

            if news:
                click.echo("Computing news decomposition...")
                from forecastbox.nowcasting.news import NewsDecomposition

                news_decomp = NewsDecomposition(nowcaster)
                news_result = news_decomp.decompose(df, df, target=target)
                result["news"] = {
                    "total_revision": float(news_result.total_revision),
                    "old_nowcast": float(news_result.old_nowcast),
                    "new_nowcast": float(news_result.new_nowcast),
                    "contributions": {
                        k: float(v) for k, v in news_result.contributions.items()
                    },
                }
                click.echo(f"News total revision: {result['news']['total_revision']}")

        elif method == "bridge":
            from forecastbox.nowcasting.bridge import BridgeEquation

            bridge = BridgeEquation(target=target, indicators=indicators)
            bridge.fit(df)
            fc = bridge.nowcast(reference_date=reference_date)
            result["nowcast"] = float(fc.point[0])
            result["model_name"] = fc.model_name
            result["r_squared"] = float(bridge.r_squared())

        elif method == "midas":
            from forecastbox.nowcasting.midas import MIDAS

            midas = MIDAS(target=target, high_freq=indicators)
            midas.fit(df)
            fc = midas.nowcast()
            result["nowcast"] = float(fc.point[0])
            result["model_name"] = fc.model_name

    except ImportError as e:
        click.echo(f"Error: nowcasting module not available: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error during nowcasting: {e}", err=True)
        sys.exit(1)

    click.echo(f"Nowcast: {result.get('nowcast', 'N/A')}")

    # Save or print
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        click.echo(f"Nowcast saved to {output_path}")
    else:
        click.echo(json.dumps(result, indent=2, default=str))
