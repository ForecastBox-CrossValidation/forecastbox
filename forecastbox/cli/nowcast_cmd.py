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

            nowcaster = DFMNowcaster(n_factors=factors)
            nowcaster.fit(df)
            nc = nowcaster.nowcast(target=target)
            result["nowcast"] = float(nc.point) if hasattr(nc, "point") else float(nc)
            result["model_info"] = getattr(nc, "metadata", {})

            if news:
                click.echo("Computing news decomposition...")
                try:
                    from forecastbox.nowcasting.news import NewsDecomposition

                    news_decomp = NewsDecomposition(nowcaster)
                    news_result = news_decomp.decompose(df, df)
                    result["news"] = {
                        "total_revision": float(news_result.total_revision),
                        "contributions": {
                            k: float(v) for k, v in news_result.contributions.items()
                        },
                    }
                    click.echo(f"News decomposition: {result['news']}")
                except ImportError as e:
                    click.echo(f"Warning: news module not available: {e}")

        elif method == "bridge":
            from forecastbox.nowcasting.bridge import BridgeEquation

            bridge = BridgeEquation()
            bridge.fit(df, target=target)
            nc = bridge.nowcast()
            result["nowcast"] = float(nc.point) if hasattr(nc, "point") else float(nc)

        elif method == "midas":
            from forecastbox.nowcasting.midas import MIDASNowcaster

            midas = MIDASNowcaster()
            midas.fit(df, target=target)
            nc = midas.nowcast()
            result["nowcast"] = float(nc.point) if hasattr(nc, "point") else float(nc)

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
