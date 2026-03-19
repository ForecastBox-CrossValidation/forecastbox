"""Main CLI entrypoint for forecastbox."""

from __future__ import annotations

import click

from forecastbox.__version__ import __version__


@click.group()
@click.version_option(version=__version__, prog_name="forecastbox")
def cli() -> None:
    """ForecastBox - Forecast engine for the NodesEcon ecosystem.

    A comprehensive toolkit for time series forecasting, evaluation,
    nowcasting, monitoring, and forecast combination.
    """


def register_commands() -> None:
    """Register all CLI commands."""
    from forecastbox.cli.combine_cmd import combine
    from forecastbox.cli.evaluate_cmd import evaluate
    from forecastbox.cli.forecast_cmd import forecast
    from forecastbox.cli.monitor_cmd import monitor
    from forecastbox.cli.nowcast_cmd import nowcast

    cli.add_command(forecast)
    cli.add_command(evaluate)
    cli.add_command(nowcast)
    cli.add_command(monitor)
    cli.add_command(combine)


register_commands()


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
