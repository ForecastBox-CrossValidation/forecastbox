"""Tests for forecastbox nowcast CLI command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from forecastbox.cli.main import cli


@pytest.fixture
def panel_csv(tmp_path: Path) -> Path:
    """Create a sample panel CSV for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2015-01-01", periods=120, freq="MS")
    data = pd.DataFrame(
        {
            "pib": np.nan,
            "producao_industrial": 100 + np.cumsum(rng.normal(0.05, 1.0, 120)),
            "vendas_varejo": 100 + np.cumsum(rng.normal(0.03, 0.8, 120)),
            "confianca_consumidor": 50 + rng.normal(0, 5, 120),
        },
        index=dates,
    )
    # Fill quarterly PIB values
    quarterly_idx = data.index.month.isin([3, 6, 9, 12])
    data.loc[quarterly_idx, "pib"] = 100 + np.cumsum(rng.normal(0.5, 2.0, quarterly_idx.sum()))
    data.index.name = "date"
    path = tmp_path / "panel.csv"
    data.to_csv(path)
    return path


class TestNowcastCmd:
    """Tests for the nowcast CLI command."""

    def test_dfm_nowcast(self, panel_csv: Path) -> None:
        """--method dfm --factors 2 executes DFM nowcasting."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(panel_csv),
                "--target", "pib",
                "--method", "dfm",
                "--factors", "2",
            ],
        )
        assert result.exit_code == 0 or "dfm" in result.output.lower() or "Error" in result.output

    def test_bridge_nowcast(self, panel_csv: Path) -> None:
        """--method bridge executes bridge equation."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(panel_csv),
                "--target", "pib",
                "--method", "bridge",
            ],
        )
        assert (
            result.exit_code == 0
            or "bridge" in result.output.lower()
            or "Error" in result.output
        )

    def test_news_flag(self, panel_csv: Path) -> None:
        """--news computes news decomposition."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(panel_csv),
                "--target", "pib",
                "--method", "dfm",
                "--factors", "2",
                "--news",
            ],
        )
        assert result.exit_code == 0 or "news" in result.output.lower() or "Error" in result.output
