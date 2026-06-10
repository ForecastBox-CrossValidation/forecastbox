"""Tests for forecastbox nowcast CLI command."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from forecastbox.cli.main import cli


def _make_panel(target_months: tuple[int, ...]) -> pd.DataFrame:
    """Build a panel with monthly indicators driven by a common factor.

    The quarterly target is observed only in ``target_months``. A stationary
    AR(1) factor keeps the data well conditioned.
    """
    rng = np.random.default_rng(0)
    n = 96
    dates = pd.date_range("2010-01-01", periods=n, freq="MS")
    factor = np.zeros(n)
    for t in range(1, n):
        factor[t] = 0.7 * factor[t - 1] + rng.normal(0, 1.0)
    data = pd.DataFrame(
        {
            "pib": np.nan,
            "producao_industrial": 100 + factor + rng.normal(0, 0.5, n),
            "vendas_varejo": 100 + 0.9 * factor + rng.normal(0, 0.5, n),
            "confianca_consumidor": 100 + 0.7 * factor + rng.normal(0, 0.5, n),
        },
        index=dates,
    )
    mask = data.index.month.isin(target_months)
    data.loc[mask, "pib"] = 100 + factor[mask]
    data.index.name = "date"
    return data


@pytest.fixture
def dfm_csv(tmp_path: Path) -> Path:
    """Panel where the quarterly target is observed at quarter-end months."""
    data = _make_panel(target_months=(3, 6, 9, 12))
    path = tmp_path / "panel_dfm.csv"
    data.to_csv(path)
    return path


@pytest.fixture
def bridge_csv(tmp_path: Path) -> Path:
    """Panel where the target aligns with quarter-start (resample 'QS') dates."""
    data = _make_panel(target_months=(1, 4, 7, 10))
    path = tmp_path / "panel_bridge.csv"
    data.to_csv(path)
    return path


class TestNowcastCmd:
    """Tests for the nowcast CLI command."""

    def test_dfm_nowcast(self, dfm_csv: Path) -> None:
        """--method dfm --factors 2 runs DFM nowcasting end to end."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(dfm_csv),
                "--target", "pib",
                "--method", "dfm",
                "--factors", "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Nowcasting 'pib' using dfm" in result.output
        # model_name is deterministic from the DFM contract: "DFM(<n>f)-<target>"
        assert "DFM(2f)-pib" in result.output

    def test_dfm_nowcast_output_file(self, dfm_csv: Path, tmp_path: Path) -> None:
        """--output writes a JSON report with the DFM contract fields."""
        out = tmp_path / "nowcast.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(dfm_csv),
                "--target", "pib",
                "--method", "dfm",
                "--factors", "2",
                "--output", str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        # The DFM EM solver can emit NaN literals, so parse with NaN allowed.
        payload = json.loads(out.read_text(), parse_constant=lambda c: float(c))
        assert payload["target"] == "pib"
        assert payload["method"] == "dfm"
        assert payload["factors"] == 2
        assert payload["model_name"] == "DFM(2f)-pib"
        assert payload["model_info"]["n_factors"] == 2
        assert payload["model_info"]["n_variables"] == 4
        assert "nowcast" in payload

    def test_bridge_nowcast(self, bridge_csv: Path) -> None:
        """--method bridge produces a coherent finite nowcast."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(bridge_csv),
                "--target", "pib",
                "--method", "bridge",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Bridge(pib)" in result.output
        payload = json.loads(result.output[result.output.index("{"):])
        assert payload["model_name"] == "Bridge(pib)"
        assert np.isfinite(payload["nowcast"])
        # Target is near 100; the bridge regression should recover that level.
        assert 90.0 < payload["nowcast"] < 110.0
        assert 0.0 <= payload["r_squared"] <= 1.0

    def test_midas_nowcast(self, bridge_csv: Path) -> None:
        """--method midas produces a coherent finite nowcast."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(bridge_csv),
                "--target", "pib",
                "--method", "midas",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output[result.output.index("{"):])
        assert payload["method"] == "midas"
        assert re.match(r"MIDAS\(", payload["model_name"])
        assert np.isfinite(payload["nowcast"])

    def test_news_flag(self, dfm_csv: Path) -> None:
        """--news triggers the news decomposition path without error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(dfm_csv),
                "--target", "pib",
                "--method", "dfm",
                "--factors", "2",
                "--news",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Computing news decomposition..." in result.output
        assert "News total revision:" in result.output

    def test_missing_target_column(self, dfm_csv: Path) -> None:
        """An unknown target column reports an error and exits non-zero."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "nowcast",
                "--data", str(dfm_csv),
                "--target", "does_not_exist",
                "--method", "midas",
            ],
        )
        assert result.exit_code != 0
        assert "not found in data" in result.output
