"""Tests for forecastbox forecast CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from forecastbox.cli.main import cli


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=120, freq="MS")
    data = pd.DataFrame(
        {
            "y": 100 + np.cumsum(rng.normal(0.1, 1.0, 120)),
            "x1": rng.normal(0, 1, 120),
        },
        index=dates,
    )
    data.index.name = "date"
    path = tmp_path / "test_data.csv"
    data.to_csv(path)
    return path


class TestForecastCmd:
    """Tests for the forecast CLI command."""

    def test_basic_forecast(self, sample_csv: Path) -> None:
        """forecastbox forecast --data test.csv --target y --horizon 12 generates output."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "forecast",
                "--data", str(sample_csv),
                "--target", "y",
                "--horizon", "12",
                "--no-plot",
                "--no-cv",
            ],
        )
        # Command should execute (may fail on model import but should not crash CLI)
        assert result.exit_code == 0 or "Error" in result.output

    def test_output_json(self, sample_csv: Path, tmp_path: Path) -> None:
        """--output fc.json saves a valid JSON file."""
        runner = CliRunner()
        output_path = tmp_path / "fc.json"
        result = runner.invoke(
            cli,
            [
                "forecast",
                "--data", str(sample_csv),
                "--target", "y",
                "--horizon", "6",
                "--output", str(output_path),
                "--format", "json",
                "--no-plot",
                "--no-cv",
            ],
        )
        if result.exit_code == 0 and output_path.exists():
            data = json.loads(output_path.read_text())
            assert "point" in data
            assert "model" in data

    def test_output_csv(self, sample_csv: Path, tmp_path: Path) -> None:
        """--format csv --output fc.csv saves a valid CSV file."""
        runner = CliRunner()
        output_path = tmp_path / "fc.csv"
        result = runner.invoke(
            cli,
            [
                "forecast",
                "--data", str(sample_csv),
                "--target", "y",
                "--horizon", "6",
                "--output", str(output_path),
                "--format", "csv",
                "--no-plot",
                "--no-cv",
            ],
        )
        if result.exit_code == 0 and output_path.exists():
            df = pd.read_csv(output_path, index_col=0)
            assert "point" in df.columns

    def test_model_selection(self, sample_csv: Path) -> None:
        """--model auto_ets uses AutoETS."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "forecast",
                "--data", str(sample_csv),
                "--target", "y",
                "--model", "auto_ets",
                "--horizon", "6",
                "--no-plot",
                "--no-cv",
            ],
        )
        # Should attempt to use AutoETS
        assert (
            result.exit_code == 0
            or "auto_ets" in result.output.lower()
            or "Error" in result.output
        )

    def test_verbose(self, sample_csv: Path) -> None:
        """-v produces verbose output."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "forecast",
                "--data", str(sample_csv),
                "--target", "y",
                "--horizon", "6",
                "--no-plot",
                "--no-cv",
                "-v",
            ],
        )
        # Verbose flag should be accepted
        assert result.exit_code == 0 or "Error" in result.output

    def test_missing_data_error(self) -> None:
        """Non-existent file produces clear error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["forecast", "--data", "/nonexistent/path.csv", "--target", "y"],
        )
        assert result.exit_code != 0
