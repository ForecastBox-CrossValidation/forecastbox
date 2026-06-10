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
        assert result.exit_code == 0, result.output
        assert "Fitting auto_arima on 'y'" in result.output
        assert "Horizon: 12" in result.output
        # The printed JSON payload should carry a 12-step point forecast.
        payload = json.loads(result.output[result.output.index('{\n  "model"'):])
        assert len(payload["point"]) == 12
        assert payload["horizon"] == 12
        assert payload["model"]

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
        assert result.exit_code == 0, result.output
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "point" in data
        assert "model" in data
        assert len(data["point"]) == 6
        assert data["horizon"] == 6

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
        assert result.exit_code == 0, result.output
        assert output_path.exists()
        df = pd.read_csv(output_path, index_col=0)
        assert "point" in df.columns
        assert len(df) == 6

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
        assert result.exit_code == 0, result.output
        assert "Fitting auto_ets on 'y'" in result.output
        # AutoETS produces an ETS(...) model name.
        payload = json.loads(result.output[result.output.index('{\n  "model"'):])
        assert payload["model"].startswith("ETS")

    def test_auto_select(self, sample_csv: Path) -> None:
        """--model auto_select fits AutoSelect and forecasts."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "forecast",
                "--data", str(sample_csv),
                "--target", "y",
                "--model", "auto_select",
                "--horizon", "6",
                "--no-plot",
                "--no-cv",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Fitting auto_select on 'y'" in result.output
        payload = json.loads(result.output[result.output.index('{\n  "model"'):])
        assert len(payload["point"]) == 6

    def test_cross_validation(self, sample_csv: Path) -> None:
        """--cv runs expanding-window CV and reports metrics."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "forecast",
                "--data", str(sample_csv),
                "--target", "y",
                "--horizon", "6",
                "--no-plot",
                "--cv",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Running cross-validation..." in result.output
        assert "CV Metrics:" in result.output
        assert "Warning: CV failed" not in result.output
        # The CV metrics line prints a dict repr; the JSON payload begins at the
        # multi-line object that opens with the "model" key.
        json_start = result.output.index('{\n  "model"')
        payload = json.loads(result.output[json_start:])
        assert "metrics" in payload
        assert "rmse" in payload["metrics"]

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
        assert result.exit_code == 0, result.output
        assert "Metadata:" in result.output

    def test_missing_data_error(self) -> None:
        """Non-existent file produces clear error."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["forecast", "--data", "/nonexistent/path.csv", "--target", "y"],
        )
        assert result.exit_code != 0

    def test_missing_target_error(self, sample_csv: Path) -> None:
        """Unknown target column produces a clear error and non-zero exit."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "forecast",
                "--data", str(sample_csv),
                "--target", "does_not_exist",
                "--no-plot",
                "--no-cv",
            ],
        )
        assert result.exit_code != 0
        assert "not found in data" in result.output
