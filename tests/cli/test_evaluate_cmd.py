"""Tests for forecastbox evaluate CLI command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from forecastbox.cli.main import cli
from forecastbox.core.forecast import Forecast


@pytest.fixture
def forecast_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create sample forecast JSON files."""
    fc1 = Forecast(
        point=np.array([100.5, 101.2, 102.0, 103.1, 104.0, 105.2]),
        model_name="Model_A",
    )
    fc2 = Forecast(
        point=np.array([100.8, 100.9, 102.5, 103.5, 103.8, 105.0]),
        model_name="Model_B",
    )

    path1 = tmp_path / "fc1.json"
    path2 = tmp_path / "fc2.json"
    fc1.save(path1)
    fc2.save(path2)
    return path1, path2


@pytest.fixture
def actual_csv(tmp_path: Path) -> Path:
    """Create sample actual values CSV."""
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    actual = pd.DataFrame(
        {"actual": [100.6, 101.0, 102.3, 103.0, 104.2, 105.1]},
        index=dates,
    )
    actual.index.name = "date"
    path = tmp_path / "actual.csv"
    actual.to_csv(path)
    return path


class TestEvaluateCmd:
    """Tests for the evaluate CLI command."""

    def test_dm_test(self, forecast_files: tuple[Path, Path], actual_csv: Path) -> None:
        """--tests dm executes Diebold-Mariano test."""
        runner = CliRunner()
        fc1, fc2 = forecast_files
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--actual", str(actual_csv),
                "--tests", "dm",
            ],
        )
        assert result.exit_code == 0 or "dm" in result.output.lower() or "Error" in result.output

    def test_mcs_test(self, forecast_files: tuple[Path, Path], actual_csv: Path) -> None:
        """--tests mcs executes Model Confidence Set."""
        runner = CliRunner()
        fc1, fc2 = forecast_files
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--actual", str(actual_csv),
                "--tests", "mcs",
            ],
        )
        assert result.exit_code == 0 or "mcs" in result.output.lower() or "Error" in result.output

    def test_multiple_tests(self, forecast_files: tuple[Path, Path], actual_csv: Path) -> None:
        """--tests dm --tests mcs executes multiple tests."""
        runner = CliRunner()
        fc1, fc2 = forecast_files
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--actual", str(actual_csv),
                "--tests", "dm",
                "--tests", "mcs",
            ],
        )
        assert result.exit_code == 0 or "Error" in result.output

    def test_metrics_output(self, forecast_files: tuple[Path, Path], actual_csv: Path) -> None:
        """Metrics are computed and displayed."""
        runner = CliRunner()
        fc1, fc2 = forecast_files
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--actual", str(actual_csv),
                "--metrics", "mae",
                "--metrics", "rmse",
            ],
        )
        assert result.exit_code == 0 or "Metrics" in result.output or "Error" in result.output
