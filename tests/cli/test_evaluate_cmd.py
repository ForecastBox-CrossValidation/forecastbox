"""Tests for forecastbox evaluate CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from forecastbox.cli.main import cli
from forecastbox.core.forecast import Forecast


@pytest.fixture
def forecast_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create sample forecast JSON files.

    Model_A is a more accurate forecast than Model_B so the statistical
    tests have a clear, deterministic outcome.
    """
    rng = np.random.default_rng(0)
    n = 40
    actual = 100.0 + np.cumsum(rng.standard_normal(n))

    fc1 = Forecast(
        point=actual + rng.standard_normal(n) * 0.5,
        model_name="Model_A",
    )
    fc2 = Forecast(
        point=actual + rng.standard_normal(n) * 1.2,
        model_name="Model_B",
    )

    path1 = tmp_path / "fc1.json"
    path2 = tmp_path / "fc2.json"
    fc1.save(path1)
    fc2.save(path2)

    # Stash actual on the fixture dir for the actual_csv fixture to reuse.
    np.save(tmp_path / "_actual.npy", actual)
    return path1, path2


@pytest.fixture
def actual_csv(tmp_path: Path, forecast_files: tuple[Path, Path]) -> Path:
    """Create the matching actual values CSV (same series used for forecasts)."""
    actual = np.load(tmp_path / "_actual.npy")
    dates = pd.date_range("2020-01-01", periods=len(actual), freq="MS")
    df = pd.DataFrame({"actual": actual}, index=dates)
    df.index.name = "date"
    path = tmp_path / "actual.csv"
    df.to_csv(path)
    return path


class TestEvaluateCmd:
    """Tests for the evaluate CLI command."""

    def test_dm_test(self, forecast_files: tuple[Path, Path], actual_csv: Path) -> None:
        """--tests dm executes Diebold-Mariano test successfully."""
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
        assert result.exit_code == 0, result.output
        assert "DM Test" in result.output
        assert "Statistic:" in result.output
        assert "P-value:" in result.output

    def test_mcs_test(self, forecast_files: tuple[Path, Path], actual_csv: Path) -> None:
        """--tests mcs executes Model Confidence Set successfully."""
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
        assert result.exit_code == 0, result.output
        assert "MCS Test" in result.output
        assert "Included models:" in result.output
        # Model_A is the more accurate model and must survive.
        assert "Model_A" in result.output

    def test_gw_test(self, forecast_files: tuple[Path, Path], actual_csv: Path) -> None:
        """--tests gw executes Giacomini-White test successfully."""
        runner = CliRunner()
        fc1, fc2 = forecast_files
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--actual", str(actual_csv),
                "--tests", "gw",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "GW Test" in result.output
        assert "Statistic:" in result.output

    def test_mz_test(self, forecast_files: tuple[Path, Path], actual_csv: Path) -> None:
        """--tests mz executes Mincer-Zarnowitz test successfully."""
        runner = CliRunner()
        fc1, fc2 = forecast_files
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--actual", str(actual_csv),
                "--tests", "mz",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "MZ Test" in result.output
        assert "Slope (beta):" in result.output

    def test_encompassing_test(
        self, forecast_files: tuple[Path, Path], actual_csv: Path
    ) -> None:
        """--tests encompassing executes the encompassing test successfully."""
        runner = CliRunner()
        fc1, fc2 = forecast_files
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--actual", str(actual_csv),
                "--tests", "encompassing",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "ENCOMPASSING Test" in result.output
        assert "Lambda:" in result.output

    def test_multiple_tests(
        self, forecast_files: tuple[Path, Path], actual_csv: Path
    ) -> None:
        """--tests dm --tests mcs executes multiple tests successfully."""
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
        assert result.exit_code == 0, result.output
        assert "DM Test" in result.output
        assert "MCS Test" in result.output

    def test_metrics_output(
        self, forecast_files: tuple[Path, Path], actual_csv: Path
    ) -> None:
        """Metrics are computed and displayed for each model."""
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
        assert result.exit_code == 0, result.output
        assert "=== Metrics ===" in result.output
        assert "Model_A:" in result.output
        assert "Model_B:" in result.output
        assert "mae:" in result.output
        assert "rmse:" in result.output

    def test_output_file(
        self, forecast_files: tuple[Path, Path], actual_csv: Path, tmp_path: Path
    ) -> None:
        """--output writes a JSON results file with metrics and test results."""
        runner = CliRunner()
        fc1, fc2 = forecast_files
        out_path = tmp_path / "results.json"
        result = runner.invoke(
            cli,
            [
                "evaluate",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--actual", str(actual_csv),
                "--tests", "dm",
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert data["models"] == ["Model_A", "Model_B"]
        assert "Model_A" in data["metrics"]
        assert "mae" in data["metrics"]["Model_A"]
        assert "dm" in data["tests"]
        assert "statistic" in data["tests"]["dm"]
        assert "pvalue" in data["tests"]["dm"]
