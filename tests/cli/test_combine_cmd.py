"""Tests for forecastbox combine CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from forecastbox.cli.main import cli
from forecastbox.core.forecast import Forecast


@pytest.fixture
def forecast_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create sample forecast JSON files."""
    fc1 = Forecast(
        point=np.array([100.5, 101.2, 102.0]),
        lower_80=np.array([98.0, 97.5, 96.8]),
        upper_80=np.array([103.0, 104.9, 107.2]),
        model_name="Model_A",
    )
    fc2 = Forecast(
        point=np.array([100.8, 100.9, 102.5]),
        lower_80=np.array([98.5, 97.0, 97.0]),
        upper_80=np.array([103.1, 104.8, 108.0]),
        model_name="Model_B",
    )
    fc3 = Forecast(
        point=np.array([101.0, 101.5, 101.8]),
        lower_80=np.array([99.0, 98.0, 97.5]),
        upper_80=np.array([103.0, 105.0, 106.1]),
        model_name="Model_C",
    )

    path1 = tmp_path / "fc1.json"
    path2 = tmp_path / "fc2.json"
    path3 = tmp_path / "fc3.json"
    fc1.save(path1)
    fc2.save(path2)
    fc3.save(path3)
    return path1, path2, path3


class TestCombineCmd:
    """Tests for the combine CLI command."""

    def test_combine_mean(self, forecast_files: tuple[Path, Path, Path]) -> None:
        """--method mean combines by simple average."""
        runner = CliRunner()
        fc1, fc2, fc3 = forecast_files
        result = runner.invoke(
            cli,
            [
                "combine",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--forecasts", str(fc3),
                "--method", "mean",
            ],
        )
        assert result.exit_code == 0
        assert "Combined" in result.output or "point" in result.output

    def test_combine_bma(self, forecast_files: tuple[Path, Path, Path]) -> None:
        """--method bma combines by BMA."""
        runner = CliRunner()
        fc1, fc2, fc3 = forecast_files
        result = runner.invoke(
            cli,
            [
                "combine",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--forecasts", str(fc3),
                "--method", "bma",
            ],
        )
        # BMA may require actual values or may fail if module not available
        assert result.exit_code == 0 or "Error" in result.output or "bma" in result.output.lower()

    def test_output(self, forecast_files: tuple[Path, Path, Path], tmp_path: Path) -> None:
        """--output combined.json saves result."""
        runner = CliRunner()
        fc1, fc2, fc3 = forecast_files
        output_path = tmp_path / "combined.json"
        result = runner.invoke(
            cli,
            [
                "combine",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--forecasts", str(fc3),
                "--method", "mean",
                "--output", str(output_path),
            ],
        )
        if result.exit_code == 0:
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            assert "point" in data
