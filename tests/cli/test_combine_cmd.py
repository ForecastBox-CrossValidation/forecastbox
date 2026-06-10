"""Tests for forecastbox combine CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from forecastbox.cli.main import cli
from forecastbox.core.forecast import Forecast

_POINTS = np.array(
    [[100.5, 101.2, 102.0], [100.8, 100.9, 102.5], [101.0, 101.5, 101.8]]
)


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


@pytest.fixture
def actuals_file(tmp_path: Path) -> Path:
    """Create a CSV of realized values aligned with the forecast horizon."""
    idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    series = pd.Series([100.6, 101.1, 102.2], index=idx, name="y")
    path = tmp_path / "actuals.csv"
    series.to_csv(path)
    return path


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
        assert result.exit_code == 0, result.output
        assert "Combined(Simple-mean)" in result.output
        # Point forecast is the elementwise mean of the three inputs.
        payload = json.loads(result.output[result.output.index("{") :])
        assert payload["method"] == "mean"
        assert payload["n_models"] == 3
        assert payload["models"] == ["Model_A", "Model_B", "Model_C"]
        np.testing.assert_allclose(payload["point"], np.mean(_POINTS, axis=0))

    def test_combine_median(self, forecast_files: tuple[Path, Path, Path]) -> None:
        """--method median combines by elementwise median."""
        runner = CliRunner()
        fc1, fc2, fc3 = forecast_files
        result = runner.invoke(
            cli,
            [
                "combine",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--forecasts", str(fc3),
                "--method", "median",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Combined(Simple-median)" in result.output
        payload = json.loads(result.output[result.output.index("{") :])
        np.testing.assert_allclose(payload["point"], np.median(_POINTS, axis=0))

    def test_combine_inverse_mse(
        self, forecast_files: tuple[Path, Path, Path], actuals_file: Path
    ) -> None:
        """--method inverse_mse fits weights from actuals."""
        runner = CliRunner()
        fc1, fc2, fc3 = forecast_files
        result = runner.invoke(
            cli,
            [
                "combine",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--forecasts", str(fc3),
                "--method", "inverse_mse",
                "--actual", str(actuals_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Combined(WeightedCombiner)" in result.output
        payload = json.loads(result.output[result.output.index("{") :])
        assert payload["method"] == "inverse_mse"
        assert len(payload["point"]) == 3

    def test_combine_ols(
        self, forecast_files: tuple[Path, Path, Path], actuals_file: Path
    ) -> None:
        """--method ols fits OLS combination weights."""
        runner = CliRunner()
        fc1, fc2, fc3 = forecast_files
        result = runner.invoke(
            cli,
            [
                "combine",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--forecasts", str(fc3),
                "--method", "ols",
                "--actual", str(actuals_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Combined(OLSCombiner)" in result.output

    def test_combine_bma(
        self, forecast_files: tuple[Path, Path, Path], actuals_file: Path
    ) -> None:
        """--method bma combines by Bayesian Model Averaging."""
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
                "--actual", str(actuals_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Combined(BMA)" in result.output
        payload = json.loads(result.output[result.output.index("{") :])
        # BMA produces intervals via the posterior variance.
        assert "lower_95" in payload
        assert "upper_95" in payload

    def test_combine_stacking(
        self, forecast_files: tuple[Path, Path, Path], actuals_file: Path
    ) -> None:
        """--method stacking trains a meta-learner (requires scikit-learn)."""
        runner = CliRunner()
        fc1, fc2, fc3 = forecast_files
        result = runner.invoke(
            cli,
            [
                "combine",
                "--forecasts", str(fc1),
                "--forecasts", str(fc2),
                "--forecasts", str(fc3),
                "--method", "stacking",
                "--actual", str(actuals_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Combined(Stacking" in result.output

    def test_fit_method_requires_actual(
        self, forecast_files: tuple[Path, Path, Path]
    ) -> None:
        """Fit-based methods error out cleanly when --actual is missing."""
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
        assert result.exit_code == 1
        assert "requires --actual" in result.output

    def test_requires_two_forecasts(
        self, forecast_files: tuple[Path, Path, Path]
    ) -> None:
        """At least two forecasts are required."""
        runner = CliRunner()
        fc1, _fc2, _fc3 = forecast_files
        result = runner.invoke(
            cli,
            ["combine", "--forecasts", str(fc1), "--method", "mean"],
        )
        assert result.exit_code == 1
        assert "at least 2 forecasts" in result.output

    def test_output(
        self, forecast_files: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        """--output combined.json saves the result as JSON."""
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
        assert result.exit_code == 0, result.output
        assert output_path.exists()
        # The saved file round-trips through Forecast.load.
        loaded = Forecast.load(output_path)
        assert loaded.model_name == "Combined(Simple-mean)"
        assert len(loaded.point) == 3

        data = json.loads(output_path.read_text())
        assert "point" in data
        np.testing.assert_allclose(data["point"], np.mean(_POINTS, axis=0))
