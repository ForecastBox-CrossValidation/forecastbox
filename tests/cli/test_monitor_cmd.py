"""Tests for forecastbox monitor CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from forecastbox.cli.main import cli


@pytest.fixture
def pipeline_yaml(tmp_path: Path) -> Path:
    """Create a sample pipeline config (JSON is a valid YAML subset)."""
    config = {
        "name": "test_pipeline",
        "model": "auto_arima",
        "target": "ipca",
        "horizon": 12,
        "thresholds": {
            "rmse": 0.5,
            "mae": 0.4,
        },
    }
    path = tmp_path / "pipeline.yaml"
    with open(path, "w") as f:
        json.dump(config, f)
    return path


@pytest.fixture
def actual_csv(tmp_path: Path) -> Path:
    """Create a CSV with date index, actual, forecast and 95% interval columns."""
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(42)
    actual = 100 + np.cumsum(rng.normal(0, 1, 24))
    forecast = actual + rng.normal(0, 0.8, 24)
    df = pd.DataFrame(
        {
            "actual": actual,
            "forecast": forecast,
            "lower_95": forecast - 3.0,
            "upper_95": forecast + 3.0,
        },
        index=dates,
    )
    df.index.name = "date"
    path = tmp_path / "actual.csv"
    df.to_csv(path)
    return path


@pytest.fixture
def actual_csv_no_intervals(tmp_path: Path) -> Path:
    """CSV with only actual and forecast columns (no intervals)."""
    dates = pd.date_range("2024-01-01", periods=18, freq="MS")
    rng = np.random.default_rng(7)
    actual = 50 + np.cumsum(rng.normal(0, 0.5, 18))
    forecast = actual + rng.normal(0, 0.3, 18)
    df = pd.DataFrame({"actual": actual, "forecast": forecast}, index=dates)
    df.index.name = "date"
    path = tmp_path / "actual_no_int.csv"
    df.to_csv(path)
    return path


class TestMonitorCmd:
    """Tests for the monitor CLI command."""

    def test_monitor_basic(self, pipeline_yaml: Path, actual_csv: Path) -> None:
        """Monitor runs end-to-end and prints the accuracy report."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(actual_csv),
                "--window", "6",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "FORECAST MONITOR REPORT" in result.output
        assert "Overall Metrics" in result.output
        assert "Hit Rate" in result.output
        assert "Rolling RMSE" in result.output

    def test_alerts_triggered(self, pipeline_yaml: Path, actual_csv: Path) -> None:
        """--alerts reports threshold breaches from the config thresholds."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(actual_csv),
                "--alerts",
                "--window", "6",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output
        # thresholds rmse=0.5/mae=0.4 are small -> at least one alert expected
        assert "=== ALERTS ===" in result.output
        assert "exceeds threshold" in result.output

    def test_no_alerts_high_thresholds(
        self, tmp_path: Path, actual_csv: Path
    ) -> None:
        """With generous thresholds and no degradation, no alerts fire."""
        config = {
            "model": "auto_arima",
            "horizon": 12,
            "thresholds": {"rmse": 1000.0, "mae": 1000.0},
        }
        pipeline_path = tmp_path / "pipeline_high.json"
        with open(pipeline_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_path),
                "--actual", str(actual_csv),
                "--window", "12",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "No alerts triggered." in result.output

    def test_output_report_saved(
        self, pipeline_yaml: Path, actual_csv: Path, tmp_path: Path
    ) -> None:
        """--output writes a JSON report with the real metric keys."""
        out = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(actual_csv),
                "--window", "6",
                "--output", str(out),
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        data = json.loads(out.read_text())
        assert set(data["overall_metrics"]) == {"rmse", "mae", "mfe", "mape"}
        for key in ("rmse", "mae", "mfe", "mape"):
            assert isinstance(data["overall_metrics"][key], (int, float))
        assert data["window"] == 6
        assert data["metric"] == "rmse"
        assert data["hit_rate"] == 1.0  # all actuals within +/-3 band
        assert isinstance(data["alerts"], list)
        assert isinstance(data["rolling_metric"], dict)

    def test_metric_mae(self, pipeline_yaml: Path, actual_csv: Path) -> None:
        """--metric mae selects the MAE rolling series."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(actual_csv),
                "--metric", "mae",
                "--window", "6",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Rolling MAE (window=6)" in result.output

    def test_no_intervals_hit_rate_zero(
        self, pipeline_yaml: Path, actual_csv_no_intervals: Path
    ) -> None:
        """Without interval columns the command still succeeds (hit rate 0)."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(actual_csv_no_intervals),
                "--window", "6",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "FORECAST MONITOR REPORT" in result.output

    def test_plot_saved(
        self, pipeline_yaml: Path, actual_csv: Path, tmp_path: Path
    ) -> None:
        """--plot saves a PNG next to the output report."""
        out = tmp_path / "report.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(actual_csv),
                "--window", "6",
                "--output", str(out),
                "--plot",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "report.png").exists()

    def test_missing_columns_error(
        self, pipeline_yaml: Path, tmp_path: Path
    ) -> None:
        """A CSV without 'forecast' column errors out cleanly."""
        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        df = pd.DataFrame({"actual": np.arange(6.0)}, index=dates)
        df.index.name = "date"
        bad = tmp_path / "bad.csv"
        df.to_csv(bad)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(bad),
                "--no-plot",
            ],
        )
        assert result.exit_code == 1
        assert "must contain 'actual' and 'forecast' columns" in result.output
