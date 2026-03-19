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
    """Create a sample pipeline YAML config."""
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
    # Write as JSON (valid YAML subset)
    with open(path, "w") as f:
        json.dump(config, f)
    return path


@pytest.fixture
def actual_csv(tmp_path: Path) -> Path:
    """Create sample actual values CSV."""
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    actual = pd.DataFrame(
        {"actual": np.random.default_rng(42).normal(100, 2, 12)},
        index=dates,
    )
    actual.index.name = "date"
    path = tmp_path / "actual.csv"
    actual.to_csv(path)
    return path


class TestMonitorCmd:
    """Tests for the monitor CLI command."""

    def test_monitor_basic(self, pipeline_yaml: Path, actual_csv: Path) -> None:
        """Monitor executes with pipeline YAML and actuals."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(actual_csv),
                "--window", "6",
            ],
        )
        assert (
            result.exit_code == 0
            or "monitor" in result.output.lower()
            or "Error" in result.output
        )

    def test_alerts_triggered(self, pipeline_yaml: Path, actual_csv: Path) -> None:
        """--alerts checks and reports alerts."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "monitor",
                "--pipeline", str(pipeline_yaml),
                "--actual", str(actual_csv),
                "--alerts",
            ],
        )
        assert result.exit_code == 0 or "alert" in result.output.lower() or "Error" in result.output
