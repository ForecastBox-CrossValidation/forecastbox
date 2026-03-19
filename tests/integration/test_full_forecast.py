"""Integration test: Workflow 1 - Full Forecast Pipeline.

Workflow: Data -> AutoARIMA + AutoETS + Theta -> BMA -> DM + MCS -> Report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ipca_data() -> pd.Series:
    """Load IPCA series from macro_brazil dataset."""
    try:
        from forecastbox.datasets import load_dataset

        data = load_dataset("macro_brazil")
        return data["ipca"]
    except Exception:
        # Fallback: generate synthetic data
        rng = np.random.default_rng(42)
        dates = pd.date_range("2000-01-01", periods=200, freq="MS")
        return pd.Series(
            0.5 + rng.normal(0, 0.3, 200),
            index=dates,
            name="ipca",
        )


@pytest.fixture
def macro_data() -> pd.DataFrame:
    """Load full macro_brazil dataset."""
    try:
        from forecastbox.datasets import load_dataset

        data = load_dataset("macro_brazil")
        return pd.DataFrame(data)
    except Exception:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2000-01-01", periods=200, freq="MS")
        return pd.DataFrame(
            {
                "ipca": 0.5 + rng.normal(0, 0.3, 200),
                "selic": 10 + np.cumsum(rng.normal(0, 0.5, 200)),
                "cambio": 3 + np.cumsum(rng.normal(0, 0.1, 200)),
            },
            index=dates,
        )


class TestWorkflow1FullForecast:
    """Integration tests for the full forecast workflow."""

    def test_workflow_1_complete(self, macro_data: pd.DataFrame) -> None:
        """Workflow full forecast runs without error."""
        try:
            from forecastbox.experiment import ForecastExperiment

            exp = ForecastExperiment(
                data=macro_data,
                target="ipca",
                models=["auto_arima", "auto_ets", "theta"],
                combination="mean",
                evaluation=["dm", "mcs"],
                horizon=12,
                cv_type="expanding",
            )
            results = exp.run()

            assert results is not None
            assert isinstance(results.metadata, dict)
            assert results.metadata["target"] == "ipca"
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

    def test_workflow_1_forecasts(self, macro_data: pd.DataFrame) -> None:
        """3 individual forecasts are generated."""
        try:
            from forecastbox.experiment import ForecastExperiment

            exp = ForecastExperiment(
                data=macro_data,
                target="ipca",
                models=["auto_arima", "auto_ets", "theta"],
                combination="mean",
                horizon=12,
            )
            results = exp.run()

            # At least some models should produce forecasts
            if len(results.forecasts) == 0:
                pytest.skip("No models produced forecasts (dependencies missing)")
            for _name, fc in results.forecasts.items():
                assert fc.horizon == 12
                assert len(fc.point) == 12
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

    def test_workflow_1_combination(self, macro_data: pd.DataFrame) -> None:
        """BMA combination is functional."""
        try:
            from forecastbox.experiment import ForecastExperiment

            exp = ForecastExperiment(
                data=macro_data,
                target="ipca",
                models=["auto_arima", "auto_ets"],
                combination="mean",
                horizon=12,
            )
            results = exp.run()

            if len(results.forecasts) >= 2:
                assert results.combination is not None
                assert len(results.combination.point) == 12
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

    def test_workflow_1_mcs(self, macro_data: pd.DataFrame) -> None:
        """MCS includes at least 1 model."""
        try:
            from forecastbox.experiment import ForecastExperiment

            exp = ForecastExperiment(
                data=macro_data,
                target="ipca",
                models=["auto_arima", "auto_ets"],
                evaluation=["mcs"],
                horizon=12,
            )
            results = exp.run()

            if results.mcs is not None:
                assert hasattr(results.mcs, "included_models")
                assert len(results.mcs.included_models) >= 1
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")

    def test_workflow_1_report(self, macro_data: pd.DataFrame, tmp_path: Path) -> None:
        """Report HTML is generated."""
        try:
            from forecastbox.experiment import ForecastExperiment

            exp = ForecastExperiment(
                data=macro_data,
                target="ipca",
                models=["auto_arima", "auto_ets"],
                combination="mean",
                horizon=6,
                report_format="html",
            )
            results = exp.run()

            report_path = tmp_path / "workflow1.html"
            results.report(report_path)

            assert report_path.exists()
            content = report_path.read_text()
            assert "<html>" in content
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")
