"""End-to-end integration tests for all forecast combination methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecastbox.combination import (
    BMACombiner,
    OLSCombiner,
    OptimalCombiner,
    SimpleCombiner,
    TimeVaryingCombiner,
    WeightedCombiner,
)
from forecastbox.core.forecast import Forecast


@pytest.fixture
def integration_data() -> dict:
    """Create synthetic data for integration testing.

    Returns a dict with:
    - actual_train: realized values for training (T=120)
    - forecasts_train: 3 forecast arrays for training
    - forecasts: 3 Forecast objects for combination (H=12)
    - actual_test: realized values for the forecast horizon
    """
    rng = np.random.default_rng(42)
    t_train = 120
    h = 12

    # Generate actual training data
    actual_train = 100.0 + np.cumsum(rng.normal(0.1, 1.0, size=t_train))

    # Generate training forecasts with different accuracy levels
    # Model A: best (low noise, slight bias)
    fc_train_a = actual_train + rng.normal(0.1, 1.0, size=t_train)
    # Model B: medium (moderate noise)
    fc_train_b = actual_train + rng.normal(0.0, 2.5, size=t_train)
    # Model C: worst (high noise, negative bias)
    fc_train_c = actual_train + rng.normal(-0.5, 4.0, size=t_train)

    forecasts_train = [fc_train_a, fc_train_b, fc_train_c]

    # Generate test period
    actual_test = actual_train[-1] + np.cumsum(rng.normal(0.1, 1.0, size=h))

    # Create Forecast objects for the test horizon
    index = pd.date_range("2024-01", periods=h, freq="MS")

    fc_a_point = actual_test + rng.normal(0.1, 1.0, size=h)
    fc_b_point = actual_test + rng.normal(0.0, 2.5, size=h)
    fc_c_point = actual_test + rng.normal(-0.5, 4.0, size=h)

    fc_a = Forecast(
        point=fc_a_point,
        lower_80=fc_a_point - 1.28 * 1.0,
        upper_80=fc_a_point + 1.28 * 1.0,
        lower_95=fc_a_point - 1.96 * 1.0,
        upper_95=fc_a_point + 1.96 * 1.0,
        index=index,
        model_name="ARIMA",
    )
    fc_b = Forecast(
        point=fc_b_point,
        lower_80=fc_b_point - 1.28 * 2.5,
        upper_80=fc_b_point + 1.28 * 2.5,
        lower_95=fc_b_point - 1.96 * 2.5,
        upper_95=fc_b_point + 1.96 * 2.5,
        index=index,
        model_name="ETS",
    )
    fc_c = Forecast(
        point=fc_c_point,
        lower_80=fc_c_point - 1.28 * 4.0,
        upper_80=fc_c_point + 1.28 * 4.0,
        lower_95=fc_c_point - 1.96 * 4.0,
        upper_95=fc_c_point + 1.96 * 4.0,
        index=index,
        model_name="Theta",
    )

    return {
        "actual_train": actual_train,
        "forecasts_train": forecasts_train,
        "forecasts": [fc_a, fc_b, fc_c],
        "actual_test": actual_test,
    }


class TestIntegration:
    """End-to-end integration tests for all combiners."""

    def test_all_combiners_produce_valid_forecast(
        self, integration_data: dict
    ) -> None:
        """All 7 combiners produce a valid Forecast object."""
        actual_train = integration_data["actual_train"]
        forecasts_train = integration_data["forecasts_train"]
        forecasts = integration_data["forecasts"]

        combiners = [
            ("SimpleCombiner(mean)", SimpleCombiner(method="mean")),
            ("SimpleCombiner(median)", SimpleCombiner(method="median")),
            ("SimpleCombiner(trimmed)", SimpleCombiner(method="trimmed", trim_fraction=0.1)),
            ("WeightedCombiner(inverse_mse)", WeightedCombiner(method="inverse_mse")),
            ("WeightedCombiner(aic_weights)", WeightedCombiner(method="aic_weights")),
            ("WeightedCombiner(bic_weights)", WeightedCombiner(method="bic_weights")),
            ("OLSCombiner(constrained)", OLSCombiner(intercept=False, constrained=True)),
            ("OLSCombiner(unconstrained)", OLSCombiner(intercept=False, constrained=False)),
            ("OLSCombiner(intercept)", OLSCombiner(intercept=True, constrained=False)),
            ("BMACombiner", BMACombiner()),
            ("TimeVaryingCombiner", TimeVaryingCombiner(decay=0.95)),
            ("OptimalCombiner", OptimalCombiner(shrinkage=0.0)),
        ]

        for name, combiner in combiners:
            # Fit (SimpleCombiner's fit is a no-op)
            if not isinstance(combiner, SimpleCombiner):
                combiner.fit(forecasts_train, actual_train)

            # Combine
            result = combiner.combine(forecasts)

            # Validate result
            assert isinstance(result, Forecast), f"{name}: result is not Forecast"
            assert len(result.point) == 12, f"{name}: wrong horizon"
            assert not np.any(np.isnan(result.point)), f"{name}: NaN in point"
            assert result.model_name != "", f"{name}: empty model_name"

    def test_all_weights_sum_to_one(self, integration_data: dict) -> None:
        """All fitted combiner weights sum to 1 (except OLS unconstrained)."""
        actual_train = integration_data["actual_train"]
        forecasts_train = integration_data["forecasts_train"]

        combiners_constrained = [
            ("WeightedCombiner(inverse_mse)", WeightedCombiner(method="inverse_mse")),
            ("WeightedCombiner(aic_weights)", WeightedCombiner(method="aic_weights")),
            ("WeightedCombiner(bic_weights)", WeightedCombiner(method="bic_weights")),
            ("OLSCombiner(constrained)", OLSCombiner(intercept=False, constrained=True)),
            ("BMACombiner", BMACombiner()),
            ("TimeVaryingCombiner", TimeVaryingCombiner(decay=0.95)),
            ("OptimalCombiner", OptimalCombiner(shrinkage=0.0)),
        ]

        for name, combiner in combiners_constrained:
            combiner.fit(forecasts_train, actual_train)
            assert combiner.weights_ is not None, f"{name}: no weights"
            weight_sum = np.sum(combiner.weights_)
            assert abs(weight_sum - 1.0) < 1e-6, (
                f"{name}: weights sum to {weight_sum}, expected 1.0"
            )

    def test_combined_rmse_leq_worst_individual(
        self, integration_data: dict
    ) -> None:
        """Combined RMSE is <= worst individual model RMSE for fitted combiners."""
        actual_train = integration_data["actual_train"]
        actual_test = integration_data["actual_test"]
        forecasts_train = integration_data["forecasts_train"]
        forecasts = integration_data["forecasts"]

        # Compute individual RMSEs on test set
        individual_rmses = []
        for fc in forecasts:
            rmse = np.sqrt(np.mean((actual_test - fc.point) ** 2))
            individual_rmses.append(rmse)
        worst_rmse = max(individual_rmses)

        combiners = [
            ("WeightedCombiner", WeightedCombiner(method="inverse_mse")),
            ("OLSCombiner", OLSCombiner(constrained=True)),
            ("BMACombiner", BMACombiner()),
            ("TimeVaryingCombiner", TimeVaryingCombiner(decay=0.95)),
            ("OptimalCombiner", OptimalCombiner(shrinkage=0.1)),
        ]

        for name, combiner in combiners:
            combiner.fit(forecasts_train, actual_train)
            combined = combiner.combine(forecasts)

            combined_rmse = np.sqrt(np.mean((actual_test - combined.point) ** 2))

            # Combined should not be catastrophically worse than worst individual
            # (allowing some tolerance for out-of-sample)
            assert combined_rmse <= worst_rmse * 1.5, (
                f"{name}: combined RMSE {combined_rmse:.2f} > "
                f"1.5 * worst individual RMSE {worst_rmse:.2f}"
            )

    def test_stacking_integration(self, integration_data: dict) -> None:
        """StackingCombiner integrates correctly with all other components."""
        pytest.importorskip("sklearn")
        from forecastbox.combination import StackingCombiner

        actual_train = integration_data["actual_train"]
        forecasts_train = integration_data["forecasts_train"]
        forecasts = integration_data["forecasts"]

        combiner = StackingCombiner(meta_learner="ridge", cv_folds=5)
        combiner.fit(forecasts_train, actual_train)
        result = combiner.combine(forecasts)

        assert isinstance(result, Forecast)
        assert len(result.point) == 12
        assert not np.any(np.isnan(result.point))

    def test_imports_from_combination(self) -> None:
        """All 7 combiners can be imported from forecastbox.combination."""
        # Also check StackingCombiner import
        from forecastbox.combination import (
            BaseCombiner,
            BMACombiner,
            OLSCombiner,
            OptimalCombiner,
            SimpleCombiner,
            StackingCombiner,
            TimeVaryingCombiner,
            WeightedCombiner,
        )

        # Verify they are classes
        assert isinstance(SimpleCombiner, type)
        assert isinstance(WeightedCombiner, type)
        assert isinstance(OLSCombiner, type)
        assert isinstance(StackingCombiner, type)
        assert isinstance(BMACombiner, type)
        assert isinstance(TimeVaryingCombiner, type)
        assert isinstance(OptimalCombiner, type)
        assert isinstance(BaseCombiner, type)

    def test_combined_forecast_has_intervals(
        self, integration_data: dict
    ) -> None:
        """Combined forecasts preserve prediction intervals when available."""
        actual_train = integration_data["actual_train"]
        forecasts_train = integration_data["forecasts_train"]
        forecasts = integration_data["forecasts"]

        combiner = WeightedCombiner(method="inverse_mse")
        combiner.fit(forecasts_train, actual_train)
        result = combiner.combine(forecasts)

        assert result.lower_80 is not None
        assert result.upper_80 is not None
        assert result.lower_95 is not None
        assert result.upper_95 is not None

        # Intervals should be ordered: lower_95 < lower_80 < point < upper_80 < upper_95
        assert np.all(result.lower_95 <= result.lower_80)
        assert np.all(result.lower_80 <= result.point)
        assert np.all(result.point <= result.upper_80)
        assert np.all(result.upper_80 <= result.upper_95)

    def test_combined_forecast_metadata(
        self, integration_data: dict
    ) -> None:
        """Combined forecasts include metadata about the combination."""
        forecasts = integration_data["forecasts"]

        combiner = SimpleCombiner(method="mean")
        result = combiner.combine(forecasts)

        assert "combiner" in result.metadata
        assert "models" in result.metadata
        assert len(result.metadata["models"]) == 3
        assert "ARIMA" in result.metadata["models"]
        assert "ETS" in result.metadata["models"]
        assert "Theta" in result.metadata["models"]
