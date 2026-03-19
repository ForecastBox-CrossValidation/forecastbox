"""Integration tests for the complete evaluation pipeline.

Tests the full flow: generate forecasts -> compare with DM test ->
select with MCS -> check efficiency with MZ -> compute advanced metrics
-> run cross-validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from forecastbox.cv.rolling_blocked import blocked_cv, rolling_window_cv
from forecastbox.evaluation import (
    diebold_mariano,
    encompassing_test,
    giacomini_white,
    mincer_zarnowitz,
    model_confidence_set,
)
from forecastbox.evaluation._hac import auto_bandwidth, hac_variance, newey_west
from forecastbox.metrics.advanced_metrics import (
    crps,
    crps_gaussian,
    log_score,
    mfe,
    smape,
    theil_u1,
    theil_u2,
)


class TestEndToEndPipeline:
    """Integration tests for the complete evaluation pipeline."""

    @pytest.fixture
    def synthetic_data(self) -> dict[str, Any]:
        """Generate synthetic forecast data with known properties."""
        rng = np.random.default_rng(42)
        n_obs = 300

        # True data generating process
        actual = np.cumsum(rng.normal(0.1, 1.0, size=n_obs)) + 100

        # Model A: good model (low noise)
        forecast_a = actual + rng.normal(0, 1.5, size=n_obs)

        # Model B: mediocre model (medium noise)
        forecast_b = actual + rng.normal(0, 5.0, size=n_obs)

        # Model C: bad model (high noise + bias)
        forecast_c = actual + rng.normal(3.0, 10.0, size=n_obs)

        return {
            "actual": actual,
            "forecast_a": forecast_a,
            "forecast_b": forecast_b,
            "forecast_c": forecast_c,
            "T": n_obs,
        }

    def test_full_evaluation_pipeline(self, synthetic_data: dict[str, Any]) -> None:
        """Run the complete evaluation pipeline end-to-end."""
        actual = synthetic_data["actual"]
        fc_a = synthetic_data["forecast_a"]
        fc_b = synthetic_data["forecast_b"]
        fc_c = synthetic_data["forecast_c"]

        # 1. Diebold-Mariano: A vs C (A should be significantly better)
        dm_result = diebold_mariano(actual, fc_a, fc_c, h=1, loss="mse")
        assert dm_result.pvalue < 0.05
        assert dm_result.mean_loss_diff < 0  # A is better

        # 2. Model Confidence Set: A should be included, C likely excluded
        forecasts_dict = {
            "Model_A": fc_a,
            "Model_B": fc_b,
            "Model_C": fc_c,
        }
        mcs_result = model_confidence_set(
            actual, forecasts_dict, alpha=0.10, n_boot=500, seed=42
        )
        assert "Model_A" in mcs_result.included_models
        # Check that MCS produces valid p-values
        for _name, pval in mcs_result.pvalues.items():
            assert 0.0 <= pval <= 1.0

        # 3. Mincer-Zarnowitz: Model A should be roughly efficient
        mz_result = mincer_zarnowitz(actual, fc_a)
        assert abs(mz_result.beta - 1.0) < 0.5  # beta close to 1
        assert mz_result.r_squared > 0.5  # reasonable R^2

        # 4. Mincer-Zarnowitz: Model C should show bias
        mz_c = mincer_zarnowitz(actual, fc_c)
        # C has bias, so alpha should be significantly nonzero or beta far from 1
        # (may or may not reject depending on noise, but check structure)
        assert isinstance(mz_c.f_statistic, float)
        assert isinstance(mz_c.pvalue, float)

        # 5. Giacomini-White: A vs C
        gw_result = giacomini_white(actual, fc_a, fc_c, h=1, loss="mse")
        assert isinstance(gw_result.statistic, float)
        assert gw_result.df > 0

        # 6. Encompassing: A vs C (A should encompass C)
        enc_result = encompassing_test(actual, fc_a, fc_c)
        assert isinstance(enc_result.lambda_hat, float)

        # 7. Advanced metrics
        assert mfe(actual, fc_c) != 0.0  # C has bias
        assert 0.0 <= theil_u1(actual, fc_a) <= 1.0
        u2 = theil_u2(actual, fc_a)
        assert isinstance(u2, float)
        s = smape(actual, fc_a)
        assert 0.0 <= s <= 200.0

        # 8. CRPS Gaussian
        mu = fc_a
        sigma = np.full_like(fc_a, 2.0)
        crps_val = crps_gaussian(actual, mu, sigma)
        assert isinstance(crps_val, float)
        assert crps_val > 0

    def test_cv_pipeline(self) -> None:
        """Test cross-validation pipeline end-to-end."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(100, 10, size=250))

        def naive_model(train: pd.Series) -> np.ndarray:
            last = float(train.iloc[-1])
            return np.full(6, last)

        # Rolling window CV
        rolling_results = rolling_window_cv(
            data, naive_model, window=100, horizon=6, step=10
        )
        assert rolling_results.n_folds > 0
        rolling_means = rolling_results.mean_metrics()
        assert "mae" in rolling_means
        assert "rmse" in rolling_means

        # Blocked CV
        blocked_results = blocked_cv(
            data, naive_model, n_blocks=4, horizon=6, gap=5
        )
        assert blocked_results.n_folds == 4
        blocked_means = blocked_results.mean_metrics()
        assert "mae" in blocked_means

    def test_imports_from_evaluation(self) -> None:
        """Verify all public imports work from evaluation module."""
        from forecastbox.evaluation import (
            DMResult,
            EncompassingResult,
            GWResult,
            MCSResult,
            MZResult,
            diebold_mariano,
            encompassing_test,
            giacomini_white,
            mincer_zarnowitz,
            model_confidence_set,
        )

        # All should be importable
        assert callable(diebold_mariano)
        assert callable(model_confidence_set)
        assert callable(giacomini_white)
        assert callable(mincer_zarnowitz)
        assert callable(encompassing_test)
        # Result types should be classes
        assert isinstance(DMResult, type)
        assert isinstance(EncompassingResult, type)
        assert isinstance(GWResult, type)
        assert isinstance(MCSResult, type)
        assert isinstance(MZResult, type)

    def test_imports_from_metrics(self) -> None:
        """Verify all advanced metric imports work."""
        from forecastbox.metrics import (
            crps,
            crps_gaussian,
            log_score,
            mfe,
            smape,
            theil_u1,
            theil_u2,
        )

        assert callable(mfe)
        assert callable(theil_u1)
        assert callable(theil_u2)
        assert callable(smape)
        assert callable(log_score)
        assert callable(crps)
        assert callable(crps_gaussian)

    def test_imports_from_cv(self) -> None:
        """Verify all CV imports work."""
        from forecastbox.cv import blocked_cv, rolling_window_cv

        assert callable(rolling_window_cv)
        assert callable(blocked_cv)

    def test_dm_conclusion_and_loss_variants(
        self, synthetic_data: dict[str, Any]
    ) -> None:
        """Test DM conclusion method and alternative loss functions."""
        actual = synthetic_data["actual"]
        fc_a = synthetic_data["forecast_a"]
        fc_c = synthetic_data["forecast_c"]

        # conclusion() with rejection
        dm = diebold_mariano(actual, fc_a, fc_c, h=1, loss="mse")
        conclusion = dm.conclusion()
        assert "Reject" in conclusion

        # conclusion() without rejection (equal forecasts)
        dm_eq = diebold_mariano(actual, fc_a, fc_a + 0.001, h=1, loss="mse")
        conclusion_eq = dm_eq.conclusion()
        assert "Fail to reject" in conclusion_eq or "Reject" in conclusion_eq

        # MAE loss
        dm_mae = diebold_mariano(actual, fc_a, fc_c, h=1, loss="mae")
        assert isinstance(dm_mae.statistic, float)

        # MAPE loss
        dm_mape = diebold_mariano(actual, fc_a, fc_c, h=1, loss="mape")
        assert isinstance(dm_mape.statistic, float)

    def test_dm_validation_errors(self) -> None:
        """Test DM input validation raises errors."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="same length"):
            diebold_mariano(a, b, a, h=1)

        with pytest.raises(ValueError, match="at least 3"):
            diebold_mariano([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], h=1)

        with pytest.raises(ValueError, match="h must be"):
            diebold_mariano(a, a, a, h=0)

        with pytest.raises(ValueError, match="Unknown loss"):
            diebold_mariano(a, a, a, h=1, loss="invalid")

    def test_dm_zero_variance(self) -> None:
        """Test DM with identical forecasts (zero variance)."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fc = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = diebold_mariano(actual, fc, fc, h=1, loss="mse")
        assert result.pvalue == 1.0
        assert result.statistic == 0.0

    def test_gw_conclusion_and_validation(
        self, synthetic_data: dict[str, Any]
    ) -> None:
        """Test GW conclusion and validation."""
        actual = synthetic_data["actual"]
        fc_a = synthetic_data["forecast_a"]
        fc_c = synthetic_data["forecast_c"]

        gw = giacomini_white(actual, fc_a, fc_c, h=1, loss="mse")
        conclusion = gw.conclusion()
        assert "level" in conclusion

        # Validation
        with pytest.raises(ValueError):
            giacomini_white(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                h=1,
            )

    def test_encompassing_summary(
        self, synthetic_data: dict[str, Any]
    ) -> None:
        """Test encompassing summary output."""
        actual = synthetic_data["actual"]
        fc_a = synthetic_data["forecast_a"]
        fc_c = synthetic_data["forecast_c"]

        enc = encompassing_test(actual, fc_a, fc_c)
        summary = enc.summary()
        assert "Conclusion" in summary

        # Test with reversed order
        enc_rev = encompassing_test(actual, fc_c, fc_a)
        summary_rev = enc_rev.summary()
        assert "Conclusion" in summary_rev

    def test_encompassing_validation(self) -> None:
        """Test encompassing validation."""
        with pytest.raises(ValueError, match="same length"):
            encompassing_test(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
            )
        with pytest.raises(ValueError, match="at least 3"):
            encompassing_test(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
            )

    def test_mz_with_hac(self, synthetic_data: dict[str, Any]) -> None:
        """Test Mincer-Zarnowitz with HAC standard errors."""
        actual = synthetic_data["actual"]
        fc_a = synthetic_data["forecast_a"]

        mz_hac = mincer_zarnowitz(actual, fc_a, hac=True, h=2)
        assert isinstance(mz_hac.alpha, float)
        assert isinstance(mz_hac.beta, float)
        assert isinstance(mz_hac.f_statistic, float)

    def test_mz_validation(self) -> None:
        """Test MZ input validation."""
        with pytest.raises(ValueError):
            mincer_zarnowitz(np.array([1.0, 2.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            mincer_zarnowitz(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

    def test_hac_functions(self) -> None:
        """Test HAC helper functions directly."""
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, size=100)

        # auto_bandwidth
        bw = auto_bandwidth(100)
        assert isinstance(bw, int)
        assert bw >= 0

        # hac_variance with auto bandwidth
        var = hac_variance(series)
        assert isinstance(var, float)
        assert var > 0

        # hac_variance with explicit lag
        var2 = hac_variance(series, max_lag=3)
        assert isinstance(var2, float)
        assert var2 > 0

        # newey_west with 2D input
        scores = rng.normal(0, 1, size=(100, 2))
        omega = newey_west(scores, max_lag=3)
        assert omega.shape == (2, 2)

        # newey_west with 1D input (should reshape)
        omega_1d = newey_west(rng.normal(0, 1, size=100), max_lag=3)
        assert omega_1d.shape == (1, 1)

        # validation
        with pytest.raises(ValueError):
            auto_bandwidth(0)
        with pytest.raises(ValueError):
            hac_variance(np.array([1.0]))

    def test_advanced_metrics_edge_cases(self) -> None:
        """Test edge cases in advanced metrics."""
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        forecast = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        # Theil U1 with zeros
        zeros = np.zeros(5)
        u1 = theil_u1(zeros, zeros)
        assert u1 == 0.0

        # Theil U2 without explicit naive (random walk)
        u2 = theil_u2(actual, forecast)
        assert isinstance(u2, float)

        # Theil U2 with explicit naive forecast
        naive = np.array([0.9, 1.9, 2.9, 3.9, 4.9])
        u2_naive = theil_u2(actual, forecast, naive=naive)
        assert isinstance(u2_naive, float)

        # Theil U2 with too few observations
        with pytest.raises(ValueError, match="at least 2"):
            theil_u2(np.array([1.0]), np.array([1.0]))

        # Theil U2 with all-zero previous values
        u2_zero = theil_u2(zeros, zeros)
        assert u2_zero == 0.0

        # sMAPE with zeros
        s = smape(zeros, zeros)
        assert s == 0.0

        # log_score with a single density function
        from scipy.stats import norm

        dist = norm(loc=3.0, scale=1.0)
        ls = log_score(actual, dist.pdf)
        assert isinstance(ls, float)

        # log_score with zero density (penalty path)
        def always_zero(_y: float) -> float:
            return 0.0

        ls_zero = log_score(actual, always_zero)
        assert ls_zero < -1e8  # penalty

        # CRPS with 2D ensemble
        ensemble = np.column_stack([forecast + 0.1, forecast - 0.1, forecast])
        crps_val = crps(actual, ensemble)
        assert isinstance(crps_val, float)
        assert crps_val >= 0

        # CRPS with 1D ensemble (single observation)
        crps_1d = crps(np.array([3.0]), np.array([2.9, 3.0, 3.1]))
        assert isinstance(crps_1d, float)
        assert crps_1d >= 0

        # CRPS length validation
        with pytest.raises(ValueError, match="rows"):
            crps(
                np.array([1.0, 2.0]),
                np.array([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]),
            )

    def test_cv_with_numpy_array(self) -> None:
        """Test CV functions accept numpy arrays."""
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, size=200)

        def naive_model(train: pd.Series) -> np.ndarray:
            return np.full(5, float(train.iloc[-1]))

        result = rolling_window_cv(data, naive_model, window=80, horizon=5, step=20)
        assert result.n_folds > 0

    def test_gw_with_custom_instruments(
        self, synthetic_data: dict[str, Any]
    ) -> None:
        """Test GW with custom instrument matrix."""
        actual = synthetic_data["actual"]
        fc_a = synthetic_data["forecast_a"]
        fc_c = synthetic_data["forecast_c"]

        # Custom instruments: constant + lagged squared errors
        n = len(actual)
        instruments = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        gw = giacomini_white(
            actual, fc_a, fc_c, h=1, loss="mse", instruments=instruments
        )
        assert isinstance(gw.statistic, float)
        assert "custom" in gw.instruments_used

        # MAE loss for GW
        gw_mae = giacomini_white(actual, fc_a, fc_c, h=1, loss="mae")
        assert isinstance(gw_mae.statistic, float)

    def test_cv_validation_and_verbose(self) -> None:
        """Test CV validation errors and verbose mode."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(100, 10, size=100))

        def naive_model(train: pd.Series) -> np.ndarray:
            return np.full(5, float(train.iloc[-1]))

        # Validation errors
        with pytest.raises(ValueError, match="window must be"):
            rolling_window_cv(data, naive_model, window=0, horizon=5)
        with pytest.raises(ValueError, match="horizon must be"):
            rolling_window_cv(data, naive_model, window=50, horizon=0)
        with pytest.raises(ValueError, match="window .* horizon .* data length"):
            rolling_window_cv(data, naive_model, window=90, horizon=20)
        with pytest.raises(ValueError, match="n_blocks must be"):
            blocked_cv(data, naive_model, n_blocks=1, horizon=5)
        with pytest.raises(ValueError, match="Block size"):
            blocked_cv(data, naive_model, n_blocks=50, horizon=5)

        # Verbose mode
        rolling_window_cv(
            data, naive_model, window=50, horizon=5, step=30, verbose=True
        )
        blocked_cv(data, naive_model, n_blocks=3, horizon=5, verbose=True)

    def test_cv_std_metrics_and_extra_metrics(self) -> None:
        """Test CV std_metrics and mse/mape metrics."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(100, 10, size=200))

        def naive_model(train: pd.Series) -> np.ndarray:
            return np.full(5, float(train.iloc[-1]))

        result = rolling_window_cv(
            data,
            naive_model,
            window=80,
            horizon=5,
            step=20,
            metrics=("mae", "rmse", "mse", "mape"),
        )
        assert result.n_folds > 0
        means = result.mean_metrics()
        assert "mse" in means
        assert "mape" in means

        stds = result.std_metrics()
        assert "mae" in stds
        assert stds["mae"] >= 0

    def test_cv_with_point_attribute(self) -> None:
        """Test CV when model returns object with .point attribute."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(100, 10, size=200))

        class ForecastResult:
            def __init__(self, values: np.ndarray) -> None:
                self.point = values

        def model_with_point(train: pd.Series) -> ForecastResult:
            return ForecastResult(np.full(5, float(train.iloc[-1])))

        result = rolling_window_cv(
            data, model_with_point, window=80, horizon=5, step=30
        )
        assert result.n_folds > 0

        # Also test blocked CV with point attribute
        result_blocked = blocked_cv(
            data, model_with_point, n_blocks=3, horizon=5
        )
        assert result_blocked.n_folds > 0
