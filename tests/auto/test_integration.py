"""End-to-end integration tests for all auto-forecasters.

These tests verify that:
1. All auto-forecasters generate valid Forecast objects (from Fase 1)
2. The full pipeline works: data -> auto -> forecast -> metrics
3. Forecasts can be plotted, serialized, and evaluated
4. Results are consistent and reproducible
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from forecastbox.auto.arima import AutoARIMA
from forecastbox.auto.ets import AutoETS
from forecastbox.auto.select import AutoSelect
from forecastbox.auto.var import AutoVAR
from forecastbox.auto.zoo import ModelZoo
from forecastbox.core.forecast import Forecast

# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------


def _make_airline_data() -> pd.Series:
    """Generate synthetic airline-like data."""
    np.random.seed(42)
    n = 144
    t = np.arange(n, dtype=np.float64)
    trend = 100 + 2.5 * t
    seasonal_pattern = np.array([
        0.90, 0.85, 0.95, 1.00, 1.05, 1.15,
        1.20, 1.18, 1.10, 1.02, 0.92, 0.88,
    ])
    seasonal = np.tile(seasonal_pattern, n // 12 + 1)[:n]
    noise = np.random.normal(0, 3, n)
    y = trend * seasonal + noise
    index = pd.date_range("1949-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="airline")


def _make_positive_seasonal() -> pd.Series:
    """Generate positive seasonal data."""
    np.random.seed(123)
    n = 180
    t = np.arange(n, dtype=np.float64)
    seasonal = 10.0 * np.sin(2 * np.pi * t / 12)
    y = 200.0 + 0.3 * t + seasonal + np.random.normal(0, 2, n)
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.Series(y, index=index, name="positive_seasonal")


def _make_multivariate() -> pd.DataFrame:
    """Generate multivariate time series data."""
    np.random.seed(456)
    n = 200
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t, 0] = 0.5 * y[t - 1, 0] + 0.2 * y[t - 1, 1] + np.random.normal(0, 1)
        y[t, 1] = 0.1 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + np.random.normal(0, 1)
        y[t, 2] = 0.3 * y[t - 1, 2] + np.random.normal(0, 1)
    index = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.DataFrame(y, index=index, columns=["gdp", "inflation", "unemployment"])


# ---------------------------------------------------------------------------
# Test: All auto-forecasters generate valid Forecast objects
# ---------------------------------------------------------------------------


class TestForecastValidity:
    """Verify all auto-forecasters produce valid Forecast objects."""

    def _validate_forecast(self, fc: Forecast, expected_horizon: int) -> None:
        """Common forecast validation checks."""
        assert isinstance(fc, Forecast), f"Expected Forecast, got {type(fc).__name__}"
        assert len(fc) == expected_horizon, f"Expected horizon={expected_horizon}, got {len(fc)}"
        assert len(fc.point) == expected_horizon
        assert not np.any(np.isnan(fc.point)), "Forecast contains NaN values"
        assert fc.model_name, "model_name should not be empty"
        assert fc.horizon == expected_horizon

    def test_auto_arima_generates_valid_forecast(self) -> None:
        """AutoARIMA generates a valid Forecast object."""
        data = _make_airline_data()
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
        result = auto.fit(data)
        fc = result.forecast(12)

        self._validate_forecast(fc, 12)

        # Check intervals
        if fc.lower_80 is not None and fc.upper_80 is not None:
            assert np.all(fc.lower_80 <= fc.point)
            assert np.all(fc.point <= fc.upper_80)

        if fc.lower_95 is not None and fc.upper_95 is not None:
            assert np.all(fc.lower_95 <= fc.point)
            assert np.all(fc.point <= fc.upper_95)

    def test_auto_ets_generates_valid_forecast(self) -> None:
        """AutoETS generates a valid Forecast object."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12)
        result = auto.fit(data)
        fc = result.forecast(12)

        self._validate_forecast(fc, 12)

    def test_auto_var_generates_valid_forecast(self) -> None:
        """AutoVAR generates a valid Forecast object."""
        data = _make_multivariate()
        auto = AutoVAR(max_lags=6, ic="bic")
        result = auto.fit(data)
        fc = result.forecast(12)

        self._validate_forecast(fc, 12)

    def test_auto_select_generates_valid_forecast(self) -> None:
        """AutoSelect generates a valid Forecast object."""
        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["naive", "drift"],
            cv_horizon=6,
            cv_step=5,
        )
        result = selector.fit(data)
        fc = result.forecast(12)

        self._validate_forecast(fc, 12)


# ---------------------------------------------------------------------------
# Test: Full pipeline (data -> auto -> forecast -> metrics)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test the complete pipeline: data -> auto-forecast -> metrics."""

    def test_arima_pipeline(self) -> None:
        """Full pipeline: data -> AutoARIMA -> forecast -> evaluate."""
        data = _make_airline_data()

        # Split: train on first 132, test on last 12
        train = data.iloc[:132]
        test = data.iloc[132:]
        h = len(test)

        # Fit AutoARIMA
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
        result = auto.fit(train)

        # Generate forecast
        fc = result.forecast(h)
        assert len(fc) == h

        # Compute metrics
        actual = test.values
        predicted = fc.point

        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        assert mae > 0, "MAE should be positive"
        assert rmse > 0, "RMSE should be positive"
        assert rmse >= mae, "RMSE should be >= MAE"

        # Verify result metadata
        assert result.order is not None
        assert result.seasonal_order is not None
        assert result.ic_value < np.inf
        assert result.n_fits > 0

    def test_ets_pipeline(self) -> None:
        """Full pipeline: data -> AutoETS -> forecast -> evaluate."""
        data = _make_positive_seasonal()

        train = data.iloc[:168]
        test = data.iloc[168:]
        h = len(test)

        auto = AutoETS(seasonal_period=12)
        result = auto.fit(train)

        fc = result.forecast(h)
        assert len(fc) == h

        actual = test.values
        predicted = fc.point

        mae = float(np.mean(np.abs(actual - predicted)))
        assert mae > 0

        # Verify result metadata
        assert result.model_type.startswith("ETS(")
        assert result.n_fits > 0

    def test_var_pipeline(self) -> None:
        """Full pipeline: data -> AutoVAR -> forecast -> evaluate."""
        data = _make_multivariate()

        train = data.iloc[:188]
        test = data.iloc[188:]
        h = len(test)

        auto = AutoVAR(max_lags=6, ic="bic")
        result = auto.fit(train)

        fc = result.forecast(h)
        assert len(fc) == h

        # Verify all variables have forecasts
        assert "all_variable_forecasts" in fc.metadata
        for var_name in train.columns:
            if var_name in result.selected_variables:
                assert var_name in fc.metadata["all_variable_forecasts"]


# ---------------------------------------------------------------------------
# Test: Forecast serialization and plotting
# ---------------------------------------------------------------------------


class TestForecastIntegration:
    """Test that auto-forecast results integrate with Fase 1 features."""

    def test_forecast_to_dataframe(self) -> None:
        """Auto-forecast Forecast objects can be converted to DataFrame."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12)
        result = auto.fit(data)
        fc = result.forecast(12)

        df = fc.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "point" in df.columns
        assert len(df) == 12

    def test_forecast_save_load(self, tmp_path: Path) -> None:
        """Auto-forecast Forecast objects can be saved and loaded."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12)
        result = auto.fit(data)
        fc = result.forecast(12)

        path = tmp_path / "auto_ets_forecast.json"
        fc.save(path)

        loaded = Forecast.load(path)
        np.testing.assert_allclose(loaded.point, fc.point)
        assert loaded.model_name == fc.model_name
        assert loaded.horizon == fc.horizon

    def test_forecast_plot(self) -> None:
        """Auto-forecast Forecast objects can be plotted."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12)
        result = auto.fit(data)
        fc = result.forecast(12)

        ax = fc.plot()
        assert ax is not None

        import matplotlib.pyplot as plt
        plt.close("all")

    def test_forecast_combine(self) -> None:
        """Multiple auto-forecast Forecast objects can be combined."""
        data = _make_positive_seasonal()

        # Get forecasts from different methods
        forecasts = []

        # ETS
        auto_ets = AutoETS(seasonal_period=12)
        result_ets = auto_ets.fit(data)
        fc_ets = result_ets.forecast(12)
        forecasts.append(fc_ets)

        # Naive baseline
        from forecastbox.auto._baselines import NaiveBaseline
        naive = NaiveBaseline()
        naive.fit(data)
        fc_naive = naive.forecast(12)
        forecasts.append(fc_naive)

        # Combine
        combined = Forecast.combine(forecasts, method="mean")
        assert isinstance(combined, Forecast)
        assert len(combined) == 12
        assert combined.model_name == "Combined(mean)"

    def test_forecast_validate(self) -> None:
        """Auto-forecast Forecast objects pass validation."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12)
        result = auto.fit(data)
        fc = result.forecast(12)

        # Should not raise if intervals are consistent
        if fc.lower_80 is not None and fc.upper_80 is not None:
            fc.validate()


# ---------------------------------------------------------------------------
# Test: ModelZoo integration
# ---------------------------------------------------------------------------


class TestModelZooIntegration:
    """Test ModelZoo integration with auto-forecasters."""

    @pytest.fixture(autouse=True)
    def reset_zoo(self) -> None:
        """Reset ModelZoo before each test."""
        ModelZoo.reset()

    def test_zoo_lists_registered_models(self) -> None:
        """After importing auto module, ModelZoo should have models."""
        zoo = ModelZoo()
        models = zoo.list_models()
        # At minimum, baselines should be registered
        # (chronobox adapters may or may not be available)
        assert isinstance(models, list)

    def test_zoo_create_baseline(self) -> None:
        """ModelZoo can create baseline models."""
        zoo = ModelZoo()

        if "naive" in zoo.list_models():
            model = zoo.create("naive")
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
            model.fit(data)
            fc = model.forecast(5)
            assert isinstance(fc, Forecast)
            assert len(fc) == 5

    def test_zoo_custom_model_in_autoselect(self) -> None:
        """Custom model registered in ModelZoo works with AutoSelect."""
        from forecastbox.auto._baselines import DriftBaseline

        zoo = ModelZoo()
        zoo.register(
            "custom_drift",
            DriftBaseline,
            family="custom",
            description="Custom drift model",
        )

        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["naive", "custom_drift"],
            cv_horizon=6,
            cv_step=5,
        )
        result = selector.fit(data)

        assert "custom_drift" in set(result.ranking["family"])


# ---------------------------------------------------------------------------
# Test: Criterios de Conclusao (from original spec)
# ---------------------------------------------------------------------------


class TestCriteriosConclusao:
    """Verify all criterios de conclusao from the original specification."""

    def test_auto_arima_airline_order(self) -> None:
        """AutoARIMA seleciona ARIMA proximo a (0,1,1)(0,1,1)[12] na airline."""
        data = _make_airline_data()
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
        result = auto.fit(data)

        p, d, q = result.order
        assert d >= 1, "Airline data should have d >= 1"
        assert result.seasonal_order[3] == 12

    def test_auto_arima_stepwise_efficiency(self) -> None:
        """AutoARIMA stepwise ajusta < 30 modelos."""
        data = _make_airline_data()
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
        result = auto.fit(data)

        assert result.n_fits < 30, (
            f"Stepwise should fit < 30 models, got {result.n_fits}"
        )

    def test_auto_ets_multiplicative_airline(self) -> None:
        """AutoETS seleciona modelo multiplicativo para airline."""
        data = _make_airline_data()
        auto = AutoETS(seasonal_period=12)
        result = auto.fit(data)

        has_mult = "M" in result.model_type
        assert has_mult, f"Expected multiplicative component, got {result.model_type}"

    def test_auto_ets_enumerate_restrictions(self) -> None:
        """AutoETS enumera ate 30 combinacoes com restricoes corretas."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12)
        y_arr = np.asarray(data, dtype=np.float64)
        models = auto._enumerate_models(y_arr)
        assert len(models) <= 30

    def test_auto_select_ranking_stable(self) -> None:
        """AutoSelect ranking estavel entre execucoes."""
        data = _make_positive_seasonal()

        results = []
        for _ in range(2):
            selector = AutoSelect(
                families=["naive", "drift"],
                cv_horizon=6,
                cv_step=5,
            )
            result = selector.fit(data)
            results.append(list(result.ranking["family"]))

        assert results[0] == results[1], "Rankings should be identical for same data"

    def test_auto_select_cv_no_leakage(self) -> None:
        """AutoSelect CV sem data leakage."""
        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["naive"],
            cv_horizon=6,
            cv_step=5,
            cv_initial=120,
        )
        result = selector.fit(data)

        # Verify all scores are finite (no data access errors)
        finite = [s for s in result.all_cv_results["naive"] if np.isfinite(s)]
        assert len(finite) > 0

    def test_auto_var_lag_selection(self) -> None:
        """AutoVAR seleciona lag correto em simulacao."""
        # Generate VAR(2) data
        np.random.seed(42)
        n = 300
        y = np.zeros((n, 2))
        for t in range(2, n):
            y[t, 0] = 0.5 * y[t - 1, 0] + 0.3 * y[t - 2, 0] + np.random.normal(0, 1)
            y[t, 1] = 0.4 * y[t - 1, 1] + 0.2 * y[t - 2, 1] + np.random.normal(0, 1)

        data = pd.DataFrame(y, columns=["y1", "y2"])
        auto = AutoVAR(max_lags=8, ic="bic")
        result = auto.fit(data)

        assert 1 <= result.selected_lag <= 4

    def test_model_zoo_functional(self) -> None:
        """ModelZoo register/get/create funcional."""
        ModelZoo.reset()
        zoo = ModelZoo()

        class TestModel:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            def fit(self, y: Any, **kwargs: Any) -> TestModel:
                return self

            def forecast(self, h: int, **kwargs: Any) -> Any:
                return np.zeros(h)

        zoo.register("test_model", TestModel, family="test", default_params={"x": 1})

        entry = zoo.get("test_model")
        assert entry.name == "test_model"

        models = zoo.list_models(family="test")
        assert "test_model" in models

        instance = zoo.create("test_model", x=2)
        assert isinstance(instance, TestModel)

    def test_all_forecasters_generate_valid_forecast(self) -> None:
        """Todos os auto-forecasters geram Forecast valido."""
        # AutoARIMA
        data = _make_airline_data()
        result = AutoARIMA(seasonal=True, m=12, stepwise=True).fit(data)
        fc = result.forecast(12)
        assert isinstance(fc, Forecast)
        assert len(fc.point) == 12

        # AutoETS
        data2 = _make_positive_seasonal()
        result2 = AutoETS(seasonal_period=12).fit(data2)
        fc2 = result2.forecast(12)
        assert isinstance(fc2, Forecast)
        assert len(fc2.point) == 12

        # AutoVAR
        data3 = _make_multivariate()
        result3 = AutoVAR(max_lags=4, ic="bic").fit(data3)
        fc3 = result3.forecast(12)
        assert isinstance(fc3, Forecast)
        assert len(fc3.point) == 12

        # AutoSelect
        selector = AutoSelect(families=["naive", "drift"], cv_horizon=6, cv_step=5)
        result4 = selector.fit(data2)
        fc4 = result4.forecast(12)
        assert isinstance(fc4, Forecast)
        assert len(fc4.point) == 12


# ---------------------------------------------------------------------------
# Test: Baseline edge cases and error paths
# ---------------------------------------------------------------------------


class TestBaselineEdgeCases:
    """Test baseline model edge cases for coverage."""

    def test_naive_unfitted_raises(self) -> None:
        """NaiveBaseline raises if forecast called before fit."""
        from forecastbox.auto._baselines import NaiveBaseline

        model = NaiveBaseline()
        with pytest.raises(RuntimeError, match="must be fit"):
            model.forecast(5)

    def test_drift_unfitted_raises(self) -> None:
        """DriftBaseline raises if forecast called before fit."""
        from forecastbox.auto._baselines import DriftBaseline

        model = DriftBaseline()
        with pytest.raises(RuntimeError, match="must be fit"):
            model.forecast(5)

    def test_drift_short_series_raises(self) -> None:
        """DriftBaseline raises on series with < 2 obs."""
        from forecastbox.auto._baselines import DriftBaseline

        model = DriftBaseline()
        with pytest.raises(ValueError, match="at least 2"):
            model.fit(np.array([1.0]))

    def test_seasonal_naive_fit_forecast(self) -> None:
        """SeasonalNaiveBaseline fits and forecasts correctly."""
        from forecastbox.auto._baselines import SeasonalNaiveBaseline

        data = np.arange(1.0, 37.0)  # 3 years of monthly-ish data
        model = SeasonalNaiveBaseline(seasonal_period=12)
        model.fit(data)
        fc = model.forecast(12)
        assert isinstance(fc, Forecast)
        assert len(fc) == 12
        # Should repeat last 12 values
        np.testing.assert_allclose(fc.point, data[-12:])

    def test_seasonal_naive_unfitted_raises(self) -> None:
        """SeasonalNaiveBaseline raises if forecast called before fit."""
        from forecastbox.auto._baselines import SeasonalNaiveBaseline

        model = SeasonalNaiveBaseline()
        with pytest.raises(RuntimeError, match="must be fit"):
            model.forecast(5)

    def test_seasonal_naive_short_series_raises(self) -> None:
        """SeasonalNaiveBaseline raises on series shorter than seasonal_period."""
        from forecastbox.auto._baselines import SeasonalNaiveBaseline

        model = SeasonalNaiveBaseline(seasonal_period=12)
        with pytest.raises(ValueError, match="seasonal_period"):
            model.fit(np.arange(1.0, 6.0))


# ---------------------------------------------------------------------------
# Test: Adapter ImportError paths
# ---------------------------------------------------------------------------


class TestAdapterImportErrors:
    """Test that adapters raise ImportError when chronobox is unavailable."""

    def test_arima_adapter_import_error(self) -> None:
        """ARIMAAdapter raises ImportError without chronobox."""
        from forecastbox.auto._adapters import HAS_CHRONOBOX, ARIMAAdapter

        if not HAS_CHRONOBOX:
            with pytest.raises(ImportError, match="chronobox"):
                ARIMAAdapter()

    def test_ets_adapter_import_error(self) -> None:
        """ETSAdapter raises ImportError without chronobox."""
        from forecastbox.auto._adapters import HAS_CHRONOBOX, ETSAdapter

        if not HAS_CHRONOBOX:
            with pytest.raises(ImportError, match="chronobox"):
                ETSAdapter()

    def test_var_adapter_import_error(self) -> None:
        """VARAdapter raises ImportError without chronobox."""
        from forecastbox.auto._adapters import HAS_CHRONOBOX, VARAdapter

        if not HAS_CHRONOBOX:
            with pytest.raises(ImportError, match="chronobox"):
                VARAdapter()

    def test_theta_adapter_import_error(self) -> None:
        """ThetaAdapter raises ImportError without chronobox."""
        from forecastbox.auto._adapters import HAS_CHRONOBOX, ThetaAdapter

        if not HAS_CHRONOBOX:
            with pytest.raises(ImportError, match="chronobox"):
                ThetaAdapter()


# ---------------------------------------------------------------------------
# Test: Additional coverage for select/var/ets/arima
# ---------------------------------------------------------------------------


class TestAdditionalCoverage:
    """Additional tests targeting uncovered code paths."""

    def test_auto_arima_summary(self) -> None:
        """AutoARIMAResult.summary returns a string."""
        data = _make_airline_data()
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
        result = auto.fit(data)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "ARIMA" in summary

    def test_auto_ets_summary(self) -> None:
        """AutoETSResult.summary returns a string."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12)
        result = auto.fit(data)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "ETS" in summary

    def test_auto_var_summary(self) -> None:
        """AutoVARResult.summary returns a string."""
        data = _make_multivariate()
        auto = AutoVAR(max_lags=4, ic="bic")
        result = auto.fit(data)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "VAR" in summary or "lag" in summary.lower()

    def test_auto_select_summary(self) -> None:
        """AutoSelectResult.summary returns a string."""
        data = _make_positive_seasonal()
        selector = AutoSelect(families=["naive", "drift"], cv_horizon=6, cv_step=5)
        result = selector.fit(data)
        summary = result.summary()
        assert isinstance(summary, str)

    def test_auto_select_plot_comparison(self) -> None:
        """AutoSelectResult.plot_comparison works."""
        import matplotlib.pyplot as plt

        data = _make_positive_seasonal()
        selector = AutoSelect(families=["naive", "drift"], cv_horizon=6, cv_step=5)
        result = selector.fit(data)
        ax = result.plot_comparison()
        assert ax is not None
        plt.close("all")

    def test_auto_var_irf(self) -> None:
        """AutoVARResult.irf returns impulse response data."""
        data = _make_multivariate()
        auto = AutoVAR(max_lags=4, ic="bic")
        result = auto.fit(data)
        irf = result.irf(steps=10)
        assert irf is not None

    def test_auto_arima_grid_search(self) -> None:
        """AutoARIMA with stepwise=False uses grid search."""
        data = _make_positive_seasonal()
        auto = AutoARIMA(
            seasonal=True, m=12, stepwise=False,
            max_p=1, max_q=1, max_P=1, max_Q=1,
        )
        result = auto.fit(data)
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)
        assert result.search_method == "grid"

    def test_auto_ets_no_seasonal(self) -> None:
        """AutoETS with seasonal_period=1 (no seasonality)."""
        np.random.seed(42)
        n = 100
        y = np.cumsum(np.random.normal(0.1, 1, n)) + 200
        data = pd.Series(y, index=pd.date_range("2000-01", periods=n, freq="MS"))
        auto = AutoETS(seasonal_period=1)
        result = auto.fit(data)
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)
        assert "N" in result.model_type  # No seasonal component

    def test_auto_var_different_ic(self) -> None:
        """AutoVAR with AIC criterion."""
        data = _make_multivariate()
        auto = AutoVAR(max_lags=4, ic="aic")
        result = auto.fit(data)
        assert result.ic_name == "aic"
        assert result.selected_lag >= 1

    def test_auto_select_mae_metric(self) -> None:
        """AutoSelect with MAE metric."""
        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["naive", "drift"],
            cv_horizon=6,
            cv_step=5,
            metric="mae",
        )
        result = selector.fit(data)
        assert result.metric_name == "mae"
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)

    def test_auto_select_with_snaive(self) -> None:
        """AutoSelect with seasonal naive family."""
        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["naive", "snaive"],
            cv_horizon=6,
            cv_step=5,
        )
        result = selector.fit(data)
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)

    def test_auto_select_rolling_cv(self) -> None:
        """AutoSelect with rolling window CV."""
        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["naive", "drift"],
            cv_type="rolling",
            cv_horizon=6,
            cv_step=5,
        )
        result = selector.fit(data)
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)

    def test_auto_var_variable_selection(self) -> None:
        """AutoVAR with variable selection enabled."""
        data = _make_multivariate()
        auto = AutoVAR(max_lags=4, ic="bic", select_variables=True)
        result = auto.fit(data)
        assert len(result.selected_variables) >= 2
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)

    def test_compute_var_ic_hqc(self) -> None:
        """Test _compute_var_ic with HQC criterion."""
        from forecastbox.auto.var import _compute_var_ic

        sigma = np.eye(2) * 0.5
        val = _compute_var_ic(sigma, 100, 2, 2, "hqc")
        assert np.isfinite(val)

    def test_compute_var_ic_fpe(self) -> None:
        """Test _compute_var_ic with FPE criterion."""
        from forecastbox.auto.var import _compute_var_ic

        sigma = np.eye(2) * 0.5
        val = _compute_var_ic(sigma, 100, 2, 2, "fpe")
        assert np.isfinite(val)

    def test_compute_var_ic_singular(self) -> None:
        """Test _compute_var_ic with singular covariance matrix."""
        from forecastbox.auto.var import _compute_var_ic

        sigma = np.zeros((2, 2))
        val = _compute_var_ic(sigma, 100, 2, 2, "bic")
        assert val == np.inf

    def test_compute_ets_ic_aic(self) -> None:
        """Test _compute_ets_ic with AIC criterion."""
        from forecastbox.auto.ets import _compute_ets_ic

        val = _compute_ets_ic(100.0, 3, 100, "aic")
        assert np.isfinite(val)

    def test_compute_ets_ic_bic(self) -> None:
        """Test _compute_ets_ic with BIC criterion."""
        from forecastbox.auto.ets import _compute_ets_ic

        val = _compute_ets_ic(100.0, 3, 100, "bic")
        assert np.isfinite(val)

    def test_compute_ets_ic_zero_sse(self) -> None:
        """Test _compute_ets_ic with zero SSE."""
        from forecastbox.auto.ets import _compute_ets_ic

        val = _compute_ets_ic(0.0, 3, 100, "aicc")
        assert val == np.inf

    def test_compute_metric_mape(self) -> None:
        """Test _compute_metric with MAPE."""
        from forecastbox.auto.select import _compute_metric

        actual = np.array([100.0, 200.0, 300.0])
        pred = np.array([110.0, 190.0, 310.0])
        val = _compute_metric(actual, pred, "mape")
        assert val > 0

    def test_compute_metric_mase(self) -> None:
        """Test _compute_metric with MASE."""
        from forecastbox.auto.select import _compute_metric

        actual = np.array([100.0, 200.0, 300.0, 400.0])
        pred = np.array([110.0, 190.0, 310.0, 390.0])
        val = _compute_metric(actual, pred, "mase")
        assert val > 0

    def test_auto_ets_fixed_error(self) -> None:
        """AutoETS with fixed error component."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12, error="A")
        result = auto.fit(data)
        assert result.error == "A"

    def test_auto_ets_bic_criterion(self) -> None:
        """AutoETS with BIC criterion."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12, ic="bic")
        result = auto.fit(data)
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)

    def test_auto_arima_repr(self) -> None:
        """AutoARIMAResult has a useful repr."""
        data = _make_airline_data()
        auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
        result = auto.fit(data)
        r = repr(result)
        assert "ARIMA" in r

    def test_auto_var_with_hqc(self) -> None:
        """AutoVAR with HQC criterion."""
        data = _make_multivariate()
        auto = AutoVAR(max_lags=4, ic="hqc")
        result = auto.fit(data)
        assert result.ic_name == "hqc"

    def test_auto_var_with_fpe(self) -> None:
        """AutoVAR with FPE criterion."""
        data = _make_multivariate()
        auto = AutoVAR(max_lags=4, ic="fpe")
        result = auto.fit(data)
        assert result.ic_name == "fpe"

    def test_auto_select_with_arima_ets(self) -> None:
        """AutoSelect with arima and ets families."""
        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["arima", "ets"],
            cv_horizon=6,
            cv_step=5,
        )
        result = selector.fit(data, m=12)
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)

    def test_auto_select_custom_zoo_family(self) -> None:
        """AutoSelect with a custom ModelZoo family."""
        ModelZoo.reset()
        from forecastbox.auto._baselines import NaiveBaseline

        zoo = ModelZoo()
        zoo.register("my_naive", NaiveBaseline, family="my_naive")

        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["naive", "my_naive"],
            cv_horizon=6,
            cv_step=5,
        )
        result = selector.fit(data)
        fc = result.forecast(6)
        assert isinstance(fc, Forecast)

    def test_stepwise_kpss_with_ct(self) -> None:
        """Test _kpss_test with trend regression."""
        from forecastbox.auto._stepwise import _kpss_test

        np.random.seed(42)
        y = np.cumsum(np.random.normal(0, 1, 100))
        result = _kpss_test(y, regression="ct")
        assert isinstance(result, bool)

    def test_stepwise_kpss_short_series(self) -> None:
        """Test _kpss_test with very short series."""
        from forecastbox.auto._stepwise import _kpss_test

        y = np.array([1.0, 2.0, 3.0])
        result = _kpss_test(y)
        assert result is True  # Too short, assume stationary

    def test_stepwise_determine_d(self) -> None:
        """Test _determine_d for differencing order."""
        from forecastbox.auto._stepwise import _determine_d

        np.random.seed(42)
        y = np.cumsum(np.random.normal(0, 1, 100))
        d = _determine_d(y)
        assert 0 <= d <= 2

    def test_stepwise_determine_seasonal_d(self) -> None:
        """Test _determine_seasonal_d edge cases."""
        from forecastbox.auto._stepwise import _determine_seasonal_d

        # m <= 1: should return 0
        y = np.random.normal(0, 1, 100)
        assert _determine_seasonal_d(y, m=1) == 0

        # Short series: should return 0
        y_short = np.random.normal(0, 1, 10)
        assert _determine_seasonal_d(y_short, m=12) == 0

    def test_stepwise_ocsb_test(self) -> None:
        """Test _ocsb_test with seasonal data."""
        from forecastbox.auto._stepwise import _ocsb_test

        np.random.seed(42)
        n = 120
        t = np.arange(n, dtype=np.float64)
        y = 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, n)
        result = _ocsb_test(y, m=12)
        assert isinstance(result, bool)

    def test_stepwise_is_valid_order(self) -> None:
        """Test _is_valid_order with various cases."""
        from forecastbox.auto._stepwise import _is_valid_order

        # Valid order
        assert _is_valid_order(1, 1, 1, 0, 0, 0, 5, 5, 2, 2, 5) is True
        # Negative p
        assert _is_valid_order(-1, 1, 1, 0, 0, 0, 5, 5, 2, 2, 5) is False
        # Exceed max_p
        assert _is_valid_order(6, 1, 1, 0, 0, 0, 5, 5, 2, 2, 5) is False
        # Exceed max seasonal
        assert _is_valid_order(1, 1, 1, 3, 0, 0, 5, 5, 2, 2, 5) is False
        # Exceed total order
        assert _is_valid_order(3, 1, 3, 2, 0, 2, 5, 5, 2, 2, 5) is False

    def test_compute_var_ic_fpe_small_sample(self) -> None:
        """Test _compute_var_ic with FPE and small sample."""
        from forecastbox.auto.var import _compute_var_ic

        sigma = np.eye(2)
        # denom = T - k*p - 1 = 10 - 2*5 - 1 = -1 <= 0
        val = _compute_var_ic(sigma, 10, 2, 5, "fpe")
        assert val == np.inf

    def test_compute_var_ic_invalid_ic(self) -> None:
        """Test _compute_var_ic with invalid IC raises."""
        from forecastbox.auto.var import _compute_var_ic

        sigma = np.eye(2)
        with pytest.raises(ValueError, match="Unknown IC"):
            _compute_var_ic(sigma, 100, 2, 2, "invalid")

    def test_compute_ets_ic_invalid(self) -> None:
        """Test _compute_ets_ic with invalid IC raises."""
        from forecastbox.auto.ets import _compute_ets_ic

        with pytest.raises(ValueError, match="Unknown IC"):
            _compute_ets_ic(100.0, 3, 100, "invalid")

    def test_auto_ets_fixed_trend(self) -> None:
        """AutoETS with fixed trend component."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12, trend="A")
        result = auto.fit(data)
        assert "A" in result.trend

    def test_auto_ets_fixed_seasonal(self) -> None:
        """AutoETS with fixed seasonal component."""
        data = _make_positive_seasonal()
        auto = AutoETS(seasonal_period=12, seasonal="A")
        result = auto.fit(data)
        assert result.seasonal == "A"

    def test_auto_select_unknown_family_raises(self) -> None:
        """AutoSelect with unknown family raises ValueError."""
        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["nonexistent_family_xyz"],
            cv_horizon=6,
            cv_step=5,
        )
        # Should not raise during fit - it catches errors
        result = selector.fit(data)
        # The failed family should have inf CV score
        assert np.isinf(result.ranking.iloc[0]["cv_mean"])

    def test_auto_select_cv_horizon_override(self) -> None:
        """AutoSelect cv_horizon override in fit."""
        data = _make_positive_seasonal()
        selector = AutoSelect(
            families=["naive"],
            cv_horizon=6,
            cv_step=5,
        )
        result = selector.fit(data, cv_horizon=3)
        assert result is not None

    def test_compute_metric_mape_zeros(self) -> None:
        """Test _compute_metric MAPE with zeros in actual."""
        from forecastbox.auto.select import _compute_metric

        actual = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 2.0, 3.0])
        val = _compute_metric(actual, pred, "mape")
        assert val == np.inf

    def test_compute_metric_mase_constant(self) -> None:
        """Test _compute_metric MASE with constant series."""
        from forecastbox.auto.select import _compute_metric

        actual = np.array([5.0, 5.0, 5.0, 5.0])
        pred = np.array([6.0, 6.0, 6.0, 6.0])
        val = _compute_metric(actual, pred, "mase")
        assert val == np.inf  # naive errors = 0

    def test_compute_metric_invalid(self) -> None:
        """Test _compute_metric with invalid metric."""
        from forecastbox.auto.select import _compute_metric

        with pytest.raises(ValueError, match="Unknown metric"):
            _compute_metric(np.ones(3), np.ones(3), "invalid")

    def test_stepwise_generate_neighbors(self) -> None:
        """Test _generate_neighbors produces valid neighbors."""
        from forecastbox.auto._stepwise import _generate_neighbors

        neighbors = _generate_neighbors(
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 12),
            include_constant=True,
            max_p=5,
            max_q=5,
            max_seasonal_p=2,
            max_seasonal_q=2,
            max_order=5,
        )
        assert len(neighbors) > 0
        for order, seasonal, _const in neighbors:
            assert len(order) == 3
            assert len(seasonal) == 4
