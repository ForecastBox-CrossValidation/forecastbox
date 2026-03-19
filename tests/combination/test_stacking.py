"""Tests for StackingCombiner."""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from forecastbox.core.forecast import Forecast


class TestStackingCombiner:
    """Tests for StackingCombiner (meta-learner combination)."""

    def test_stacking_ridge(self) -> None:
        """Ridge meta-learner produces valid combined forecast."""
        pytest.importorskip("sklearn")
        from forecastbox.combination.stacking import StackingCombiner

        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)
        fc_c = actual + rng.normal(0, 3.0, size=t)

        combiner = StackingCombiner(meta_learner="ridge")
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.is_fitted_
        assert combiner.weights_ is not None
        assert len(combiner.weights_) == 3
        assert pytest.approx(np.sum(combiner.weights_), abs=1e-6) == 1.0

        # Create Forecast objects and combine
        fc1 = Forecast(point=rng.normal(100, 5, size=4), model_name="A")
        fc2 = Forecast(point=rng.normal(100, 5, size=4), model_name="B")
        fc3 = Forecast(point=rng.normal(100, 5, size=4), model_name="C")
        result = combiner.combine([fc1, fc2, fc3])

        assert len(result.point) == 4
        assert result.model_name.startswith("Combined(Stacking")

    def test_stacking_rf(self) -> None:
        """Random Forest meta-learner produces valid combined forecast."""
        pytest.importorskip("sklearn")
        from forecastbox.combination.stacking import StackingCombiner

        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)

        combiner = StackingCombiner(meta_learner="rf", cv_folds=3)
        combiner.fit([fc_a, fc_b], actual)

        assert combiner.is_fitted_
        assert combiner.weights_ is not None
        assert len(combiner.weights_) == 2

        fc1 = Forecast(point=rng.normal(100, 5, size=4), model_name="A")
        fc2 = Forecast(point=rng.normal(100, 5, size=4), model_name="B")
        result = combiner.combine([fc1, fc2])

        assert len(result.point) == 4

    def test_sklearn_not_installed(self) -> None:
        """ImportError with useful message if sklearn is not installed."""
        with patch.dict(sys.modules, {"sklearn": None}):
            from forecastbox.combination.stacking import _check_sklearn

            with pytest.raises(ImportError, match="scikit-learn"):
                _check_sklearn()

    def test_cv_predictions(self) -> None:
        """Out-of-fold predictions are used when use_cv_predictions=True."""
        pytest.importorskip("sklearn")
        from forecastbox.combination.stacking import StackingCombiner

        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 2.0, size=t)

        # With CV
        combiner_cv = StackingCombiner(
            meta_learner="ridge", use_cv_predictions=True, cv_folds=5
        )
        combiner_cv.fit([fc_a, fc_b], actual)

        # Without CV
        combiner_no_cv = StackingCombiner(
            meta_learner="ridge", use_cv_predictions=False
        )
        combiner_no_cv.fit([fc_a, fc_b], actual)

        # Both should produce valid weights, but they may differ
        assert combiner_cv.weights_ is not None
        assert combiner_no_cv.weights_ is not None
        assert combiner_cv.is_fitted_
        assert combiner_no_cv.is_fitted_

    def test_feature_importances(self) -> None:
        """Feature importances available for models that support it."""
        pytest.importorskip("sklearn")
        from forecastbox.combination.stacking import StackingCombiner

        rng = np.random.default_rng(42)
        t = 100
        actual = rng.normal(100, 5, size=t)
        fc_a = actual + rng.normal(0, 1.0, size=t)
        fc_b = actual + rng.normal(0, 3.0, size=t)
        fc_c = actual + rng.normal(0, 5.0, size=t)

        combiner = StackingCombiner(meta_learner="rf", cv_folds=3)
        combiner.fit([fc_a, fc_b, fc_c], actual)

        assert combiner.feature_importances_ is not None
        assert len(combiner.feature_importances_) == 3
        assert pytest.approx(np.sum(combiner.feature_importances_), abs=1e-6) == 1.0

        # Model A (lowest noise) should have highest importance
        assert combiner.feature_importances_[0] > combiner.feature_importances_[2]
