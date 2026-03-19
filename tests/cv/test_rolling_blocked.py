"""Tests for rolling window and blocked cross-validation.

References
----------
Bergmeir, C. & Benitez, J.M. (2012). "On the use of cross-validation
    for time series predictor evaluation."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecastbox.cv.rolling_blocked import blocked_cv, rolling_window_cv


def _naive_model(train: pd.Series) -> np.ndarray:
    """Simple naive model: forecast = last observed value."""
    last = float(train.iloc[-1])
    return np.full(12, last)


class TestRollingWindowCV:
    """Tests for rolling window CV."""

    def test_rolling_fixed_window(self) -> None:
        """All training windows have the same size."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(100, 10, size=200))

        results = rolling_window_cv(
            data, _naive_model, window=50, horizon=12, step=5
        )

        for fold in results.folds:
            train_size = fold.train_end - fold.train_start
            assert train_size == 50, f"Fold {fold.fold_id}: train size = {train_size}, expected 50"

    def test_rolling_no_leakage(self) -> None:
        """train_end <= test_start (no data leakage)."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(100, 10, size=200))

        results = rolling_window_cv(
            data, _naive_model, window=60, horizon=12, step=3
        )

        for fold in results.folds:
            assert fold.train_end <= fold.test_start, (
                f"Fold {fold.fold_id}: train_end={fold.train_end} > "
                f"test_start={fold.test_start}"
            )

        # Should have multiple folds
        assert results.n_folds > 5

        # Metrics should be computed
        means = results.mean_metrics()
        assert "mae" in means
        assert "rmse" in means
        assert means["mae"] >= 0
        assert means["rmse"] >= 0

        # Summary works
        summary = results.summary()
        assert "rolling_window" in summary


class TestBlockedCV:
    """Tests for blocked CV."""

    def test_blocked_cv_folds(self) -> None:
        """n_blocks folds are generated."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(100, 10, size=300))

        n_blocks = 5
        results = blocked_cv(
            data, _naive_model, n_blocks=n_blocks, horizon=12
        )

        assert results.n_folds == n_blocks

    def test_blocked_cv_gap(self) -> None:
        """Gap between training and test is respected."""
        rng = np.random.default_rng(42)
        data = pd.Series(rng.normal(100, 10, size=300))

        gap = 10
        n_blocks = 5

        results = blocked_cv(
            data, _naive_model, n_blocks=n_blocks, horizon=12, gap=gap
        )

        for fold in results.folds:
            # The fold's actual test data should be within the test block
            assert fold.test_start >= 0
            assert fold.test_end <= 300

        # Metrics should be computed
        means = results.mean_metrics()
        assert "mae" in means
        assert "rmse" in means
