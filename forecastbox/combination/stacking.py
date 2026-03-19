"""Stacking forecast combination with meta-learner."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from forecastbox.combination.base import BaseCombiner
from forecastbox.core.forecast import Forecast


def _check_sklearn() -> None:
    """Check if scikit-learn is installed."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        msg = (
            "StackingCombiner requires scikit-learn. "
            "Install it with: pip install scikit-learn"
        )
        raise ImportError(msg) from None


def _get_meta_learner(name: str) -> Any:
    """Get a scikit-learn estimator by name.

    Parameters
    ----------
    name : str
        Estimator name: 'ridge', 'lasso', 'rf', 'gbm'.

    Returns
    -------
    Any
        scikit-learn estimator instance.
    """
    _check_sklearn()

    if name == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)
    elif name == "lasso":
        from sklearn.linear_model import Lasso

        return Lasso(alpha=0.01, max_iter=10000)
    elif name == "rf":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif name == "gbm":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        msg = f"Unknown meta_learner: '{name}'. Use 'ridge', 'lasso', 'rf', or 'gbm'."
        raise ValueError(msg)


class StackingCombiner(BaseCombiner):
    """Forecast combination via stacking (meta-learner).

    Treats individual forecasts as features and trains a machine learning
    model to predict the realized value. Supports both linear (ridge, lasso)
    and non-linear (random forest, gradient boosting) meta-learners.

    The meta-learner is trained using out-of-fold predictions to avoid
    overfitting: training data is split into ``cv_folds`` time-ordered
    folds, and each fold's forecasts serve as features for the meta-learner.

    Parameters
    ----------
    meta_learner : str or sklearn estimator
        Meta-learner to use. String options: 'ridge', 'lasso', 'rf', 'gbm'.
        Alternatively, pass any scikit-learn estimator with fit/predict.
    cv_folds : int
        Number of cross-validation folds for out-of-fold predictions.
        Default is 5.
    use_cv_predictions : bool
        Whether to use out-of-fold predictions for training. Default True.

    Attributes
    ----------
    weights_ : NDArray[np.float64]
        Approximate model weights (from coefficients or feature importances).
    meta_model_ : Any
        Fitted scikit-learn estimator.
    feature_importances_ : NDArray[np.float64] | None
        Feature importances from the meta-learner (if available).

    Examples
    --------
    >>> combiner = StackingCombiner(meta_learner='ridge')
    >>> combiner.fit(forecasts_train, actual)
    >>> fc_combined = combiner.combine(forecasts)
    """

    def __init__(
        self,
        meta_learner: str | Any = "ridge",
        cv_folds: int = 5,
        use_cv_predictions: bool = True,
    ) -> None:
        super().__init__()
        self.meta_learner_param = meta_learner
        self.cv_folds = cv_folds
        self.use_cv_predictions = use_cv_predictions
        self.meta_model_: Any = None
        self.feature_importances_: NDArray[np.float64] | None = None

    def fit(
        self,
        forecasts_train: list[NDArray[np.float64]],
        actual: NDArray[np.float64],
    ) -> StackingCombiner:
        """Train the meta-learner on historical forecasts.

        Parameters
        ----------
        forecasts_train : list[NDArray[np.float64]]
            List of K arrays, each of shape (T,), with historical forecasts.
        actual : NDArray[np.float64]
            Array of shape (T,) with realized values.

        Returns
        -------
        StackingCombiner
            self, for method chaining.
        """
        _check_sklearn()

        actual = np.asarray(actual, dtype=np.float64)
        k = len(forecasts_train)
        t = len(actual)
        self.n_models_ = k

        # Build feature matrix: shape (T, K)
        x = np.column_stack(
            [np.asarray(fc, dtype=np.float64) for fc in forecasts_train]
        )

        # Get or create meta-learner
        if isinstance(self.meta_learner_param, str):
            self.meta_model_ = _get_meta_learner(self.meta_learner_param)
        else:
            from sklearn.base import clone

            self.meta_model_ = clone(self.meta_learner_param)

        if self.use_cv_predictions and t >= self.cv_folds * 2:
            # Out-of-fold predictions
            fold_size = t // self.cv_folds

            for fold_idx in range(self.cv_folds):
                val_start = fold_idx * fold_size
                val_end = (
                    (fold_idx + 1) * fold_size
                    if fold_idx < self.cv_folds - 1
                    else t
                )

                # Train on all data except this fold
                train_mask = np.ones(t, dtype=bool)
                train_mask[val_start:val_end] = False

                x_train = x[train_mask]
                y_train = actual[train_mask]

                if isinstance(self.meta_learner_param, str):
                    fold_model = _get_meta_learner(self.meta_learner_param)
                else:
                    from sklearn.base import clone

                    fold_model = clone(self.meta_learner_param)

                fold_model.fit(x_train, y_train)

            # Final fit on all data
            self.meta_model_.fit(x, actual)
        else:
            # Direct fit without CV
            self.meta_model_.fit(x, actual)

        # Extract weights / feature importances
        self._extract_weights()

        self.is_fitted_ = True
        return self

    def _extract_weights(self) -> None:
        """Extract approximate weights from the fitted meta-learner."""
        if hasattr(self.meta_model_, "coef_"):
            coef = np.asarray(self.meta_model_.coef_, dtype=np.float64).ravel()
            abs_coef = np.abs(coef)
            total = np.sum(abs_coef)
            if total > 0:
                self.weights_ = abs_coef / total
            else:
                self.weights_ = np.full(len(coef), 1.0 / len(coef))
            self.feature_importances_ = self.weights_.copy()

        elif hasattr(self.meta_model_, "feature_importances_"):
            fi = np.asarray(
                self.meta_model_.feature_importances_, dtype=np.float64
            )
            total = np.sum(fi)
            if total > 0:
                self.weights_ = fi / total
            else:
                self.weights_ = np.full(len(fi), 1.0 / len(fi))
            self.feature_importances_ = self.weights_.copy()

        else:
            # Fallback: equal weights
            self.weights_ = np.full(self.n_models_, 1.0 / self.n_models_)
            self.feature_importances_ = None

    def combine(self, forecasts: list[Forecast]) -> Forecast:
        """Combine forecasts using the trained meta-learner.

        Parameters
        ----------
        forecasts : list[Forecast]
            List of K Forecast objects to combine.

        Returns
        -------
        Forecast
            Combined forecast with point predictions from the meta-learner.
        """
        self._validate_forecasts(forecasts)

        if self.meta_model_ is None:
            msg = "StackingCombiner must be fitted before calling combine()."
            raise ValueError(msg)

        # Build feature matrix from forecasts: shape (H, K)
        x = np.column_stack([fc.point for fc in forecasts])

        # Predict using meta-learner
        combined_point = self.meta_model_.predict(x)

        # For intervals, use weighted combination with extracted weights
        lower_80 = None
        upper_80 = None
        lower_95 = None
        upper_95 = None

        if self.weights_ is not None:
            if all(fc.lower_80 is not None for fc in forecasts):
                lower_80_arr = np.array([fc.lower_80 for fc in forecasts])
                lower_80 = self.weights_ @ lower_80_arr
            if all(fc.upper_80 is not None for fc in forecasts):
                upper_80_arr = np.array([fc.upper_80 for fc in forecasts])
                upper_80 = self.weights_ @ upper_80_arr
            if all(fc.lower_95 is not None for fc in forecasts):
                lower_95_arr = np.array([fc.lower_95 for fc in forecasts])
                lower_95 = self.weights_ @ lower_95_arr
            if all(fc.upper_95 is not None for fc in forecasts):
                upper_95_arr = np.array([fc.upper_95 for fc in forecasts])
                upper_95 = self.weights_ @ upper_95_arr

        model_names = [fc.model_name for fc in forecasts]
        learner_name = (
            self.meta_learner_param
            if isinstance(self.meta_learner_param, str)
            else type(self.meta_learner_param).__name__
        )

        return Forecast(
            point=combined_point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            index=forecasts[0].index,
            model_name=f"Combined(Stacking-{learner_name})",
            metadata={
                "combiner": "StackingCombiner",
                "meta_learner": learner_name,
                "models": model_names,
            },
        )
