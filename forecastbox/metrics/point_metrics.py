"""Point forecast evaluation metrics.

References
----------
Hyndman, R.J. & Koehler, A.B. (2006). "Another look at measures of forecast accuracy."
International Journal of Forecasting, 22(4), 679-688.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from forecastbox.utils.validation import check_array, check_same_length


def _validate_inputs(
    actual: object, predicted: object
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and convert inputs to numpy arrays.

    Parameters
    ----------
    actual : array-like
        Actual values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    tuple[NDArray, NDArray]
        Validated arrays.
    """
    a = check_array(actual, "actual")
    p = check_array(predicted, "predicted")
    check_same_length(a, p, "actual", "predicted")
    return a, p


def mae(actual: object, predicted: object) -> float:
    """Mean Absolute Error.

    MAE = (1/n) * sum(|y_t - f_t|)

    Parameters
    ----------
    actual : array-like
        Actual values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        MAE value.
    """
    a, p = _validate_inputs(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: object, predicted: object) -> float:
    """Root Mean Squared Error.

    RMSE = sqrt((1/n) * sum((y_t - f_t)^2))

    Parameters
    ----------
    actual : array-like
        Actual values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    a, p = _validate_inputs(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: object, predicted: object) -> float:
    """Mean Absolute Percentage Error.

    MAPE = (100/n) * sum(|y_t - f_t| / |y_t|)

    Parameters
    ----------
    actual : array-like
        Actual values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE value in percentage. Returns np.inf if any actual value is zero.
    """
    a, p = _validate_inputs(actual, predicted)

    if np.any(a == 0):
        warnings.warn(
            "MAPE is undefined when actual values contain zeros. Returning np.inf.",
            stacklevel=2,
        )
        return float(np.inf)

    return float(100.0 * np.mean(np.abs((a - p) / a)))


def mase(
    actual: object,
    predicted: object,
    training_series: object,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    MASE = MAE / MAE_naive
    where MAE_naive is the in-sample MAE of the naive (random walk) forecast.

    Parameters
    ----------
    actual : array-like
        Actual values (test set).
    predicted : array-like
        Predicted values.
    training_series : array-like
        In-sample (training) data for computing naive forecast MAE.

    Returns
    -------
    float
        MASE value. < 1 means better than naive forecast.
    """
    a, p = _validate_inputs(actual, predicted)
    train = check_array(training_series, "training_series", min_length=2)

    # Naive forecast MAE (random walk): |y_t - y_{t-1}| for t=2..T
    naive_errors = np.abs(np.diff(train))
    naive_mae = np.mean(naive_errors)

    if naive_mae == 0:
        warnings.warn(
            "Naive MAE is zero (constant training series). Returning np.inf.",
            stacklevel=2,
        )
        return float(np.inf)

    forecast_mae = np.mean(np.abs(a - p))
    return float(forecast_mae / naive_mae)


def me(actual: object, predicted: object) -> float:
    """Mean Error (bias).

    ME = (1/n) * sum(y_t - f_t)

    Parameters
    ----------
    actual : array-like
        Actual values.
    predicted : array-like
        Predicted values.

    Returns
    -------
    float
        ME value. Positive means under-prediction, negative means over-prediction.
    """
    a, p = _validate_inputs(actual, predicted)
    return float(np.mean(a - p))
