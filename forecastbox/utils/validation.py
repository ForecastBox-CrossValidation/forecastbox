"""Validation utilities for forecastbox."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def check_array(x: object, name: str, min_length: int = 1) -> NDArray[np.float64]:
    """Validate and convert to numpy array.

    Parameters
    ----------
    x : array-like
        Input to validate.
    name : str
        Name for error messages.
    min_length : int
        Minimum required length.

    Returns
    -------
    NDArray[np.float64]
        Validated numpy array.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        msg = f"{name} must be array-like, got scalar"
        raise ValueError(msg)
    if len(arr) < min_length:
        msg = f"{name} must have at least {min_length} elements, got {len(arr)}"
        raise ValueError(msg)
    if np.any(np.isnan(arr)):
        msg = f"{name} contains NaN values"
        raise ValueError(msg)
    return arr


def check_same_length(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    name_a: str,
    name_b: str,
) -> None:
    """Validate that two arrays have the same length."""
    if len(a) != len(b):
        msg = f"{name_a} and {name_b} must have same length, got {len(a)} and {len(b)}"
        raise ValueError(msg)


def check_positive(x: float, name: str) -> None:
    """Validate that a value is positive."""
    if x <= 0:
        msg = f"{name} must be positive, got {x}"
        raise ValueError(msg)


def check_probability(p: float, name: str) -> None:
    """Validate that a value is a valid probability (0 < p < 1)."""
    if not 0 < p < 1:
        msg = f"{name} must be between 0 and 1 (exclusive), got {p}"
        raise ValueError(msg)


def check_forecast(fc: object) -> None:
    """Validate a Forecast object for consistency.

    Parameters
    ----------
    fc : Forecast
        Forecast to validate.
    """
    from forecastbox.core.forecast import Forecast

    if not isinstance(fc, Forecast):
        msg = f"Expected Forecast, got {type(fc).__name__}"
        raise TypeError(msg)

    # Check intervals consistency: lower < point < upper
    if fc.lower_80 is not None and fc.upper_80 is not None:
        if not np.all(fc.lower_80 <= fc.point):
            msg = "lower_80 must be <= point for all horizons"
            raise ValueError(msg)
        if not np.all(fc.point <= fc.upper_80):
            msg = "point must be <= upper_80 for all horizons"
            raise ValueError(msg)

    if fc.lower_95 is not None and fc.upper_95 is not None:
        if not np.all(fc.lower_95 <= fc.point):
            msg = "lower_95 must be <= point for all horizons"
            raise ValueError(msg)
        if not np.all(fc.point <= fc.upper_95):
            msg = "point must be <= upper_95 for all horizons"
            raise ValueError(msg)

    # Check 95 wider than 80
    if (
        fc.lower_80 is not None
        and fc.lower_95 is not None
        and fc.upper_80 is not None
        and fc.upper_95 is not None
    ):
        if not np.all(fc.lower_95 <= fc.lower_80):
            msg = "lower_95 must be <= lower_80 for all horizons"
            raise ValueError(msg)
        if not np.all(fc.upper_80 <= fc.upper_95):
            msg = "upper_80 must be <= upper_95 for all horizons"
            raise ValueError(msg)
