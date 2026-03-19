"""Forecast horizon abstractions."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


class ForecastHorizon:
    """Abstraction for forecast horizon h=1..H.

    Parameters
    ----------
    h : int
        Number of steps ahead.
    freq : str or None
        Pandas frequency string (e.g., 'MS', 'QS', 'YS').
    origin : str or pd.Timestamp or None
        Origin date for generating temporal index.
    """

    def __init__(
        self,
        h: int,
        freq: str | None = None,
        origin: str | pd.Timestamp | None = None,
    ) -> None:
        if h < 1:
            msg = f"Horizon must be >= 1, got {h}"
            raise ValueError(msg)
        self.h = h
        self.freq = freq
        self.origin = pd.Timestamp(origin) if origin is not None else None

    def to_index(self) -> pd.DatetimeIndex:
        """Generate temporal index for the forecast horizon.

        Returns
        -------
        pd.DatetimeIndex
            Index with h dates starting after origin.
        """
        if self.origin is None or self.freq is None:
            msg = "Cannot generate index without origin and freq"
            raise ValueError(msg)
        # Generate h periods starting after origin
        start = self.origin + pd.tseries.frequencies.to_offset(self.freq)
        return pd.date_range(start=start, periods=self.h, freq=self.freq)

    def __len__(self) -> int:
        return self.h

    def __iter__(self) -> Iterator[int]:
        return iter(range(1, self.h + 1))

    def __repr__(self) -> str:
        freq_str = f", freq='{self.freq}'" if self.freq else ""
        origin_str = f", origin='{self.origin}'" if self.origin else ""
        return f"ForecastHorizon(h={self.h}{freq_str}{origin_str})"


class MultiHorizon:
    """Non-contiguous forecast horizons (e.g., h={1, 3, 6, 12}).

    Parameters
    ----------
    horizons : list[int]
        List of specific horizon steps.
    """

    def __init__(self, horizons: list[int]) -> None:
        if not horizons:
            msg = "horizons must be non-empty"
            raise ValueError(msg)
        self.horizons = sorted(horizons)

    def __len__(self) -> int:
        return len(self.horizons)

    def __iter__(self) -> Iterator[int]:
        return iter(self.horizons)

    def __contains__(self, h: int) -> bool:
        return h in self.horizons

    def __repr__(self) -> str:
        return f"MultiHorizon({self.horizons})"


def h_step_ahead(
    h: int, freq: str, origin: str | pd.Timestamp
) -> pd.DatetimeIndex:
    """Generate DatetimeIndex for h-step-ahead forecast.

    Parameters
    ----------
    h : int
        Number of steps ahead.
    freq : str
        Pandas frequency string.
    origin : str or pd.Timestamp
        Origin date.

    Returns
    -------
    pd.DatetimeIndex
        Forecast dates.
    """
    fh = ForecastHorizon(h, freq, origin)
    return fh.to_index()


def quarterly_from_monthly(
    origin: str | pd.Timestamp, n_quarters: int
) -> pd.DatetimeIndex:
    """Generate quarterly dates from monthly origin.

    Parameters
    ----------
    origin : str or pd.Timestamp
        Monthly origin date.
    n_quarters : int
        Number of quarters.

    Returns
    -------
    pd.DatetimeIndex
        Quarterly dates.
    """
    return h_step_ahead(n_quarters, "QS", origin)
