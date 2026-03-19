"""Real-time data manager for nowcasting with publication calendars and ragged edge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SeriesInfo:
    """Metadata for a macroeconomic time series.

    Parameters
    ----------
    name : str
        Series identifier.
    frequency : str
        Data frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly).
    release_calendar : str
        Publication frequency: 'daily', 'weekly', 'monthly', 'quarterly'.
    lag_days : int
        Publication delay in days after end of reference period.
    source : str
        Data source (e.g. 'IBGE', 'BCB', 'BLS').
    transform : str
        Transformation applied: 'none', 'diff', 'log_diff', 'yoy'.
    seasonal_adjustment : bool
        Whether the series is seasonally adjusted.
    """

    name: str
    frequency: str
    release_calendar: str
    lag_days: int
    source: str = ""
    transform: str = "none"
    seasonal_adjustment: bool = True

    def __post_init__(self) -> None:
        valid_freq = {"D", "W", "M", "Q"}
        if self.frequency not in valid_freq:
            msg = f"frequency must be one of {valid_freq}, got '{self.frequency}'"
            raise ValueError(msg)
        valid_cal = {"daily", "weekly", "monthly", "quarterly"}
        if self.release_calendar not in valid_cal:
            msg = f"release_calendar must be one of {valid_cal}, got '{self.release_calendar}'"
            raise ValueError(msg)
        if self.lag_days < 0:
            msg = f"lag_days must be >= 0, got {self.lag_days}"
            raise ValueError(msg)
        valid_transforms = {"none", "diff", "log_diff", "yoy"}
        if self.transform not in valid_transforms:
            msg = f"transform must be one of {valid_transforms}, got '{self.transform}'"
            raise ValueError(msg)


def _end_of_period(period_date: date, frequency: str) -> date:
    """Compute end of the reference period for a given date and frequency.

    Parameters
    ----------
    period_date : date
        A date within the reference period.
    frequency : str
        Frequency: 'D', 'W', 'M', 'Q'.

    Returns
    -------
    date
        Last day of the reference period.
    """
    if frequency == "D":
        return period_date
    elif frequency == "W":
        # End of ISO week (Sunday)
        days_to_sunday = 6 - period_date.weekday()
        return period_date + timedelta(days=days_to_sunday)
    elif frequency == "M":
        # End of month
        if period_date.month == 12:
            return date(period_date.year + 1, 1, 1) - timedelta(days=1)
        return date(period_date.year, period_date.month + 1, 1) - timedelta(days=1)
    elif frequency == "Q":
        # End of quarter
        quarter_end_month = ((period_date.month - 1) // 3 + 1) * 3
        if quarter_end_month == 12:
            return date(period_date.year + 1, 1, 1) - timedelta(days=1)
        return date(period_date.year, quarter_end_month + 1, 1) - timedelta(days=1)
    else:
        msg = f"Unknown frequency: {frequency}"
        raise ValueError(msg)


class RealTimeDataManager:
    """Manager for real-time data with publication calendars and ragged edge.

    Handles macroeconomic data published at different dates with different
    lags. Integrates with DataVintage from Phase 1 for revision tracking.

    Examples
    --------
    >>> rtdm = RealTimeDataManager()
    >>> rtdm.add_series('ipca', frequency='M', release_calendar='monthly', lag_days=15)
    >>> rtdm.add_series('pib', frequency='Q', release_calendar='quarterly', lag_days=60)
    """

    def __init__(self) -> None:
        self._series: dict[str, SeriesInfo] = {}
        self._data: dict[str, pd.Series] = {}
        self._vintages: dict[str, list[tuple[str, pd.Series]]] = {}

    @property
    def series(self) -> dict[str, SeriesInfo]:
        """Return registered series metadata."""
        return dict(self._series)

    @property
    def series_names(self) -> list[str]:
        """Return list of registered series names."""
        return sorted(self._series.keys())

    def add_series(
        self,
        name: str,
        frequency: str = "M",
        release_calendar: str = "monthly",
        lag_days: int = 30,
        source: str = "",
        transform: str = "none",
        seasonal_adjustment: bool = True,
        data: pd.Series | None = None,
    ) -> None:
        """Register a new series with its publication metadata.

        Parameters
        ----------
        name : str
            Series identifier.
        frequency : str
            Data frequency: 'D', 'W', 'M', 'Q'.
        release_calendar : str
            Publication frequency: 'daily', 'weekly', 'monthly', 'quarterly'.
        lag_days : int
            Publication delay in days after end of reference period.
        source : str
            Data source name.
        transform : str
            Transformation: 'none', 'diff', 'log_diff', 'yoy'.
        seasonal_adjustment : bool
            Whether the series is seasonally adjusted.
        data : pd.Series or None
            Initial data for the series. Index must be DatetimeIndex.
        """
        info = SeriesInfo(
            name=name,
            frequency=frequency,
            release_calendar=release_calendar,
            lag_days=lag_days,
            source=source,
            transform=transform,
            seasonal_adjustment=seasonal_adjustment,
        )
        self._series[name] = info
        self._vintages[name] = []

        if data is not None:
            self._data[name] = data.copy()
            self._vintages[name].append(
                (pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), data.copy())
            )
        else:
            self._data[name] = pd.Series(dtype=np.float64)

    def remove_series(self, name: str) -> None:
        """Remove a registered series.

        Parameters
        ----------
        name : str
            Series name to remove.

        Raises
        ------
        KeyError
            If the series is not registered.
        """
        if name not in self._series:
            msg = f"Series '{name}' not found. Available: {self.series_names}"
            raise KeyError(msg)
        del self._series[name]
        self._data.pop(name, None)
        self._vintages.pop(name, None)

    def update(self, new_data: dict[str, pd.Series]) -> None:
        """Update series with new data, creating a new vintage.

        Parameters
        ----------
        new_data : dict[str, pd.Series]
            Dictionary mapping series name to new data. Each Series should have
            a DatetimeIndex. New observations are appended; existing observations
            may be revised.
        """
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        for name, series in new_data.items():
            if name not in self._series:
                msg = f"Series '{name}' not registered. Call add_series() first."
                raise KeyError(msg)

            # Combine old and new data, new values overwrite revisions
            if name in self._data and len(self._data[name]) > 0:
                combined = self._data[name].copy()
                for idx_val in series.index:
                    combined[idx_val] = series[idx_val]
                combined = combined.sort_index()
            else:
                combined = series.copy().sort_index()

            self._data[name] = combined
            self._vintages[name].append((timestamp, combined.copy()))

    def get_available_data(
        self, reference_date: str | date | pd.Timestamp
    ) -> pd.DataFrame:
        """Get data available as of a reference date, respecting publication lags.

        Parameters
        ----------
        reference_date : str, date, or Timestamp
            The date at which to evaluate data availability. Observations are
            only available if their publication date <= reference_date.

        Returns
        -------
        pd.DataFrame
            DataFrame with available data. Columns are series names, index is
            the union of all available observation dates. Missing values are NaN.
        """
        if isinstance(reference_date, str):
            ref = pd.Timestamp(reference_date).date()
        elif isinstance(reference_date, pd.Timestamp):
            ref = reference_date.date()
        else:
            ref = reference_date

        available: dict[str, pd.Series] = {}

        for name, info in self._series.items():
            if name not in self._data or len(self._data[name]) == 0:
                continue

            series = self._data[name]
            mask = []
            for obs_date in series.index:
                obs_d = pd.Timestamp(obs_date).date()
                end_period = _end_of_period(obs_d, info.frequency)
                delay = max(0, info.lag_days - 1) if info.lag_days > 0 else 0
                pub_date = end_period + timedelta(days=delay)
                mask.append(pub_date <= ref)

            available_series = series[mask]
            if len(available_series) > 0:
                available[name] = available_series

        if not available:
            return pd.DataFrame()

        return pd.DataFrame(available)

    def get_ragged_edge(
        self, reference_date: str | date | pd.Timestamp
    ) -> dict[str, date | None]:
        """Get the last available observation date for each series.

        The 'ragged edge' is the irregular pattern of data availability
        across different series at a given reference date.

        Parameters
        ----------
        reference_date : str, date, or Timestamp
            The date at which to evaluate availability.

        Returns
        -------
        dict[str, date | None]
            Dictionary mapping series name to the date of its last available
            observation. None if no data is available.
        """
        available_df = self.get_available_data(reference_date)
        result: dict[str, date | None] = {}

        for name in self._series:
            if name in available_df.columns:
                valid = available_df[name].dropna()
                if len(valid) > 0:
                    result[name] = pd.Timestamp(valid.index[-1]).date()
                else:
                    result[name] = None
            else:
                result[name] = None

        return result

    def get_missing_pattern(
        self, reference_date: str | date | pd.Timestamp
    ) -> pd.DataFrame:
        """Get a boolean panel showing data availability.

        Parameters
        ----------
        reference_date : str, date, or Timestamp
            The date at which to evaluate availability.

        Returns
        -------
        pd.DataFrame
            Boolean DataFrame where True = data available, False = missing.
            Columns are series names, index is the union of expected periods.
        """
        available_df = self.get_available_data(reference_date)

        if available_df.empty:
            return pd.DataFrame()

        return available_df.notna()

    def simulate_publication(
        self,
        start_date: str | date | pd.Timestamp,
        end_date: str | date | pd.Timestamp,
    ) -> list[dict[str, Any]]:
        """Simulate the sequence of data publications between two dates.

        Parameters
        ----------
        start_date : str, date, or Timestamp
            Start of simulation period.
        end_date : str, date, or Timestamp
            End of simulation period.

        Returns
        -------
        list[dict]
            List of publication events, each with keys:
            - 'date': publication date
            - 'series': series name
            - 'period': reference period of the observation
            - 'lag_days': publication lag
        """
        if isinstance(start_date, str):
            start = pd.Timestamp(start_date).date()
        elif isinstance(start_date, pd.Timestamp):
            start = start_date.date()
        else:
            start = start_date

        if isinstance(end_date, str):
            end = pd.Timestamp(end_date).date()
        elif isinstance(end_date, pd.Timestamp):
            end = end_date.date()
        else:
            end = end_date

        events: list[dict[str, Any]] = []

        for name, info in self._series.items():
            if name not in self._data or len(self._data[name]) == 0:
                continue

            for obs_date in self._data[name].index:
                obs_d = pd.Timestamp(obs_date).date()
                end_period = _end_of_period(obs_d, info.frequency)
                delay = max(0, info.lag_days - 1) if info.lag_days > 0 else 0
                pub_date = end_period + timedelta(days=delay)

                if start <= pub_date <= end:
                    events.append({
                        "date": pub_date,
                        "series": name,
                        "period": obs_d,
                        "lag_days": info.lag_days,
                    })

        # Sort by publication date, then by series name
        events.sort(key=lambda x: (x["date"], x["series"]))
        return events

    def plot_ragged_edge(
        self,
        reference_date: str | date | pd.Timestamp,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot the ragged edge showing data availability across series.

        Parameters
        ----------
        reference_date : str, date, or Timestamp
            The reference date for availability.
        ax : matplotlib Axes or None
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(14, max(4, len(self._series) * 0.6)))

        missing = self.get_missing_pattern(reference_date)

        if missing.empty:
            ax.text(
                0.5, 0.5, "No data available",
                ha="center", va="center", transform=ax.transAxes,
            )
            return ax

        # Plot heatmap-style availability
        series_names = list(missing.columns)
        n_series = len(series_names)
        n_periods = len(missing.index)

        for i, name in enumerate(series_names):
            for j, period in enumerate(missing.index):
                color = "green" if missing.loc[period, name] else "red"
                alpha = 0.7 if missing.loc[period, name] else 0.3
                ax.barh(
                    i, 1, left=j, height=0.8, color=color, alpha=alpha,
                    edgecolor="white", linewidth=0.5,
                )

        ax.set_yticks(range(n_series))
        ax.set_yticklabels(series_names)

        # X-axis: show a subset of dates
        if n_periods > 0:
            step = max(1, n_periods // 8)
            tick_positions = list(range(0, n_periods, step))
            tick_labels = [str(missing.index[p])[:10] for p in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        ax.set_title(f"Ragged Edge (reference: {reference_date})")
        ax.set_xlabel("Period")

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="Available"),
            Patch(facecolor="red", alpha=0.3, label="Missing"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()
        return ax

    def plot_release_calendar(
        self,
        period: str | tuple[str, str] = "2024-Q1",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot a calendar view of publication dates.

        Parameters
        ----------
        period : str or tuple[str, str]
            Either a quarter string ('2024-Q1') or a (start, end) tuple of dates.
        ax : matplotlib Axes or None
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(14, max(4, len(self._series) * 0.6)))

        if isinstance(period, str):
            # Parse quarter string
            year = int(period.split("-")[0])
            quarter = int(period.split("Q")[1])
            start = date(year, (quarter - 1) * 3 + 1, 1)
            if quarter == 4:
                end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = date(year, quarter * 3 + 1, 1) - timedelta(days=1)
        else:
            start = pd.Timestamp(period[0]).date()
            end = pd.Timestamp(period[1]).date()

        events = self.simulate_publication(start, end)

        if not events:
            ax.text(
                0.5, 0.5, "No publications in period",
                ha="center", va="center", transform=ax.transAxes,
            )
            return ax

        # Plot events on timeline
        series_names = sorted(set(e["series"] for e in events))
        series_idx = {name: i for i, name in enumerate(series_names)}
        colors = plt.cm.Set3(np.linspace(0, 1, len(series_names)))

        for event in events:
            idx = series_idx[event["series"]]
            day_offset = (event["date"] - start).days
            ax.barh(
                idx, 1, left=day_offset, height=0.6,
                color=colors[idx], edgecolor="black", linewidth=0.5,
            )

        ax.set_yticks(range(len(series_names)))
        ax.set_yticklabels(series_names)
        ax.set_xlabel("Days from start")
        ax.set_title(f"Release Calendar ({start} to {end})")

        plt.tight_layout()
        return ax

    def __repr__(self) -> str:
        n = len(self._series)
        names = ", ".join(list(self._series.keys())[:5])
        if n > 5:
            names += ", ..."
        return f"RealTimeDataManager(n_series={n}, series=[{names}])"
