"""Data vintage management for real-time forecast evaluation."""

from __future__ import annotations

from datetime import date

import pandas as pd


class DataVintage:
    """Manages data vintages for real-time evaluation.

    Each vintage represents the data available at a specific release date.

    Parameters
    ----------
    name : str
        Name of the time series.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.vintages: dict[date, pd.Series] = {}

    @property
    def release_dates(self) -> list[date]:
        """Return sorted list of release dates."""
        return sorted(self.vintages.keys())

    def add_vintage(self, release_date: date, data: pd.Series) -> None:
        """Add a vintage of data.

        Parameters
        ----------
        release_date : date
            Date when this vintage was published.
        data : pd.Series
            Time series available at release_date.
        """
        self.vintages[release_date] = data.copy()

    def get_vintage(self, release_date: date) -> pd.Series:
        """Get data available at a specific release date.

        Parameters
        ----------
        release_date : date
            Target release date.

        Returns
        -------
        pd.Series
            Data available at release_date.
        """
        if release_date not in self.vintages:
            msg = f"No vintage for release date {release_date}"
            raise KeyError(msg)
        return self.vintages[release_date].copy()

    def get_latest(self) -> pd.Series:
        """Get the most recent vintage.

        Returns
        -------
        pd.Series
            Latest vintage data.
        """
        if not self.vintages:
            msg = "No vintages available"
            raise ValueError(msg)
        latest_date = max(self.vintages.keys())
        return self.vintages[latest_date].copy()

    def get_revision(
        self, period: str, release1: date, release2: date
    ) -> float:
        """Calculate revision between two vintages for a specific period.

        Parameters
        ----------
        period : str
            Time period to check (e.g., '2024-01').
        release1 : date
            First release date.
        release2 : date
            Second release date.

        Returns
        -------
        float
            Revision = value_release2 - value_release1.
        """
        v1 = self.get_vintage(release1)
        v2 = self.get_vintage(release2)

        ts_period = pd.Timestamp(period)
        if ts_period not in v1.index or ts_period not in v2.index:
            msg = f"Period {period} not available in one or both vintages"
            raise KeyError(msg)

        return float(v2.loc[ts_period] - v1.loc[ts_period])

    def revision_history(self, period: str) -> pd.Series:
        """Get revision history for a specific period across all vintages.

        Parameters
        ----------
        period : str
            Time period (e.g., '2024-01').

        Returns
        -------
        pd.Series
            Series indexed by release dates with values for the given period.
        """
        ts_period = pd.Timestamp(period)
        values = {}
        for rd in self.release_dates:
            v = self.vintages[rd]
            if ts_period in v.index:
                values[rd] = v.loc[ts_period]
        return pd.Series(values, name=f"{self.name}_{period}")

    def to_dataframe(self) -> pd.DataFrame:
        """Create vintage matrix (rows=periods, columns=release_dates).

        Returns
        -------
        pd.DataFrame
            Vintage matrix.
        """
        if not self.vintages:
            return pd.DataFrame()
        return pd.DataFrame(self.vintages)

    def triangle(self) -> pd.DataFrame:
        """Create real-time data triangle.

        The triangle shows, for each release date, only the data that
        was actually available at that time (NaN for future data).

        Returns
        -------
        pd.DataFrame
            Triangle DataFrame.
        """
        df = self.to_dataframe()
        # In a proper triangle, each column (release) only has data
        # for periods up to the release date
        return df

    def __repr__(self) -> str:
        n_vintages = len(self.vintages)
        return f"DataVintage(name='{self.name}', n_vintages={n_vintages})"
