"""Forecast container for point, interval, and density predictions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class Forecast:
    """Container for a single model's forecast.

    Stores point forecasts, prediction intervals (80% and 95%),
    optional density draws, temporal index, and metadata.

    Parameters
    ----------
    point : NDArray[np.float64]
        Point forecasts for h=1..H.
    lower_80 : NDArray[np.float64] or None
        Lower bound of 80% prediction interval.
    upper_80 : NDArray[np.float64] or None
        Upper bound of 80% prediction interval.
    lower_95 : NDArray[np.float64] or None
        Lower bound of 95% prediction interval.
    upper_95 : NDArray[np.float64] or None
        Upper bound of 95% prediction interval.
    density : NDArray[np.float64] or None
        Density draws of shape (H, N) where N is number of draws.
    index : pd.DatetimeIndex or None
        Temporal index for the forecast horizon.
    model_name : str
        Name of the model that generated this forecast.
    horizon : int or None
        Number of steps ahead. Defaults to len(point).
    metadata : dict or None
        Additional metadata (parameters, information criteria, etc.).
    """

    def __init__(
        self,
        point: NDArray[np.float64],
        lower_80: NDArray[np.float64] | None = None,
        upper_80: NDArray[np.float64] | None = None,
        lower_95: NDArray[np.float64] | None = None,
        upper_95: NDArray[np.float64] | None = None,
        density: NDArray[np.float64] | None = None,
        index: pd.DatetimeIndex | None = None,
        model_name: str = "",
        horizon: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.point = np.asarray(point, dtype=np.float64)
        self.lower_80 = np.asarray(lower_80, dtype=np.float64) if lower_80 is not None else None
        self.upper_80 = np.asarray(upper_80, dtype=np.float64) if upper_80 is not None else None
        self.lower_95 = np.asarray(lower_95, dtype=np.float64) if lower_95 is not None else None
        self.upper_95 = np.asarray(upper_95, dtype=np.float64) if upper_95 is not None else None
        self.density = np.asarray(density, dtype=np.float64) if density is not None else None
        self.index = index
        self.model_name = model_name
        self.horizon = horizon if horizon is not None else len(self.point)
        self.created_at = datetime.now()
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """Return the forecast horizon."""
        return self.horizon

    def __getitem__(self, i: int) -> dict[str, float]:
        """Return point and intervals for horizon i.

        Parameters
        ----------
        i : int
            Horizon index (0-based).

        Returns
        -------
        dict[str, float]
            Dictionary with point, lower_80, upper_80, lower_95, upper_95.
        """
        result: dict[str, float] = {"point": float(self.point[i])}
        if self.lower_80 is not None:
            result["lower_80"] = float(self.lower_80[i])
        if self.upper_80 is not None:
            result["upper_80"] = float(self.upper_80[i])
        if self.lower_95 is not None:
            result["lower_95"] = float(self.lower_95[i])
        if self.upper_95 is not None:
            result["upper_95"] = float(self.upper_95[i])
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert forecast to DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: point, lower_80, upper_80, lower_95, upper_95.
        """
        data: dict[str, NDArray[np.float64]] = {"point": self.point}
        if self.lower_80 is not None:
            data["lower_80"] = self.lower_80
        if self.upper_80 is not None:
            data["upper_80"] = self.upper_80
        if self.lower_95 is not None:
            data["lower_95"] = self.lower_95
        if self.upper_95 is not None:
            data["upper_95"] = self.upper_95

        idx = self.index if self.index is not None else range(self.horizon)
        return pd.DataFrame(data, index=idx)

    @classmethod
    def from_distribution(
        cls,
        draws: NDArray[np.float64],
        index: pd.DatetimeIndex | None = None,
        model_name: str = "",
        quantiles: tuple[float, ...] = (0.10, 0.90, 0.025, 0.975),
    ) -> Forecast:
        """Create Forecast from distribution draws.

        Parameters
        ----------
        draws : NDArray[np.float64]
            Array of shape (H, N) with N draws for each of H horizons.
        index : pd.DatetimeIndex or None
            Temporal index.
        model_name : str
            Model name.
        quantiles : tuple[float, ...]
            Quantiles for intervals. Default: (0.10, 0.90, 0.025, 0.975)
            which gives 80% and 95% intervals.

        Returns
        -------
        Forecast
            Forecast with point = median, intervals from quantiles.
        """
        draws = np.asarray(draws, dtype=np.float64)
        point = np.median(draws, axis=1)
        q = np.quantile(draws, quantiles, axis=1)

        return cls(
            point=point,
            lower_80=q[0],
            upper_80=q[1],
            lower_95=q[2],
            upper_95=q[3],
            density=draws,
            index=index,
            model_name=model_name,
        )

    def plot(
        self,
        actual: NDArray[np.float64] | None = None,
        ax: plt.Axes | None = None,
        title: str | None = None,
        show_intervals: bool = True,
    ) -> plt.Axes:
        """Plot forecast with confidence bands.

        Parameters
        ----------
        actual : NDArray[np.float64] or None
            Actual values to overlay.
        ax : matplotlib Axes or None
            Axes to plot on. Creates new figure if None.
        title : str or None
            Plot title.
        show_intervals : bool
            Whether to show prediction intervals.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        x = self.index if self.index is not None else np.arange(self.horizon)

        ax.plot(x, self.point, "b-", label=f"Forecast ({self.model_name})", linewidth=2)

        if show_intervals:
            if self.lower_95 is not None and self.upper_95 is not None:
                ax.fill_between(
                    x, self.lower_95, self.upper_95,
                    alpha=0.15, color="blue", label="95% CI",
                )
            if self.lower_80 is not None and self.upper_80 is not None:
                ax.fill_between(
                    x, self.lower_80, self.upper_80,
                    alpha=0.30, color="blue", label="80% CI",
                )

        if actual is not None:
            ax.plot(x, actual, "ro-", label="Actual", linewidth=1.5)

        ax.set_title(title or f"Forecast: {self.model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    @staticmethod
    def combine(forecasts: list[Forecast], method: str = "mean") -> Forecast:
        """Combine multiple forecasts.

        Parameters
        ----------
        forecasts : list[Forecast]
            List of forecasts to combine.
        method : str
            Combination method: 'mean' or 'median'.

        Returns
        -------
        Forecast
            Combined forecast.
        """
        if not forecasts:
            msg = "Cannot combine empty list of forecasts"
            raise ValueError(msg)

        points = np.array([fc.point for fc in forecasts])

        if method == "mean":
            combined_point = np.mean(points, axis=0)
        elif method == "median":
            combined_point = np.median(points, axis=0)
        else:
            msg = f"Unknown combination method: {method}. Use 'mean' or 'median'."
            raise ValueError(msg)

        # Combine intervals if available
        lower_80 = None
        upper_80 = None
        lower_95 = None
        upper_95 = None

        if all(fc.lower_80 is not None for fc in forecasts):
            lower_80_arr = np.array([fc.lower_80 for fc in forecasts])
            agg = np.mean if method == "mean" else np.median
            lower_80 = agg(lower_80_arr, axis=0)
        if all(fc.upper_80 is not None for fc in forecasts):
            upper_80_arr = np.array([fc.upper_80 for fc in forecasts])
            agg = np.mean if method == "mean" else np.median
            upper_80 = agg(upper_80_arr, axis=0)
        if all(fc.lower_95 is not None for fc in forecasts):
            lower_95_arr = np.array([fc.lower_95 for fc in forecasts])
            agg = np.mean if method == "mean" else np.median
            lower_95 = agg(lower_95_arr, axis=0)
        if all(fc.upper_95 is not None for fc in forecasts):
            upper_95_arr = np.array([fc.upper_95 for fc in forecasts])
            agg = np.mean if method == "mean" else np.median
            upper_95 = agg(upper_95_arr, axis=0)

        model_names = [fc.model_name for fc in forecasts]

        return Forecast(
            point=combined_point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            index=forecasts[0].index,
            model_name=f"Combined({method})",
            metadata={"method": method, "models": model_names},
        )

    def validate(self) -> None:
        """Validate forecast consistency.

        Raises
        ------
        ValueError
            If intervals are inconsistent.
        """
        from forecastbox.utils.validation import check_forecast

        check_forecast(self)

    def save(self, path: str | Path) -> None:
        """Save forecast to JSON file.

        Parameters
        ----------
        path : str or Path
            File path to save to.
        """
        data: dict[str, Any] = {
            "point": self.point.tolist(),
            "model_name": self.model_name,
            "horizon": self.horizon,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
        if self.lower_80 is not None:
            data["lower_80"] = self.lower_80.tolist()
        if self.upper_80 is not None:
            data["upper_80"] = self.upper_80.tolist()
        if self.lower_95 is not None:
            data["lower_95"] = self.lower_95.tolist()
        if self.upper_95 is not None:
            data["upper_95"] = self.upper_95.tolist()
        if self.density is not None:
            data["density"] = self.density.tolist()
        if self.index is not None:
            data["index"] = [str(d) for d in self.index]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Forecast:
        """Load forecast from JSON file.

        Parameters
        ----------
        path : str or Path
            File path to load from.

        Returns
        -------
        Forecast
            Loaded forecast.
        """
        with open(path) as f:
            data = json.load(f)

        index = None
        if "index" in data:
            index = pd.DatetimeIndex(data["index"])

        fc = cls(
            point=np.array(data["point"]),
            lower_80=np.array(data["lower_80"]) if "lower_80" in data else None,
            upper_80=np.array(data["upper_80"]) if "upper_80" in data else None,
            lower_95=np.array(data["lower_95"]) if "lower_95" in data else None,
            upper_95=np.array(data["upper_95"]) if "upper_95" in data else None,
            density=np.array(data["density"]) if "density" in data else None,
            index=index,
            model_name=data.get("model_name", ""),
            horizon=data.get("horizon"),
            metadata=data.get("metadata", {}),
        )
        return fc

    def __repr__(self) -> str:
        intervals = ""
        if self.lower_80 is not None:
            intervals += ", 80%CI"
        if self.lower_95 is not None:
            intervals += ", 95%CI"
        density = ", density" if self.density is not None else ""
        return (
            f"Forecast(model='{self.model_name}', horizon={self.horizon}"
            f"{intervals}{density})"
        )
