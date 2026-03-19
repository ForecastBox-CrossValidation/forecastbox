"""Fan chart visualization for forecast uncertainty (Bank of England style).

Fan charts display the predictive distribution as colored bands around
the median, with intensity decreasing from center to tails.

References
----------
Bank of England (2005). "The Bank of England Quarterly Model."
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

_DEFAULT_QUANTILES = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# Bank of England style color pairs (symmetric around median)
_DEFAULT_BAND_PAIRS = [
    (0.10, 0.90),  # outermost (lightest)
    (0.20, 0.80),
    (0.30, 0.70),
    (0.40, 0.60),  # innermost (darkest)
]


class FanChart:
    """Fan chart for forecast uncertainty visualization.

    Displays predictive distribution as nested colored bands around
    the median, inspired by the Bank of England inflation reports.

    Parameters
    ----------
    median : ndarray (H,)
        Median forecast values.
    quantiles : dict[float, ndarray]
        Mapping from quantile level (0-1) to values array (H,).
    history : ndarray or None
        Historical data for context (plotted before the forecast).
    index : DatetimeIndex or None
        Temporal index for the forecast horizon.

    Examples
    --------
    >>> fan = FanChart.from_ensemble(draws)
    >>> fan.plot(title="Inflation Forecast")
    """

    def __init__(
        self,
        median: NDArray[np.float64],
        quantiles: dict[float, NDArray[np.float64]],
        history: NDArray[np.float64] | None = None,
        index: pd.DatetimeIndex | None = None,
    ) -> None:
        self.median = np.asarray(median, dtype=np.float64)
        self.quantiles: dict[float, NDArray[np.float64]] = {
            k: np.asarray(v, dtype=np.float64) for k, v in quantiles.items()
        }
        self.history = (
            np.asarray(history, dtype=np.float64) if history is not None else None
        )
        self.index = index
        self._steps = len(self.median)

    @classmethod
    def from_ensemble(
        cls,
        draws: NDArray[np.float64],
        quantile_levels: list[float] | None = None,
        history: NDArray[np.float64] | None = None,
        index: pd.DatetimeIndex | None = None,
    ) -> FanChart:
        """Create FanChart from ensemble of simulated paths.

        Parameters
        ----------
        draws : ndarray (N, H)
            N simulated trajectories, each of length H.
        quantile_levels : list[float] or None
            Quantile levels to compute. Default: [0.10, 0.20, ..., 0.90].
        history : ndarray or None
            Historical data for context.
        index : DatetimeIndex or None
            Temporal index.

        Returns
        -------
        FanChart
            Fan chart with empirical quantiles.
        """
        draws = np.asarray(draws, dtype=np.float64)

        if draws.ndim != 2:
            msg = f"draws must be 2D (N, H), got shape {draws.shape}"
            raise ValueError(msg)

        levels = quantile_levels if quantile_levels is not None else _DEFAULT_QUANTILES

        median = np.median(draws, axis=0)
        quantiles: dict[float, NDArray[np.float64]] = {}
        for q in levels:
            quantiles[q] = np.quantile(draws, q, axis=0)

        return cls(median=median, quantiles=quantiles, history=history, index=index)

    @classmethod
    def from_gaussian(
        cls,
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
        quantile_levels: list[float] | None = None,
        history: NDArray[np.float64] | None = None,
        index: pd.DatetimeIndex | None = None,
    ) -> FanChart:
        """Create FanChart from Gaussian parameters.

        q_alpha(h) = mean(h) + std(h) * Phi^{-1}(alpha)

        Parameters
        ----------
        mean : ndarray (H,)
            Forecast mean at each horizon.
        std : ndarray (H,)
            Forecast standard deviation at each horizon.
        quantile_levels : list[float] or None
            Quantile levels. Default: [0.10, 0.20, ..., 0.90].
        history : ndarray or None
            Historical data for context.
        index : DatetimeIndex or None
            Temporal index.

        Returns
        -------
        FanChart
            Fan chart with Gaussian quantiles.
        """
        from scipy.stats import norm

        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)

        levels = quantile_levels if quantile_levels is not None else _DEFAULT_QUANTILES

        quantiles: dict[float, NDArray[np.float64]] = {}
        for q in levels:
            z = norm.ppf(q)
            quantiles[q] = mean + std * z

        # Median = mean for Gaussian
        median = mean.copy()

        return cls(median=median, quantiles=quantiles, history=history, index=index)

    def plot(
        self,
        ax: plt.Axes | None = None,
        title: str | None = None,
        color: str = "steelblue",
        alpha_range: tuple[float, float] = (0.15, 0.5),
        history_periods: int = 36,
        show_median: bool = True,
    ) -> plt.Axes:
        """Plot fan chart with colored probability bands.

        Parameters
        ----------
        ax : matplotlib Axes or None
            Axes to plot on. Creates new figure if None.
        title : str or None
            Plot title.
        color : str
            Base color for the bands.
        alpha_range : tuple[float, float]
            Min and max alpha transparency for bands.
            Outer bands use min, inner bands use max.
        history_periods : int
            Number of historical periods to show.
        show_median : bool
            Whether to show the median line.

        Returns
        -------
        plt.Axes
            The matplotlib Axes.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        h = self._steps

        # X-axis for forecast period
        x_forecast = self.index[:h] if self.index is not None else np.arange(1, h + 1)

        # Plot history if available
        if self.history is not None:
            n_hist = min(len(self.history), history_periods)
            if self.index is not None and len(self.index) > h:
                x_hist = self.index[-n_hist - h : -h]
                hist_data = self.history[-n_hist:]
            else:
                x_hist = np.arange(-n_hist, 0)
                hist_data = self.history[-n_hist:]
            ax.plot(x_hist, hist_data, "k-", linewidth=1.5, label="History")

        # Determine available band pairs
        available_pairs: list[tuple[float, float]] = []
        for lower_q, upper_q in _DEFAULT_BAND_PAIRS:
            if lower_q in self.quantiles and upper_q in self.quantiles:
                available_pairs.append((lower_q, upper_q))

        # Plot bands from outer to inner
        n_bands = len(available_pairs)
        for band_idx, (lower_q, upper_q) in enumerate(available_pairs):
            lower = self.quantiles[lower_q]
            upper = self.quantiles[upper_q]

            # Alpha increases from outer to inner
            t = band_idx / (n_bands - 1) if n_bands > 1 else 1.0
            band_alpha = alpha_range[0] + t * (alpha_range[1] - alpha_range[0])

            prob = upper_q - lower_q
            label = (
                f"{prob * 100:.0f}% CI"
                if band_idx == 0 or band_idx == n_bands - 1
                else None
            )
            ax.fill_between(
                x_forecast,
                lower,
                upper,
                alpha=band_alpha,
                color=color,
                label=label,
            )

        # Plot median
        if show_median:
            ax.plot(
                x_forecast, self.median, "-", color=color, linewidth=2, label="Median"
            )

        ax.set_title(title or "Fan Chart Forecast")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        return ax

    def to_dataframe(self) -> pd.DataFrame:
        """Convert fan chart to DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with median and all quantiles as columns.
        """
        data: dict[str, NDArray[np.float64]] = {"median": self.median}
        for q_level in sorted(self.quantiles.keys()):
            data[f"q{q_level:.2f}"] = self.quantiles[q_level]

        index: pd.DatetimeIndex | range = (
            self.index if self.index is not None else range(1, self._steps + 1)
        )
        return pd.DataFrame(data, index=index)

    def width_at_horizon(self, h: int, level: float = 0.90) -> float:
        """Compute width of the confidence band at a given horizon.

        width(h) = q_{(1+level)/2}(h) - q_{(1-level)/2}(h)

        Parameters
        ----------
        h : int
            Horizon index (0-based).
        level : float
            Confidence level (e.g., 0.90 for 90% band).

        Returns
        -------
        float
            Width of the band at horizon h.

        Raises
        ------
        ValueError
            If required quantiles are not available.
        """
        lower_q = (1 - level) / 2
        upper_q = (1 + level) / 2

        # Find closest available quantiles
        lower_val = self._get_nearest_quantile(lower_q, h)
        upper_val = self._get_nearest_quantile(upper_q, h)

        return float(upper_val - lower_val)

    def _get_nearest_quantile(self, target_q: float, h: int) -> float:
        """Get the value of the nearest available quantile at horizon h.

        Parameters
        ----------
        target_q : float
            Target quantile level.
        h : int
            Horizon index.

        Returns
        -------
        float
            Quantile value.
        """
        available = sorted(self.quantiles.keys())

        # Find nearest
        nearest_q = min(available, key=lambda q: abs(q - target_q))

        if abs(nearest_q - target_q) > 0.05:
            msg = (
                f"Quantile {target_q:.3f} not available. "
                f"Nearest is {nearest_q:.3f}. Available: {available}"
            )
            raise ValueError(msg)

        return float(self.quantiles[nearest_q][h])

    def contains(
        self,
        actual: NDArray[np.float64],
        level: float = 0.90,
    ) -> NDArray[np.bool_]:
        """Check which realized values fall within the confidence band.

        Parameters
        ----------
        actual : ndarray (H,)
            Realized values at each horizon.
        level : float
            Confidence level (e.g., 0.90).

        Returns
        -------
        ndarray[bool] (H,)
            True where actual falls within the band.
        """
        actual = np.asarray(actual, dtype=np.float64)
        lower_q = (1 - level) / 2
        upper_q = (1 + level) / 2

        # Get lower and upper bounds
        available = sorted(self.quantiles.keys())
        nearest_lower = min(available, key=lambda q: abs(q - lower_q))
        nearest_upper = min(available, key=lambda q: abs(q - upper_q))

        lower = self.quantiles[nearest_lower]
        upper = self.quantiles[nearest_upper]

        n = min(len(actual), len(lower))
        return (lower[:n] <= actual[:n]) & (actual[:n] <= upper[:n])
