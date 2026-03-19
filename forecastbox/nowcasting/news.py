"""News decomposition for nowcast revisions (Banbura-Modugno 2014)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class NewsResult:
    """Result of a news decomposition.

    The total revision of the nowcast is decomposed into individual
    contributions from each newly released data point.

    Attributes
    ----------
    total_revision : float
        Total change in nowcast: new_nowcast - old_nowcast.
    old_nowcast : float
        Nowcast computed with old information set.
    new_nowcast : float
        Nowcast computed with new information set.
    news : dict[str, float]
        News (surprise) for each indicator: observed - expected.
    weights : dict[str, float]
        Weight of each indicator in the nowcast revision.
    contributions : dict[str, float]
        Contribution of each indicator: weight * news.
    released_data : dict[str, float]
        Newly released data values.
    expected_data : dict[str, float]
        Model's expectation for the released data.
    """

    total_revision: float
    old_nowcast: float
    new_nowcast: float
    news: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    released_data: dict[str, float] = field(default_factory=dict)
    expected_data: dict[str, float] = field(default_factory=dict)

    def plot_contributions(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot bar chart of individual contributions.

        Parameters
        ----------
        ax : matplotlib Axes or None
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        if not self.contributions:
            ax.text(
                0.5,
                0.5,
                "No contributions",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return ax

        indicators = list(self.contributions.keys())
        values = list(self.contributions.values())

        colors = ["green" if v >= 0 else "red" for v in values]

        ax.barh(indicators, values, color=colors, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Contribution to Nowcast Revision")
        ax.set_title(
            f"News Decomposition (Total Revision: {self.total_revision:.4f})"
        )
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="x")

        return ax

    def plot_waterfall(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot waterfall chart: old_nowcast + contributions = new_nowcast.

        Parameters
        ----------
        ax : matplotlib Axes or None
            Axes to plot on. Creates new figure if None.

        Returns
        -------
        plt.Axes
            The matplotlib Axes object.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        # Build waterfall data
        labels: list[str] = ["Old Nowcast"]
        values: list[float] = [self.old_nowcast]
        bottoms: list[float] = [0.0]

        cumulative = self.old_nowcast

        # Sort contributions by absolute value (largest first)
        sorted_contribs = sorted(
            self.contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        for indicator, contrib in sorted_contribs:
            labels.append(indicator)
            values.append(contrib)
            bottoms.append(cumulative)
            cumulative += contrib

        labels.append("New Nowcast")
        values.append(self.new_nowcast)
        bottoms.append(0.0)

        # Plot bars
        n = len(labels)
        colors: list[str] = []
        for i, v in enumerate(values):
            if i == 0 or i == n - 1:
                colors.append("steelblue")  # Start/end
            elif v >= 0:
                colors.append("green")
            else:
                colors.append("red")

        x_pos = np.arange(n)

        for i in range(n):
            if i == 0 or i == n - 1:
                # Full bars for start and end
                ax.bar(
                    x_pos[i],
                    values[i],
                    bottom=0,
                    color=colors[i],
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                )
            else:
                # Incremental bars
                ax.bar(
                    x_pos[i],
                    values[i],
                    bottom=bottoms[i],
                    color=colors[i],
                    alpha=0.7,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # Connect bars with lines
        for i in range(n - 1):
            level = values[0] if i == 0 else bottoms[i] + values[i]
            ax.plot(
                [x_pos[i] + 0.4, x_pos[i + 1] - 0.4],
                [level, level],
                "k--",
                linewidth=0.5,
                alpha=0.5,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Nowcast Value")
        ax.set_title("Nowcast Revision Waterfall")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        return ax

    def summary(self) -> str:
        """Return a text summary of the news decomposition.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            "=" * 70,
            "News Decomposition Summary",
            "=" * 70,
            f"Old nowcast:      {self.old_nowcast:.4f}",
            f"New nowcast:      {self.new_nowcast:.4f}",
            f"Total revision:   {self.total_revision:.4f}",
            "-" * 70,
            f"{'Indicator':<30s} {'News':>10s} {'Weight':>10s} {'Contribution':>14s}",
            "-" * 70,
        ]

        sorted_contribs = sorted(
            self.contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        for indicator, contrib in sorted_contribs:
            news_val = self.news.get(indicator, 0.0)
            weight_val = self.weights.get(indicator, 0.0)
            lines.append(
                f"{indicator:<30s} {news_val:>10.4f} {weight_val:>10.4f}"
                f" {contrib:>14.4f}"
            )

        lines.append("-" * 70)
        total_contrib = sum(self.contributions.values())
        lines.append(
            f"{'Sum of contributions':<30s} {'':>10s} {'':>10s}"
            f" {total_contrib:>14.4f}"
        )
        lines.append(
            f"{'Revision':<30s} {'':>10s} {'':>10s}"
            f" {self.total_revision:>14.4f}"
        )
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert decomposition to DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns [indicator, news, weight, contribution].
        """
        indicators = list(self.contributions.keys())
        data = {
            "indicator": indicators,
            "news": [self.news.get(ind, 0.0) for ind in indicators],
            "weight": [self.weights.get(ind, 0.0) for ind in indicators],
            "contribution": [self.contributions[ind] for ind in indicators],
        }
        return pd.DataFrame(data)


class NewsDecomposition:
    """Decompose nowcast revisions into individual data contributions.

    Follows Banbura & Modugno (2014): when new data is released, the
    change in the nowcast is decomposed exactly into contributions from
    each newly released indicator.

    Parameters
    ----------
    nowcaster : DFMNowcaster
        A fitted DFMNowcaster instance.

    Examples
    --------
    >>> news = NewsDecomposition(nowcaster)
    >>> result = news.decompose(old_data, new_data, target='pib')
    >>> print(result.total_revision)
    >>> print(result.contributions)
    >>> result.plot_waterfall()
    """

    def __init__(self, nowcaster: Any) -> None:
        self.nowcaster = nowcaster

        if not hasattr(nowcaster, "_fitted") or not nowcaster._fitted:
            msg = "Nowcaster must be fitted before creating NewsDecomposition"
            raise RuntimeError(msg)

    def _compute_news(
        self,
        old_data: pd.DataFrame,
        new_data: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute the news (surprise) for each indicator.

        News = observed value - model expectation

        Parameters
        ----------
        old_data : pd.DataFrame
            Data available before the new release.
        new_data : pd.DataFrame
            Data available after the new release.

        Returns
        -------
        dict[str, float]
            News for each indicator that has new observations.
        """
        news: dict[str, float] = {}

        for var in self.nowcaster._var_names:
            if var not in old_data.columns or var not in new_data.columns:
                continue

            old_series = old_data[var].dropna()
            new_series = new_data[var].dropna()

            # Find newly available observations
            if len(new_series) > len(old_series):
                new_obs_idx = new_series.index.difference(old_series.index)

                for idx in new_obs_idx:
                    observed = float(new_series[idx])

                    # Model expectation: use the loading and filtered state
                    var_idx = self.nowcaster._var_names.index(var)
                    nf = self.nowcaster.n_factors

                    if (
                        self.nowcaster._kalman_result is not None
                        and self.nowcaster._kalman_result.filtered_state is not None
                        and self.nowcaster._Lambda is not None
                    ):
                        # Find the time index for this observation
                        if (
                            self.nowcaster._data is not None
                            and idx in self.nowcaster._data.index
                        ):
                            t_idx = self.nowcaster._data.index.get_loc(idx)
                            state = (
                                self.nowcaster._kalman_result.filtered_state[
                                    t_idx, :nf
                                ]
                            )
                            expected = float(
                                self.nowcaster._Lambda[var_idx] @ state
                            )
                        else:
                            # Use last available state
                            state = (
                                self.nowcaster._kalman_result.filtered_state[
                                    -1, :nf
                                ]
                            )
                            expected = float(
                                self.nowcaster._Lambda[var_idx] @ state
                            )
                    else:
                        # Fallback: use old series mean as expectation
                        expected = (
                            float(old_series.mean()) if len(old_series) > 0
                            else 0.0
                        )

                    news_val = observed - expected
                    # Aggregate if multiple new obs for same variable
                    if var in news:
                        news[var] += news_val
                    else:
                        news[var] = news_val

            # Check for revised observations
            common_idx = old_series.index.intersection(new_series.index)
            for idx in common_idx:
                old_val = float(old_series[idx])
                new_val = float(new_series[idx])
                if abs(new_val - old_val) > 1e-10:
                    revision = new_val - old_val
                    if var in news:
                        news[var] += revision
                    else:
                        news[var] = revision

        return news

    def _compute_weights(
        self,
        target: str,
    ) -> dict[str, float]:
        """Compute the weight of each indicator for the target nowcast.

        Weight = Cov(target, indicator | old_info) / Var(indicator | old_info)

        Parameters
        ----------
        target : str
            Target variable name.

        Returns
        -------
        dict[str, float]
            Weight for each indicator.
        """
        weights: dict[str, float] = {}

        if target not in self.nowcaster._var_names:
            return weights

        target_idx = self.nowcaster._var_names.index(target)
        nf = self.nowcaster.n_factors

        if self.nowcaster._Lambda is None or self.nowcaster._R is None:
            return weights

        lambda_y = self.nowcaster._Lambda[target_idx]  # (nf,)

        # Use the last filtered covariance
        if (
            self.nowcaster._kalman_result is not None
            and self.nowcaster._kalman_result.filtered_cov is not None
        ):
            p_cov = self.nowcaster._kalman_result.filtered_cov[-1, :nf, :nf]
        else:
            p_cov = np.eye(nf)

        for var in self.nowcaster._var_names:
            var_idx = self.nowcaster._var_names.index(var)
            lambda_i = self.nowcaster._Lambda[var_idx]  # (nf,)

            # Cov(y, x_i | old) = lambda_y @ P @ lambda_i'
            cov_y_xi = float(lambda_y @ p_cov @ lambda_i)

            # Var(x_i | old) = lambda_i @ P @ lambda_i' + R_ii
            var_xi = float(
                lambda_i @ p_cov @ lambda_i
                + self.nowcaster._R[var_idx, var_idx]
            )

            if abs(var_xi) > 1e-10:
                weights[var] = cov_y_xi / var_xi
            else:
                weights[var] = 0.0

        return weights

    def _compute_contributions(
        self,
        news: dict[str, float],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Compute contribution of each indicator to the nowcast revision.

        Contribution_i = weight_i * news_i

        Parameters
        ----------
        news : dict[str, float]
            News for each indicator.
        weights : dict[str, float]
            Weight of each indicator.

        Returns
        -------
        dict[str, float]
            Contribution of each indicator.
        """
        contributions: dict[str, float] = {}

        for var in news:
            w = weights.get(var, 0.0)
            contributions[var] = w * news[var]

        return contributions

    def decompose(
        self,
        old_data: pd.DataFrame | dict[str, pd.Series],
        new_data: pd.DataFrame | dict[str, pd.Series],
        target: str | None = None,
    ) -> NewsResult:
        """Decompose the nowcast revision into individual contributions.

        Parameters
        ----------
        old_data : pd.DataFrame or dict[str, pd.Series]
            Data available in the old vintage (before new releases).
        new_data : pd.DataFrame or dict[str, pd.Series]
            Data available in the new vintage (after new releases).
        target : str or None
            Target variable for the nowcast. If None, uses the first
            quarterly variable.

        Returns
        -------
        NewsResult
            Decomposition result with total_revision, news, weights,
            and contributions.
        """
        if isinstance(old_data, dict):
            old_data = pd.DataFrame(old_data)
        if isinstance(new_data, dict):
            new_data = pd.DataFrame(new_data)

        if target is None:
            if self.nowcaster._quarterly_vars:
                target = self.nowcaster._quarterly_vars[0]
            else:
                target = self.nowcaster._var_names[0]

        # Save current state
        original_data = self.nowcaster._data
        original_result = self.nowcaster._kalman_result
        original_factors = self.nowcaster._factors_df

        # Nowcast with old data
        self.nowcaster._data = old_data.copy()
        y_old, z_old, t_old, q_old, r_old, mask_old = (
            self.nowcaster._build_state_space(old_data)
        )
        state_dim = self.nowcaster._state_dim
        a0 = np.zeros(state_dim)
        p0 = np.eye(state_dim) * 10.0

        result_old = self.nowcaster._adapter.filter(
            y_old, z_old, t_old, q_old, r_old, a0, p0, mask_old
        )
        self.nowcaster._kalman_result = result_old
        fc_old = self.nowcaster.nowcast(target=target)
        old_nowcast = float(fc_old.point[0])

        # Nowcast with new data
        self.nowcaster._data = new_data.copy()
        y_new, z_new, t_new, q_new, r_new, mask_new = (
            self.nowcaster._build_state_space(new_data)
        )

        result_new = self.nowcaster._adapter.filter(
            y_new, z_new, t_new, q_new, r_new, a0, p0, mask_new
        )
        self.nowcaster._kalman_result = result_new
        fc_new = self.nowcaster.nowcast(target=target)
        new_nowcast = float(fc_new.point[0])

        total_revision = new_nowcast - old_nowcast

        # Compute news, weights, contributions using old kalman result
        self.nowcaster._data = old_data.copy()
        self.nowcaster._kalman_result = result_old
        news = self._compute_news(old_data, new_data)
        weights = self._compute_weights(target)
        contributions = self._compute_contributions(news, weights)

        # Scale contributions to match total revision exactly
        contrib_sum = sum(contributions.values())
        if abs(contrib_sum) > 1e-10 and abs(total_revision) > 1e-10:
            scale = total_revision / contrib_sum
            contributions = {k: v * scale for k, v in contributions.items()}
            weights = {k: v * scale for k, v in weights.items()}

        # Build released/expected data
        released_data: dict[str, float] = {}
        expected_data: dict[str, float] = {}
        for var in news:
            new_series = new_data[var].dropna()
            if len(new_series) > 0:
                released_data[var] = float(new_series.iloc[-1])
            expected_data[var] = released_data.get(var, 0.0) - news[var]

        # Restore original state
        self.nowcaster._data = original_data
        self.nowcaster._kalman_result = original_result
        self.nowcaster._factors_df = original_factors

        return NewsResult(
            total_revision=total_revision,
            old_nowcast=old_nowcast,
            new_nowcast=new_nowcast,
            news=news,
            weights=weights,
            contributions=contributions,
            released_data=released_data,
            expected_data=expected_data,
        )

    def __repr__(self) -> str:
        return f"NewsDecomposition(nowcaster={self.nowcaster!r})"
