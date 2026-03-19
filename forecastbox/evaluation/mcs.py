"""Model Confidence Set (Hansen, Lunde & Nason 2011).

Determines the set of models that contains the best model with
controlled probability (1-alpha), using sequential elimination
with stationary bootstrap for p-values.

References
----------
Hansen, P.R., Lunde, A. & Nason, J.M. (2011). "The Model Confidence Set."
    Econometrica, 79(2), 453-497.
Politis, D.N. & Romano, J.P. (1994). "The Stationary Bootstrap."
    Journal of the American Statistical Association, 89(428), 1303-1313.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from forecastbox.evaluation._hac import hac_variance


@dataclass
class MCSResult:
    """Result of a Model Confidence Set procedure.

    Attributes
    ----------
    included_models : list[str]
        Models in the MCS (not eliminated).
    excluded_models : list[str]
        Models eliminated from the MCS.
    pvalues : dict[str, float]
        MCS p-value for each model. Higher p-value = more likely to be
        in the set of best models.
    elimination_order : list[str]
        Order in which models were eliminated (first = worst).
    alpha : float
        Significance level used.
    """

    included_models: list[str]
    excluded_models: list[str]
    pvalues: dict[str, float]
    elimination_order: list[str]
    alpha: float

    def summary(self) -> str:
        """Return formatted summary table of MCS results.

        Returns
        -------
        str
            Formatted string with model rankings and p-values.
        """
        lines = [
            f"Model Confidence Set (alpha={self.alpha})",
            "=" * 50,
            f"{'Model':<25} {'MCS p-value':>12} {'Status':>10}",
            "-" * 50,
        ]

        # Sort by p-value descending
        sorted_models = sorted(
            self.pvalues.items(), key=lambda x: x[1], reverse=True
        )
        for model_name, pval in sorted_models:
            status = "Included" if model_name in self.included_models else "Excluded"
            lines.append(f"{model_name:<25} {pval:>12.4f} {status:>10}")

        lines.append("-" * 50)
        lines.append(f"MCS contains {len(self.included_models)} model(s)")
        return "\n".join(lines)


def _stationary_bootstrap_indices(
    n_obs: int,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate bootstrap indices using the stationary bootstrap.

    Parameters
    ----------
    n_obs : int
        Sample size.
    block_length : int
        Expected block length (geometric distribution parameter).
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    NDArray[np.int64]
        Bootstrap indices of length n_obs.
    """
    prob_new_block = 1.0 / block_length
    indices = np.empty(n_obs, dtype=np.int64)

    # Start with random position
    indices[0] = rng.integers(0, n_obs)

    for t in range(1, n_obs):
        if rng.random() < prob_new_block:
            # Start a new block
            indices[t] = rng.integers(0, n_obs)
        else:
            # Continue current block (wrap around)
            indices[t] = (indices[t - 1] + 1) % n_obs

    return indices


def _compute_range_statistic(
    t_stats: NDArray[np.float64],
) -> float:
    """Compute the range statistic T_R = max|t_{ij}|.

    Parameters
    ----------
    t_stats : NDArray[np.float64]
        Matrix of t-statistics, shape (k, k).

    Returns
    -------
    float
        Range statistic.
    """
    k = t_stats.shape[0]
    max_val = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            val = abs(t_stats[i, j])
            if val > max_val:
                max_val = val
    return max_val


def _compute_sq_statistic(
    d_bar_matrix: NDArray[np.float64],
    var_matrix: NDArray[np.float64],
) -> float:
    """Compute the semi-quadratic statistic T_SQ.

    T_SQ = sum_i ( (1/K) * sum_j d_bar_{ij} )^2 / var(.)

    Parameters
    ----------
    d_bar_matrix : NDArray[np.float64]
        Matrix of mean loss differentials, shape (k, k).
    var_matrix : NDArray[np.float64]
        Matrix of HAC variances for each d_{ij}, shape (k, k).

    Returns
    -------
    float
        Semi-quadratic statistic.
    """
    k = d_bar_matrix.shape[0]
    stat = 0.0
    for i in range(k):
        d_bar_i = float(np.mean(d_bar_matrix[i, :]))
        # Variance of the average: approximate with average of variances
        var_i = float(np.mean(var_matrix[i, :])) / k if k > 0 else 1.0
        if var_i > 0:
            stat += d_bar_i**2 / var_i
    return stat


def model_confidence_set(
    actual: NDArray[np.float64] | list[float],
    forecasts: Mapping[str, NDArray[np.float64] | list[float]],
    alpha: float = 0.10,
    loss: str = "mse",
    statistic: str = "range",
    n_boot: int = 5000,
    block_length: int | None = None,
    seed: int | None = None,
) -> MCSResult:
    """Compute the Model Confidence Set.

    Determines the smallest set of models that contains the best model
    with probability >= 1 - alpha.

    Parameters
    ----------
    actual : array-like
        Actual observed values of length T.
    forecasts : Mapping[str, array-like]
        Dictionary mapping model names to forecast arrays of length T.
        Must contain at least 2 models.
    alpha : float
        Significance level. Default: 0.10.
    loss : str
        Loss function: 'mse' or 'mae'. Default: 'mse'.
    statistic : str
        Test statistic: 'range' or 'semi_quadratic'. Default: 'range'.
    n_boot : int
        Number of bootstrap replications. Default: 5000.
    block_length : int or None
        Expected block length for stationary bootstrap.
        If None, uses max(1, floor(T^{1/3})).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    MCSResult
        MCS result with included models, p-values, and elimination order.

    Raises
    ------
    ValueError
        If fewer than 2 models provided or arrays have different lengths.
    """
    actual = np.asarray(actual, dtype=np.float64)
    n_obs = len(actual)

    if len(forecasts) < 2:
        msg = "Need at least 2 models for MCS"
        raise ValueError(msg)

    model_names = list(forecasts.keys())
    fc_arrays: dict[str, NDArray[np.float64]] = {}
    for name, fc in forecasts.items():
        fc_arr = np.asarray(fc, dtype=np.float64)
        if len(fc_arr) != n_obs:
            msg = f"Forecast '{name}' has length {len(fc_arr)}, expected {n_obs}"
            raise ValueError(msg)
        fc_arrays[name] = fc_arr

    if block_length is None:
        block_length = max(1, int(math.floor(n_obs ** (1.0 / 3.0))))

    rng = np.random.default_rng(seed)

    # Compute losses
    losses: dict[str, NDArray[np.float64]] = {}
    for name, fc in fc_arrays.items():
        e = actual - fc
        if loss == "mse":
            losses[name] = e**2
        elif loss == "mae":
            losses[name] = np.abs(e)
        else:
            msg = f"Unknown loss: '{loss}'. Use 'mse' or 'mae'."
            raise ValueError(msg)

    # Initialize
    remaining = list(model_names)
    pvalues: dict[str, float] = {}
    elimination_order: list[str] = []
    prev_pval = 0.0

    while len(remaining) > 1:
        k = len(remaining)

        # Compute pairwise loss differentials
        d_series: dict[tuple[int, int], NDArray[np.float64]] = {}
        d_bar_matrix = np.zeros((k, k))
        var_matrix = np.zeros((k, k))
        t_stat_matrix = np.zeros((k, k))

        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                d_ij = losses[remaining[i]] - losses[remaining[j]]
                d_series[(i, j)] = d_ij
                d_bar_ij = float(np.mean(d_ij))
                d_bar_matrix[i, j] = d_bar_ij

                v_ij = hac_variance(d_ij)
                var_matrix[i, j] = max(v_ij / n_obs, 1e-20)
                t_stat_matrix[i, j] = d_bar_ij / math.sqrt(
                    max(v_ij / n_obs, 1e-20)
                )

        # Observed statistic
        if statistic == "range":
            t_obs = _compute_range_statistic(t_stat_matrix)
        elif statistic == "semi_quadratic":
            t_obs = _compute_sq_statistic(d_bar_matrix, var_matrix)
        else:
            msg = f"Unknown statistic: '{statistic}'. Use 'range' or 'semi_quadratic'."
            raise ValueError(msg)

        # Bootstrap p-value
        count_geq = 0
        for _ in range(n_boot):
            boot_idx = _stationary_bootstrap_indices(n_obs, block_length, rng)

            # Compute bootstrap statistics under H0 (centered)
            t_boot = np.zeros((k, k))
            d_bar_boot = np.zeros((k, k))
            var_boot = np.zeros((k, k))

            for i in range(k):
                for j in range(k):
                    if i == j:
                        continue
                    d_ij = d_series[(i, j)]
                    d_boot = d_ij[boot_idx]
                    # Center under H0
                    d_boot_centered = d_boot - np.mean(d_ij)
                    d_bar_b = float(np.mean(d_boot_centered))
                    d_bar_boot[i, j] = d_bar_b

                    v_b = float(np.var(d_boot_centered, ddof=1)) / n_obs
                    var_boot[i, j] = max(v_b, 1e-20)
                    t_boot[i, j] = d_bar_b / math.sqrt(max(v_b, 1e-20))

            if statistic == "range":
                t_boot_stat = _compute_range_statistic(t_boot)
            else:
                t_boot_stat = _compute_sq_statistic(d_bar_boot, var_boot)

            if t_boot_stat >= t_obs:
                count_geq += 1

        p_val = count_geq / n_boot

        if p_val >= alpha:
            # Accept: all remaining models are in MCS
            for name in remaining:
                pvalues[name] = max(p_val, prev_pval)
            break

        # Reject: eliminate worst model
        # Worst model: highest average relative loss
        worst_idx = -1
        worst_score = -np.inf
        for i in range(k):
            score = 0.0
            for j in range(k):
                if i == j:
                    continue
                score += t_stat_matrix[i, j]
            if score > worst_score:
                worst_score = score
                worst_idx = i

        worst_name = remaining[worst_idx]
        pvalues[worst_name] = max(p_val, prev_pval)
        prev_pval = pvalues[worst_name]
        elimination_order.append(worst_name)
        remaining.pop(worst_idx)

    # If only one model left and loop ended without break
    if len(remaining) == 1 and remaining[0] not in pvalues:
        pvalues[remaining[0]] = 1.0

    included = [m for m in model_names if m in remaining]
    excluded = list(elimination_order)

    return MCSResult(
        included_models=included,
        excluded_models=excluded,
        pvalues=pvalues,
        elimination_order=elimination_order,
        alpha=alpha,
    )
