"""Bridge equation nowcasting: quarterly prediction from monthly indicators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class BridgeEquation:
    """Bridge equation for nowcasting a quarterly variable from monthly indicators.

    Aggregates monthly indicators to quarterly frequency and estimates a linear
    regression. Handles partial quarters by projecting missing months.

    Parameters
    ----------
    target : str
        Name of the quarterly target variable.
    indicators : list[str]
        Names of monthly indicator variables.
    aggregation : str
        Aggregation method: 'mean', 'sum', 'last', 'first'.
    fill_method : str
        Method for projecting missing months: 'ar1', 'last', 'mean'.
    include_lags : bool
        Whether to include lagged target as a regressor.

    Examples
    --------
    >>> bridge = BridgeEquation(
    ...     target='pib_quarterly',
    ...     indicators=['producao_industrial', 'vendas_varejo', 'pmi'],
    ...     aggregation='mean'
    ... )
    >>> bridge.fit(data)
    >>> nowcast = bridge.nowcast()
    """

    def __init__(
        self,
        target: str,
        indicators: list[str],
        aggregation: str = "mean",
        fill_method: str = "ar1",
        include_lags: bool = False,
    ) -> None:
        valid_agg = {"mean", "sum", "last", "first"}
        if aggregation not in valid_agg:
            msg = f"aggregation must be one of {valid_agg}, got '{aggregation}'"
            raise ValueError(msg)

        valid_fill = {"ar1", "last", "mean"}
        if fill_method not in valid_fill:
            msg = f"fill_method must be one of {valid_fill}, got '{fill_method}'"
            raise ValueError(msg)

        self.target = target
        self.indicators = list(indicators)
        self.aggregation = aggregation
        self.fill_method = fill_method
        self.include_lags = include_lags

        # Fitted attributes
        self._fitted = False
        self._beta: NDArray[np.float64] | None = None
        self._intercept: float = 0.0
        self._sigma2: float = 0.0
        self._XtX_inv: NDArray[np.float64] | None = None
        self._r_squared_val: float = 0.0
        self._n_obs: int = 0
        self._target_data: pd.Series | None = None  # type: ignore[type-arg]
        self._indicator_data: pd.DataFrame | None = None
        self._data: pd.DataFrame | None = None

    def _aggregate(self, monthly_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate monthly indicators to quarterly frequency.

        Parameters
        ----------
        monthly_data : pd.DataFrame
            Monthly indicator data with DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Quarterly aggregated data.
        """
        if self.aggregation == "mean":
            return monthly_data.resample("QS").mean()
        elif self.aggregation == "sum":
            return monthly_data.resample("QS").sum()
        elif self.aggregation == "last":
            return monthly_data.resample("QS").last()
        elif self.aggregation == "first":
            return monthly_data.resample("QS").first()
        else:
            msg = f"Unknown aggregation method: {self.aggregation}"
            raise ValueError(msg)

    def _fill_missing_months(
        self, data: pd.DataFrame, method: str | None = None
    ) -> pd.DataFrame:
        """Project missing months using the specified fill method.

        Parameters
        ----------
        data : pd.DataFrame
            Monthly data that may have trailing NaNs.
        method : str or None
            Fill method. If None, uses self.fill_method.

        Returns
        -------
        pd.DataFrame
            Data with missing trailing months filled.
        """
        if method is None:
            method = self.fill_method

        result = data.copy()

        for col in result.columns:
            series = result[col]
            valid = series.dropna()

            if len(valid) == 0:
                continue

            if method == "ar1":
                # Fit AR(1) on available data
                if len(valid) >= 3:
                    y = valid.values[1:]
                    x = valid.values[:-1]
                    x_with_const = np.column_stack([np.ones(len(x)), x])
                    try:
                        params = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        params = np.array([0.0, 0.9])
                    c, phi = params[0], params[1]
                    phi = np.clip(phi, -0.99, 0.99)
                else:
                    c = 0.0
                    phi = 0.9

                last_val = valid.values[-1]
                for idx in series.index:
                    if pd.isna(series[idx]) and idx > valid.index[-1]:
                        new_val = c + phi * last_val
                        result.loc[idx, col] = new_val
                        last_val = new_val

            elif method == "last":
                last_val = valid.values[-1]
                for idx in series.index:
                    if pd.isna(series[idx]) and idx > valid.index[-1]:
                        result.loc[idx, col] = last_val

            elif method == "mean":
                mean_val = valid.mean()
                result[col] = result[col].fillna(mean_val)

        return result

    def fit(self, data: pd.DataFrame | dict[str, pd.Series]) -> BridgeEquation:  # type: ignore[type-arg]
        """Estimate the bridge equation via OLS.

        Parameters
        ----------
        data : pd.DataFrame or dict[str, pd.Series]
            Panel data containing the target (quarterly) and indicators (monthly).

        Returns
        -------
        BridgeEquation
            Self, for method chaining.
        """
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        self._data = data.copy()

        if self.target not in data.columns:
            msg = f"Target '{self.target}' not found in data columns"
            raise ValueError(msg)

        target_series = data[self.target].dropna()

        missing_indicators = [ind for ind in self.indicators if ind not in data.columns]
        if missing_indicators:
            msg = f"Indicators not found in data: {missing_indicators}"
            raise ValueError(msg)

        monthly_indicators = data[self.indicators].copy()

        # Aggregate monthly indicators to quarterly
        quarterly_indicators = self._aggregate(monthly_indicators)

        # Align target and indicators
        common_idx = target_series.index.intersection(quarterly_indicators.index)

        if len(common_idx) < 3:
            msg = f"Need at least 3 common quarters, got {len(common_idx)}"
            raise ValueError(msg)

        y = target_series.loc[common_idx].values.astype(np.float64)
        x_mat = quarterly_indicators.loc[common_idx].values.astype(np.float64)

        # Drop rows with NaN
        valid_mask = ~np.any(np.isnan(x_mat), axis=1) & ~np.isnan(y)
        y = y[valid_mask]
        x_mat = x_mat[valid_mask]

        if len(y) < 3:
            msg = f"Need at least 3 valid observations, got {len(y)}"
            raise ValueError(msg)

        # Include lagged target if requested
        if self.include_lags:
            y_lag = np.roll(y, 1)
            y_lag[0] = y[0]
            x_mat = np.column_stack([x_mat, y_lag])

        # Add intercept
        n = len(y)
        x_with_const = np.column_stack([np.ones(n), x_mat])

        # OLS estimation: beta = (X'X)^{-1} X'y
        try:
            xtx = x_with_const.T @ x_with_const
            xtx_inv = np.linalg.inv(xtx)
            beta = xtx_inv @ x_with_const.T @ y
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(x_with_const.T @ x_with_const)
            beta = xtx_inv @ x_with_const.T @ y

        self._intercept = float(beta[0])
        self._beta = beta[1:].astype(np.float64)
        self._XtX_inv = xtx_inv.astype(np.float64)

        # Residual variance
        y_hat = x_with_const @ beta
        residuals = y - y_hat
        self._sigma2 = float(np.sum(residuals**2) / max(1, n - len(beta)))

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self._r_squared_val = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        self._n_obs = n
        self._target_data = target_series
        self._indicator_data = monthly_indicators

        self._fitted = True
        return self

    def nowcast(
        self,
        data: pd.DataFrame | None = None,
        reference_date: str | pd.Timestamp | None = None,
    ) -> Any:
        """Generate nowcast for the current/next quarter.

        Parameters
        ----------
        data : pd.DataFrame or None
            Updated data. If None, uses the data from fit().
        reference_date : str, Timestamp, or None
            Reference date. If None, uses latest available.

        Returns
        -------
        Forecast
            Nowcast as a Forecast object.
        """
        if not self._fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        from forecastbox.core.forecast import Forecast

        if data is not None:
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            monthly_indicators = data[self.indicators].copy()
        else:
            monthly_indicators = self._indicator_data

        if monthly_indicators is None:
            msg = "No indicator data available"
            raise RuntimeError(msg)

        # Fill missing months for partial quarter
        filled = self._fill_missing_months(monthly_indicators, self.fill_method)

        # Aggregate to quarterly
        quarterly = self._aggregate(filled)

        # Use the last quarter
        last_q = quarterly.iloc[[-1]].values.astype(np.float64)

        if self.include_lags and self._target_data is not None:
            last_target = (
                self._target_data.iloc[-1] if len(self._target_data) > 0 else 0.0
            )
            last_q = np.column_stack([last_q, [[last_target]]])

        # Predict
        assert self._beta is not None
        x_new = np.concatenate([[1.0], last_q.flatten()])
        point_est = float(x_new @ np.concatenate([[self._intercept], self._beta]))

        # Prediction variance
        if self._XtX_inv is not None:
            pred_var = self._sigma2 * (1 + x_new @ self._XtX_inv @ x_new)
        else:
            pred_var = self._sigma2

        std_est = np.sqrt(max(pred_var, 1e-10))

        point = np.array([point_est])
        lower_80 = np.array([point_est - 1.28 * std_est])
        upper_80 = np.array([point_est + 1.28 * std_est])
        lower_95 = np.array([point_est - 1.96 * std_est])
        upper_95 = np.array([point_est + 1.96 * std_est])

        return Forecast(
            point=point,
            lower_80=lower_80,
            upper_80=upper_80,
            lower_95=lower_95,
            upper_95=upper_95,
            model_name=f"Bridge({self.target})",
            horizon=1,
            metadata={
                "target": self.target,
                "indicators": self.indicators,
                "aggregation": self.aggregation,
                "fill_method": self.fill_method,
                "r_squared": self._r_squared_val,
            },
        )

    def coefficients(self) -> pd.DataFrame:
        """Return estimated coefficients as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['variable', 'coefficient'].
        """
        if not self._fitted or self._beta is None:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        names = ["intercept"] + list(self.indicators)
        if self.include_lags:
            names.append(f"{self.target}_lag1")

        coefs = np.concatenate([[self._intercept], self._beta])

        return pd.DataFrame({
            "variable": names,
            "coefficient": coefs,
        })

    def r_squared(self) -> float:
        """Return the R-squared of the bridge equation.

        Returns
        -------
        float
            R-squared value between 0 and 1.
        """
        if not self._fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)
        return self._r_squared_val

    def summary(self) -> str:
        """Return a text summary of the bridge equation.

        Returns
        -------
        str
            Formatted summary string.
        """
        if not self._fitted:
            msg = "Model not fitted. Call fit() first."
            raise RuntimeError(msg)

        coefs = self.coefficients()

        lines = [
            "=" * 60,
            f"Bridge Equation: {self.target}",
            "=" * 60,
            f"Aggregation: {self.aggregation}",
            f"Fill method: {self.fill_method}",
            f"Observations: {self._n_obs}",
            f"R-squared: {self._r_squared_val:.4f}",
            f"Residual std: {np.sqrt(self._sigma2):.4f}",
            "-" * 60,
            "Coefficients:",
        ]

        for _, row in coefs.iterrows():
            lines.append(f"  {row['variable']:30s}  {row['coefficient']:10.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"BridgeEquation(target='{self.target}', "
            f"indicators={self.indicators}, "
            f"aggregation='{self.aggregation}', {status})"
        )
