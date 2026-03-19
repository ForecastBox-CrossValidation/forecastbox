"""Auto-forecast module: automatic model selection for time series.

Provides AutoARIMA, AutoETS, AutoVAR, AutoSelect, and ModelZoo for
automatic selection of the best forecasting model.

Usage
-----
>>> from forecastbox.auto import AutoARIMA, AutoETS, AutoSelect, AutoVAR, ModelZoo
>>> auto = AutoARIMA(seasonal=True, m=12, stepwise=True)
>>> result = auto.fit(data)
>>> forecast = result.forecast(12)
"""

from forecastbox.auto.arima import AutoARIMA, AutoARIMAResult
from forecastbox.auto.ets import AutoETS, AutoETSResult
from forecastbox.auto.select import AutoSelect, AutoSelectResult
from forecastbox.auto.var import AutoVAR, AutoVARResult
from forecastbox.auto.zoo import ModelZoo

__all__ = [
    "AutoARIMA",
    "AutoARIMAResult",
    "AutoETS",
    "AutoETSResult",
    "AutoSelect",
    "AutoSelectResult",
    "AutoVAR",
    "AutoVARResult",
    "ModelZoo",
]
