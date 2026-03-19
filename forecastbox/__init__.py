"""forecastbox - Forecast containers, evaluation metrics, and cross-validation for time series."""

from forecastbox.__version__ import __version__
from forecastbox.core.forecast import Forecast
from forecastbox.experiment import ExperimentResults, ForecastExperiment

__all__ = [
    "__version__",
    "Forecast",
    "ForecastExperiment",
    "ExperimentResults",
]
