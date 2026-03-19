"""Forecast combination methods.

This module provides multiple strategies for combining individual forecasts
into a single combined forecast, including simple averaging, weighted methods,
OLS regression, stacking, Bayesian Model Averaging, time-varying weights,
and optimal (Bates-Granger) combination.
"""

from forecastbox.combination.base import BaseCombiner
from forecastbox.combination.bma import BMACombiner
from forecastbox.combination.ols import OLSCombiner
from forecastbox.combination.optimal import OptimalCombiner
from forecastbox.combination.simple import SimpleCombiner
from forecastbox.combination.stacking import StackingCombiner
from forecastbox.combination.time_varying import TimeVaryingCombiner
from forecastbox.combination.weighted import WeightedCombiner

__all__ = [
    "BaseCombiner",
    "BMACombiner",
    "OLSCombiner",
    "OptimalCombiner",
    "SimpleCombiner",
    "StackingCombiner",
    "TimeVaryingCombiner",
    "WeightedCombiner",
]
