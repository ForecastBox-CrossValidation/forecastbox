"""Forecast evaluation tests and comparison tools.

Includes statistical tests for comparing forecast accuracy (Diebold-Mariano,
Model Confidence Set, Giacomini-White), calibration tests (Mincer-Zarnowitz),
and forecast encompassing tests (HLN 1998).
"""

from forecastbox.evaluation.diebold_mariano import DMResult, diebold_mariano
from forecastbox.evaluation.encompassing import EncompassingResult, encompassing_test
from forecastbox.evaluation.giacomini_white import GWResult, giacomini_white
from forecastbox.evaluation.mcs import MCSResult, model_confidence_set
from forecastbox.evaluation.mincer_zarnowitz import MZResult, mincer_zarnowitz

__all__ = [
    "DMResult",
    "EncompassingResult",
    "GWResult",
    "MCSResult",
    "MZResult",
    "diebold_mariano",
    "encompassing_test",
    "giacomini_white",
    "mincer_zarnowitz",
    "model_confidence_set",
]
