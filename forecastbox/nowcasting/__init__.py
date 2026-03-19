"""Nowcasting toolkit: DFM, bridge equations, MIDAS, news decomposition.

This module provides tools for nowcasting (predicting the present) using:
- DFMNowcaster: Dynamic Factor Model with mixed frequencies
- BridgeEquation: Quarterly prediction from monthly indicators
- MIDAS: Mixed Data Sampling regression (Ghysels et al 2004)
- NewsDecomposition: Revision decomposition (Banbura-Modugno 2014)
- RealTimeDataManager: Real-time data management with publication calendars

References
----------
- Giannone, Reichlin & Small (2008). "Nowcasting: The real-time informational
  content of macroeconomic data." Journal of Monetary Economics, 55(4), 665-676.
- Banbura & Modugno (2014). "Maximum likelihood estimation of factor models
  on datasets with arbitrary pattern of missing data." JAE, 29(1), 133-160.
- Ghysels, Santa-Clara & Valkanov (2004). "The MIDAS Touch." CIRANO.
- Mariano & Murasawa (2003). "A new coincident index of business cycles."
  Journal of Applied Econometrics, 18(4), 427-443.
"""

from forecastbox.nowcasting.bridge import BridgeEquation
from forecastbox.nowcasting.dfm import DFMNowcaster
from forecastbox.nowcasting.midas import MIDAS
from forecastbox.nowcasting.news import NewsDecomposition, NewsResult
from forecastbox.nowcasting.realtime import RealTimeDataManager, SeriesInfo

__all__ = [
    "BridgeEquation",
    "DFMNowcaster",
    "MIDAS",
    "NewsDecomposition",
    "NewsResult",
    "RealTimeDataManager",
    "SeriesInfo",
]
