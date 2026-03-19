"""Type aliases for forecastbox."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | list[float]
ModelFn = Callable[[pd.Series], Any]
MetricFn = Callable[[np.ndarray, np.ndarray], float]
