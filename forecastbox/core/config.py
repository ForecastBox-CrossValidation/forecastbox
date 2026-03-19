"""Global configuration for forecastbox."""

from dataclasses import dataclass


@dataclass
class ForecastBoxConfig:
    """Global configuration."""

    default_confidence_levels: tuple[float, ...] = (0.80, 0.95)
    default_point_method: str = "median"
    plot_style: str = "seaborn-v0_8"
    plot_figsize: tuple[int, int] = (12, 6)
    n_density_draws: int = 1000
    float_precision: int = 4


# Singleton global config
config = ForecastBoxConfig()
