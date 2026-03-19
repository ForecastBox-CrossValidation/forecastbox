"""ModelZoo - Registry of models with plugin pattern.

ModelZoo is a singleton registry that allows registering, listing, and
instantiating forecasting models. It supports both built-in models
(via chronobox adapters) and custom user models.

Usage:
    from forecastbox.auto import ModelZoo

    zoo = ModelZoo()
    zoo.register('my_model', MyModelClass, default_params={'order': (1,1,1)})
    model = zoo.create('my_model', order=(2,1,1))
    print(zoo.list_models())
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from forecastbox._logging import get_logger

logger = get_logger("auto.zoo")


@runtime_checkable
class ForecastModel(Protocol):
    """Protocol that all forecastbox-compatible models must satisfy."""

    def fit(self, y: Any, **kwargs: Any) -> Any:
        """Fit the model to data."""
        ...

    def forecast(self, h: int, **kwargs: Any) -> Any:
        """Generate forecasts for h steps ahead."""
        ...


@dataclass
class ModelEntry:
    """Entry in the ModelZoo registry.

    Attributes
    ----------
    name : str
        Unique name for the model.
    model_class : type
        The model class to instantiate.
    default_params : dict[str, Any]
        Default parameters for model instantiation.
    family : str
        Model family (e.g., 'arima', 'ets', 'var', 'garch', 'baseline', 'custom').
    description : str
        Human-readable description of the model.
    """

    name: str
    model_class: type
    default_params: dict[str, Any] = field(default_factory=dict)
    family: str = "custom"
    description: str = ""


class ModelZoo:
    """Singleton registry of forecasting models.

    ModelZoo maintains a global registry of model classes that can be
    instantiated by name. It supports plugin pattern for extensibility.

    Examples
    --------
    >>> zoo = ModelZoo()
    >>> zoo.register('my_arima', MyARIMA, family='arima', default_params={'order': (1,1,1)})
    >>> zoo.list_models()
    ['my_arima']
    >>> zoo.list_models(family='arima')
    ['my_arima']
    >>> model = zoo.create('my_arima', order=(2,1,1))
    """

    _instance: ModelZoo | None = None
    _registry: dict[str, ModelEntry] = {}
    _initialized: bool = False

    def __new__(cls) -> ModelZoo:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._registry = {}
            cls._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the ModelZoo and register built-in models."""
        if not self._initialized:
            self._register_builtins()
            ModelZoo._initialized = True

    def _register_builtins(self) -> None:
        """Register built-in model adapters.

        Attempts to register chronobox models if available.
        Always registers forecastbox baseline adapters.
        """
        # Register chronobox adapters if available
        try:
            from forecastbox.auto._adapters import (
                ARIMAAdapter,
                ETSAdapter,
                ThetaAdapter,
                VARAdapter,
            )

            self.register(
                name="arima",
                model_class=ARIMAAdapter,
                family="arima",
                default_params={"order": (1, 1, 1)},
                description="ARIMA model via chronobox adapter",
            )
            self.register(
                name="ets",
                model_class=ETSAdapter,
                family="ets",
                default_params={},
                description="ETS model via chronobox adapter",
            )
            self.register(
                name="var",
                model_class=VARAdapter,
                family="var",
                default_params={"maxlags": 12},
                description="VAR model via chronobox adapter",
            )
            self.register(
                name="theta",
                model_class=ThetaAdapter,
                family="theta",
                default_params={},
                description="Theta method via chronobox adapter",
            )
            logger.info("Registered chronobox model adapters")
        except ImportError:
            logger.info("chronobox not available; built-in model adapters not registered")

        # Register baseline models (always available)
        try:
            from forecastbox.auto._baselines import (
                DriftBaseline,
                NaiveBaseline,
                SeasonalNaiveBaseline,
            )

            self.register(
                name="naive",
                model_class=NaiveBaseline,
                family="baseline",
                default_params={},
                description="Naive forecast (last value repeated)",
            )
            self.register(
                name="snaive",
                model_class=SeasonalNaiveBaseline,
                family="baseline",
                default_params={"seasonal_period": 12},
                description="Seasonal naive forecast",
            )
            self.register(
                name="drift",
                model_class=DriftBaseline,
                family="baseline",
                default_params={},
                description="Random walk with drift",
            )
            logger.info("Registered baseline models")
        except ImportError:
            logger.info("Baseline models not yet available")

    def register(
        self,
        name: str,
        model_class: type,
        default_params: dict[str, Any] | None = None,
        family: str = "custom",
        description: str = "",
    ) -> None:
        """Register a model in the zoo.

        Parameters
        ----------
        name : str
            Unique name for the model.
        model_class : type
            The model class.
        default_params : dict[str, Any] or None
            Default parameters for instantiation.
        family : str
            Model family (e.g., 'arima', 'ets', 'var', 'baseline', 'custom').
        description : str
            Human-readable description.

        Warns
        -----
        UserWarning
            If a model with the same name is already registered (it will be overwritten).
        """
        if name in self._registry:
            warnings.warn(
                f"Model '{name}' is already registered in ModelZoo. "
                f"Overwriting with new registration.",
                UserWarning,
                stacklevel=2,
            )

        entry = ModelEntry(
            name=name,
            model_class=model_class,
            default_params=default_params or {},
            family=family,
            description=description,
        )
        self._registry[name] = entry
        logger.debug("Registered model '%s' (family=%s)", name, family)

    def get(self, name: str) -> ModelEntry:
        """Retrieve a model entry by name.

        Parameters
        ----------
        name : str
            The registered model name.

        Returns
        -------
        ModelEntry
            The model entry with class, params, and metadata.

        Raises
        ------
        KeyError
            If the model name is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            msg = f"Model '{name}' not found in ModelZoo. Available: {available}"
            raise KeyError(msg)
        return self._registry[name]

    def list_models(self, family: str | None = None) -> list[str]:
        """List registered model names.

        Parameters
        ----------
        family : str or None
            If provided, filter by model family.

        Returns
        -------
        list[str]
            Sorted list of model names.
        """
        if family is not None:
            return sorted(
                name for name, entry in self._registry.items() if entry.family == family
            )
        return sorted(self._registry.keys())

    def create(self, name: str, **kwargs: Any) -> Any:
        """Create a model instance by name.

        Parameters
        ----------
        name : str
            The registered model name.
        **kwargs : Any
            Parameters to override defaults.

        Returns
        -------
        Any
            An instance of the model class.

        Raises
        ------
        KeyError
            If the model name is not registered.
        """
        entry = self.get(name)
        params = {**entry.default_params, **kwargs}
        return entry.model_class(**params)

    def unregister(self, name: str) -> None:
        """Remove a model from the registry.

        Parameters
        ----------
        name : str
            The model name to remove.

        Raises
        ------
        KeyError
            If the model name is not registered.
        """
        if name not in self._registry:
            msg = f"Model '{name}' not found in ModelZoo."
            raise KeyError(msg)
        del self._registry[name]

    def clear(self) -> None:
        """Remove all registered models."""
        self._registry.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None
        cls._registry = {}
        cls._initialized = False

    def __repr__(self) -> str:
        n = len(self._registry)
        families = sorted(set(e.family for e in self._registry.values()))
        return f"ModelZoo({n} models, families={families})"
