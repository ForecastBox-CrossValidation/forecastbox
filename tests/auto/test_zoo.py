"""Tests for ModelZoo registry."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from forecastbox.auto.zoo import ModelEntry, ModelZoo


class DummyModel:
    """A dummy model for testing."""

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1), **kwargs: Any) -> None:
        self.order = order
        self.kwargs = kwargs
        self._fitted = False

    def fit(self, y: Any, **kwargs: Any) -> DummyModel:
        self._fitted = True
        return self

    def forecast(self, h: int, **kwargs: Any) -> Any:
        return np.zeros(h)


class AnotherDummyModel:
    """Another dummy model for testing."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, y: Any, **kwargs: Any) -> AnotherDummyModel:
        return self

    def forecast(self, h: int, **kwargs: Any) -> Any:
        return np.ones(h)


@pytest.fixture(autouse=True)
def reset_zoo() -> None:
    """Reset ModelZoo singleton before each test."""
    ModelZoo.reset()


class TestModelZoo:
    """Tests for ModelZoo registry."""

    def test_register_get(self) -> None:
        """Register a model and retrieve it by name."""
        zoo = ModelZoo()
        zoo.register(
            name="test_arima",
            model_class=DummyModel,
            family="arima",
            default_params={"order": (1, 1, 1)},
            description="Test ARIMA model",
        )

        entry = zoo.get("test_arima")
        assert isinstance(entry, ModelEntry)
        assert entry.name == "test_arima"
        assert entry.model_class is DummyModel
        assert entry.family == "arima"
        assert entry.default_params == {"order": (1, 1, 1)}
        assert entry.description == "Test ARIMA model"

    def test_get_not_found(self) -> None:
        """Getting a non-existent model raises KeyError."""
        zoo = ModelZoo()
        with pytest.raises(KeyError, match="not found"):
            zoo.get("nonexistent")

    def test_list_models(self) -> None:
        """list_models() returns all registered model names."""
        zoo = ModelZoo()
        zoo.clear()
        zoo.register("model_a", DummyModel, family="arima")
        zoo.register("model_b", AnotherDummyModel, family="ets")
        zoo.register("model_c", DummyModel, family="arima")

        models = zoo.list_models()
        assert "model_a" in models
        assert "model_b" in models
        assert "model_c" in models
        assert models == sorted(models)  # Should be sorted

    def test_list_by_family(self) -> None:
        """list_models(family='arima') filters correctly."""
        zoo = ModelZoo()
        zoo.clear()
        zoo.register("arima_1", DummyModel, family="arima")
        zoo.register("arima_2", DummyModel, family="arima")
        zoo.register("ets_1", AnotherDummyModel, family="ets")
        zoo.register("var_1", DummyModel, family="var")

        arima_models = zoo.list_models(family="arima")
        assert arima_models == ["arima_1", "arima_2"]

        ets_models = zoo.list_models(family="ets")
        assert ets_models == ["ets_1"]

        var_models = zoo.list_models(family="var")
        assert var_models == ["var_1"]

        custom_models = zoo.list_models(family="custom")
        assert custom_models == []

    def test_create_instance(self) -> None:
        """create('model', order=(2,1,1)) returns a correctly configured instance."""
        zoo = ModelZoo()
        zoo.register(
            "test_arima",
            DummyModel,
            family="arima",
            default_params={"order": (1, 1, 1)},
        )

        # Create with defaults
        model = zoo.create("test_arima")
        assert isinstance(model, DummyModel)
        assert model.order == (1, 1, 1)

        # Create with overrides
        model2 = zoo.create("test_arima", order=(2, 1, 1))
        assert isinstance(model2, DummyModel)
        assert model2.order == (2, 1, 1)

    def test_create_not_found(self) -> None:
        """Creating a non-existent model raises KeyError."""
        zoo = ModelZoo()
        with pytest.raises(KeyError):
            zoo.create("nonexistent")

    def test_duplicate_warning(self) -> None:
        """Registering the same name twice emits a UserWarning."""
        zoo = ModelZoo()
        zoo.register("dup_model", DummyModel, family="arima")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            zoo.register("dup_model", AnotherDummyModel, family="ets")
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "already registered" in str(w[0].message)

        # The second registration should overwrite
        entry = zoo.get("dup_model")
        assert entry.model_class is AnotherDummyModel
        assert entry.family == "ets"

    def test_unregister(self) -> None:
        """Unregister removes a model from the registry."""
        zoo = ModelZoo()
        zoo.register("temp_model", DummyModel)
        assert "temp_model" in zoo.list_models()

        zoo.unregister("temp_model")
        assert "temp_model" not in zoo.list_models()

    def test_unregister_not_found(self) -> None:
        """Unregistering a non-existent model raises KeyError."""
        zoo = ModelZoo()
        with pytest.raises(KeyError):
            zoo.unregister("nonexistent")

    def test_singleton_pattern(self) -> None:
        """ModelZoo is a singleton — two instances share the same registry."""
        zoo1 = ModelZoo()
        zoo1.register("singleton_test", DummyModel)

        zoo2 = ModelZoo()
        assert "singleton_test" in zoo2.list_models()

    def test_clear(self) -> None:
        """clear() removes all models from the registry."""
        zoo = ModelZoo()
        zoo.register("a", DummyModel)
        zoo.register("b", AnotherDummyModel)
        assert len(zoo.list_models()) >= 2

        zoo.clear()
        assert len(zoo.list_models()) == 0

    def test_repr(self) -> None:
        """__repr__ shows model count and families."""
        zoo = ModelZoo()
        zoo.clear()
        zoo.register("m1", DummyModel, family="arima")
        zoo.register("m2", AnotherDummyModel, family="ets")
        repr_str = repr(zoo)
        assert "2 models" in repr_str
        assert "arima" in repr_str
        assert "ets" in repr_str
