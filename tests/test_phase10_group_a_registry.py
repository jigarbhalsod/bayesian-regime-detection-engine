import pytest

from src.ensemble import (
    BaseEnsembleStrategy,
    EnsembleRegistry,
    EnsembleResult,
)


class DummyStrategy(BaseEnsembleStrategy):

    STRATEGY_NAME = "dummy"

    @property
    def strategy_name(self) -> str:
        return self.STRATEGY_NAME

    def combine(self, model_outputs):
        return EnsembleResult(
            prediction="risk_on",
            participating_models=["dummy_model"],
            strategy=self.strategy_name,
        )


class AnotherDummyStrategy(BaseEnsembleStrategy):

    STRATEGY_NAME = "another_dummy"

    @property
    def strategy_name(self) -> str:
        return self.STRATEGY_NAME

    def combine(self, model_outputs):
        return EnsembleResult(
            prediction="risk_off",
            participating_models=["another_dummy_model"],
            strategy=self.strategy_name,
        )


class InvalidStrategy:
    STRATEGY_NAME = "invalid"


class MissingNameStrategy(BaseEnsembleStrategy):

    @property
    def strategy_name(self) -> str:
        return "missing_name"

    def combine(self, model_outputs):
        return EnsembleResult()


def test_registry_starts_empty():
    registry = EnsembleRegistry()

    assert registry.count == 0
    assert registry.list_strategies() == []


def test_registry_registers_strategy():
    registry = EnsembleRegistry()

    registry.register(DummyStrategy)

    assert registry.count == 1
    assert registry.has("dummy") is True


def test_registry_retrieves_strategy():
    registry = EnsembleRegistry()

    registry.register(DummyStrategy)

    strategy_class = registry.get("dummy")

    assert strategy_class is DummyStrategy


def test_registry_lists_strategies_in_registration_order():
    registry = EnsembleRegistry()

    registry.register(DummyStrategy)
    registry.register(AnotherDummyStrategy)

    assert registry.list_strategies() == [
        "dummy",
        "another_dummy",
    ]


def test_registry_rejects_duplicate_strategy():
    registry = EnsembleRegistry()

    registry.register(DummyStrategy)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(DummyStrategy)


def test_registry_unregisters_strategy():
    registry = EnsembleRegistry()

    registry.register(DummyStrategy)

    registry.unregister("dummy")

    assert registry.count == 0
    assert registry.has("dummy") is False


def test_registry_rejects_unregistering_missing_strategy():
    registry = EnsembleRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.unregister("dummy")


def test_registry_rejects_getting_missing_strategy():
    registry = EnsembleRegistry()

    with pytest.raises(KeyError, match="not registered"):
        registry.get("dummy")


def test_registry_rejects_non_strategy_class():
    registry = EnsembleRegistry()

    with pytest.raises(TypeError, match="inherit"):
        registry.register(InvalidStrategy)


def test_registry_rejects_non_class():
    registry = EnsembleRegistry()

    with pytest.raises(TypeError, match="must be a class"):
        registry.register("invalid")


def test_registry_rejects_strategy_without_constant_name():
    registry = EnsembleRegistry()

    with pytest.raises(ValueError, match="STRATEGY_NAME"):
        registry.register(MissingNameStrategy)


def test_registry_validates_empty_strategy_name():
    registry = EnsembleRegistry()

    with pytest.raises(ValueError, match="non-empty"):
        registry.has("")


def test_registry_validates_non_string_strategy_name():
    registry = EnsembleRegistry()

    with pytest.raises(TypeError, match="must be a string"):
        registry.get(123)


def test_registry_clear():
    registry = EnsembleRegistry()

    registry.register(DummyStrategy)
    registry.register(AnotherDummyStrategy)

    registry.clear()

    assert registry.count == 0
    assert registry.list_strategies() == []