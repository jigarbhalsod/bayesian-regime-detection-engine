import pytest

from src.integration.config import IntegrationConfig
from src.integration.registry import ModelRegistry
from src.integration.service import IntegrationService


class AddModel:
    def __init__(self, value):
        self.value = value


class FailingModel:
    def __init__(self, message="execution failed"):
        self.message = message


def add_executor(model, value):
    return model.value + value


def mixed_executor(model, value):
    if isinstance(model, FailingModel):
        raise RuntimeError(model.message)

    return model.value + value


def build_registry():
    registry = ModelRegistry()

    registry.register(
        "hmm",
        AddModel(10),
    )

    registry.register(
        "bayesian",
        AddModel(20),
    )

    return registry


def test_default_service():
    service = IntegrationService()

    assert isinstance(service.registry, ModelRegistry)
    assert isinstance(service.config, IntegrationConfig)


def test_custom_registry_and_config():
    registry = ModelRegistry()

    config = IntegrationConfig(
        integration_name="advanced_models",
        min_models=1,
        fail_fast=False,
        include_metadata=False,
    )

    service = IntegrationService(
        registry=registry,
        config=config,
    )

    assert service.registry is registry
    assert service.config is config


@pytest.mark.parametrize(
    "registry",
    [
        "invalid",
        {},
        [],
        123,
    ],
)
def test_invalid_registry_type(registry):
    with pytest.raises(TypeError):
        IntegrationService(
            registry=registry
        )


@pytest.mark.parametrize(
    "config",
    [
        "invalid",
        {},
        [],
        123,
    ],
)
def test_invalid_config_type(config):
    with pytest.raises(TypeError):
        IntegrationService(
            config=config
        )


def test_execute_successfully():
    service = IntegrationService(
        registry=build_registry()
    )

    result = service.execute(
        add_executor,
        5,
    )

    assert result.outputs == {
        "hmm": 15,
        "bayesian": 25,
    }

    assert result.errors == {}
    assert result.success is True
    assert result.successful_models == 2
    assert result.failed_models == 0
    assert result.total_models == 2


def test_execute_forwards_keyword_arguments():
    registry = ModelRegistry()
    registry.register(
        "hmm",
        AddModel(10),
    )

    service = IntegrationService(
        registry=registry
    )

    def executor(model, *, multiplier):
        return model.value * multiplier

    result = service.execute(
        executor,
        multiplier=3,
    )

    assert result.outputs == {
        "hmm": 30,
    }


def test_non_callable_executor_rejected():
    service = IntegrationService(
        registry=build_registry()
    )

    with pytest.raises(TypeError):
        service.execute(
            "not_callable"
        )


def test_insufficient_models_rejected():
    registry = ModelRegistry()
    registry.register(
        "hmm",
        AddModel(10),
    )

    config = IntegrationConfig(
        min_models=2,
    )

    service = IntegrationService(
        registry=registry,
        config=config,
    )

    with pytest.raises(
        ValueError,
        match="insufficient registered models",
    ):
        service.execute(
            add_executor,
            5,
        )


def test_fail_fast_raises_model_error():
    registry = ModelRegistry()

    registry.register(
        "hmm",
        AddModel(10),
    )

    registry.register(
        "bayesian",
        FailingModel("bayesian failed"),
    )

    service = IntegrationService(
        registry=registry,
        config=IntegrationConfig(
            fail_fast=True,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="bayesian failed",
    ):
        service.execute(
            mixed_executor,
            5,
        )


def test_non_fail_fast_collects_errors():
    registry = ModelRegistry()

    registry.register(
        "hmm",
        AddModel(10),
    )

    registry.register(
        "bayesian",
        FailingModel("bayesian failed"),
    )

    registry.register(
        "particle_filter",
        AddModel(30),
    )

    service = IntegrationService(
        registry=registry,
        config=IntegrationConfig(
            fail_fast=False,
        ),
    )

    result = service.execute(
        mixed_executor,
        5,
    )

    assert result.outputs == {
        "hmm": 15,
        "particle_filter": 35,
    }

    assert result.errors == {
        "bayesian": "bayesian failed",
    }

    assert result.success is False
    assert result.successful_models == 2
    assert result.failed_models == 1
    assert result.total_models == 3


def test_non_fail_fast_uses_exception_name_for_empty_message():
    registry = ModelRegistry()

    registry.register(
        "hmm",
        AddModel(10),
    )

    registry.register(
        "bayesian",
        FailingModel(""),
    )

    service = IntegrationService(
        registry=registry,
        config=IntegrationConfig(
            fail_fast=False,
        ),
    )

    result = service.execute(
        mixed_executor,
        5,
    )

    assert result.outputs == {
        "hmm": 15,
    }

    assert result.errors == {
        "bayesian": "RuntimeError",
    }


def test_metadata_is_included_by_default():
    service = IntegrationService(
        registry=build_registry()
    )

    result = service.execute(
        add_executor,
        5,
    )

    assert result.metadata == {
        "integration_name": "model_integration",
        "registered_models": 2,
        "successful_models": 2,
        "failed_models": 0,
    }


def test_metadata_can_be_disabled():
    service = IntegrationService(
        registry=build_registry(),
        config=IntegrationConfig(
            include_metadata=False,
        ),
    )

    result = service.execute(
        add_executor,
        5,
    )

    assert result.metadata == {}


def test_custom_integration_name_in_metadata():
    service = IntegrationService(
        registry=build_registry(),
        config=IntegrationConfig(
            integration_name="phase9_models",
        ),
    )

    result = service.execute(
        add_executor,
        5,
    )

    assert result.metadata[
        "integration_name"
    ] == "phase9_models"


def test_execution_uses_registry_order():
    registry = ModelRegistry()

    registry.register(
        "first",
        AddModel(1),
    )

    registry.register(
        "second",
        AddModel(2),
    )

    registry.register(
        "third",
        AddModel(3),
    )

    service = IntegrationService(
        registry=registry
    )

    execution_order = []

    def executor(model):
        execution_order.append(
            model.value
        )
        return model.value

    result = service.execute(executor)

    assert execution_order == [1, 2, 3]

    assert tuple(result.outputs.keys()) == (
        "first",
        "second",
        "third",
    )