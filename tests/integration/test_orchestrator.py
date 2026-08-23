import pytest

from src.integration.adapter import ModelOutputAdapter
from src.integration.input import AdvancedModelInput
from src.integration.orchestrator import (
    ModelOrchestrator,
)
from src.integration.registry import ModelRegistry
from src.integration.result import IntegrationResult


class PredictModel:
    def __init__(self, value):
        self.value = value

    def predict(self, model_input):
        return {
            "prediction": self.value,
            "n_observations": (
                model_input.n_observations
            ),
        }


class RunModel:
    def __init__(self, value):
        self.value = value

    def run(self, model_input):
        return self.value


class CallableModel:
    def __init__(self, value):
        self.value = value

    def __call__(self, model_input):
        return {
            "value": self.value,
        }


class FailingModel:
    def predict(self, model_input):
        raise RuntimeError("model failed")


class InvalidModel:
    pass


@pytest.fixture
def registry():
    return ModelRegistry()


@pytest.fixture
def model_input():
    return AdvancedModelInput(
        features=[1, 2, 3],
    )


def test_requires_model_registry():
    with pytest.raises(TypeError):
        ModelOrchestrator(
            registry="invalid",
        )


def test_invalid_adapter_rejected(registry):
    with pytest.raises(TypeError):
        ModelOrchestrator(
            registry=registry,
            adapter="invalid",
        )


def test_default_adapter_created(registry):
    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    assert isinstance(
        orchestrator.adapter,
        ModelOutputAdapter,
    )


def test_supplied_adapter_used(registry):
    adapter = ModelOutputAdapter()

    orchestrator = ModelOrchestrator(
        registry=registry,
        adapter=adapter,
    )

    assert orchestrator.adapter is adapter


def test_registry_property(registry):
    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    assert orchestrator.registry is registry


def test_run_predict_model(
    registry,
    model_input,
):
    registry.register(
        "predictor",
        PredictModel("up"),
    )

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    result = orchestrator.run(
        model_input
    )

    assert isinstance(
        result,
        IntegrationResult,
    )

    assert result.outputs["predictor"] == {
        "model_name": "predictor",
        "prediction": "up",
        "probabilities": None,
        "uncertainty": None,
        "metadata": {
            "n_observations": 3,
        },
    }

    assert result.errors == {}


def test_run_method_supported(
    registry,
    model_input,
):
    registry.register(
        "runner",
        RunModel("down"),
    )

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    result = orchestrator.run(
        model_input
    )

    assert result.outputs["runner"][
        "prediction"
    ] == "down"


def test_callable_model_supported(
    registry,
    model_input,
):
    registry.register(
        "callable_model",
        CallableModel("sideways"),
    )

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    result = orchestrator.run(
        model_input
    )

    assert result.outputs["callable_model"][
        "prediction"
    ] == "sideways"


def test_selected_models_only(
    registry,
    model_input,
):
    registry.register(
        "first",
        PredictModel(1),
    )

    registry.register(
        "second",
        PredictModel(2),
    )

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    result = orchestrator.run(
        model_input,
        model_names=["second"],
    )

    assert tuple(result.outputs.keys()) == (
        "second",
    )


def test_model_failure_does_not_stop_others(
    registry,
    model_input,
):
    registry.register(
        "working",
        PredictModel("up"),
    )

    registry.register(
        "failing",
        FailingModel(),
    )

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    result = orchestrator.run(
        model_input
    )

    assert "working" in result.outputs
    assert "failing" in result.errors
    assert "RuntimeError" in (
        result.errors["failing"]
    )


def test_invalid_model_is_captured(
    registry,
    model_input,
):
    registry.register(
        "invalid",
        InvalidModel(),
    )

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    result = orchestrator.run(
        model_input
    )

    assert "invalid" in result.errors
    assert "TypeError" in (
        result.errors["invalid"]
    )


def test_invalid_input_rejected(registry):
    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    with pytest.raises(TypeError):
        orchestrator.run(
            model_input="invalid",
        )


def test_empty_model_names_rejected(
    registry,
    model_input,
):
    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            model_input,
            model_names=[],
        )


@pytest.mark.parametrize(
    "model_names",
    [
        "predictor",
        b"predictor",
        123,
    ],
)
def test_invalid_model_names_container_rejected(
    registry,
    model_input,
    model_names,
):
    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    with pytest.raises(TypeError):
        orchestrator.run(
            model_input,
            model_names=model_names,
        )


@pytest.mark.parametrize(
    "model_names",
    [
        [None],
        [123],
        [True],
    ],
)
def test_invalid_model_name_type_rejected(
    registry,
    model_input,
    model_names,
):
    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    with pytest.raises(TypeError):
        orchestrator.run(
            model_input,
            model_names=model_names,
        )


@pytest.mark.parametrize(
    "model_names",
    [
        [""],
        ["   "],
    ],
)
def test_empty_model_name_rejected(
    registry,
    model_input,
    model_names,
):
    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    with pytest.raises(ValueError):
        orchestrator.run(
            model_input,
            model_names=model_names,
        )


def test_metadata_contains_execution_summary(
    registry,
    model_input,
):
    registry.register(
        "working",
        PredictModel("up"),
    )

    registry.register(
        "failing",
        FailingModel(),
    )

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    result = orchestrator.run(
        model_input
    )

    assert result.metadata[
        "requested_models"
    ] == [
        "working",
        "failing",
    ]

    assert result.metadata[
        "successful_models"
    ] == [
        "working",
    ]

    assert result.metadata[
        "failed_models"
    ] == [
        "failing",
    ]