from src.integration import (
    AdvancedModelInput,
    IntegrationConfig,
    IntegrationService,
    ModelOrchestrator,
    ModelRegistry,
)


class HMMStub:
    def predict(self, model_input):
        return {
            "prediction": "risk_on",
            "probabilities": {
                "risk_on": 0.8,
                "risk_off": 0.2,
            },
        }


class BayesianStub:
    def predict(self, model_input):
        return {
            "prediction": "risk_on",
            "uncertainty": 0.15,
            "posterior_samples": 500,
        }


class FoundationStub:
    def predict(self, model_input):
        return {
            "value": 0.62,
            "forecast_horizon": 5,
        }


class BrokenStub:
    def predict(self, model_input):
        raise ValueError("simulated failure")


def build_registry():
    registry = ModelRegistry()

    registry.register(
        "hmm",
        HMMStub(),
    )

    registry.register(
        "bayesian",
        BayesianStub(),
    )

    registry.register(
        "foundation",
        FoundationStub(),
    )

    registry.register(
        "broken",
        BrokenStub(),
    )

    return registry


def test_complete_advanced_model_flow():
    registry = build_registry()

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    model_input = AdvancedModelInput(
        features=[
            [100.0, 0.01],
            [101.0, 0.02],
            [99.0, -0.01],
        ],
        timestamps=[
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ],
        regime_context="transitional",
        metadata={
            "symbol": "TEST",
        },
    )

    result = orchestrator.run(
        model_input
    )

    assert set(result.outputs.keys()) == {
        "hmm",
        "bayesian",
        "foundation",
    }

    assert set(result.errors.keys()) == {
        "broken",
    }

    assert result.outputs["hmm"][
        "prediction"
    ] == "risk_on"

    assert result.outputs["hmm"][
        "probabilities"
    ] == {
        "risk_on": 0.8,
        "risk_off": 0.2,
    }

    assert result.outputs["bayesian"][
        "uncertainty"
    ] == 0.15

    assert result.outputs["bayesian"][
        "metadata"
    ] == {
        "posterior_samples": 500,
    }

    assert result.outputs["foundation"][
        "prediction"
    ] == 0.62

    assert result.outputs["foundation"][
        "metadata"
    ] == {
        "forecast_horizon": 5,
    }

    assert result.metadata[
        "successful_models"
    ] == [
        "hmm",
        "bayesian",
        "foundation",
    ]

    assert result.metadata[
        "failed_models"
    ] == [
        "broken",
    ]


def test_orchestrator_can_run_selected_models():
    registry = build_registry()

    orchestrator = ModelOrchestrator(
        registry=registry,
    )

    model_input = AdvancedModelInput(
        features=[1, 2, 3],
    )

    result = orchestrator.run(
        model_input,
        model_names=[
            "hmm",
            "foundation",
        ],
    )

    assert set(result.outputs.keys()) == {
        "hmm",
        "foundation",
    }

    assert result.errors == {}


def test_existing_integration_components_remain_available():
    config = IntegrationConfig()
    registry = ModelRegistry()
    service = IntegrationService(
        registry=registry,
    )

    assert config is not None
    assert registry is not None
    assert service is not None