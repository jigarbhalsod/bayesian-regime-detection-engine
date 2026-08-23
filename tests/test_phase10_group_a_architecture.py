import pytest

from src.ensemble import (
    BaseEnsembleStrategy,
    EnsembleConfig,
    EnsembleResult,
)


def test_ensemble_config_defaults():
    config = EnsembleConfig()

    assert config.name == "default_ensemble"
    assert config.strategy == "base"
    assert config.min_models == 1
    assert config.allow_missing_models is True


def test_ensemble_config_validates_successfully():
    config = EnsembleConfig(
        name="test_ensemble",
        strategy="weighted",
        enabled_models=["hmm", "bayesian"],
        model_weights={
            "hmm": 0.5,
            "bayesian": 0.5,
        },
        min_models=2,
    )

    config.validate()


def test_ensemble_config_rejects_invalid_min_models():
    config = EnsembleConfig(min_models=0)

    with pytest.raises(ValueError, match="min_models"):
        config.validate()


def test_ensemble_config_rejects_negative_weights():
    config = EnsembleConfig(
        model_weights={
            "hmm": -0.5,
        }
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        config.validate()


def test_ensemble_config_rejects_duplicate_models():
    config = EnsembleConfig(
        enabled_models=["hmm", "hmm"],
    )

    with pytest.raises(ValueError, match="duplicate"):
        config.validate()


def test_ensemble_result_defaults():
    result = EnsembleResult()

    assert result.success is True
    assert result.strategy == "base"
    assert result.model_count == 0


def test_ensemble_result_validation():
    result = EnsembleResult(
        prediction="risk_on",
        probabilities={
            "risk_on": 0.7,
            "risk_off": 0.3,
        },
        participating_models=[
            "hmm",
            "bayesian",
        ],
        strategy="weighted",
    )

    result.validate()

    assert result.model_count == 2


def test_ensemble_result_rejects_negative_probability():
    result = EnsembleResult(
        probabilities={
            "risk_on": -0.1,
        }
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        result.validate()


def test_ensemble_result_rejects_duplicate_models():
    result = EnsembleResult(
        participating_models=[
            "hmm",
            "hmm",
        ]
    )

    with pytest.raises(ValueError, match="duplicates"):
        result.validate()


def test_base_strategy_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseEnsembleStrategy()


def test_concrete_strategy_follows_contract():

    class DummyStrategy(BaseEnsembleStrategy):

        @property
        def strategy_name(self) -> str:
            return "dummy"

        def combine(self, model_outputs):
            return EnsembleResult(
                prediction="risk_on",
                participating_models=["dummy_model"],
                strategy=self.strategy_name,
            )

    strategy = DummyStrategy()

    result = strategy.combine([])

    validated_result = strategy.validate_result(result)

    assert validated_result.prediction == "risk_on"
    assert validated_result.strategy == "dummy"


def test_base_strategy_rejects_invalid_result():

    class DummyStrategy(BaseEnsembleStrategy):

        @property
        def strategy_name(self) -> str:
            return "dummy"

        def combine(self, model_outputs):
            return EnsembleResult()

    strategy = DummyStrategy()

    with pytest.raises(TypeError, match="EnsembleResult"):
        strategy.validate_result("invalid_result")