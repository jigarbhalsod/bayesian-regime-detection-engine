import pytest

from src.ensemble import (
    DynamicWeightedEnsembleStrategy,
    EnsembleConfig,
    EnsembleModelOutput,
    PreparedEnsembleInput,
)


def make_prepared_input():
    return PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(
                model_name="hmm",
                prediction="risk_on",
                probabilities={
                    "risk_on": 0.8,
                    "risk_off": 0.2,
                },
            ),
            EnsembleModelOutput(
                model_name="bayesian",
                prediction="risk_off",
                probabilities={
                    "risk_on": 0.5,
                    "risk_off": 0.5,
                },
            ),
        ],
        regime_labels=[
            "risk_on",
            "risk_off",
        ],
    )


def make_config():
    return EnsembleConfig(
        metadata={
            "performance_scores": {
                "hmm": 0.9,
                "bayesian": 0.6,
            }
        }
    )


def test_dynamic_strategy_name():
    strategy = DynamicWeightedEnsembleStrategy(make_config())

    assert strategy.strategy_name == "dynamic_weighted"


def test_dynamic_strategy_uses_performance_scores():
    strategy = DynamicWeightedEnsembleStrategy(make_config())

    result = strategy.combine(make_prepared_input())

    assert result.metadata["weights"]["hmm"] == pytest.approx(0.6)
    assert result.metadata["weights"]["bayesian"] == pytest.approx(0.4)


def test_dynamic_strategy_combines_probabilities():
    strategy = DynamicWeightedEnsembleStrategy(make_config())

    result = strategy.combine(make_prepared_input())

    assert result.prediction == "risk_on"
    assert result.probabilities["risk_on"] == pytest.approx(0.68)
    assert result.probabilities["risk_off"] == pytest.approx(0.32)


def test_dynamic_strategy_preserves_performance_scores():
    strategy = DynamicWeightedEnsembleStrategy(make_config())

    result = strategy.combine(make_prepared_input())

    assert result.metadata["performance_scores"] == {
        "hmm": 0.9,
        "bayesian": 0.6,
    }


def test_dynamic_strategy_rejects_missing_performance_scores():
    strategy = DynamicWeightedEnsembleStrategy(
        EnsembleConfig()
    )

    with pytest.raises(ValueError, match="performance_scores"):
        strategy.combine(make_prepared_input())


def test_dynamic_strategy_rejects_non_dictionary_scores():
    config = EnsembleConfig(
        metadata={
            "performance_scores": ["invalid"],
        }
    )

    strategy = DynamicWeightedEnsembleStrategy(config)

    with pytest.raises(TypeError, match="dictionary"):
        strategy.combine(make_prepared_input())


def test_dynamic_strategy_rejects_missing_model_score():
    config = EnsembleConfig(
        metadata={
            "performance_scores": {
                "hmm": 0.9,
            }
        }
    )

    strategy = DynamicWeightedEnsembleStrategy(config)

    with pytest.raises(ValueError, match="Missing performance score"):
        strategy.combine(make_prepared_input())


def test_dynamic_strategy_rejects_non_numeric_score():
    config = EnsembleConfig(
        metadata={
            "performance_scores": {
                "hmm": "high",
                "bayesian": 0.6,
            }
        }
    )

    strategy = DynamicWeightedEnsembleStrategy(config)

    with pytest.raises(TypeError, match="must be numeric"):
        strategy.combine(make_prepared_input())


def test_dynamic_strategy_rejects_negative_score():
    config = EnsembleConfig(
        metadata={
            "performance_scores": {
                "hmm": -0.1,
                "bayesian": 0.6,
            }
        }
    )

    strategy = DynamicWeightedEnsembleStrategy(config)

    with pytest.raises(ValueError, match="cannot be negative"):
        strategy.combine(make_prepared_input())


def test_dynamic_strategy_rejects_zero_total_score():
    config = EnsembleConfig(
        metadata={
            "performance_scores": {
                "hmm": 0.0,
                "bayesian": 0.0,
            }
        }
    )

    strategy = DynamicWeightedEnsembleStrategy(config)

    with pytest.raises(ValueError, match="total performance score"):
        strategy.combine(make_prepared_input())


def test_dynamic_strategy_respects_minimum_models():
    config = EnsembleConfig(
        min_models=2,
        metadata={
            "performance_scores": {
                "hmm": 0.9,
            }
        },
    )

    strategy = DynamicWeightedEnsembleStrategy(config)

    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(
                model_name="hmm",
                probabilities={
                    "risk_on": 1.0,
                },
            )
        ],
        regime_labels=["risk_on"],
    )

    with pytest.raises(ValueError, match="At least 2"):
        strategy.combine(prepared)


def test_dynamic_strategy_rejects_invalid_input_type():
    strategy = DynamicWeightedEnsembleStrategy(make_config())

    with pytest.raises(
        TypeError,
        match="PreparedEnsembleInput",
    ):
        strategy.combine([])


def test_dynamic_strategy_returns_normalized_probabilities():
    strategy = DynamicWeightedEnsembleStrategy(make_config())

    result = strategy.combine(make_prepared_input())

    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_dynamic_strategy_result_is_successful():
    strategy = DynamicWeightedEnsembleStrategy(make_config())

    result = strategy.combine(make_prepared_input())

    assert result.success is True
    assert result.strategy == "dynamic_weighted"