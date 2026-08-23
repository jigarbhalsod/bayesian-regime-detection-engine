import pytest

from src.ensemble import (
    EnsembleConfig,
    EnsembleModelOutput,
    PreparedEnsembleInput,
    WeightedEnsembleStrategy,
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


def test_weighted_strategy_name():
    strategy = WeightedEnsembleStrategy()

    assert strategy.strategy_name == "weighted"


def test_weighted_strategy_combines_equal_weights():
    strategy = WeightedEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.prediction == "risk_on"
    assert result.probabilities["risk_on"] == pytest.approx(0.65)
    assert result.probabilities["risk_off"] == pytest.approx(0.35)
    assert result.model_count == 2


def test_weighted_strategy_uses_configured_weights():
    config = EnsembleConfig(
        model_weights={
            "hmm": 0.6,
            "bayesian": 0.4,
        }
    )

    strategy = WeightedEnsembleStrategy(config)

    result = strategy.combine(make_prepared_input())

    assert result.prediction == "risk_on"
    assert result.probabilities["risk_on"] == pytest.approx(0.68)
    assert result.probabilities["risk_off"] == pytest.approx(0.32)


def test_weighted_strategy_normalizes_weights():
    config = EnsembleConfig(
        model_weights={
            "hmm": 6.0,
            "bayesian": 4.0,
        },
        normalize_weights=True,
    )

    strategy = WeightedEnsembleStrategy(config)

    result = strategy.combine(make_prepared_input())

    weights = result.metadata["weights"]

    assert weights["hmm"] == pytest.approx(0.6)
    assert weights["bayesian"] == pytest.approx(0.4)


def test_weighted_strategy_uses_default_weight_for_missing_config():
    config = EnsembleConfig(
        model_weights={
            "hmm": 2.0,
        }
    )

    strategy = WeightedEnsembleStrategy(config)

    result = strategy.combine(make_prepared_input())

    weights = result.metadata["weights"]

    assert weights["hmm"] == pytest.approx(2 / 3)
    assert weights["bayesian"] == pytest.approx(1 / 3)


def test_weighted_strategy_handles_missing_regime_probability():
    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(
                model_name="hmm",
                prediction="risk_on",
                probabilities={
                    "risk_on": 1.0,
                },
            ),
            EnsembleModelOutput(
                model_name="bayesian",
                prediction="risk_off",
                probabilities={
                    "risk_off": 1.0,
                },
            ),
        ],
        regime_labels=[
            "risk_on",
            "risk_off",
        ],
    )

    strategy = WeightedEnsembleStrategy()

    result = strategy.combine(prepared)

    assert result.probabilities["risk_on"] == pytest.approx(0.5)
    assert result.probabilities["risk_off"] == pytest.approx(0.5)


def test_weighted_strategy_respects_minimum_models():
    config = EnsembleConfig(
        min_models=2,
    )

    strategy = WeightedEnsembleStrategy(config)

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


def test_weighted_strategy_rejects_invalid_input_type():
    strategy = WeightedEnsembleStrategy()

    with pytest.raises(
        TypeError,
        match="PreparedEnsembleInput",
    ):
        strategy.combine([])


def test_weighted_strategy_rejects_empty_probabilities():
    strategy = WeightedEnsembleStrategy()

    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(
                model_name="hmm",
                prediction="risk_on",
                probabilities={},
            )
        ],
        regime_labels=["risk_on"],
    )

    with pytest.raises(
        ValueError,
        match="positive total",
    ):
        strategy.combine(prepared)


def test_weighted_strategy_preserves_model_information():
    strategy = WeightedEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.model_predictions == {
        "hmm": "risk_on",
        "bayesian": "risk_off",
    }

    assert result.model_probabilities["hmm"]["risk_on"] == pytest.approx(0.8)
    assert result.model_probabilities["bayesian"]["risk_off"] == pytest.approx(
        0.5
    )


def test_weighted_strategy_returns_normalized_probabilities():
    strategy = WeightedEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_weighted_strategy_result_is_successful():
    strategy = WeightedEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.success is True
    assert result.strategy == "weighted"