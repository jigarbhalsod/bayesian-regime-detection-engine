import pytest

from src.ensemble import (
    EnsembleConfig,
    EnsembleModelOutput,
    PreparedEnsembleInput,
    VotingEnsembleStrategy,
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
                    "risk_on": 0.3,
                    "risk_off": 0.7,
                },
            ),
            EnsembleModelOutput(
                model_name="advanced",
                prediction="risk_on",
                probabilities={
                    "risk_on": 0.9,
                    "risk_off": 0.1,
                },
            ),
        ],
        regime_labels=[
            "risk_on",
            "risk_off",
        ],
    )


def test_voting_strategy_name():
    strategy = VotingEnsembleStrategy()

    assert strategy.strategy_name == "voting"


def test_voting_strategy_selects_majority_prediction():
    strategy = VotingEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.prediction == "risk_on"


def test_voting_strategy_calculates_vote_probabilities():
    strategy = VotingEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.probabilities["risk_on"] == pytest.approx(2 / 3)
    assert result.probabilities["risk_off"] == pytest.approx(1 / 3)


def test_voting_strategy_preserves_model_predictions():
    strategy = VotingEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.model_predictions == {
        "hmm": "risk_on",
        "bayesian": "risk_off",
        "advanced": "risk_on",
    }


def test_voting_strategy_preserves_model_probabilities():
    strategy = VotingEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.model_probabilities["hmm"]["risk_on"] == pytest.approx(0.8)
    assert result.model_probabilities["bayesian"]["risk_off"] == pytest.approx(
        0.7
    )


def test_voting_strategy_stores_vote_counts():
    strategy = VotingEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.metadata["vote_counts"] == {
        "risk_on": 2,
        "risk_off": 1,
    }


def test_voting_strategy_resolves_tie_by_regime_label_order():
    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(
                model_name="model_a",
                prediction="risk_off",
                probabilities={
                    "risk_off": 1.0,
                },
            ),
            EnsembleModelOutput(
                model_name="model_b",
                prediction="risk_on",
                probabilities={
                    "risk_on": 1.0,
                },
            ),
        ],
        regime_labels=[
            "risk_on",
            "risk_off",
        ],
    )

    strategy = VotingEnsembleStrategy()

    result = strategy.combine(prepared)

    assert result.prediction == "risk_on"


def test_voting_strategy_handles_single_model():
    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(
                model_name="hmm",
                prediction="risk_on",
                probabilities={
                    "risk_on": 1.0,
                },
            )
        ],
        regime_labels=["risk_on"],
    )

    strategy = VotingEnsembleStrategy()

    result = strategy.combine(prepared)

    assert result.prediction == "risk_on"
    assert result.probabilities["risk_on"] == pytest.approx(1.0)


def test_voting_strategy_respects_minimum_models():
    config = EnsembleConfig(
        min_models=2,
    )

    strategy = VotingEnsembleStrategy(config)

    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(
                model_name="hmm",
                prediction="risk_on",
                probabilities={
                    "risk_on": 1.0,
                },
            )
        ],
        regime_labels=["risk_on"],
    )

    with pytest.raises(ValueError, match="At least 2"):
        strategy.combine(prepared)


def test_voting_strategy_rejects_invalid_input_type():
    strategy = VotingEnsembleStrategy()

    with pytest.raises(
        TypeError,
        match="PreparedEnsembleInput",
    ):
        strategy.combine([])


def test_voting_strategy_rejects_missing_prediction():
    prepared = PreparedEnsembleInput(
        outputs=[
            EnsembleModelOutput(
                model_name="hmm",
                prediction=None,
                probabilities={
                    "risk_on": 1.0,
                },
            )
        ],
        regime_labels=["risk_on"],
    )

    strategy = VotingEnsembleStrategy()

    with pytest.raises(ValueError, match="does not have a prediction"):
        strategy.combine(prepared)


def test_voting_strategy_returns_normalized_probabilities():
    strategy = VotingEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_voting_strategy_result_is_successful():
    strategy = VotingEnsembleStrategy()

    result = strategy.combine(make_prepared_input())

    assert result.success is True
    assert result.strategy == "voting"